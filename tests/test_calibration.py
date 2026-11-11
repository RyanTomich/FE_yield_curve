"""Numerical checks for staged estimation and bounded least-squares curve fitting."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from calibration import (KAPPA_BOUNDS, estimate_short_rate_volatility, fit_yield_curve,
                         calibrate_yield_curve)
from vasicek import bond_yield


class VolatilityTests(unittest.TestCase):
    def setUp(self):
        self.dates = np.datetime64('2019-01-01') + np.arange(61)
        z = np.sin(np.arange(60, dtype=float))
        z = (z - z.mean()) / z.std(ddof=1)
        self.rates = .03 + np.r_[0, np.cumsum(.008*z/np.sqrt(365.25) + .002/365.25)]

    def test_recovers_volatility_and_removed_drift(self):
        fit = estimate_short_rate_volatility(self.dates[::-1], self.rates[::-1])
        self.assertAlmostEqual(fit.sigma, .008, places=12)
        self.assertAlmostEqual(fit.drift, .002, places=12)
        self.assertEqual(fit.change_count, 60)
        self.assertEqual(fit.observation_count, 61)
        self.assertEqual(fit.start_date, '2019-01-01')
        self.assertEqual(fit.end_date, '2019-03-02')

    def test_gaps_missing_values_and_drift_correction(self):
        dates = self.dates.copy()
        dates[15:] += np.timedelta64(2, 'D')
        dates[40:] += np.timedelta64(20, 'D')
        rates = self.rates.copy()
        rates[10] = np.nan
        fit = estimate_short_rate_volatility(dates, rates)
        valid = [(d, r) for d, r in zip(dates, rates) if np.isfinite(r)]
        changes = []
        intervals = []
        for (d0, r0), (d1, r1) in zip(valid[:-1], valid[1:]):
            gap = int((d1-d0).astype(int))
            if gap <= 7:
                changes.append(r1-r0)
                intervals.append(gap/365.25)
        mu = sum(changes)/sum(intervals)
        variance = sum((dr-mu*dt)**2/dt for dr, dt in zip(changes, intervals))/(len(changes)-1)
        self.assertAlmostEqual(fit.sigma, np.sqrt(variance), places=12)
        self.assertEqual(fit.change_count, 58)
        self.assertEqual(fit.observation_count, 60)
        np.testing.assert_allclose(fit.scaled_shocks,
                                   (np.asarray(changes)-mu*np.asarray(intervals))/np.sqrt(intervals))
        self.assertEqual(len(fit.shock_dates), 58)
        self.assertNotIn(dates[40], fit.shock_dates)

    def test_zero_is_preserved_and_warned(self):
        fit = estimate_short_rate_volatility(self.dates, np.full(61, .02))
        self.assertEqual(fit.sigma, 0)
        self.assertTrue(any('zero' in warning for warning in fit.warnings))

    def test_invalid_and_insufficient_history(self):
        cases = [
            (self.dates[:20], self.rates[:20]),
            (np.r_[self.dates, self.dates[-1]], np.r_[self.rates, np.nan]),
            (self.dates, np.where(np.arange(61) == 5, np.inf, self.rates)),
            (self.dates, self.rates[:-1]),
            (self.dates, np.full(61, np.nan)),
            (np.datetime64('2020-01-01') + np.arange(61)*8, self.rates),
        ]
        for dates, rates in cases:
            with self.subTest(size=len(dates)), self.assertRaises(ValueError):
                estimate_short_rate_volatility(dates, rates)

class CurveFitTests(unittest.TestCase):
    def setUp(self):
        self.maturities = np.array([.25, .5, 1, 2, 5, 10, 20, 30])
        self.yields = bond_yield(.035, .42, .05, .012, self.maturities)

    def fit(self, yields=None, sigma=.012):
        return calibrate_yield_curve(self.maturities, self.yields if yields is None else yields,
                                     .035, sigma)

    def test_recovers_parameters_and_compatibility(self):
        fit = self.fit()
        np.testing.assert_allclose([fit.kappa, fit.theta], [.42, .05], atol=1e-7)
        np.testing.assert_allclose(fit.model_yields, self.yields, atol=1e-9)
        np.testing.assert_allclose(fit.residuals, 0, atol=1e-9)
        self.assertAlmostEqual(fit.half_life, np.log(2)/.42, places=6)
        self.assertAlmostEqual(fit.limiting_yield, .05-.012**2/(2*.42**2), places=7)
        self.assertEqual(fit_yield_curve(self.maturities, self.yields, .035, .012),
                         (fit.kappa, fit.theta))
        self.assertEqual(len(fit.solver_attempts), 3)
        self.assertTrue(all(a['success'] for a in fit.solver_attempts))
        self.assertEqual(self.fit().solver_attempts, fit.solver_attempts)

    def test_noisy_residuals_and_rmse(self):
        noisy = self.yields + np.array([2, -1, 3, -2, 1, 0, -1, 2])*1e-5
        fit = self.fit(noisy)
        np.testing.assert_allclose(fit.residuals, fit.model_yields-noisy)
        self.assertAlmostEqual(fit.rmse_bp, np.sqrt(np.mean(fit.residuals**2))*10000)
        self.assertAlmostEqual(fit.rmse_bp, min(a['rmse_bp'] for a in fit.solver_attempts))

    def test_kappa_boundaries(self):
        for kappa in KAPPA_BOUNDS:
            with self.subTest(kappa=kappa):
                fit = self.fit(bond_yield(.035, kappa, .05, .012, self.maturities))
                self.assertAlmostEqual(fit.kappa, kappa, delta=1e-6)
                self.assertTrue(any('kappa is at' in w for w in fit.warnings))

    def test_theta_boundary_and_poor_fit(self):
        fit = self.fit(np.full(8, .5), sigma=0)
        self.assertAlmostEqual(fit.theta, .2)
        self.assertTrue(any('theta is at' in w for w in fit.warnings))
        self.assertTrue(any('20 bp' in w for w in fit.warnings))

    def test_flat_zero_volatility_warns_about_starting_guesses(self):
        fit = calibrate_yield_curve(self.maturities, np.full(8, .03), .03, 0)
        self.assertLess(fit.rmse_bp, 1e-6)
        self.assertTrue(any('starting-guess sensitivity' in w for w in fit.warnings))
        self.assertTrue(any('not a confidence interval' in w for w in fit.warnings))

    def test_longest_maturity_initialization_and_clipping(self):
        order = [7, 0, 6, 1, 5, 2, 4, 3]
        fit = calibrate_yield_curve(self.maturities[order], self.yields[order], .035, .012)
        self.assertTrue(all(a['initial_theta'] == self.yields[-1] for a in fit.solver_attempts))
        for level in (-.5, .5):
            fit = self.fit(np.full(8, level), sigma=0)
            self.assertTrue(all(a['initial_theta'] == np.clip(level, -.19, .19)
                                for a in fit.solver_attempts))

    @staticmethod
    def result(kappa=.42, theta=.05, success=True):
        return SimpleNamespace(x=np.array([kappa, theta]), success=success,
                               message='converged' if success else 'evaluation limit reached')

    def test_partial_failures_are_recorded_and_json_safe(self):
        with patch('calibration.least_squares', side_effect=[
                self.result(success=False), self.result(kappa=np.nan), self.result()]):
            fit = self.fit()
        self.assertEqual([a['success'] for a in fit.solver_attempts], [False, False, True])
        self.assertIsNone(fit.solver_attempts[1]['kappa'])
        self.assertIsNone(fit.solver_attempts[1]['rmse_bp'])
        self.assertEqual(sum('failed:' in w for w in fit.warnings), 2)
        json.dumps(fit.solver_attempts, allow_nan=False)

    def test_solver_exception_does_not_discard_other_starts(self):
        with patch('calibration.least_squares', side_effect=[
                ValueError('nonfinite residuals'), self.result(), self.result()]):
            fit = self.fit()
        self.assertFalse(fit.solver_attempts[0]['success'])
        self.assertIn('nonfinite residuals', fit.solver_attempts[0]['message'])
        json.dumps(fit.solver_attempts, allow_nan=False)

    def test_all_failed_starts_raise(self):
        with patch('calibration.least_squares', return_value=self.result(success=False)):
            with self.assertRaisesRegex(RuntimeError, 'all three starting guesses'):
                self.fit()

    def test_exact_ties_preserve_starting_order_and_theta_sensitivity(self):
        # Constant mocked model isolates tie selection and the theta-spread warning.
        with patch('calibration.bond_yield', return_value=np.full(8, .03)), \
             patch('calibration.least_squares', side_effect=[
                 self.result(.4, .04), self.result(.4, .06), self.result(.4, .05)]):
            fit = self.fit(np.full(8, .03))
        self.assertEqual(fit.theta, .04)
        self.assertTrue(any('starting-guess sensitivity' in w for w in fit.warnings))

    def test_reference_curves_and_local_minimum_regression(self):
        cases = json.loads((Path(__file__).parent / 'fixtures' / 'calibration_regression.json').read_text())
        for case in cases:
            with self.subTest(date=case['date']):
                fit = calibrate_yield_curve(case['maturities'], case['yields'], case['r0'], case['sigma'])
                self.assertAlmostEqual(fit.rmse_bp, case['reference_rmse_bp'], delta=.001)
                if case['date'] == '2019-06-03':
                    self.assertGreater(fit.solver_attempts[1]['rmse_bp'] - fit.rmse_bp, .5)
                    self.assertAlmostEqual(fit.kappa, 5, delta=1e-6)
                    self.assertTrue(any('starting-guess sensitivity' in w for w in fit.warnings))

    def test_rejects_invalid_curve_inputs(self):
        for maturities, yields, r0, sigma in [
                ([1, 1], [.02, .03], .02, .01), ([0, 1], [.02, .03], .02, .01),
                ([1], [.02], .02, .01), ([1, 2], [.02, np.nan], .02, .01),
                ([1, 2], [.02, .03], np.inf, .01), ([1, 2], [.02, .03], .02, -.01)]:
            with self.subTest(maturities=maturities), self.assertRaises(ValueError):
                calibrate_yield_curve(maturities, yields, r0, sigma)


if __name__ == '__main__':
    unittest.main()
