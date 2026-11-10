"""Numerical checks for staged estimation and one-dimensional curve profiling."""

import unittest

import numpy as np
from scipy.optimize import least_squares

from calibration import (KAPPA_BOUNDS, estimate_short_rate_volatility, fit_yield_curve,
                         fit_yield_curve_profile, historical_yield_volatility)
from vasicek import bond_yield


class VolatilityTests(unittest.TestCase):
    def setUp(self):
        self.dates = np.datetime64('2019-01-01') + np.arange(61)
        z = np.random.default_rng(7).normal(size=60)
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

    def test_diagnostic_omits_unsupported_maturities(self):
        panel = np.column_stack([self.rates, self.rates*2, self.rates])
        panel[10:, 2] = np.nan
        vol, counts = historical_yield_volatility(self.dates, panel)
        np.testing.assert_allclose(vol[:2], [.008, .016], atol=1e-12)
        self.assertTrue(np.isnan(vol[2]))
        np.testing.assert_array_equal(counts, [60, 60, 9])


class CurveProfileTests(unittest.TestCase):
    def setUp(self):
        self.maturities = np.array([.25, .5, 1, 2, 5, 10, 20, 30])
        self.params = (.035, .42, .05, .012)
        self.yields = bond_yield(*self.params, self.maturities)

    def test_recovers_parameters_and_compatibility(self):
        fit = fit_yield_curve_profile(self.maturities, self.yields, .035, .012)
        np.testing.assert_allclose([fit.kappa, fit.theta], [.42, .05], atol=1e-7)
        np.testing.assert_allclose(fit.model_yields, self.yields, atol=1e-9)
        np.testing.assert_allclose(fit.residuals, 0, atol=1e-9)
        self.assertAlmostEqual(fit.half_life, np.log(2)/.42, places=6)
        self.assertAlmostEqual(fit.limiting_yield, .05-.012**2/(2*.42**2), places=7)
        self.assertEqual(fit_yield_curve(self.maturities, self.yields, .035, .012),
                         (fit.kappa, fit.theta))
        self.assertGreaterEqual(len(fit.profile_kappa), 501)
        self.assertEqual(fit.profile_kappa[0], KAPPA_BOUNDS[0])
        self.assertEqual(fit.profile_kappa[-1], KAPPA_BOUNDS[1])

    def test_matches_independent_numerical_least_squares(self):
        noisy = self.yields + np.array([2, -1, 3, -2, 1, 0, -1, 2])*1e-5
        fit = fit_yield_curve_profile(self.maturities, noisy, .035, .012)
        reference = least_squares(
            lambda p: 1e4*(bond_yield(.035, p[0], p[1], .012, self.maturities)-noisy),
            [.3, .04], bounds=([.005, -.2], [5, .2]),
            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        np.testing.assert_allclose([fit.kappa, fit.theta], reference.x, rtol=1e-5)
        self.assertAlmostEqual(fit.rmse_bp, np.sqrt(np.mean(reference.fun**2)), places=7)
        for i in [0, 150, 300, -1]:
            k, theta = fit.profile_kappa[i], fit.profile_theta[i]
            reference_theta = least_squares(
                lambda p: 1e4*(bond_yield(.035, k, p[0], .012, self.maturities)-noisy),
                [.04], bounds=([-.2], [.2]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
            self.assertAlmostEqual(theta, reference_theta.x[0], places=6)

    def test_both_kappa_boundaries_are_candidates(self):
        for kappa in KAPPA_BOUNDS:
            with self.subTest(kappa=kappa):
                yields = bond_yield(.035, kappa, .05, .012, self.maturities)
                fit = fit_yield_curve_profile(self.maturities, yields, .035, .012)
                self.assertAlmostEqual(fit.kappa, kappa, delta=1e-7)
                self.assertTrue(any('kappa' in warning for warning in fit.warnings))

    def test_theta_boundary_and_poor_fit_are_reported(self):
        fit = fit_yield_curve_profile(self.maturities, np.full(8, .5), .035, 0)
        self.assertEqual(fit.theta, .2)
        self.assertTrue(any('theta' in warning for warning in fit.warnings))
        self.assertTrue(any('20 bp' in warning for warning in fit.warnings))

    def test_flat_curve_exposes_broad_minimum(self):
        fit = fit_yield_curve_profile(self.maturities, np.full(8, .03), .03, 0)
        self.assertLess(fit.rmse_bp, 1e-6)
        self.assertGreater(fit.near_optimal_half_life_range[1]/fit.near_optimal_half_life_range[0], 10)
        self.assertTrue(any('broad minimum' in warning for warning in fit.warnings))

    def test_rejects_invalid_curve_inputs(self):
        for maturities, yields, r0, sigma in [
                ([1, 1], [.02, .03], .02, .01), ([0, 1], [.02, .03], .02, .01),
                ([1], [.02], .02, .01), ([1, 2], [.02, np.nan], .02, .01),
                ([1, 2], [.02, .03], np.inf, .01), ([1, 2], [.02, .03], .02, -.01)]:
            with self.subTest(maturities=maturities), self.assertRaises(ValueError):
                fit_yield_curve_profile(maturities, yields, r0, sigma)


if __name__ == '__main__':
    unittest.main()
