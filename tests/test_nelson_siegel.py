"""Fixed-decay curve calibration and deterministic coefficient stresses."""

import unittest

import numpy as np

from applications import apply_nelson_siegel
from calibration import calibrate_nelson_siegel
from nelson_siegel import discount_factor, factor_loadings, zero_yield


class NelsonSiegelTests(unittest.TestCase):
    def setUp(self):
        self.maturities = np.array([.08, .25, .5, 1, 2, 5, 10, 20, 30])
        self.betas = np.array([.04, -.05, -.02])

    def test_synthetic_parameter_recovery_including_negative_yields(self):
        yields = zero_yield(self.maturities, *self.betas)
        self.assertLess(yields[0], 0)
        fit = calibrate_nelson_siegel(self.maturities, yields)
        np.testing.assert_allclose([fit.beta0, fit.beta1, fit.beta2], self.betas, atol=1e-14)
        np.testing.assert_allclose(fit.model_yields, yields, atol=1e-14)
        np.testing.assert_allclose(fit.residuals, 0, atol=1e-14)
        self.assertLess(fit.rmse_bp, 1e-9)
        self.assertEqual(fit.decay, .5)

    def test_zero_and_small_maturity_limits(self):
        times = np.array([0, 1e-14, 1e-10])
        np.testing.assert_array_equal(factor_loadings([0]), [[1, 1, 0]])
        yields = zero_yield(times, *self.betas)
        self.assertTrue(np.all(np.isfinite(yields)))
        np.testing.assert_allclose(yields, self.betas[0] + self.betas[1], atol=1e-11)
        discounts = discount_factor(times, *self.betas)
        self.assertEqual(discounts[0], 1)
        np.testing.assert_allclose(discounts, np.exp(-times * yields))
        self.assertAlmostEqual(zero_yield([1e8], *self.betas)[0], self.betas[0], places=8)

    def test_invalid_calibration_observations(self):
        cases = [([1, 2], [.02, .03]),
                 ([1, 2, 3], [.02, .03]),
                 ([1, 1, 3], [.02, .03, .04]),
                 ([0, 1, 2], [.02, .03, .04]),
                 ([-1, 1, 2], [.02, .03, .04]),
                 ([1, 2, np.inf], [.02, .03, .04]),
                 ([1, 2, 3], [.02, np.nan, .04]),
                 ([1, 2, 3], [.02, np.inf, .04]),
                 ([[1, 2, 3]], [[.02, .03, .04]])]
        for maturities, yields in cases:
            with self.subTest(maturities=maturities, yields=yields), self.assertRaises(ValueError):
                calibrate_nelson_siegel(maturities, yields)

    def test_rank_deficient_calibration(self):
        # Distinct long maturities have numerically indistinguishable slope/curvature columns.
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            calibrate_nelson_siegel([1000, 2000, 3000], [.03, .04, .05])

    def test_model_rejects_invalid_inputs(self):
        for times in ([], [-1], [np.nan], [[1]]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                zero_yield(times, *self.betas)
        for decay in (0, -1, np.inf, np.nan):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                factor_loadings([1], decay)
        with self.assertRaises(ValueError):
            zero_yield([1], np.nan, 0, 0)

    def test_noisy_fit_residual_sign_and_least_squares(self):
        yields = zero_yield(self.maturities, *self.betas)
        yields[3] += .001
        fit = calibrate_nelson_siegel(self.maturities, yields)
        np.testing.assert_allclose(fit.residuals, fit.model_yields - yields)
        self.assertAlmostEqual(fit.rmse_bp, 10_000 * np.sqrt(np.mean(fit.residuals**2)))
        # Least-squares residuals must be orthogonal to every loading column.
        np.testing.assert_allclose(factor_loadings(self.maturities).T @ fit.residuals, 0, atol=1e-14)

    def test_factor_stresses_and_analytic_price_ratio(self):
        fit = calibrate_nelson_siegel(self.maturities, zero_yield(self.maturities, *self.betas))
        result = apply_nelson_siegel(fit, self.maturities)
        yields = np.array(result['yields'])
        prices = np.array(result['prices'])
        changes = np.array(result['price_change_percent'])
        np.testing.assert_allclose(yields[1] - yields[0], .01)
        np.testing.assert_allclose(prices[1] / prices[0], np.exp(-.01 * self.maturities))
        np.testing.assert_allclose(yields[2] - yields[0],
                                   .01 * (1 - np.exp(-.5 * self.maturities)) / (.5 * self.maturities))
        np.testing.assert_allclose(yields[3] - yields[0],
                                   yields[2] - yields[0] - .01 * np.exp(-.5 * self.maturities))
        np.testing.assert_allclose(prices, 100 * np.exp(-self.maturities * yields))
        np.testing.assert_allclose(changes, 100 * (prices / prices[0] - 1))
        np.testing.assert_array_equal(changes[0], 0)
        np.testing.assert_allclose([fit.beta0, fit.beta1, fit.beta2], self.betas)
        self.assertNotIn('measure', result)
        self.assertNotIn('simulation', result)


if __name__ == '__main__':
    unittest.main()
