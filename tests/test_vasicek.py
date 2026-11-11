"""Small numerical checks using only the Python standard library."""

import unittest

import numpy as np

from calibration import fit_yield_curve
from vasicek import bond_price, bond_yield, simulate_short_rates


class VasicekTests(unittest.TestCase):
    def test_simulation_matches_analytic_moments(self):
        r0, kappa, theta, sigma = .03, .4, .05, .02
        times, paths = simulate_short_rates(r0, kappa, theta, sigma,
                                           years=5, steps=20, n_paths=50000, seed=7)
        expected_mean = theta + (r0 - theta) * np.exp(-kappa * times)
        expected_variance = sigma**2 * (1 - np.exp(-2*kappa*times)) / (2*kappa)
        np.testing.assert_allclose(paths.mean(axis=0), expected_mean, atol=.0004)
        np.testing.assert_allclose(paths.var(axis=0), expected_variance, rtol=.025, atol=1e-12)
        np.testing.assert_array_equal(paths[:, 0], r0)

    def test_deterministic_simulation_and_reproducible_randomness(self):
        times, paths = simulate_short_rates(-.01, .4, .03, 0, n_paths=3)
        expected = .03 + (-.01 - .03) * np.exp(-.4 * times)
        for path in paths:
            np.testing.assert_allclose(path, expected, atol=1e-14)
        params = (.03, .4, .05, .02)
        first = simulate_short_rates(*params, n_paths=3, seed=1)[1]
        np.testing.assert_array_equal(first, simulate_short_rates(*params, n_paths=3, seed=1)[1])
        self.assertFalse(np.array_equal(first, simulate_short_rates(*params, n_paths=3, seed=2)[1]))

    def test_simulation_rejects_invalid_inputs(self):
        for kwargs in ({'years': 0}, {'steps': 0}, {'n_paths': 1.5}, {'n_paths': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_short_rates(.03, .4, .05, .01, **kwargs)

    def test_bond_limits(self):
        params = (0.04, 0.5, 0.045, 0.01)
        self.assertAlmostEqual(bond_price(*params, [0])[0], 1.0)
        yields = bond_yield(*params, [0, 1e-6])
        np.testing.assert_allclose(yields, 0.04, atol=1e-6)

    def test_synthetic_parameter_recovery(self):
        maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 20, 30])
        r0, kappa, theta, sigma = 0.035, 0.42, 0.05, 0.012
        yields = bond_yield(r0, kappa, theta, sigma, maturities)
        fitted_kappa, fitted_theta = fit_yield_curve(maturities, yields, r0, sigma)
        self.assertAlmostEqual(fitted_kappa, kappa, delta=1e-5)
        self.assertAlmostEqual(fitted_theta, theta, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
