"""Small numerical checks using only the Python standard library."""

import unittest

import numpy as np

from calibration import fit_yield_curve
from vasicek import bond_price, bond_yield, rate_moments, simulate_rates


class VasicekTests(unittest.TestCase):
    def test_simulation_matches_moments(self):
        times = np.linspace(0, 5, 21)
        params = (0.03, 0.6, 0.05, 0.02)
        samples = simulate_rates(*params, times, 40_000, np.random.default_rng(1))
        means, stds = rate_moments(*params, times)
        for i in (5, 10, 20):
            self.assertAlmostEqual(samples[:, i].mean(), means[i], delta=5e-4)
            self.assertAlmostEqual(samples[:, i].std(), stds[i], delta=5e-4)

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
