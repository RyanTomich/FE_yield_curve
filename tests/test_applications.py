"""Pricing scenarios preserve fitted parameters and use actual rate shocks."""

import unittest

import numpy as np

from applications import apply_model
from calibration import calibrate_yield_curve
from vasicek import bond_yield


class ApplicationTests(unittest.TestCase):
    def test_rate_shock_matches_analytic_price_ratio(self):
        maturities = np.array([.25, 1, 5, 10, 30])
        fit = calibrate_yield_curve(maturities, bond_yield(.03, .4, .05, .01, maturities), .03, .01)
        result = apply_model(fit, maturities)
        sensitivity = result['sensitivity']
        prices = np.array(sensitivity['prices'])
        loading = -np.expm1(-fit.kappa * maturities) / fit.kappa
        np.testing.assert_allclose(prices[2] / prices[1], np.exp(-.01 * loading))
        self.assertTrue(np.all(prices[0] > prices[1]))
        self.assertTrue(np.all(prices[1] > prices[2]))
        self.assertTrue(np.all(np.diff(sensitivity['price_change_up_100bp_percent']) < 0))
        self.assertEqual(result['measure'], 'Q')
        self.assertEqual(len(result['simulation']['sample_paths']), 10)

