"""Historical context uses past dates and stays separate from calibration."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from market import comparison_curves, observe_market
from calibration import calibrate_market


class MarketTests(unittest.TestCase):
    def test_anniversaries_use_prior_observation_and_handle_leap_day(self):
        dates = np.array(['2021-02-26', '2023-02-27', '2023-03-01', '2024-02-29'])
        history = np.arange(8).reshape(4, 2)
        curves, warnings = comparison_curves(dates, history, '2024-02-29')
        self.assertEqual([curve[1] for curve in curves], ['2024-02-29', '2023-02-27', '2021-02-26'])
        np.testing.assert_array_equal(curves[1][2], history[1])
        self.assertEqual(warnings, [])

    def test_missing_comparisons_are_not_replaced_with_distant_dates(self):
        dates = np.array(['2020-01-01', '2024-02-29'])
        curves, warnings = comparison_curves(dates, np.ones((2, 2)), '2024-02-29')
        self.assertEqual(len(curves), 1)
        self.assertEqual(len(warnings), 2)

    def test_full_display_history_does_not_expand_calibration_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'curves.csv'
            dates = np.datetime64('2024-01-01') + np.arange(121)
            def write(old_rate):
                lines = ['Date,0.25Y,1Y,10Y', f'2020-01-01,{old_rate},.03,.04']
                lines += [f'{day},{.03 + .001*np.sin(i)},.035,.04' for i, day in enumerate(dates)]
                lines += ['2024-05-01,bad,bad,bad']  # Excluded future observation.
                path.write_text('\n'.join(lines) + '\n')
            write(.1)
            market = observe_market(path, str(dates[-1]))
            first_fit, first = calibrate_market(market, path, 60)
            self.assertEqual(market.short_rates.dates[0], '2020-01-01')
            self.assertEqual(first['historically_estimated']['observation_count'], 61)
            self.assertEqual(market.date, str(dates[-1]))
            write(.5)
            second_fit, second = calibrate_market(observe_market(path, str(dates[-1])), path, 60)
            self.assertEqual(first['curve_fitted'], second['curve_fitted'])
            self.assertEqual(first_fit.sigma, second_fit.sigma)
