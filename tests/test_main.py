"""Offline integration and output checks using explicitly synthetic rate files."""

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import main as app
from calibration import fit_yield_curve_profile
from data import load_short_rate_history
from vasicek import bond_yield, rate_moments


FIXTURES = Path(__file__).parent / 'fixtures'


class StagedIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'data').mkdir()
        self.curve_path = self.root / 'data' / 'feds200628.csv'

    def create_inputs(self, date, old_curve_shift=0):
        fixture = FIXTURES / f'synthetic_dgs3mo_{date[:4]}.csv'
        shutil.copyfile(fixture, self.root / 'data' / 'DGS3MO.csv')
        history = load_short_rate_history(fixture, date)
        maturities = np.arange(1, 31, dtype=float)
        columns = ['Date'] + [f'SVENY{i:02d}' for i in range(1, 31)]
        with self.curve_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            for day, rate in zip(history.dates, history.rates):
                yields = bond_yield(rate, .2, .045, .004, maturities)
                if day != date:
                    yields += old_curve_shift * np.sin(maturities / 5)
                writer.writerow([day, *(yields * 100)])

    def run_app(self, date, extra=()):
        with patch.object(app, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()) as stdout:
            report = app.main(['--date', date, *extra])
        return report, stdout.getvalue()

    def test_offline_2020_and_2026_outputs_and_sensitivity(self):
        for date in ['2020-01-03', '2026-06-01']:
            with self.subTest(date=date):
                self.create_inputs(date)
                with patch.object(app, 'plot_rate_paths', wraps=app.plot_rate_paths) as paths_plot, \
                     patch.object(app, 'plot_rate_distribution', wraps=app.plot_rate_distribution) as bands_plot:
                    report, stdout = self.run_app(date)
                self.assertEqual(report['selected_date'], date)
                self.assertEqual(report['lookback_days'], 60)
                self.assertEqual(report['curve_fitted']['fit_dates'], [date])
                self.assertFalse(report['historical_context']['used_in_curve_objective'])
                self.assertGreaterEqual(report['historically_estimated']['eligible_change_count'], 20)
                self.assertIn('Pricing', report['scenarios']['interpretation'])
                self.assertNotIn('curve_score', report)
                self.assertNotIn('short_rates', report)
                self.assertNotIn('bootstrap', stdout)
                self.assertEqual(len(list((self.root / 'output').glob('*.png'))), 7)
                self.assertEqual(json.loads((self.root / 'output' / 'calibration.json').read_text()),
                                 json.loads(json.dumps(report)))
                fit = report['curve_fitted']
                times, paths, theta, plot_date = paths_plot.call_args.args
                np.testing.assert_allclose(paths[:, 0], fit['r0'])
                self.assertEqual(theta, fit['theta'])
                self.assertEqual(plot_date, date)
                bt, means, stds, plot_date = bands_plot.call_args.args
                expected_mean, expected_std = rate_moments(
                    fit['r0'], fit['kappa'], fit['theta'], fit['sigma'], bt)
                np.testing.assert_allclose(means, expected_mean)
                np.testing.assert_allclose(stds, expected_std)
                self.assertEqual(plot_date, date)
                for case in report['sigma_sensitivity']:
                    self.assertEqual(case['sigma'], fit['sigma'] * case['sigma_multiplier'])
                    reference = fit_yield_curve_profile(fit['maturities'], fit['market_yields'],
                                                        fit['r0'], case['sigma'])
                    self.assertAlmostEqual(case['kappa'], reference.kappa)
                    self.assertAlmostEqual(case['theta'], reference.theta)
                    self.assertAlmostEqual(case['delta_rmse_bp'], case['rmse_bp'] - fit['rmse_bp'])
                self.assertEqual(plt.get_fignums(), [])

    def test_historical_curves_do_not_enter_selected_date_fit(self):
        self.create_inputs('2020-01-03')
        first, _ = self.run_app('2020-01-03')
        self.create_inputs('2020-01-03', old_curve_shift=.01)
        second, _ = self.run_app('2020-01-03')
        self.assertEqual(first['curve_fitted'], second['curve_fitted'])
        self.assertEqual(first['half_life_profile'], second['half_life_profile'])
        self.assertNotEqual(first['historical_context']['observed_annualized_yield_volatility'],
                            second['historical_context']['observed_annualized_yield_volatility'])

    def test_cli_rejects_missing_rate_data_and_removed_option(self):
        self.create_inputs('2020-01-03')
        path = self.root / 'data' / 'DGS3MO.csv'
        path.unlink()
        cases = [([], 'short-rate file not found'),
                 (['--volatility-weight', '1'], 'was removed'),
                 (['--volatility-weight'], 'was removed')]
        for extra, message in cases:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    self.run_app('2020-01-03', extra)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn('Traceback', stderr.getvalue())
        self.assertFalse((self.root / 'output').exists())
        self.create_inputs('2020-01-03')
        rows = path.read_text().splitlines()
        path.write_text('\n'.join(rows[:-1]) + '\n')
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            self.run_app('2020-01-03')
        self.assertIn('exact date', stderr.getvalue())
        self.assertFalse((self.root / 'output').exists())

    def test_custom_short_rate_path(self):
        self.create_inputs('2020-01-03')
        default = self.root / 'data' / 'DGS3MO.csv'
        custom = self.root / 'my local rates.csv'
        default.rename(custom)
        report, _ = self.run_app('2020-01-03', ['--short-rate-file', str(custom)])
        self.assertEqual(report['observed_proxy']['source_path'], str(custom.resolve()))


if __name__ == '__main__':
    unittest.main()
