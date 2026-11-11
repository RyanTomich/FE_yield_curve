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
from data import load_short_rate_history
from vasicek import bond_yield


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
            report = app.main(['--date', date, '--curve-file', str(self.curve_path),
                               '--short-rate-file', str(self.root / 'data' / 'DGS3MO.csv'), *extra])
        return report, stdout.getvalue()

    def test_offline_2020_and_2026_outputs(self):
        for date in ['2020-01-03', '2026-06-01']:
            with self.subTest(date=date):
                self.create_inputs(date)
                report, stdout = self.run_app(date)
                self.assertEqual(report['selected_date'], date)
                self.assertEqual(report['lookback_days'], 60)
                self.assertEqual(report['curve_fitted']['fit_dates'], [date])
                self.assertGreaterEqual(report['historically_estimated']['eligible_change_count'], 20)
                self.assertEqual(report['schema_version'], 6)
                self.assertNotIn('scenarios', report)
                self.assertNotIn('curve_score', report)
                self.assertNotIn('short_rates', report)
                self.assertNotIn('bootstrap', stdout)
                self.assertEqual(
                    {str(path.relative_to(self.root / 'output')) for path in (self.root / 'output').rglob('*.png')},
                    {'01_market_data/yield_curve_historical_context.png',
                     '01_market_data/historical_short_rates.png',
                     '02_calibration/short_rate_estimation.png',
                     '02_calibration/fitted_yield_curve.png',
                     '02_calibration/yield_fit_residuals.png',
                     '03_applications/rate_sensitivity.png',
                     '03_applications/simulated_short_rates.png'})
                self.assertEqual(report['applications']['measure'], 'Q')
                self.assertIn('not real-world forecasts', stdout)
                self.assertIn('Parameter', stdout)
                self.assertEqual(report['applications']['simulation']['path_count'], 1000)
                self.assertEqual(json.loads((self.root / 'output' / 'calibration.json').read_text()),
                                 json.loads(json.dumps(report)))
                fit = report['curve_fitted']
                self.assertEqual(len(fit['solver_attempts']), 3)
                self.assertTrue(any(a['success'] for a in fit['solver_attempts']))
                for removed in ('half_life_profile', 'sigma_sensitivity', 'historical_context'):
                    self.assertNotIn(removed, report)
                self.assertEqual(plt.get_fignums(), [])

    def test_decimal_curve_default_and_plot_titles(self):
        path = self.root / 'data' / 'zero_curve_2001_2026.csv'
        maturities = np.array([.08, .25, .5, 1, 5, 10, 30])
        dates = np.datetime64('2026-09-01') + np.arange(60)
        with path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', *(f'{value:g}Y' for value in maturities)])
            for i, date in enumerate(dates):
                yields = bond_yield(.04 + .0001*np.sin(i), .2, .05, .004, maturities)
                writer.writerow([str(date), *yields])
        self.addCleanup(plt.close, 'all')
        with patch.object(app, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(plt, 'close'):
            report = app.main([])
            self.assertEqual(len(plt.get_fignums()), 7)
            stage_counts = {'1.': 0, '2.': 0, '3.': 0}
            for number in plt.get_fignums():
                fig = plt.figure(number)
                title = fig.axes[0].get_title()
                stage_counts[title[:2]] += 1
                if 'Yield curves across dates' in title:
                    self.assertEqual(len(fig.axes[0].lines), 1)
                    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), yields)
                    self.assertNotIn('Vasicek', str(fig.axes[0].get_legend_handles_labels()))
                if 'Yield fit residuals' in title:
                    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets()[:, 1],
                                               report['curve_fitted']['residuals_bp'])
                if 'Select r₀ and estimate σ' in title:
                    shocks = np.asarray(fig.axes[1].collections[1].get_offsets()[:, 1], dtype=float) / 10000
                    estimate = report['historically_estimated']
                    self.assertEqual(shocks.size, estimate['eligible_change_count'])
                    self.assertAlmostEqual(np.sqrt(np.sum(shocks**2)/(shocks.size-1)), estimate['sigma'])
            self.assertEqual(stage_counts, {'1.': 2, '2.': 3, '3.': 2})
        plt.close('all')
        self.assertEqual(report['selected_date'], str(dates[-1]))
        np.testing.assert_allclose(report['curve_fitted']['maturities'], maturities)
        np.testing.assert_allclose(report['curve_fitted']['market_yields'], yields)
        self.assertEqual(report['observed_proxy']['series'], '0.25Y')
        self.assertIsNone(report['observed_proxy']['quote_percent'])
        self.assertEqual(report['observed_proxy']['continuous_rate'], yields[1])
        self.assertIn('none', report['observed_proxy']['conversion'])
        self.assertEqual(report['historically_estimated']['observation_count'], 60)
        fit = report['curve_fitted']
        np.testing.assert_allclose(fit['residuals_bp'],
                                   (np.array(fit['market_yields']) - fit['model_yields']) * 10000)

    def test_historical_curves_do_not_enter_selected_date_fit(self):
        self.create_inputs('2020-01-03')
        first, _ = self.run_app('2020-01-03')
        self.create_inputs('2020-01-03', old_curve_shift=.01)
        second, _ = self.run_app('2020-01-03')
        self.assertEqual(first['curve_fitted'], second['curve_fitted'])

    def test_cli_rejects_missing_rate_data_and_removed_option(self):
        self.create_inputs('2020-01-03')
        path = self.root / 'data' / 'DGS3MO.csv'
        path.unlink()
        cases = [([], 'short-rate file not found'),
                 (['--volatility-weight', '1'], 'was removed'),
                 (['--volatility-weight'], 'was removed'),
                 (['--diagnostics'], '--diagnostics was removed')]
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

    def test_obsolete_outputs_are_removed_after_success(self):
        self.create_inputs('2020-01-03')
        self.run_app('2020-01-03')
        obsolete = [self.root / 'output' / 'fitted_yield_curve.png']
        for folder in ('output', 'output/02_calibration'):
            for name in ('parameter_selection.png', 'half_life_sensitivity.png', 'yield_volatility.png'):
                obsolete.append(self.root / folder / name)
        for path in obsolete:
            path.write_bytes(b'old generated plot')
        report, _ = self.run_app('2020-01-03')
        self.assertTrue(all(not path.exists() for path in obsolete))
        self.assertEqual(len(report['outputs']), 7)
        self.assertEqual(len(list((self.root / 'output').rglob('*.png'))), 7)


if __name__ == '__main__':
    unittest.main()
