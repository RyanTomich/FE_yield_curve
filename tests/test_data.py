"""Checks for Fed date selection, rate units, and missing observations."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data import (load_yields, load_yield_history, load_short_rate_history,
                  load_curve_short_rate_history)


class FedDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fed.csv'
        columns = ['Date'] + [f'SVENY{n:02d}' for n in range(1, 31)]
        with self.path.open('w', newline='') as file:
            file.write('Research data\n\nSeries,Convention\n\n')
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for date, values in [
                ('2026-10-29', {1: '4.5', 30: '5.0'}),
                ('2026-10-28', {1: '4.0', 2: '4.2'}),
                ('2026-10-30', {}),
            ]:
                writer.writerow({'Date': date, **{
                    f'SVENY{n:02d}': values.get(n, 'NA') for n in range(1, 31)
                }})

    def test_latest_usable_row_and_percentage_conversion(self):
        date, maturities, yields = load_yields(self.path)
        self.assertEqual(date, '2026-10-29')
        np.testing.assert_array_equal(maturities, [1, 30])
        np.testing.assert_allclose(yields, [.045, .05])

    def test_exact_date_with_partial_maturities(self):
        date, maturities, yields = load_yields(self.path, '2026-10-28')
        self.assertEqual(date, '2026-10-28')
        np.testing.assert_array_equal(maturities, [1, 2])
        np.testing.assert_allclose(yields, [.04, .042])

    def test_missing_empty_and_invalid_dates_fail(self):
        for date in ['2026-10-27', '2026-10-30', '10/29/2026']:
            with self.subTest(date=date), self.assertRaises(ValueError):
                load_yields(self.path, date)

    def test_history_sorts_and_keeps_missing_cells(self):
        dates, maturities, panel = load_yield_history(self.path)
        np.testing.assert_array_equal(dates, ['2026-10-28', '2026-10-29'])
        np.testing.assert_array_equal(maturities, [1, 2, 30])
        np.testing.assert_allclose(panel, [[.04, .042, np.nan], [.045, np.nan, .05]])

    def test_history_window_and_no_future_observations(self):
        columns = ['Date'] + [f'SVENY{n:02d}' for n in range(1, 31)]
        with self.path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for date in ['2020-01-06', '2020-01-03', '2019-01-03', '2019-01-02']:
                value = 'invalid future value' if date > '2020-01-03' else '2.0'
                writer.writerow({'Date': date, 'SVENY01': value, 'SVENY02': value})
        dates, _, panel = load_yield_history(self.path, '2020-01-03', lookback_days=365)
        np.testing.assert_array_equal(dates, ['2019-01-03', '2020-01-03'])
        np.testing.assert_allclose(panel, .02)

    def test_history_rejects_invalid_window_and_empty_endpoint(self):
        for days in [0, -1, 1.5]:
            with self.subTest(days=days), self.assertRaises(ValueError):
                load_yield_history(self.path, lookback_days=days)
        with self.assertRaises(ValueError):
            load_yield_history(self.path, '2026-10-30')


class ShortRateDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'DGS3MO.csv'

    def write(self, rows, date_column='observation_date'):
        with self.path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([date_column, 'DGS3MO'])
            writer.writerows(rows)

    def test_both_headers_conversion_sorting_and_missing_quotes(self):
        for header in ['observation_date', 'DATE']:
            with self.subTest(header=header):
                self.write([['2020-01-03', '1.5'], ['2019-12-30', '2.0'],
                            ['2019-12-31', '.'], ['2020-01-01', 'NA'], ['2020-01-02', '']], header)
                result = load_short_rate_history(self.path, '2020-01-03')
                np.testing.assert_array_equal(result.dates, ['2019-12-30', '2020-01-03'])
                np.testing.assert_allclose(result.quotes_percent, [2, 1.5])
                np.testing.assert_allclose(result.rates, 2*np.log1p(np.array([2, 1.5])/200))
                self.assertEqual(result.r0, result.rates[-1])
                self.assertEqual(result.source_path, str(self.path.resolve()))

    def test_window_and_future_values_excluded(self):
        self.write([['2019-11-03', 'bad'], ['2019-11-04', '2'], ['2020-01-03', '1.5'],
                    ['2020-01-06', 'invalid future value']])
        result = load_short_rate_history(self.path, '2020-01-03', 60)
        np.testing.assert_array_equal(result.dates, ['2019-11-04', '2020-01-03'])

    def test_missing_endpoint_is_not_substituted(self):
        for missing in ['', '.', 'NA']:
            self.write([['2020-01-02', '2'], ['2020-01-03', missing]])
            with self.assertRaisesRegex(ValueError, 'exact date'):
                load_short_rate_history(self.path, '2020-01-03')

    def test_duplicates_and_invalid_values_fail(self):
        cases = [
            [['2020-01-03', '2'], ['2020-01-03', '.']],
            [['01/03/2020', '2']], [['2020-01-03', 'inf']],
            [['2020-01-03', 'nan']], [['2020-01-03', '-200']],
            [['2020-01-03', 'bad']], [['2020-01-03']],
        ]
        for rows in cases:
            self.write(rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_short_rate_history(self.path, '2020-01-03')

    def test_missing_file_schema_window_and_date_errors(self):
        with self.assertRaisesRegex(ValueError, '--short-rate-file'):
            load_short_rate_history(self.path, '2020-01-03')
        self.path.write_text('date,rate\n2020-01-03,2\n')
        with self.assertRaisesRegex(ValueError, 'DGS3MO'):
            load_short_rate_history(self.path, '2020-01-03')
        for days in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                load_short_rate_history(self.path, '2020-01-03', days)
        for day in ['01/03/2020', '2020-13-01']:
            with self.assertRaises(ValueError):
                load_short_rate_history(self.path, day)


class DecimalCurveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'zero.csv'
        self.path.write_text('Date,30Y,0.25Y,0.08Y,1Y\n'
                             '2026-10-28,0.05,0.041,0.04,NA\n'
                             '2026-10-30,0.054,0.042,0.039,0.044\n'
                             '2026-10-29,0.053,.,0.038,0.043\n')

    def test_decimal_units_sorted_fractional_maturities_and_missing_cells(self):
        date, maturities, rates = load_yields(self.path)
        self.assertEqual(date, '2026-10-30')
        np.testing.assert_array_equal(maturities, [.08, .25, 1, 30])
        np.testing.assert_allclose(rates, [.039, .042, .044, .054])
        dates, maturities, panel = load_yield_history(self.path)
        np.testing.assert_array_equal(dates, ['2026-10-28', '2026-10-29', '2026-10-30'])
        np.testing.assert_allclose(panel, [[.04, .041, np.nan, .05],
                                          [.038, np.nan, .043, .053], [.039, .042, .044, .054]])

    def test_proxy_has_no_second_conversion_and_requires_exact_date(self):
        proxy = load_curve_short_rate_history(self.path, '2026-10-30', 2)
        np.testing.assert_allclose(proxy.rates, [.041, .042])
        self.assertIsNone(proxy.quotes_percent)
        self.assertEqual(proxy.r0, .042)
        with self.assertRaisesRegex(ValueError, 'exact date'):
            load_curve_short_rate_history(self.path, '2026-10-29')
        self.path.write_text('Date,0.08Y,1Y\n2026-10-30,.04,.05\n')
        with self.assertRaisesRegex(ValueError, '0.25Y'):
            load_curve_short_rate_history(self.path, '2026-10-30')

    def test_window_excludes_future_values_and_older_dates(self):
        self.path.write_text(self.path.read_text() + '2026-10-31,bad,bad,bad,bad\n')
        dates, _, panel = load_yield_history(self.path, '2026-10-30', 1)
        np.testing.assert_array_equal(dates, ['2026-10-29', '2026-10-30'])
        self.assertEqual(panel.shape, (2, 4))
        proxy = load_curve_short_rate_history(self.path, '2026-10-30', 1)
        np.testing.assert_allclose(proxy.rates, [.042])

    def test_rejects_duplicate_maturities_dates_and_bad_rows(self):
        for content in [
            'Date,0.25Y,0.250Y\n2026-10-30,.04,.04\n',
            'Date,0Y,1Y\n2026-10-30,.04,.05\n',
            'Date,0.25Y,1Y\n2026-10-30,.04,.05\n2026-10-30,.04,.05\n',
            'Date,0.25Y,1Y\n2026-10-30,inf,.05\n',
            'Date,0.25Y,1Y\n2026-10-30,.04\n',
            'Date,0.25Y,1Y\n10/30/2026,.04,.05\n',
        ]:
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(ValueError):
                    load_yields(self.path)
