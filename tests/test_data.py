"""Checks for Fed date selection, rate units, and missing observations."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data import load_yields, load_yield_history, load_short_rate_history


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
