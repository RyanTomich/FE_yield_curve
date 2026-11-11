"""Load dated zero-yield curves and short-rate proxies from local CSV files."""

import csv
from dataclasses import dataclass
from datetime import date as calendar_date, timedelta
from itertools import chain
from pathlib import Path

import numpy as np


def _validate_date(day):
    try:
        if calendar_date.fromisoformat(day).isoformat() != day:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(f"date must use YYYY-MM-DD: {day!r}") from None


def _read_curve(path, start=None, end=None):
    """Read Fed percentage columns or maturity-labelled decimal zero yields."""
    observations = {}
    with open(path, newline="", encoding="utf-8-sig") as source:
        header = next((line for line in source if line.startswith("Date,")), None)
        if header is None:
            raise ValueError("curve CSV must contain a Date header")
        rows = csv.DictReader(chain([header], source))
        fields = rows.fieldnames
        if len(set(fields)) != len(fields):
            raise ValueError("duplicate curve column")
        if any(column.startswith('SVENY') for column in fields):
            columns = [f"SVENY{n:02d}" for n in range(1, 31)]
            if not set(columns).issubset(fields):
                raise ValueError("Fed CSV must contain SVENY01 through SVENY30")
            maturities = np.arange(1, 31, dtype=float)
            scale = .01
        else:
            columns = fields[1:]
            try:
                if not columns or any(not column.endswith('Y') for column in columns):
                    raise ValueError
                maturities = np.array([float(column[:-1]) for column in columns])
            except ValueError:
                raise ValueError("zero-curve columns must be year maturities such as 0.25Y and 10Y") from None
            if (not np.all(np.isfinite(maturities)) or np.any(maturities <= 0)
                    or len(np.unique(maturities)) != len(maturities)):
                raise ValueError("curve maturities must be distinct, positive and finite")
            scale = 1.0  # Already continuously compounded decimal yields.
        order = np.argsort(maturities)
        for row in rows:
            day = row['Date']
            _validate_date(day)
            if (start is not None and day < start) or (end is not None and day > end):
                continue
            if day in observations:
                raise ValueError(f"Duplicate curve date: {day}")
            if None in row or any(row[column] is None for column in columns):
                raise ValueError(f"malformed curve row on {day}")
            values = np.full(len(columns), np.nan)
            for i, column in enumerate(columns):
                value = row[column].strip()
                if value in ('', '.', 'NA'):
                    continue
                try:
                    values[i] = float(value) * scale
                except ValueError:
                    raise ValueError(f"Invalid {column} yield on {day}: {value!r}") from None
                if not np.isfinite(values[i]):
                    raise ValueError(f"Nonfinite {column} yield on {day}")
            observations[day] = values[order]
    return maturities[order], observations


def load_yields(path, date=None):
    """Return exact/latest usable date, maturities in years, and decimal yields.

    SVENY01–SVENY30 values are percentages. Columns such as 0.08Y and 0.25Y
    contain continuous decimal yields and are used without conversion.
    """
    if date is not None:
        _validate_date(date)
    maturities, observations = _read_curve(path, date, date)
    usable = [day for day, values in observations.items() if np.isfinite(values).sum() >= 2]
    if not usable:
        raise ValueError(f"No usable yield curve with at least two spot yields for {date or 'any date'}")
    selected = max(usable)
    values = observations[selected]
    available = np.isfinite(values)
    return selected, maturities[available], values[available]


def load_yield_history(path, date=None, lookback_days=60):
    """Return dates, maturities and panel for the selected historical window.

    Missing cells remain NaN; rows with fewer than two yields are omitted.
    Set lookback_days=None for all available history through the selected date.
    """
    endpoint, _, _ = load_yields(path, date)
    start = _window_start(endpoint, lookback_days)
    maturities, observations = _read_curve(path, start, endpoint)
    dates = sorted(day for day, values in observations.items() if np.isfinite(values).sum() >= 2)
    panel = np.array([observations[day] for day in dates])
    available = np.any(np.isfinite(panel), axis=0)
    return np.array(dates), maturities[available], panel[:, available]


def _window_start(date, lookback_days):
    """None requests all available history through the selected date."""
    if lookback_days is None:
        return '0001-01-01'
    if not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or lookback_days < 1:
        raise ValueError("lookback_days must be a positive integer")
    return (calendar_date.fromisoformat(date) - timedelta(days=lookback_days)).isoformat()


@dataclass
class ShortRateHistory:
    dates: np.ndarray
    rates: np.ndarray
    quotes_percent: np.ndarray | None
    source_path: str
    series: str = 'DGS3MO'
    conversion: str = '2 * log1p(quote_percent / 200)'
    interpretation: str = 'Three-month Treasury yield used as a short-rate proxy.'


    @property
    def r0(self):
        return float(self.rates[-1])


def load_short_rate_history(path, date, lookback_days=60):
    """Load a local FRED DGS3MO CSV, requiring the exact endpoint observation.

    Quotes are percentage bond-equivalent yields. Convert to continuous rates
    as 2*log(1 + (quote/100)/2). This is still a three-month short-rate proxy,
    not an observation of the instantaneous short rate. Blank, '.', and 'NA'
    entries mean missing. No interpolation, fallback dates, or network access.
    """
    try:
        endpoint = calendar_date.fromisoformat(date)
        if endpoint.isoformat() != date:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("date must use YYYY-MM-DD") from None
    start = _window_start(date, lookback_days)
    path = Path(path)
    try:
        source = path.open(newline='', encoding='utf-8-sig')
    except FileNotFoundError:
        raise ValueError(f"short-rate file not found: {path}. Save a local FRED DGS3MO CSV "
                         "there or pass --short-rate-file PATH; no data is downloaded") from None
    observations = {}
    seen = set()
    with source:
        rows = csv.DictReader(source)
        fields = rows.fieldnames or []
        date_column = next((key for key in ('observation_date', 'DATE') if key in fields), None)
        if date_column is None or 'DGS3MO' not in fields:
            raise ValueError("short-rate CSV must contain DATE or observation_date, and DGS3MO")
        for row in rows:
            day = row.get(date_column, '')
            try:
                if calendar_date.fromisoformat(day).isoformat() != day:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(f"invalid short-rate date: {day!r}") from None
            if not start <= day <= date:
                continue
            if day in seen:
                raise ValueError(f"duplicate short-rate date: {day}")
            seen.add(day)
            quote = row.get('DGS3MO')
            if quote is None:
                raise ValueError(f"malformed DGS3MO row on {day}")
            if quote.strip() in ('', '.', 'NA'):
                continue
            try:
                value = float(quote)
            except ValueError:
                raise ValueError(f"invalid DGS3MO quote on {day}: {quote!r}") from None
            if not np.isfinite(value) or value <= -200:
                raise ValueError(f"invalid DGS3MO quote on {day}: {quote!r}")
            observations[day] = value
    if date not in observations:
        raise ValueError(f"no valid DGS3MO observation on {date}; provide a local file containing "
                         "that exact date (no earlier observation is substituted)")
    dates = np.array(sorted(observations))
    quotes = np.array([observations[day] for day in dates])
    rates = 2 * np.log1p(quotes / 200)
    return ShortRateHistory(dates, rates, quotes, str(path.resolve()))


def load_curve_short_rate_history(path, date, lookback_days=60):
    """Use the 0.25Y continuous decimal yield as the three-month rate proxy."""
    _validate_date(date)
    start = _window_start(date, lookback_days)
    maturities, observations = _read_curve(path, start, date)
    matches = np.flatnonzero(maturities == .25)
    if not len(matches):
        raise ValueError("curve file must contain 0.25Y for the short-rate proxy; "
                         "otherwise pass --short-rate-file PATH to a local DGS3MO CSV")
    index = matches[0]
    if date not in observations or not np.isfinite(observations[date][index]):
        raise ValueError(f"no valid 0.25Y observation on the exact date {date}")
    dates = np.array(sorted(day for day, values in observations.items() if np.isfinite(values[index])))
    rates = np.array([observations[day][index] for day in dates])
    return ShortRateHistory(dates, rates, None, str(Path(path).resolve()),
                            series='0.25Y', conversion='none (continuous decimal yield)',
                            interpretation='Three-month zero yield used as a short-rate proxy.')
