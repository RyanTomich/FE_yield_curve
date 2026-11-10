"""Load a dated cross-section of Fed continuously compounded spot yields."""

import csv
from dataclasses import dataclass
from datetime import date as calendar_date, timedelta
from itertools import chain
from pathlib import Path

import numpy as np


def load_yields(path, date=None):
    """Return (selected_date, maturities, decimal_yields).

    Select an exact ISO date, or the latest row with at least two spot yields.
    Missing maturities are omitted; they are never filled or treated as zero.
    """
    if date is not None:
        try:
            if calendar_date.fromisoformat(date).isoformat() != date:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("date must use YYYY-MM-DD") from None
    columns = [f"SVENY{n:02d}" for n in range(1, 31)]
    selected = None
    with open(path, newline="", encoding="utf-8-sig") as source:
        header = next((line for line in source if line.startswith("Date,")), None)
        if header is None:
            raise ValueError("Fed CSV must contain a Date header")
        rows = csv.DictReader(chain([header], source))
        if not set(columns).issubset(rows.fieldnames):
            raise ValueError("Fed CSV must contain SVENY01 through SVENY30")
        for row in rows:
            row_date = row["Date"]
            if date is not None and row_date != date:
                continue
            points = []
            for maturity, column in enumerate(columns, start=1):
                value = row[column]
                if value is None or value.strip() in ("", "NA"):
                    continue
                rate = float(value) / 100
                if not np.isfinite(rate):
                    raise ValueError(f"Nonfinite {column} yield on {row_date}")
                points.append((maturity, rate))
            if len(points) < 2:
                if date is not None:
                    raise ValueError(f"Fewer than two available spot yields on {date}")
                continue
            if selected is None or row_date > selected[0]:
                values = np.asarray(points, dtype=float)
                selected = (row_date, values[:, 0], values[:, 1])
    if selected is None:
        raise ValueError(f"No usable Fed yield curve for {date or 'any date'}")
    return selected


def load_yield_history(path, date=None, lookback_days=60):
    """Return (dates, maturities, panel) in [selected - lookback_days, selected].

    Select the endpoint using load_yields' exact-date rules. Missing cells stay
    NaN, empty dates are skipped, and observations after the endpoint are never
    included. The returned columns include every maturity available in-window.
    """
    if not isinstance(lookback_days, int) or lookback_days < 1:
        raise ValueError("lookback_days must be a positive integer")
    endpoint, _, _ = load_yields(path, date)
    start = (calendar_date.fromisoformat(endpoint)
             - timedelta(days=lookback_days)).isoformat()
    columns = [f"SVENY{n:02d}" for n in range(1, 31)]
    observations = {}
    with open(path, newline="", encoding="utf-8-sig") as source:
        header = next(line for line in source if line.startswith("Date,"))
        for row in csv.DictReader(chain([header], source)):
            row_date = row["Date"]
            if not start <= row_date <= endpoint:
                continue
            values = np.full(30, np.nan)
            for i, column in enumerate(columns):
                value = row[column]
                if value is None or value.strip() in ("", "NA"):
                    continue
                values[i] = float(value) / 100
                if not np.isfinite(values[i]):
                    raise ValueError(f"Nonfinite {column} yield on {row_date}")
            if np.isfinite(values).sum() >= 2:
                if row_date in observations:
                    raise ValueError(f"Duplicate curve date: {row_date}")
                observations[row_date] = values
    dates = sorted(observations)
    panel = np.array([observations[day] for day in dates])
    available = np.any(np.isfinite(panel), axis=0)
    return np.array(dates), np.arange(1, 31, dtype=float)[available], panel[:, available]


@dataclass
class ShortRateHistory:
    dates: np.ndarray
    rates: np.ndarray
    quotes_percent: np.ndarray
    source_path: str

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
    if not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or lookback_days < 1:
        raise ValueError("lookback_days must be a positive integer")
    start = (endpoint - timedelta(days=lookback_days)).isoformat()
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
