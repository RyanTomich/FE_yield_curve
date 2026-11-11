"""1. Observe the market: selected curve, past curves, and short-rate history."""

from dataclasses import dataclass
from datetime import date as calendar_date

import numpy as np

from data import (ShortRateHistory, load_curve_short_rate_history,
                  load_short_rate_history, load_yield_history)


@dataclass
class MarketData:
    date: str
    dates: np.ndarray
    maturities: np.ndarray
    history: np.ndarray
    short_rates: ShortRateHistory
    comparisons: list
    warnings: list[str]

    @property
    def selected_curve(self):
        available = np.isfinite(self.history[-1])
        return self.maturities[available], self.history[-1, available]


def comparison_curves(dates, history, date):
    """Use the last available curve on/before each anniversary, within 7 days."""
    selected = calendar_date.fromisoformat(date)
    curves = [('Selected date', date, history[-1])]
    warnings = []
    for years in (1, 3):
        try:
            target = selected.replace(year=selected.year - years)
        except ValueError:  # February 29 in a non-leap year.
            target = selected.replace(year=selected.year - years, day=28)
        eligible = np.flatnonzero(dates <= target.isoformat())
        if eligible.size:
            index = eligible[-1]
            gap = (target - calendar_date.fromisoformat(str(dates[index]))).days
            if gap <= 7:
                curves.append((f'{years} year{"s" if years > 1 else ""} ago',
                               str(dates[index]), history[index]))
                continue
        warnings.append(f'no curve within 7 days on/before {target}; {years}-year comparison omitted')
    return curves, warnings


def observe_market(curve_path, date=None, short_rate_path=None):
    """Load all available past context; calibration chooses its own window."""
    dates, maturities, history = load_yield_history(curve_path, date, lookback_days=None)
    selected = str(dates[-1])
    if short_rate_path is None:
        short_rates = load_curve_short_rate_history(curve_path, selected, lookback_days=None)
    else:
        short_rates = load_short_rate_history(short_rate_path, selected, lookback_days=None)
    comparisons, warnings = comparison_curves(dates, history, selected)
    return MarketData(selected, dates, maturities, history, short_rates, comparisons, warnings)
