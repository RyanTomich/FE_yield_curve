"""Presentation-only functions: return figures for main.py to save."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_yield_curve(curve_maturities, curve_yields, market_maturities, market_yields, date=None, fit=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve_maturities, curve_yields, label="Fitted Vasicek curve", linewidth=2)
    ax.scatter(market_maturities, market_yields, color="black", zorder=3,
               label="Observed zero-coupon yields")
    ax.set(xlabel="Maturity (years)", ylabel="Zero-coupon yield", title=f"2. Calibration | Market vs. Vasicek{' — ' + date if date else ''}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    if fit is not None:
        fig.text(.5, .025,
                 f'r₀ = {fit.r0:.2%}   θ (Q) = {fit.theta:.2%}   κ (Q) = {fit.kappa:.4f}/year\n'
                 f'σ (historical) = {fit.sigma:.2%}/√year   Yield RMSE = {fit.rmse_bp:.2f} bp',
                 ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12 if fit is not None else 0, 1, 1))
    return fig


def plot_yield_curve_history(market):
    """Compare dated market curves without any model overlay."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, day, yields in market.comparisons:
        ax.plot(market.maturities, yields, marker='o', markersize=3,
                linewidth=2, label=f'{label} — {day}')
    ax.set(xlabel="Maturity (years)", ylabel="Zero-coupon yield",
           title=f"1. Market data | Yield curves across dates — {market.date}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    if market.warnings:
        fig.text(.5, .02, 'Some historical comparisons are unavailable; see report.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    return fig


def plot_historical_short_rates(market):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    short = market.short_rates
    ax.plot(np.asarray(short.dates, dtype='datetime64[D]'), short.rates, linewidth=1,
            label=f'{short.series} — three-month short-rate proxy')
    ax.set(xlabel='Date', ylabel='Continuously compounded rate',
           title=f'1. Market data | Historical short rates through {market.date}')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_residuals(maturities, residuals, date=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.scatter(maturities, -residuals * 10_000, zorder=3)
    ax.plot(maturities, -residuals * 10_000, alpha=0.55)
    ax.set(xlabel="Maturity (years)", ylabel="Market − model (bp)",
           title=f"2. Calibration | Yield fit residuals{' — ' + date if date else ''}")
    fig.tight_layout()
    return fig


def plot_short_rate_estimation(market, report):
    """Show the observed proxy and the exact shocks used by the estimator."""
    estimate = report['historically_estimated']
    end = np.datetime64(market.date)
    start = end - np.timedelta64(report['lookback_days'], 'D')
    dates = np.asarray(market.short_rates.dates, dtype='datetime64[D]')
    # Include a little preceding context so the shaded estimation window is visible.
    visible = dates >= start - np.timedelta64(max(7, report['lookback_days'] // 3), 'D')
    fig, (rate_ax, shock_ax) = plt.subplots(figsize=(9, 7), nrows=2, sharex=True)
    rate_ax.plot(dates[visible], market.short_rates.rates[visible], linewidth=1.5,
                 label=f'{market.short_rates.series} — three-month proxy')
    rate_ax.axvspan(start, end, color='tab:blue', alpha=.09,
                   label=f'{report["lookback_days"]}-day estimation window')
    rate_ax.scatter([end], [report['observed_proxy']['continuous_rate']],
                    color='black', zorder=3,
                    label=f'r₀ = {report["observed_proxy"]["continuous_rate"]:.3%} (selected observation)')
    rate_ax.set(ylabel='Continuously compounded rate',
                title=f'2. Calibration | Select r₀ and estimate σ — {market.date}')
    rate_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))
    rate_ax.legend(fontsize=8, loc='best')
    shock_dates = np.asarray(estimate['shock_dates'], dtype='datetime64[D]')
    shocks = np.asarray(estimate['scaled_shocks']) * 10_000
    sigma = estimate['sigma'] * 10_000
    shock_ax.axhline(0, color='grey', linewidth=.8)
    shock_ax.vlines(shock_dates, 0, shocks, color='tab:blue', alpha=.6, linewidth=1)
    shock_ax.scatter(shock_dates, shocks, color='tab:blue', s=14,
                     label='Drift-adjusted changes / √Δt')
    shock_ax.axhline(sigma, color='tab:orange', linestyle='--',
                     label=f'±σ = ±{sigma:.1f} bp / √year (scale references)')
    shock_ax.axhline(-sigma, color='tab:orange', linestyle='--')
    shock_ax.set(xlabel='Observation date', ylabel='Scaled rate shock (bp / √year)',
                 title=f'{estimate["eligible_change_count"]} eligible changes; σ held fixed in the curve fit')
    shock_ax.title.set_fontsize(10)
    shock_ax.legend(fontsize=8, loc='best')
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    shock_ax.xaxis.set_major_locator(locator)
    shock_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in (rate_ax, shock_ax):
        ax.grid(alpha=.15)
    fig.text(.5, .02,
             'σ² = sum(scaled shocks²) / (n − 1); ACT/365.25; gaps over 7 days excluded.\n'
             'Constant drift removed; ±σ lines are scale references, not confidence bounds.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    return fig


def plot_rate_sensitivity(sensitivity, date):
    fig, (ax, change_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                      gridspec_kw={'height_ratios': [2, 1]})
    maturities = sensitivity['maturities']
    for label, rate, prices in zip(('−1 percentage point', 'Calibrated baseline', '+1 percentage point'),
                                   sensitivity['short_rates'], sensitivity['prices']):
        ax.plot(maturities, prices, linewidth=2, label=f'{label}: r₀ = {rate:.2%}')
    ax.set(ylabel='Bond price ($100 face value)',
           title=f'3. Applications | Bond prices under short-rate shocks — {date}')
    ax.legend(fontsize=9)
    change_ax.axhline(0, color='grey', linewidth=.8)
    change_ax.plot(maturities, sensitivity['price_change_up_100bp_percent'], color='tab:red')
    change_ax.set(xlabel='Maturity (years)', ylabel='Price change (%)',
                  title='Price response to +1 percentage point in r₀; κ, θ, σ held fixed')
    change_ax.title.set_fontsize(10)
    fig.tight_layout()
    return fig


def plot_simulated_short_rates(simulation, fit, date):
    fig, ax = plt.subplots(figsize=(8, 5))
    times = simulation['times']
    ax.fill_between(times, simulation['percentile_05'], simulation['percentile_95'],
                    alpha=.18, color='tab:blue', label='Pointwise 5th–95th percentiles')
    for index, path in enumerate(simulation['sample_paths']):
        ax.plot(times, path, linewidth=.7, alpha=.4, color='tab:blue',
                label='10 sample paths' if index == 0 else None)
    ax.plot(times, simulation['mean'], linewidth=2, color='black', label='Mean of 1,000 paths')
    ax.axhline(fit.theta, linestyle='--', linewidth=1, color='tab:orange', label='Risk-neutral long-run rate θ')
    ax.set(xlabel='Years from selected date', ylabel='Short rate',
           title=f'3. Applications | Risk-neutral short-rate scenarios — {date}')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.legend(fontsize=8, loc='best')
    fig.text(.5, .02, f'Pricing scenarios, not real-world forecasts. Mean-reversion half-life: {fit.half_life:.2f} years.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig
