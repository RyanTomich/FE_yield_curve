"""Presentation-only functions: return figures for main.py to save."""

import matplotlib.pyplot as plt
import numpy as np


def plot_rate_paths(times, paths, theta, date=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, paths.T, alpha=0.5, linewidth=0.9)
    ax.axhline(theta, color="black", linestyle="--", linewidth=1.5, label="Pricing long-run mean")
    ax.set(xlabel="Time from valuation (years)", ylabel="Short rate",
           title=f"A. Risk-neutral paths — {date or 'illustration'}\nPricing scenarios, not forecasts")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rate_distribution(times, means, stds, date=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    z95 = 1.6448536269514722
    ax.fill_between(times, means - z95 * stds, means + z95 * stds,
                    alpha=0.25, label="Pointwise 5th–95th percentile under Q")
    ax.plot(times, means, linewidth=2, label="Expected short rate under Q")
    ax.set(xlabel="Time from valuation (years)", ylabel="Short rate",
           title=f"B. Risk-neutral distribution — {date or 'illustration'}\nPricing scenarios, not forecasts")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_yield_curve(curve_maturities, curve_yields, market_maturities, market_yields, date=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve_maturities, curve_yields, label="Fitted Vasicek curve", linewidth=2)
    ax.scatter(market_maturities, market_yields, color="black", zorder=3,
               label="Fed zero-coupon yields")
    ax.set(xlabel="Maturity (years)", ylabel="Zero-coupon yield", title=f"C. Fitted yield curve{' — ' + date if date else ''}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_yield_curve_all_data(curve_maturities, curve_yields, market_maturities,
                            history, start, end):
    """Show the selected-date curve with history used only as context."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    maturity_grid = np.broadcast_to(market_maturities, history.shape)
    available = np.isfinite(history)
    ax.scatter(maturity_grid[available], history[available], color="black",
               alpha=0.06, s=10, linewidths=0,
               label="Fed history for context; not fitted")
    ax.plot(curve_maturities, curve_yields, linewidth=2, zorder=3,
            label=f"Fitted Vasicek curve — {end}")
    ax.set(xlabel="Maturity (years)", ylabel="Zero-coupon yield",
           title=f"C. Selected curve with historical context\n{start} to {end}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(maturities, residuals, date=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.scatter(maturities, residuals * 10_000, zorder=3)
    ax.plot(maturities, residuals * 10_000, alpha=0.55)
    ax.set(xlabel="Maturity (years)", ylabel="Model − market (bp)",
           title=f"D. Yield fit residuals{' — ' + date if date else ''}")
    fig.tight_layout()
    return fig


def plot_yield_volatility(maturities, observed, model, start, end):
    """Compare historical and model volatility as a diagnostic, without fitting."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    supported = np.isfinite(observed)
    ax.scatter(maturities[supported], observed[supported] * 10_000,
               color='black', s=18, label='Historical estimate (≥20 changes)')
    ax.plot(maturities, model * 10_000, linewidth=2, label='Fitted Vasicek volatility')
    ax.set(xlabel='Maturity (years)', ylabel='Yield volatility (bp / √year)',
           title=f'E. Volatility diagnostic — {start} to {end}\nNot used in curve fitting')
    if not np.any(supported):
        ax.text(.5, .5, 'Insufficient history for observed volatility',
                transform=ax.transAxes, ha='center')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_half_life_profile(kappas, rmse_bp, selected_kappa, selected_rmse_bp, date):
    """Display the profiled objective, not a parameter confidence interval."""
    half_lives = np.log(2) / kappas
    order = np.argsort(half_lives)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(half_lives[order], rmse_bp[order], label='Best theta at each half-life')
    ax.scatter([np.log(2)/selected_kappa], [selected_rmse_bp], color='black',
               zorder=3, label='Selected fit')
    for i, bound in enumerate((half_lives.min(), half_lives.max())):
        ax.axvline(bound, color='grey', linestyle=':',
                   label='Computational boundaries' if i == 0 else None)
    ax.axhline(selected_rmse_bp + 1, color='orange', linestyle='--', label='Best RMSE + 1 bp')
    ax.set(xscale='log', xlabel='Pricing mean-reversion half-life (years)',
           ylabel='Selected-date yield RMSE (bp)',
           title=f'F. Half-life sensitivity — {date}\nSensitivity diagnostic, not a confidence interval')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
