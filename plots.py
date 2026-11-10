"""Presentation-only functions: return figures for main.py to save."""

import matplotlib.pyplot as plt


def plot_rate_paths(times, paths, theta):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, paths.T, alpha=0.5, linewidth=0.9)
    ax.axhline(theta, color="black", linestyle="--", linewidth=1.5, label="Long-run mean")
    ax.set(xlabel="Time (years)", ylabel="Short rate", title="A. Simulated short-rate paths")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rate_distribution(times, means, stds):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    z95 = 1.6448536269514722
    ax.fill_between(times, means - z95 * stds, means + z95 * stds,
                    alpha=0.25, label="5th–95th percentile")
    ax.plot(times, means, linewidth=2, label="Expected short rate")
    ax.set(xlabel="Time (years)", ylabel="Short rate", title="B. Analytical rate distribution")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_yield_curve(curve_maturities, curve_yields, market_maturities, market_yields):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve_maturities, curve_yields, label="Fitted Vasicek curve", linewidth=2)
    ax.scatter(market_maturities, market_yields, color="black", zorder=3,
               label="Illustrative observations")
    ax.set(xlabel="Maturity (years)", ylabel="Zero-coupon yield", title="C. Fitted yield curve")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(maturities, residuals):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.scatter(maturities, residuals * 10_000, zorder=3)
    ax.plot(maturities, residuals * 10_000, alpha=0.55)
    ax.set(xlabel="Maturity (years)", ylabel="Model − market (bp)",
           title="D. Yield fit residuals")
    fig.tight_layout()
    return fig
