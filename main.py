"""Run illustrative short-rate dynamics and fit a single zero-coupon yield curve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibration import fit_yield_curve
from data import load_yields
from plots import (plot_rate_distribution, plot_rate_paths,
                   plot_residuals, plot_yield_curve)
from vasicek import bond_yield, rate_moments, simulate_rates


ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "output"
    output.mkdir(exist_ok=True)

    # A/B: Illustrative real-world dynamics; NOT inferred from market yields.
    r0, kappa, theta, sigma = 0.035, 0.4, 0.045, 0.015
    times = np.linspace(0, 10, 251)
    paths = simulate_rates(r0, kappa, theta, sigma, times, 20, np.random.default_rng(42))
    means, stds = rate_moments(r0, kappa, theta, sigma, times)

    figures = {
        "A_rate_paths.png": plot_rate_paths(times, paths, theta),
        "B_rate_distribution.png": plot_rate_distribution(times, means, stds),
    }

    # C/D: Risk-neutral pricing calibration; fixed r0/sigma are assumptions.
    maturities, market_yields = load_yields(ROOT / "data" / "yields.csv")
    pricing_r0, pricing_sigma = 0.035, 0.012
    fitted_kappa, fitted_theta = fit_yield_curve(
        maturities, market_yields, pricing_r0, pricing_sigma
    )
    model_yields = bond_yield(
        pricing_r0, fitted_kappa, fitted_theta, pricing_sigma, maturities
    )
    residuals = model_yields - market_yields
    dense_maturities = np.linspace(0.25, 30, 300)
    dense_yields = bond_yield(
        pricing_r0, fitted_kappa, fitted_theta, pricing_sigma, dense_maturities
    )
    figures["C_yield_curve.png"] = plot_yield_curve(
        dense_maturities, dense_yields, maturities, market_yields
    )
    figures["D_residuals.png"] = plot_residuals(maturities, residuals)

    for filename, fig in figures.items():
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    rmse_bp = np.sqrt(np.mean(residuals**2)) * 10_000
    print("Data: illustrative zero-coupon yields (not observed Treasury data)")
    print(f"Fixed pricing r0: {pricing_r0:.4%}; sigma: {pricing_sigma:.4%}/sqrt(year)")
    print(f"Fitted risk-neutral kappa: {fitted_kappa:.6f} /year")
    print(f"Fitted risk-neutral theta: {fitted_theta:.4%}")
    print(f"Yield RMSE: {rmse_bp:.2f} bp")
    print(f"Saved four figures in {output}")


if __name__ == "__main__":
    main()
