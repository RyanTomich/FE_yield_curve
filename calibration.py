"""Calibrate two risk-neutral Vasicek parameters to a zero-coupon curve."""

import numpy as np
from scipy.optimize import least_squares

from vasicek import bond_yield


def fit_yield_curve(maturities, market_yields, r0, sigma):
    """Fit risk-neutral (kappa, theta) by least-squares yield residuals.

    r0 and sigma are held fixed. This fit does not estimate physical dynamics.
    """
    maturities = np.asarray(maturities, dtype=float)
    market_yields = np.asarray(market_yields, dtype=float)
    if (maturities.ndim != 1 or market_yields.shape != maturities.shape
            or maturities.size < 2 or not np.all(np.isfinite(maturities))
            or np.any(maturities <= 0) or not np.all(np.isfinite(market_yields))):
        raise ValueError("provide matching finite yields at at least two positive maturities")
    if np.unique(maturities).size != maturities.size:
        raise ValueError("maturities must be distinct")

    def residuals(params):
        kappa, theta = params
        return bond_yield(r0, kappa, theta, sigma, maturities) - market_yields

    initial_theta = float(np.clip(market_yields[np.argmax(maturities)], -0.19, 0.19))
    result = least_squares(
        residuals, x0=[0.3, initial_theta],
        bounds=([0.005, -0.2], [5.0, 0.2]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"curve calibration failed: {result.message}")
    return tuple(float(value) for value in result.x)
