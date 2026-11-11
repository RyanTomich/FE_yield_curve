"""Vasicek risk-neutral zero-coupon bond pricing for yield-curve calibration.

Rates are decimal fractions, and all times and maturities are in years.
"""

import numpy as np


def _check_parameters(kappa, sigma):
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be positive and finite")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be nonnegative and finite")


def _check_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("times must be nonnegative and finite")
    return times


def bond_price(r0, kappa, theta, sigma, maturities):
    """Risk-neutral prices of zero-coupon bonds paying 1 at maturity."""
    _check_parameters(kappa, sigma)
    maturities = _check_times(maturities)
    b = -np.expm1(-kappa * maturities) / kappa
    log_a = (
        (theta - sigma**2 / (2 * kappa**2)) * (b - maturities)
        - sigma**2 * b**2 / (4 * kappa)
    )
    return np.exp(log_a - b * r0)


def bond_yield(r0, kappa, theta, sigma, maturities):
    """Continuously compounded spot yields, with y(0) defined as r0."""
    maturities = _check_times(maturities)
    prices = bond_price(r0, kappa, theta, sigma, maturities)
    yields = np.full(maturities.shape, float(r0))
    np.divide(-np.log(prices), maturities, out=yields, where=maturities > 0)
    return yields


def simulate_short_rates(r0, kappa, theta, sigma, *, years=10, steps=120,
                         n_paths=1000, seed=42):
    """Exact Vasicek transitions under the measure of the supplied parameters.

    Curve-calibrated parameters give Q pricing scenarios, not P forecasts.
    Return times and a (n_paths, steps + 1) array; negative rates are allowed.
    """
    _check_parameters(kappa, sigma)
    if not all(np.isfinite(value) for value in (r0, theta, years)) or years <= 0:
        raise ValueError('r0 and theta must be finite; years must be positive and finite')
    for value in (steps, n_paths):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError('steps and n_paths must be positive integers')
    times = np.linspace(0, years, steps + 1)
    dt = years / steps
    decay = np.exp(-kappa * dt)
    shock_sd = sigma * np.sqrt(-np.expm1(-2 * kappa * dt) / (2 * kappa))
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, steps + 1))
    paths[:, 0] = r0
    for step in range(steps):
        paths[:, step + 1] = theta + (paths[:, step] - theta) * decay + shock_sd * rng.standard_normal(n_paths)
    return times, paths
