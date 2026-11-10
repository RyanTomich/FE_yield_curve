"""Vasicek short-rate dynamics and risk-neutral zero-coupon bond pricing.

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


def simulate_rates(r0, kappa, theta, sigma, times, n_paths, rng):
    """Sample exact Vasicek transitions; return (n_paths, n_times) rates."""
    _check_parameters(kappa, sigma)
    times = _check_times(times)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must start at zero and be strictly increasing")
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")

    paths = np.empty((n_paths, times.size), dtype=float)
    paths[:, 0] = r0
    for i, dt in enumerate(np.diff(times), start=1):
        decay = np.exp(-kappa * dt)
        innovation_std = sigma * np.sqrt(-np.expm1(-2 * kappa * dt) / (2 * kappa))
        paths[:, i] = (
            theta + (paths[:, i - 1] - theta) * decay
            + innovation_std * rng.standard_normal(n_paths)
        )
    return paths


def rate_moments(r0, kappa, theta, sigma, times):
    """Return analytical unconditional mean and standard deviation at each time."""
    _check_parameters(kappa, sigma)
    times = _check_times(times)
    decay = np.exp(-kappa * times)
    means = theta + (r0 - theta) * decay
    stds = sigma * np.sqrt(-np.expm1(-2 * kappa * times) / (2 * kappa))
    return means, stds


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
