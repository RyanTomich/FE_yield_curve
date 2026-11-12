"""Static Nelson–Siegel zero yields; decimal rates and maturities in years."""

import numpy as np


FIXED_DECAY = 0.5  # Per year, using x = decay * maturity.


def factor_loadings(maturities, decay=FIXED_DECAY):
    """Return level, slope, and curvature columns, including their limits at zero."""
    times = np.asarray(maturities, dtype=float)
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or np.any(times < 0)):
        raise ValueError('maturities must be a nonempty array of finite nonnegative years')
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError('decay must be positive and finite')
    x = decay * times
    slope = np.ones_like(x)
    np.divide(-np.expm1(-x), x, out=slope, where=x != 0)
    curvature = slope - np.exp(-x)
    return np.column_stack((np.ones_like(x), slope, curvature))


def zero_yield(maturities, beta0, beta1, beta2, decay=FIXED_DECAY):
    """Continuously compounded zero yields, with y(0) = beta0 + beta1."""
    betas = np.array([beta0, beta1, beta2], dtype=float)
    if not np.all(np.isfinite(betas)):
        raise ValueError('beta coefficients must be finite')
    return factor_loadings(maturities, decay) @ betas


def discount_factor(maturities, beta0, beta1, beta2, decay=FIXED_DECAY):
    """Discount factors for deterministic cash flows, with D(0) = 1."""
    yields = zero_yield(maturities, beta0, beta1, beta2, decay)
    return np.exp(-np.asarray(maturities, dtype=float) * yields)
