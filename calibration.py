"""Staged Vasicek calibration: historical volatility, then one pricing curve."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from vasicek import bond_yield


# Computational limits, not economic priors.
KAPPA_BOUNDS = (0.005, 5.0)
THETA_BOUNDS = (-0.2, 0.2)
MIN_CHANGES = 20


@dataclass
class VolatilityEstimate:
    sigma: float
    drift: float
    start_date: str
    end_date: str
    observation_count: int
    change_count: int
    warnings: tuple[str, ...]


@dataclass
class CurveFit:
    r0: float
    sigma: float
    kappa: float
    theta: float
    model_yields: np.ndarray
    residuals: np.ndarray
    rmse_bp: float
    profile_kappa: np.ndarray
    profile_theta: np.ndarray
    profile_rmse_bp: np.ndarray
    near_optimal_half_life_range: tuple[float, float]
    warnings: tuple[str, ...]

    @property
    def half_life(self):
        return float(np.log(2) / self.kappa)

    @property
    def limiting_yield(self):
        return float(self.theta - self.sigma**2 / (2 * self.kappa**2))


def _sorted_observations(dates, rates):
    dates = np.asarray(dates, dtype='datetime64[D]')
    rates = np.asarray(rates, dtype=float)
    if (dates.ndim != 1 or rates.shape != dates.shape
            or np.any(np.isnat(dates)) or np.any(np.isinf(rates))):
        raise ValueError("provide matching valid dates and finite rates (NaN denotes missing)")
    order = np.argsort(dates)
    dates, rates = dates[order], rates[order]
    if np.any(np.diff(dates).astype(int) == 0):
        raise ValueError("rate dates must be distinct")
    valid = np.isfinite(rates)
    return dates[valid], rates[valid]


def _estimate_volatility(dates, rates):
    """Shared estimator; None means too few eligible changes, not bad data."""
    dates, rates = _sorted_observations(dates, rates)
    gaps = np.diff(dates).astype(int)
    eligible = (gaps > 0) & (gaps <= 7)
    n = int(eligible.sum())
    if n < MIN_CHANGES:
        return None, n
    dt = gaps[eligible] / 365.25
    changes = np.diff(rates)[eligible]
    drift = float(changes.sum() / dt.sum())
    variance = float(np.sum((changes - drift * dt)**2 / dt) / (n - 1))
    sigma = float(np.sqrt(variance))
    warnings = () if sigma > 0 else (
        "zero historical shock volatility: the window does not inform nonzero volatility",)
    return VolatilityEstimate(sigma, drift, str(dates[0]), str(dates[-1]),
                              int(dates.size), n, warnings), n


def estimate_short_rate_volatility(dates, rates):
    """Estimate annualized proxy volatility from small-interval rate shocks.

    Sort observations, omit NaNs without filling, and use successive valid
    observations up to seven calendar days apart (ACT/365.25). Remove constant
    drift; this does NOT estimate physical Vasicek mean reversion. At least 20
    eligible changes are required. No volatility floor is imposed.
    """
    estimate, count = _estimate_volatility(dates, rates)
    if estimate is None:
        raise ValueError(f"need at least {MIN_CHANGES} eligible short-rate changes; found {count}; "
                         "supply more history or increase --lookback-days")
    return estimate


def historical_yield_volatility(dates, panel):
    """Descriptive annualized yield volatilities, never calibration targets.

    Unsupported maturities return NaN; counts disclose actual usable evidence.
    Uses the same gap convention and constant-drift removal as the proxy.
    """
    panel = np.asarray(panel, dtype=float)
    dates = np.asarray(dates, dtype='datetime64[D]')
    if panel.ndim != 2 or panel.shape[0] != dates.size:
        raise ValueError("provide a yield panel with one row per date")
    observed = np.full(panel.shape[1], np.nan)
    counts = np.zeros(panel.shape[1], dtype=int)
    for i in range(panel.shape[1]):
        estimate, counts[i] = _estimate_volatility(dates, panel[:, i])
        if estimate is not None:
            observed[i] = estimate.sigma
    return observed, counts


def fit_yield_curve_profile(maturities, yields, r0, sigma):
    """Fit kappa_Q and theta_Q to one curve with fixed r0/sigma, equal weights.

    Profile theta analytically over 501 logarithmic kappa grid points. Refine
    every interior candidate minimum, retaining grid points and both endpoints.
    The near-optimal range is an RMSE sensitivity diagnostic, not a confidence
    interval. Pricing parameters are not physical forecasting parameters.
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    if (maturities.ndim != 1 or yields.shape != maturities.shape
            or maturities.size < 2 or not np.all(np.isfinite(maturities))
            or np.any(maturities <= 0) or not np.all(np.isfinite(yields))):
        raise ValueError("provide matching finite yields at at least two positive maturities")
    if np.unique(maturities).size != maturities.size:
        raise ValueError("maturities must be distinct")
    if not np.isfinite(r0) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("r0 must be finite and sigma must be nonnegative and finite")

    def evaluate(kappa):
        loading = -np.expm1(-kappa * maturities) / (kappa * maturities)
        a = 1 - loading
        c = bond_yield(r0, kappa, 0, sigma, maturities)
        theta = float(np.clip(np.dot(a, yields - c) / np.dot(a, a), *THETA_BOUNDS))
        residuals = a * theta + c - yields
        mse = float(np.mean(residuals**2))
        if not np.isfinite(mse):
            raise ValueError("nonfinite curve evaluation; check input rate units and magnitudes")
        return theta, mse

    grid = np.geomspace(*KAPPA_BOUNDS, 501)
    evaluations = [evaluate(k) for k in grid]
    costs = np.array([v[1] for v in evaluations])
    candidates = [(float(k), theta, cost) for k, (theta, cost) in zip(grid, evaluations)]
    for i in range(1, len(grid) - 1):
        if costs[i] <= costs[i-1] and costs[i] <= costs[i+1]:
            result = minimize_scalar(lambda k: evaluate(k)[1],
                                     bounds=(grid[i-1], grid[i+1]), method='bounded',
                                     options={'xatol': 1e-12})
            if not result.success:
                raise RuntimeError(f"curve profile refinement failed: {result.message}")
            theta, cost = evaluate(result.x)
            candidates.append((float(result.x), theta, cost))
    kappa, theta, mse = min(candidates, key=lambda v: v[2])
    rmse_bp = float(np.sqrt(mse) * 10_000)
    # Include refined minima in the exported profile and sensitivity range.
    profile = np.array(sorted(set(candidates)))
    profile_rmse = np.sqrt(profile[:, 2]) * 10_000
    near = profile[:, 0][profile_rmse <= rmse_bp + 1]
    half_lives = np.log(2) / near
    near_range = (float(half_lives.min()), float(half_lives.max()))
    warnings = []
    for name, value, bounds in [('kappa', kappa, KAPPA_BOUNDS), ('theta', theta, THETA_BOUNDS)]:
        lo, hi = bounds
        if min(value - lo, hi - value) <= 1e-4 * (hi - lo):
            warnings.append(f"{name} is at or near a computational bound; not an economic validation")
    if rmse_bp > 20:
        warnings.append("selected-date yield RMSE exceeds 20 bp; Vasicek may not fit this curve")
    if near_range[1] / near_range[0] > 10:
        warnings.append("broad minimum: half-lives within 1 bp of best RMSE span over 10x; "
                        "this is sensitivity, not a confidence interval")
    model = bond_yield(r0, kappa, theta, sigma, maturities)
    return CurveFit(float(r0), float(sigma), kappa, theta, model, model - yields,
                    rmse_bp, profile[:, 0], profile[:, 1], profile_rmse,
                    near_range, tuple(warnings))


def fit_yield_curve(maturities, market_yields, r0, sigma):
    """Compatibility API returning (kappa, theta) with fixed r0 and sigma."""
    fit = fit_yield_curve_profile(maturities, market_yields, r0, sigma)
    return fit.kappa, fit.theta
