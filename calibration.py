"""Staged Vasicek calibration and a fixed-decay Nelson–Siegel curve fit."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from vasicek import bond_yield
from nelson_siegel import FIXED_DECAY, factor_loadings


# Computational limits, not economic priors.
KAPPA_BOUNDS = (0.005, 5.0)
THETA_BOUNDS = (-0.2, 0.2)
MIN_CHANGES = 20


@dataclass
class NelsonSiegelFit:
    beta0: float
    beta1: float
    beta2: float
    decay: float
    model_yields: np.ndarray
    residuals: np.ndarray
    rmse_bp: float


def calibrate_nelson_siegel(maturities, yields):
    """Fit three factors by equal-weight least squares; decay stays at 0.5/year."""
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    if (maturities.ndim != 1 or yields.shape != maturities.shape
            or maturities.size < 3 or not np.all(np.isfinite(maturities))
            or np.any(maturities <= 0) or not np.all(np.isfinite(yields))):
        raise ValueError('Nelson–Siegel requires matching finite yields at at least three positive maturities')
    if np.unique(maturities).size != maturities.size:
        raise ValueError('Nelson–Siegel maturities must be distinct')
    loadings = factor_loadings(maturities)
    try:
        betas, _, rank, _ = np.linalg.lstsq(loadings, yields, rcond=None)
    except np.linalg.LinAlgError as error:
        raise RuntimeError('Nelson–Siegel least-squares fit failed') from error
    if rank < 3:
        raise ValueError('Nelson–Siegel loading matrix is rank deficient')
    model_yields = loadings @ betas
    residuals = model_yields - yields
    rmse_bp = float(np.sqrt(np.mean((10_000 * residuals)**2)))
    if not np.all(np.isfinite(betas)) or not np.isfinite(rmse_bp):
        raise RuntimeError('Nelson–Siegel fit produced nonfinite results')
    return NelsonSiegelFit(*map(float, betas), FIXED_DECAY, model_yields, residuals, rmse_bp)


@dataclass
class VolatilityEstimate:
    sigma: float
    drift: float
    start_date: str
    end_date: str
    observation_count: int
    change_count: int
    warnings: tuple[str, ...]
    shock_dates: np.ndarray
    scaled_shocks: np.ndarray


@dataclass
class CurveFit:
    r0: float
    sigma: float
    kappa: float
    theta: float
    model_yields: np.ndarray
    residuals: np.ndarray
    rmse_bp: float
    solver_attempts: list[dict]
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
    scaled_shocks = (changes - drift * dt) / np.sqrt(dt)
    variance = float(np.sum(scaled_shocks**2) / (n - 1))
    sigma = float(np.sqrt(variance))
    warnings = () if sigma > 0 else (
        "zero historical shock volatility: the window does not inform nonzero volatility",)
    return VolatilityEstimate(sigma, drift, str(dates[0]), str(dates[-1]),
                              int(dates.size), n, warnings,
                              dates[1:][eligible], scaled_shocks), n


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


def calibrate_yield_curve(maturities, yields, r0, sigma):
    """Fit risk-neutral kappa/theta with three bounded least-squares starts.

    Hold r0 and sigma fixed and weight selected-date yield errors equally.
    Starting-guess sensitivity is not a confidence interval or evidence that
    parameters are identified. These pricing parameters are not forecasts.
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

    def residuals_bp(parameters):
        return 10_000 * (bond_yield(r0, parameters[0], parameters[1], sigma, maturities) - yields)

    initial_theta = float(np.clip(yields[np.argmax(maturities)], -.19, .19))
    attempts, candidates = [], []
    lower = np.array([KAPPA_BOUNDS[0], THETA_BOUNDS[0]])
    upper = np.array([KAPPA_BOUNDS[1], THETA_BOUNDS[1]])
    for initial_kappa in (.03, .3, 3.0):
        attempt = {
            'initial_kappa': initial_kappa, 'initial_theta': initial_theta,
            'success': False, 'kappa': None, 'theta': None, 'rmse_bp': None,
            'message': '',
        }
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                result = least_squares(
                    residuals_bp, [initial_kappa, initial_theta],
                    bounds=(lower, upper), method='trf', loss='linear', x_scale='jac',
                    ftol=1e-12, xtol=1e-12, gtol=1e-10, max_nfev=1000)
                attempt['message'] = str(result.message)
                for name, value in zip(('kappa', 'theta'), result.x):
                    attempt[name] = float(value) if np.isfinite(value) else None
                valid = (np.all(np.isfinite(result.x))
                         and np.all(result.x >= lower) and np.all(result.x <= upper))
                if valid:
                    mse = float(np.mean(residuals_bp(result.x)**2))
                    valid = np.isfinite(mse)
                    if valid:
                        attempt['rmse_bp'] = float(np.sqrt(mse))
                attempt['success'] = bool(result.success and valid)
                if not valid:
                    attempt['message'] += '; rejected nonfinite or out-of-bounds result'
                if attempt['success']:
                    candidates.append((mse, attempt))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempt['message'] = str(error)
        attempts.append(attempt)

    if not candidates:
        raise RuntimeError('curve calibration failed for all three starting guesses: '
                           + '; '.join(attempt['message'] for attempt in attempts))
    # min preserves starting order on exact ties.
    _, best = min(candidates, key=lambda candidate: candidate[0])
    kappa, theta, rmse_bp = best['kappa'], best['theta'], best['rmse_bp']
    warnings = [f"starting guess kappa={attempt['initial_kappa']:g} failed: {attempt['message']}"
                for attempt in attempts if not attempt['success']]
    for name, value, bounds in [('kappa', kappa, KAPPA_BOUNDS), ('theta', theta, THETA_BOUNDS)]:
        lo, hi = bounds
        if min(value - lo, hi - value) <= 1e-4 * (hi - lo):
            warnings.append(f"{name} is at or near a computational bound; not an economic validation")
    if rmse_bp > 20:
        warnings.append("selected-date yield RMSE exceeds 20 bp; Vasicek may not fit this curve")
    near = [attempt for _, attempt in candidates if attempt['rmse_bp'] <= rmse_bp + 1]
    kappas = [attempt['kappa'] for attempt in near]
    thetas = [attempt['theta'] for attempt in near]
    if max(kappas) > 10 * min(kappas) or max(thetas) - min(thetas) > .01:
        warnings.append('starting-guess sensitivity: fits within 1 bp of best RMSE have '
                        'kappa spanning over 10x or theta spanning over 1 percentage point; '
                        'not a confidence interval or proof of identification')
    model = bond_yield(r0, kappa, theta, sigma, maturities)
    return CurveFit(float(r0), float(sigma), kappa, theta, model, model - yields,
                    rmse_bp, attempts, tuple(warnings))


def fit_yield_curve(maturities, market_yields, r0, sigma):
    """Compatibility API returning (kappa, theta) with fixed r0 and sigma."""
    fit = calibrate_yield_curve(maturities, market_yields, r0, sigma)
    return fit.kappa, fit.theta


def fit_summary(fit):
    """JSON-ready summary of the selected fit."""
    return {
        'r0': fit.r0, 'sigma': fit.sigma, 'kappa': fit.kappa, 'theta': fit.theta,
        'half_life_years': fit.half_life, 'limiting_yield': fit.limiting_yield,
        'rmse_bp': fit.rmse_bp, 'warnings': list(fit.warnings),
    }


def calibrate_market(market, curve_path, lookback_days=60):
    """2. Estimate sigma from recent history, then fit one risk-neutral curve.

    Display history never enters the curve objective. Return the fit and its
    audit report; plotting and application scenarios are separate stages.
    """
    if not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or lookback_days < 1:
        raise ValueError('lookback_days must be a positive integer')
    selected_date = market.date
    start = str(np.datetime64(selected_date) - np.timedelta64(lookback_days, 'D'))
    short_history = market.short_rates
    recent_rates = short_history.dates >= start
    volatility = estimate_short_rate_volatility(short_history.dates[recent_rates],
                                                short_history.rates[recent_rates])
    selected_maturities, market_yields = market.selected_curve
    fit = calibrate_yield_curve(selected_maturities, market_yields,
                                short_history.r0, volatility.sigma)
    warnings = [*volatility.warnings, *fit.warnings]
    span = int((np.datetime64(volatility.end_date) - np.datetime64(volatility.start_date)).astype(int))
    if span < 0.9 * lookback_days:
        warnings.append('available short-rate history covers less than 90% of the requested window')

    report = {
        'schema_version': 7, 'method': 'staged_vasicek', 'selected_date': selected_date,
        'curve_source': str(curve_path.resolve()), 'lookback_days': lookback_days,
        'data_vintage_note': 'Local snapshots; no claim of unrevised historical vintages.',
        'observed_proxy': {
            'series': short_history.series, 'source_path': short_history.source_path,
            'date': selected_date, 'quote_percent': (float(short_history.quotes_percent[-1])
                             if short_history.quotes_percent is not None else None),
            'continuous_rate': fit.r0, 'is_instantaneous_rate': False,
            'conversion': short_history.conversion,
            'interpretation': short_history.interpretation,
        },
        'historically_estimated': {
            'sigma': fit.sigma, 'units': 'decimal rate / sqrt(year)',
            'start_date': volatility.start_date, 'end_date': volatility.end_date,
            'observation_count': volatility.observation_count,
            'eligible_change_count': volatility.change_count,
            'removed_constant_drift': volatility.drift,
            'shock_dates': volatility.shock_dates.astype(str).tolist(),
            'scaled_shocks': volatility.scaled_shocks.tolist(),
            'day_count': 'ACT/365.25', 'maximum_gap_days': 7,
            'interpretation': 'Small-interval proxy volatility; not physical mean reversion.',
        },
        'curve_fitted': {
            **fit_summary(fit), 'weighting': 'equal squared yield errors',
            'fit_dates': [selected_date], 'measure': 'Q',
            'maturities': selected_maturities.tolist(),
            'market_yields': market_yields.tolist(), 'model_yields': fit.model_yields.tolist(),
            'residuals_bp': (-fit.residuals * 10_000).tolist(),
            'residual_convention': 'market minus model',
            'solver': 'bounded nonlinear least squares; three starting guesses',
            'solver_attempts': fit.solver_attempts,
            'computational_bounds': {'kappa': KAPPA_BOUNDS, 'theta': THETA_BOUNDS},
        },
        'warnings': warnings,
    }
    return fit, report
