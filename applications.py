"""3. Use the fitted pricing model: bond sensitivity and rate scenarios."""

import numpy as np

from vasicek import bond_price, simulate_short_rates


def apply_model(fit, maturities, *, seed=42):
    """Hold fitted parameters fixed for ±1 percentage point short-rate shocks."""
    shocks = np.array([-.01, 0, .01])
    prices = np.array([100 * bond_price(fit.r0 + shock, fit.kappa, fit.theta,
                                      fit.sigma, maturities) for shock in shocks])
    times, paths = simulate_short_rates(fit.r0, fit.kappa, fit.theta, fit.sigma,
                                       years=10, steps=120, n_paths=1000, seed=seed)
    return {
        'measure': 'Q',
        'interpretation': 'Risk-neutral pricing scenarios, not real-world forecasts.',
        'sensitivity': {
            'maturities': np.asarray(maturities).tolist(), 'face_value': 100,
            'short_rate_shocks': shocks.tolist(),
            'short_rates': (fit.r0 + shocks).tolist(),
            'prices': prices.tolist(),
            'price_change_up_100bp_percent': (100 * (prices[2] / prices[1] - 1)).tolist(),
        },
        'simulation': {
            'seed': seed, 'path_count': paths.shape[0], 'times': times.tolist(),
            'sample_paths': paths[:10].tolist(), 'mean': paths.mean(axis=0).tolist(),
            'percentile_05': np.quantile(paths, .05, axis=0).tolist(),
            'percentile_95': np.quantile(paths, .95, axis=0).tolist(),
            'method': 'Exact Gaussian transitions; monthly observations over 10 years.',
        },
    }
