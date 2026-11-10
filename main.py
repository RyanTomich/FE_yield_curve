"""Fit a Vasicek curve using a local short-rate proxy and staged calibration."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibration import (KAPPA_BOUNDS, THETA_BOUNDS, estimate_short_rate_volatility,
                         fit_yield_curve_profile, historical_yield_volatility)
from data import load_short_rate_history, load_yield_history
from plots import (plot_half_life_profile, plot_rate_distribution, plot_rate_paths,
                   plot_residuals, plot_yield_curve, plot_yield_curve_all_data,
                   plot_yield_volatility)
from vasicek import bond_yield, rate_moments, simulate_rates


ROOT = Path(__file__).resolve().parent


class RemovedVolatilityWeight(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error('--volatility-weight was removed: staged calibration estimates sigma '
                     'separately and fits the selected curve with equal weights')


def fit_summary(fit):
    """Small JSON-ready summary shared by the central and sensitivity fits."""
    return {
        'r0': fit.r0, 'sigma': fit.sigma, 'kappa': fit.kappa, 'theta': fit.theta,
        'half_life_years': fit.half_life, 'limiting_yield': fit.limiting_yield,
        'rmse_bp': fit.rmse_bp, 'warnings': list(fit.warnings),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='Fed curve date (YYYY-MM-DD); default: latest available')
    parser.add_argument('--short-rate-file', type=Path, default=ROOT / 'data' / 'DGS3MO.csv',
                        help='local FRED DGS3MO CSV (default: data/DGS3MO.csv)')
    parser.add_argument('--lookback-days', type=int, default=60,
                        help='calendar-day window for proxy volatility and context (default: 60)')
    parser.add_argument('--volatility-weight', action=RemovedVolatilityWeight, nargs='?',
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    curve_path = ROOT / 'data' / 'feds200628.csv'
    try:
        dates, maturities, history = load_yield_history(curve_path, args.date, args.lookback_days)
        selected_date = str(dates[-1])
        short_history = load_short_rate_history(args.short_rate_file, selected_date, args.lookback_days)
        volatility = estimate_short_rate_volatility(short_history.dates, short_history.rates)
        available = np.isfinite(history[-1])
        selected_maturities = maturities[available]
        market_yields = history[-1, available]
        fit = fit_yield_curve_profile(selected_maturities, market_yields,
                                      short_history.r0, volatility.sigma)
        sensitivity = []
        for multiplier in (0.75, 1.25):
            alternative = fit_yield_curve_profile(selected_maturities, market_yields,
                                                  fit.r0, multiplier * fit.sigma)
            sensitivity.append({
                'sigma_multiplier': multiplier, **fit_summary(alternative),
                'delta_kappa': alternative.kappa - fit.kappa,
                'delta_theta': alternative.theta - fit.theta,
                'delta_rmse_bp': alternative.rmse_bp - fit.rmse_bp,
            })
        observed_vol, change_counts = historical_yield_volatility(dates, history)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))

    warnings = [*volatility.warnings, *fit.warnings]
    span = int((np.datetime64(volatility.end_date) - np.datetime64(volatility.start_date)).astype(int))
    if span < 0.9 * args.lookback_days:
        warnings.append('available short-rate history covers less than 90% of the requested window')

    # All scenario outputs use the SAME fitted Q parameters; they are not forecasts.
    times = np.linspace(0, 10, 251)
    params = (fit.r0, fit.kappa, fit.theta, fit.sigma)
    paths = simulate_rates(*params, times, 20, np.random.default_rng(42))
    means, stds = rate_moments(*params, times)
    dense_maturities = np.linspace(0.25, 30, 300)
    dense_yields = bond_yield(*params, dense_maturities)
    model_vol = fit.sigma * (-np.expm1(-fit.kappa * maturities) / (fit.kappa * maturities))
    figures = {
        'A_rate_paths.png': plot_rate_paths(times, paths, fit.theta, selected_date),
        'B_rate_distribution.png': plot_rate_distribution(times, means, stds, selected_date),
        'C_yield_curve.png': plot_yield_curve(dense_maturities, dense_yields,
                                             selected_maturities, market_yields, selected_date),
        'C_yield_curve_all_data.png': plot_yield_curve_all_data(
            dense_maturities, dense_yields, maturities, history, str(dates[0]), selected_date),
        'D_residuals.png': plot_residuals(selected_maturities, fit.residuals, selected_date),
        'E_yield_volatility.png': plot_yield_volatility(
            maturities, observed_vol, model_vol, str(dates[0]), selected_date),
        'F_half_life_profile.png': plot_half_life_profile(
            fit.profile_kappa, fit.profile_rmse_bp, fit.kappa, fit.rmse_bp, selected_date),
    }
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    for filename, fig in figures.items():
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    def finite_list(values):
        return [float(v) if np.isfinite(v) else None for v in values]

    report = {
        'schema_version': 2, 'method': 'staged_vasicek', 'selected_date': selected_date,
        'curve_source': str(curve_path.resolve()), 'lookback_days': args.lookback_days,
        'data_vintage_note': 'Local snapshots; no claim of unrevised historical vintages.',
        'observed_proxy': {
            'series': 'DGS3MO', 'source_path': short_history.source_path,
            'date': selected_date, 'quote_percent': float(short_history.quotes_percent[-1]),
            'continuous_rate': fit.r0, 'is_instantaneous_rate': False,
            'conversion': '2 * log1p(quote_percent / 200)',
            'interpretation': 'Three-month Treasury yield used as a short-rate proxy.',
        },
        'historically_estimated': {
            'sigma': fit.sigma, 'units': 'decimal rate / sqrt(year)',
            'start_date': volatility.start_date, 'end_date': volatility.end_date,
            'observation_count': volatility.observation_count,
            'eligible_change_count': volatility.change_count,
            'removed_constant_drift': volatility.drift,
            'day_count': 'ACT/365.25', 'maximum_gap_days': 7,
            'interpretation': 'Small-interval proxy volatility; not physical mean reversion.',
        },
        'curve_fitted': {
            **fit_summary(fit), 'weighting': 'equal squared yield errors',
            'fit_dates': [selected_date], 'measure': 'Q',
            'maturities': selected_maturities.tolist(),
            'market_yields': market_yields.tolist(), 'model_yields': fit.model_yields.tolist(),
            'residuals_bp': (fit.residuals * 10_000).tolist(),
            'computational_bounds': {'kappa': KAPPA_BOUNDS, 'theta': THETA_BOUNDS},
        },
        'half_life_profile': {
            'kappa': fit.profile_kappa.tolist(), 'theta': fit.profile_theta.tolist(),
            'half_life_years': (np.log(2) / fit.profile_kappa).tolist(),
            'rmse_bp': fit.profile_rmse_bp.tolist(),
            'within_1bp_half_life_range': fit.near_optimal_half_life_range,
            'interpretation': 'Sensitivity diagnostic, not a confidence interval.',
        },
        'sigma_sensitivity': sensitivity,
        'historical_context': {
            'start_date': str(dates[0]), 'end_date': selected_date, 'dates': dates.tolist(),
            'used_in_curve_objective': False, 'maturities': maturities.tolist(),
            'observed_annualized_yield_volatility': finite_list(observed_vol),
            'model_annualized_yield_volatility': finite_list(model_vol),
            'eligible_change_counts': change_counts.tolist(),
        },
        'scenarios': {'measure': 'Q', 'interpretation': 'Pricing scenarios, not forecasts.',
                      'horizon_years': 10, 'seed': 42, 'path_count': 20},
        'warnings': warnings,
    }
    (output / 'calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Selected curve: {selected_date} ({len(market_yields)} maturities); equal weights')
    print(f'Observed proxy: DGS3MO {short_history.quotes_percent[-1]:.4f}% quoted; r0 = {fit.r0:.4%} continuous')
    print(f'Local short-rate source: {short_history.source_path}')
    print(f'Historically estimated sigma: {fit.sigma:.4%}/sqrt(year); '
          f'{volatility.start_date} to {volatility.end_date}; {volatility.change_count} eligible changes')
    print(f'Curve-fitted Q kappa: {fit.kappa:.6f}/year; half-life: {fit.half_life:.2f} years')
    print(f'Curve-fitted Q theta: {fit.theta:.4%}; limiting yield: {fit.limiting_yield:.4%}')
    print(f'Selected-date yield RMSE: {fit.rmse_bp:.2f} bp')
    print('Computational bounds (not economic priors): kappa [0.005, 5], theta [-20%, 20%]')
    print('Half-lives within 1 bp of best RMSE: '
          f'{fit.near_optimal_half_life_range[0]:.2f}–{fit.near_optimal_half_life_range[1]:.2f} years '
          '(sensitivity, not a confidence interval)')
    for case in sensitivity:
        print(f"Sigma ×{case['sigma_multiplier']:.2f}: kappa {case['kappa']:.6f}, "
              f"theta {case['theta']:.4%}, RMSE {case['rmse_bp']:.2f} bp "
              f"(change {case['delta_rmse_bp']:+.2f} bp)")
    print('A/B are fitted risk-neutral pricing scenarios, not real-world forecasts.')
    print('Historical yield curves and volatility are diagnostics only; no future observations used.')
    print('Local snapshots are not guaranteed to be unrevised historical vintages.')
    if selected_maturities[-1] < 30:
        print(f'Beyond {selected_maturities[-1]:g} years the curve is model extrapolation.')
    for warning in warnings:
        print(f'Warning: {warning}.')
    print(f'Saved {len(figures)} figures and calibration.json in {output}')
    return report


if __name__ == '__main__':
    main()
