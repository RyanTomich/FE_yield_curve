"""Observe market data, calibrate Vasicek, and use the fitted pricing model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from applications import apply_model
from calibration import calibrate_market
from market import observe_market
from plots import (plot_short_rate_estimation, plot_historical_short_rates,
                   plot_rate_sensitivity, plot_residuals, plot_simulated_short_rates,
                   plot_yield_curve, plot_yield_curve_history)
from vasicek import bond_yield

ROOT = Path(__file__).resolve().parent


class RemovedVolatilityWeight(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error('--volatility-weight was removed: calibration estimates sigma '
                     'separately and fits the selected curve with equal weights')


class RemovedDiagnostics(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error('--diagnostics was removed: simplified calibration retains core fit plots '
                     'and basic solver warnings only')


def save_figures(output, stage, figures):
    folder = output / stage
    folder.mkdir(parents=True, exist_ok=True)
    for filename, fig in figures.items():
        fig.savefig(folder / filename, dpi=160)
        plt.close(fig)
    return [f'{stage}/{name}' for name in figures]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='Exact curve date (YYYY-MM-DD); default: latest available')
    parser.add_argument('--curve-file', type=Path, default=ROOT / 'data' / 'zero_curve_2001_2026.csv')
    parser.add_argument('--short-rate-file', type=Path, help='Optional local FRED DGS3MO CSV')
    parser.add_argument('--lookback-days', type=int, default=60,
                        help='Calendar days for volatility estimation only (default: 60)')
    parser.add_argument('--diagnostics', action=RemovedDiagnostics, nargs=0, help=argparse.SUPPRESS)
    parser.add_argument('--volatility-weight', action=RemovedVolatilityWeight, nargs='?', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        market = observe_market(args.curve_file, args.date, args.short_rate_file)
        fit, report = calibrate_market(market, args.curve_file, args.lookback_days)
        maturities, yields = market.selected_curve
        dense = np.linspace(maturities[0], maturities[-1], 300)
        applications = apply_model(fit, dense)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))

    output = ROOT / 'output'
    # 1. Observe the market. These figures contain no fitted model.
    files = save_figures(output, '01_market_data', {
        'yield_curve_historical_context.png': plot_yield_curve_history(market),
        'historical_short_rates.png': plot_historical_short_rates(market),
    })

    # 2. Calibrate. Show parameter estimation, then the resulting curve fit.
    files += save_figures(output, '02_calibration', {
        'short_rate_estimation.png': plot_short_rate_estimation(market, report),
        'fitted_yield_curve.png': plot_yield_curve(
            dense, bond_yield(fit.r0, fit.kappa, fit.theta, fit.sigma, dense),
            maturities, yields, market.date, fit),
        'yield_fit_residuals.png': plot_residuals(maturities, fit.residuals, market.date),
    })
    # 3. Apply the fitted model. Keep pricing scenarios distinct from forecasts.
    files += save_figures(output, '03_applications', {
        'rate_sensitivity.png': plot_rate_sensitivity(applications['sensitivity'], market.date),
        'simulated_short_rates.png': plot_simulated_short_rates(applications['simulation'], fit, market.date),
    })
    report['market_data'] = {
        'comparison_dates': [day for _, day, _ in market.comparisons],
        'short_rate_history_start': str(market.short_rates.dates[0]),
        'short_rate_history_end': market.date, 'warnings': market.warnings,
    }
    report['applications'] = applications
    report['outputs'] = files
    (output / 'calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    # Retire the previous flat output layout and any optional plots from an older run.
    for name in ('fitted_yield_curve.png', 'yield_curve_historical_context.png',
                 'yield_fit_residuals.png', 'yield_volatility.png', 'half_life_sensitivity.png',
                 'parameter_selection.png'):
        (output / name).unlink(missing_ok=True)
    for name in ('parameter_selection.png', 'half_life_sensitivity.png', 'yield_volatility.png'):
        (output / '02_calibration' / name).unlink(missing_ok=True)

    print(f'1. Market data — {market.date}: {len(yields)} maturities; {market.short_rates.series} short-rate proxy')
    print('2. Calibration — historical sigma held fixed; kappa and theta fitted with three least-squares starts')
    print(f'{"Parameter":<12} {"Value":>14}  Interpretation')
    for name, value, meaning in [
        ('r0', f'{fit.r0:.4%}', 'Current short-rate proxy'),
        ('theta', f'{fit.theta:.4%}', 'Risk-neutral long-run rate'),
        ('kappa', f'{fit.kappa:.6f}/yr', 'Risk-neutral mean-reversion speed'),
        ('sigma', f'{fit.sigma:.4%}/√yr', 'Historical proxy volatility'),
        ('RMSE', f'{fit.rmse_bp:.2f} bp', 'Root mean squared yield error'),
    ]:
        print(f'{name:<12} {value:>14}  {meaning}')
    print('3. Applications — ±1 percentage point rate shocks and 1,000 ten-year pricing scenarios')
    print('Risk-neutral scenarios are not real-world forecasts.')
    for warning in [*market.warnings, *report['warnings']]:
        print(f'Warning: {warning}.')
    print(f'Saved {len(files)} plots in three stage folders and calibration.json in {output}')
    return report


if __name__ == '__main__':
    main()
