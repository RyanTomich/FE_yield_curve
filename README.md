# Yield curves with Vasicek

**Observe the market → calibrate a model → answer questions with the model.**

Vasicek models the instantaneous short rate as a mean-reverting stochastic
process. Its parameters imply zero-coupon bond prices, which imply a yield curve:

```text
dr = kappa * (theta - r) * dt + sigma * dW
model parameters → bond prices → yields
```

This project keeps those three stages separate, using local CSV data only.

## Run

```bash
python3 -m pip install -e .
python3 main.py --date 2020-01-03
```

Omit `--date` to use the latest usable observation in the file. A specified date
must exist; weekends and missing dates are not silently substituted.

Each run prints a parameter table and writes seven plots into three folders:

```text
output/
  01_market_data/
    yield_curve_historical_context.png
    historical_short_rates.png
  02_calibration/
    short_rate_estimation.png
    fitted_yield_curve.png
    yield_fit_residuals.png
  03_applications/
    rate_sensitivity.png
    simulated_short_rates.png
  calibration.json
```

Outputs are replaced on each successful run. Read the dates on the plots and
in the report; an unsuccessful run can leave earlier outputs in place.

## 1. Observe the market

**What does the selected curve look like, and how has the market changed?**

- `yield_curve_historical_context.png` shows the selected curve alongside curves
  from one and three calendar years earlier. It contains **no model fit**.
  Anniversary comparisons use the most recent observation on or before the
  target, at most seven days earlier. Actual dates appear in the legend;
  unavailable comparisons are omitted and noted. February 29 maps to February 28.
- `historical_short_rates.png` shows all available three-month proxy history
  through the selected date. It provides context for fluctuations and rate
  regimes; the plot alone does not establish mean reversion.

The default file is `data/zero_curve_2001_2026.csv`. Its columns are `Date` and
maturities such as `0.08Y`, `0.25Y`, and `10Y`, with **continuously compounded
decimal zero yields** (`0.04` means 4%). Maturities are interpreted literally
in years. The file has no source metadata, so plots describe local zero-yield
observations without asserting Treasury provenance or unrevised historical
vintages. The program neither downloads nor refreshes data.

## 2. Calibrate Vasicek

**How well can Vasicek explain this date's yield curve?**

1. Set `r0` to the selected date's three-month yield, a proxy for the
   instantaneous short rate.
2. Estimate `sigma` from recent changes in that proxy and hold it fixed.
3. Fit risk-neutral `kappa` and `theta` to the selected curve, minimizing equally
   weighted squared yield errors.

`short_rate_estimation.png` shows the three-month proxy, the selected `r0`,
and the highlighted volatility estimation window. Its lower panel shows the
exact drift-adjusted changes divided by the square root of their year fractions,
with reference lines at ±`sigma`. These indicate scale, not confidence bounds.

`fitted_yield_curve.png` overlays market yields and fitted yields, with the
parameters and RMSE beneath the plot. `yield_fit_residuals.png` shows **market
minus model** in basis points: positive values mean the model underestimates
the market yield. Curves are drawn only over the observed maturity range.

| Parameter | Interpretation | How obtained |
| --- | --- | --- |
| `r0` | Current short-rate proxy | Selected date's three-month yield |
| `theta` | Risk-neutral long-run short rate | Curve fit |
| `kappa` | Risk-neutral mean-reversion speed, per year | Curve fit |
| `sigma` | Rate volatility, decimal rate per √year | Historical proxy changes |
| RMSE | Root mean squared yield error, in basis points | Market vs. fitted yields |

A single curve can weakly identify the parameters, especially volatility.
Estimating `sigma` separately reduces this problem; it does not guarantee
precise or economically plausible parameters.

The default estimation window is 60 calendar days, independent of the longer
history displayed in stage 1. Change it with `--lookback-days 365`. At least
20 changes with gaps of at most seven days are required. The estimator uses
ACT/365.25 intervals and removes a constant drift:

```text
mu      = sum(delta_r) / sum(delta_t)
sigma²  = sum((delta_r - mu * delta_t)² / delta_t) / (n - 1)
```

This estimates small-interval proxy volatility, **not real-world mean-reversion
dynamics**. Only the selected curve enters the yield-fitting objective.

The solver fits `kappa` and `theta` jointly using bounded nonlinear least
squares, with yield errors measured in basis points. It tries three initial
`kappa` values (`0.03`, `0.3`, and `3.0`), starting `theta` at the longest-maturity
observed yield clipped to `[-0.19, 0.19]`, and selects the successful finite fit
with the lowest squared error. Exact ties retain starting-guess order.
Computational bounds remain `0.005 ≤ kappa ≤ 5` and `−0.2 ≤ theta ≤ 0.2`;
these are not economic priors.

Each attempt uses SciPy's trust-region reflective solver with ordinary
squared-error loss, Jacobian scaling, `ftol=xtol=1e-12`, `gtol=1e-10`, and at
most 1,000 function evaluations. The report records the initial and final
parameters, success, RMSE, and termination message for all three attempts.
Failed attempts are reported as warnings; calibration fails if none succeeds.
Nonfinite attempt values are recorded as `null`.

Boundary fits and RMSE above 20 bp are flagged. Starting-guess sensitivity is
also flagged when successful fits within 1 bp of the best RMSE have `kappa`
spanning more than 10× or `theta` spanning more than one percentage point.
These checks neither guarantee a global minimum nor establish parameter
identification or confidence intervals.

The former parameter profile, volatility sensitivity recalibrations, and
yield-volatility diagnostic have been removed. `--diagnostics` now reports a
removal error. Successful runs remove obsolete diagnostic plots from the output
folders. Calibration retains the short-rate estimation, fitted curve, and
residual plots.

## 3. Use the calibrated model

**How do bond prices respond to rates, and what rate scenarios does the model imply?**

- `rate_sensitivity.png` prices zero-coupon bonds per $100 face value at the
  calibrated `r0`, `r0 − 1 percentage point`, and `r0 + 1 percentage point`,
  holding `kappa`, `theta`, and `sigma` fixed. A second panel shows the percentage
  price change for the upward shock, making maturity sensitivity explicit.
- `simulated_short_rates.png` generates 1,000 paths over ten years with exact
  Gaussian Vasicek transitions at monthly intervals. It shows ten sample paths,
  the simulated mean, and pointwise 5th–95th percentile bands. A fixed seed (42)
  makes the result reproducible. The model allows negative rates.

**These are risk-neutral pricing scenarios, not real-world forecasts.**
Forecasting requires separately estimated real-world dynamics. Historical
volatility estimation does not turn the curve-fitted pricing drift into a
forecast. The bands describe scenario dispersion at each time, not parameter
confidence or a band containing 90% of entire paths.

The model's mean-reversion half-life is `log(2)/kappa`. Its long-maturity yield
limit is `theta − sigma²/(2*kappa²)`, which differs from the long-run short rate.

## Files and data options

The three main components are plain Python modules, with shared helpers:

| File | Responsibility |
| --- | --- |
| `market.py` | Stage 1: load market context and select comparison dates |
| `calibration.py` | Stage 2: estimate volatility, fit the curve, report solver results |
| `applications.py` | Stage 3: price shocked bonds and generate rate scenarios |
| `main.py` | Run the three stages, save plots, print parameters |
| `data.py` | Read and validate local CSVs |
| `vasicek.py` | Bond-pricing formulas and exact rate simulation |
| `plots.py` | Plot presentation only |

```bash
python3 main.py --curve-file /path/to/zero_curve.csv
python3 main.py --curve-file /path/to/feds200628.csv --short-rate-file /path/to/DGS3MO.csv
```

The original Fed `Date,SVENY01,...,SVENY30` format is also supported, including
metadata before the header. These yields are percentages and are divided by
100. Because it lacks a three-month maturity, provide a separate DGS3MO file.
That file must contain `DATE` or `observation_date`, plus `DGS3MO` percentage
quotes. Its bond-equivalent quotes are converted with
`2 * log1p(quote_percent / 200)`. The default `0.25Y` values need no conversion.
Both proxy sources require an observation on the exact selected date.

Missing yields remain missing; duplicate dates, invalid values, and duplicate
or nonpositive maturities are rejected. No future observations enter the plots
or estimation. Comparison curves do not affect calibration.

`calibration.json` (schema 6) records source paths, conventions, observed proxy,
historical volatility, curve-fitted parameters, solver attempts, market comparison
dates, application scenarios, and the paths of this run's plots. Schema 6
replaces the former profile and volatility diagnostics with
`curve_fitted.solver_attempts`; the old `half_life_profile`, `sigma_sensitivity`,
and `historical_context` sections are absent. Market context remains in
`market_data`. Reports and plots retain **market minus model** residuals
(the same convention as schema 5). All rates in the report are decimals unless named
`quote_percent`, `*_percent`, or `*_bp`.

## Checks

```bash
python3 -m unittest discover -s tests -v
```

Tests cover data selection, calibration, analytic simulation moments, bond
sensitivity, and offline generation of the stage outputs. Integration tests use
explicitly synthetic rate histories; solver regression tests use four fixed
local-data curve snapshots. See `tests/fixtures/README.md` for their provenance.
