# Staged Vasicek yield-curve calibration

Fit one **risk-neutral** Vasicek curve in three explicit stages:

1. Anchor `r0` to an observed three-month Treasury **proxy** on the valuation date.
2. Estimate `sigma` from the proxy's historical movements, then hold it fixed.
3. Fit only `kappa` and `theta` to the selected date's zero-coupon yields, with
   equal squared-yield-error weights.

The program uses local files only. It does not download or refresh data.

## Local inputs and running

Install the project dependencies in your Python environment:

```bash
pip install -e .
```

Download both input files and save them in the project's `data/` directory:

1. **Fed zero-coupon yield curves → `data/feds200628.csv`.**
   Download the [Fed yield-curve CSV](https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv),
   or open the [Federal Reserve Nominal Yield Curve page](https://www.federalreserve.gov/data/nominal-yield-curve.htm)
   and select **Data: CSV**. Save the file as `feds200628.csv`. Keep its metadata,
   headers, and all columns intact; the loader skips the metadata and selects
   `SVENY01`–`SVENY30` automatically.
2. **Three-month Treasury rate → `data/DGS3MO.csv`.**
   Open [FRED's DGS3MO series](https://fred.stlouisfed.org/series/DGS3MO), then
   choose **Download → CSV** above the graph. Keep the frequency **Daily** and
   units **Percent**, without transformations. Download the full available
   history, or a range covering your valuation date and at least the preceding
   `--lookback-days` calendar days (60 by default). Save or rename the downloaded
   file to `DGS3MO.csv`. See [FRED's download instructions](https://fred.stlouisfed.org/help/data/downloading/using-the-download-data-link/)
   if needed. A different local location can be supplied with `--short-rate-file`.

For example, `--date 2020-01-03` with the default lookback needs short-rate data
covering **2019-11-04 through 2020-01-03**, including a valid observation on
January 3. Keep the downloaded percentage values unchanged; the program handles
unit conversion.

The short-rate CSV must have `DATE` or `observation_date`, plus `DGS3MO`.
Values are percentage quotes; blank cells, `.` and `NA` mean missing. For
example, a quote of `1.50` means 1.50%, not a decimal rate of 1.50.

```bash
python main.py --date 2020-01-03
python main.py --date 2026-06-01 --short-rate-file /path/to/DGS3MO.csv
python main.py --date 2020-01-03 --lookback-days 365
```

The default lookback is **60 calendar days**, inclusive of both endpoints.
It controls the historical volatility sample and contextual plots; only the
selected date's curve enters the fitting objective. Without `--date`, the latest
usable row in the Fed curve file is selected. The short-rate file must have a
valid observation on that exact date. Neither an earlier date nor the one-year
yield is silently substituted. Missing files and insufficient history produce
an actionable error before any output is written.

At least two distinct positive curve maturities and **20 eligible short-rate
changes** are required. Unsupported historical yield-volatility points are
omitted from plot E; they do not prevent curve fitting. Dates after the valuation
date are excluded. Local snapshots may contain revisions: this is not a claim
that the observations were available in this form on their historical dates.

## Stage 1: observed proxy

DGS3MO is a three-month constant-maturity Treasury yield, not an instantaneous
short rate. Its quoted bond-equivalent rate `q` (in decimals) is converted to a
continuous equivalent:

```text
r = 2 * log(1 + q / 2)
  = 2 * log1p(quote_percent / 200)
```

See the [Treasury quotation conventions](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions).
This conversion does not eliminate the three-month-versus-instantaneous-rate
approximation. The report records the original quote, converted rate, date,
series, and local source path. The selected-date converted rate fixes `r0`.

## Stage 2: historical volatility

Sort valid observations, reject duplicate dates and invalid values, omit missing
rates without filling, and form successive valid-observation changes. Keep gaps
of at most seven calendar days and measure elapsed time using ACT/365.25.
For `n` eligible changes, estimate:

```text
mu       = sum(delta_r) / sum(delta_t)
sigma^2  = sum((delta_r - mu * delta_t)^2 / delta_t) / (n - 1)
```

This removes a constant drift before measuring shock volatility. It is a
**small-interval proxy-volatility approximation**, not an estimate of physical
Vasicek mean reversion. The report includes the actual history span, observation
count, eligible change count, and removed drift. A truly zero estimate remains
zero and triggers a warning; no hidden floor is applied. Short windows, policy
jumps, and the use of a three-month proxy can affect this estimate.

## Stage 3: selected-date curve

With `r0` and `sigma` fixed, yields are affine in `theta` for each trial `kappa`:

```text
y_i = a_i * theta + c_i
best theta = clip(sum(a_i * (observed_y_i - c_i)) / sum(a_i^2), -0.2, 0.2)
```

The solver evaluates 501 logarithmically spaced `kappa` values, refines every
interior candidate minimum with bounded scalar minimization, and compares the
results with the grid and both endpoints. It minimizes the mean squared yield
error without per-maturity uncertainty weights or a volatility fitting term.

Computational bounds remain `0.005 <= kappa <= 5` and `-0.2 <= theta <= 0.2`.
They are **not economic priors**. Boundary solutions and curve RMSE above 20 bp
are reported. The output also includes:

- Pricing mean-reversion half-life: `log(2) / kappa`.
- Limiting bond yield: `theta - sigma^2 / (2*kappa^2)`; this is not `theta`.
- The half-life range on the evaluated profile within 1 bp of the best RMSE.
  A span over a factor of ten triggers a broad-minimum warning. This is a
  sensitivity diagnostic, **not a confidence interval**.
- Recalibration at `0.75 * sigma` and `1.25 * sigma`, including parameter and
  RMSE changes. These do not replace the central fit.

No forecast interpretation is assigned to the fitted pricing drift. A low RMSE
cannot establish economic plausibility. Vasicek still restricts the possible
curve shapes, ties all maturities to one factor, and predicts yield volatility
that declines with maturity. The staged procedure exposes these limitations
rather than guaranteeing a fit or forcing attractive parameters.

## Outputs

A successful run writes **seven PNGs** and `calibration.json` to `output/`:

| Output | Interpretation |
| --- | --- |
| `A_rate_paths.png` | Twenty short-rate paths under the fitted risk-neutral parameters. Pricing scenarios, not real-world forecasts. |
| `B_rate_distribution.png` | Expected rate and pointwise 5th–95th percentile bands under the same pricing parameters. The band is not a 90% guarantee for an entire path. |
| `C_yield_curve.png` | Selected-date fitted curve versus Fed zero-coupon yield estimates. |
| `C_yield_curve_all_data.png` | The same selected-date curve with historical observations for context; those older curves are not fitted. |
| `D_residuals.png` | Model minus observed yields in basis points; positive means the model is too high. |
| `E_yield_volatility.png` | Historical versus model-implied annualized yield volatility, purely diagnostic. No bootstrap error bars or fitting score. |
| `F_half_life_profile.png` | Selected-date RMSE versus pricing half-life on a logarithmic axis, with the selected solution, computational limits, and best-RMSE-plus-1-bp line. |

Plots A/B use the same calibrated `r0`, `sigma`, `kappa`, and `theta` as the curve,
a ten-year horizon, and the existing fixed random seed (42). Plot E uses
`sigma * (1-exp(-kappa*T))/(kappa*T)` and the historical estimator described
above, separately for each maturity. Fewer than 20 eligible changes means that
historical point is omitted. Historical curves are smoothed estimates, not
independent raw bond quotes.

The fitted curve is drawn from 0.25 through 30 years. Its extension outside
available observed maturities is model extrapolation. Every figure identifies
the valuation date or relevant historical interval.

`calibration.json` uses schema version 2 and separates `observed_proxy`,
`historically_estimated`, `curve_fitted`, `half_life_profile`, `sigma_sensitivity`,
`historical_context`, and `scenarios`. It records the sources, dates, conventions,
counts, computational bounds, profile values, and warnings. The old panel
scores, empirical uncertainty weights, and daily inferred rates are removed.

Failed runs do not replace earlier output files; check the date and report before
interpreting an existing plot as the result of a new run.

## API and migration

- `load_short_rate_history(path, date, lookback_days=60)` returns sorted proxy
  observations, original percentage quotes, and the source path.
- `estimate_short_rate_volatility(dates, rates)` returns the estimate, counts,
  date span, removed drift, and warnings. Decimal continuous rates are expected;
  NaN denotes a missing observation.
- `fit_yield_curve_profile(maturities, yields, r0, sigma)` returns the selected
  parameters, yields, residuals, profile, and warnings. It accepts no history.
- `fit_yield_curve(...)` retains its existing `(kappa, theta)` return value as a
  wrapper around the profile solver.

The joint panel fitter and its bootstrap/weighting machinery have been removed.
`--volatility-weight` now produces an explicit migration error. Previous runs
need a local DGS3MO file before they can use staged calibration.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover quote conventions, exact-date and historical-window selection,
missing data and date gaps, volatility recovery, independent numerical
least-squares comparisons, parameter bounds, broad minima, sensitivity results,
and offline rendering of all seven figures. Integration fixtures for January
2020 and June 2026 are **explicitly synthetic**, not downloaded Treasury data.
They must not be used as market observations or copied to `data/DGS3MO.csv`.

`vasicek.py` holds the dynamics/pricing math, `calibration.py` the staged
estimators, `data.py` the local loaders, `plots.py` presentation, and `main.py`
orchestration and reporting.
