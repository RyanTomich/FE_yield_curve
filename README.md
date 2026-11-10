# Minimal Vasicek yield-curve model

Simulate the Vasicek short-rate process, plot its analytical distribution, and
fit a two-parameter **risk-neutral** Vasicek zero-coupon yield curve.

## Run

```bash
pip install -e .
python main.py
```

This writes four PNGs to `output/` and prints fitted `kappa`, `theta` and
RMSE in basis points. The random seed is fixed for reproducible simulations.

Run the numerical checks with:

```bash
python -m unittest discover -s tests -v
```

## Inputs and assumptions

`data/yields.csv` contains **illustrative synthetic/example values**, **not**
observed U.S. Treasury yields. Replace them with *continuously compounded
zero-coupon spot yields* if using market data. Standard quoted par/coupon
Treasury yields are not interchangeable with zero-coupon yields; obtain or
bootstrap a compatible zero-coupon curve first.

CSV columns are `maturity_years,yield`, with years for maturities and decimal
rates (e.g. `0.04` = 4%). Values must be finite and maturities distinct and
positive.

- Figures A/B use illustrative dynamics: `r0=3.5%`, `kappa=0.4/year`,
  `theta=4.5%`, `sigma=1.5%/sqrt(year)`.
- Figures C/D fit risk-neutral `kappa` and `theta` while assuming
  `r0=3.5%` and `sigma=1.2%/sqrt(year)`. They do **not** estimate real-world
  forecasting parameters. The calibration ignores observational weights and
  model risk.

Files are intentionally small: `vasicek.py` holds the math, `calibration.py`
fits pricing parameters, `data.py` loads CSV values, `plots.py` constructs
figures without saving them, and `main.py` orchestrates execution and saves
figures.
