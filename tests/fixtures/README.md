# Synthetic short-rate fixtures

These CSVs have FRED's `observation_date,DGS3MO` format, but their rates are
**synthetic test values, not observations downloaded from FRED**. Do not copy
them into `data/DGS3MO.csv` for market analysis.

They cover weekday dates in the 60-calendar-day windows ending January 3, 2020
and June 1, 2026. They use seeded Gaussian increments around illustrative rate
levels, converted to percentage bond-equivalent quotes. Holiday calendars are
not modeled. They support reproducible offline integration tests only.

## Calibration regression curves

`calibration_regression.json` contains four selected curves and fixed short-rate
proxy/volatility inputs captured from the local `zero_curve_2001_2026.csv` file.
Unlike the synthetic short-rate CSVs above, these are local-data snapshots; their
original market provenance is unspecified. They are not refreshed by tests.
Reference RMSEs were measured with the previous 501-point profile solver before
its removal. The June 2019 case also exercises competing local minima. These
small fixed inputs make solver comparisons independent of the working data file.
