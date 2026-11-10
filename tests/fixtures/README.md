# Synthetic short-rate fixtures

These CSVs have FRED's `observation_date,DGS3MO` format, but their rates are
**synthetic test values, not observations downloaded from FRED**. Do not copy
them into `data/DGS3MO.csv` for market analysis.

They cover weekday dates in the 60-calendar-day windows ending January 3, 2020
and June 1, 2026. They use seeded Gaussian increments around illustrative rate
levels, converted to percentage bond-equivalent quotes. Holiday calendars are
not modeled. They support reproducible offline integration tests only.
