"""Load a single cross-section of continuously compounded zero-coupon yields."""

import numpy as np


def load_yields(path):
    """Return sorted (maturity_years, yield) arrays from the documented CSV."""
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if rows.dtype.names != ("maturity_years", "yield"):
        raise ValueError("CSV header must be maturity_years,yield")
    rows = np.atleast_1d(rows)
    maturities = np.asarray(rows["maturity_years"], dtype=float)
    yields = np.asarray(rows["yield"], dtype=float)
    if (maturities.size < 2 or not np.all(np.isfinite(maturities))
            or np.any(maturities <= 0) or not np.all(np.isfinite(yields))
            or np.unique(maturities).size != maturities.size):
        raise ValueError("CSV requires distinct positive maturities and finite yields")
    order = np.argsort(maturities)
    return maturities[order], yields[order]
