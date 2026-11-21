"""Joint Mira-cycle model: B-spline in time per band (+ per-sector offsets for TESS), in magnitudes."""
import numpy as np, pandas as pd
from scipy.interpolate import BSpline

def design(t, t0, t1, dk, k=3):
    inner = np.arange(t0, t1 + dk, dk)
    kn = np.r_[[inner[0]]*k, inner, [inner[-1]]*k]
    n = len(kn) - k - 1
    return BSpline.design_matrix(np.clip(t, inner[0], inner[-1]), kn, k).toarray()

def fit(t, m, w, groups=None, t0=None, t1=None, dk=15.0):
    B = design(t, t0, t1, dk)
    cols = [B]
    if groups is not None:
        u = np.unique(groups)
        cols.append((groups[:, None] == u[None, 1:]).astype(float))   # offsets relative to first group
    X = np.hstack(cols)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], m * sw, rcond=None)
    return m - X @ beta, X @ beta

def load(S, sectors=range(76, 84)):
    t = pd.read_parquet(f"{S}/xdra_tess.parquet"); t = t[t.sector.isin(list(sectors))]
    t = t[t.sap > 0].copy(); t["m"] = -2.5*np.log10(t.sap)
    z = pd.read_parquet(f"{S}/xdra_ztf.parquet")
    z = z[(z.mjd >= t.mjd.min()) & (z.mjd <= t.mjd.max())].copy()
    return t, z
