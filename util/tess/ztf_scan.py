import sys; sys.path.insert(0, r"C:\Users\guill\cepheid-internet\src")
import numpy as np, pandas as pd, astropy.units as u, warnings; warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord
from stage1_pulse import _ztf_raw
S = sys.argv[1]
m = pd.read_csv(f"{S}/miras_tic.csv").dropna(subset=["tic"]).drop_duplicates("tic")
m["beta"] = SkyCoord(m.ra.values*u.deg, m.dec.values*u.deg).barycentrictrueecliptic.lat.deg
c = m[m.beta > 78].sort_values("max")
for _, r in c.iterrows():
    try:
        df = _ztf_raw(f"RADEC {r.ra} {r.dec}", pos=(r.ra, r.dec))
    except Exception as e:
        print(r.Name, "ERR", repr(e)[:80]); continue
    if df is None or df.empty: print(r.Name, "none"); continue
    df.to_parquet(f"{S}/ztf/{r.Name.replace(' ','_')}.parquet")
    ok = (df.catflags == 0) & (df.mag > 12.5)
    s = df[ok].groupby("filtercode").size().to_dict()
    print(f"{r.Name:10} V {r['max']}-{r['min']}  P={r.Period}  unsat good epochs {s}  span {df.mjd.min():.0f}-{df.mjd.max():.0f}", flush=True)
