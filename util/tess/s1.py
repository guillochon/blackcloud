import sys; sys.path.insert(0, r"C:\Users\guill\cepheid-internet\src")
import numpy as np, pandas as pd
from astroquery.vizier import Vizier
from stage0_build_targets import crossmatch_tic
out = sys.argv[1]
v = Vizier(columns=["OID","Name","Type","RAJ2000","DEJ2000","max","n_max","min","Period"], row_limit=-1)
t = v.query_constraints(catalog="B/vsx/vsx", Type="=M")[0]
df = t.to_pandas()
print(len(df), "VSX type M")
df = df.rename(columns={"OID":"source_id","RAJ2000":"ra","DEJ2000":"dec"})
df["max"] = pd.to_numeric(df["max"], errors="coerce")
df = df[df["max"] < 11].copy()   # bright at maximum
print(len(df), "with max < 11 mag")
df = crossmatch_tic(df, 3.0)
df.to_csv(out, index=False)
print(df["tic"].notna().sum(), "TIC matched")
