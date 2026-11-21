"""All VSX Miras (type M, any brightness), cross-matched to the TIC; keep Tmag < 14 (TESS-SPOC FFI reach).
usage: python p1_targets.py SCRATCH_DIR  ->  SCRATCH_DIR/targets.csv"""
import sys; sys.path.insert(0, r"C:\Users\guill\cepheid-internet\src")
import pandas as pd
from astroquery.vizier import Vizier
from stage0_build_targets import crossmatch_tic
S = sys.argv[1]
v = Vizier(columns=["OID", "Name", "Type", "RAJ2000", "DEJ2000", "max", "n_max", "min", "n_min", "Period"], row_limit=-1)
df = v.query_constraints(catalog="B/vsx/vsx", Type="=M")[0].to_pandas()
print(len(df), "VSX type M")
df = df.rename(columns={"OID": "source_id", "RAJ2000": "ra", "DEJ2000": "dec"})
for c in ["max", "min", "Period"]: df[c] = pd.to_numeric(df[c], errors="coerce")
df = crossmatch_tic(df, 3.0).dropna(subset=["tic"])
df = df[df.tmag < 14].drop_duplicates("tic")
df.to_csv(f"{S}/targets.csv", index=False)
print(len(df), "with TIC and Tmag < 14")
