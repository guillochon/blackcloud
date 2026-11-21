"""MAST inventory of SPOC (2-min) and TESS-SPOC (FFI) light-curve observations for the targets.
usage: python p2_products.py SCRATCH_DIR  ->  SCRATCH_DIR/obs.csv"""
import sys, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from astroquery.mast import Observations
S = sys.argv[1]
tics = pd.read_csv(f"{S}/targets.csv").tic.astype(int).astype(str).unique().tolist()
COLS = ["obsid", "target_name", "obs_collection", "provenance_name", "sequence_number", "t_exptime", "t_min", "t_max"]
def q(i):
    for k in range(3):
        try:
            r = Observations.query_criteria(target_name=tics[i:i+300], obs_collection=["TESS", "HLSP"],
                                            provenance_name=["SPOC", "TESS-SPOC"], dataproduct_type="timeseries")
            return r.to_pandas()[COLS] if len(r) else None
        except Exception as e: err = e
    print(i, "FAILED", repr(err)[:100], flush=True)
with ThreadPoolExecutor(6) as ex: parts = list(ex.map(q, range(0, len(tics), 300)))
o = pd.concat([p for p in parts if p is not None])
o.to_csv(f"{S}/obs.csv", index=False)
print(len(o), "observations,", o.target_name.nunique(), "stars"); print(o.groupby(["provenance_name", "t_exptime"]).size())
