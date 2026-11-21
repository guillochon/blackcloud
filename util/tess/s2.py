import sys, pandas as pd, numpy as np
from astroquery.mast import Observations
df = pd.read_csv(sys.argv[1]).dropna(subset=["tic"])
df["tic"] = df["tic"].astype(int).astype(str)
tics = df["tic"].unique().tolist()
parts = []
for i in range(0, len(tics), 200):
    r = Observations.query_criteria(target_name=tics[i:i+200], obs_collection="TESS",
                                    dataproduct_type="timeseries")
    if len(r): parts.append(r.to_pandas()[["target_name","obs_collection","provenance_name","sequence_number","t_exptime","t_min","t_max"]])
    print(i, flush=True)
o = pd.concat(parts)
o.to_csv(sys.argv[2], index=False)
