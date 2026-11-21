import sys, pandas as pd, numpy as np
m = pd.read_csv(sys.argv[1]).dropna(subset=["tic"]); m["tic"]=m["tic"].astype(int)
o = pd.read_csv(sys.argv[2]); o["tic"]=o["target_name"].astype(int)
print(o.provenance_name.value_counts().to_string()); print(o.t_exptime.value_counts().head().to_string())
g = o.groupby("tic").agg(n_sec=("sequence_number","nunique"), first=("sequence_number","min"), last=("sequence_number","max"),
     tmin=("t_min","min"), tmax=("t_max","max"), fast=("t_exptime", lambda x:(x<60).sum()))
g["span_yr"]=(g.tmax-g.tmin)/365.25
j = g.join(m.drop_duplicates("tic").set_index("tic")[["Name","max","min","Period","tmag","ra","dec"]])
print(len(j), "Miras with SPOC 2-min/20-s light curves")
print(j.n_sec.describe().to_string())
for k in [1,2,5,10,20,30]: print(f">= {k} sectors:", (j.n_sec>=k).sum())
pd.set_option("display.width",250)
print(j.sort_values("n_sec",ascending=False).head(40)[["Name","n_sec","first","last","span_yr","fast","Period","max","min","tmag"]].to_string())
j.to_csv(sys.argv[3])
