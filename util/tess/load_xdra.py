import sys, numpy as np, pandas as pd, lightkurve as lk, warnings; warnings.filterwarnings("ignore")
S = sys.argv[1]
sr = lk.search_lightcurve("TIC 219111486", mission="TESS", author="TESS-SPOC")
lcs = sr.download_all(download_dir=f"{S}/tessdl_xdra", quality_bitmask="hard")
rows = []
for lc in lcs:
    l = lc.select_flux("sap_flux").remove_nans()
    rows.append(pd.DataFrame({"btjd": l.time.value, "sap": np.asarray(l.flux.value, float),
                              "sector": lc.sector, "exp": float(lc.meta.get("TIMEDEL", np.nan))*86400}))
t = pd.concat(rows); t["mjd"] = t.btjd + 2457000 - 2400000.5
t.to_parquet(f"{S}/xdra_tess.parquet")
print(t.groupby("sector").agg(n=("sap","size"), mjd0=("mjd","min"), mjd1=("mjd","max"), med=("sap","median"), exp=("exp","first")))
z = pd.read_parquet(f"{S}/ztf/X_Dra.parquet")
z = z[(z.catflags == 0) & (z.mag > 12.5)]
w = z[(z.mjd > t.mjd.min()-5) & (z.mjd < t.mjd.max()+5)]
print("ZTF in TESS window:", w.groupby("filtercode").agg(n=("mag","size"), nights=("mjd", lambda x: np.unique(np.floor(x)).size), mag_lo=("mag","min"), mag_hi=("mag","max"), err=("magerr","median")))
w.to_parquet(f"{S}/xdra_ztf.parquet")
