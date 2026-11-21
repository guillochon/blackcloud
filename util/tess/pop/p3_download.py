"""Download SPOC 2-min (preferred) or TESS-SPOC FFI light curves, keep hard-masked SAP flux in 30-min bins.
One parquet per star in SCRATCH_DIR/lc/; the FITS files are deleted after reading. Resumable.
usage: python p3_download.py SCRATCH_DIR"""
import sys, os, io, time, numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
from astroquery.mast import Observations
from lightkurve.utils import TessQualityFlags
S = sys.argv[1]; os.makedirs(f"{S}/lc", exist_ok=True)
o = pd.read_csv(f"{S}/obs.csv"); o = o[o.t_exptime > 60]
o["rank"] = (o.provenance_name != "SPOC").astype(int)                # prefer SPOC 2-min over TESS-SPOC per sector
o = o.sort_values("rank").drop_duplicates(["target_name", "sequence_number"])
if not os.path.exists(f"{S}/products.csv"):
    parts = []
    for i in range(0, len(o), 500):
        p = Observations.get_product_list([str(x) for x in o.obsid.values[i:i+500]]).to_pandas()
        parts.append(p[p.productFilename.str.endswith("lc.fits") & ~p.productFilename.str.contains("fast")][["obsID", "dataURI", "productFilename"]])
        print("products", i, flush=True)
    pd.concat(parts).to_csv(f"{S}/products.csv", index=False)
p = pd.read_csv(f"{S}/products.csv").merge(o.assign(obsID=o.obsid)[["obsID", "target_name", "provenance_name"]], on="obsID")
p = p.drop_duplicates("productFilename")
BIN = 1/48
def read(buf):
    with fits.open(io.BytesIO(buf)) as h:
        d, hd = h[1].data, h[0].header
        t, f, q = np.asarray(d["TIME"], float), np.asarray(d["SAP_FLUX"], float), np.asarray(d["QUALITY"], int)
        cad = float(np.nanmedian(np.diff(t)))
        k = np.isfinite(t) & np.isfinite(f) & (f > 0) & ((q & TessQualityFlags.HARD_BITMASK) == 0)
        df = pd.DataFrame({"t": t[k], "f": f[k]}); df["b"] = np.floor(df.t/BIN).astype(int)
        g = df.groupby("b").agg(btjd=("t", "mean"), sap=("f", "mean"), n=("f", "size")).reset_index(drop=True)
        g["nexp"] = BIN/cad; g["sector"] = hd["SECTOR"]; g["camera"] = hd["CAMERA"]; g["ccd"] = hd["CCD"]
        return g
def star(tic, rows):
    out = f"{S}/lc/{tic}.parquet"
    if os.path.exists(out): return 0
    secs = []
    for _, r in rows.iterrows():
        for k in range(4):
            try:
                x = requests.get("https://mast.stsci.edu/api/v0.1/Download/file", params={"uri": r.dataURI}, timeout=120)
                x.raise_for_status(); secs.append(read(x.content).assign(prov=r.provenance_name)); break
            except Exception as e: time.sleep(5*(k+1)); err = e
        else: print(tic, r.productFilename, "FAILED", repr(err)[:80], flush=True)
    if secs: pd.concat(secs).to_parquet(out)
    return len(secs)
groups = list(p.groupby("target_name")); n = 0
with ThreadPoolExecutor(12) as ex:
    for i, k in enumerate(ex.map(lambda g: star(*g), groups)):
        n += k
        if i % 200 == 0: print(f"{i}/{len(groups)} stars, {n} sectors", flush=True)
print("done", len(groups), "stars", n, "sectors")
