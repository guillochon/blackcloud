"""Inject superstep-cadence burst trains (paper Sec. 'Workloads across the computing hierarchy') into X Dra TESS
residuals and compare a broadband band-power detector with a periodogram line search.

Signal: one burst per global superstep, Gaussian with the unresolved broadcast FWHM of the paper's fig5
(backbone 1.6-4 d -> 2.8 d with snub-cube hubs; mesh 8-14 d -> 11 d), superstep periods from outputs/cells.json (9.4 d over
the 24-hub backbone, between the 13.7 d TESS orbit and its second harmonic; 25.7 d over the mesh), 5% cycle-to-cycle
period jitter, random start phase, scaled to a given rms of the TESS flux. Injected before the cycle fit.
usage: python superstep_inject.py SCRATCH_DIR   (needs xdra_tess.parquet from load_xdra.py)"""
import sys, json; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np, pandas as pd
from astropy.timeseries import LombScargle
from cyc import load, fit
S = sys.argv[1]; rng = np.random.default_rng(7)
cells = json.load(open(__import__("pathlib").Path(__file__).parents[2] / "outputs" / "cells.json"))["workloads"]["cloud"]
CASES = {"backbone": dict(P=cells["superstep_backbone_d"], fwhm=2.8, search=(0.085, 0.135)),
         "mesh": dict(P=cells["superstep_mesh_d"], fwhm=11.0, search=(0.025, 0.06))}

t, _ = load(S); t0, t1 = t.mjd.min(), t.mjd.max(); T = t1 - t0
t = t.sort_values("mjd").reset_index(drop=True); t["orbit"] = 0
for s, g in t.groupby("sector"):
    gap = g.mjd.values[np.argmax(np.diff(g.mjd.values))]
    t.loc[g.index, "orbit"] = int(s)*2 + (g.mjd > gap).astype(int)
t["bin"] = np.floor((t.mjd - t0)/0.25)
KEEP = None
def tess_resid(tt):
    global KEEP
    tb = tt.groupby(["orbit", "bin"]).agg(mjd=("mjd","mean"), m=("m","mean"), n=("m","size")).reset_index()
    tb = tb[tb.n > 20]
    if KEEP is None:
        r, _ = fit(tb.mjd.values, tb.m.values, np.ones(len(tb)), tb.orbit.values, t0, t1)
        k = np.abs(r - np.median(r)) < 5*1.4826*np.median(np.abs(r - np.median(r)))
        KEEP = set(zip(tb.orbit[k], tb.bin[k]))
    tb = tb[[(o, b) in KEEP for o, b in zip(tb.orbit, tb.bin)]]
    r, _ = fit(tb.mjd.values, tb.m.values, np.ones(len(tb)), tb.orbit.values, t0, t1)
    return tb.mjd.values, r, tb.orbit.values
def band_power(tt, x, lo, hi):
    f = np.arange(np.ceil(lo*T), np.floor(hi*T)+1)/T
    X = np.hstack([np.cos(2*np.pi*np.outer(tt - t0, f)), np.sin(2*np.pi*np.outer(tt - t0, f))])
    c, *_ = np.linalg.lstsq(X, x - x.mean(), rcond=None); return np.var(X @ c)
def burst_train(tt, P, fwhm, rms):
    sig = fwhm/2.3548; n = int((tt.max() - tt.min())/P) + 4
    centers = tt.min() - P*rng.uniform(0, 1) + np.cumsum(P*(1 + 0.05*rng.normal(size=n)))
    s = np.exp(-0.5*((tt[:, None] - centers[None, :])/sig)**2).sum(1)
    s -= s.mean(); return s*rms/np.std(s)

tb, rT, orb = tess_resid(t)
A_lo, A_hi = 0.03, 0.25
P0 = band_power(tb, rT, A_lo, A_hi)
jk = np.array([band_power(tb[orb != o], rT[orb != o], A_lo, A_hi) for o in np.unique(orb)])
sigP = np.sqrt((len(jk)-1)*np.var(jk))
out = {}
for name, cs in CASES.items():
    fs = np.linspace(*cs["search"], 400)
    pk0 = LombScargle(tb, rT).power(fs).max()
    res = {}
    for A in [300, 1000, 2000, 4000, 8000]:
        bb, ln, surv = [], [], []
        for k in range(20):
            tt = t.copy(); L = burst_train(tt.mjd.values, cs["P"], cs["fwhm"], A*1e-6); tt["m"] = tt.m - 1.0857*L
            tbi, rTi, _ = tess_resid(tt)
            bb.append(band_power(tbi, rTi, A_lo, A_hi) - P0 > 3*sigP)
            pw = LombScargle(tbi, rTi).power(fs); j = np.argmax(pw)
            ln.append(bool(pw[j] > 2*pk0 and abs(fs[j] - 1/cs["P"]) < 2/T))
            Lb = pd.Series(L).groupby([tt.orbit, tt.bin]).mean()
            Lb = Lb.reindex(pd.MultiIndex.from_arrays([orb, np.floor((tb - t0)/0.25)])).values
            surv.append(np.var(rTi - rT)/np.var(1.0857*Lb))
        res[A] = {"broadband": float(np.mean(bb)), "line": float(np.mean(ln)), "surviving_variance": float(np.median(surv))}
        print(name, A, res[A], flush=True)
    out[name] = {"period_d": cs["P"], "fwhm_d": cs["fwhm"], "search_1_per_d": cs["search"], "real_peak_power": float(pk0), "injections": res}
json.dump(out, open(__import__("pathlib").Path(__file__).parent / "xdra_superstep.json", "w"), indent=1)
