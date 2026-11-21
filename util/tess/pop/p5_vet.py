"""Calibrate, rank and vet superstep-template peaks from p4_search.

The analytic null (R ~ Exp(1)) fails: TESS leaves lines at the 13.7-d orbit harmonics (0.146, 0.29 1/d) and the
momentum-dump cadence, and R is inflated across the band (2922 segments at FAP < 0.1 against 789 expected).
So the null is empirical. For each (sector, camera) group with >= 100 segments (else the whole sector), and at each
frequency of the common grid, the upper quartile of R over all Miras in that group is fit with an exponential tail,
p(R) = 0.25 exp(-(R - q75)/b). For pure Exp(1) noise this returns p = e^-R exactly. A segment's p(f) is the largest
(most conservative) over the groups it belongs to; its FAP is 1 - (1 - min_f p)^N_indep over P = 3-30 d, f >= 3/T.

Vetting at the calibrated peak (the per-group null already absorbs lines common to a sector and camera):
  cluster      another star's segment in a shared sector with FAP < 1e-3 peaking within 0.3/T (one systematic line
               hitting several stars; a +-1/T window or an R > 5 fraction vetoes everything in crowded sectors)
  repeat       R at that frequency in the star's other segments that can resolve it
  shape        brightening bursts fit better than dips (the leak only adds light)
  cm_frac_gt5  fraction of co-sector Miras with R > 5 at that frequency, reported, not cut on
usage: python p5_vet.py SCRATCH_DIR  ->  SCRATCH_DIR/vetted.parquet, candidates.csv, calib.npz"""
import sys, numpy as np, pandas as pd
S = sys.argv[1]
d = pd.read_parquet(f"{S}/search.parquet").reset_index(drop=True)
g = np.load(f"{S}/rgrid.npz"); F, R = g["f"], g["R"]
tg = pd.read_csv(f"{S}/targets.csv")[["tic", "Name", "max", "min", "Period", "tmag", "ra", "dec"]]
tg["tic"] = tg.tic.astype(int); d = d.merge(tg, on="tic", how="left")
N = len(d)

def group_keys(cams):
    return sorted({tuple(map(int, c.split("-")[:2])) for c in cams.split(";")})
memb = {}
for i, c in enumerate(d.cams):
    for s, cam in group_keys(c):
        memb.setdefault((s, cam), []).append(i); memb.setdefault((s, 0), []).append(i)

def tail(rows):
    x = R[rows]; q = np.nanquantile(x, 0.75, axis=0)
    b = np.array([np.nanmean(x[:, j][x[:, j] > q[j]]) - q[j] if np.isfinite(q[j]) else np.nan for j in range(len(F))])
    return q, np.maximum(b, 1.0)                                  # never more permissive than Exp(1)
cal = {k: tail(v) for k, v in memb.items() if len(v) >= 100 or k[1] == 0}

P = np.full_like(R, np.nan)
for i, c in enumerate(d.cams):
    ps = []
    for s, cam in group_keys(c):
        q, b = cal.get((s, cam), cal[(s, 0)])
        ps.append(np.minimum(1, 0.25*np.exp(-(R[i] - q)/b)))
    P[i] = np.nanmax(ps, axis=0)
lo = np.maximum(1/30, 3/d["T"].values)[:, None]
band = (F[None, :] >= lo) & (F[None, :] <= 1/3) & np.isfinite(P)
Pb = np.where(band, P, np.inf); j = np.argmin(Pb, axis=1)
d["p_min"] = Pb[np.arange(N), j]; d["f_cal"] = F[j]; d["R_cal"] = R[np.arange(N), j]; d["P_cal_d"] = 1/d.f_cal
d["fap_emp"] = -np.expm1(d.n_indep*np.log1p(-d.p_min)); d["fap_emp_pop"] = np.minimum(1, d.fap_emp*N)
np.savez_compressed(f"{S}/calib.npz", f=F, keys=np.array(list(cal.keys())), q=np.array([v[0] for v in cal.values()]),
                    b=np.array([v[1] for v in cal.values()]))

secs = [set(map(int, s.split(","))) for s in d.sectors]
by_sec = {}
for i, s in enumerate(secs):
    for x in s: by_sec.setdefault(x, []).append(i)
def at(i, f):
    k = np.argmin(np.abs(F - f)); w = R[i, max(0, k-2):k+3]
    return np.nanmax(w) if np.isfinite(w).any() else np.nan
cm, rep_R, rep_n = [], [], []
for i, r in d.iterrows():
    others = [o for o in set(sum((by_sec[x] for x in secs[i]), [])) if d.tic[o] != r.tic]
    v = np.array([at(o, r.f_cal) for o in others]); v = v[np.isfinite(v)]
    cm.append(np.mean(v > 5) if len(v) else np.nan)               # null for a max over 5 bins: ~0.03
    same = [s for s in d.index[(d.tic == r.tic) & (d.index != i)] if r.f_cal >= 3/d["T"][s]]
    w = np.array([at(s, r.f_cal) for s in same]); w = w[np.isfinite(w)]
    rep_R.append(np.min(w) if len(w) else np.nan); rep_n.append(len(w))
d["cm_frac_gt5"], d["repeat_min_R"], d["repeat_n"] = cm, rep_R, rep_n
d.to_parquet(f"{S}/vetted.parquet")

print(N, "segments,", d.tic.nunique(), "stars")
for q in [0.5, 0.1, 0.01, 1e-3, 1e-4]:
    print(f"calibrated per-segment FAP < {q:g}: {np.sum(d.fap_emp < q)} (expect {N*q:.1f})")
c = d[d.fap_emp < 1e-2].copy()
c["pass_shape"] = c["shape"] > 0
c["pass_repeat"] = (c.repeat_n == 0) | (c.repeat_min_R > 3)
strong = d[d.fap_emp < 1e-3]; ss = [set(map(int, s.split(","))) for s in strong.sectors]
c["n_cluster"] = [sum(1 for k in range(len(strong)) if strong.tic.iloc[k] != r.tic and ss[k] & set(map(int, r.sectors.split(",")))
                      and abs(strong.f_cal.iloc[k] - r.f_cal) < 0.3/r["T"]) for _, r in c.iterrows()]
c["pass_cm"] = c.n_cluster == 0
c = c.sort_values("fap_emp"); c.to_csv(f"{S}/candidates.csv", index=False)
cols = ["Name", "tic", "tmag", "sectors", "T", "P_cal_d", "R_cal", "fap_emp", "fap_emp_pop", "amp_peak_ppm", "rms_ppm",
        "mira_ptp_mag", "cm_frac_gt5", "shape", "repeat_n", "repeat_min_R"]
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
ok = c.pass_cm & c.pass_shape & c.pass_repeat
print(len(c), "segments with calibrated FAP < 0.01;", int(ok.sum()), "pass all three vetting tests")
print(c[ok][cols].head(30).to_string(index=False))
