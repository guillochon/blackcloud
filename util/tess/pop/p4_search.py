"""Population search of TESS Mira residuals for the superstep burst train (research/tess-spectral-search.md).

Per star, per contiguous TESS segment (sectors closer than 10 d):
  1. 0.25-d bins of SAP magnitude, Mira cycle removed by the X Dra model (15-d B-spline + per-orbit offsets, cyc.fit),
     one 5xMAD clip for scattered-light ramps.
  2. Lomb-Scargle PSD of the residual on [1/T, 2] 1/d; red-noise continuum C(f) from a robust quadratic fit in
     log-log (ratio_spectrum). R(f) = P/C is ~Exp(1) under the null.
  3. Template search: the burst train (FWHM 0.3 P) puts 87% of its detectable power in the fundamental, so the
     statistic is max R over P = 3-30 d (f >= 3/T). Per-segment FAP = 1 - (1 - e^-Rmax)^N_indep.
  4. Shape check at the peak: brightening Gaussian bursts vs dips (the leak only adds light; -r is flux-like).
  5. Band rms (0.03-0.25, 0.25-0.5 1/d) for the broadband stochastic model, and R on a common absolute grid
     for the common-mode systematics test (p5).
usage: python p4_search.py SCRATCH_DIR  ->  SCRATCH_DIR/search.parquet, SCRATCH_DIR/rgrid.npz"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor
from astropy.timeseries import LombScargle
from cyc import fit

PMIN, PMAX, FWHM = 3.0, 30.0, 0.3
FGRID = np.arange(0.02, 0.5, 0.0005)                       # common grid for the common-mode test

def segments(lc):
    """0.25-d binned magnitudes for each contiguous run of sectors, with orbit ids."""
    lc = lc[lc.n >= 0.5*lc.nexp].copy(); lc["mjd"] = lc.btjd + 56999.5
    lc = lc.sort_values("mjd").reset_index(drop=True); lc["orbit"] = 0
    for s, g in lc.groupby("sector"):
        d = np.diff(g.mjd.values)
        if len(d) == 0: continue
        gap = g.mjd.values[np.argmax(d)]
        lc.loc[g.index, "orbit"] = int(s)*2 + (g.mjd > gap).astype(int)
    lc["seg"] = np.r_[0, np.cumsum(np.diff(lc.mjd.values) > 10)]
    for _, g in lc.groupby("seg"):
        t0 = g.mjd.min(); g = g.assign(bin=np.floor((g.mjd - t0)/0.25), cov=g.n/g.nexp/12, nt=g.mjd*g.n, nf=g.sap*g.n)
        b = g.groupby(["orbit", "bin"])[["n", "nt", "nf", "cov"]].sum().reset_index()
        b["mjd"], b["sap"] = b.nt/b.n, b.nf/b.n
        b = b[b["cov"] > 0.5]
        yield g, b

def resid(mjd, m, orbit, k=None):
    """Cycle-fit residual. The 5xMAD clip only touches bins within 1.5 d of an orbit edge, where the scattered-light
    ramps are: clipping everywhere removes the peaks of a strong burst train. Pass k to reuse a mask (injections)."""
    t0, t1 = mjd.min(), mjd.max()
    if k is None:
        r, _ = fit(mjd, m, np.ones(len(m)), orbit, t0, t1)
        s = pd.Series(mjd); edge = np.minimum(s - s.groupby(orbit).transform("min"), s.groupby(orbit).transform("max") - s) < 1.5
        mad = 1.4826*np.median(np.abs(r - np.median(r))); k = ~(edge.values & (np.abs(r - np.median(r)) > 5*mad))
    r2, mod = fit(mjd[k], m[k], np.ones(k.sum()), orbit[k], t0, t1)
    return k, r2, mod

def continuum(f, p, use):
    """Quadratic in log-log over the bins in `use`, refit without points > 4x or < 1/30x the model (an Exp(1) variate
    falls there 2% and 3% of the time), log-bias corrected (E[ln Exp(1)] = -gamma)."""
    lf, lp, k = np.log(f), np.log(p), use.copy()
    for _ in range(5):
        c = np.exp(np.polyval(np.polyfit(lf[k], lp[k], 2), lf) + np.euler_gamma)
        k = use & (p < 4*c) & (p > c/30)
    return c

def ratio_spectrum(t, r, T):
    """R = P/C with C from the prewhitened series: up to three lines with R > 10 (fundamental + first harmonic) are
    subtracted in time before the continuum is refit. Without this a strong burst train leaks through the TESS window
    into many bins and lifts its own continuum (a 3% rms, 8-d train reached only R ~ 45)."""
    f = np.arange(1/T, 2.0, 1/(5*T))
    p = LombScargle(t, r, normalization="psd").power(f)
    x, px, use = r.copy(), p, np.ones(len(f), bool); c = continuum(f, px, use)
    for _ in range(3):
        i = np.argmax(px/c)
        if px[i]/c[i] < 10: break
        X = np.column_stack([np.ones_like(t)] + [fn(2*np.pi*k*f[i]*t) for k in (1, 2) for fn in (np.cos, np.sin)])
        x = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
        use &= (np.abs(f - f[i]) > 1/T) & (np.abs(f - 2*f[i]) > 1/T)          # the subtracted lobes leave holes
        px = LombScargle(t, x, normalization="psd").power(f); c = continuum(f, px, use)
    return f, p/c, p, c

def burst(t, P, phi, fwhm=FWHM):
    sig = fwhm*P/2.3548; ph = ((t/P - phi + 0.5) % 1) - 0.5
    return np.exp(-0.5*(ph*P/sig)**2)

def shape_check(t, r, P):
    """Best positive-burst amplitude minus best dip amplitude (chi2 gain of bursts over dips), phase scanned."""
    best = {1: 0.0, -1: 0.0}
    for phi in np.linspace(0, 1, 40, endpoint=False):
        s = burst(t, P, phi); s -= s.mean(); a = np.dot(s, r)/np.dot(s, s)
        g = a*np.dot(s, r)
        sg = 1 if a > 0 else -1; best[sg] = max(best[sg], g)
    return (best[1] - best[-1])/np.var(r)/len(r)

def band_rms(t, r, T, lo, hi):
    f = np.arange(np.ceil(lo*T), np.floor(hi*T) + 1)/T
    if len(f) == 0: return np.nan
    X = np.hstack([np.cos(2*np.pi*np.outer(t - t.min(), f)), np.sin(2*np.pi*np.outer(t - t.min(), f))])
    c, *_ = np.linalg.lstsq(X, r - r.mean(), rcond=None); return float(np.std(X @ c))

def one(path):
    tic = int(os.path.basename(path)[:-8]); rows, grids = [], []
    try: lc = pd.read_parquet(path)
    except Exception: return rows, grids
    for j, (g, b) in enumerate(segments(lc)):
        T = b.mjd.max() - b.mjd.min() if len(b) else 0
        if T < 20 or len(b) < 60: continue
        mjd, m, orb = b.mjd.values, -2.5*np.log10(b.sap.values), b.orbit.values
        k, r, mod = resid(mjd, m, orb); t = mjd[k]
        f, R, p, c = ratio_spectrum(t, r, T)
        sel = (f >= max(1/PMAX, 3/T)) & (f <= 1/PMIN)
        if sel.sum() < 5: continue
        i = np.argmax(np.where(sel, R, -1)); n_ind = (f[sel].max() - f[sel].min())*T
        fap = -np.expm1(n_ind*np.log1p(-np.exp(-R[i])))
        secs = sorted(g.sector.unique()); cams = sorted(set(zip(g.sector, g.camera, g.ccd)))
        rows.append(dict(tic=tic, seg=j, mjd0=mjd.min(), T=T, nbin=len(r), sectors=",".join(map(str, secs)),
                         cams=";".join(f"{a}-{b_}-{c_}" for a, b_, c_ in cams), prov=",".join(sorted(g.prov.unique())),
                         mira_ptp_mag=float(np.ptp(mod)), rms_ppm=float(np.std(r)/1.0857e-6),
                         p2p_ppm=float(np.median(np.abs(np.diff(r)))*1.4826/np.sqrt(2)/1.0857e-6),
                         bandA_ppm=band_rms(t, r, T, 0.03, 0.25)/1.0857e-6, bandB_ppm=band_rms(t, r, T, 0.25, 0.5)/1.0857e-6,
                         f_peak=float(f[i]), R_peak=float(R[i]), n_indep=float(n_ind), fap=float(fap),
                         amp_peak_ppm=float(np.sqrt(4*p[i]/len(r))/1.0857e-6), shape=float(shape_check(t, -r, 1/f[i]))))
        grids.append(np.interp(FGRID, f, R, left=np.nan, right=np.nan).astype(np.float32))
    return rows, grids

if __name__ == "__main__":
    S = sys.argv[1]; files = sorted(glob.glob(f"{S}/lc/*.parquet"))
    if len(sys.argv) > 2: files = files[:int(sys.argv[2])]
    rows, grids = [], []
    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
        for i, (r, g) in enumerate(ex.map(one, files, chunksize=8)):
            rows += r; grids += g
            if i % 500 == 0: print(i, len(files), len(rows), flush=True)
    d = pd.DataFrame(rows); d.to_parquet(f"{S}/search.parquet")
    np.savez_compressed(f"{S}/rgrid.npz", f=FGRID, R=np.array(grids))
    print(len(d), "segments from", d.tic.nunique(), "stars")
