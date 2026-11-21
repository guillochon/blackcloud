"""Population completeness for the superstep burst train: inject into real binned TESS magnitudes before the
cycle fit (as superstep_inject.py does for X Dra), rerun the p4 statistic, count recoveries.

Per sampled segment: P log-uniform in [3, min(30, T/3)] d, FWHM 0.3 P, 5% period jitter, random phase,
rms in {300, 1000, 3000, 10000, 30000} ppm of the flux. Recovered = calibrated peak (p5's empirical null for the
segment's sector/camera groups) within 2/T of 1/P or 2/P (the train has harmonics), and significant across the population: FAP x N_segments < 0.01.
usage: python p6_inject.py SCRATCH_DIR [N_SEGMENTS]  ->  SCRATCH_DIR/inject.parquet"""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ProcessPoolExecutor
from p4_search import segments, resid, ratio_spectrum, PMIN, PMAX, FWHM, FGRID
AMPS, NREAL, POP_FAP = [300, 1000, 3000, 10000, 30000], 4, 0.01
CAL = None

def pcal(Rg, cams):
    ps = []
    for key in sorted({tuple(map(int, c.split("-")[:2])) for c in cams.split(";")}):
        q, b = CAL.get(key, CAL[(key[0], 0)]); ps.append(np.minimum(1, 0.25*np.exp(-(Rg - q)/b)))
    return np.nanmax(ps, axis=0)

def train(t, P, rms, rng):
    sig = FWHM*P/2.3548; n = int(np.ptp(t)/P) + 4
    c = t.min() - P*rng.uniform() + np.cumsum(P*(1 + 0.05*rng.normal(size=n)))
    s = np.exp(-0.5*((t[:, None] - c[None, :])/sig)**2).sum(1); s -= s.mean()
    return s*rms/np.std(s)

def one(args):
    path, seg, seed, cams, nseg, S = args; rng = np.random.default_rng(seed); out = []
    global CAL
    if CAL is None:
        c = np.load(f"{S}/calib.npz"); CAL = {tuple(k): (q, b) for k, q, b in zip(c["keys"], c["q"], c["b"])}
    lc = pd.read_parquet(path)
    for j, (g, b) in enumerate(segments(lc)):
        if j != seg: continue
        mjd, m0, orb = b.mjd.values, -2.5*np.log10(b.sap.values), b.orbit.values; T = np.ptp(mjd)
        keep, _, _ = resid(mjd, m0, orb)                                   # ramp mask from the real data, reused
        for A in AMPS:
            for k in range(NREAL):
                P = np.exp(rng.uniform(np.log(PMIN), np.log(min(PMAX, T/3))))
                kk, r, _ = resid(mjd, m0 - 1.0857*train(mjd, P, A*1e-6, rng), orb, keep); t = mjd[kk]
                f, R, _, _ = ratio_spectrum(t, r, T)
                p = pcal(np.interp(FGRID, f, R, left=np.nan, right=np.nan), cams)
                sel = (FGRID >= max(1/PMAX, 3/T)) & (FGRID <= 1/PMIN) & np.isfinite(p)
                i = np.argmin(np.where(sel, p, np.inf)); n_ind = (1/PMIN - max(1/PMAX, 3/T))*T
                fap = -np.expm1(n_ind*np.log1p(-p[i]))
                out.append(dict(tic=int(os.path.basename(path)[:-8]), seg=seg, T=T, amp_ppm=A, P=P, fap=fap,
                                f_det=FGRID[i], det=bool(fap*nseg < POP_FAP and min(abs(FGRID[i] - k/P) for k in (1, 2)) < 2/T)))
    return out

if __name__ == "__main__":
    S = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    d = pd.read_parquet(f"{S}/search.parquet"); nseg = len(d); d = d.sample(n, random_state=1)
    jobs = [(f"{S}/lc/{r.tic}.parquet", int(r.seg), i, r.cams, nseg, S) for i, r in enumerate(d.itertuples())]
    rows = []
    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
        for i, o in enumerate(ex.map(one, jobs, chunksize=4)):
            rows += o
            if i % 100 == 0: print(i, flush=True)
    x = pd.DataFrame(rows).merge(d[["tic", "seg", "rms_ppm", "p2p_ppm", "prov", "sectors"]], on=["tic", "seg"])
    x.to_parquet(f"{S}/inject.parquet")
    print(x.groupby("amp_ppm").det.mean())
