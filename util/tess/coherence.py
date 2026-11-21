"""TESS vs ZTF cross-band coherence of Mira-cycle residuals for X Dra, plus leak injection.

Leak model (notes/tess-spectral-search.md): red-band-only stochastic excess, PSD ~ sinc^2(2 f R/c) / (1 + (f/f_c)^2),
R/c = 1.27 d, f_c = v/(pi R) with v = 0.38c -> 0.095 1/d.  Injected into TESS only, before the cycle fit.
"""
import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd, json
from cyc import load, fit
S = sys.argv[1]; rng = np.random.default_rng(42)
BANDS = {"A 0.03-0.25": (0.03, 0.25), "B 0.25-0.5": (0.25, 0.5)}
RC, FC = 1.27, 0.38/(np.pi*1.27)

t, z = load(S); t0, t1 = t.mjd.min(), t.mjd.max(); T = t1 - t0
# orbit id: split each sector at its largest internal gap
t = t.sort_values("mjd").reset_index(drop=True); t["orbit"] = 0
for s, g in t.groupby("sector"):
    gap = g.mjd.values[np.argmax(np.diff(g.mjd.values))]
    t.loc[g.index, "orbit"] = int(s)*2 + (g.mjd > gap).astype(int)
t["bin"] = np.floor((t.mjd - t0)/0.25)

KEEP = None
def tess_resid(tt):
    """Binned TESS residuals. The ramp clip is decided once, on the real data, and reused for injections."""
    global KEEP
    tb = tt.groupby(["orbit", "bin"]).agg(mjd=("mjd","mean"), m=("m","mean"), n=("m","size")).reset_index()
    tb = tb[tb.n > 20]
    if KEEP is None:
        r, _ = fit(tb.mjd.values, tb.m.values, np.ones(len(tb)), tb.orbit.values, t0, t1)
        k = np.abs(r - np.median(r)) < 5*1.4826*np.median(np.abs(r - np.median(r)))   # scattered-light ramps
        KEEP = set(zip(tb.orbit[k], tb.bin[k]))
    tb = tb[[(o, b) in KEEP for o, b in zip(tb.orbit, tb.bin)]]
    r, _ = fit(tb.mjd.values, tb.m.values, np.ones(len(tb)), tb.orbit.values, t0, t1)
    return tb.mjd.values, r, tb.orbit.values

def ztf_resid(fc):
    zz = z[z.filtercode == fc]
    r, _ = fit(zz.mjd.values, zz.mag.values, 1/zz.magerr.values**2, None, t0, t1)
    return zz.mjd.values, r, zz.magerr.values

def fourier_basis(tt, lo, hi):
    f = np.arange(np.ceil(lo*T), np.floor(hi*T)+1)/T
    return f, np.hstack([np.cos(2*np.pi*np.outer(tt, f)), np.sin(2*np.pi*np.outer(tt, f))])

def band_power(tt, x, lo, hi):
    f, X = fourier_basis(tt - t0, lo, hi)
    c, *_ = np.linalg.lstsq(X, x - x.mean(), rcond=None)
    return np.var(X @ c)

def coherence(tx, x, ty, y, lo, hi):
    """Band-averaged coherence of two series on the same epochs (DFT at 1/T spacing)."""
    f = np.arange(np.ceil(lo*T), np.floor(hi*T)+1)/T
    E = np.exp(-2j*np.pi*np.outer(f, tx - t0))
    X, Y = E @ (x - x.mean()), E @ (y - y.mean())
    sxy = np.sum(X*np.conj(Y)); return np.abs(sxy)**2/(np.sum(np.abs(X)**2)*np.sum(np.abs(Y)**2)), sxy/np.sum(np.abs(Y)**2)

def paired(tb, rT, tz, rz, half=0.25):
    out = []
    for i, tt in enumerate(tz):
        k = np.abs(tb - tt) < half
        if k.any(): out.append((tt, rT[k].mean(), rz[i]))
    a = np.array(out); return a[:, 0], a[:, 1], a[:, 2]

def leak(tt, rms):
    dt = 0.01; n = int(np.ceil((tt.max() - tt.min())/dt)) + 1
    n2 = 1 << int(np.ceil(np.log2(2*n)))
    f = np.fft.rfftfreq(n2, dt); P = np.sinc(2*f*RC)**2/(1 + (f/FC)**2)
    s = np.fft.irfft(np.sqrt(P)*(rng.normal(size=f.size) + 1j*rng.normal(size=f.size)), n2)[:n]
    s *= rms/np.std(s); return np.interp(tt, tt.min() + dt*np.arange(n), s)

tb, rT, orb = tess_resid(t)
res = {"star": "X Dra (TIC 219111486)", "sectors": "76-83 TESS-SPOC SAP, 200 s", "T_days": T,
       "tess_resid_rms_mag": float(np.std(rT)), "bands": {}}
for fc in ["zg", "zr"]:
    tz, rz, ez = ztf_resid(fc)
    tp, xT, yZ = paired(tb, rT, tz, rz)
    res[f"{fc}_n_paired"] = int(len(tp)); res[f"{fc}_resid_rms_mag"] = float(np.std(rz)); res[f"{fc}_med_err_mag"] = float(np.median(ez))
    for name, (lo, hi) in BANDS.items():
        g2, H = coherence(tp, xT, tp, yZ, lo, hi)
        null = np.array([coherence(tp, xT, tp, rng.permutation(yZ), lo, hi)[0] for _ in range(2000)])
        d = res["bands"].setdefault(name, {})
        d[fc] = {"gamma2": float(g2), "null_median": float(np.median(null)), "p": float((np.sum(null >= g2)+1)/(len(null)+1)),
                 "transfer_TESS_per_ZTF": float(np.abs(H))}
for name, (lo, hi) in BANDS.items():
    P = band_power(tb, rT, lo, hi)
    jk = np.array([band_power(tb[orb != o], rT[orb != o], lo, hi) for o in np.unique(orb)])
    sig = np.sqrt((len(jk)-1)*np.var(jk))
    res["bands"][name].update({"tess_band_rms_ppm": float(np.sqrt(P)*0.921e6), "tess_band_power_jk_sigma_frac": float(sig/P)})

# injection: recovered TESS band power vs injected leak rms (ppm of flux), 20 realizations each
amps = [300, 1000, 2000, 4000, 8000, 16000]
inj = {}
P0 = {n: band_power(tb, rT, *b) for n, b in BANDS.items()}
sig = {n: res["bands"][n]["tess_band_power_jk_sigma_frac"]*P0[n] for n in BANDS}
for A in amps:
    det = {n: [] for n in BANDS}; surv = []
    for k in range(20):
        tt = t.copy(); L = leak(tt.mjd.values, A*1e-6); tt["m"] = tt.m - 1.0857*L
        tbi, rTi, _ = tess_resid(tt)
        # fraction of injected leak variance that survives the cycle + offset fit
        Lb = pd.Series(L, index=tt.index).groupby([tt.orbit, tt.bin]).mean().reindex(pd.MultiIndex.from_arrays([orb, np.floor((tb - t0)/0.25)])).values  # tb/orb from the null run
        for n, b in BANDS.items():
            det[n].append(band_power(tbi, rTi, *b) - P0[n] > 3*sig[n])
        surv.append(np.var(rTi - rT)/np.var(1.0857*Lb[~np.isnan(Lb)]))
    inj[A] = {"det_frac": {n: float(np.mean(v)) for n, v in det.items()}, "variance_surviving_fit": float(np.median(surv))}
    print(A, inj[A], flush=True)
res["injection"] = inj
json.dump(res, open(f"{S}/xdra_coherence.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "injection"}, indent=1))
