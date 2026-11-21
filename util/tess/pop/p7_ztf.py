"""ZTF veto for TESS template candidates. The leak sits in the 1-um comm band, inside TESS and outside ZTF g/r,
so a line that ZTF also shows is stellar (or blended), not leak. ZTF cannot confirm, only veto.

Per candidate north of dec -28: ZTF g, r (catflags = 0, mag > 12.5 to avoid saturation), Mira cycle removed with
the same 15-d spline per observing season, Lomb-Scargle amplitude at the TESS peak frequency (+-1/T_TESS) and its
noise from the same band away from the peak.
usage: python p7_ztf.py SCRATCH_DIR [N_TOP]  ->  SCRATCH_DIR/ztf_veto.csv"""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\guill\cepheid-internet\src")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from astropy.timeseries import LombScargle
from stage1_pulse import _ztf_raw
from cyc import fit
S = sys.argv[1]; NTOP = int(sys.argv[2]) if len(sys.argv) > 2 else 60
c = pd.read_csv(f"{S}/candidates.csv")
c = c[c.pass_cm & c.pass_shape & c.pass_repeat & (c.fap_emp_pop < 0.01) & (c.dec > -28)].head(NTOP)
os.makedirs(f"{S}/ztf", exist_ok=True); rows = []
for _, r in c.iterrows():
    fn = f"{S}/ztf/{r.tic}.parquet"
    try:
        z = pd.read_parquet(fn) if os.path.exists(fn) else _ztf_raw(f"RADEC {r.ra} {r.dec}", pos=(r.ra, r.dec))
    except Exception as e:
        print(r.Name, "ERR", repr(e)[:80]); continue
    if z is None or z.empty: rows.append(dict(tic=r.tic, Name=r.Name)); continue
    z.to_parquet(fn); out = dict(tic=r.tic, Name=r.Name, f_peak=r.f_cal, tess_amp_ppm=r.amp_peak_ppm)
    for fc in ["zg", "zr"]:
        zz = z[(z.filtercode == fc) & (z.catflags == 0) & (z.mag > 12.5)].sort_values("mjd")
        if len(zz) < 40: continue
        season = np.r_[0, np.cumsum(np.diff(zz.mjd.values) > 60)]
        res = []
        for s in np.unique(season):
            k = season == s; t = zz.mjd.values[k]
            if k.sum() < 15 or np.ptp(t) < 30: continue
            rr, _ = fit(t, zz.mag.values[k], 1/zz.magerr.values[k]**2, None, t.min(), t.max())
            res.append(pd.DataFrame({"t": t, "r": rr}))
        if not res: continue
        x = pd.concat(res); n = len(x)
        df = 1/r["T"]; fgrid = np.arange(0.02, 0.5, 0.1*df)
        A = np.sqrt(4*LombScargle(x.t, x.r, normalization="psd").power(fgrid)/n)/1.0857e-6
        win = lambda f0: A[np.abs(fgrid - f0) < df].max()                 # ZTF resolves far finer than TESS
        on = win(r.f_cal)
        off = np.array([win(f0) for f0 in np.arange(0.03, 0.49, 2*df) if abs(f0 - r.f_cal) > 3*df])
        out.update({f"{fc}_n": n, f"{fc}_amp_ppm": float(on), f"{fc}_noise_ppm": float(np.median(off)),
                    f"{fc}_pctile": float(np.mean(off < on))})
    rows.append(out); print(out, flush=True)
v = pd.DataFrame(rows); v.to_csv(f"{S}/ztf_veto.csv", index=False)
