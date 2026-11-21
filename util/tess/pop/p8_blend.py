"""Blend check for vetted candidates. A TESS pixel is 21", so a periodic neighbor can put a line in the Mira's SAP.
For every population-significant peak (FAP x N_segments < 0.01), flags a known variable within 2' (VSX) or a Gaia DR3 eclipsing binary or rotator within 1' whose
frequency, or half or twice it, lies within the Rayleigh resolution 1/T of the TESS peak.
usage: python p8_blend.py SCRATCH_DIR [N_TOP]  ->  SCRATCH_DIR/blend.csv"""
import sys, numpy as np, pandas as pd, astropy.units as u, warnings; warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
S = sys.argv[1]; NTOP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
c = pd.read_csv(f"{S}/candidates.csv")
c = c[c.fap_emp_pop < 0.01].head(NTOP)                               # vetted or not: blend statistics
vsx = Vizier(columns=["Name", "Type", "Period", "max", "_r"], row_limit=-1)
gv = Vizier(columns=["Source", "Class", "_r"], row_limit=-1)
geb = Vizier(columns=["Source", "Freq", "_r"], row_limit=-1)
grot = Vizier(columns=["Source", "BestPeriod", "_r"], row_limit=-1)
def match(P, Pn):                                                    # within the TESS Rayleigh resolution 1/T
    return any(abs(k/Pn - 1/P) < 1/T for k in (0.5, 1, 2))
rows = []
for _, r in c.iterrows():
    pos = SkyCoord(r.ra*u.deg, r.dec*u.deg); P, T = r.P_cal_d, r["T"]; out = dict(Name=r.Name, tic=r.tic, P_cal_d=P)
    t = vsx.query_region(pos, radius=2*u.arcmin, catalog="B/vsx/vsx")
    nb = t[0].to_pandas().rename(columns={"_r": "sep"}) if len(t) else pd.DataFrame(columns=["Name", "Type", "Period", "sep"])
    nb = nb[nb.Name != r.Name]
    out["vsx_neighbors"] = "; ".join(f"{x.Name} {x.Type} P={x.Period} r={float(x.sep)*60:.0f}\"" for x in nb.itertuples())
    out["vsx_match"] = any(np.isfinite(x.Period) and match(P, x.Period) for x in nb.itertuples())
    t = gv.query_region(pos, radius=1*u.arcmin, catalog="I/358/vclassre")
    out["gaia_vari"] = "; ".join(f"{x.Source} {x.Class} r={float(x.sep)*60:.0f}\"" for x in t[0].to_pandas().rename(columns={"_r": "sep"}).itertuples()) if len(t) else ""
    gm = []
    t = geb.query_region(pos, radius=1*u.arcmin, catalog="I/358/veb")
    if len(t): gm += [f"EB {x.Source} P={1/x.Freq:.3f}" for x in t[0].to_pandas().rename(columns={"_r": "sep"}).itertuples() if x.Freq > 0 and match(P, 1/x.Freq)]
    t = grot.query_region(pos, radius=1*u.arcmin, catalog="I/358/vrot")
    if len(t): gm += [f"ROT {x.Source} P={x.BestPeriod:.3f}" for x in t[0].to_pandas().rename(columns={"_r": "sep"}).itertuples() if np.isfinite(x.BestPeriod) and match(P, x.BestPeriod)]
    out["gaia_match"] = "; ".join(gm)
    rows.append(out); print(out, flush=True)
pd.DataFrame(rows).to_csv(f"{S}/blend.csv", index=False)
