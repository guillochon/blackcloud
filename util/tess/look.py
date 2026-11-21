import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cyc import load, fit
S = sys.argv[1]; t, z = load(S); t0, t1 = t.mjd.min(), t.mjd.max()
# 0.25-d bins for TESS to keep the fit small
t["bin"] = np.floor((t.mjd - t0)/0.25)
tb = t.groupby(["sector", "bin"]).agg(mjd=("mjd","mean"), m=("m","mean"), n=("m","size"), s=("m","std")).reset_index()
tb = tb[tb.n > 20]
rT, mT = fit(tb.mjd.values, tb.m.values, np.ones(len(tb)), tb.sector.values, t0, t1)
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(tb.mjd, tb.m, ".", ms=2, label="TESS SAP (inst mag)"); ax[0].plot(tb.mjd, mT, "k-", lw=0.8); ax[0].invert_yaxis()
for fc, col in [("zg","g"),("zr","r")]:
    zz = z[z.filtercode == fc]
    r, mm = fit(zz.mjd.values, zz.mag.values, 1/zz.magerr.values**2, None, t0, t1)
    ax[1].errorbar(zz.mjd, zz.mag, zz.magerr, fmt=".", color=col, ms=3, label=fc); ax[1].plot(zz.mjd, mm, "-", color=col, lw=0.8)
    ax[2].plot(zz.mjd, r, ".", color=col, ms=3, label=f"{fc} resid rms {np.std(r)*1e3:.1f} mmag (med err {np.median(zz.magerr)*1e3:.1f})")
    print(fc, len(zz), "resid rms", np.std(r), "median err", np.median(zz.magerr))
ax[1].invert_yaxis(); ax[1].legend()
ax[2].plot(tb.mjd, rT, "k.", ms=1, label=f"TESS resid rms {np.std(rT)*1e3:.2f} mmag")
ax[2].legend(fontsize=7); ax[2].set_ylim(-0.08, 0.08)
fig.savefig(f"{S}/xdra_look.png", dpi=110)
print("TESS bins", len(tb), "resid rms", np.std(rT))
