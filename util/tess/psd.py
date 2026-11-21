import sys, numpy as np, lightkurve as lk, warnings; warnings.filterwarnings("ignore")
dl = sys.argv[1]
sr = lk.search_lightcurve("TIC 38828619", mission="TESS", author="SPOC", exptime=120)
sec = np.array(sr.table["sequence_number"]); keep = (sec >= 27) & (sec <= 39)
print("sectors:", sorted(set(sec[keep])))
lcs = sr[keep].download_all(download_dir=dl, quality_bitmask="hard")
out = []
for lc in lcs:
    for col in ["pdcsap_flux", "sap_flux"]:
        l = lc.select_flux(col).remove_nans()
        t = l.time.value; y = l.flux.value / np.nanmedian(l.flux.value) - 1
        # remove the Mira cycle + offsets: quadratic per orbit (split at largest gap)
        g = np.argmax(np.diff(t)); res = np.empty_like(y)
        for sl in (slice(0, g+1), slice(g+1, None)):
            p = np.polyfit(t[sl]-t[sl].mean(), y[sl], 2); res[sl] = y[sl] - np.polyval(p, t[sl]-t[sl].mean())
        f = np.linspace(0.1, 300, 30000)  # 1/day
        from astropy.timeseries import LombScargle
        P = LombScargle(t, res, normalization="psd").power(f) * 4 / len(t)   # amplitude^2 per bin
        out.append((lc.sector, col, np.std(res), f, P, np.ptp(t)))
np.save(sys.argv[2], np.array([(o[0], o[1], o[2]) for o in out], dtype=object))
for band in [(0.1,0.4),(0.4,1),(1,3),(3,10),(10,30),(30,100),(100,300)]:
    row = []
    for col in ["pdcsap_flux","sap_flux"]:
        A = [np.sqrt(np.median(o[4][(o[3]>=band[0])&(o[3]<band[1])])) for o in out if o[1]==col]
        row.append(f"{1e6*np.median(A):8.1f}")
    print(f"{band[0]:>5}-{band[1]:<5} 1/d  median amplitude (ppm) PDC/SAP:", *row)
print("residual rms (ppm), PDC:", [round(1e6*o[2]) for o in out if o[1]=="pdcsap_flux"])
print("residual rms (ppm), SAP:", [round(1e6*o[2]) for o in out if o[1]=="sap_flux"])
