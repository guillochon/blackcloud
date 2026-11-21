"""Pixel-level localization of a TESS line (p5 survivors): where on the sky does the power at f come from?

Per star and sector, an 11x11 TESScut cutout (hard quality mask, per-frame background from the faintest 40% of
pixels) is binned like p4 (0.25 d, per orbit). Each pixel's flux is fit with the p4 cycle model (15-d spline +
per-orbit offsets) plus a sinusoid at the TESS line frequency f. The in-phase amplitude image, phased on the pixels
around the target, is the line's own image on the detector. A 2-D Gaussian (width fit to the mean image at the
target) gives its centroid. The centroid error comes from refitting with noise images: the same projection at
40 frequencies away from f.

Diagnostics:
  aperture    in-phase amplitude summed over r < 1.5 px of the target, as a fraction of that aperture's flux and in
              sigma; it should reproduce the SAP line if the line is on the target
  net/gross   sum(A)/sum(|A|) over r < 2 px: ~1 for a brightness change at the target, ~0 for the +/- dipole a
              shifting image leaves (pointing or focus at f)
  peak        the most significant pixel of the phase-free amplitude in the cutout, and the Gaussian centroid fit in the 5x5
              box around it; Gaia DR3 sources (G < 19) within 1.5 px of that centroid, with the fractional amplitude
              each would need to make the line alone
usage: python p10_pixels.py SCRATCH_DIR [TIC ...]  ->  SCRATCH_DIR/pix/*.png, util/tess/pop/pixel_localization.json"""
import sys, os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lightkurve as lk, astropy.units as u, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from scipy.optimize import least_squares
from cyc import design
S = sys.argv[1]; os.makedirs(f"{S}/pix", exist_ok=True)
TICS = [int(x) for x in sys.argv[2:]] or [11032792, 156296547, 417728675]
c = pd.read_csv(f"{S}/candidates.csv"); c = c[c.tic.isin(TICS)].sort_values("fap_emp").drop_duplicates("tic")
NPIX, PX = 11, 21.0                                                     # cutout size, arcsec per pixel

def binned(tpf, f):
    t = tpf.time.value; F = tpf.flux.value.reshape(len(t), -1).astype(float)
    ok = np.isfinite(t) & np.all(np.isfinite(F), axis=1); t, F = t[ok], F[ok]
    bkg = np.array([np.median(np.sort(row)[:int(0.4*row.size)]) for row in F]); F -= bkg[:, None]
    gap = t[np.argmax(np.diff(t))]; orbit = (t > gap).astype(int)
    b = np.floor((t - t.min())/0.25) + 1000*orbit
    df = pd.DataFrame(F); df["t"], df["b"], df["o"] = t, b, orbit
    g = df.groupby("b"); n = g.size(); m = g.mean()[n >= 0.5*n.max()]
    return m.t.values, m.o.values.astype(int), m.drop(columns=["t", "o"]).values, tpf

def fit_pixels(t, o, F, f):
    X = np.hstack([design(t, t.min(), t.max(), 15.0), (o[:, None] == 1).astype(float),
                   np.cos(2*np.pi*f*t)[:, None], np.sin(2*np.pi*f*t)[:, None]])
    beta, *_ = np.linalg.lstsq(X, F, rcond=None)
    return beta[-2], beta[-1]                                          # cos, sin amplitude per pixel (e-/s)

def gauss(p, yy, xx):
    a, x0, y0, s, c0 = p; return a*np.exp(-0.5*((xx - x0)**2 + (yy - y0)**2)/s**2) + c0

def centroid(img, x0, y0, s, box=2):
    yy, xx = np.mgrid[:NPIX, :NPIX]
    m = ((np.abs(xx - round(x0)) <= box) & (np.abs(yy - round(y0)) <= box)).ravel()
    r = least_squares(lambda p: (gauss([p[0], p[1], p[2], s, p[3]], yy, xx) - img).ravel()[m],
                      [img.flat[int(round(y0))*NPIX + int(round(x0))], x0, y0, 0.0],
                      bounds=([-np.inf, x0 - box, y0 - box, -np.inf], [np.inf, x0 + box, y0 + box, np.inf]))
    return r.x

out = []
Vizier.ROW_LIMIT = -1
for _, r in c.iterrows():
    pos = SkyCoord(r.ra*u.deg, r.dec*u.deg); f = r.f_cal
    gaia = Vizier(row_limit=-1, columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP"], column_filters={"Gmag": "<19"}).query_region(
        pos, radius=NPIX*PX/np.sqrt(2)*u.arcsec, catalog="I/355/gaiadr3")[0].to_pandas()
    for sec in map(int, r.sectors.split(",")):
        tpf = lk.search_tesscut(pos, sector=sec).download(cutout_size=NPIX, download_dir=f"{S}/tesscut", quality_bitmask="hard")
        t, o, F, tpf = binned(tpf, f)
        tx, ty = tpf.wcs.world_to_pixel(pos)
        a_c, a_s = fit_pixels(t, o, F, f)
        k = (np.hypot(*np.meshgrid(np.arange(NPIX) - tx, np.arange(NPIX) - ty)) < 1.5).ravel()
        phi = np.arctan2(a_s[k].sum(), a_c[k].sum())                    # phase of the line on the target pixels
        A = (a_c*np.cos(phi) + a_s*np.sin(phi)).reshape(NPIX, NPIX)
        mean = F.mean(0).reshape(NPIX, NPIX); yy, xx = np.mgrid[:NPIX, :NPIX]; rr = np.hypot(xx - tx, yy - ty)
        s_psf = least_squares(lambda p: (gauss([p[0], tx, ty, p[1], p[2]], yy, xx) - mean)[rr < 1.5],
                              [mean.max(), 1.0, 0.0], bounds=([0, 0.4, -np.inf], [np.inf, 3, np.inf])).x[1]
        T = np.ptp(t); offs = [fo for fo in np.linspace(0.12, 0.5, 200) if abs(fo - f) > 3/T and abs(fo - 2*f) > 2/T][::5][:40]
        noise = []
        for fo in offs:
            nc, ns = fit_pixels(t, o, F, fo); ph = np.random.default_rng(int(fo*1e4)).uniform(0, 2*np.pi)
            noise.append((nc*np.cos(ph) + ns*np.sin(ph)).reshape(NPIX, NPIX))
        noise = np.array(noise); sig = noise.std(0); Z = A/sig
        ap = rr < 1.5; ap_amp = A[ap].sum(); ap_sig = noise[:, ap].sum(1).std()
        near2 = rr < 2; net_gross = A[near2].sum()/np.abs(A[near2]).sum()
        Zt = np.hypot(a_c, a_s).reshape(NPIX, NPIX)/sig                # phase-free: a neighbor's line need not share the target phase
        py, px = np.unravel_index(np.argmax(Zt), Zt.shape)
        ph = np.arctan2(a_s[py*NPIX + px], a_c[py*NPIX + px])
        Ap = (a_c*np.cos(ph) + a_s*np.sin(ph)).reshape(NPIX, NPIX)      # phased on the peak pixel for the centroid
        cA = centroid(Ap, px, py, s_psf)
        mc = np.array([centroid(gauss([cA[0], cA[1], cA[2], s_psf, cA[3]], yy, xx) + nz, cA[1], cA[2], s_psf)[1:3] for nz in noise])
        sx, sy = mc.std(0); dx, dy = cA[1] - tx, cA[2] - ty
        gx, gy = tpf.wcs.world_to_pixel(SkyCoord(gaia.RA_ICRS.values*u.deg, gaia.DE_ICRS.values*u.deg))
        gaia["dist_line_px"] = np.hypot(gx - cA[1], gy - cA[2]); gaia["dist_target_px"] = np.hypot(gx - tx, gy - ty)
        line_flux = abs(cA[0])*2*np.pi*s_psf**2                          # e-/s, integrated over the Gaussian
        star_flux = 15000*10**(-0.4*((gaia.Gmag - 0.43*gaia["BP-RP"].fillna(1.5)) - 10))   # rough TESS e-/s
        gaia["frac_amp_needed"] = line_flux/star_flux
        near = gaia[gaia.dist_line_px < 1.5].sort_values("dist_line_px")
        res = dict(Name=r.Name, tic=int(r.tic), sector=sec, f=f, P_d=1/f, sap_amp_ppm=float(r.amp_peak_ppm),
                   aperture_amp_ppm=float(ap_amp/mean[ap].sum()*1e6), aperture_amp_nsigma=float(ap_amp/ap_sig),
                   net_over_gross=float(net_gross), peak_pixel_nsigma=float(Zt[py, px]), peak_pixel_dist_px=float(np.hypot(px - tx, py - ty)),
                   psf_sigma_px=float(s_psf), centroid_offset_px=[float(dx), float(dy)], centroid_sigma_px=[float(sx), float(sy)],
                   centroid_offset_arcsec=float(np.hypot(dx, dy)*PX), centroid_offset_nsigma=float(np.hypot(dx/sx, dy/sy)),
                   gaia_near_line=near[["Source", "Gmag", "BP-RP", "dist_line_px", "dist_target_px", "frac_amp_needed"]].round(3).to_dict("records"))
        out.append(res); print(json.dumps(res, default=str), flush=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
        for a_, img, ttl in [(ax[0], np.log10(np.clip(mean, 1, None)), "mean image, log10 e-/s"), (ax[1], Z, f"in-phase amplitude / pixel noise at {1/f:.2f} d")]:
            im = a_.imshow(img, origin="lower", cmap="viridis" if a_ is ax[0] else "RdBu_r",
                           **({} if a_ is ax[0] else dict(vmin=-np.abs(img).max(), vmax=np.abs(img).max()))); plt.colorbar(im, ax=a_, fraction=0.046)
            a_.scatter(gx, gy, s=np.clip(3*(19 - gaia.Gmag)**1.5, 2, 200), facecolors="none", edgecolors="w" if a_ is ax[0] else "k", lw=0.6)
            a_.plot(tx, ty, "r+", ms=14, mew=2, label="target"); a_.set_xlim(-0.5, NPIX - 0.5); a_.set_ylim(-0.5, NPIX - 0.5); a_.set_title(ttl, fontsize=9)
        ax[1].errorbar(cA[1], cA[2], xerr=sx, yerr=sy, fmt="x", color="orange", ms=10, mew=2, label="line centroid")
        ax[1].legend(fontsize=8, loc="upper right"); fig.suptitle(f"{r.Name}, TIC {r.tic}, sector {sec}", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{S}/pix/{r.tic}_s{sec}.png", dpi=110); plt.close(fig)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pixel_localization.json"), "w"), indent=1, default=str)
