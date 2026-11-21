"""Collect the population search into util/tess/pop/population_summary.json.
usage: python p9_summary.py SCRATCH_DIR"""
import sys, os, json, numpy as np, pandas as pd
S = sys.argv[1]
d = pd.read_parquet(f"{S}/vetted.parquet"); c = pd.read_csv(f"{S}/candidates.csv"); x = pd.read_parquet(f"{S}/inject.parquet")
b = pd.read_csv(f"{S}/blend.csv"); z = pd.read_csv(f"{S}/ztf_veto.csv") if os.path.exists(f"{S}/ztf_veto.csv") else pd.DataFrame()
N = len(d); sig = c[c.fap_emp_pop < 0.01]; ok = sig[sig.pass_cm & sig.pass_shape & sig.pass_repeat]

def a50(g):
    m = g.groupby("amp_ppm").det.mean(); a, v = np.log10(m.index.values), m.values
    if v[-1] < 0.5: return np.nan
    if v[0] >= 0.5: return 10**a[0]
    k = np.argmax(v >= 0.5); return 10**np.interp(0.5, [v[k-1], v[k]], [a[k-1], a[k]])
x["Pbin"] = pd.cut(x.P, [3, 4, 6, 8, 10, 15, 30]).astype(str)
r50 = x.groupby(["tic", "seg"])[["amp_ppm", "det"]].apply(a50)
out = {
    "sample": {"vsx_type_M": 81453, "tic_tmag_lt_14": int(pd.read_csv(f"{S}/targets.csv").shape[0]),
               "stars_with_tess_lc": int(len(os.listdir(f"{S}/lc"))), "segments_searched": N, "stars_searched": int(d.tic.nunique()),
               "segment_length_d_quantiles": d["T"].quantile([.1, .5, .9]).round(1).tolist(),
               "residual_rms_ppm_quantiles": d.rms_ppm.quantile([.1, .5, .9]).round(0).tolist()},
    "null_calibration": {f"fap_lt_{q:g}": {"observed": int((d.fap_emp < q).sum()), "expected": round(N*q, 1)}
                         for q in [0.5, 0.1, 0.01, 1e-3, 1e-4]},
    "population_significant": {"segments": int(len(sig)), "stars": int(sig.tic.nunique()),
                               "fail_cluster": int((~sig.pass_cm).sum()), "fail_shape": int((~sig.pass_shape).sum()),
                               "fail_repeat": int((~sig.pass_repeat).sum()),
                               "known_periodic_neighbor": int((b.vsx_match | b.gaia_match.notna()).sum())},
    "survivors": [{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in r.items()}
                  for r in ok[["Name", "tic", "tmag", "sectors", "T", "P_cal_d", "fap_emp_pop", "amp_peak_ppm", "rms_ppm"]].to_dict("records")],
    "ztf": z.to_dict("records"),
    "injection": {"n_segments": int(x.groupby(["tic", "seg"]).ngroups), "realizations_per_amp": 4,
                  "detected_fraction_by_amp_ppm": x.groupby("amp_ppm").det.mean().round(3).to_dict(),
                  "detected_fraction_by_period": x.pivot_table(index="Pbin", columns="amp_ppm", values="det").round(2).to_dict("index"),
                  "per_segment_50pct_rms_ppm_quantiles": {str(q): round(float(r50.quantile(q)), -1) for q in [.1, .25, .5, .75]},
                  "fraction_not_reaching_50pct_at_3pct": round(float(r50.isna().mean()), 2)},
}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "population_summary.json"), "w"), indent=1, default=str)
print(json.dumps(out["population_significant"]), json.dumps(out["injection"]["per_segment_50pct_rms_ppm_quantiles"]))
