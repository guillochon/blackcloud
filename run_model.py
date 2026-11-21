"""Run the toy Black Cloud megacomputer model and write figures + results.json to outputs/."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from blackcloud.params import CloudParams, c, AU, day, kpc, Jy, L_sun, k_B, LN2, M_earth, yr
from blackcloud import thermo, network as net, observe as obs, plotstyle as ps

OUT = Path(__file__).parent / 'outputs'
OUT.mkdir(exist_ok=True)
p = CloudParams()
T = p.T_out
R = p.R_out
res = {}
ps.apply()

# ============ 1. Energetics ======================================================
L_abs = p.covering * p.L_star
column = p.M_cloud / (4 * np.pi * R**2)
res['energetics'] = {
    'T_out_K': T,
    'L_star_Lsun': p.L_star / L_sun,
    'landauer_bits_per_s': thermo.landauer_rate(L_abs, T),
    'entropy_bound_bits_per_s': thermo.entropy_bound_rate(L_abs, T, p.T_star),
    'kTln2_J': k_B * T * LN2,
    'areal_mass_kg_m2': column,
    'thermal_time_constant_upper_s': thermo.thermal_time_constant(column, T),
    # energy a thermal store would have to absorb to flatten half a Mira cycle (integral of the
    # sinusoid's excess over half a period = dL * L * P / pi), and the temperature swing it implies
    # for the whole mass at the Dulong-Petit heat capacity (a lower bound on the swing)
    'mira_half_cycle_excess_J': p.mira_amp * L_abs * p.mira_period / np.pi,
    'buffer_dT_lower_K': p.mira_amp * L_abs * p.mira_period / np.pi / (p.M_cloud * thermo.CP_DULONG_PETIT_CARBON),
    'light_crossing_radius_hr': R / c / 3600,
    'mira_period_d': p.mira_period / day,
}

# ============ 2. SEDs =============================================================
nu = np.logspace(11, 14.8, 4000)
lam_um = c / nu * 1e6
L_eng = thermo.Lnu_shell(nu, T, R)
L_leak = (1 - p.covering) * p.L_star * np.pi * thermo.planck_nu(nu, p.T_star) / (thermo.sigma_SB * p.T_star**4)
L_IR = 4 * np.pi * R**2 * thermo.sigma_SB * T**4
seds = {'Engineered shell (single T)': L_eng}
for Y in (5, 30, 300):
    Tin = thermo.match_natural_Tin(T, Y=Y)
    seds[f'Natural dust envelope, r_out/r_in={Y} (T_in={Tin:.0f} K)'] = thermo.Lnu_natural(nu, L_IR, Tin, Y=Y)
res['sed_width_dex'] = {k: thermo.log_width_dex(nu, v) for k, v in seds.items()}
# long-wavelength spectral index alpha (F_nu ∝ nu^alpha) between 450 and 850 µm
nu_a, nu_b = c / 450e-6, c / 850e-6
res['submm_index_450_850'] = {k: float(np.log(np.interp(nu_a, nu, v) / np.interp(nu_b, nu, v)) / np.log(nu_a / nu_b))
                              for k, v in seds.items()}

sed_labels = ['engineered shell'] + [rf'natural, $r_{{\rm out}}/r_{{\rm in}}={Y}$' for Y in (5, 30, 300)]
fig, ax = plt.subplots(figsize=(ps.WC, 2.6))
for (name, Lnu), lab, ls in zip(seds.items(), sed_labels, ['-', '--', '-.', ':']):
    ax.loglog(lam_um, nu * Lnu / L_sun, ls, lw=1.8 if ls == '-' else 1.3,
              label=f'{lab} (FWHM {thermo.log_width_dex(nu, Lnu):.2f} dex)')
ax.loglog(lam_um, nu * L_leak / L_sun, color='gray', lw=1, label=f'leaked starlight ({1 - p.covering:.1%} uncovered)')
for b, lab in [(12, 'W3'), (22, 'W4')]:
    ax.axvline(b, color='k', alpha=0.15)
    ax.text(b * 0.96, 1.4, lab, rotation=90, fontsize=ps.SMALL, va='bottom', ha='right', alpha=0.7)
ax.set(xlim=(1, 3000), ylim=(1, 2e5), xlabel='wavelength [µm]', ylabel=r'$\nu L_\nu$ [$L_\odot$]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24))
ps.save(fig, OUT / 'fig1_sed.png')

# ============ 3. Response to Mira pulsation ===================================
lam_grid = np.logspace(0.5, 3, 200) * 1e-6
bands = {'W3 12um': 12e-6, 'W4 22um': 22e-6, 'MIRI 15um': 15e-6, '70um': 70e-6, '850um': 850e-6}
resp = {m: thermo.band_response(m, lam_grid, p, T) for m in ('thermostat', 'passive', 'natural')}
res['band_response_dlnF_dlnL'] = {
    m: {b: float(thermo.band_response(m, l, p, T)[0]) for b, l in bands.items()} for m in resp}

phase = np.linspace(0, 1, 200)
Lrel = 1 + p.mira_amp * np.sin(2 * np.pi * phase)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(ps.W, 2.8))
for m, col in zip(resp, ['C0', 'C1', 'C3']):
    a1.semilogx(lam_grid * 1e6, resp[m], color=col, label=m)
    # exact light curves in W3/W4 (power-law response is fine for a toy)
    r3, r4 = res['band_response_dlnF_dlnL'][m]['W3 12um'], res['band_response_dlnF_dlnL'][m]['W4 22um']
    W4 = -2.5 * np.log10(Lrel ** r4)
    col34 = -2.5 * np.log10(Lrel ** r3) - W4
    a2.plot(col34, W4, color=col, label=m)
a1.set(xlabel='wavelength [µm]', ylabel=r'$d\ln F_\nu / d\ln L_*$')
a1.legend()
a2.invert_yaxis()
a2.set(xlabel='Δ(W3 − W4) [mag]', ylabel='ΔW4 [mag]')
fig.tight_layout(w_pad=1.5); ps.save(fig, OUT / 'fig2_mira_response.png')

# ============ 4. Link economics ======================================================
D_mesh = np.pi * R             # antipodal distance along the shell
d_hop = np.logspace(1, 9, 400)
E, t = net.multihop(D_mesh, d_hop, p)
d0 = net.fresnel_hop(p)
E0, t0 = net.multihop(D_mesh, d0, p)
E_bb = net.backbone_energy_per_bit(2 * R, p)
t_bb = 2 * R / c + 2 * p.tau_hub
P_comm = p.eta_comm * L_abs
res['network'] = {
    'fresnel_hop_m': d0,
    'mesh_speed_over_c': net.mesh_speed(p) / c,
    'mesh_hops_antipode': D_mesh / d0,
    'mesh_energy_per_bit_antipode_J': float(E0),
    'mesh_latency_antipode_d': float(t0 / day),
    'backbone_energy_per_bit_J': float(E_bb),
    'backbone_latency_antipode_d': t_bb / day,
    'global_bitrate_mesh_bits_s': P_comm / float(E0),
    'global_bitrate_backbone_bits_s': P_comm / float(E_bb),
    'local_compute_bits_s': thermo.landauer_rate((1 - p.eta_comm) * L_abs, T),
}
res['network']['global_to_local_ratio_backbone'] = (res['network']['global_bitrate_backbone_bits_s']
                                                    / res['network']['local_compute_bits_s'])

fig, ax = plt.subplots(figsize=(ps.WC, 2.7))
sc = ax.scatter(t / day, E / (k_B * T * LN2), c=np.log10(d_hop), s=6, cmap='viridis')
ax.plot(t0 / day, E0 / (k_B * T * LN2), 'r*', ms=12, label=f'mesh, Fresnel hop A/λ = {d0/1e3:.0f} km')
ax.plot(t_bb / day, E_bb / (k_B * T * LN2), 'ks', ms=8, label=f'backbone, {p.hub_aperture/1e3:.0f} km hubs')
ax.axvline(D_mesh / c / day, color='gray', ls=':', label='light time along shell (πR/c)')
ax.axvline(2 * R / c / day, color='gray', ls='--', label='light time across (2R/c)')
ax.set(xscale='log', yscale='log', xlabel='latency to antipode [days]', ylabel=r'energy per bit [$kT\ln 2$]')
plt.colorbar(sc, label=r'$\log_{10}$ hop length [m]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24))
ps.save(fig, OUT / 'fig3_link_economics.png')

# ============ 5. Signal propagation on the shell =====================================
src = 0
arch = {}
for name, nh in [('mesh', 0), ('backbone', 24)]:
    x, W, hubs = net.build_graph(p, n_cells=6000, n_hubs=nh, seed=3)
    ta = net.arrival_times(W, src)
    arch[name] = (x, ta, hubs)
    res['network'][f'{name}_max_arrival_d'] = float(ta.max() / day)
    res['network'][f'{name}_median_arrival_d'] = float(np.median(ta) / day)
    if nh:
        res['network']['backbone_covering_radius_deg'] = net.covering_radius_deg(x, hubs)
# for comparison: the same 24 hubs scattered at random leave wide gaps between them
x_r, W_r, hubs_r = net.build_graph(p, n_cells=6000, n_hubs=24, seed=3, hub_layout='random')
ta_r = net.arrival_times(W_r, src)
res['network']['backbone_random_max_arrival_d'] = float(ta_r.max() / day)
res['network']['backbone_random_median_arrival_d'] = float(np.median(ta_r) / day)
res['network']['backbone_random_covering_radius_deg'] = net.covering_radius_deg(x_r, hubs_r)

t_max = max(ta.max() for _, ta, _ in arch.values()) / day     # one color scale for both maps
# Mollweide (equal-area) maps of the whole shell: longitude and latitude of each cell's direction from
# the star, about the model's arbitrary polar (z) axis; graticule every 60 deg in longitude, 30 deg in latitude
fig, axs = plt.subplots(1, 2, figsize=(ps.W, 2.2), subplot_kw=dict(projection='mollweide'))
for ax, (name, (x, ta, hubs)), title in zip(axs, arch.items(), ['mesh only', 'mesh + 24 backbone hubs (snub cube)']):
    lon = np.arctan2(x[:, 1], x[:, 0]); lat = np.arcsin(x[:, 2])
    s = ax.scatter(lon, lat, c=ta / day, s=1.6, lw=0, cmap='magma', vmin=0, vmax=t_max)
    if len(hubs):
        ax.scatter(lon[hubs], lat[hubs], marker='^', s=18, facecolor='none', edgecolor='cyan', lw=1.0)
    ax.scatter(lon[src], lat[src], marker='*', s=90, color='lime', edgecolor='k', lw=0.5)
    ax.set_longitude_grid(60); ax.set_latitude_grid(30)
    ax.grid(alpha=0.3); ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title(title, fontsize=ps.SMALL, pad=4)
res['network']['source_lon_lat_deg'] = [float(np.degrees(np.arctan2(x[src, 1], x[src, 0]))),
                                        float(np.degrees(np.arcsin(x[src, 2])))]
fig.colorbar(s, ax=axs, orientation='horizontal', fraction=0.07, pad=0.06, aspect=50,
             label='arrival time of a broadcast [days]')
ps.save(fig, OUT / 'fig4_propagation.png')

# ============ 6. What an observer sees of one broadcast wave =========================
t_grid = np.arange(-5, 25, 0.02) * day
fig, axs = plt.subplots(1, 2, figsize=(ps.W, 2.9), sharey=True)
res['broadcast_observed_fwhm_d'] = {}
for ax, (name, (x, ta, hubs)) in zip(axs, arch.items()):
    s_dir = x[src]
    for ang in (0, 60, 120, 180):
        a = np.radians(ang)
        perp = np.cross(s_dir, [0, 0, 1.0]); perp /= np.linalg.norm(perp)
        n_hat = np.cos(a) * s_dir + np.sin(a) * perp
        lc = obs.observed_burst(x, ta, n_hat, R, t_grid, width=0.25 * day)
        ax.plot(t_grid / day, lc / lc.max(), label=f'{ang}°')
        above = t_grid[lc >= 0.5 * lc.max()]
        res['broadcast_observed_fwhm_d'][f'{name}_{ang}deg'] = float((above.max() - above.min()) / day)
    ax.set(title=name, xlabel='observed time [days]', ylim=(0, 1.3), xlim=(-5, 25))
axs[0].set_ylabel('normalized in-band flux')
axs[1].legend(title='from source', loc='upper right', title_fontsize=ps.SMALL, columnspacing=1.0)
fig.tight_layout(w_pad=0.8); ps.save(fig, OUT / 'fig5_broadcast_lightcurves.png')

# ============ 7. Light-travel smearing (fig6 is drawn in run_cells.py, beside the supersteps) ===
res['time_domain'] = {'2R_over_c_d': 2 * R / c / day,
                      'amplitude_retained_at_1d': float(obs.shell_transfer(1 / day, R)),
                      'amplitude_retained_at_mira': float(obs.shell_transfer(1 / p.mira_period, R))}

# ============ 8. Comm-band spectrum ===================================================
nu_s = np.linspace(p.nu_comm * 5 / 6, p.nu_comm * 7 / 6, 20000)
Tc, Tt = obs.comm_band_spectrum(nu_s, p)
edge = p.nu_comm + p.B_comm / 2
nu_z = np.linspace(edge - 4 * edge * p.v_exp / c, edge + 4 * edge * p.v_exp / c, 4000)   # +-4 smear widths
Tcz, Ttz = obs.comm_band_spectrum(nu_z, p)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(ps.W, 2.8), sharey=True)
a1.plot(nu_s / 1e9, Tc, 'k--', label='thermal only')
a1.plot(nu_s / 1e9, Tt, 'C0', label=f'with traffic (SNR = {p.snr_design:g})')
a1.set(xlabel='frequency [GHz]', ylabel='brightness temperature [K]', ylim=(0, 1.35 * Tt.max()))
a1.legend(loc='upper left')
a2.plot((nu_z - edge) / 1e6, Ttz, 'C0')
a2.axvline(edge * p.v_exp / c / 1e6, color='gray', ls=':', label=r'$\nu\,v_{\rm exp}/c$')
a2.set(xlabel='offset from band edge [MHz]')
a2.legend(loc='upper right')
fig.tight_layout(w_pad=0.8); ps.save(fig, OUT / 'fig7_comm_band.png')

P_leak = obs.comm_leak_power(p)
res['comm_leak'] = {'P_leak_W': P_leak, 'P_leak_over_L': P_leak / L_abs,
                    'P_leak_over_P_comm': P_leak / P_comm,
                    'band_edge_smear_MHz': edge * p.v_exp / c / 1e6,
                    'excess_Tb_K': p.snr_design * T}

# ============ 9. Detectability table ===================================================
det = {}
for dk in p.distances_kpc:
    d = dk * kpc
    row = {'angular_diameter_arcsec': 2 * R / d * 206265}
    for lab, lam in [('F_12um_Jy', 12e-6), ('F_22um_Jy', 22e-6), ('F_850um_Jy', 850e-6)]:
        row[lab] = float(obs.flux_density(thermo.Lnu_shell(c / lam, T, R), d) / Jy)
    row['comm_excess_Jy'] = float(obs.flux_density(P_leak / p.B_comm, d) / Jy)
    row['comm_excess_over_continuum'] = row['comm_excess_Jy'] / float(
        obs.flux_density(thermo.Lnu_shell(p.nu_comm, T, R), d) / Jy)
    det[f'{dk} kpc'] = row
res['detectability'] = det
res['sensitivity_reference_Jy'] = obs.SENSITIVITY_JY

json.dump(res, open(OUT / 'results.json', 'w'), indent=2, default=float)


def show(d, ind=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(' ' * ind + k); show(v, ind + 2)
        else:
            print(' ' * ind + f'{k:45s} {v:.4g}' if isinstance(v, (float, int)) else f'{k}: {v}')


show(res)
