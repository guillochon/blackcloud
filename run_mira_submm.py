"""How the 850 um continuum and the sub-mm communication plateau respond to the Mira cycle,
for a natural dust envelope and for engineered shells under different scheduling choices.
Writes outputs/mira_submm.json and outputs/fig12_mind_scheduling.png.

Light travel: a thin shell of radius r answers a modulation at frequency f with the complex
transfer sinc(2 f r / c) exp(-2 pi i f r / c) (delays uniform on [0, 2r/c], mean r/c).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blackcloud.params import CloudParams, c, h, k_B, AU, day
from blackcloud import thermo, plotstyle as ps

OUT = Path(__file__).parent / 'outputs'
p = CloudParams()
T = p.T_out
f = 1 / p.mira_period
nu = c / 850e-6
dL = p.mira_amp


def semi_amp(alpha, amp=dL):
    """Semi-amplitude of (1 + amp sin)^alpha, as (max - min) / (max + min)."""
    hi, lo = (1 + amp) ** alpha, (1 - amp) ** alpha
    return (hi - lo) / (hi + lo)


def shell_transfer(r):
    x = 2 * f * r / c
    return np.sinc(x) * np.exp(-1j * np.pi * x)


def rj_response(Td, alpha_T):
    """d ln B_nu / d ln L when the emitter's temperature scales as L^alpha_T."""
    x = h * nu / (k_B * Td)
    return alpha_T * x * np.exp(x) / np.expm1(x)


res = {'period_d': p.mira_period / day, 'bolometric_semi_amplitude': dL}

# --- natural envelope: optically thin, n ~ r^-2, kappa ~ nu^beta (beta=1), T_d ~ r^(-2/5) ~ L^(1/5),
# peak matched to the engineered shell; inner radius placed where blackbody-equilibrium scaling
# from the 220 AU shell gives T_in (toy geometry), extending 30x outward.
beta, Y = 1.0, 30.0
Tin = thermo.match_natural_Tin(T, Y=Y, beta=beta)
r_in = p.R_out * (T / Tin) ** ((4 + beta) / 2)
r = r_in * np.logspace(0, np.log10(Y), 600)
Td = Tin * (r / r_in) ** (-2 / (4 + beta))
w = thermo.planck_nu(nu, Td) * np.gradient(r)          # 850 um emission per shell (n r^2 = const)
w /= w.sum()
resp = rj_response(Td, 1 / (4 + beta))
H = np.sum(w * resp * shell_transfer(r)) / np.sum(w * resp)
alpha_nat = float(np.sum(w * resp))
res['natural'] = dict(
    r_in_AU=r_in / AU, r_out_AU=r_in * Y / AU,
    emission_weighted_radius_AU=float(np.sum(w * r) / AU),
    dlnF_dlnL=alpha_nat,
    semi_amplitude_no_light_travel=semi_amp(alpha_nat),
    light_travel_amplitude_factor=float(abs(H)),
    semi_amplitude=semi_amp(alpha_nat) * float(abs(H)),
    lag_d=float(-np.angle(H) / (2 * np.pi * f) / day))

# --- engineered shells: continuum at 850 um (single radius, lag R/c) ---
H_eng = shell_transfer(p.R_out)
alpha_passive = float(rj_response(T, 0.25))
res['passive_continuum'] = dict(dlnF_dlnL=alpha_passive, semi_amplitude=semi_amp(alpha_passive) * abs(H_eng),
                                lag_d=float(-np.angle(H_eng) / (2 * np.pi * f) / day))
res['thermostat_continuum'] = dict(dlnF_dlnL=1.0, semi_amplitude=semi_amp(1.0) * abs(H_eng),
                                   lag_d=res['passive_continuum']['lag_d'])

# --- the continuum beside the communication plateau, at the carrier frequency (deeper in the
# Rayleigh-Jeans regime than 850 um, so a passive shell responds more weakly there) ---
alpha_passive_band = float(h * p.nu_comm / (k_B * T) * 0.25 / -np.expm1(-h * p.nu_comm / (k_B * T)))
res['passive_continuum_band'] = dict(nu_GHz=p.nu_comm / 1e9, dlnF_dlnL=alpha_passive_band,
                                     semi_amplitude=semi_amp(alpha_passive_band) * abs(H_eng))

# --- sub-mm communication plateau: excess brightness = SNR * T per mode, and in the low-SNR regime
# the traffic carried is proportional to SNR * bandwidth, so the in-band excess scales as traffic * T.
# throughput: passive shell L/T ~ L^(3/4); thermostatted shell ~ L.
cases = {
    'passive, traffic follows power': (0.75, 0.25, alpha_passive_band),
    'passive, mind traffic held constant': (0.0, 0.25, alpha_passive_band),
    'thermostat, traffic follows power': (1.0, 0.0, 1.0),
    'thermostat, mind traffic held constant': (0.0, 0.0, 1.0),
}
res['plateau'] = {}
for name, (a_traffic, a_T, a_cont) in cases.items():
    a_plateau = a_traffic + a_T
    res['plateau'][name] = dict(
        plateau_dlnF_dlnL=a_plateau, plateau_semi_amplitude=semi_amp(a_plateau),
        continuum_semi_amplitude=semi_amp(a_cont),
        ratio_dln_dlnL=a_plateau - a_cont, ratio_semi_amplitude=semi_amp(abs(a_plateau - a_cont)),
        ratio_sign='in phase' if a_plateau >= a_cont else 'anti-phase')

json.dump(res, open(OUT / 'mira_submm.json', 'w'), indent=2)

# --- figure: plateau-to-continuum ratio over two Mira cycles for the four scheduling cases.
# Plateau and continuum come from the same shell, so light travel delays both equally and
# drops out of the ratio.
ps.apply()
phase = np.linspace(0, 2, 801)
L = 1 + dL * np.sin(2 * np.pi * phase)
style = {'passive, traffic follows power': ('C1', '-'),
         'passive, mind traffic held constant': ('C1', '--'),
         'thermostat, traffic follows power': ('C0', '-'),
         'thermostat, mind traffic held constant': ('C0', '--')}
fig, ax = plt.subplots(figsize=(ps.W, 3.3))
ax.plot(phase, L, color='orange', ls=':', label=r'starlight $L_\ast/\langle L_\ast\rangle$')
for name, (col, ls) in style.items():
    q = L ** res['plateau'][name]['ratio_dln_dlnL']
    ax.plot(phase, q, color=col, ls=ls, label=name.replace('mind traffic held constant', 'minds held steady'))
ax.set(xlabel='Mira phase [cycles]', ylabel='plateau-to-continuum ratio', xlim=(0, 2), ylim=(0.5, 1.75))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0)
ps.save(fig, OUT / 'fig12_mind_scheduling.png')
print(json.dumps(res, indent=1))
