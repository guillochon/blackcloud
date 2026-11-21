"""What a distant, unresolved observer receives.

Key geometric fact (Archimedes' hat-box theorem): for a uniform thin shell,
the extra light-travel delay  -R (x . n)/c  is *uniformly* distributed on
[-R/c, +R/c].  Any process that is simultaneous in the cloud's frame is
therefore smeared by a boxcar of width 2R/c, i.e. its Fourier amplitude is
multiplied by |sinc(2 f R / c)|.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .params import c, k_B, h, LN2
from .thermo import planck_nu


# --- time domain ------------------------------------------------------------------
def observed_burst(x, t_arrival, n_hat, R, t_grid, width):
    """Unresolved light curve of a relay wave: each cell emits a Gaussian burst of
    traffic leakage centered on its arrival time; observer along unit vector n_hat."""
    t_obs = t_arrival - R * (x @ n_hat) / c
    dt = t_grid[1] - t_grid[0]
    hist, _ = np.histogram(t_obs, bins=np.append(t_grid, t_grid[-1] + dt))
    lc = gaussian_filter1d(hist.astype(float), width / dt, mode='constant')
    return lc / len(x)


def shell_transfer(f, R):
    """Amplitude transfer of a frame-simultaneous modulation at frequency f."""
    return np.abs(np.sinc(2 * f * R / c))


def modulation_amplitude(x, n_hat, R, f):
    """Observed fractional amplitude of a frame-simultaneous sinusoidal modulation at
    frequencies f, summed over shell elements at unit positions x (Monte Carlo shell_transfer)."""
    delay = -R * (x @ n_hat) / c
    return np.abs(np.exp(2j * np.pi * np.outer(f, delay)).mean(axis=1))


# --- sub-mm communication band ------------------------------------------------------
def doppler_kernel(v_exp, optically_thick, n=401):
    """Line-of-sight velocity distribution of the radiating surface of an expanding shell.
    Thick: only the approaching hemisphere is seen, weight ∝ mu -> p(v) ∝ v on [0, v_exp].
    Thin: whole shell, flat on [-v_exp, v_exp]."""
    if optically_thick:
        v = np.linspace(0, v_exp, n)
        w = v.copy()
    else:
        v = np.linspace(-v_exp, v_exp, n)
        w = np.ones_like(v)
    return v, w / w.sum()


def comm_band_spectrum(nu, p):
    """Brightness temperature seen from outside: the thermal surface at T_out, plus the
    traffic leaking from the outermost tau~1 layer (excess snr*T_out inside the band),
    with band edges Doppler-smeared by the outflow (approaching side = blueshift)."""
    v, w = doppler_kernel(p.v_exp, p.tau_band > 1)
    lo, hi = p.nu_comm - p.B_comm / 2, p.nu_comm + p.B_comm / 2
    excess = np.zeros_like(nu)
    for vi, wi in zip(v, w):
        s = 1 + vi / c  # approaching -> blueshift
        excess += wi * ((nu >= lo * s) & (nu <= hi * s))
    T_cont = h * nu / k_B / np.expm1(h * nu / (k_B * p.T_out))  # Planck brightness temp
    return T_cont, T_cont + p.snr_design * p.T_out * excess


def comm_leak_power(p):
    """Power escaping in the comm band (Rayleigh-Jeans excess over the outer surface)."""
    return 4 * np.pi * p.R_out**2 * np.pi * 2 * k_B * p.snr_design * p.T_out * p.nu_comm**2 / c**2 * p.B_comm


def flux_density(L_nu, d):
    return L_nu / (4 * np.pi * d**2)


# Rough point-source sensitivities (order of magnitude only; Jy).
SENSITIVITY_JY = {
    'WISE W3 12um (5sig, AllWISE)': 0.7e-3,
    'WISE W4 22um (5sig, AllWISE)': 5e-3,
    'JWST MIRI F2100W (10sig, 10ks)': 2e-5,
    'ALMA Band 7 continuum (1 hr rms)': 5e-5,
}
