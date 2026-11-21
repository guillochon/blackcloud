"""Thermodynamics of the cloud: computing budget, SEDs, and how IR responds to the Mira.

Three emitters are compared throughout:
  * 'thermostat' - engineered shell held at fixed T; flux changes by modulating
                   radiating area / emissivity (achromatic variability).
  * 'passive'    - engineered opaque shell of fixed area; T ~ L^(1/4).
  * 'natural'    - optically thin dusty AGB envelope, n ~ r^-2, kappa ~ nu^beta,
                   T_d ~ r^(-2/(4+beta)), matched to the same peak wavelength.
"""
import numpy as np
from scipy.optimize import brentq
from .params import c, h, k_B, sigma_SB, LN2


def planck_nu(nu, T):
    x = np.clip(h * nu / (k_B * T), 1e-12, 700)
    return 2 * h * nu**3 / c**2 / np.expm1(x)


# --- computing budget ----------------------------------------------------------
def landauer_rate(P, T):
    """Irreversible bit operations per second if all power P is dissipated at T."""
    return P / (k_B * T * LN2)


def entropy_bound_rate(L, T_out, T_in):
    """Upper bound from the radiation entropy budget: blackbody light arrives
    carrying 4L/(3 T_in) entropy and leaves carrying 4L/(3 T_out)."""
    return (4.0 / 3.0) * L * (1 / T_out - 1 / T_in) / (k_B * LN2)


# Dulong-Petit limit for carbon: 3R per mole of atoms.  The true heat capacity of graphite at
# ~180 K is lower, so times and temperatures computed with it are upper / lower bounds.
CP_DULONG_PETIT_CARBON = 3 * 8.314462618 / 12.011e-3      # J kg^-1 K^-1


def thermal_time_constant(column_kg_m2, T, cp=CP_DULONG_PETIT_CARBON, emissivity=1.0):
    """Radiative cooling time of a sheet of given areal mass (both faces radiate).
    With the default Dulong-Petit heat capacity this is an upper bound."""
    return column_kg_m2 * cp / (2 * 4 * emissivity * sigma_SB * T**3)


# --- spectral energy distributions (luminosity density L_nu, W/Hz) ------------
def Lnu_shell(nu, T, R):
    return 4 * np.pi * R**2 * np.pi * planck_nu(nu, T)


def _natural_profile(nu, T_in, Y, beta, n_r=400):
    """Unnormalized L_nu of an optically thin r^-2 dust envelope r_in..Y r_in."""
    x = np.logspace(0, np.log10(Y), n_r)               # r / r_in
    T = T_in * x ** (-2.0 / (4 + beta))
    # n r^2 = const  ->  dL_nu ∝ kappa_nu B_nu(T) dr
    B = planck_nu(nu[:, None], T[None, :])
    return nu**beta * np.trapezoid(B, x, axis=1)


def Lnu_natural(nu, L_IR, T_in, Y=30.0, beta=1.0):
    grid = np.logspace(10, 15.5, 3000)
    norm = np.trapezoid(_natural_profile(grid, T_in, Y, beta), grid)
    return L_IR * _natural_profile(nu, T_in, Y, beta) / norm


def peak_wavelength(nu, Lnu):
    """Wavelength of the peak of lambda*F_lambda (= nu*F_nu)."""
    return c / nu[np.argmax(nu * Lnu)]


def match_natural_Tin(T_shell, Y=30.0, beta=1.0):
    """Inner dust temperature that puts the natural envelope's nuLnu peak where
    a T_shell blackbody peaks, so the comparison is about *shape* only."""
    nu = np.logspace(11, 14.5, 3000)
    target = peak_wavelength(nu, planck_nu(nu, T_shell))
    f = lambda Tin: peak_wavelength(nu, _natural_profile(nu, Tin, Y, beta)) - target
    return brentq(f, T_shell * 0.8, 3000.0, xtol=0.1)


def log_width_dex(nu, Lnu):
    """Full width at half maximum of nu*L_nu, in dex of wavelength."""
    y = nu * Lnu
    above = np.where(y >= 0.5 * y.max())[0]
    return np.log10(nu[above.max()] / nu[above.min()])


# --- response of IR bands to stellar pulsation ---------------------------------
def band_response(model, lam, p, T_shell, natural_kw=None, eps=0.02):
    """d ln F_nu / d ln L* at wavelength(s) lam for one of the three emitters."""
    nu = c / np.atleast_1d(lam)
    if model == 'thermostat':
        return np.ones_like(nu)
    if model == 'passive':
        f = lambda s: Lnu_shell(nu, T_shell * s**0.25, p.R_out)
    elif model == 'natural':
        kw = dict(Y=30.0, beta=1.0) | (natural_kw or {})
        Tin = match_natural_Tin(T_shell, **kw)
        f = lambda s: Lnu_natural(nu, s, Tin * s ** (1 / (4 + kw['beta'])), **kw)
    else:
        raise ValueError(model)
    return (np.log(f(1 + eps)) - np.log(f(1 - eps))) / (np.log(1 + eps) - np.log(1 - eps))
