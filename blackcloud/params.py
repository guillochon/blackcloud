"""Physical constants and fiducial parameters for the toy Black Cloud megacomputer.

Fiducial numbers follow the Laughlin/Adams "Black Cloud" sketch: an AGB host,
a ~220 AU computational structure at 150-200 K holding ~1 Earth mass, whose
nodes talk over millimeter-wave wireless links (their 3 mm carrier).  Everything here is a knob.
"""
from dataclasses import dataclass, field
import numpy as np

# --- constants (SI) ---------------------------------------------------------
c = 2.99792458e8
h = 6.62607015e-34
k_B = 1.380649e-23
sigma_SB = 5.670374419e-8
G = 6.674e-11
L_sun = 3.828e26
M_sun = 1.989e30
M_earth = 5.972e24
AU = 1.495978707e11
pc = 3.0857e16
kpc = 1e3 * pc
day = 86400.0
yr = 3.15576e7
Jy = 1e-26
LN2 = np.log(2.0)


@dataclass
class CloudParams:
    # host star (AGB, Mira-like)
    L_star: float = 2000 * L_sun        # mean bolometric luminosity
    T_star: float = 2800.0              # K, photospheric temperature
    M_star: float = 1.0 * M_sun
    mira_period: float = 400 * day
    mira_amp: float = 0.4               # fractional bolometric semi-amplitude

    # structure
    R_out: float = 220 * AU             # radiating (outer) radius
    shell_frac: float = 0.1             # radial thickness / R_out of compute layer
    M_cloud: float = 1.0 * M_earth
    covering: float = 0.995             # fraction of starlight intercepted
    v_exp: float = 10e3                 # m/s, outflow speed of the shell material

    # internal communications (millimeter wave, the 3 mm carrier of Laughlin et al.)
    nu_comm: float = 100e9              # Hz, band center
    B_comm: float = 10e9                # Hz, total band in use
    node_aperture: float = 20.0         # m, side of a mesh node's phased tile
    hub_aperture: float = 10e3          # m, side of a backbone hub aperture
    tau_proc: float = 1e-3              # s, store-and-forward latency per mesh hop
    tau_hub: float = 3600.0             # s, processing latency at a backbone hub
    eta_comm: float = 0.01              # fraction of power spent on communication
    snr_design: float = 1.0             # per-mode SNR the links run at (low-SNR = efficient)
    link_margin: float = 2.0            # practical overhead above the Shannon limit
    tau_band: float = 30.0              # in-band (comm) optical depth of the shell

    distances_kpc: tuple = (0.5, 1.0, 3.0, 8.0)

    # derived ----------------------------------------------------------------
    @property
    def lam_comm(self):
        return c / self.nu_comm

    @property
    def T_out(self):
        """Radiative-equilibrium temperature of the opaque outer surface."""
        return (self.covering * self.L_star / (4 * np.pi * sigma_SB * self.R_out**2)) ** 0.25

    @property
    def light_radius_time(self):
        return self.R_out / c
