"""How signals move through the cloud.

Link physics (per bit):
  * Receiver noise floor is the cloud itself: T_sys ~ T_out in the comm band.
    Shannon's low-SNR limit gives E_bit >= k T_sys ln2 at the receiver.
  * Friis between apertures A_t, A_r at distance d:  P_r/P_t = A_t A_r / (lambda d)^2,
    capped at 1 inside the near field (d < A/lambda).
  * Every relay hop that decodes and re-encodes pays ~k T ln2 per bit as well.

Two architectures are simulated on a coarse-grained shell:
  * 'mesh'     - nearest-neighbor relaying along the shell surface.
  * 'backbone' - mesh plus a few large hubs that beam straight across the cavity. The 24 hubs sit
                 at the vertices of a snub cube, the evenly spread arrangement of 24 points on a
                 sphere, so that no cell is far from a hub; hub_layout='random' scatters them instead.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from .params import c, k_B, LN2


# --- analytic link budget --------------------------------------------------------
def path_loss(d, A_t, A_r, lam):
    return np.maximum(1.0, (lam * d) ** 2 / (A_t * A_r))


def energy_per_bit_hop(d, p, A=None):
    A = p.node_aperture**2 if A is None else A
    kTln2 = k_B * p.T_out * LN2
    return kTln2 * (p.link_margin * path_loss(d, A, A, p.lam_comm) + 1.0)


def multihop(D, d, p, A=None):
    """Energy/bit and latency to move a bit a distance D in hops of length d."""
    n = np.ceil(D / d)
    return n * energy_per_bit_hop(d, p, A), n * (d / c + p.tau_proc)


def fresnel_hop(p, A=None):
    """Hop length that minimizes energy per bit-distance: the near-field edge A/lambda."""
    A = p.node_aperture**2 if A is None else A
    return A / p.lam_comm


def mesh_speed(p):
    d = fresnel_hop(p)
    return d / (d / c + p.tau_proc)


def backbone_energy_per_bit(D, p):
    A = p.hub_aperture**2
    return energy_per_bit_hop(D, p, A)


# --- coarse-grained shell graph ---------------------------------------------------
def fibonacci_sphere(n, rng, jitter=0.3):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    xyz = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    xyz += jitter * np.sqrt(4 * np.pi / n) * rng.standard_normal(xyz.shape)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def snub_cube():
    """The 24 unit vertices of a snub cube: even permutations of (1, 1/t, t) with an even number of
    minus signs and odd permutations with an odd number, t the tribonacci constant. They are the best
    known separation of 24 points on a sphere (Tammes problem; Robinson 1961)."""
    t = 1.839286755214161
    base = np.array([1, 1 / t, t])
    pts = []
    for perms, parity in (([(0, 1, 2), (1, 2, 0), (2, 0, 1)], 1), ([(0, 2, 1), (2, 1, 0), (1, 0, 2)], -1)):
        for perm in perms:
            for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
                if np.prod(s) == parity:
                    pts.append(s * base[list(perm)])
    pts = np.array(pts)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def covering_radius_deg(x, hubs):
    """Largest angle from any cell to its nearest hub."""
    return float(np.degrees(np.arccos(np.clip((x @ x[hubs].T).max(1), -1, 1))).max())


def build_graph(p, n_cells=6000, k=8, n_hubs=0, seed=1, hub_layout='snub'):
    """Return (unit positions, sparse travel-time graph [s], hub indices).

    hub_layout 'snub' puts the 24 hubs at the cells nearest the snub-cube vertices; 'random' draws
    n_hubs cells at random."""
    rng = np.random.default_rng(seed)
    x = fibonacci_sphere(n_cells, rng)
    R, v = p.R_out, mesh_speed(p)
    dist, idx = cKDTree(x).query(x, k + 1)
    rows = np.repeat(np.arange(n_cells), k)
    cols = idx[:, 1:].ravel()
    arc = 2 * np.arcsin(np.clip(dist[:, 1:].ravel() / 2, 0, 1)) * R
    w = [arc / v]
    r_, c_ = [rows], [cols]

    hubs = np.array([], dtype=int)
    if n_hubs:
        if hub_layout == 'snub':
            assert n_hubs == 24, 'the snub-cube layout has 24 hubs'
            hubs = cKDTree(x).query(snub_cube())[1]
        else:
            hubs = rng.choice(n_cells, n_hubs, replace=False)
        hi, hj = np.meshgrid(hubs, hubs)
        m = hi != hj
        chord = np.linalg.norm(x[hi[m]] - x[hj[m]], axis=1) * R
        r_.append(hi[m]); c_.append(hj[m]); w.append(chord / c + p.tau_hub)

    W = coo_matrix((np.concatenate(w), (np.concatenate(r_), np.concatenate(c_))),
                   shape=(n_cells, n_cells)).tocsr()
    W = W.maximum(W.T)  # undirected
    return x, W, hubs


def arrival_times(W, source):
    return dijkstra(W, directed=False, indices=source)
