"""Computing cells: the scale at which a Black Cloud stops being one machine.

Design assumptions (all optional knobs):
  * Nodes sit on a 3D lattice inside a shell of thickness dR and intercept starlight with
    total radial covering tau.  A convex node of surface S has mean projected area S/4
    (Cauchy), so covering fixes the node surface density:  n S = 4 tau / dR.
  * A node has one flat aperture per nearest neighbor (z faces of area A each, S = z A),
    and each link is Fresnel-matched (d = A / lambda): one spatial mode, no path loss.
  * A link carries b = f_B * nu bits/s (fractional bandwidth f_B, ~1 bit per mode per Hz).

Consequences derived here:
  * node spacing   d = kappa^(3/2) sqrt(z lambda dR / (4 tau))
  * bisection bandwidth per unit area   sigma_b = tau f_B c / (dR lambda^2)   (lattice-free)
  * bandwidth-balanced (Rent) cell size, where a region's boundary bandwidth equals
    beta times the bit operations inside it:
        l_cell = 2 tau f_B c kT ln2 / (beta F lambda^2),   F = absorbed flux per area
    which depends on neither the lattice nor the shell thickness.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import Voronoi, ConvexHull
from .params import c, h, k_B, LN2

# --- lattices --------------------------------------------------------------------------
# primitive vectors as columns; neighbor steps in primitive (integer) coordinates
_s3 = np.sqrt(3) / 2
LATTICES = {
    'sc': dict(label='simple cubic', shape='cube', inspiration='3D torus (Blue Gene/L, Cray T3E)',
               P=np.eye(3),
               nbrs=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    'hex': dict(label='simple hexagonal', shape='hexagonal prism',
                inspiration='stacked 2D meshes (wafer-scale, on-chip NoC layers)',
                P=np.array([[1, 0.5, 0], [0, _s3, 0], [0, 0, 1.0]]),
                nbrs=[(1, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 0, 1)]),
    'bcc': dict(label='body-centered cubic', shape='truncated octahedron',
                inspiration='8-port mesh with diagonal links',
                P=0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1.0]]),
                nbrs=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]),
    'fcc': dict(label='face-centered cubic', shape='rhombic dodecahedron',
                inspiration='high-radix mesh (Tofu 6D mesh/torus, 10 ports)',
                P=0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]]),
                nbrs=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (0, 1, -1), (1, 0, -1)]),
}


def _steps(L):
    s = np.array(L['nbrs'])
    return np.vstack([s, -s])


def lattice_stats(name, K=9, n_dirs=2000, seed=0):
    """Coordination z, kappa = d_nn n^(1/3), routing stretch (graph hops x d_nn over Euclidean
    distance), and bond-crossing density of a plane in units of n^(2/3)."""
    L = LATTICES[name]
    P, steps = L['P'], _steps(L)
    vecs = steps @ P.T
    d_nn = np.linalg.norm(vecs, axis=1).min()
    n = 1 / abs(np.linalg.det(P))
    kappa = d_nn * n ** (1 / 3)

    # BFS on a finite block of the lattice
    g = np.arange(-K, K + 1)
    I = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    index = {tuple(v): i for i, v in enumerate(I)}
    rows, cols = [], []
    for s in steps:
        J = I + s
        ok = np.all(np.abs(J) <= K, axis=1)
        rows.append(np.nonzero(ok)[0]); cols.append([index[tuple(v)] for v in J[ok]])
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    G = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(I), len(I))).tocsr()
    hops = shortest_path(G, unweighted=True, indices=index[(0, 0, 0)])
    X = I @ P.T
    r = np.linalg.norm(X, axis=1)
    # radius of the largest sphere inside the block (parallelepiped K * P)
    r_in = min(K * abs(np.linalg.det(P)) / np.linalg.norm(np.cross(P[:, j], P[:, k]))
               for j, k in [(1, 2), (0, 2), (0, 1)])
    sel = (r > 0.35 * r_in) & (r < 0.7 * r_in)
    stretch = hops[sel] * d_nn / r[sel]

    # bonds crossing a plane, per unit area / n^(2/3), averaged and worst case over orientation
    u = np.random.default_rng(seed).standard_normal((n_dirs, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cross = 0.5 * n * np.abs(vecs @ u.T).sum(0) / n ** (2 / 3)
    return dict(z=len(steps), kappa=kappa, stretch_mean=float(stretch.mean()),
                stretch_max=float(stretch.max()), sigma_hat_mean=float(cross.mean()),
                sigma_hat_min=float(cross.min()), d_nn=d_nn, n=n)


def wigner_seitz(name):
    """Faces (list of vertex arrays) of the lattice's Wigner-Seitz cell, in units of d_nn."""
    L = LATTICES[name]
    P = L['P']
    g = np.arange(-2, 3)
    I = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    X = I @ P.T
    d_nn = np.linalg.norm(_steps(L) @ P.T, axis=1).min()
    X = X / d_nn
    vor = Voronoi(X)
    center = int(np.argmin(np.linalg.norm(X, axis=1)))
    return hull_polygons(vor.vertices[vor.regions[vor.point_region[center]]])


def hull_polygons(verts):
    """Faces of the convex hull of verts as ordered vertex arrays (coplanar triangles merged)."""
    verts = np.asarray(verts, float)
    hull = ConvexHull(verts)
    faces = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq, 6))
        faces.setdefault(key, set()).update(simplex)
    polys = []
    for key, idx in faces.items():
        pts = verts[list(idx)]
        ctr = pts.mean(0)
        nrm = np.array(key[:3])
        a = pts[0] - ctr; a /= np.linalg.norm(a)
        b = np.cross(nrm, a)
        ang = np.arctan2((pts - ctr) @ b, (pts - ctr) @ a)
        polys.append(pts[np.argsort(ang)])
    return polys


# --- cell shape --------------------------------------------------------------------------------
# Inter-cell traffic crosses a cell's boundary on the links it cuts, so the best cell shape is the
# space-filling one that cuts the fewest node links per cell at fixed node count. A plane with unit
# normal u cuts gamma(u) = (n/2) sum_b |b.u| links per unit area (the same density lattice_stats
# averages). gamma is the support function of the zonotope sum_b [-b, b] (n/2), so the
# shape minimizing the total cut for its volume (the Wulff shape) is that zonotope. For FCC
# nodes the six bond directions are the A3 roots e_i - e_j, whose zonotope is the permutohedron:
# the regular truncated octahedron, Kelvin's cell, which tiles space on a BCC superlattice.
SUPERLATTICE_SHAPE = {'sc': 'cube', 'fcc': 'rhombic dodecahedron', 'bcc': 'truncated octahedron'}


def _faces(verts):
    """(unit normal, area) of each face of the convex hull of verts."""
    hull = ConvexHull(verts)
    faces = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        a, b, c3 = verts[simplex]
        key = tuple(np.round(eq[:3], 6))
        faces[key] = faces.get(key, 0.0) + 0.5 * np.linalg.norm(np.cross(b - a, c3 - a))
    return [(np.array(k), A) for k, A in faces.items()], hull.volume


def _superlattice_cell(name):
    """Vertices of the Wigner-Seitz cell of superlattice `name`, axes aligned with the node cube."""
    P = LATTICES[name]['P']
    g = np.arange(-2, 3)
    X = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3) @ P.T
    vor = Voronoi(X)
    center = int(np.argmin(np.linalg.norm(X, axis=1)))
    return vor.vertices[vor.regions[vor.point_region[center]]]


def cut_links(name, node='fcc', R=None):
    """Links cut per cell over N^(2/3), N = nodes per cell, for cells shaped like superlattice
    `name`'s Wigner-Seitz cell (optionally rotated by R), in the continuum limit. Also returns the
    Euclidean surface area at unit volume (Kelvin's measure)."""
    L = LATTICES[node]
    b = _steps(L) @ L['P'].T
    n = 1 / abs(np.linalg.det(L['P']))
    V = _superlattice_cell(name)
    if R is not None:
        V = V @ R.T
    faces, vol = _faces(V)
    N = n * vol
    cut = 0.5 * sum(A * 0.5 * n * np.abs(b @ u).sum() for u, A in faces)   # each face shared by 2 cells
    area = sum(A for _, A in faces) / vol ** (2 / 3)
    return dict(cut_per_N23=float(cut / N ** (2 / 3)), area_unit_volume=float(area))


def _rotations(k, seed=0):
    q = np.random.default_rng(seed).standard_normal((k, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def cut_links_counted(name, M, node='fcc'):
    """Direct count on a periodic box of 2 x 2 x 2 superlattice periods: FCC sites (cube = 1/2
    in half-integer units) assigned to the nearest superlattice point, links between different
    cells counted. Returns cut per cell over N^(2/3); tends to cut_links(name) as M grows."""
    assert node == 'fcc'
    box = 2 * M                                           # in units of half the node cube
    g = np.arange(box)
    I = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    I = I[I.sum(1) % 2 == 0]
    centers = [np.array(c) * M for c in np.ndindex(2, 2, 2)]
    if name == 'bcc':
        centers += [c + M / 2 for c in centers]
    elif name == 'fcc':
        centers += [c + np.array(o) * M / 2 for c in list(centers) for o in [(1, 1, 0), (1, 0, 1), (0, 1, 1)]]
    C = np.array(centers) + 0.1 * np.array([1, 2, 3]) / 7   # off-lattice offset breaks ties
    d = I[:, None, :] - C[None]
    d -= box * np.round(d / box)
    lab = (d ** 2).sum(-1).argmin(1)
    index = -np.ones((box,) * 3, int)
    index[tuple(I.T)] = np.arange(len(I))
    steps = np.array([s for s in np.ndindex(3, 3, 3) if sum(abs(x - 1) for x in s) == 2]) - 1
    cut = 0
    for s in steps:
        j = index[tuple(((I + s) % box).T)]
        cut += np.count_nonzero(lab != lab[j])
    cut //= 2                                             # each link seen from both ends
    N = len(I) / len(C)
    return float(cut / len(C) / N ** (2 / 3))


def cell_shapes(n_rot=3000):
    """Cut links per cell for the three cubic superlattice tilings of an FCC node lattice:
    axis-aligned, the best of n_rot random orientations, and a direct count for finite cells."""
    out = {}
    rots = _rotations(n_rot)
    for name, shape in SUPERLATTICE_SHAPE.items():
        aligned = cut_links(name)
        best = min(cut_links(name, R=R)['cut_per_N23'] for R in rots)
        out[shape] = dict(cut_per_N23=aligned['cut_per_N23'], cut_per_N23_best_rotation=min(best, aligned['cut_per_N23']),
                          area_unit_volume=aligned['area_unit_volume'],
                          counted={str(M): cut_links_counted(name, M) for M in (8, 16)})
    cube = out['cube']['cut_per_N23']
    for v in out.values():
        v['vs_cube'] = v['cut_per_N23'] / cube
    # a cell spanning the whole shell thickness tiles a surface: hexagons (Hales 2001) have
    # perimeter 2 sqrt(2 sqrt 3) per sqrt(area), against 4 for squares
    out['hexagon_vs_square_perimeter'] = float(2 * np.sqrt(2 * np.sqrt(3)) / 4)
    return out


# --- link energetics -------------------------------------------------------------------------
def energy_per_bit(nu, T, margin=2.0):
    """Coherent-detection cost per bit at the noise floor hnu(n+1) = hnu/(1-exp(-hnu/kT)),
    times a practical margin, plus ~kT ln2 to relay (decode/re-encode)."""
    x = h * nu / (k_B * T)
    return margin * LN2 * h * nu / -np.expm1(-x) + k_B * T * LN2


# --- cell design ---------------------------------------------------------------------------
def design(p, lam, dR, beta, lattice='fcc', tau=1.0, f_B=0.1, tau_p=1e-6, hops_local=2.0,
           stats=None):
    st = stats or lattice_stats(lattice)
    z, kappa, s_L = st['z'], st['kappa'], st['stretch_mean']
    nu = c / lam
    T = p.T_out
    F = p.covering * p.L_star / (4 * np.pi * p.R_out ** 2)
    kTln2 = k_B * T * LN2
    cdot_v = F / (dR * kTln2)                       # bit ops per s per m^3

    d = kappa ** 1.5 * np.sqrt(z * lam * dR / (4 * tau))
    n = (kappa / d) ** 3
    A = lam * d                                      # Fresnel-matched face area
    b = f_B * nu
    sigma_b = tau * f_B * c / (dR * lam ** 2)
    l_cell = 2 * sigma_b / (beta * cdot_v)
    t_hop = d / c + tau_p
    v = d / t_hop
    tau_cell = s_L * l_cell / v
    E_bit = energy_per_bit(nu, T)
    vol = l_cell ** 3 if l_cell < dR else l_cell ** 2 * dR
    return dict(lam=lam, nu=nu, dR=dR, beta=beta, lattice=lattice, z=z, d=d, n=n,
                N_z=dR / d, A_face=A, a_face=np.sqrt(A), P_node=F / (dR * n),
                c_node=cdot_v / n, b_link=b, beta_node=z * b * n / cdot_v,
                sigma_b=sigma_b, l_cell=l_cell, cells_per_side=l_cell / d,
                nodes_per_cell=n * vol, t_hop=t_hop, v_mesh=v, tau_cell=tau_cell,
                f_cell=1 / tau_cell, E_cell=F * vol / dR * tau_cell,
                P_cell=F * vol / dR, ops_cell=cdot_v * vol,
                E_bit=E_bit, E_bit_over_kTln2=E_bit / kTln2,
                eta_comm=beta * hops_local * E_bit / kTln2,
                energy_per_bit_m=s_L * E_bit / d)


def lam_energy_limit(p, beta, eta_max, hops_local=2.0):
    """Shortest wavelength at which local traffic still costs <= eta_max of the power."""
    f = lambda loglam: beta * hops_local * energy_per_bit(c / 10 ** loglam, p.T_out) / (
        k_B * p.T_out * LN2) - eta_max
    lo, hi = -9.0, 0.0
    if f(hi) > 0:
        return np.nan      # unaffordable even in the Rayleigh-Jeans limit
    if f(lo) < 0:
        return 10 ** lo
    return 10 ** brentq(f, lo, hi)


def optimize(p, beta, eta_max=0.01, lam_material=1e-6, cells_per_side=10, dR_max_frac=0.1,
             lattice='fcc', **kw):
    """Carrier: the shortest wavelength allowed by energy and materials (l_cell ∝ lambda^-2).
    Thickness: the thickest shell whose node spacing still leaves cells_per_side nodes
    across a cell (larger d = fewer, faster, cheaper long-range hops)."""
    lam = max(lam_energy_limit(p, beta, eta_max, kw.get('hops_local', 2.0)), lam_material)
    st = kw.pop('stats', None) or lattice_stats(lattice)
    probe = design(p, lam, 1.0, beta, lattice, stats=st, **kw)
    d_target = probe['l_cell'] / cells_per_side
    # d ∝ sqrt(dR): solve for dR giving d_target
    dR = (d_target / probe['d']) ** 2 * 1.0
    dR = min(dR, dR_max_frac * p.R_out)
    # at least one layer of nodes: d <= dR  <=>  dR >= kappa^3 z lambda / (4 tau)
    dR = max(dR, st['kappa'] ** 3 * st['z'] * lam / (4 * kw.get('tau', 1.0)))
    return design(p, lam, dR, beta, lattice, stats=st, **kw)


def prolate_lambda0(c, n=200):
    """Largest eigenvalue of the sinc kernel sin(c(x-y))/(pi(x-y)) on [-1, 1] (Slepian & Pollak 1961):
    the fraction of a band-limited beam's power that one aperture can deliver to another."""
    x, w = np.polynomial.legendre.leggauss(n)
    D = x[:, None] - x[None, :]
    K = np.where(np.abs(D) < 1e-14, c / np.pi, np.sin(c * D) / (np.pi * np.where(D == 0, 1, D)))
    M = np.sqrt(w)[:, None] * K * np.sqrt(w)[None, :]
    return np.linalg.eigvalsh(M)[-1]


def fresnel_spill(N_F=1.0):
    """Power not captured between two facing square apertures of area A at distance d with
    Fresnel number N_F = A/(lambda d).  The problem separates per axis with c = (pi/2) N_F."""
    return 1 - prolate_lambda0(np.pi / 2 * N_F) ** 2


def leakage(p, des, spill=None, tau=1.0):
    """Comm power escaping the shell: diffraction spill of local links (derived for the
    Fresnel-matched links of the design unless given), escaping through the outer covering
    (sources spread uniformly in optical depth)."""
    spill = fresnel_spill(1.0) if spill is None else spill
    f_esc = (1 - np.exp(-tau)) / (2 * tau)
    return des['eta_comm'] * p.covering * p.L_star * spill * f_esc
