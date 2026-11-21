"""Computing-cell analysis: lattice comparison, carrier optimization, scale ladder, 3D local volume,
and the light-travel filter set against the global superstep cadence.
Writes outputs/fig6 and fig8-fig11 and outputs/cells.json."""
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from blackcloud.params import CloudParams, c, h, k_B, AU, kpc, Jy, L_sun, sigma_SB, LN2
from blackcloud import cells, thermo, network as net, plotstyle as ps
from blackcloud.observe import shell_transfer, modulation_amplitude

OUT = Path(__file__).parent / 'outputs'
OUT.mkdir(exist_ok=True)
ps.apply()
p = CloudParams()
res = {}

# ---------- lattices ------------------------------------------------------------------------
stats = {k: cells.lattice_stats(k) for k in cells.LATTICES}
res['lattices'] = {k: {**{a: float(b) for a, b in s.items()},
                       'shape': cells.LATTICES[k]['shape'],
                       'inspiration': cells.LATTICES[k]['inspiration']} for k, s in stats.items()}

fig = plt.figure(figsize=(ps.WC, 2.5))
abbr = {'sc': 'SC', 'hex': 'Hex', 'bcc': 'BCC', 'fcc': 'FCC'}     # as in the lattice table of the paper
for i, k in enumerate(['sc', 'hex', 'bcc', 'fcc']):
    x0, y0 = 0.5 * (i % 2), 0.5 * (1 - i // 2)       # 2 x 2 grid; label to the left of each polyhedron
    ax = fig.add_axes((x0 + 0.23, y0, 0.28, 0.5), projection='3d')
    polys = cells.wigner_seitz(k)
    ax.add_collection3d(Poly3DCollection(polys, facecolor='C0', edgecolor='k', alpha=0.35, lw=0.6))
    L = cells.LATTICES[k]
    vecs = cells._steps(L) @ L['P'].T
    vecs /= np.linalg.norm(vecs, axis=1).min()
    for v in vecs:
        if np.isclose(np.linalg.norm(v), 1):
            ax.plot(*np.c_[0.5 * v, 0.95 * v], color='C3', lw=1)
    lim = 0.95
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_box_aspect((1, 1, 1), zoom=1.3); ax.set_axis_off()
    s = stats[k]
    fig.text(x0 + 0.005, y0 + 0.25, f"{abbr[k]}\n{L['shape'].replace(' ', chr(10))}\n"
             f"z = {s['z']}\nstretch {s['stretch_mean']:.2f}", va='center', fontsize=ps.SMALL)
ps.save(fig, OUT / 'fig8_lattices.png')

# ---------- carrier optimization -----------------------------------------------------------------
eta_max, lam_mat = 0.01, 1e-6
lam = np.logspace(-6.7, -2, 300)
betas = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
cols = plt.cm.viridis(np.linspace(0, 0.9, len(betas)))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(ps.W, 3.0))
res['optimum'] = {}
for b, col in zip(betas, cols):
    D = [cells.design(p, l, 0.1 * p.R_out, b, 'fcc', stats=stats['fcc']) for l in lam]
    lc = np.array([d['l_cell'] for d in D]); eta = np.array([d['eta_comm'] for d in D])
    ok = (eta <= eta_max) & (lam >= lam_mat)
    a1.loglog(lam * 1e6, lc, color=col, ls=':', lw=1)
    blab = rf'$\beta = 10^{{{np.log10(b):.0f}}}$'
    a1.loglog(lam[ok] * 1e6, lc[ok], color=col, lw=2, label=blab)
    a2.loglog(lam * 1e6, eta, color=col, lw=1.5, label=blab)
    o = cells.optimize(p, b, eta_max, lam_mat, stats=stats['fcc'])
    a1.plot(o['lam'] * 1e6, o['l_cell'], 'o', color=col, mec='k')
    o['leak_W'] = cells.leakage(p, o)
    res['optimum'][f'{b:.0e}'] = {k: (float(v) if np.isscalar(v) and not isinstance(v, str) else v)
                                   for k, v in o.items()}
a1.axvline(lam_mat * 1e6, color='gray', ls='--')
a1.text(lam_mat * 1.15e6, 1e-7, 'material\nlimit', fontsize=ps.SMALL, va='bottom', color='gray')
a1.axvline(p.lam_comm * 1e6, color='C3', ls='--')
a1.text(p.lam_comm * 1.15e6, 1e6, 'mm\n(L–A)', fontsize=ps.SMALL, va='top', color='C3')
a1.set(xlabel='carrier wavelength [µm]', ylabel=r'cell size $\ell_{\rm cell}$ [m]', ylim=(1e-8, 1e7),
       xlim=(0.15, 3e4))
a2.axhline(eta_max, color='k', ls='--', lw=1)
a2.text(2.5e4, eta_max * 1.4, rf'$\eta_{{\rm max}}$ = {eta_max}', fontsize=ps.SMALL, ha='right', va='bottom')
a2.axvline(h * c / (k_B * p.T_out) * 1e6, color='gray', ls=':')
a2.text(h * c / (k_B * p.T_out) * 1.15e6, 3e-9, 'hν = kT', fontsize=ps.SMALL, color='gray', va='bottom')
a2.set(xlabel='carrier wavelength [µm]', ylabel=r'local-link power fraction $\eta$', ylim=(1e-9, 1),
       xlim=(0.15, 3e4))
for ax in (a1, a2):                                   # fewer log ticks so 9 pt labels fit a half-width panel
    ax.set_xticks([1e0, 1e2, 1e4])
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
fig.legend(*a2.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5,
           handlelength=1.5, columnspacing=1.0)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9))
ps.save(fig, OUT / 'fig9_carrier_optimum.png')

# ---------- fiducial optimum and the Laughlin-Adams sub-mm design --------------------------
beta0 = 1e-6
opt = cells.optimize(p, beta0, eta_max, lam_mat, stats=stats['fcc'])
la = cells.design(p, p.lam_comm, 0.1 * p.R_out, beta0, 'sc', stats=stats['sc'], tau_p=1e-3)
F = p.covering * p.L_star / (4 * np.pi * p.R_out ** 2)
bis = lambda d: 2 * np.pi * p.R_out * d['sigma_b'] * d['dR']
res['fiducial'] = {
    'beta': beta0, 'eta_max': eta_max, 'flux_W_m2': F,
    'optimum': {k: float(v) for k, v in opt.items() if np.isscalar(v) and not isinstance(v, str)},
    'laughlin_adams_submm': {k: float(v) for k, v in la.items() if np.isscalar(v) and not isinstance(v, str)},
    'global_bisection_bits_s_optimum': float(bis(opt)),
    'global_bisection_bits_s_submm': float(bis(la)),
    'total_compute_bits_s': float(p.covering * p.L_star / (k_B * p.T_out * LN2)),
    'n_cells_optimum': float(4 * np.pi * p.R_out ** 2 * opt['dR'] / (opt['l_cell'] ** 2 * min(opt['l_cell'], opt['dR']))),
}

# leakage at the optimum vs leaked starlight in the same band, at 1 kpc
nu0, B = opt['nu'], 0.1 * opt['nu']
nus = np.linspace(nu0 - B / 2, nu0 + B / 2, 200)
star_band = (1 - p.covering) * p.L_star * np.pi * np.trapezoid(thermo.planck_nu(nus, p.T_star), nus) / (sigma_SB * p.T_star ** 4)
therm_band = thermo.Lnu_shell(nu0, p.T_out, p.R_out) * B
leak = cells.leakage(p, opt)
d1 = 1 * kpc
res['fiducial']['leak'] = {
    'P_leak_W': float(leak), 'P_starlight_in_band_W': float(star_band), 'P_thermal_in_band_W': float(therm_band),
    'leak_over_starlight': float(leak / star_band),
    'F_leak_1kpc_Jy': float(leak / B / (4 * np.pi * d1 ** 2) / Jy),
    'F_star_1kpc_Jy': float(star_band / B / (4 * np.pi * d1 ** 2) / Jy)}

# ---------- scale ladder ------------------------------------------------------------------------
def ladder(d, s_L):
    pts = [('carrier λ', d['lam'], 1 / d['nu']),
           ('node face', d['a_face'], d['a_face'] / c),
           ('hop d', d['d'], d['t_hop']),
           ('cell ℓ', d['l_cell'], d['tau_cell']),
           ('shell thickness ΔR', d['dR'], s_L * d['dR'] / d['v_mesh']),
           ('half circumference πR', np.pi * p.R_out, s_L * np.pi * p.R_out / d['v_mesh'])]
    return pts

fig, ax = plt.subplots(figsize=(ps.WC, 3.0))
x = np.logspace(-7, 15, 10)
# label offsets (points) where the default would collide: the 3 mm λ sits right beside its cell size
nudge = {('C3', 'carrier λ'): ((2, -9), 'center', 'top')}
ax.loglog(x, x / c, color='gray', lw=0.8, ls=':')
short = {'carrier λ': 'λ', 'node face': 'face', 'hop d': 'd', 'cell ℓ': r'$\ell_{\rm cell}$',
         'shell thickness ΔR': 'ΔR', 'half circumference πR': 'πR'}
for d, s_L, col, lab, mk, line in [
        (opt, stats['fcc']['stretch_mean'], 'C0', f'optimum (β = {ps.sci(beta0, 0)}, {opt["lam"]*1e6:.0f} µm, FCC)', 'o', '-'),
        (la, stats['sc']['stretch_mean'], 'C3', f'Laughlin–Adams-like ({p.lam_comm * 1e3:.0f} mm, SC)', 's', 'none')]:
    pts = ladder(d, s_L)
    X = np.array([q[1] for q in pts]); Y = np.array([q[2] for q in pts])
    ax.loglog(X, Y, marker=mk, ls=line, color=col, label=lab, ms=6, mfc=col if line == '-' else 'none')
    for name, xx, yy in pts:
        below = col == 'C0'                       # optimum labeled below the diagonal, sub-mm above it
        off, ha, va = nudge.get((col, name), ((6, -6), 'left', 'top') if below else ((-6, 6), 'right', 'bottom'))
        ax.annotate(short[name], (xx, yy), textcoords='offset points', xytext=off, ha=ha, va=va,
                    fontsize=ps.SMALL, color=col)
ax.annotate('', xy=(la['l_cell'], la['tau_cell']), xytext=(la['d'], la['t_hop']),
            arrowprops=dict(arrowstyle='->', color='C3', lw=0.8, ls='--'))
ax.set(xlabel='length scale [m]', ylabel='characteristic time [s]', xlim=(1e-7, 1e16), ylim=(1e-18, 1e14))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
fig.canvas.draw()                                 # rotate the light-cone label to the line's on-screen slope
(x0, y0), (x1, y1) = ax.transData.transform([(1e6, 1e6 / c), (1e8, 1e8 / c)])
ax.text(3e6, 3e6 / c * 1e-6, 'light crossing x/c', rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)),
        rotation_mode='anchor', fontsize=ps.SMALL, color='gray')
ps.save(fig, OUT / 'fig10_scale_ladder.png')

# ---------- one cell, zooming in to the FCC lattice and to one node ----------------------------------
# Left: a cell drawn as the truncated octahedron (Kelvin's cell) that tiles the FCC node lattice with
# the fewest cut links, holding the fiducial ~1400 nodes. Right: zooms into a patch of the lattice
# (top) and into one node (bottom).
L = cells.LATTICES['fcc']
P = L['P'] / (np.sqrt(2) / 2)                         # d_nn = 1; cube axes, conventional cube a = sqrt(2)
a_fcc = np.sqrt(2)
n_nodes = 1 / abs(np.linalg.det(P))                   # nodes per d_nn^3
N_cell = res['fiducial']['optimum']['nodes_per_cell']
M = (2 * N_cell / n_nodes) ** (1 / 3)                 # square-to-square width: volume M^3/2 holds N_cell nodes
g = np.arange(-14, 15)
X = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3) @ P.T
inside = lambda Y, e=1e-9: (np.abs(Y).max(1) <= M / 2 + e) & (np.abs(Y).sum(1) <= 0.75 * M + e)
Xc = X[inside(X)]
to_verts = M / 4 * np.array(sorted({q for s1 in (-1, 1) for s2 in (-1, 1)
                                    for q in itertools.permutations((0, s1, 2 * s2))}))
to_faces = cells.hull_polygons(to_verts)
ws = cells.wigner_seitz('fcc')
cmap, VIEW = plt.cm.cividis, (20, -58)
depth = lambda z: cmap(0.5 + z / M)                    # color encodes radial depth, as in the cell


def to_fig(ax, pts):
    """Figure coordinates of 3D points drawn in ax (after the figure has been drawn)."""
    x, y, _ = proj3d.proj_transform(*np.asarray(pts, float).T, ax.get_proj())
    return fig.transFigure.inverted().transform(ax.transData.transform(np.c_[x, y]))


def box_edges(c0, h):
    corners = np.array(list(itertools.product((-h, h), repeat=3))) + c0
    return [np.stack([corners[i], corners[j]]) for i in range(8) for j in range(i + 1, 8)
            if np.isclose(np.abs(corners[i] - corners[j]).sum(), 2 * h)], corners


fig = plt.figure(figsize=(ps.W, 3.15))
axc = fig.add_axes((-0.02, -0.07, 0.64, 1.14), projection='3d')
axc.add_collection3d(Poly3DCollection(to_faces, facecolor='C0', edgecolor='none', alpha=0.07))
axc.add_collection3d(Line3DCollection([np.stack([f[k], f[(k + 1) % len(f)]]) for f in to_faces for k in range(len(f))],
                                     colors='k', lw=0.7))
axc.scatter(*Xc.T, c=[depth(z) for z in Xc[:, 2]], s=2.2, depthshade=False, lw=0)
# zoom region: a 2a cube around a node near the cell's front, upper edge
cam = np.array([np.cos(np.radians(VIEW[0])) * np.cos(np.radians(VIEW[1])),
                np.cos(np.radians(VIEW[0])) * np.sin(np.radians(VIEW[1])), np.sin(np.radians(VIEW[0]))])
target = 0.62 * M * (cam + np.array([0, 0, 0.6])) / np.linalg.norm(cam + np.array([0, 0, 0.6]))
c0 = Xc[np.argmin(np.linalg.norm(Xc - target, axis=1))]
edges, corners = box_edges(c0, a_fcc)
axc.add_collection3d(Line3DCollection(edges, colors='k', lw=1.1))
# starlight arrives from below: a row of arrows across the view, under the inner square face
perp = np.array([-np.sin(np.radians(VIEW[1])), np.cos(np.radians(VIEW[1])), 0])
for o in np.linspace(-0.42 * M, 0.42 * M, 5):
    axc.quiver(*(o * perp + [0, 0, -M / 2 - 3.3]), 0, 0, 1.9, color='gold', lw=1.8, arrow_length_ratio=0.3)
lim = 0.5 * M + 0.3
axc.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim - 3.3, lim))
axc.set_box_aspect((1, 1, (2 * lim + 3.3) / (2 * lim)), zoom=1.32); axc.view_init(*VIEW); axc.set_axis_off()

# lattice patch: the nodes within one conventional cube of the zoom center
axl = fig.add_axes((0.6, 0.5, 0.4, 0.5), projection='3d')
Xl = X[np.all(np.abs(X) <= a_fcc + 1e-9, axis=1)]
D = np.linalg.norm(Xl[:, None] - Xl[None], axis=-1)
ii, jj = np.nonzero(np.triu(np.isclose(D, 1.0, atol=1e-6)))
axl.add_collection3d(Line3DCollection(np.stack([Xl[ii], Xl[jj]], 1), colors='C3', lw=0.5, alpha=0.45))
# the node zoomed into next sits on the bottom face, nearest the viewer, so the zoom lines clear the patch
bottom = np.nonzero(np.isclose(Xl[:, 2], Xl[:, 2].min()))[0]
xn = Xl[bottom[np.argmax(Xl[bottom] @ cam)]]
nb = np.isclose(np.linalg.norm(Xl - xn, axis=1), 1.0)
axl.add_collection3d(Line3DCollection(np.stack([0 * Xl[nb] + xn, Xl[nb]], 1), colors='C3', lw=1.4))
for x0 in Xl:
    center = np.allclose(x0, xn)
    axl.add_collection3d(Poly3DCollection([0.2 * f + x0 for f in ws], edgecolor='k', lw=0.2, alpha=0.95,
                                          facecolor='C0' if center else depth(c0[2] + x0[2])))
small, small_c = box_edges(xn, 0.34)
axl.add_collection3d(Line3DCollection(small, colors='k', lw=1.0))
axl.set(xlim=(-a_fcc, a_fcc), ylim=(-a_fcc, a_fcc), zlim=(-a_fcc, a_fcc))
axl.set_box_aspect((1, 1, 1), zoom=0.82); axl.view_init(*VIEW); axl.set_axis_off()

# one node: a rhombic dodecahedron with an aperture on each face toward a neighbor
axn = fig.add_axes((0.6, 0.0, 0.4, 0.5), projection='3d')
axn.add_collection3d(Poly3DCollection(ws, facecolor='C0', edgecolor='k', alpha=0.5, lw=0.8))
for v in cells._steps(L) @ P.T:
    tip = 1.6 * v
    axn.plot(*np.c_[0.5 * v, tip], color='C3', lw=1.2)
    axn.scatter(*tip, color='C3', s=10, depthshade=False)
axn.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), zlim=(-1.4, 1.4))
axn.set_box_aspect((1, 1, 1), zoom=1.1); axn.view_init(*VIEW); axn.set_axis_off()

# frames around the zooms and connector lines from each zoom region to its view
fig.canvas.draw()
for ax_ in (axl, axn):
    x0_, y0_, w_, h_ = ax_.get_position().bounds
    fig.add_artist(Rectangle((x0_ + 0.01, y0_ + 0.015), w_ - 0.02, h_ - 0.03, transform=fig.transFigure,
                             fill=False, ec='0.5', lw=0.7))
def connect(src, dst):
    for (sx, sy), (dx, dy) in zip(src, dst):
        fig.add_artist(Line2D([sx, dx], [sy, dy], transform=fig.transFigure, color='0.5', lw=0.6))


def frame(ax_):
    x0_, y0_, w_, h_ = ax_.get_position().bounds
    return x0_ + 0.01, y0_ + 0.015, x0_ + w_ - 0.01, y0_ + h_ - 0.015


q = to_fig(axc, corners); fx0, fy0, fx1, fy1 = frame(axl)        # cell -> lattice: rightward
connect([(q[:, 0].max(), q[:, 1].max()), (q[:, 0].max(), q[:, 1].min())], [(fx0, fy1), (fx0, fy0)])
q = to_fig(axl, small_c); fx0, fy0, fx1, fy1 = frame(axn)        # lattice -> node: downward
connect([(q[:, 0].min(), q[:, 1].min()), (q[:, 0].max(), q[:, 1].min())], [(fx0, fy1), (fx1, fy1)])
fig.text(0.02, 0.97, 'cell', fontsize=ps.SMALL, va='top', bbox=ps.BOX)
for ax_, txt in ((axl, 'FCC lattice'), (axn, 'node')):
    x0_, y0_, w_, h_ = ax_.get_position().bounds
    fig.text(x0_ + 0.025, y0_ + h_ - 0.03, txt, fontsize=ps.SMALL, va='top', bbox=ps.BOX)
ps.save(fig, OUT / 'fig11_local_volume.png')

# ---------- one cell compared with terrestrial computers ---------------------------------------
# Terrestrial figures: TOP500 June 2026 (HPL Rmax, listed power) and the Laughlin et al. draft
# (research/laughlinadams2021/) for a SHA-256 mining ASIC (1e-3 erg per hash of ~2000 bit operations).
# A double-precision flop is taken as ~3e4 irreversible bit operations (order of magnitude, range 1e4-1e5), as in the beta estimate.
BITOPS_PER_FLOP = 3e4
landauer = lambda T: 1 / (k_B * T * LN2)               # bit ops per joule at the Landauer limit
terrestrial = {
    'LineShine (TOP500 #1, 2026)': dict(P=42.220e6, flops=2.1984e18),
    'El Capitan (TOP500 #2, 2026)': dict(P=29.685e6, flops=1.809e18),
    'Fugaku (2020-21 #1)': dict(P=29.899e6, flops=442.01e15),
}
comp = {'bitops_per_flop': BITOPS_PER_FLOP,
        'landauer_bitops_per_J_177K': landauer(p.T_out), 'landauer_bitops_per_J_300K': landauer(300.0)}
for name, m in terrestrial.items():
    ops = m['flops'] * BITOPS_PER_FLOP
    comp[name] = dict(P_W=m['P'], flops=m['flops'], bitops_s=ops, bitops_per_J=ops / m['P'],
                      below_landauer_300K=landauer(300.0) / (ops / m['P']))
comp['mining ASIC (Laughlin et al.)'] = dict(bitops_per_J=2e6 * 1e7,
                                             below_landauer_300K=landauer(300.0) / 2e13)
for k, o in res['optimum'].items():
    comp[f'cell beta={k}'] = dict(P_W=o['P_cell'], bitops_s=o['ops_cell'],
                                  flop_equiv=o['ops_cell'] / BITOPS_PER_FLOP, nodes=o['nodes_per_cell'],
                                  l_cell_m=o['l_cell'], f_cell_Hz=o['f_cell'],
                                  el_capitans=o['ops_cell'] / comp['El Capitan (TOP500 #2, 2026)']['bitops_s'],
                                  power_ratio_vs_el_capitan_same_ops=(
                                      comp['El Capitan (TOP500 #2, 2026)']['P_W'] * o['ops_cell']
                                      / comp['El Capitan (TOP500 #2, 2026)']['bitops_s'] / o['P_cell']))
M_cell = p.M_cloud / res['fiducial']['n_cells_optimum']          # kg, fiducial design
comp['cell beta=1e-06'].update(mass_kg=M_cell, oklo=opt['ops_cell'] / (M_cell * 1e3))  # bit ops / g / s
comp['node beta=1e-06'] = dict(P_W=opt['P_node'], bitops_s=opt['c_node'], flop_equiv=opt['c_node'] / BITOPS_PER_FLOP)
comp['whole cloud'] = dict(P_W=p.covering * p.L_star, bitops_s=res['fiducial']['total_compute_bits_s'],
                           el_capitans=res['fiducial']['total_compute_bits_s'] / comp['El Capitan (TOP500 #2, 2026)']['bitops_s'])
res['terrestrial_comparison'] = comp

# ---------- workloads at each level of the hierarchy (fiducial design) ----------------------------
yr = 3.15576e7
AGB_LIFETIME, REBUILD = 1e4 * yr, 300 * yr               # Laughlin & Adams timescales
s_fcc = stats['fcc']['stretch_mean']
node_bw = opt['z'] * opt['b_link']
t_mesh = s_fcc * np.pi * p.R_out / opt['v_mesh']          # antipodal latency over the mesh
# The backbone of the propagation model has 24 hubs at the snub-cube vertices. A broadcast leaves the
# root hub, crosses the cavity to every hub at c (processing at both ends), then spreads over the mesh,
# so a cell hears it at the minimum over hubs of chord/c + 2 tau_hub + its mesh time from that hub. The
# gather is the time reverse, so a superstep takes twice the slowest broadcast. A backbone dense enough
# that no mesh leg remains reaches the light-crossing bound, 2R/c + 2 tau_hub each way.
hubs24 = net.snub_cube()
x_ss = net.fibonacci_sphere(200000, np.random.default_rng(5))
ang_ss = np.arccos(np.clip(x_ss @ hubs24.T, -1, 1))
chord_ss = 2 * p.R_out * np.sin(np.arccos(np.clip(hubs24 @ hubs24[0], -1, 1)) / 2)
t_bb = (chord_ss / c + 2 * p.tau_hub + s_fcc * ang_ss * p.R_out / opt['v_mesh']).min(1).max()
t_bound = 2 * p.R_out / c + 2 * p.tau_hub
cover24_deg = float(np.degrees(ang_ss.min(1).max()))
sync_bb, sync_bound, sync_mesh = 2 * t_bb, 2 * t_bound, 2 * t_mesh   # gather + broadcast = one global superstep
ops_total = res['fiducial']['total_compute_bits_s']
bis = res['fiducial']['global_bisection_bits_s_optimum']
n_cells = res['fiducial']['n_cells_optimum']
# 3D stencil on a node: b bits exchanged per halo point, c bit-ops per point update (~10 flops)
b_pt, c_pt = 64.0, 10 * BITOPS_PER_FLOP
L_bw = 6 * b_pt * opt['c_node'] / (c_pt * node_bw)                     # halo time = compute time
L_lat = (opt['t_hop'] * opt['c_node'] / c_pt) ** (1 / 3)                 # compute time = hop latency
L_st = max(L_bw, L_lat)
step = L_st ** 3 * c_pt / opt['c_node']
pts_total = L_st ** 3 * n_cells * opt['nodes_per_cell']
OPS_PER_TRIAL = 1e4                                      # bit ops to test one key / candidate
res['workloads'] = {
    'node': dict(P_W=opt['P_node'], bitops_s=opt['c_node'], link_bw_bits_s=node_bw,
                 bw_per_op=node_bw / opt['c_node'], neighbor_latency_s=opt['t_hop']),
    'cell': dict(P_W=opt['P_cell'], bitops_s=opt['ops_cell'], sync_s=opt['tau_cell'],
                 bisection_bits_s=opt['sigma_b'] * opt['l_cell'] ** 2,
                 ops_per_sync=opt['ops_cell'] * opt['tau_cell']),
    'cloud': dict(bitops_s=ops_total, bisection_bits_s=bis,
                  superstep_backbone_d=sync_bb / 86400, superstep_mesh_d=sync_mesh / 86400,
                  superstep_backbone_bound_d=sync_bound / 86400, broadcast_backbone_d=t_bb / 86400,
                  backbone_cover_deg=cover24_deg, mesh_speed_c=opt['v_mesh'] / c, fcc_stretch=s_fcc,
                  supersteps_agb_backbone=AGB_LIFETIME / sync_bb, supersteps_agb_mesh=AGB_LIFETIME / sync_mesh,
                  supersteps_rebuild_backbone=REBUILD / sync_bb,
                  ops_per_superstep=ops_total * sync_bb,
                  global_bits_per_superstep=bis * sync_bb,
                  global_bits_per_cell_per_superstep=bis * sync_bb / n_cells,
                  ops_lifetime=ops_total * AGB_LIFETIME,
                  energy_lifetime_J=p.covering * p.L_star * AGB_LIFETIME,
                  key_bits_searchable=np.log2(ops_total * AGB_LIFETIME / OPS_PER_TRIAL),
                  landauer_J_count_2_256=2.0 ** 256 * k_B * p.T_out * LN2,
                  H_superstep_backbone=float(shell_transfer(1 / sync_bb, p.R_out)),
                  H_superstep_backbone_bound=float(shell_transfer(1 / sync_bound, p.R_out)),
                  H_superstep_mesh=float(shell_transfer(1 / sync_mesh, p.R_out)),
                  H_cell_clock=float(shell_transfer(opt['f_cell'], p.R_out))),
    'stencil': dict(bits_per_halo_point=b_pt, bitops_per_update=c_pt, L_bandwidth=L_bw, L_latency=L_st,
                    points_per_node=L_st ** 3, step_s=step, updates_per_s=1 / step,
                    halo_time_s=6 * L_st ** 2 * b_pt / node_bw, points_total=pts_total,
                    grid_signal_speed_m_s=(opt['d'] / L_st) / step,
                    grid_crossing_yr=np.pi * p.R_out / ((opt['d'] / L_st) / step) / yr),
}

# ---------- light-travel filter against the cadence of coordinated activity ----------------
# Cells running independent workloads (minds) are not filtered but average away as N^-1/2;
# what an unresolved observer can see is shell-wide modulation, the monopole of Eq. (kernel).
day = 86400.0
N_MC = 20000
x_mc = net.fibonacci_sphere(N_MC, np.random.default_rng(3))
n_hat = np.array([0.3, -0.4, 0.866]); n_hat /= np.linalg.norm(n_hat)
P_sim = np.logspace(-1, 2.5, 29) * day
H_sim = modulation_amplitude(x_mc, n_hat, p.R_out, 1 / P_sim)
H_ana = shell_transfer(1 / P_sim, p.R_out)
f = np.logspace(-3.5, 1.5, 2000) / day
T_lc = 2 * p.R_out / c / day
fig, ax = plt.subplots(figsize=(ps.W, 3.3))
ax.axvspan(sync_bound / day, sync_mesh / day, color='C0', alpha=0.15, lw=0)
ax.text(np.sqrt(sync_bound * sync_mesh) / day, 1.4e-3, 'global\nsupersteps', ha='center', va='bottom',
        fontsize=ps.SMALL, color='C0')
ax.loglog(1 / f / day, shell_transfer(f, p.R_out), color='C0', label=r'analytic, $|\mathrm{sinc}(2fR/c)|$')
ax.loglog(P_sim / day, H_sim, 'o', color='k', ms=3.5, label=f'simulated, {N_MC} cells')
ax.axvline(T_lc, color='gray', ls='--')
ax.text(T_lc / 1.12, 1.4e-3, f'2R/c = {T_lc:.2f} d', rotation=90, ha='right', va='bottom',
        fontsize=ps.SMALL, color='gray')
ax.axvline(p.mira_period / day, color='C3', ls=':')
ax.text(p.mira_period / day * 1.12, 1.4e-3, 'Mira period', rotation=90, ha='left', va='bottom',
        fontsize=ps.SMALL, color='C3')
ax.set(xlabel='period of shell-wide modulation [days]', ylabel='observed fractional amplitude',
       xlim=(0.03, 1000), ylim=(1e-3, 10))
ax.legend(loc='upper left')
ps.save(fig, OUT / 'fig6_light_travel_filter.png')
res['light_travel_filter'] = dict(n_cells_mc=N_MC,
                                  max_abs_dev_P_gt_2Rc=float(np.abs(H_sim - H_ana)[P_sim > 2 * p.R_out / c].max()),
                                  mc_floor_below_half_day=float(np.median(H_sim[P_sim < 0.5 * day])))

# cell shape: links cut per cell for cube, rhombic-dodecahedron and truncated-octahedron tilings
res['cell_shapes'] = cells.cell_shapes()
print('cell shapes (cut links per cell / N^(2/3))')
[print(f"  {k}: {v['cut_per_N23']:.3f} ({v['vs_cube']:.3f} of cube), best rotation {v['cut_per_N23_best_rotation']:.3f}")
 for k, v in res['cell_shapes'].items() if isinstance(v, dict)]

json.dump(res, open(OUT / 'cells.json', 'w'), indent=2, default=float)
print("workloads"); [print(" ", k, {a: f"{b:.3g}" for a, b in v.items()}) for k, v in res["workloads"].items()]
print('terrestrial comparison')
for k, v in comp.items():
    print(' ', k, v if not isinstance(v, dict) else {a: f'{b:.3g}' for a, b in v.items()})

print('lattices'); [print(' ', k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()}) for k, v in res['lattices'].items()]
print('optima by beta')
for k, o in res['optimum'].items():
    print(f"  beta={k}: lam={o['lam']*1e6:.3g}um dR={o['dR']/AU:.3g}AU d={o['d']:.3g}m a={o['a_face']:.3g}m "
          f"l={o['l_cell']:.3g}m l/d={o['cells_per_side']:.3g} Nz={o['N_z']:.3g} f={o['f_cell']:.3g}Hz "
          f"eta={o['eta_comm']:.2g} P_node={o['P_node']:.3g}W leak/L={o['leak_W']/(p.L_star):.2g}")
fd = res['fiducial']
print('fiducial global bisection opt/submm/compute:', f"{fd['global_bisection_bits_s_optimum']:.3g}",
      f"{fd['global_bisection_bits_s_submm']:.3g}", f"{fd['total_compute_bits_s']:.3g}", 'n_cells', f"{fd['n_cells_optimum']:.3g}")
print('LA submm: l_cell', f"{la['l_cell']:.3g}", 'd', f"{la['d']:.3g}", 'beta_node', f"{la['beta_node']:.3g}")
print('leak', {k: f'{v:.3g}' for k, v in fd['leak'].items()})
print('opt E_cell', f"{opt['E_cell']:.3g}", 'nodes/cell', f"{opt['nodes_per_cell']:.3g}", 'tau_cell', f"{opt['tau_cell']:.3g}", 'v/c', f"{opt['v_mesh']/c:.3g}")
