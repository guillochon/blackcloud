"""Check: unresolved shell light curve at frequency f sees angular mode l with weight (2l+1) j_l^2(2 pi f R/c)."""
import numpy as np
from scipy.special import spherical_jn, eval_legendre
from scipy.integrate import quad
Rc = 1.27  # R/c in days (220 AU)
# 1) per-mode transfer: 1/2 int P_l(mu) exp(i x mu) dmu = i^l j_l(x)
for l in [0, 1, 2, 5]:
    for x in [0.5, 3.0, 10.0]:
        re = quad(lambda m: eval_legendre(l, m)*np.cos(x*m), -1, 1)[0]/2
        im = quad(lambda m: eval_legendre(l, m)*np.sin(x*m), -1, 1)[0]/2
        assert abs(np.hypot(re, im) - abs(spherical_jn(l, x))) < 1e-8
print("per-mode transfer = |j_l(x)|: ok")
# 2) angularly white activity: sum (2l+1) j_l^2 = 1 at every x -> incoherent cells unfiltered
for x in [0.1, 3, 30, 300]:
    ls = np.arange(0, int(x)+200)
    print(f"x={x:6}: sum(2l+1)j_l^2 = {np.sum((2*ls+1)*spherical_jn(ls, x)**2):.6f}")
# 3) Monte-Carlo: 20000 shell cells, activity correlated on angle theta_c, white in time
rng = np.random.default_rng(1)
N = 20000
mu = rng.uniform(-1, 1, N); ph = rng.uniform(0, 2*np.pi, N); s = np.sqrt(1-mu**2)
X = np.c_[s*np.cos(ph), s*np.sin(ph), mu]
f = np.array([0.02, 0.1, 0.2, 0.39, 0.6, 1.0, 2.0])  # 1/day
x = 2*np.pi*f*Rc
for name, lcut in [("monopole only", 0), ("theta_c ~ 30 deg (l<=6)", 6), ("uncorrelated cells", None)]:
    if lcut is None:
        pred = np.ones_like(x) / N  # white: |sum e^{ix mu}/N|^2 expectation = 1/N
        sims = []
        for _ in range(200):
            a = rng.normal(size=N) + 1j*rng.normal(size=N)
            sims.append(np.abs((a[None, :]*np.exp(1j*np.outer(x, mu))).sum(1)/N)**2 / 2)
        sim = np.mean(sims, 0)
    else:
        ls = np.arange(lcut+1); Cl = np.ones(lcut+1)
        pred = np.array([np.sum((2*ls+1)*Cl*spherical_jn(ls, xi)**2) for xi in x]) / np.sum((2*ls+1)*Cl)
        sim = pred  # analytic; checked by (1)
    print(name.ljust(26), " ".join(f"{v:9.3g}" for v in pred/pred[0]), "| sim/pred:", " ".join(f"{v:.2f}" for v in sim/pred) if lcut is None else "")
print("freqs (1/d):", f)
