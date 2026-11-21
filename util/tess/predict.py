import numpy as np
h,c,k=6.626e-34,2.998e8,1.381e-23
lam=np.linspace(0.3e-6,3e-6,20000)
B=lam**-5/np.expm1(h*c/(lam*k*2800))
R=np.interp(lam*1e9,[580,650,900,1050],[0,1,1,0],left=0,right=0)   # crude TESS response
phiT=np.trapezoid(B*R,lam)/np.trapezoid(B,lam)            # fraction of L* in TESS (resp-weighted)
band=(lam>0.95e-6)&(lam<1.05e-6)
gC=np.trapezoid(R[band],lam[band])/(0.1e-6)                 # mean TESS response over the comm band
print(f"2800 K: response-weighted fraction of L* in TESS band = {phiT:.3f}; mean response over 0.95-1.05 um = {gC:.2f}")
import json, pathlib
_opt = json.load(open(pathlib.Path(__file__).resolve().parents[2] / 'outputs' / 'cells.json'))['optimum']
_L = 2000 * 3.828e26   # fiducial L* (blackcloud.params)
for beta,leak in [(1e-6, _opt['1e-06']['leak_W'] / _L), (1e-4, _opt['1e-04']['leak_W'] / _L)]:
    for fc in [0.995,0.5,0.1]:
        L_leak=leak*fc/0.995*gC; L_tr=(1-fc)*phiT
        print(f"beta={beta:.0e} f_c={fc:5}: leak / transmitted starlight in TESS = {L_leak/L_tr:.2e}  "
              f"(host dimmed by {-2.5*np.log10(1-fc):.1f} mag)")
# coordination ceiling: excess correlated power at scale l can't vary faster than its relay-crossing time
Rc=1.27
for name,v in [("mesh 0.25c",0.25),("NIR mesh 0.38c",0.38),("backbone ~c",1.0)]:
    print(f"{name}: monopole coordinated only below f ~ v/(pi R) = {v/(np.pi*Rc):.3f} 1/d  (P > {np.pi*Rc/v:.0f} d)")
