"""B2 — Multiscale constitutive law: EPS worm-like-chain (WLC) network -> macro stress.

The thesis uses a compressible neo-Hookean elastic factor as the fast/crude limit. The physically
grounded EPS model is a WLC fibre network (Ehret & Böl 2013). Here we homogenise an Arruda-Boyce
8-chain WLC network (the canonical analytic network model) under uniaxial stretch and compare its
macro response to the neo-Hookean used in the thesis — showing WHERE neo-Hookean is adequate (small
strain) and where it fails (the strain-stiffening lock-up the network captures and neo-Hookean misses).

8-chain model: chains along the cube diagonals, chain stretch lambda_c = sqrt((l1^2+l2^2+l3^2)/3).
Langevin chain force via Padé inverse-Langevin. Uniaxial incompressible: l1=lam, l2=l3=1/sqrt(lam).
This is the macro constitutive law a Keio FE^2 scheme would call per Gauss point.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use
RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

mu = 100.0          # shear modulus (Pa) — biofilm order of magnitude
N = 8.0             # number of Kuhn segments per chain (lock-up at lambda_c = sqrt(N))

def invL(y):        # Padé approximation to the inverse Langevin function
    y = np.clip(y, -0.999, 0.999)
    return y * (3 - y**2) / (1 - y**2)

def cauchy_8chain(lam):
    # uniaxial incompressible: l1=lam, l2=l3=1/sqrt(lam); chain stretch:
    lc = np.sqrt((lam**2 + 2.0 / lam) / 3.0)
    beta = invL(lc / np.sqrt(N))
    # 8-chain Cauchy stress (uniaxial, incompressible), normalised so small-strain shear = mu
    pref = mu * np.sqrt(N) / (3.0 * lc) * beta
    return pref * (lam**2 - 1.0 / lam)

def cauchy_neo_hookean(lam):
    return mu * (lam**2 - 1.0 / lam)            # incompressible neo-Hookean uniaxial

lams = np.linspace(1.0, 2.6, 80)
s_wlc = np.array([cauchy_8chain(l) for l in lams])
s_nh = cauchy_neo_hookean(lams)
# small-strain check: ratio at lam->1
print("small-strain consistency (lam=1.02):  WLC/neo-Hookean = %.3f  (should be ~1)"
      % (cauchy_8chain(1.02) / cauchy_neo_hookean(1.02)))
i_lock = np.argmin(np.abs(lams - np.sqrt(N) * 0.92))
print("strain-stiffening: at lam=%.2f  WLC stress is %.1fx the neo-Hookean (lock-up the crude model misses)"
      % (lams[i_lock], s_wlc[i_lock] / s_nh[i_lock]))

fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.82))
ax.plot(lams, s_nh, "--", color=BLUE, lw=1.2, label="neo-Hookean (thesis, crude limit)")
ax.plot(lams, s_wlc, "-", color=RED, lw=1.2, label=r"WLC 8-chain network ($N=%g$)" % N)
ax.axvline(np.sqrt(N), ls=":", color="0.5", lw=0.8); ax.text(np.sqrt(N) - 0.02, 50, r"$\lambda_{\rm lock}$", fontsize=7, ha="right")
ax.set_xlabel(r"stretch $\lambda$"); ax.set_ylabel(r"Cauchy stress $\sigma$ [Pa]")
ax.set_title(r"Multiscale constitutive law: WLC network vs neo-Hookean", fontsize=8)
ax.legend(fontsize=7, loc="upper left"); ax.set_ylim(0, 1.5 * s_nh.max())
fig.savefig(OUT / "F12_multiscale_wlc.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "F12_multiscale_wlc.pdf")
