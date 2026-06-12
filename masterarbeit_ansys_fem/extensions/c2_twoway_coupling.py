"""C2 — Two-way coupling: growth-induced stress feeds back on the reaction (mechanotransduction).

The thesis FEM is ONE-WAY: composition phi_Pg(z) drives growth -> stress. Biofilms also show the
reverse: compressive mechanical stress suppresses local growth/metabolism (a known mechanotransductive
feedback). Here we prototype the closed loop in 1-D depth:

  d phi/dt = D d^2phi/dz^2  +  g(z) * phi * (K(sigma) - phi)         (logistic with mechano-modulated K)
  sigma(z) = boundary-layer residual stress built by the growth so far  ~  W * (gamma * phi)
  K(sigma) = 1 / (1 + (sigma/sigma_c)^2)                             (compressive stress lowers carrying capacity)

One-way = K==1 (no feedback). Two-way lets stress lower the local carrying capacity, so the fixed
point at the high-stress substratum drops below 1 -> Pg is redistributed away from the substratum.
Pure numpy; this is the C2 morphogenesis prototype.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use
RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

Nz = 60; L = 1.0; dz = L / Nz
z = (np.arange(Nz) + 0.5) * dz
D = 2e-3; gamma = 3.0; sigma_c = 0.5
g = np.exp(-z / 0.3)                       # intrinsic growth bias toward substratum (anaerobic niche)
Wker = np.exp(-z / 0.25)                   # substratum boundary-layer stress kernel (verified physics)

def stress(phi):
    # cumulative boundary-layer residual stress from growth below a depth (free top relieves above)
    return Wker * np.cumsum((gamma * phi)[::-1])[::-1] * dz

def lap(p):
    q = np.empty_like(p)
    q[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dz**2
    q[0] = (p[1] - p[0]) / dz**2; q[-1] = (p[-2] - p[-1]) / dz**2   # no-flux walls
    return q

def integrate(feedback):
    phi = 0.05 + 0.0 * z; dt = 0.2 * dz**2 / D
    for _ in range(60000):
        sig = stress(phi)
        K = 1.0 / (1.0 + (sig / sigma_c)**2) if feedback else 1.0   # stress lowers carrying capacity
        phi = phi + dt * (D * lap(phi) + g * phi * (K - phi))
        phi = np.clip(phi, 0, 1)
    return phi, stress(phi)

phi1, s1 = integrate(feedback=False)       # one-way (thesis)
phi2, s2 = integrate(feedback=True)        # two-way (mechanotransduction)
zc = np.sum(z * phi2) / np.sum(phi2) - np.sum(z * phi1) / np.sum(phi1)
print("one-way  : substratum phi = %.3f, peak stress = %.3f" % (phi1[0], s1.max()))
print("two-way  : substratum phi = %.3f, peak stress = %.3f" % (phi2[0], s2.max()))
print("feedback shifts Pg centre-of-mass UP by dz = %.3f and cuts peak interface stress by %.0f%%"
      % (zc, 100 * (1 - s2.max() / s1.max())))

fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.5), sharey=True)
a1.plot(phi1, z, "--", color=BLUE, lw=1.2, label="one-way (thesis)")
a1.plot(phi2, z, "-", color=RED, lw=1.2, label="two-way (mechano-gated)")
a1.set_xlabel(r"$\phi_{Pg}(z)$"); a1.set_ylabel(r"depth $z$ (0 = substratum)"); a1.legend(fontsize=7)
a1.set_title(r"Composition", fontsize=8)
a2.plot(s1, z, "--", color=BLUE, lw=1.2); a2.plot(s2, z, "-", color=RED, lw=1.2)
a2.set_xlabel(r"residual stress $\sigma(z)$"); a2.set_title(r"Stress self-limits growth", fontsize=8)
fig.suptitle(r"Two-way coupling: stress feedback relieves dysbiotic sequestration", fontsize=8, y=1.0)
fig.savefig(OUT / "F14_twoway_coupling.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "F14_twoway_coupling.pdf")
