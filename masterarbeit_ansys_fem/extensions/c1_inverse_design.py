"""C1 — Inverse design: mechanical probiotic steering.

Forward model (from the VERIFIED column physics, EXTENSIONS_RECIPES A4/tall-column): the residual
stress that drives detachment is a SUBSTRATUM BOUNDARY LAYER — only growth within ~one film-width of
the implant surface builds interface stress; growth higher up relieves by free upward expansion. So a
defensible surrogate for the detachment driver is

    J[phi] = integral_0^1  w(z) * phi_Pg(z) dz ,   w(z) = exp(-z / z0)   (boundary-layer weight, z0~0.25)

i.e. Pg mass near the substratum is mechanically costly, Pg mass near the surface is (almost) free.

Inverse-design question (probiotic steering): given a FIXED total Pg load M = integral phi dz that the
ecology delivers, how should it be DISTRIBUTED in depth to minimise the detachment driver J? This is
the mechanical target a microbiome-steering intervention should aim for. Constraints 0<=phi<=phi_max.

The optimum is analytic (push all Pg as far from the substratum as the cap allows), but we solve it
numerically and compare against the observed DYSBIOTIC profile (Pg sequestered AT the substratum, the
worst case) and a uniform profile. Output: the three profiles, their detachment driver J, and the
achievable risk reduction. No Abaqus needed — this is a design study on top of the verified mechanics.
"""
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use
RED, BLUE, GREEN = "#d62728", "#1f77b4", "#2ca02c"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

Nz = 40
z = (np.arange(Nz) + 0.5) / Nz                 # cell-centre normalized depth (0 = substratum)
z0 = 0.25
w = np.exp(-z / z0)                            # boundary-layer cost weight
dz = 1.0 / Nz
M = 0.20                                        # fixed total Pg load (integral phi dz)
phi_max = 0.6

def J(phi):
    return float(np.sum(w * phi) * dz)

# observed dysbiotic profile: Pg sequestered at the substratum (decreasing with z) -> worst case
phi_DH = np.exp(-z / 0.18); phi_DH *= M / (np.sum(phi_DH) * dz)
phi_DH = np.minimum(phi_DH, phi_max)
phi_uniform = np.full(Nz, M)                    # = M since integral over [0,1]

# optimum: minimise sum(w*phi)*dz  s.t.  sum(phi)*dz = M, 0<=phi<=phi_max  (linear program)
c = w * dz
A_eq = np.ones((1, Nz)) * dz; b_eq = [M]
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, phi_max)] * Nz, method="highs")
phi_opt = res.x

print("detachment driver J  (lower = less detachment):")
print("  dysbiotic (Pg at substratum):  J = %.4f" % J(phi_DH))
print("  uniform:                       J = %.4f" % J(phi_uniform))
print("  optimal (Pg steered to surface): J = %.4f" % J(phi_opt))
red = 100 * (1 - J(phi_opt) / J(phi_DH))
print("  => mechanical risk reduction vs dysbiotic: %.0f%%  (same total Pg load M=%.2f)" % (red, M))

fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.85))
ax.plot(phi_DH, z, "-", color=RED, lw=1.2, label=r"dysbiotic (observed): $J=%.3f$" % J(phi_DH))
ax.plot(phi_uniform, z, "--", color="0.5", lw=1.0, label=r"uniform: $J=%.3f$" % J(phi_uniform))
ax.plot(phi_opt, z, "-", color=GREEN, lw=1.2, label=r"steered optimum: $J=%.3f$" % J(phi_opt))
ax.axhspan(0, z0, color=RED, alpha=0.06); ax.text(0.42, 0.05, "substratum cost zone", fontsize=6, color="0.35")
ax.set_xlabel(r"$\phi_{Pg}(z)$"); ax.set_ylabel(r"depth $z$ (0 = implant surface)")
ax.set_title(r"Probiotic steering: same Pg load, %.0f\%% less detachment driver" % red, fontsize=8)
ax.legend(fontsize=6.5, loc="upper right")
fig.savefig(OUT / "F11_inverse_design.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "F11_inverse_design.pdf")
