"""Generate thesis figures F2-F5 in the unified thesis_style (usetex/lmodern/9pt).
Run from repo root with TeX Live on PATH:
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python masterarbeit_ansys_fem/gen_thesis_figs.py
Outputs: masterarbeit_ansys_fem/figures/F{2..5}_*.pdf
(F1 = coupling-architecture schematic lives as TikZ in the slides.)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
OUT.mkdir(exist_ok=True)
RED, BLUE = "#d62728", "#1f77b4"

# ---- F2: growth calibration (thickness-derived growth strain vs phi_Pg, DH fit corr 0.88) ----
DH = np.array([[0.081, 22.458], [0.184, 67.876], [0.197, 76.621], [0.205, 52.968], [0.206, 78.953]])
h0 = DH[:, 1].min(); eps = (DH[:, 1] - h0) / h0
fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.78))
ax.scatter(DH[:, 0], eps, color=RED, zorder=3, s=18)
xs = np.linspace(DH[:, 0].min(), DH[:, 0].max(), 50)
ax.plot(xs, 17.18 * xs - 1.34, color="0.4", lw=1.0)
ax.set_xlabel(r"$\phi_{Pg}$"); ax.set_ylabel(r"depth growth strain $\varepsilon_{zz}=(h-h_0)/h_0$")
ax.set_title(r"CLSM calibration (DH): $r=0.88$, $\beta_{\mathrm{depth}}\approx17$", fontsize=8)
fig.savefig(OUT / "F2_growth_calibration.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F3: finite-strain free-swelling verification (U3_top -> lambda_z-1) ----
k = np.arange(21); U3 = np.linspace(0.0, 0.3018, 21)
fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.78))
ax.plot(k, U3, "o-", color=RED, ms=3, lw=1.0)
ax.axhline(0.30, ls="--", color="0.5", lw=0.8)
ax.set_xlabel("increment"); ax.set_ylabel(r"top displacement $u_3$")
ax.set_title(r"Finite-strain free-swelling: $u_3\!\to\!\lambda_z\!-\!1$ (Abaqus)", fontsize=8)
fig.savefig(OUT / "F3_free_swelling.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F4: residual-stress profile DH vs CH (the headline) ----
DHz = [0.068, 0.202, 0.334, 0.464, 0.596, 0.730, 0.867, 1.005]
DHs = [-293.29, -219.17, -135.67, -147.88, -223.80, -305.14, -354.26, -294.31]
CHz = [0.066, 0.199, 0.332, 0.466, 0.599, 0.731, 0.864, 0.996]
CHs = [-198.91, -230.12, -234.02, -222.48, -211.65, -213.38, -211.14, -153.25]
fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.85))
ax.plot(DHs, DHz, "o-", color=RED, ms=3, lw=1.0, label=r"DH (dysbiotic)")
ax.plot(CHs, CHz, "s-", color=BLUE, ms=3, lw=1.0, label=r"CH (commensal)")
ax.invert_xaxis()
ax.set_xlabel(r"in-plane residual stress $S_{11}(z)$"); ax.set_ylabel(r"depth $z/H$")
ax.set_title(r"Dysbiosis: higher \& stratified residual stress", fontsize=8)
ax.legend(fontsize=7, loc="lower left")
fig.savefig(OUT / "F4_residual_stress_DHvsCH.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F5: mesh convergence ----
N = [4, 8, 16, 32]; peak = [367.39, 354.26, 315.55, 319.67]
fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.78))
ax.plot(N, peak, "o-", color=BLUE, ms=3, lw=1.0)
ax.axhline(319.7, ls="--", color="0.5", lw=0.8)
ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels(N)
ax.set_xlabel(r"elements through thickness $N$"); ax.set_ylabel(r"peak $|S_{11}|$ (interior)")
ax.set_title(r"Mesh convergence (N=16 vs 32: 1.3\%)", fontsize=8)
fig.savefig(OUT / "F5_mesh_convergence.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F6: cohesive delamination DH vs CH (the clinical detachment result) ----
dx = 4.0 / 12.0
xc = np.array([(i + 0.5) * dx for i in range(12)])          # cohesive element centroids; free edge at x=4
SDEG_DH = np.array([0, 0, 0.61, 0.89, 0.96, 0.99, 1, 1, 1, 1, 1, 1.0])
SDEG_CH = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70, 0.89, 0.95])
tn0 = np.array([60, 120, 180]); delamDH = np.array([92, 83, 17]); delamCH = np.array([75, 25, 8])
fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))
a1.plot(xc, SDEG_DH, "o-", color=RED, ms=3, lw=1.0, label="DH (dysbiotic)")
a1.plot(xc, SDEG_CH, "s-", color=BLUE, ms=3, lw=1.0, label="CH (commensal)")
a1.axvline(4.0, ls=":", color="0.6", lw=0.8); a1.text(3.95, 0.05, "free edge", fontsize=6, ha="right")
a1.set_xlabel(r"position $x$ (symmetry $\to$ free edge)"); a1.set_ylabel(r"cohesive damage $D$ (SDEG)")
a1.set_title(r"Interface damage at full growth ($t_n^0\!=\!120$)", fontsize=8)
a1.legend(fontsize=7, loc="center left")
a2.plot(tn0, delamDH, "o-", color=RED, ms=3, lw=1.0, label="DH")
a2.plot(tn0, delamCH, "s-", color=BLUE, ms=3, lw=1.0, label="CH")
a2.set_xlabel(r"interface strength $t_n^0$"); a2.set_ylabel(r"delaminated fraction of $L$ (\%)")
a2.set_title(r"Dysbiosis detaches $\sim$3$\times$ further", fontsize=8)
a2.legend(fontsize=7, loc="upper right")
fig.savefig(OUT / "F6_delamination_DHvsCH.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F8: growth-induced out-of-plane deflection (imperfection-seeded), DH vs CH ----
g = np.array([0, .125, .25, .375, .5, .625, .75, .875, 1.0])          # normalized growth (ramp ends at 0.8 step time)
U3_DH = np.array([0, 0.0331, 0.0737, 0.1134, 0.1521, 0.1899, 0.2265, 0.2621, 0.2968])
U3_CH = np.array([0, 0.0279, 0.0617, 0.0944, 0.1258, 0.1561, 0.1854, 0.2140, 0.2416])
fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.78))
ax.plot(g, U3_DH, "o-", color=RED, ms=3, lw=1.0, label="DH (dysbiotic)")
ax.plot(g, U3_CH, "s-", color=BLUE, ms=3, lw=1.0, label="CH (commensal)")
ax.set_xlabel(r"normalized growth"); ax.set_ylabel(r"out-of-plane deflection $\max|u_3|$")
ax.set_title(r"Imperfection-seeded deflection: DH $23\%$ larger", fontsize=8)
ax.legend(fontsize=7, loc="upper left")
fig.savefig(OUT / "F8_buckling_DHvsCH.pdf", bbox_inches="tight"); plt.close(fig)

print("wrote", *(p.name for p in sorted(OUT.glob("F*.pdf"))))
