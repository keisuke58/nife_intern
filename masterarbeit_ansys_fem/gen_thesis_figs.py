"""Generate ALL thesis FEM figures (F2-F8) in the unified thesis_style (usetex/lmodern/9pt).
Single source of truth for the masterarbeit figure set — consistent RED/BLUE (DH/CH), 9pt.
Run from repo root with TeX Live on PATH:
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python masterarbeit_ansys_fem/gen_thesis_figs.py
Outputs: masterarbeit_ansys_fem/figures/F{2..8}_*.pdf
(F1 = coupling-architecture schematic lives as TikZ in the slides.)
  F2 growth calibration · F3 free-swelling · F4 substratum-interface residual stress (DH vs CH) ·
  F5 socket-free mesh convergence · F6 cohesive delamination · F7 viscoelastic relaxation · F8 deflection.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
CP = ROOT / "masterarbeit_ansys_fem" / "coupling_prototype"   # data CSVs / JSON from the FEM runs
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

# ---- F4: substratum-interface residual stress, DH vs CH (tall column H=4w, the headline) ----
# Socket-free, CLSM-calibrated (iso volume-matched). Stress concentrates in the substratum
# boundary layer (z/H<0.25); the free top relieves the growth -> ~0 above. DH >> CH there.
d = np.genfromtxt(CP / "coltall_DH.csv", delimiter=",", names=True)
c = np.genfromtxt(CP / "coltall_CH.csv", delimiter=",", names=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.52), sharey=True)
a1.plot(d["S11_mu"], d["zref"], "o-", color=RED, ms=2, lw=1.0, label=r"DH (dysbiotic)")
a1.plot(c["S11_mu"], c["zref"], "s--", color=BLUE, ms=2, lw=1.0, label=r"CH (commensal)")
a1.axhspan(0, 0.25, color=RED, alpha=0.07)
a1.text(-2.4, 0.12, r"substratum interface", fontsize=6, color="0.35")
a1.text(-2.4, 0.6, r"free-surface (relieved)", fontsize=6, color="0.45")
a1.set_xlabel(r"residual stress $\sigma_{xx}/\mu$"); a1.set_ylabel(r"height above substratum $z/H$")
a1.legend(fontsize=7, loc="upper left")
a2.plot(d["phi"], d["zref"], "o-", color=RED, ms=2, lw=1.0)
a2.plot(c["phi"], c["zref"], "s--", color=BLUE, ms=2, lw=1.0)
a2.axhspan(0, 0.25, color=RED, alpha=0.07)
a2.set_xlabel(r"composition $\phi_{Pg}(z)$")
fig.suptitle(r"Substratum-interface residual stress: DH $\gg$ CH (peak $6.4\times$, mean $10\times$)",
             fontsize=8, y=0.99)
fig.savefig(OUT / "F4_residual_stress_DHvsCH.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F5: socket-free column mesh convergence (uniform plateau exact; graded interface +0.27%) ----
mu = 5000 / (2 * 1.45)
N = [4, 8, 16, 32]
flat = [abs(-1461.8) / mu] * 4                                   # uniform phi: machine-identical
grad = [abs(v) / mu for v in [-1130.8, -1726.3, -1771.9, -1776.6]]  # graded DH (confined bottom)
fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.78))
ax.plot(N, flat, "s-", color=BLUE, ms=3, lw=1.0, label=r"uniform $\phi_{Pg}$ (plateau)")
ax.plot(N, grad, "o-", color=RED, ms=3, lw=1.0, label=r"graded $\phi_{Pg}(z)$ (interface)")
ax.set_xscale("log", base=2); ax.set_xticks(N); ax.set_xticklabels(N)
ax.set_xlabel(r"elements through depth $N$"); ax.set_ylabel(r"confined $|\sigma_{xx}|/\mu$")
ax.set_title(r"Mesh convergence (uniform exact; graded N=16: $+0.27\%$)", fontsize=8)
ax.legend(fontsize=7, loc="center right")
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

# ---- F7: viscoelastic shear-stress relaxation (Burgers->Prony, FE vs analytic) ----
fe = np.loadtxt(CP / "abaqus" / "s13_relaxation.csv", delimiter=",", skiprows=1)
t, s13 = fe[:, 0], fe[:, 1]; ipk = np.abs(s13).argmax(); norm = s13 / s13[ipk]; tp = t - t[ipk]
P = json.load(open(CP / "prony_biofilm.json")); ginf = 0.05
ssum = sum(q["g_i"] for q in P["prony"])
gi = [p["g_i"] / ssum * (1 - ginf) for p in P["prony"]]; ti = [p["tau_i_s"] for p in P["prony"]]
gfun = lambda x: ginf + sum(gg * np.exp(-x / tau) for gg, tau in zip(gi, ti))  # noqa: E731
tt = np.logspace(-2, np.log10(tp.max()), 400); m = tp > 1e-6
fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.78))
ax.semilogx(tt, gfun(tt), "-", color="0.2", lw=1.0, label=r"analytic Prony $g(t)$")
ax.semilogx(tp[m], norm[m], "o", ms=2.5, mfc="none", color=RED, label=r"Abaqus FE")
ax.axhline(ginf, ls=":", color="0.5", lw=0.8)
ax.set_xlabel(r"relaxation time $t-t_0$ [s]"); ax.set_ylabel(r"normalized shear stress")
ax.set_title(r"Biofilm stress relaxation (Burgers$\to$Prony, RMS $9{\times}10^{-4}$)", fontsize=8)
ax.legend(fontsize=7, loc="lower left"); ax.set_ylim(0, 1.05)
fig.savefig(OUT / "F7_viscoelastic_relaxation.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F9: bilayer growth-induced wrinkling — DH wrinkles (short lambda), CH only bends ----
AB = CP / "abaqus"
def _wr(p):
    a = np.loadtxt(p); return a[:, 0] / 0.8, a[:, 1]            # frac/0.8 = normalized growth, amplitude
gDH, aDH = _wr(AB / "film_bilayer_wrinkle.txt")
gCH, aCH = _wr(AB / "film_bilayer_ch_wrinkle.txt")
pDH = np.genfromtxt(AB / "film_bilayer_topprofile.csv", delimiter=",", names=True)
pCH = np.genfromtxt(AB / "film_bilayer_ch_topprofile.csv", delimiter=",", names=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))
mDH, mCH = gDH <= 1.0, gCH <= 1.0
a1.plot(gDH[mDH], aDH[mDH], "o-", color=RED, ms=2, lw=1.0, label="DH (dysbiotic)")
a1.plot(gCH[mCH], aCH[mCH], "s-", color=BLUE, ms=2, lw=1.0, label="CH (commensal)")
a1.axvspan(0.7, 0.85, color=RED, alpha=0.06); a1.text(0.72, 0.02, "onset", fontsize=6, color="0.35")
a1.set_xlabel(r"normalized growth"); a1.set_ylabel(r"surface amplitude")
a1.set_title(r"Wrinkle amplitude: super-linear onset (DH)", fontsize=8)
a1.legend(fontsize=7, loc="upper left")
a2.plot(pDH["x"], pDH["u3"], "-", color=RED, lw=1.0, label=r"DH: $\lambda\!\approx\!1.8$ (mode 9)")
a2.plot(pCH["x"], pCH["u3"], "--", color=BLUE, lw=1.0, label=r"CH: $\lambda\!=\!16$ (mode 1, bend)")
a2.set_xlabel(r"position $x$"); a2.set_ylabel(r"surface deflection $u_3$")
a2.set_title(r"Wavelength selection: DH wrinkles, CH only bends", fontsize=8)
a2.legend(fontsize=7, loc="upper right")
fig.savefig(OUT / "F9_wrinkling_DHvsCH.pdf", bbox_inches="tight"); plt.close(fig)

# ---- F10: patient-specific detachment risk — 10 Dieckow patients on the mechanics risk curve ----
# Risk curve = cohesive-delamination fraction vs growth driver phi_Pg (clean clinical low-phi branch).
phi_curve = np.array([0.0, 0.06, 0.10, 0.14, 0.18])
risk_curve = np.array([0.0, 0.0, 8.0, 33.0, 33.0])                 # delaminated % (monotone branch)
pg_pat = np.array([0.001, 0.002, 0.010, 0.028, 0.050, 0.058, 0.062, 0.062, 0.079, 0.122])  # Dieckow Bacteroidia(Pg)
risk_pat = np.interp(pg_pat, phi_curve, risk_curve)
fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))
xx = np.linspace(0, 0.18, 100)
a1.plot(xx, np.interp(xx, phi_curve, risk_curve), "-", color="0.4", lw=1.2)
a1.axvspan(0.10, 0.14, color=RED, alpha=0.07); a1.text(0.105, 2, "onset", fontsize=6, color="0.35")
a1.scatter(pg_pat, risk_pat, color=RED, zorder=3, s=20)
a1.set_xlabel(r"patient \textit{P.\,gingivalis} load $\phi_{Pg}$"); a1.set_ylabel(r"predicted detachment risk (\%)")
a1.set_title(r"Mechanics risk curve $+$ 10 Dieckow patients", fontsize=8)
order = np.argsort(pg_pat)
a2.barh(range(10), risk_pat[order], color=[RED if r > 10 else BLUE for r in risk_pat[order]])
a2.set_yticks(range(10)); a2.set_yticklabels([r"%.3f" % v for v in pg_pat[order]], fontsize=6)
a2.set_xlabel(r"detachment risk (\%)"); a2.set_ylabel(r"patient $\phi_{Pg}$ (sorted)")
a2.set_title(r"Risk stratification: one high-Pg patient at risk", fontsize=8)
fig.savefig(OUT / "F10_patient_risk.pdf", bbox_inches="tight"); plt.close(fig)

print("wrote", *(p.name for p in sorted(OUT.glob("F*.pdf"))))
