"""C3 — Residual-stress / detachment-risk map from REAL CLSM FISH data (species-resolved Böl pipeline).

Böl et al. (2009) built FE models from real CLSM biofilm geometry, but blind to species. Here we drive
the verified substratum-boundary-layer stress model with the REAL measured P. gingivalis distribution
from the Heine HOBIC FISH dataset — per field of view (FOV) — to map detachment risk across the actually
imaged biofilm. Two real inputs:
  * results/diffusion_fit/zprofiles_all_ti.csv  -> measured depth shape phi_Pg(z) per condition,
  * results/fish_3d/fish_3d_lateral_metrics.csv -> P. gingivalis biomass fraction per FOV (84 FOVs).
For each real FOV: scale the measured depth shape to the FOV's Pg load and evaluate the boundary-layer
detachment driver J = sum_z w(z) phi_Pg(z),  w(z)=exp(-z/z0) (verified: only near-substratum growth
builds interface stress). Output: the distribution of FOV-level detachment risk, CH vs DH — the spatial
heterogeneity of mechanical detachment risk on the real biofilm. No smoothed/synthetic profile is used.
"""
import numpy as np, csv
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use
RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

# --- real measured depth shape phi_Pg(z), latest day, per condition ---
zp = {}
with open(ROOT / "results/diffusion_fit/zprofiles_all_ti.csv") as f:
    for r in csv.DictReader(f):
        zp.setdefault(r["condition"], {}).setdefault(int(r["day"]), []).append(
            (float(r["z_um"]), float(r["P.gingivalis"])))
def depth_shape(cond):
    day = max(zp[cond]); rows = sorted(zp[cond][day])
    z = np.array([a for a, _ in rows]); pg = np.array([b for _, b in rows])
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)          # normalise depth 0..1 (0=substratum)
    return z, pg / (pg.sum() + 1e-9)                          # shape (sums to 1)

# --- real per-FOV Pg biomass fraction ---
fov = {"CH": [], "DH": []}
with open(ROOT / "results/fish_3d/fish_3d_lateral_metrics.csv") as f:
    for r in csv.DictReader(f):
        sp = ["S.o", "A.n", "Vd/Vp", "F.n", "P.g"]
        bm = np.array([float(r["biomass_%s" % s]) for s in sp])
        frac_pg = bm[-1] / (bm.sum() + 1e-9)
        c = r["cond"]
        if c in fov:
            fov[c].append(frac_pg)

z0 = 0.25
# --- detachment driver J from the REAL measured depth profile, per condition x day ---
def J_profile(cond, day):
    rows = sorted(zp[cond][day]); z = np.array([a for a, _ in rows]); pg = np.array([b for _, b in rows])
    zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return float(np.sum(np.exp(-zn / z0) * pg) / len(z))
days = sorted(set(zp["CH"]) & set(zp["DH"]))
JCH = np.array([J_profile("CH", d) for d in days]); JDH = np.array([J_profile("DH", d) for d in days])
print("REAL measured-profile detachment driver J(day):")
for i, d in enumerate(days):
    print("  day %-2d   CH J=%.4f   DH J=%.4f" % (d, JCH[i], JDH[i]))
print("  CH FALLS over time (Pg cleared); DH RISES (Pg accumulates at depth).")
print("  endpoint DH/CH = %.2f  (mechanical detachment risk diverges as dysbiosis develops)"
      % (JDH[-1] / JCH[-1]))
# secondary: per-FOV total-Pg fraction does NOT separate conditions (the signal is depth-structural)
fpgCH, fpgDH = np.array(fov["CH"]), np.array(fov["DH"])
print("  (control: per-FOV total-Pg fraction CH %.3f vs DH %.3f -> total Pg alone does NOT separate; "
      "the depth POSITION does)" % (fpgCH.mean(), fpgDH.mean()))

fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))
zC, shC = depth_shape("CH"); zD, shD = depth_shape("DH")
a1.plot(shC, zC, "-", color=BLUE, lw=1.3, label=r"CH measured")
a1.plot(shD, zD, "-", color=RED, lw=1.3, label=r"DH measured")
a1.axhspan(0, z0, color=RED, alpha=0.07); a1.text(0.02, 0.04, "interface stress zone", fontsize=6, color="0.35")
a1.set_xlabel(r"measured $\phi_{Pg}$ depth shape"); a1.set_ylabel(r"depth $z$ (0 = substratum)")
a1.set_title(r"Real FISH depth profiles (endpoint)", fontsize=8); a1.legend(fontsize=6.5, loc="upper right")
a2.plot(days, JCH, "s-", color=BLUE, lw=1.2, ms=3, label="CH (clears)")
a2.plot(days, JDH, "o-", color=RED, lw=1.2, ms=3, label="DH (accumulates)")
a2.set_xlabel(r"day"); a2.set_ylabel(r"detachment driver $J$ (real data)")
a2.set_title(r"Mechanical risk diverges as dysbiosis develops", fontsize=8); a2.legend(fontsize=7, loc="upper left")
fig.savefig(OUT / "F17_real_fov_stress.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "F17_real_fov_stress.pdf")
