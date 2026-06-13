"""F1b — Data-provenance bridge: patient 16S (composition, 0D) -> depth assumption
(in-vitro HOBIC CLSM) -> FEM residual stress.

Makes explicit the LAYER SEPARATION that the thesis must state honestly:
  * Panel A  real PATIENT data (Dieckow, peri-implant 16S) is COMPOSITION-ONLY (0D, no depth).
  * Panel B  the depth SHAPE phi_Pg(z) comes from the in-vitro HOBIC CLSM/FISH biofilm,
             not from the patient -> the bridge assumption: shape (CLSM) x load (patient).
  * Panel C  the FEM material model turns phi_Pg(z) into a residual-stress column S11(z).

Real inputs, all already in the repo:
  results/dieckow_otu/phi_guild.npy                 (10 patients x 3 wk x 10 guilds; Pg = Bacteroidia)
  results/diffusion_fit/zprofiles_all_ti.csv        (HOBIC DH day-21 phi_Pg(z), corrected Fn=blue cap red decode)
  masterarbeit_ansys_fem/coupling_prototype/coltall_DH.csv  (verified growth-column S11(z)/mu)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
BLUE, RED, GREY = "#1f77b4", "#d62728", "#555555"

# ---- A) real patient composition (0D): Dieckow Bacteroidia (Pg-containing guild), week 3 ----
phi_g = np.load(ROOT / "results/dieckow_otu/phi_guild.npy")   # (10,3,10)
pg_patient = phi_g[:, -1, 2]                                   # guild idx 2 = Bacteroidia
load_repr = float(pg_patient.mean())                          # representative patient Pg load

# ---- B) in-vitro depth shape phi_Pg(z): HOBIC DH, day 21 (corrected decode) ----
zp = pd.read_csv(ROOT / "results/diffusion_fit/zprofiles_all_ti.csv")
dh = zp[(zp.condition == "DH") & (zp.day == 21)].sort_values("z_um")
z_um = dh.z_um.values
zN = (z_um - z_um.min()) / (z_um.max() - z_um.min())
phi_z = dh["P.gingivalis"].values
shape = phi_z / phi_z.mean()                                  # unit-mean depth template
phi_patient_z = load_repr * shape                            # patient load x CLSM shape

# ---- C) FEM residual-stress column S11(z)/mu (verified growth column) ----
col = pd.read_csv(ROOT / "masterarbeit_ansys_fem/coupling_prototype/coltall_DH.csv")

# ---------------------------------------------------------------- figure ----
fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.40))
gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.08, right=0.97, bottom=0.20, top=0.86)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

# A: patient composition, 0D ------------------------------------------------
order = np.argsort(pg_patient)[::-1]
axA.bar(np.arange(10), pg_patient[order] * 100, color=BLUE, width=0.72, edgecolor="none")
axA.axhline(load_repr * 100, color=GREY, ls="--", lw=0.8)
axA.set_xticks([]); axA.set_xlabel(r"10 peri-implant patients")
axA.set_ylabel(r"Pg-guild abundance [\%]")
axA.set_title(r"\textbf{A}\quad composition (0D)", loc="left")
axA.text(0.97, 0.93, r"real patient 16S" + "\n" + r"\emph{no depth}",
         transform=axA.transAxes, ha="right", va="top", fontsize=7, color=GREY)

# B: depth shape from in-vitro CLSM ----------------------------------------
axB.plot(phi_z * 100, zN, "-", color=RED, lw=1.6, label=r"HOBIC CLSM shape")
axB.plot(phi_patient_z * 100, zN, "--", color=BLUE, lw=1.4,
         label=r"patient load $\times$ shape")
axB.invert_yaxis()
axB.set_xlabel(r"$\varphi_{Pg}(z)$ [\%]")
axB.set_ylabel(r"depth $z/h$  (0 = substratum)")
axB.set_title(r"\textbf{B}\quad depth assumption", loc="left")
axB.legend(loc="lower right", frameon=False, fontsize=6.5, handlelength=1.4)

# C: FEM residual stress ----------------------------------------------------
axC.plot(col.S11_mu, col.zref, "-", color="#222222", lw=1.6)
axC.axvline(0, color=GREY, lw=0.6, ls=":")
axC.invert_yaxis()
axC.set_xlabel(r"$S_{11}/\mu$  (residual)")
axC.set_ylabel(r"depth $z/h$")
axC.set_title(r"\textbf{C}\quad FEM mechanics", loc="left")
axC.text(0.05, 0.10, r"compressive" + "\n" + r"near substratum",
         transform=axC.transAxes, ha="left", va="bottom", fontsize=7, color=GREY)

# bridge arrows between panels (figure coords) ------------------------------
for x, lab in ((0.345, r"CLSM shape"), (0.655, r"$F\!=\!F_eF_g$")):
    fig.text(x, 0.95, r"$\Rightarrow$", fontsize=15, ha="center", va="center", color=GREY)
    fig.text(x, 0.90, lab, fontsize=6.5, ha="center", va="top", color=GREY)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "F1b_data_bridge.pdf", bbox_inches="tight")
fig.savefig(OUT / "F1b_data_bridge.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "F1b_data_bridge.pdf")
print(f"  patient Pg-guild load (mean) = {load_repr*100:.1f}%   "
      f"HOBIC phi_Pg(z) mean = {phi_z.mean()*100:.1f}%   "
      f"S11/mu in [{col.S11_mu.min():.2f}, {col.S11_mu.max():.2f}]")
