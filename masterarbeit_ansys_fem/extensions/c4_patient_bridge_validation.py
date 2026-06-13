"""C4 — Patient-bridge validation: project the REAL Dieckow cohort through the verified
mechanics and ask whether the load x shape bridge assumption is load-bearing.

The bridge (Fig F1b) assumes a patient's depth profile is phi_Pg(z) = (patient Pg LOAD)
x (in-vitro CLSM depth SHAPE). The load is measured per patient (Dieckow 16S, 0D); the
shape is borrowed from the in-vitro HOBIC biofilm and is NOT measured per patient. So the
honest question is: how much does the borrowed shape matter?

Verified reduced model: only near-substratum growth builds interface stress, so the
detachment driver is  J = mean_z exp(-z/z0) phi_Pg(z),  z0 = 0.25  (same kernel as c3,
where S11(z) vs phi is depth-dominated, NOT a local-phi law: polyfit R^2 = 0.03).

Because phi_Pg(z) = load x shape and the kernel is linear,  J_patient = load x J(shape).
=> two levers:
  * LOAD  (measured, per patient)  -> the inter-patient spread.
  * SHAPE (borrowed, in-vitro)     -> a multiplicative factor; its uncertainty is bounded
    by the two REAL measured shapes (CH vs DH), both unit-mean normalised.

Real inputs (all in repo):
  results/dieckow_otu/phi_guild.npy            (10 patients x 3 wk x 10 guilds; Pg=Bacteroidia idx2)
  results/diffusion_fit/zprofiles_all_ti.csv   (in-vitro CH/DH day-21 depth shapes, corrected decode)
Output: figures/F1c_patient_bridge_validation.pdf + printed numbers for ch5 sec.6.
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
Z0 = 0.25  # verified boundary-layer depth scale (interface-stress zone), same as c3

# ---- patient Pg load (Dieckow, Bacteroidia guild, week 3) -----------------
loads = np.load(ROOT / "results/dieckow_otu/phi_guild.npy")[:, -1, 2]   # (10,)

# ---- two REAL measured depth shapes (unit-mean), in-vitro day 21 ----------
zp = pd.read_csv(ROOT / "results/diffusion_fit/zprofiles_all_ti.csv")
def unit_shape(cond):
    s = zp[(zp.condition == cond) & (zp.day == 21)].sort_values("z_um")
    z = s.z_um.values
    zn = (z - z.min()) / (z.max() - z.min())
    pg = s["P.gingivalis"].values
    return zn, pg / pg.mean()
zn, shD = unit_shape("DH")
_,  shC = unit_shape("CH")

def J_of_shape(sh):                       # detachment kernel on a unit-mean shape
    return float(np.mean(np.exp(-zn / Z0) * sh))
JD, JC = J_of_shape(shD), J_of_shape(shC)   # shape factors (DH borrowed; CH bound)

# ---- per-patient detachment driver under the bridge -----------------------
J_patient = loads * JD                                   # load x (DH shape)
J_lo = loads * min(JC, JD)                               # shape-uncertainty band
J_hi = loads * max(JC, JD)
shape_lever = abs(JD / JC - 1.0)                         # how much the borrowed shape moves J
cv_load = J_patient.std() / J_patient.mean()             # inter-patient spread (load-driven)

print(f"shape factors:  DH={JD:.3f}  CH={JC:.3f}  -> shape lever = {shape_lever*100:.0f}%")
print(f"per-patient J:  min={J_patient.min():.4f}  max={J_patient.max():.4f}  CV={cv_load:.2f}")
print(f"inter-patient range / shape-band width = "
      f"{(J_patient.max()-J_patient.min()) / (J_hi-J_lo).mean():.1f}x")

# ------------------------------------------------------------------- figure ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.42))
fig.subplots_adjust(wspace=0.34, left=0.09, right=0.97, bottom=0.20, top=0.88)

# A: per-patient detachment driver, sorted, with shape-uncertainty error bars
order = np.argsort(J_patient)[::-1]
x = np.arange(10)
yerr = np.vstack([(J_patient - J_lo)[order], (J_hi - J_patient)[order]])
axA.bar(x, J_patient[order], color=BLUE, width=0.7, edgecolor="none", zorder=2)
axA.errorbar(x, J_patient[order], yerr=yerr, fmt="none", ecolor=GREY,
             elinewidth=0.9, capsize=2, zorder=3, label=r"shape band (CH$\leftrightarrow$DH)")
axA.set_xticks([]); axA.set_xlabel(r"10 Dieckow patients (sorted)")
axA.set_ylabel(r"detachment driver $J$")
axA.set_title(r"\textbf{A}\quad per-patient mechanical risk", loc="left")
axA.legend(loc="upper right", frameon=False, fontsize=6.5)
axA.text(0.5, 0.55, r"inter-patient spread CV $=%.2f$" % cv_load + "\n"
         + r"(set by measured Pg load)", transform=axA.transAxes,
         fontsize=7, color=GREY, ha="center")

# B: J vs load — the linear bridge law, shape uncertainty as a thin wedge
ls = np.linspace(0, loads.max() * 1.05, 50)
axB.fill_between(ls * 100, ls * min(JC, JD), ls * max(JC, JD),
                 color=GREY, alpha=0.25, lw=0, label=r"shape band ($\pm%.0f\%%$)" % (shape_lever * 100))
axB.plot(ls * 100, ls * JD, "-", color=GREY, lw=0.8)
axB.plot(loads * 100, J_patient, "o", color=BLUE, ms=5, zorder=3, label=r"patient")
axB.set_xlabel(r"measured Pg-guild load [\%]")
axB.set_ylabel(r"detachment driver $J$")
axB.set_title(r"\textbf{B}\quad load is the lever, not shape", loc="left")
axB.legend(loc="upper left", frameon=False, fontsize=6.5)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "F1c_patient_bridge_validation.pdf", bbox_inches="tight")
fig.savefig(OUT / "F1c_patient_bridge_validation.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "F1c_patient_bridge_validation.pdf")
