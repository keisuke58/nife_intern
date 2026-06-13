"""F1d / fem_periimplant — re-frame the verified growth-column result in the correct
PERI-IMPLANT geometry (NIFE = implant research; Dieckow = abutment biofilm; substratum = Ti).

No new mechanics: the substratum-bonded, free-topped growth column of Section fem_residual
IS the radial idealisation of a biofilm film on the titanium abutment wall inside the
peri-implant sulcus. This figure places the VERIFIED residual-stress column (coltall_DH,
S11(z)/mu) onto an idealised implant cross-section, so the compressive interface stress
sits where it physically acts: the biofilm--abutment interface, whose failure (detachment)
seeds peri-implantitis.

Inputs (verified, already in repo):
  coupling_prototype/coltall_DH.csv  (zref = radial coord from Ti wall; S11_mu = S11/mu)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
TI, BONE, MUCOSA, GREY = "#9aa3ad", "#e9dcc3", "#e8b7b0", "#555555"

col = pd.read_csv(ROOT / "masterarbeit_ansys_fem/coupling_prototype/coltall_DH.csv")
zr = col.zref.values            # 0 = Ti abutment wall (substratum), 1 = sulcus (free)
s11 = col.S11_mu.values         # verified residual stress / mu

# ---------------------------------------------------------------- figure ----
fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.52))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.30,
                      left=0.02, right=0.95, bottom=0.13, top=0.92)
axS = fig.add_subplot(gs[0, 0]); axS.set_axis_off()
axS.set_xlim(0, 10); axS.set_ylim(0, 11); axS.set_aspect("equal")
axC = fig.add_subplot(gs[0, 1])

# --- A: idealised peri-implant cross-section (axisymmetric, centreline at x=5) --
cx, crest = 5.0, 4.6                      # implant centreline; alveolar crest height
# bone: one full-width block below the crest; the fixture is drawn on top of it
axS.add_patch(Rectangle((0.4, 0.4), 9.2, crest-0.4, facecolor=BONE,
                        edgecolor="0.6", lw=0.6, zorder=1))
axS.text(1.7, 2.3, r"bone", fontsize=7, color="0.45", ha="center")
axS.text(8.3, 2.3, r"bone", fontsize=7, color="0.45", ha="center")
# implant fixture: clean tapered screw body (solid), with simple thread notches
fix = Polygon([(cx-0.85, crest), (cx+0.85, crest), (cx+0.30, 0.6), (cx-0.30, 0.6)],
              closed=True, facecolor=TI, edgecolor="0.35", lw=0.8, zorder=3)
axS.add_patch(fix)
for y in np.linspace(0.9, crest-0.35, 7):              # thread marks on both flanks
    w = 0.30 + (0.55) * (y-0.6)/(crest-0.6)
    axS.plot([cx-w, cx-w+0.16], [y, y-0.18], color="0.35", lw=0.5, zorder=4)
    axS.plot([cx+w, cx+w-0.16], [y, y-0.18], color="0.35", lw=0.5, zorder=4)
# abutment: smooth collar above the crest
axS.add_patch(Polygon([(cx-0.55, crest), (cx+0.55, crest), (cx+0.42, 8.4),
                       (cx-0.42, 8.4)], closed=True, facecolor=TI,
                      edgecolor="0.35", lw=0.8, zorder=3))
axS.text(cx, 2.2, r"Ti implant", fontsize=6.5, color="0.12", ha="center",
         rotation=90, zorder=5)
axS.annotate(r"abutment", xy=(cx-0.48, 7.2), xytext=(1.5, 8.0), fontsize=7,
             color="0.2", arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"))
# mucosa cuff, leaving a sulcus gap against the abutment near the top
for side in (-1, 1):
    xin = cx + side*0.62
    axS.add_patch(Polygon([(xin, crest), (cx+side*3.8, crest), (cx+side*3.8, 8.7),
                           (cx+side*1.5, 8.7), (xin, 7.0)], closed=True,
                          facecolor=MUCOSA, edgecolor="0.7", lw=0.5, zorder=2))
axS.text(8.4, 6.6, r"mucosa", fontsize=7, color="0.45", ha="center")
axS.annotate(r"peri-implant sulcus", xy=(cx+0.75, 7.6), xytext=(7.9, 9.4),
             fontsize=7, color="0.25", ha="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"))

# --- the verified biofilm column, mapped onto the abutment wall in the sulcus ---
# a thin sleeve hugging the abutment wall (both sides), coloured by S11/mu across its
# radial thickness (wall = zref 0 -> sulcus lumen = zref 1).
ny = 60
ys = np.linspace(7.05, 8.55, ny)         # sulcus extent along the wall
zr_grid = np.linspace(0, 1, 24)
S = np.interp(zr_grid, zr, s11)[None, :].repeat(ny, 0)   # (ny, nz)
sleeve_t = 0.5
for side in (-1, 1):
    x0 = cx + side*0.42
    X = x0 + side*np.linspace(0, sleeve_t, 24)[None, :].repeat(ny, 0)
    Y = ys[:, None].repeat(24, 1)
    axS.pcolormesh(X, Y, S, cmap="RdBu_r", vmin=-5, vmax=5, shading="gouraud", zorder=5)
axS.annotate(r"biofilm (stress-coloured)", xy=(cx-0.7, 7.9), xytext=(1.4, 9.4),
             fontsize=7, color="0.2", ha="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"))

# --- B: the verified residual-stress column, re-labelled for the implant wall ---
axC.fill_betweenx(zr, s11, 0, where=(s11 <= 0), color="#4575b4", alpha=0.35, lw=0)
axC.plot(s11, zr, "-", color="#222222", lw=1.6, zorder=3)
axC.axvline(0, color=GREY, ls=":", lw=0.7)
axC.set_ylim(1, 0)                                   # 0 = wall at top
axC.set_xlabel(r"residual stress $S_{11}/\mu$")
axC.set_ylabel(r"radial distance from abutment wall")
axC.set_title(r"\textbf{B}\quad verified interface-stress column", loc="left", fontsize=9)
axC.text(0.04, 0.06, r"Ti abutment wall" + "\n" + r"(substratum): compression",
         transform=axC.transAxes, fontsize=6.8, color="#4575b4", va="bottom")
axC.text(0.96, 0.93, r"sulcus (free): $\sigma\!\approx\!0$", transform=axC.transAxes,
         fontsize=6.8, color="0.4", ha="right", va="top")
axS.text(0.5, 0.97, r"\textbf{A}\quad idealised peri-implant section", transform=axS.transAxes,
         fontsize=9, ha="center", va="top")

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"F1d_periimplant_schematic.{ext}",
                dpi=200 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "F1d_periimplant_schematic.pdf")
print(f"  mapped verified column: S11/mu in [{s11.min():.2f},{s11.max():.2f}] "
      f"(compression at Ti wall, ~0 at sulcus)")
