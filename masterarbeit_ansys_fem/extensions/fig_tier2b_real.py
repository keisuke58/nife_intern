"""Tier-2(b) figure: FULL real-shape coupled implant + tooth + alveolar-bone FEM (Open-Full-Jaw P1).

A single conforming/TIE-coupled model built from the REAL mandible, REAL tooth-24 and a root-form
titanium implant at the (extracted) tooth-23 site, with a conforming PDL layer and dysbiotic biofilm
collars.  Mesiodistal (y = -41) section through both columns:
  (A) the real multi-material anatomy (bone / implant-Ti / dentin / PDL / biofilm);
  (B) occlusal-load von Mises -- load is transmitted through the SHARED bone to both columns; the
      osseointegrated implant concentrates stress at the crestal bone while the PDL-supported tooth
      distributes it along the root (the peri-implant marginal-bone-loss signature).

Run:  python masterarbeit_ansys_fem/extensions/fig_tier2b_real.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

FIELD = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real/tier2b_real_field.json")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
MAT_ORDER = ["BONE", "PDL", "DENTIN", "TI", "BIOFILM"]
MAT_COL = {"BONE": "#e7d8b0", "PDL": "#ff8c00", "DENTIN": "#9ecae1",
           "TI": "#636363", "BIOFILM": "#d62728"}
T23, T24 = -69.4, -63.9   # mesiodistal x of implant / tooth axes


def main():
    use()
    d = json.load(open(FIELD))["els"]
    x = np.array([r["x"] for r in d]); y = np.array([r["y"] for r in d])
    z = np.array([r["z"] for r in d]); mat = np.array([r["mat"] for r in d])
    vmo = np.array([r["vmo"] for r in d])
    m = np.abs(y - (-41.0)) < 0.9                      # mesiodistal slice through both axes

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.5))

    # ---- Panel A: real multi-material anatomy ----
    code = np.array([MAT_ORDER.index(mm) if mm in MAT_ORDER else 0 for mm in mat])
    cmap = ListedColormap([MAT_COL[k] for k in MAT_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(MAT_ORDER)), cmap.N)
    axA.scatter(x[m], z[m], c=code[m], cmap=cmap, norm=norm, s=6, marker="s", linewidths=0)
    axA.set_title(r"(A) real anatomy: implant $+$ tooth $+$ bone", fontsize=7.5)
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=MAT_COL[k], mec="none",
                          label={"TI": "implant (Ti)", "DENTIN": "tooth (dentin)"}.get(k, k.title()))
               for k in MAT_ORDER]
    axA.legend(handles=handles, fontsize=5.3, loc="lower center", ncol=3,
               handletextpad=0.2, columnspacing=0.7, framealpha=0.9)

    # ---- Panel B: occlusal von Mises ----
    vclip = 8.0
    sc = axB.scatter(x[m], z[m], c=np.clip(vmo[m], 0, vclip), cmap="inferno",
                     s=6, marker="s", vmin=0, vmax=vclip, linewidths=0)
    axB.set_title(r"(B) occlusal load: shared-bone coupling", fontsize=7.5)
    cb = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.03)
    cb.set_label(r"von Mises $\sigma_\mathrm{vM}$ (MPa)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)

    for ax in (axA, axB):
        ax.axvline(T23, color="0.4", lw=0.5, ls=":")
        ax.axvline(T24, color="0.4", lw=0.5, ls=":")
        ax.set_xlabel(r"mesiodistal $x$ (mm)", fontsize=6.5)
        ax.set_xlim(-74, -60); ax.set_ylim(16.5, 31.5); ax.set_aspect("equal")
        ax.tick_params(labelsize=5.5)
    axB.annotate("implant\n(no PDL)", xy=(T23, 17.2), ha="center", fontsize=5.4, color="w")
    axB.annotate("tooth 24\n(PDL)", xy=(T24, 17.2), ha="center", fontsize=5.4, color="w")
    axA.set_ylabel(r"depth $z$ (mm)", fontsize=6.5)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_tier2b_real.pdf", bbox_inches="tight")
    plt.close(fig)
    print("slice elems=%d  wrote %s" % (m.sum(), OUT / "fem_tier2b_real.pdf"))


if __name__ == "__main__":
    main()
