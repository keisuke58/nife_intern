"""Tier-2 (a) visualisation: implant <-> bone <-> adjacent tooth coupling under occlusal load.

Reads the parametric alveolar-bone FEM extract (coupling_prototype/abaqus/tier2_bone_3d.json, step 2 =
dysbiotic growth + occlusal load) and shows a mid-buccolingual (y = Ly/2) section in the x-z plane,
coloured by von Mises stress. The occlusal load on both crowns is transmitted through the SHARED bone,
so the implant and the adjacent tooth are mechanically coupled -- the Tier-2 effect Tier-1 omits.

Run:  python masterarbeit_ansys_fem/extensions/fig_tier2_bone.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

AB = ROOT / "masterarbeit_ansys_fem" / "coupling_prototype" / "abaqus"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
IMP, TOO = (5.0, 5.0), (13.0, 5.0)
R_IMP, R_ROOT, R_PDL, Z_CREST = 2.0, 2.5, 3.0, 10.0


def main():
    use()
    e = json.load(open(AB / "tier2_bone_3d.json"))["els"]
    x = np.array([r["x"] for r in e]); y = np.array([r["y"] for r in e])
    z = np.array([r["z"] for r in e]); vm = np.array([r["vm"] for r in e])
    m = np.abs(y - 5.0) < 0.75                         # mid buccolingual slice
    vmax = np.quantile(vm[m], 0.97)

    fig, ax = plt.subplots(figsize=use(width_frac=0.7, aspect=0.78))
    sc = ax.scatter(x[m], z[m], c=vm[m], cmap="inferno", s=12, vmin=0, vmax=vmax, marker="s")
    # region outlines
    for (cx, _), r, col, lab, zb in ((IMP, R_IMP, "#4da6ff", "implant (Ti)", 0.0),
                                     (TOO, R_ROOT, "#7CFC00", "tooth root", 3.0)):
        ax.add_patch(plt.Rectangle((cx - r, zb), 2 * r, 13.0 - zb, fill=False, edgecolor=col, lw=1.0, ls="--"))
        ax.text(cx, 13.4, lab, ha="center", fontsize=6.5, color=col)
    ax.add_patch(plt.Rectangle((TOO[0] - R_PDL, 3.0), 2 * R_PDL, Z_CREST - 3.0, fill=False,
                               edgecolor="#ff8c00", lw=0.8, ls=":"))
    ax.text(TOO[0] + R_PDL + 0.3, 6, "PDL", fontsize=5.5, color="#ff8c00")
    ax.axhline(Z_CREST, color="0.5", lw=0.7, ls="-")
    ax.text(0.3, Z_CREST + 0.2, "bone crest", fontsize=5.5, color="0.4")
    ax.text(0.3, 4, "alveolar bone", fontsize=6, color="0.85")
    ax.set_xlabel(r"mesiodistal $x$ (mm)"); ax.set_ylabel(r"depth $z$ (mm)")
    ax.set_title(r"Tier-2: implant $\leftrightarrow$ bone $\leftrightarrow$ tooth coupling (occlusal load)", fontsize=7.5)
    ax.set_xlim(0, 18); ax.set_ylim(0, 14); ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"von Mises stress (MPa)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_tier2_bone.pdf", bbox_inches="tight")
    plt.close(fig)
    # report coupling: peak vM in bone between implant and tooth
    mid = m & (x > 7) & (x < 11) & (z < Z_CREST)
    print("slice elems=%d  bone-bridge (7<x<11) peak vM=%.1f MPa  overall slice peak=%.1f"
          % (m.sum(), vm[mid].max() if mid.any() else 0, vm[m].max()))
    print("wrote", OUT / "fem_tier2_bone.pdf")


if __name__ == "__main__":
    main()
