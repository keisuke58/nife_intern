"""Side-by-side: BARE screw+abutment (load at the platform, moment arm ~0) vs RESTORED implant
(real-tooth ceramic crown + gingiva, load at the occlusal table -> crown-height moment arm).  Both
Panel-A anatomy views; the restored case is where the crestal peri-implant bone peak rises x1.6.

Reuses the renderers from fig_implant_crown_fem.py.  Run:
  python masterarbeit_ansys_fem/extensions/fig_implant_bare_vs_restored.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EXT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXT))
import fig_implant_crown_fem as cf  # noqa: E402  (boundary_faces, shade, style3d, label3d, overlays)

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
FEMDIR = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"


def panel(ax, meta, title, restored):
    a = np.load(meta, allow_pickle=True)
    nodes = a["nodes"]; conn = a["conn"]; mat = a["mat"]
    b = (nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max(),
         nodes[:, 2].min(), nodes[:, 2].max())
    ymid = (b[2] + b[3]) / 2
    # bone (back half only)
    bidx = np.where(np.isin(mat, cf.BONE_MATS))[0]
    by = nodes[conn[bidx]][:, :, 1].mean(axis=1)
    fbB, _ = cf.boundary_faces(conn[bidx[by >= ymid]])
    if len(fbB) > 12000:
        fbB = fbB[np.random.default_rng(0).choice(len(fbB), 12000, replace=False)]
    surfB = nodes[fbB]
    ax.add_collection3d(Poly3DCollection(surfB, facecolors=cf.shade(surfB, to_rgb(cf.COL["BONE"])),
                                         edgecolors="none", alpha=0.26, rasterized=True))
    for M in ["DENTIN", "PDL", "TI", "BIOFILM", "CROWN"]:
        idx = np.where(mat == M)[0]
        if not len(idx) or (M == "CROWN" and restored):     # restored crown drawn as real-tooth overlay
            continue
        fb, _ = cf.boundary_faces(conn[idx])
        tris = nodes[fb]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=cf.shade(tris, to_rgb(cf.COL[M]), *cf.SH[M]),
                                             edgecolors=(0, 0, 0, 0.10), linewidths=0.05, rasterized=True))
    if restored:
        cr_top = nodes[conn[mat == "CROWN"]].reshape(-1, 3)[:, 2].max()
        gum = cf.gingiva_cuff()
        ax.add_collection3d(Poly3DCollection(gum, facecolors=cf.shade(gum, to_rgb(cf.COL["GINGIVA"]), 0.6, 0.35),
                                             edgecolors=(0.6, 0.2, 0.3, 0.25), linewidths=0.05,
                                             alpha=0.55, rasterized=True))
        cm = cf.real_tooth_crown(cf.STL_TOOTH, h_target=cr_top - 31.0, base_z=31.0)
        ax.add_collection3d(Poly3DCollection(cm, facecolors=cf.shade(cm, to_rgb(cf.COL["CROWN"]), 0.62, 0.34, 0.30),
                                             edgecolors=(0, 0, 0, 0.12), linewidths=0.04, rasterized=True))
        ztop = cr_top + 3.0
    else:
        ztop = nodes[conn[mat == "TI"]].reshape(-1, 3)[:, 2].max() + 2.6
    yf = b[2] + 0.18 * (b[3] - b[2])
    ax.text(cf.T23[0], yf, ztop, r"$\downarrow$ 100 N, 30$^\circ$", fontsize=5.4, ha="center",
            color="#b30000", fontweight="bold")
    ax.set_title(title, fontsize=6.8, pad=1)
    cf.style3d(ax, b)


def main():
    cf.use()
    fig = plt.figure(figsize=cf.use(width_frac=1.0, aspect=0.52))
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    panel(axA, FEMDIR / "tier2b_generic_meta.npz", r"(A) bare screw $+$ abutment (load at platform)", False)
    pk_b = cf.peri_implant_bone_peak(FEMDIR / "tier2b_generic_field.json")
    axA.text2D(0.04, 0.06, r"crestal bone peak %.0f MPa" % pk_b, transform=axA.transAxes,
               fontsize=5.6, color="0.25")
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    panel(axB, FEMDIR / "tier2b_crown_meta.npz", r"(B) restored: crown $+$ gingiva (load at occlusal table)", True)
    pk_c = cf.peri_implant_bone_peak(FEMDIR / "tier2b_crown_field.json")
    axB.text2D(0.04, 0.06, r"crestal bone peak %.0f MPa ($\times$%.1f)" % (pk_c, pk_c / pk_b),
               transform=axB.transAxes, fontsize=5.6, color="#b30000", fontweight="bold")
    fig.suptitle(r"Crown moment arm: load moves from the abutment platform to the occlusal table "
                 r"$\Rightarrow$ crestal peri-implant bone peak $\times$%.1f" % (pk_c / pk_b),
                 fontsize=7.2, y=0.99)
    fig.subplots_adjust(left=0.0, right=0.99, bottom=0.0, top=0.9, wspace=0.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_bare_vs_restored.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "fem_implant_bare_vs_restored.png", dpi=300, bbox_inches="tight")
    print("wrote fem_implant_bare_vs_restored  bone-peak %.0f -> %.0f (x%.2f)" % (pk_b, pk_c, pk_c / pk_b))


if __name__ == "__main__":
    main()
