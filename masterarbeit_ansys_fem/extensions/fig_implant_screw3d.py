"""Clearest 3-D implant FEM figure: the GENERIC screw implant (threaded, unmistakably an
implant) next to the natural tooth, with explicit IMPLANT / TOOTH labels.  Real tet mesh
(tier2b_generic_meta.npz) rendered as shaded boundary surfaces; front bone clipped away.
  (A) labelled anatomy, (B) occlusal von-Mises stress.

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_screw3d.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

FEMDIR = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
META = FEMDIR / "tier2b_generic_meta.npz"
FIELD = FEMDIR / "tier2b_generic_field.json"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
COL = {"BONE": "#d6c193", "TI": "#5a5a5a", "DENTIN": "#9ecae1", "PDL": "#ff8c00", "BIOFILM": "#d62728"}
LAB = {"TI": "implant (Ti)", "DENTIN": "tooth (dentin)", "PDL": "PDL", "BIOFILM": "biofilm", "BONE": "bone"}
BONE_MATS = ("CORTICAL", "CANCELLOUS")
ELEV, AZIM = 14, -58
LIGHT = np.array([0.35, 0.55, 0.75]); LIGHT = LIGHT / np.linalg.norm(LIGHT)


def boundary_faces(tets):
    nt = len(tets)
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                        tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]], axis=0)
    parent = np.tile(np.arange(nt), 4)
    key = np.sort(F, axis=1)
    order = np.lexsort(key.T)
    ks = key[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    dup = np.zeros(len(ks), bool); dup[1:] |= same; dup[:-1] |= same
    sel = order[~dup]
    return F[sel], parent[sel]


def shade(tris, rgb):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    s = 0.45 + 0.55 * np.clip(np.abs(n @ LIGHT), 0, 1)
    return np.clip(np.asarray(rgb)[None, :] * s[:, None], 0, 1)


def style3d(ax, b):
    x0, x1, y0, y1, z0, z1 = b
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.set_xlabel(r"$x$ (mm)", fontsize=5.5, labelpad=-6)
    ax.set_ylabel(r"$y$ (mm)", fontsize=5.5, labelpad=-6)
    ax.set_zlabel(r"depth $z$ (mm)", fontsize=6, labelpad=-3)
    ax.tick_params(labelsize=4.5, pad=-2)
    ax.set_xticks([-72, -62]); ax.set_yticks([-46, -39]); ax.set_zticks([18, 26, 34])
    try:
        ax.set_proj_type("persp", focal_length=0.55)
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.03); axis.pane.set_edgecolor((0, 0, 0, 0.10))


def label3d(ax, x, y, z, text, color):
    ax.text(x, y, z, text, fontsize=6.2, fontweight="bold", color=color, ha="center", va="bottom",
            zorder=20, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.6, alpha=0.9))


def main():
    use()
    a = np.load(META, allow_pickle=True)
    nodes = a["nodes"]; conn = a["conn"]; mat = a["mat"]
    vmo = np.array([r["vmo"] for r in json.load(open(FIELD))["els"]])
    b = (nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max(),
         nodes[:, 2].min(), nodes[:, 2].max())
    ymid = (b[2] + b[3]) / 2

    surf = {}
    for M in ["TI", "DENTIN", "PDL", "BIOFILM"]:
        idx = np.where(mat == M)[0]
        fb, par = boundary_faces(conn[idx])
        surf[M] = (nodes[fb], vmo[idx[par]])
    bidx = np.where(np.isin(mat, BONE_MATS))[0]
    by = nodes[conn[bidx]][:, :, 1].mean(axis=1)
    bk = bidx[by >= ymid]
    fbB, _ = boundary_faces(conn[bk])
    if len(fbB) > 14000:
        fbB = fbB[np.random.default_rng(0).choice(len(fbB), 14000, replace=False)]
    surfB = nodes[fbB]

    # label anchor points (top of each column, front face)
    yf = b[2] + 0.18 * (b[3] - b[2])
    ti_c = nodes[conn[mat == "TI"]].reshape(-1, 3); de_c = nodes[conn[mat == "DENTIN"]].reshape(-1, 3)
    imp_xyz = (ti_c[:, 0].mean(), yf, ti_c[:, 2].max() + 2.2)
    too_xyz = (de_c[:, 0].mean(), yf, de_c[:, 2].max() + 2.2)

    fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.52))

    # ---------- Panel A: labelled anatomy ----------
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    axA.add_collection3d(Poly3DCollection(surfB, facecolors=shade(surfB, to_rgb(COL["BONE"])),
                                          edgecolors="none", alpha=0.28, rasterized=True))
    for M in ["DENTIN", "PDL", "TI", "BIOFILM"]:
        tris, _ = surf[M]
        axA.add_collection3d(Poly3DCollection(tris, facecolors=shade(tris, to_rgb(COL[M])),
                                              edgecolors=(0, 0, 0, 0.10), linewidths=0.05, rasterized=True))
    label3d(axA, *imp_xyz, "IMPLANT (Ti screw)", "#222222")
    label3d(axA, *too_xyz, "natural tooth", "#b35806")
    axA.set_title(r"(A) screw implant $+$ natural tooth (real mesh)", fontsize=6.8, pad=-4)
    style3d(axA, b)
    present = [k for k in ["TI", "DENTIN", "PDL", "BIOFILM", "BONE"] if (np.isin(mat, [k]).any() or k == "BONE")]
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=COL[k], mec="0.5", mew=0.3, label=LAB[k])
               for k in present]
    axA.legend(handles=handles, fontsize=5.0, loc="upper left", handletextpad=0.25,
               labelspacing=0.3, framealpha=0.85, bbox_to_anchor=(-0.02, 0.97))

    # ---------- Panel B: von Mises ----------
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    allv = np.concatenate([surf[M][1] for M in surf])
    vclip = float(np.nanpercentile(allv, 96)); norm = Normalize(0, vclip); cmap = plt.get_cmap("inferno")
    axB.add_collection3d(Poly3DCollection(surfB, facecolors=(0.7, 0.7, 0.7, 0.06),
                                          edgecolors="none", rasterized=True))
    for M in ["DENTIN", "PDL", "TI", "BIOFILM"]:
        tris, fv = surf[M]
        base = cmap(norm(np.clip(np.nan_to_num(fv), 0, vclip)))[:, :3]
        sh = shade(tris, [1, 1, 1])[:, 0]
        pc = Poly3DCollection(tris, edgecolors="none", rasterized=True)
        pc.set_facecolor(np.clip(base * (0.5 + 0.5 * sh)[:, None], 0, 1))
        axB.add_collection3d(pc)
    label3d(axB, *imp_xyz, "IMPLANT", "#222222")
    label3d(axB, *too_xyz, "tooth", "#b35806")
    axB.set_title(r"(B) occlusal von Mises $\sigma_\mathrm{vM}$", fontsize=6.8, pad=-4)
    style3d(axB, b)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axB, fraction=0.028, pad=0.0, shrink=0.55)
    cb.set_label(r"$\sigma_\mathrm{vM}$ (MPa)", fontsize=6); cb.ax.tick_params(labelsize=5)

    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.0, top=0.99, wspace=0.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_screw3d.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "fem_implant_screw3d.png", dpi=300, bbox_inches="tight")
    print("wrote fem_implant_screw3d  struct-faces=%d bone-faces=%d vclip=%.2f"
          % (sum(len(surf[M][0]) for M in surf), len(surfB), vclip))


if __name__ == "__main__":
    main()
