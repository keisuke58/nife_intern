"""Hero 3-D figure of the implant FEM: voxel cutaway of the real implant + tooth + bone
assembly (tier2b_real).  The front half of the alveolar bone is cut away so the titanium
implant, the natural tooth root (dentin + PDL) and the dysbiotic biofilm collars stand
revealed inside the translucent bone block.  (A) material anatomy, (B) von-Mises stress.

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_voxel3d.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

FIELD = "/home/nishioka/IKM_Hiwi/FEM/tier2b_real/tier2b_real_field.json"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
COL = {"BONE": "#dcc79a", "TI": "#5a5a5a", "DENTIN": "#9ecae1", "PDL": "#ff8c00", "BIOFILM": "#d62728"}
LAB = {"TI": "implant (Ti)", "DENTIN": "tooth (dentin)", "PDL": "PDL", "BIOFILM": "biofilm", "BONE": "bone"}
PRI = ["BONE", "PDL", "DENTIN", "TI", "BIOFILM"]    # display priority: high index wins a shared cell
ELEV, AZIM = 14, -58


def dims(cell, b):
    x0, x1, y0, y1, z0, z1 = b
    nx = int(np.ceil((x1 - x0) / cell)); ny = int(np.ceil((y1 - y0) / cell)); nz = int(np.ceil((z1 - z0) / cell))
    ex = x0 + np.arange(nx + 1) * cell; ey = y0 + np.arange(ny + 1) * cell; ez = z0 + np.arange(nz + 1) * cell
    return nx, ny, nz, ex, ey, ez


def idx(a, lo, cell, n):
    return np.clip(((a - lo) / cell).astype(int), 0, n - 1)


def matgrid(x, y, z, vmo, code, cell, b):
    nx, ny, nz, ex, ey, ez = dims(cell, b)
    ix = idx(x, b[0], cell, nx); iy = idx(y, b[2], cell, ny); iz = idx(z, b[4], cell, nz)
    cm = np.full((nx, ny, nz), -1)
    order = np.argsort(code, kind="stable")
    cm[ix[order], iy[order], iz[order]] = code[order]
    sv = np.zeros((nx, ny, nz)); cnt = np.zeros((nx, ny, nz))
    np.add.at(sv, (ix, iy, iz), vmo); np.add.at(cnt, (ix, iy, iz), 1)
    mv = np.where(cnt > 0, sv / np.maximum(cnt, 1.0), np.nan)
    X, Y, Z = np.meshgrid(ex, ey, ez, indexing="ij")
    return cm, mv, X, Y, Z, (ex, ey, ez)


def occ(x, y, z, sel, vmo, cell, b):
    """Boolean occupancy + mean value for a material selection, on a regular grid."""
    nx, ny, nz, ex, ey, ez = dims(cell, b)
    ix = idx(x[sel], b[0], cell, nx); iy = idx(y[sel], b[2], cell, ny); iz = idx(z[sel], b[4], cell, nz)
    g = np.zeros((nx, ny, nz), bool); g[ix, iy, iz] = True
    sv = np.zeros((nx, ny, nz)); cnt = np.zeros((nx, ny, nz))
    np.add.at(sv, (ix, iy, iz), vmo[sel]); np.add.at(cnt, (ix, iy, iz), 1)
    mv = np.where(cnt > 0, sv / np.maximum(cnt, 1.0), np.nan)
    X, Y, Z = np.meshgrid(ex, ey, ez, indexing="ij")
    return g, mv, X, Y, Z, (ex, ey, ez)


def style3d(ax, b):
    x0, x1, y0, y1, z0, z1 = b
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.set_xlabel(r"$x$ (mm)", fontsize=5.5, labelpad=-6)
    ax.set_ylabel(r"$y$ (mm)", fontsize=5.5, labelpad=-6)
    ax.set_zlabel(r"depth $z$ (mm)", fontsize=6, labelpad=-3)
    ax.tick_params(labelsize=4.5, pad=-2)
    ax.set_xticks([-74, -64]); ax.set_yticks([-46, -39]); ax.set_zticks([18, 26, 34])
    try:
        ax.set_proj_type("persp", focal_length=0.55)
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.03); axis.pane.set_edgecolor((0, 0, 0, 0.10))


def main():
    use()
    d = json.load(open(FIELD))["els"]
    x = np.array([r["x"] for r in d]); y = np.array([r["y"] for r in d]); z = np.array([r["z"] for r in d])
    mat = np.array([r["mat"] for r in d]); vmo = np.array([r["vmo"] for r in d])
    b = (x.min(), x.max(), y.min(), y.max(), z.min(), z.max())
    ymid = (b[2] + b[3]) / 2
    code = np.array([PRI.index(m) if m in PRI else 0 for m in mat])

    cmF, mvF, XF, YF, ZF, _ = matgrid(x, y, z, vmo, code, 0.45, b)           # fine: structures
    struct = cmF >= 1
    boneG, boneV, XB, YB, ZB, (exB, eyB, ezB) = occ(x, y, z, mat == "BONE", vmo, 1.2, b)  # coarse bone
    cyB = (eyB[:-1] + eyB[1:]) / 2
    back = np.broadcast_to(cyB[None, :, None] >= ymid, boneG.shape)          # keep BACK half (cutaway front)
    bone_show = boneG & back

    fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.52))

    # ---------- Panel A: material anatomy ----------
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    fcA = np.zeros(struct.shape + (4,))
    for c, name in enumerate(PRI):
        if c:
            fcA[cmF == c] = to_rgba(COL[name], 1.0)
    fcBone = np.zeros(bone_show.shape + (4,)); fcBone[bone_show] = to_rgba(COL["BONE"], 0.22)
    axA.voxels(XB, YB, ZB, bone_show, facecolors=fcBone, edgecolor=(0, 0, 0, 0.05), shade=False)
    axA.voxels(XF, YF, ZF, struct, facecolors=fcA, edgecolor=(0, 0, 0, 0.16), linewidth=0.07, shade=True)
    axA.set_title(r"(A) anatomy: implant $+$ tooth in bone (front cut away)", fontsize=6.8, pad=-4)
    style3d(axA, b)
    present = [k for k in ["TI", "DENTIN", "PDL", "BIOFILM", "BONE"] if (mat == k).any()]
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=COL[k], mec="0.5", mew=0.3, label=LAB[k])
               for k in present]
    axA.legend(handles=handles, fontsize=5.0, loc="upper left", ncol=1, handletextpad=0.25,
               labelspacing=0.3, framealpha=0.85, bbox_to_anchor=(-0.02, 0.97))

    # ---------- Panel B: von Mises stress ----------
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    vclip = float(np.nanpercentile(mvF[struct], 96))
    norm = Normalize(0, vclip); cmap = plt.get_cmap("inferno")
    fcS = np.zeros(struct.shape + (4,))
    fcS[struct] = cmap(norm(np.clip(np.nan_to_num(mvF[struct]), 0, vclip)))
    hi = bone_show & (boneV > np.nanpercentile(boneV[boneG], 85))            # crestal high-stress bone
    fcHi = np.zeros(hi.shape + (4,)); fcHi[hi] = cmap(norm(np.clip(np.nan_to_num(boneV[hi]), 0, vclip)))
    fcHi[hi, 3] = 0.6
    axB.voxels(XB, YB, ZB, hi, facecolors=fcHi, edgecolor="none", shade=False)
    axB.voxels(XF, YF, ZF, struct, facecolors=fcS, edgecolor=(0, 0, 0, 0.08), linewidth=0.05, shade=True)
    axB.set_title(r"(B) occlusal von Mises (crestal-bone hot zone)", fontsize=6.8, pad=-4)
    style3d(axB, b)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axB, fraction=0.028, pad=0.0, shrink=0.55)
    cb.set_label(r"$\sigma_\mathrm{vM}$ (MPa)", fontsize=6); cb.ax.tick_params(labelsize=5)

    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.0, top=0.99, wspace=0.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_voxel3d.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_implant_voxel3d.png", dpi=300, bbox_inches="tight")
    print("wrote fem_implant_voxel3d  struct=%d bone=%d hi=%d vclip=%.2f"
          % (struct.sum(), bone_show.sum(), hi.sum(), vclip))


if __name__ == "__main__":
    main()
