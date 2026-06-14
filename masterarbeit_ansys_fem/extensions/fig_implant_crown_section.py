"""Clean ANATOMICAL HEMISECTION of the restored-implant assembly: every body cut at the mesiodistal
midplane (y = y_mid) and viewed toward the cut, so the layered peri-implant anatomy reads as a
cross-section -- cortical/cancellous bone, the titanium screw thread in section, the abutment, the
ceramic crown, the PDL-sheathed neighbour tooth, the gingival cuff, and the BIOFILM in the sulcus.
Companion to the 3-D view fig_implant_crown_fem.py (kept); this one is the textbook section.

(A) labelled anatomy section ; (B) occlusal von Mises section.
Run: python masterarbeit_ansys_fem/extensions/fig_implant_crown_section.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EXT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXT))
import fig_implant_crown_fem as cf  # noqa: E402

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
FEMDIR = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
META = FEMDIR / "tier2b_crown_meta.npz"
FIELD = FEMDIR / "tier2b_crown_field.json"
ELEV, AZIM = 6, -90                       # look toward +y -> the x-z cut plane faces the viewer
STRUCT = ["DENTIN", "PDL", "TI", "BIOFILM", "CROWN"]
BONE = ("CORTICAL", "CANCELLOUS", "BONE")


def half(tris, ymid):
    return tris[tris.mean(axis=1)[:, 1] >= ymid]


def sec_style(ax, b):
    x0, x1, y0, y1, z0, z1 = b
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((x1 - x0, (y1 - y0) * 0.35, z1 - z0))
    ax.set_xlabel(r"mesiodistal $x$ (mm)", fontsize=6, labelpad=-4)
    ax.set_zlabel(r"depth $z$ (mm)", fontsize=6.5, labelpad=-2)
    ax.set_yticks([])
    ax.tick_params(labelsize=5, pad=-2)
    ax.set_xticks([-72, -66, -60]); ax.set_zticks([20, 25, 30, 35, 40])
    try:
        ax.set_proj_type("persp", focal_length=0.9)
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0); axis.pane.set_edgecolor((0, 0, 0, 0.0))


def main():
    cf.use()
    a = np.load(META, allow_pickle=True)
    nodes, conn, mat = a["nodes"], a["conn"], a["mat"]
    vmo = np.array([r["vmo"] for r in json.load(open(FIELD))["els"]])
    b = (nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max(),
         nodes[:, 2].min(), nodes[:, 2].max())
    ymid = (b[2] + b[3]) / 2

    surf = {}
    for M in STRUCT + list(BONE):
        idx = np.where(mat == M)[0]
        if not len(idx):
            continue
        fb, par = cf.boundary_faces(conn[idx])
        keep = nodes[fb].mean(axis=1)[:, 1] >= ymid     # back half -> the section face
        surf[M] = (nodes[fb][keep], vmo[idx[par]][keep])

    fig = plt.figure(figsize=cf.use(width_frac=1.0, aspect=0.62))

    # ---------- (A) anatomy ----------
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    order = list(BONE) + ["PDL", "DENTIN", "TI", "BIOFILM"]
    for M in order:
        if M not in surf:
            continue
        tris = surf[M][0]
        col = cf.COL["BONE"] if M in BONE else cf.COL[M]
        al = {"CORTICAL": 0.9, "CANCELLOUS": 0.55, "BONE": 0.85, "BIOFILM": 0.85}.get(M, 1.0)
        sh = (0.6, 0.42, 0.12) if M == "BIOFILM" else (0.55, 0.5, 0.0)
        pc = Poly3DCollection(tris, facecolors=cf.shade(tris, to_rgb(col), *sh),
                              edgecolors=(0.2, 0.1, 0.1, 0.25), linewidths=0.06, rasterized=True)
        pc.set_alpha(al); axA.add_collection3d(pc)
    # real-tooth ceramic crown + gingiva cuff (back half)
    cr_top = nodes[conn[mat == "CROWN"]].reshape(-1, 3)[:, 2].max()
    cm = half(cf.real_tooth_crown(cf.STL_TOOTH, h_target=cr_top - 31.0, base_z=31.0), ymid)
    axA.add_collection3d(Poly3DCollection(cm, facecolors=cf.shade(cm, to_rgb(cf.COL["CROWN"]), 0.62, 0.34, 0.30),
                                          edgecolors=(0, 0, 0, 0.15), linewidths=0.05, rasterized=True))
    def neck_c(M, zl):
        c = nodes[conn[mat == M]].reshape(-1, 3); s = (c[:, 2] >= zl - 0.6) & (c[:, 2] <= zl + 0.6)
        ct = c[s, :2].mean(0)
        return ct, float(np.percentile(np.hypot(c[s, 0] - ct[0], c[s, 1] - ct[1]), 99))
    ic, ir = neck_c("TI", 31.0); tc, tr = neck_c("DENTIN", 31.0)
    gum = half(cf.gingiva_mound(ic, ir, tc, tr, ymid), ymid)
    pg = Poly3DCollection(gum, facecolors=cf.shade(gum, to_rgb(cf.COL["GINGIVA"]), 0.6, 0.35),
                          edgecolors=(0.6, 0.2, 0.3, 0.3), linewidths=0.06, rasterized=True)
    pg.set_alpha(0.85); axA.add_collection3d(pg)
    sec_style(axA, b)
    axA.set_title("(A) restored-implant hemisection (anatomy)", fontsize=7.2, pad=-2)
    labels = [("crown (ceramic)", cf.COL["CROWN"], 39), ("gingiva", cf.COL["GINGIVA"], 32.2),
              ("biofilm (sulcus)", "#d62728", 30.4), ("implant (Ti)", cf.COL["TI"], 24),
              ("tooth + PDL", cf.COL["PDL"], 21), ("cortical / cancellous bone", cf.COL["BONE"], 17)]
    hs = [plt.Line2D([], [], marker="s", ls="", mfc=c, mec="0.4", mew=0.3, label=t) for t, c, _ in labels]
    axA.legend(handles=hs, fontsize=5.4, loc="center left", bbox_to_anchor=(-0.08, 0.5),
               framealpha=0.9, handletextpad=0.3, labelspacing=0.4)

    # ---------- (B) von Mises ----------
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    allv = np.concatenate([surf[M][1] for M in STRUCT if M in surf])
    vclip = float(np.nanpercentile(allv, 90)); norm = Normalize(0, vclip); cmap = plt.get_cmap("viridis")
    for M in BONE:
        if M in surf:
            axB.add_collection3d(Poly3DCollection(surf[M][0], facecolors=(0.7, 0.7, 0.7, 0.10),
                                                  edgecolors="none", rasterized=True))
    for M in ["PDL", "DENTIN", "TI", "BIOFILM"]:
        if M not in surf:
            continue
        tris, fv = surf[M]
        base = cmap(norm(np.clip(np.nan_to_num(fv), 0, vclip)))[:, :3]
        s = cf._brightness(tris, 0.72, 0.28)
        pc = Poly3DCollection(tris, edgecolors="none", rasterized=True)
        pc.set_facecolor(np.clip(base * s[:, None], 0, 1)); axB.add_collection3d(pc)
    sec_style(axB, b)
    axB.set_title(r"(B) occlusal von Mises $\sigma_\mathrm{vM}$ (section)", fontsize=7.2, pad=-2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axB, fraction=0.026, pad=-0.02, shrink=0.5)
    cb.set_label(r"$\sigma_\mathrm{vM}$ (MPa)", fontsize=6); cb.ax.tick_params(labelsize=5)

    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.0, top=0.96, wspace=0.02)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_crown_section.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "fem_implant_crown_section.png", dpi=300, bbox_inches="tight")
    print("wrote fem_implant_crown_section  (hemisection, %d struct-faces)"
          % sum(len(surf[M][0]) for M in STRUCT if M in surf))


if __name__ == "__main__":
    main()
