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

# argv: <job> (default tier2b_real = patient root-analog; tier2b_generic = standard screw implant)
JOB = sys.argv[1] if len(sys.argv) > 1 else "tier2b_real"
TITLE_A = {"tier2b_real": r"(A) real anatomy: root-analog implant $+$ tooth $+$ bone",
           "tier2b_generic": r"(A) real bone: generic Ø4.1 screw implant $+$ tooth"}.get(
    JOB, "(A) real anatomy")
FIELD = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real/%s_field.json" % JOB)
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
MAT_ORDER = ["CANCELLOUS", "BONE", "CORTICAL", "PDL", "DENTIN", "TI", "BIOFILM"]
MAT_COL = {"BONE": "#e7d8b0", "CANCELLOUS": "#efe3c2", "CORTICAL": "#c9a96a", "PDL": "#ff8c00",
           "DENTIN": "#9ecae1", "TI": "#636363", "BIOFILM": "#d62728"}
MAT_LAB = {"TI": "implant (Ti)", "DENTIN": "tooth (dentin)", "CORTICAL": "cortical bone",
           "CANCELLOUS": "cancellous bone", "PDL": "PDL", "BIOFILM": "biofilm", "BONE": "bone"}
T23, T24 = -69.4, -63.9   # mesiodistal x of implant / tooth axes


def main():
    use()
    d = json.load(open(FIELD))["els"]
    x = np.array([r["x"] for r in d]); y = np.array([r["y"] for r in d])
    z = np.array([r["z"] for r in d]); mat = np.array([r["mat"] for r in d])
    vmo = np.array([r["vmo"] for r in d])
    m = np.abs(y - (-41.0)) < 0.9                      # mesiodistal slice through both axes

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.5))

    # ---- shared raster via nearest-neighbour fill: solid fields (the coarse implant/dentin
    #      meshes fill in), with true white only OUTSIDE the anatomy (distance mask).
    from scipy.spatial import cKDTree
    x0, x1, z0, z1 = -74.0, -60.0, 16.5, 31.5
    cell = 0.10
    NX = int(round((x1 - x0) / cell)); NZ = int(round((z1 - z0) / cell))
    code = np.array([MAT_ORDER.index(mm) if mm in MAT_ORDER else 0 for mm in mat])
    inb = m & (x >= x0 - 1) & (x < x1 + 1) & (z >= z0 - 1) & (z < z1 + 1)
    xb, zb, vb, cb_, mb = x[inb], z[inb], vmo[inb], code[inb], mat[inb]
    tree = cKDTree(np.column_stack([xb, zb]))
    gx = x0 + (np.arange(NX) + 0.5) * (x1 - x0) / NX
    gz = z0 + (np.arange(NZ) + 0.5) * (z1 - z0) / NZ
    GX, GZ = np.meshgrid(gx, gz)
    dist, idx = tree.query(np.column_stack([GX.ravel(), GZ.ravel()]), k=1)
    dist = dist.reshape(NZ, NX); idx = idx.reshape(NZ, NX)
    # adaptive fill radius = a few median nearest-neighbour spacings (fills coarse meshes,
    # masks the genuine exterior so it stays white)
    sub = np.random.default_rng(0).choice(len(xb), size=min(3000, len(xb)), replace=False)
    dnn, _ = tree.query(np.column_stack([xb[sub], zb[sub]]), k=2)
    thr = max(0.45, 3.0 * float(np.median(dnn[:, 1])))
    masked = dist > thr

    # ---- Panel A: real multi-material anatomy ----
    A = np.where(masked, np.nan, cb_[idx].astype(float))
    cmapA = ListedColormap([MAT_COL[k] for k in MAT_ORDER]); cmapA.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm(np.arange(-0.5, len(MAT_ORDER)), cmapA.N)
    axA.imshow(np.ma.masked_invalid(A), origin="lower", extent=[x0, x1, z0, z1],
               cmap=cmapA, norm=norm, interpolation="nearest", aspect="equal")
    axA.set_title(TITLE_A, fontsize=7.0)
    present = [k for k in MAT_ORDER if (mb == k).any()]
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=MAT_COL[k], mec="none", label=MAT_LAB[k])
               for k in present]
    axA.legend(handles=handles, fontsize=5.0, loc="lower center", ncol=3,
               handletextpad=0.2, columnspacing=0.7, framealpha=0.9)

    # ---- Panel B: occlusal von Mises ----
    Bv = vb[idx]
    vclip = float(np.ceil(np.nanpercentile(Bv[~masked], 98)))   # adaptive (oblique load)
    B = np.where(masked, np.nan, np.clip(Bv, 0, vclip))
    cmapB = plt.get_cmap("inferno").copy(); cmapB.set_bad((1, 1, 1, 0))
    sc = axB.imshow(np.ma.masked_invalid(B), origin="lower", extent=[x0, x1, z0, z1],
                    cmap=cmapB, vmin=0, vmax=vclip, interpolation="nearest", aspect="equal")
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
    outpdf = OUT / ("fem_%s.pdf" % JOB)
    fig.savefig(outpdf, bbox_inches="tight")
    plt.close(fig)
    print("slice elems=%d  wrote %s" % (m.sum(), outpdf))


if __name__ == "__main__":
    main()
