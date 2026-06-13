"""Restored-implant FEM (LOAD-BEARING crown): the ceramic crown is a real meshed body seated on the
abutment, so the 30deg-oblique occlusal bite is applied at the crown occlusal table (z~40.8), not the
bare abutment top (z=32.5).  The crown-height moment arm bends the implant neck and overloads the
peri-implant crestal bone -- the region where peri-implantitis bone loss occurs.

  (A) labelled restored anatomy (crown = real ceramic body);  (B) occlusal von Mises, crown included.

Reads tier2b_crown_meta.npz + tier2b_crown_field.json (build_assembly.py job=tier2b_crown).
Run:  python masterarbeit_ansys_fem/extensions/fig_implant_crown_fem.py
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
META = FEMDIR / "tier2b_crown_meta.npz"
FIELD = FEMDIR / "tier2b_crown_field.json"
GEN_FIELD = FEMDIR / "tier2b_generic_field.json"   # no-crown baseline (for the moment-arm delta)
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
COL = {"BONE": "#d6c193", "TI": "#7c8893", "DENTIN": "#9ecae1", "PDL": "#ff8c00",
       "BIOFILM": "#d62728", "CROWN": "#f6f1e7"}   # bright tooth-coloured ceramic
LAB = {"TI": "implant (Ti)", "DENTIN": "tooth (dentin)", "PDL": "PDL", "BIOFILM": "biofilm",
       "BONE": "bone", "CROWN": "crown (ceramic)"}
STRUCT = ["DENTIN", "PDL", "TI", "BIOFILM", "CROWN"]
BONE_MATS = ("CORTICAL", "CANCELLOUS")
ELEV, AZIM = 11, -50
LIGHT = np.array([0.32, 0.5, 0.8]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
SH = {"TI": (0.32, 0.80, 0.55), "DENTIN": (0.5, 0.5, 0.0), "PDL": (0.5, 0.5, 0.0),
      "BIOFILM": (0.5, 0.5, 0.0), "BONE": (0.5, 0.5, 0.0), "CROWN": (0.66, 0.32, 0.35)}  # bright + glossy
T23 = np.array([-69.4, -41.0]); CREST = 29.0
# --- illustrative overlays (toggle on/off; default ON, disable per-flag) ---
ARGS = set(sys.argv[1:])
GINGIVA = "--no-gingiva" not in ARGS       # peri-implant mucosa cuff (z 29->31.9), transmucosal-consistent
REALCROWN = "--no-realcrown" not in ARGS   # draw a REAL segmented molar crown (cusps/grooves) in Panel A
STL_TOOTH = Path("/home/nishioka/IKM_Hiwi/FEM/external_tooth_models/OpenJaw_Dataset/"
                 "Patient_1/Teeth/P1_Tooth_30.stl")   # real mandibular molar (occlusal anatomy)
COL["GINGIVA"] = "#f2a0ad"                  # mucosa pink, matches fig_transmucosal.py
# gingival cuff as a rounded collar HUGGING the implant emergence (r,z profile, lathed about the axis):
# narrow at the bone crest, a slight free-gingiva bulge, tapering to the margin against the crown.
GUM_PROFILE = [(2.45, CREST), (3.05, CREST + 0.9), (2.95, CREST + 1.9), (2.35, 31.9)]


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


def _brightness(tris, lo=0.45, span=0.55, spec=0.0):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    d = np.clip(np.abs(n @ LIGHT), 0, 1)
    s = lo + span * d
    if spec:
        s = s + spec * d ** 12
    return s


def shade(tris, rgb, lo=0.45, span=0.55, spec=0.0):
    s = _brightness(tris, lo, span, spec)
    return np.clip(np.asarray(rgb)[None, :] * s[:, None], 0, 1)


def style3d(ax, b):
    x0, x1, y0, y1, z0, z1 = b
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.set_xlabel(r"$x$ (mm)", fontsize=5.5, labelpad=-6)
    ax.set_ylabel(r"$y$ (mm)", fontsize=5.5, labelpad=-6)
    ax.set_zlabel(r"depth $z$ (mm)", fontsize=6, labelpad=-3)
    ax.tick_params(labelsize=4.5, pad=-2)
    ax.set_xticks([-72, -62]); ax.set_yticks([-46, -39]); ax.set_zticks([20, 30, 40])
    try:
        ax.set_proj_type("persp", focal_length=0.55)
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.03); axis.pane.set_edgecolor((0, 0, 0, 0.10))


def label3d(ax, x, y, z, text, color):
    ax.text(x, y, z, text, fontsize=6.2, fontweight="bold", color=color, ha="center", va="bottom",
            zorder=20, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.6, alpha=0.9))


def load_stl(fn):
    """Binary-STL triangles -> (n,3,3) float array."""
    import struct
    with open(fn, "rb") as f:
        f.read(80); n = struct.unpack("<I", f.read(4))[0]; data = f.read(n * 50)
    rec = np.frombuffer(data, dtype=np.uint8).reshape(n, 50)
    return rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float)


def real_tooth_crown(stl, h_target=7.0, base_z=31.0, cx=T23[0], cy=T23[1], crown_frac=0.40):
    """Extract the clinical crown of a real segmented molar STL, upright it (long axis -> z, occlusal
    up), scale to h_target height, and seat it on the implant axis (base at base_z). Returns triangles
    for an ILLUSTRATIVE overlay (real cusps/grooves) -- not a FEM body."""
    tris = load_stl(stl)
    V = tris.reshape(-1, 3); c = V.mean(0); Vc = V - c
    _, _, vt = np.linalg.svd(Vc - Vc.mean(0), full_matrices=False)
    a = vt[0] / np.linalg.norm(vt[0]); t = Vc @ a
    perp = Vc - np.outer(t, a); rad = np.linalg.norm(perp, axis=1)
    if rad[t > np.percentile(t, 80)].mean() < rad[t < np.percentile(t, 20)].mean():
        a = -a                                  # make the WIDE (crown) end point at +t
    zhat = np.array([0, 0, 1.0]); v = np.cross(a, zhat); sang = np.linalg.norm(v); cang = a @ zhat
    if sang < 1e-8:
        R = np.eye(3) if cang > 0 else np.diag([1.0, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cang) / (sang ** 2))
    P = ((tris.reshape(-1, 3) - c) @ R.T).reshape(-1, 3, 3)     # upright, occlusal at +z
    zc = P[:, :, 2].mean(axis=1); zmax, zmin = zc.max(), zc.min()
    P = P[zc >= zmax - crown_frac * (zmax - zmin)]              # keep the occlusal crown band
    sc = h_target / (P[:, :, 2].max() - P[:, :, 2].min()); P *= sc
    m = P.reshape(-1, 3).mean(0)
    P[:, :, 0] += cx - m[0]; P[:, :, 1] += cy - m[1]; P[:, :, 2] += base_z - P[:, :, 2].min()
    return P


def gingiva_cuff(cx=T23[0], cy=T23[1], prof=None, nth=64):
    """Peri-implant mucosa cuff (illustrative): a rounded collar hugging the implant emergence, built
    by lathing the (r,z) profile about the implant axis. Narrow at the crest, slight free-gingiva bulge,
    tapering to the margin -- so it FITS the implant rather than flaring into a wide skirt."""
    prof = prof if prof is not None else GUM_PROFILE
    th = np.linspace(0, 2 * np.pi, nth)
    f = []
    for j in range(len(prof) - 1):
        r0, z0 = prof[j]; r1, z1 = prof[j + 1]
        for i in range(nth - 1):
            p00 = [cx + r0 * np.cos(th[i]), cy + r0 * np.sin(th[i]), z0]
            p01 = [cx + r0 * np.cos(th[i + 1]), cy + r0 * np.sin(th[i + 1]), z0]
            p10 = [cx + r1 * np.cos(th[i]), cy + r1 * np.sin(th[i]), z1]
            p11 = [cx + r1 * np.cos(th[i + 1]), cy + r1 * np.sin(th[i + 1]), z1]
            f.append([p00, p01, p11, p10])
    return np.array(f)


def peri_implant_bone_peak(field_json):
    """Robust (p95) occlusal vM in the crestal peri-implant bone (cortical/cancellous, r<3 mm, near
    crest). p95 not max: with anatomically intact crestal bone the absolute max is a neck/TIE-edge
    geometric singularity (insensitive to the crown), so the broad crestal stress (p95) is the
    physically meaningful measure of the moment-arm effect."""
    d = json.load(open(field_json))["els"]
    x = np.array([e["x"] for e in d]); y = np.array([e["y"] for e in d]); z = np.array([e["z"] for e in d])
    vm = np.array([e["vmo"] for e in d]); mat = np.array([e["mat"] for e in d])
    r = np.hypot(x - T23[0], y - T23[1])
    pib = np.isin(mat, ["CORTICAL", "CANCELLOUS", "BONE"]) & (r <= 3.0) & (z >= CREST - 3) & (z <= CREST + 1.5)
    return float(np.percentile(vm[pib], 95))


def main():
    use()
    a = np.load(META, allow_pickle=True)
    nodes = a["nodes"]; conn = a["conn"]; mat = a["mat"]
    vmo = np.array([r["vmo"] for r in json.load(open(FIELD))["els"]])
    b = (nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max(),
         nodes[:, 2].min(), nodes[:, 2].max())
    ymid = (b[2] + b[3]) / 2

    surf = {}
    for M in STRUCT:
        idx = np.where(mat == M)[0]
        if not len(idx):
            continue
        fb, par = boundary_faces(conn[idx])
        surf[M] = (nodes[fb], vmo[idx[par]])
    bidx = np.where(np.isin(mat, BONE_MATS))[0]
    by = nodes[conn[bidx]][:, :, 1].mean(axis=1)
    bk = bidx[by >= ymid]
    fbB, _ = boundary_faces(conn[bk])
    if len(fbB) > 14000:
        fbB = fbB[np.random.default_rng(0).choice(len(fbB), 14000, replace=False)]
    surfB = nodes[fbB]

    yf = b[2] + 0.18 * (b[3] - b[2])
    cr_c = nodes[conn[mat == "CROWN"]].reshape(-1, 3); de_c = nodes[conn[mat == "DENTIN"]].reshape(-1, 3)
    crown_cx = cr_c[:, 0].mean()
    imp_xyz = (crown_cx - 3.4, yf, cr_c[:, 2].max() + 0.6)   # IMPLANT label: shifted left & lower
    too_xyz = (de_c[:, 0].max() + 1.8, yf, 35.0)            # natural-tooth label: right of the tooth, mid-height

    fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.52))

    # ---------- Panel A: labelled restored anatomy ----------
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    axA.add_collection3d(Poly3DCollection(surfB, facecolors=shade(surfB, to_rgb(COL["BONE"])),
                                          edgecolors="none", alpha=0.28, rasterized=True))
    for M in STRUCT:
        if M not in surf:
            continue
        if M == "CROWN" and REALCROWN:
            continue                              # replaced by the real-tooth overlay below
        tris, _ = surf[M]
        al = 0.55 if M == "BIOFILM" else 1.0      # biofilm is a thin sulcular film -> translucent
        pc = Poly3DCollection(tris, facecolors=shade(tris, to_rgb(COL[M]), *SH[M]),
                              edgecolors=(0, 0, 0, 0.10), linewidths=0.05, rasterized=True)
        pc.set_alpha(al)
        axA.add_collection3d(pc)
    # illustrative peri-implant mucosa cuff (transmucosal-consistent: bone crest -> margin)
    if GINGIVA:
        gum = gingiva_cuff()
        axA.add_collection3d(Poly3DCollection(gum, facecolors=shade(gum, to_rgb(COL["GINGIVA"]), 0.6, 0.35),
                                              edgecolors=(0.6, 0.2, 0.3, 0.25), linewidths=0.05,
                                              alpha=0.55, rasterized=True))
    # illustrative REAL molar crown (cusps/grooves) seated on the abutment (z 31->38)
    if REALCROWN:
        cm = real_tooth_crown(STL_TOOTH, h_target=cr_c[:, 2].max() - 31.0, base_z=31.0)
        axA.add_collection3d(Poly3DCollection(cm, facecolors=shade(cm, to_rgb(COL["CROWN"]), 0.62, 0.34, 0.30),
                                              edgecolors=(0, 0, 0, 0.12), linewidths=0.04, rasterized=True))
    label3d(axA, *imp_xyz, "IMPLANT (screw + crown)", "#222222")
    label3d(axA, *too_xyz, "natural tooth", "#b35806")
    if "BIOFILM" in surf:
        label3d(axA, T23[0] - 3.2, yf, CREST + 1.6, "biofilm (sulcus)", "#b30000")
    axA.text(crown_cx, yf, cr_c[:, 2].max() + 3.0, r"$\downarrow$ 100 N, 30$^\circ$ (ISO 14801)",
             fontsize=5.4, ha="center", color="#b30000", fontweight="bold")
    axA.set_title(r"(A) restored implant: load-bearing ceramic crown", fontsize=6.6, pad=1)
    style3d(axA, b)
    LAB["GINGIVA"] = "gingiva (illustr.)"
    if REALCROWN:
        LAB["CROWN"] = "crown (real tooth, illustr.)"
    present = [k for k in ["CROWN", "GINGIVA", "TI", "DENTIN", "PDL", "BIOFILM", "BONE"]
               if (k == "BONE" or (k == "GINGIVA" and GINGIVA) or (mat == k).any())]
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=COL[k], mec="0.5", mew=0.3, label=LAB[k])
               for k in present]
    axA.legend(handles=handles, fontsize=4.6, loc="lower left", handletextpad=0.25,
               labelspacing=0.28, framealpha=0.85, bbox_to_anchor=(-0.02, -0.02))

    # ---------- Panel B: von Mises (crown included) ----------
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    allv = np.concatenate([surf[M][1] for M in surf])
    vclip = float(np.nanpercentile(allv, 80)); norm = Normalize(0, vclip); cmap = plt.get_cmap("viridis")
    axB.add_collection3d(Poly3DCollection(surfB, facecolors=(0.7, 0.7, 0.7, 0.06),
                                          edgecolors="none", rasterized=True))
    for M in STRUCT:
        if M not in surf:
            continue
        tris, fv = surf[M]
        base = cmap(norm(np.clip(np.nan_to_num(fv), 0, vclip)))[:, :3]
        sh = _brightness(tris, 0.72, 0.28)
        pc = Poly3DCollection(tris, edgecolors="none", rasterized=True)
        pc.set_facecolor(np.clip(base * sh[:, None], 0, 1))
        axB.add_collection3d(pc)
    label3d(axB, *imp_xyz, "crown", "#222222")
    # moment-arm delta vs the no-crown baseline
    pk_crown = peri_implant_bone_peak(FIELD)
    pk_gen = peri_implant_bone_peak(GEN_FIELD) if GEN_FIELD.exists() else None
    if pk_gen:
        axB.text2D(0.02, 0.04,
                   r"crestal peri-implant bone $\sigma_\mathrm{vM}$ (p95):" + "\n"
                   + r"%.0f $\to$ %.0f MPa ($\times$%.1f) via crown moment arm"
                   % (pk_gen, pk_crown, pk_crown / pk_gen),
                   transform=axB.transAxes, fontsize=5.4, color="#b30000",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b30000", lw=0.5, alpha=0.9))
    axB.set_title(r"(B) occlusal von Mises $\sigma_\mathrm{vM}$ (crown included)", fontsize=6.6, pad=-4)
    style3d(axB, b)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axB, fraction=0.028, pad=0.0, shrink=0.55)
    cb.set_label(r"$\sigma_\mathrm{vM}$ (MPa)", fontsize=6); cb.ax.tick_params(labelsize=5)

    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.0, top=0.99, wspace=0.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_crown_fem.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "fem_implant_crown_fem.png", dpi=300, bbox_inches="tight")
    print("wrote fem_implant_crown_fem  struct-faces=%d bone-faces=%d vclip=%.2f bone-peak %.0f->%.0f"
          % (sum(len(surf[M][0]) for M in surf), len(surfB), vclip, pk_gen or 0, pk_crown))


if __name__ == "__main__":
    main()
