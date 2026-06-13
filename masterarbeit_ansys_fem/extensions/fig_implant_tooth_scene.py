"""Tier-1 combined anatomical scene: implant + adjacent natural tooth, peri-implant & peri-tooth biofilm.

Places, in the real mandible coordinate frame (Open-Full-Jaw Patient 1), the conformal biofilm on a real
adjacent tooth (tooth 24, solved in Abaqus with DH growth) next to the parametric implant screw (3-D
helical, DH growth) positioned at the neighbouring tooth-23 site. Both biofilms carry the SAME dysbiotic
growth; the figure shows where the growth-induced interface stress concentrates on each --- peri-implant
(implant-thread interface, peri-implantitis) versus peri-tooth (cervical/root interface, periodontitis)
--- in one realistic scene. Absolute moduli/units differ between the two models, so each is shown on its
own relative von-Mises scale (the comparison is spatial/qualitative; per-model dysbiosis contrasts are
quantified elsewhere). This is Tier 1 (bone as a rigid bonded boundary); Tier 2 adds volumetric bone.

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_tooth_scene.py
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
FEM = Path("/home/nishioka/IKM_Hiwi/FEM")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
TOOTH23 = np.array([-69.4, -41.0, 29.0])     # implant site (real jaw coords)


def load(path):
    e = json.load(open(path))["els"]
    return (np.array([r["x"] for r in e]), np.array([r["y"] for r in e]),
            np.array([r["z"] for r in e]), np.array([r["vm"] for r in e]))


def norm(v, lo=0.10, hi=0.98):
    a, b = np.quantile(v, lo), np.quantile(v, hi)
    return np.clip((v - a) / (b - a + 1e-12), 0, 1)


def main():
    use()
    # adjacent natural tooth (real jaw coords)
    xt, yt, zt, vt = load(FEM / "tooth24_biofilm_DH_3d.json")
    # implant (normalized coords) -> place at tooth-23 site, ~4.3 mm dia x ~10 mm long, axis along z
    xi, yi, zi, vi = load(AB / "screw3d_DH_3d.json")
    sxy, sz = 0.7, 3.3
    Xi = TOOTH23[0] + xi * sxy
    Yi = TOOTH23[1] + yi * sxy
    Zi = (TOOTH23[2] - 4.0) + (zi - 1.5) * sz       # root region (lower z), ~10 mm span

    # downsample for a clean scatter
    rng = np.random.default_rng(0)
    def ds(n, k=6000): return rng.choice(n, size=min(k, n), replace=False)
    it = ds(len(xt)); ii = ds(len(Xi))

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xt[it], yt[it], zt[it], c=norm(vt)[it], cmap="inferno", s=4, alpha=0.85)
    sc = ax.scatter(Xi[ii], Yi[ii], Zi[ii], c=norm(vi)[ii], cmap="inferno", s=7, alpha=0.95)
    ax.set_xlabel("x (mm)", fontsize=6); ax.set_ylabel("y (mm)", fontsize=6); ax.set_zlabel("z (mm)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=14, azim=-72)
    # labels (relative stress, each model self-normalised)
    ax.text(TOOTH23[0] - 1, TOOTH23[1], TOOTH23[2] - 11, "implant\n(peri-implant\ninterface)",
            fontsize=6.5, color="#b03020", ha="center", fontweight="bold")
    ax.text(xt.mean() + 2, yt.mean(), zt.max() + 2, "natural tooth 24\n(peri-tooth interface)",
            fontsize=6.5, color="#1b5e20", ha="center", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.04)
    cb.set_label(r"interface $\sigma_\mathrm{vM}$ (relative, per model)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5)
    fig.suptitle(r"Tier-1 anatomical scene (Open-Full-Jaw mandible frame): implant at the tooth-23 site "
                 r"$+$ adjacent natural tooth 24, both under dysbiotic biofilm growth", fontsize=7, y=0.97)
    try:
        ax.set_box_aspect((1, 1, 1.7))
    except Exception:
        pass

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_tooth_scene.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("tooth elems=%d (vM max=%.0f), implant elems=%d" % (len(xt), vt.max(), len(Xi)))
    print("wrote", OUT / "fem_implant_tooth_scene.pdf")


if __name__ == "__main__":
    main()
