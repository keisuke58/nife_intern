"""3-D visualisation of the helical implant-screw biofilm interface stress.

Reads the C3D8 helical-screw odb extract (coupling_prototype/abaqus/screw3d_*_3d.json) and renders the
INNER (bonded) interface layer coloured by von Mises stress: (left) a 3-D perspective of the screw,
(right) the unrolled (azimuth theta, axial z) map, where the helical thread shows as diagonal stress
bands. This is the genuinely-3-D model behind the implant result.

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_3d.py [job]
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


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "screw3d_DH"
    e = json.load(open(AB / ("%s_3d.json" % job)))["els"]
    x = np.array([r["x"] for r in e]); y = np.array([r["y"] for r in e]); z = np.array([r["z"] for r in e])
    r = np.array([r["r"] for r in e]); vm = np.array([r_["vm"] for r_ in e])
    # inner (bonded interface) layer = innermost radial band
    rin = np.quantile(r, 0.25)
    m = r <= rin
    vmin = np.quantile(vm[m], 0.10); vmax = np.quantile(vm[m], 0.98)   # stretch contrast on the modulation

    use()
    fig = plt.figure(figsize=(7.0, 3.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    p = ax.scatter(x[m], y[m], z[m], c=vm[m], cmap="inferno", s=4, vmin=vmin, vmax=vmax, alpha=0.9)
    ax.set_title(r"(A) Helical screw interface $\sigma_\mathrm{vM}$", fontsize=7.5)
    ax.set_xlabel("x", fontsize=6); ax.set_ylabel("y", fontsize=6); ax.set_zlabel("z", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=18, azim=35)
    try:
        ax.set_box_aspect((1, 1, 2))
    except Exception:
        pass

    # unrolled (theta, z) interface map -> helical bands appear diagonal
    a2 = fig.add_subplot(1, 2, 2)
    th = np.degrees(np.arctan2(y[m], x[m])) % 360
    sc = a2.scatter(th, z[m], c=vm[m], cmap="inferno", s=6, vmin=vmin, vmax=vmax)
    a2.set_xlabel(r"azimuth $\theta$ (deg)"); a2.set_ylabel(r"axial $z$")
    a2.set_title(r"(B) Unrolled interface (helical bands)", fontsize=7.5)
    a2.set_xlim(0, 360)
    cb = fig.colorbar(sc, ax=a2, fraction=0.046, pad=0.03)
    cb.set_label(r"$\sigma_\mathrm{vM}$ (arb.)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)

    fig.suptitle(r"3-D helical implant-screw biofilm FEM (C3D8): growth-induced interface stress "
                 r"follows the thread helix", fontsize=7.2, y=1.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_3d.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("inner-layer elems=%d  max vM=%.0f" % (m.sum(), vm[m].max()))
    print("wrote", OUT / "fem_implant_3d.pdf")


if __name__ == "__main__":
    main()
