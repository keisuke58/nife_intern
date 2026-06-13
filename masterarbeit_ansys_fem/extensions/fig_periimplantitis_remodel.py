"""Peri-implantitis FEM, study (2): mechanobiological bone remodelling (Frost/Carter mechanostat).
Reads FEM/tier2b_real/pimp_e4_bonefield.json (bone-element r,x,y,z,vM at 4 mm bone loss, 30deg load).
Strain-energy density SED = vM^2/(2E) is the remodelling signal; bone resorbs where SED exceeds the
pathological-overload window. The map shows the overload concentrates at the CRESTAL bone on the loaded
(buccal) side -- i.e. the FEM predicts resorption initiates there and progresses as a saucer-shaped
defect (clinical "saucerisation"), the spatial complement to the progression curve.
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_remodel.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
E_BONE = 13700.0
VM_OVERLOAD = 25.0          # vM (MPa) threshold for the pathological-overload window (Frost)


def main():
    use()
    d = json.load(open(FEM / "pimp_e4_bonefield.json"))
    x = np.array([r["x"] for r in d]); y = np.array([r["y"] for r in d])
    z = np.array([r["z"] for r in d]); vm = np.array([r["vm"] for r in d])
    sed = vm ** 2 / (2.0 * E_BONE) * 1e3        # mJ/mm^3 (kPa)
    m = np.abs(y) < 0.5                          # buccolingual (x-z) section through the load plane

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.52))

    # (A) SED map of the bone section
    vmax = np.percentile(sed[m], 98)
    sc = axA.scatter(x[m], z[m], c=sed[m], cmap="inferno", s=14, vmin=0, vmax=vmax, marker="s")
    axA.add_patch(plt.Rectangle((-2.05, 0), 4.1, 6.0, fill=False, ec="0.5", lw=0.8, ls="--"))
    axA.text(0, 0.3, "implant", ha="center", fontsize=6, color="0.5")
    axA.set_title(r"(A) strain-energy density (remodelling signal)", fontsize=7.3)
    axA.set_xlabel(r"buccolingual $x$ (mm)", fontsize=7); axA.set_ylabel(r"depth $z$ (mm)", fontsize=7)
    cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label(r"SED (kPa)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    axA.tick_params(labelsize=6); axA.set_aspect("equal")

    # (B) predicted resorption zone: bone elements in the (relative) overload window -- the upper
    # tail of the remodelling signal, which localises to the crestal ring.
    thr = np.percentile(vm[m], 88)
    over = m & (vm > thr)
    axB.scatter(x[m], z[m], c="0.82", s=14, marker="s")
    axB.scatter(x[over], z[over], c="#c0392b", s=16, marker="s", label="overload (resorb)")
    axB.add_patch(plt.Rectangle((-2.05, 0), 4.1, 6.0, fill=False, ec="0.5", lw=0.8, ls="--"))
    axB.axhline(6.0, color="0.5", lw=0.6, ls=":")
    axB.text(-4.3, 5.6, "bone level", fontsize=5.6, color="0.4")
    axB.annotate("crestal saucerisation\n(initiation site)", xy=(1.8, 5.6), xytext=(0.2, 3.4),
                 fontsize=6, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7))
    axB.set_title(r"(B) predicted resorption: crestal overload", fontsize=7.3)
    axB.set_xlabel(r"buccolingual $x$ (mm)", fontsize=7)
    axB.legend(fontsize=5.8, loc="lower right"); axB.tick_params(labelsize=6); axB.set_aspect("equal")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_remodel.pdf", bbox_inches="tight")
    plt.close(fig)
    print("overload bone elems in section: %d/%d  peak SED=%.1f kPa"
          % (over.sum(), m.sum(), sed[m].max()))
    print("wrote", OUT / "fem_periimplantitis_remodel.pdf")


if __name__ == "__main__":
    main()
