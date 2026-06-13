"""Peri-implantitis FEM, study (4): loss of osseointegration -> frictional micromotion (Brunski).
Reads FEM/tier2b_real/contact_results.jsonl (fully debonded 0%-BIC implant, frictional general contact,
ISO 14801 30deg / 100 N, bone level 4/6/8 mm below platform). Plots peak interface micromotion vs bone
loss against the Brunski ~150 um fibrous-encapsulation threshold.
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_micromotion.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
BRUNSKI = 150.0


def main():
    use()
    R = [json.loads(l) for l in open(FEM / "contact_results.jsonl")]
    R.sort(key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R])
    mmax = np.array([r["micromotion_max_um"] for r in R])

    fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.74))
    ax.plot(loss, mmax, "o-", color="#c0392b", lw=1.5, ms=5, label="peak micromotion (100 N)")
    ax.axhline(BRUNSKI, color="0.3", lw=1.0, ls="--")
    ax.text(loss.min(), BRUNSKI * 1.04, r"Brunski $\sim$150\,$\mu$m (fibrous encapsulation)",
            fontsize=6, color="0.3")
    # indicative parafunctional-load band (micromotion scales ~linearly with bite force)
    ax.fill_between(loss, mmax, mmax * 6.0, color="#c0392b", alpha=0.12)
    ax.text(loss.mean(), mmax.mean() * 3.0, "parafunctional\nload range\n($\\times$ bite force)",
            fontsize=5.6, color="#a03020", ha="center")
    ax.set_yscale("log"); ax.set_ylim(3, 300)
    ax.set_xlabel("marginal bone loss (mm)", fontsize=7.5)
    ax.set_ylabel(r"peak interface micromotion ($\mu$m)", fontsize=7.5)
    ax.set_title(r"(4) debonded implant: micromotion vs.\ Brunski threshold", fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.legend(fontsize=6.2, loc="lower right")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_micromotion.pdf", bbox_inches="tight")
    plt.close(fig)
    print("loss", loss, "micromotion um", mmax)
    print("wrote", OUT / "fem_periimplantitis_micromotion.pdf")


if __name__ == "__main__":
    main()
