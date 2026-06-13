"""Peri-implantitis FEM, studies (1)+(3): progressive marginal bone loss and its biofilm-driven rate.
Reads FEM/tier2b_real/pimp_results.jsonl (bonded coupon, bone level lowered 2->8 mm).
  (A) the resorption -> stress / stiffness feedback: crestal-bone vM rises and coupon stiffness
      collapses as bone is lost -- the peri-implant vicious cycle;
  (B) biofilm severity sets the RATE: the dysbiotic interface drives resorption ~2.5x faster than the
      commensal (the Chapter-5 dysbiosis contrast), so DH reaches the accelerating regime far sooner.
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
DYS = 2.5   # dysbiotic/commensal resorption-rate ratio (= Chapter-5 interface-stress contrast)


def main():
    use()
    R = [json.loads(l) for l in open(FEM / "pimp_results.jsonl")]
    R.sort(key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R])   # marginal bone loss = mm below platform
    crest = np.array([r["crest_p95"] for r in R])
    sti = np.array([r["stiffness_N_per_um"] for r in R])
    thr = np.array([r["thread_exp"] for r in R])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.46))
    C1, C2 = "#b22222", "#1f5fa6"

    # (A) feedback: crestal stress + stiffness vs bone loss
    axA.plot(loss, crest, "o-", color=C1, lw=1.4, ms=4, label=r"crestal bone $\sigma_\mathrm{vM}$")
    axA.set_xlabel("marginal bone loss (mm)", fontsize=7)
    axA.set_ylabel(r"crestal bone $\sigma_\mathrm{vM}$ (MPa)", color=C1, fontsize=7)
    axA.tick_params(axis="y", labelcolor=C1, labelsize=6); axA.tick_params(axis="x", labelsize=6)
    a2 = axA.twinx()
    a2.plot(loss, sti, "s--", color=C2, lw=1.3, ms=4)
    a2.set_ylabel(r"coupon stiffness (N/$\mu$m)", color=C2, fontsize=7)
    a2.tick_params(axis="y", labelcolor=C2, labelsize=6)
    axA.set_title(r"(A) resorption $\to$ stress feedback (vicious cycle)", fontsize=7.3)
    axA.annotate("acceleration", xy=(loss[-1], crest[-1]), xytext=(loss[-1] - 2.2, crest[-1]),
                 fontsize=6, color=C1, arrowprops=dict(arrowstyle="->", color=C1, lw=0.7))

    # (B) biofilm-driven rate: DH vs CH time to reach each bone-loss level (rate ratio = DYS)
    # commensal rate normalised so CH reaches 6 mm at t=1 (arbitrary clinical unit); DH is DYS x faster
    t_CH = loss / loss.max()                  # normalised time for CH (rate 1)
    t_DH = t_CH / DYS                          # DH reaches same loss DYS x sooner
    axB.plot(t_CH, loss, "o-", color="#2b8cbe", lw=1.4, ms=4, label="commensal (CH)")
    axB.plot(t_DH, loss, "o-", color="#c0392b", lw=1.4, ms=4, label="dysbiotic (DH)")
    axB.axhline(6.0, color="0.5", lw=0.7, ls=":")
    axB.text(0.02, 6.2, "accelerating regime", fontsize=5.6, color="0.4")
    axB.set_xlabel("clinical time (normalised)", fontsize=7)
    axB.set_ylabel("marginal bone loss (mm)", fontsize=7)
    axB.tick_params(labelsize=6)
    axB.legend(fontsize=6, loc="lower right")
    axB.set_title(r"(B) biofilm severity sets the rate ($\times%.1f$ faster, DH)" % DYS, fontsize=7.3)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_progression.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fem_periimplantitis_progression.pdf")
    print("bone loss mm:", loss, "crest:", crest, "stiffness:", sti)


if __name__ == "__main__":
    main()
