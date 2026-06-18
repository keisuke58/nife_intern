"""Element-FORMULATION robustness of the implant-thread interface stress.

Reads coupling_prototype/abaqus/elform_results.csv (produced by run_implant_elform.sh) and shows that
the headline linear CPE4 result is not a volumetric-locking / root-under-resolution artefact:

  (A) interior interface peak vM (DH & CH thread) vs element formulation -- at the headline n108
      resolution the absolute peak is INVARIANT to <2% across linear (CPE4), linear-hybrid (CPE4H),
      quadratic (CPE8), quadratic-reduced (CPE8R) and quadratic-reduced-hybrid (CPE8RH). The hybrid
      points coincide with the standard ones -> no volumetric locking at nu=0.45; the quadratic points
      coincide with the linear -> the smooth (finite-curvature) thread is already resolved, no root
      singularity to under-predict. (A coarse n72 mesh inflates the peak ~12% -> that was a resolution
      artefact, removed at n108.)
  (B) the dimensionless DH/CH and thread/flat RATIOS (the actual reported quantities) are FLAT across
      every formulation -> the dysbiosis ratio and "thread generates the stress" conclusion are robust.

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_elform.py
"""
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

CSV = ROOT / "masterarbeit_ansys_fem" / "coupling_prototype" / "abaqus" / "elform_results.csv"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
RED, BLUE, GREY = "#c0392b", "#1f6fb4", "0.6"

ORDER = ["CPE4", "CPE4H", "CPE8", "CPE8R", "CPE8RH"]
LABEL = {"CPE4": "CPE4\n(lin)", "CPE4H": "CPE4H\n(lin,hyb)", "CPE8": "CPE8\n(quad)",
         "CPE8R": "CPE8R\n(quad,red)", "CPE8RH": "CPE8RH\n(quad,red,hyb)"}


def load():
    d = {}
    for r in csv.DictReader(open(CSV)):
        d[(r["eltype"], r["cond"], r["profile"])] = float(r["interior_peak_vm"])
    return d


def main():
    use()
    d = load()
    vm = lambda el, c, p: d[(el, c, p)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    x = np.arange(len(ORDER))

    # (A) absolute interior peak, DH & CH thread -- near-flat across formulations
    dh = [vm(el, "DH", "thread") for el in ORDER]
    ch = [vm(el, "CH", "thread") for el in ORDER]
    sprd = lambda a: 100 * (max(a) - min(a)) / np.mean(a)
    axA.plot(x, dh, "o-", color=RED, lw=1.4, ms=5, label="DH (dysbiotic)")
    axA.plot(x, ch, "s-", color=BLUE, lw=1.4, ms=4, label="CH (commensal)")
    axA.axhspan(min(dh), max(dh), color=RED, alpha=0.08, zorder=0)
    axA.annotate(r"spread $\pm$%.1f\%%" % (sprd(dh) / 2), (x[2], max(dh)), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=6, color=RED)
    axA.text(0.30, 0.66, "hybrid = standard\n$\\Rightarrow$ no locking\nquad = linear\n$\\Rightarrow$ resolved",
             transform=axA.transAxes, fontsize=5.8, color="0.35", va="top", ha="center")
    axA.set_xticks(x); axA.set_xticklabels([LABEL[e] for e in ORDER], fontsize=5.6)
    axA.set_ylabel(r"interior peak $\sigma_\mathrm{vM}$ (arb.)")
    axA.set_title("(A) Interface peak is formulation-invariant", fontsize=7.5)
    axA.legend(fontsize=6, loc="upper right"); axA.set_ylim(0, max(dh) * 1.3)

    # (B) dimensionless ratios -- the reported quantities
    dhch = [vm(el, "DH", "thread") / vm(el, "CH", "thread") for el in ORDER]
    thfl = [vm(el, "DH", "thread") / vm(el, "DH", "flat") for el in ORDER]
    axB.plot(x, dhch, "o-", color=RED, lw=1.4, ms=5, label=r"DH/CH (dysbiosis)")
    axB.plot(x, thfl, "^-", color="#2e7d32", lw=1.4, ms=5, label=r"thread/flat (geometry)")
    for xi, v in zip(x, dhch):
        axB.annotate("%.2f" % v, (xi, v), textcoords="offset points", xytext=(0, 6), ha="center",
                     fontsize=5.6, color=RED)
    axB.set_xticks(x); axB.set_xticklabels([LABEL[e] for e in ORDER], fontsize=5.6)
    axB.set_ylabel(r"interior peak ratio ($\times$)")
    axB.set_title(r"(B) Reported ratios are formulation-invariant", fontsize=7.5)
    axB.legend(fontsize=6, loc="lower right")
    axB.set_ylim(0, max(max(dhch), max(thfl)) * 1.3)
    axB.text(0.03, 0.92, r"DH/CH spread $\pm$%.1f\%%, thread/flat $\pm$%.1f\%%" % (sprd(dhch), sprd(thfl)),
             transform=axB.transAxes, fontsize=5.8, color="0.4")

    fig.suptitle(r"Element-formulation robustness (n108): the interface peak and the DH/CH \& thread/flat "
                 r"ratios are invariant ($\pm$%.1f\%%) across linear, linear-hybrid, quadratic, quadratic-"
                 r"reduced \& quadratic-reduced-hybrid elements -- the headline CPE4 carries no volumetric-"
                 r"locking ($\nu{=}0.45$) or root-under-resolution artefact." % (sprd(dh) / 2),
                 fontsize=6.6, y=1.02)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_elform.pdf", bbox_inches="tight")
    plt.close(fig)
    print("DH thread peak: CPE4=%.1f -> quad plateau %.1f (%+.1f%%)" % (dh[0], dh[-1], 100 * (dh[-1] - dh[0]) / dh[0]))
    print("DH/CH ratio across formulations:", ["%.2f" % v for v in dhch], "spread %.1f%%" % sprd(dhch))
    print("thread/flat across formulations:", ["%.2f" % v for v in thfl], "spread %.1f%%" % sprd(thfl))
    print("wrote", OUT / "fem_implant_elform.pdf")


if __name__ == "__main__":
    main()
