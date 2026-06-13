"""Restoration DESIGN sweep (study 2): which prosthetic knob drives peri-implant bone stress?

(A) Crown HEIGHT (moment arm) -- ISO-14801 coupon, bonded, expose=1 mm (FEM/tier2b_real/
design_results.jsonl): crestal bone stress, implant-neck Ti stress and crown displacement all climb
steeply with crown height. (B) Crown MATERIAL -- full coupled assembly, crown Young's modulus 10->210
GPa (composite -> e.max -> zirconia): the peri-implant bone peak is essentially FLAT (~1.6% over a 21x
stiffness range). Conclusion: crown HEIGHT is the dominant design variable; crown MATERIAL is secondary
for the bone (it only changes the crown's own stress) -- height/moment-arm, not stiffness, drives
marginal-bone risk.

Run: python masterarbeit_ansys_fem/extensions/fig_crown_design.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
# crown-material sweep (full assembly, peri-implant bone peak vM, MPa): E(MPa) -> bone peak
MAT = [(10000, 89.4, "composite"), (95000, 88.3, "lithium-disilicate"), (210000, 88.0, "zirconia")]


def main():
    use()
    rows = sorted([json.loads(l) for l in open(FEM / "design_results.jsonl")], key=lambda r: r["crown_h"])
    h = np.array([r["crown_h"] for r in rows])
    crest = np.array([r["crest_p95"] for r in rows])
    ti = np.array([r["ti_max"] for r in rows])
    disp = np.array([r["disp_um"] for r in rows])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.46))

    # ---------- (A) crown height ----------
    axA.plot(h, crest, "o-", color="#d73027", lw=1.5, ms=4, label=r"crestal bone $\sigma_\mathrm{vM}$")
    axA.plot(h, ti / 10.0, "s--", color="#4575b4", lw=1.2, ms=3.5, label=r"implant-neck Ti $\sigma_\mathrm{vM}/10$")
    axA.set_xlabel("crown height / moment arm (mm)", fontsize=7.5)
    axA.set_ylabel(r"$\sigma_\mathrm{vM}$ (MPa)", fontsize=7.5)
    axA.set_title(r"(A) crown HEIGHT dominates ($\times$%.1f over 0$\to$10 mm)" % (crest[-1] / crest[0]),
                  fontsize=7.4)
    axA.legend(fontsize=6.3, loc="upper left", framealpha=0.9); axA.tick_params(labelsize=6.5)
    axD = axA.twinx()
    axD.plot(h, disp, "^:", color="0.5", lw=1.0, ms=3)
    axD.set_ylabel(r"crown displacement ($\mu$m, dotted)", fontsize=6.6, color="0.4")
    axD.tick_params(labelsize=6.0, colors="0.4")

    # ---------- (B) crown material ----------
    E = np.array([m[0] for m in MAT]) / 1000.0
    bone = np.array([m[1] for m in MAT])
    axB.plot(E, bone, "o-", color="#1a9850", lw=1.5, ms=5)
    for (e, b, nm) in MAT:
        axB.annotate(nm, xy=(e / 1000.0, b), xytext=(e / 1000.0, b + 0.35), fontsize=5.8,
                     ha="center", color="0.3")
    axB.set_ylim(bone.min() - 2, bone.max() + 2)
    axB.set_xlabel("crown Young's modulus (GPa)", fontsize=7.5)
    axB.set_ylabel(r"peri-implant bone peak $\sigma_\mathrm{vM}$ (MPa)", fontsize=7.5)
    axB.set_title(r"(B) crown MATERIAL is secondary ($<2\%$ over 21$\times$)", fontsize=7.4)
    axB.tick_params(labelsize=6.5)
    axB.axhspan(bone.min() - 0.3, bone.max() + 0.3, color="#1a9850", alpha=0.08)

    fig.suptitle(r"Restoration design: the crown HEIGHT (moment arm) drives peri-implant bone stress, "
                 r"not the crown material", fontsize=7.8, y=1.0)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_crown_design.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_crown_design.png", dpi=200, bbox_inches="tight")
    print("wrote fem_crown_design  height x%.2f (crest %.1f->%.1f) ; material flat %.1f-%.1f MPa"
          % (crest[-1] / crest[0], crest[0], crest[-1], bone.min(), bone.max()))


if __name__ == "__main__":
    main()
