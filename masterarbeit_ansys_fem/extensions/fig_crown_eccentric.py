"""Eccentric occlusal contact -> saucerisation direction (study 3).

Crowned ISO-14801 coupon (crown_h=7 mm), bonded, lateral bite offset (+x = buccal) 0->4 mm
(FEM/tier2b_real/ci_ecc_results.jsonl). (A) Polar distribution of crestal peri-implant bone stress
around the implant: the peak sits on the LOADED (buccal) side and grows with eccentricity -> the
saucer-shaped (saucerisation) bone defect starts there. (B) Peak crestal stress vs eccentricity. Even
a centred 30-deg oblique bite is already ~3x asymmetric around the ring; eccentric contact raises the
loaded-side peak further.

Run: python masterarbeit_ansys_fem/extensions/fig_crown_eccentric.py
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


def main():
    use()
    rows = sorted([json.loads(l) for l in open(FEM / "ci_ecc_results.jsonl")],
                  key=lambda r: float(r["tag"].split("_")[1]))
    ecc = np.array([float(r["tag"].split("_")[1]) for r in rows])

    fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.46))
    axA = fig.add_subplot(1, 2, 1, projection="polar")
    axB = fig.add_subplot(1, 2, 2)

    cmap = plt.get_cmap("YlOrRd")
    for r in rows:
        e = float(r["tag"].split("_")[1])
        if e not in (0, 2, 4):
            continue
        th = np.array(r["theta"]); p90 = np.array(r["p90"])
        th = np.append(th, th[0]); p90 = np.append(p90, p90[0])      # close the ring
        axA.plot(th, p90, "-", color=cmap(0.35 + 0.22 * e), lw=1.5, label="ecc %g mm" % e)
    axA.set_theta_zero_location("E")                                 # 0 deg = +x = buccal (loaded side)
    axA.set_title(r"(A) crestal bone $\sigma_\mathrm{vM}$ around the implant", fontsize=7.2, pad=10)
    axA.tick_params(labelsize=5.5)
    axA.set_rlabel_position(135)
    axA.annotate("buccal\n(loaded)", xy=(0, axA.get_rmax()), fontsize=5.6, color="#b30000",
                 ha="center", va="bottom")
    axA.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(-0.18, -0.12), framealpha=0.9)

    peak = np.array([r["peak_p90"] for r in rows])
    asym = np.array([r["asym"] for r in rows])
    axB.plot(ecc, peak, "o-", color="#d73027", lw=1.5, ms=4, label="loaded-side peak")
    axB.set_xlabel("eccentric bite offset (mm)", fontsize=7.5)
    axB.set_ylabel(r"loaded-side crestal peak $\sigma_\mathrm{vM}$ (MPa)", fontsize=7.5)
    axB.set_title(r"(B) eccentricity raises the loaded-side peak ($\times$%.2f)" % (peak[-1] / peak[0]),
                  fontsize=7.4)
    axB.tick_params(labelsize=6.5)
    axB2 = axB.twinx()
    axB2.plot(ecc, asym, "s--", color="0.5", lw=1.0, ms=3)
    axB2.set_ylabel("ring asymmetry max/min (dashed)", fontsize=6.6, color="0.4")
    axB2.tick_params(labelsize=6.0, colors="0.4"); axB2.set_ylim(2.5, 4)
    axB.legend(fontsize=6.3, loc="upper left", framealpha=0.9)

    fig.suptitle(r"Eccentric occlusal contact concentrates crestal bone stress on the loaded side "
                 r"$\Rightarrow$ saucerisation starts there", fontsize=7.7, y=1.0)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_crown_eccentric.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_crown_eccentric.png", dpi=200, bbox_inches="tight")
    print("wrote fem_crown_eccentric  peak %.1f->%.1f MPa (x%.2f)  peak@%.0fdeg (loaded side)"
          % (peak[0], peak[-1], peak[-1] / peak[0], rows[-1]["peak_theta_deg"]))


if __name__ == "__main__":
    main()
