"""Axisymmetric implant-screw FEM: hoop-confinement interface stress vs implant curvature.

A biofilm on a cylindrical titanium implant is confined in the hoop direction; even a smooth cylinder
(no thread) carries an interface stress that a flat (plane-strain) surface does not. This plots the
inner-interface hoop stress |S33| from the CAX4 axisymmetric decks (coupling_prototype/abaqus/axi_*),
versus implant radius R0, showing it scales with curvature 1/R0; dysbiosis adds DH/CH ~2.3x.

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_axi.py
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
RED, BLUE = "#c0392b", "#1f6fb4"


def hoop(job):
    return json.load(open(AB / ("%s_axi.json" % job)))["peak_hoop"]


def main():
    use()
    R = [0.5, 1.0, 2.0, 4.0]
    tags = ["axi_DH_R0p5", "axi_DH_R1p0", "axi_DH_R2p0", "axi_DH_R4p0"]
    hDH = [hoop(t) for t in tags]
    hCH = hoop("axi_CH_R1p0")

    fig, ax = plt.subplots(figsize=use(width_frac=0.6, aspect=0.72))
    ax.plot(R, hDH, "o-", color=RED, lw=1.5, ms=5, label="DH (dysbiotic)")
    ax.scatter([1.0], [hCH], s=45, facecolor="white", edgecolor=BLUE, lw=1.4, zorder=5)
    ax.annotate(r"CH ($\mathrm{DH/CH}\approx%.1f\times$)" % (hoop("axi_DH_R1p0") / hCH),
                xy=(1.0, hCH), xytext=(1.6, hCH - 35), fontsize=6.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    ax.axhline(0, ls=":", color="0.6", lw=0.8)
    ax.text(3.4, 12, "flat (plane strain): hoop $=0$", fontsize=6, color="0.45")
    ax.annotate("microthread /\nsmall radius", xy=(0.5, hDH[0]), xytext=(0.9, hDH[0] - 5),
                fontsize=6, color="0.4", arrowprops=dict(arrowstyle="->", color="0.6", lw=0.7))
    ax.annotate("implant body", xy=(4.0, hDH[-1]), xytext=(2.7, hDH[-1] + 45), fontsize=6, color="0.4",
                arrowprops=dict(arrowstyle="->", color="0.6", lw=0.7))
    ax.set_xlabel(r"implant radius $R_0$ (film thicknesses)")
    ax.set_ylabel(r"inner-interface hoop stress $|S_{33}|$ (arb.)")
    ax.set_title(r"Cylindrical curvature generates hoop interface stress ($\propto 1/R_0$)", fontsize=7.5)
    ax.legend(fontsize=6.5, loc="upper right")

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_axi.pdf", bbox_inches="tight")
    plt.close(fig)
    print("DH hoop vs R0:", dict(zip(R, [round(x) for x in hDH])), " CH@R1=%.0f" % hCH)
    print("wrote", OUT / "fem_implant_axi.pdf")


if __name__ == "__main__":
    main()
