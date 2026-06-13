"""Clinical schematic: why dysbiosis detaches the biofilm at the implant interface (CH vs DH).

A single intuitive cross-section figure for the thesis FEM chapter. Two side-by-side titanium-implant
thread cross-sections (commensal CH vs dysbiotic DH) make the mechanism visible at a glance:
  * biofilm thickness from CLSM (CH ~64 um, DH ~79 um),
  * measured P. gingivalis depth distribution (dots): CH spreads Pg through the film, DH packs it at the
    substratum (interface zone),
  * growth-induced interface stress concentrates where Pg sits -> a hot interface band in DH, faint in CH,
  * outcome: CH stays bonded, DH delaminates (crack at the curved thread root),
  * a data strip carries the headline interface-stress contrast DH/CH ~ 2.5x (implant-thread FEM).

Data-grounded: Pg depth distribution from the pooled CLSM profile; DH/CH stress ratio from the
implant-thread FEM interior peak (fig_implant_fem.py).

Run:  python masterarbeit_ansys_fem/extensions/fig_fem_clinical_schematic.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "masterarbeit_ansys_fem" / "extensions"))
from thesis_style import use  # noqa: E402
from b4_viscoelastic_growth_stress import depth_phi_pg  # noqa: E402

RED, BLUE = "#c0392b", "#1f6fb4"
TI = "#9aa3ab"          # titanium grey
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
RNG = np.random.default_rng(3)


def pg_profile(cond):
    """Measured endpoint phi_Pg(z) (pooled diffusion-fit profile, same source as the FEM stress and c5;
    DH packs absolute Pg at the substratum). Returned on a 0..1 depth grid (0 = substratum)."""
    zn, pg = depth_phi_pg(cond)                 # raw (absolute) Pg occupancy vs normalised depth
    grid = np.linspace(0, 1, 40)
    pgi = np.interp(grid, zn, pg)
    return grid, np.clip(pgi, 0, None)


def thread(ax, x0, x1, y_base, amp, n_teeth):
    """Draw a titanium implant thread cross-section (sawtooth) filling below y_base."""
    xs = np.linspace(x0, x1, 400)
    tooth = y_base - amp * (0.5 + 0.5 * np.abs(((xs - x0) / (x1 - x0) * n_teeth) % 1.0 * 2 - 1))
    verts = [(x0, y_base - amp - 0.18)] + list(zip(xs, tooth)) + [(x1, y_base - amp - 0.18)]
    ax.add_patch(Polygon(verts, closed=True, facecolor=TI, edgecolor="#5d646b", lw=0.8, zorder=2))
    # hatch a thin "interface" just above the crest
    ax.plot(xs, tooth, color="#5d646b", lw=0.8, zorder=3)
    return xs, tooth


def panel(ax, cond, col, h_um, detaches):
    x0, x1 = 0.0, 1.0
    y_base = 0.30
    amp = 0.10
    xs, crest = thread(ax, x0, x1, y_base, amp, n_teeth=2)

    # biofilm layer on top of the crest, thickness ~ h
    h = 0.40 * (h_um / 79.0)                     # visual thickness scaled to DH=ref
    top = crest + h
    band = list(zip(xs, crest)) + list(zip(xs[::-1], top[::-1]))
    ax.add_patch(Polygon(band, closed=True, facecolor=col, alpha=0.10, edgecolor=col, lw=1.0, zorder=1))

    # interface stress hot band (concentrated near the crest), intensity from Pg-at-base
    grid, pg = pg_profile(cond)
    pg_base = pg[grid < 0.25].mean()             # Pg in interface zone
    # heat strip just above crest: alpha grows with pg_base
    for k, frac in enumerate(np.linspace(0, 0.10, 24)):
        a = (1 - frac / 0.10) * (pg_base / 0.10) * 0.55
        ax.fill_between(xs, crest + frac, crest + frac + 0.0045,
                        color=RED, alpha=float(np.clip(a, 0, 0.6)), lw=0, zorder=2)

    # Pg dots sampled by the measured depth distribution
    ndot = 150
    zc = RNG.choice(grid, size=ndot, p=pg / pg.sum())
    xpos = x0 + 0.04 + RNG.random(ndot) * (x1 - x0 - 0.08)
    ypos = np.interp(xpos, xs, crest) + zc * h
    ax.scatter(xpos, ypos, s=5, color=RED, alpha=0.7, edgecolor="none", zorder=4)

    # outcome
    if detaches:
        # jagged delamination crack along the interface
        xc = np.linspace(x0 + 0.08, x1 - 0.08, 60)
        yc = np.interp(xc, xs, crest) + 0.012 + 0.010 * RNG.standard_normal(60).cumsum() / 8
        ax.plot(xc, yc, color="k", lw=1.4, zorder=6)
        ax.annotate("DELAMINATES", xy=(0.5, np.interp(0.5, xs, crest) + 0.02),
                    xytext=(0.5, top.max() + 0.10), ha="center", fontsize=8, color="k",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
    else:
        ax.text(0.5, top.max() + 0.07, "stays bonded", ha="center", fontsize=8, color="#2e7d32")

    # labels
    ax.text(0.5, y_base - amp - 0.12, "Ti implant thread", ha="center", fontsize=6.5, color="#3d444b")
    ax.text(0.5, top.max() + 0.20, r"\textbf{%s}  ($h\approx%d\,\mu$m)" % (cond, h_um),
            ha="center", fontsize=9, color=col)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.0, 1.0); ax.axis("off")
    return pg_base


def main():
    use()  # set thesis rc
    fig = plt.figure(figsize=(6.6, 3.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 0.9], hspace=0.05, wspace=0.06)
    axCH = fig.add_subplot(gs[0, 0]); axDH = fig.add_subplot(gs[0, 1])
    axbar = fig.add_subplot(gs[1, :])

    bCH = panel(axCH, "CH", BLUE, h_um=64, detaches=False)
    bDH = panel(axDH, "DH", RED, h_um=79, detaches=True)

    # shared legend bits
    axCH.scatter([], [], s=12, color=RED, label=r"\textit{P. gingivalis}")
    axCH.fill_between([], [], color=RED, alpha=0.4, label="interface stress")
    axCH.legend(fontsize=6, loc="upper left", frameon=False)

    # data strip: FEM-grounded interface-stress contrast (two bars, DH = 3.3x CH)
    axbar.axis("off"); axbar.set_xlim(0, 1); axbar.set_ylim(0, 1)
    axbar.text(0.02, 0.80, r"Interface von Mises stress (implant-thread FEM): dysbiotic vs commensal",
               ha="left", fontsize=7.5, color="0.2")
    x_lab = 0.40
    unit = 0.16                                    # axes-width per "x" unit (so 3.3x -> ~0.53 wide)
    axbar.add_patch(Rectangle((x_lab, 0.50), 1.0 * unit, 0.20, color=BLUE, alpha=0.9))
    axbar.text(x_lab - 0.01, 0.60, "CH", ha="right", va="center", fontsize=7.5, color=BLUE)
    axbar.text(x_lab + 1.0 * unit + 0.01, 0.60, r"$1\times$", ha="left", va="center", fontsize=7)
    axbar.add_patch(Rectangle((x_lab, 0.14), 2.5 * unit, 0.20, color=RED, alpha=0.9))
    axbar.text(x_lab - 0.01, 0.24, "DH", ha="right", va="center", fontsize=7.5, color=RED)
    axbar.text(x_lab + 2.5 * unit + 0.01, 0.24, r"$\mathbf{2.5\times}$", ha="left", va="center",
               fontsize=8.5, color=RED)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_clinical_schematic.pdf", bbox_inches="tight")
    plt.close(fig)
    print("CH Pg-at-interface=%.3f  DH Pg-at-interface=%.3f" % (bCH, bDH))
    print("wrote", OUT / "fem_clinical_schematic.pdf")


if __name__ == "__main__":
    main()
