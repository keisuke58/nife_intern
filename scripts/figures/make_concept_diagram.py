#!/usr/bin/env python3
"""
make_concept_diagram.py — high-quality CONCEPTUAL diagram (graphical abstract)
for the oral-biofilm project. Schematic only (no data panels): it states the
central idea — genome-scale metabolism constrains the SIGN of ecological
interactions, and dysbiosis is a rewiring + spatial reorganisation, not a bulk
compositional shift.

Three zones, left -> right:
  (1) the defined community (5 species on a Ti implant / in-vivo 16S guilds)
  (2) the principle: AGORA FBA cross-feeding -> sign of A_ij  (magnitude from data)
  (3) commensal <-> dysbiotic: mutualism breaks, P. gingivalis centralises & sinks deep
      -> peri-implantitis

Vector PDF + PNG.  Run from repo root:  python scripts/figures/make_concept_diagram.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

HERE = Path(__file__).resolve().parents[2]
OUTDIR = HERE / "results" / "figures"
NAVY, INK, MUT = "#1a3a5c", "#222222", "#555555"
GREEN, RED = "#2e7d4f", "#b5341f"
SP = [("S. oralis", "#1f77b4"), ("A. naeslundii", "#2ca02c"),
      ("V. dispar/parvula", "#bcbd22"), ("F. nucleatum", "#9467bd"),
      ("P. gingivalis", "#d62728")]


def zone_header(ax, x, y, n, text):
    ax.add_patch(Circle((x, y), 0.22, color=NAVY, zorder=5))
    ax.text(x, y, str(n), color="white", ha="center", va="center",
            fontsize=13, weight="bold", zorder=6)
    ax.text(x + 0.42, y, text, color=NAVY, ha="left", va="center",
            fontsize=14.5, weight="bold")


def rbox(ax, x, y, w, h, fc, ec, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def arrow(ax, p0, p1, color=NAVY, lw=2.6, ms=22, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, ls=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.0, 7.4))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.4); ax.axis("off")

    fig.text(0.5, 0.955, "Metabolism constrains the ecology of the peri-implant biofilm",
             ha="center", fontsize=18, weight="bold", color=NAVY)
    fig.text(0.5, 0.912,
             "genome-scale cross-feeding fixes the $sign$ of interactions; "
             "dysbiosis is a re-wiring & spatial re-organisation — not a bulk shift",
             ha="center", fontsize=11.5, color=MUT, style="italic")

    # background zones
    for x0 in (0.25, 5.35, 10.2):
        ax.add_patch(Rectangle((x0, 0.5), 4.55 if x0 < 10 else 4.55, 6.0,
                               fc="#f5f7fa", ec="#dde4ec", lw=1.0, zorder=0))

    # ================= Zone 1: the defined community =================
    zone_header(ax, 0.65, 6.15, 1, "Defined community")
    ax.text(0.5, 5.62, "5-species HOBIC biofilm (in vitro)\n+ 10-guild in-vivo 16S (Dieckow)",
            fontsize=10.2, color=INK, va="top")
    # species discs
    cx, cy = 1.15, 4.35
    for i, (name, col) in enumerate(SP):
        yy = cy - i * 0.62
        ax.add_patch(Circle((cx, yy), 0.20, color=col, ec="white", lw=1.0, zorder=3))
        ax.text(cx + 0.42, yy, name, fontsize=10.5, va="center", color=INK, style="italic")
    # Ti implant bar
    ax.add_patch(FancyBboxPatch((0.5, 0.78), 4.05, 0.42,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc="#c9ced4", ec="#9aa1aa", lw=1.0, zorder=2))
    ax.text(2.52, 0.99, "titanium implant surface", ha="center", va="center",
            fontsize=9.5, color="#333")

    # ================= Zone 2: the principle =================
    zone_header(ax, 5.75, 6.15, 2, "The principle")
    rbox(ax, 5.55, 4.55, 4.15, 1.15, "#eef3f8", NAVY)
    ax.text(7.62, 5.45, "AGORA2 genome-scale FBA", ha="center", fontsize=11.2,
            weight="bold", color=NAVY)
    ax.text(7.62, 5.02, "who secretes / consumes which metabolite",
            ha="center", fontsize=9.6, color=INK)
    arrow(ax, (7.62, 4.5), (7.62, 4.02))
    # cross-feeding rule
    rbox(ax, 5.55, 2.95, 4.15, 1.0, "#eaf3ec", GREEN)
    ax.text(7.62, 3.62, r"$j$ secretes $X$, $i$ consumes $X$", ha="center", fontsize=10.2, color=INK)
    ax.text(7.62, 3.20, r"$\Rightarrow$ cross-feeding $\Rightarrow\ A_{ij}>0$",
            ha="center", fontsize=11.0, color=GREEN, weight="bold")
    arrow(ax, (7.62, 2.9), (7.62, 2.5), color=MUT, lw=2.0)
    rbox(ax, 5.55, 1.55, 4.15, 0.9, "#f7ece9", RED)
    ax.text(7.62, 2.16, r"$j$ secretes a toxin (H$_2$O$_2$, H$_2$S)", ha="center", fontsize=10.0, color=INK)
    ax.text(7.62, 1.78, r"$\Rightarrow\ A_{ij}<0$", ha="center", fontsize=11.0, color=RED, weight="bold")
    ax.text(7.62, 1.16, r"$\mathbf{sign}(A)$ from metabolism · magnitude from data",
            ha="center", fontsize=10.2, color=NAVY)

    # ================= Zone 3: commensal <-> dysbiotic =================
    zone_header(ax, 10.6, 6.15, 3, "Commensal ⇄ dysbiotic")
    # commensal sub-panel
    rbox(ax, 10.35, 3.55, 4.3, 2.15, "#f1f8f3", GREEN, lw=1.6)
    ax.text(12.5, 5.45, "COMMENSAL", ha="center", fontsize=11, weight="bold", color=GREEN)
    _mutualism(ax, 11.1, 4.45, "#1f77b4", "#bcbd22", "So ⇄ Vd  mutualism", GREEN)
    ax.text(13.35, 4.45, "shallow,\nbalanced", ha="center", va="center",
            fontsize=9.2, color=INK)
    # dysbiotic sub-panel
    rbox(ax, 10.35, 1.05, 4.3, 2.15, "#fbf0ee", RED, lw=1.6)
    ax.text(12.5, 2.95, "DYSBIOTIC", ha="center", fontsize=11, weight="bold", color=RED)
    _broken(ax, 11.1, 1.95, "#1f77b4", "#bcbd22")
    ax.text(11.1, 1.30, "mutualism breaks", ha="center", fontsize=8.6, color=RED)
    # Pg deep + central
    ax.add_patch(Circle((13.2, 1.7), 0.30, color="#d62728", ec="white", lw=1.2, zorder=3))
    ax.text(13.2, 1.7, "Pg", color="white", ha="center", va="center", fontsize=9, weight="bold")
    ax.annotate("", xy=(13.2, 1.18), xytext=(13.2, 2.05),
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.0))
    ax.text(13.85, 1.7, "central\n& deep", ha="center", va="center", fontsize=9.0, color=RED)

    # commensal <-> dysbiotic vertical double arrow
    ax.annotate("", xy=(12.5, 3.45), xytext=(12.5, 3.30),
                arrowprops=dict(arrowstyle="<->", color=MUT, lw=2.2))

    # big inter-zone arrows
    arrow(ax, (4.7, 3.6), (5.5, 3.6))
    arrow(ax, (9.78, 3.6), (10.3, 3.6))

    # bottom validation ribbon
    ax.add_patch(FancyBboxPatch((0.25, 0.04), 14.4, 0.40,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc=NAVY, ec="none", zorder=2))
    ax.text(7.45, 0.24, "validated:  cross-feeding direction $p=4\\times10^{-4}$   ·   "
                        "two cohorts (Dieckow × Duran-Pinedo) 89% prior-free   ·   "
                        "mechanistic COMETS dFBA reproduces the split",
            ha="center", va="center", color="white", fontsize=10.0, zorder=3)

    png = OUTDIR / "concept_overview_pub.png"
    pdf = OUTDIR / "concept_overview_pub.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", png)
    print("wrote", pdf)


def _mutualism(ax, x, y, c1, c2, label, frame):
    ax.add_patch(Circle((x, y), 0.18, color=c1, ec="white", lw=0.8, zorder=3))
    ax.add_patch(Circle((x + 0.9, y), 0.18, color=c2, ec="white", lw=0.8, zorder=3))
    ax.annotate("", xy=(x + 0.70, y + 0.10), xytext=(x + 0.20, y + 0.10),
                arrowprops=dict(arrowstyle="-|>", color=frame, lw=1.6))
    ax.annotate("", xy=(x + 0.20, y - 0.10), xytext=(x + 0.70, y - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=frame, lw=1.6))
    ax.text(x + 0.45, y - 0.52, label, ha="center", fontsize=8.6, color=INK)


def _broken(ax, x, y, c1, c2):
    ax.add_patch(Circle((x, y), 0.18, color=c1, ec="white", lw=0.8, zorder=3))
    ax.add_patch(Circle((x + 0.9, y), 0.18, color=c2, ec="white", lw=0.8, zorder=3))
    # crossed-out competition arrows
    ax.annotate("", xy=(x + 0.70, y), xytext=(x + 0.20, y),
                arrowprops=dict(arrowstyle="-", color="#b5341f", lw=1.6))
    ax.plot([x + 0.40, x + 0.50], [y + 0.13, y - 0.13], color="#b5341f", lw=2.0, zorder=4)
    ax.plot([x + 0.50, x + 0.40], [y + 0.13, y - 0.13], color="#b5341f", lw=2.0, zorder=4)


if __name__ == "__main__":
    main()
