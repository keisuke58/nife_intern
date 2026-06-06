# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_botelho_backbone.py — Cross-cohort concordant backbone network.

Strong pairs (|A_dk| > 0.1 AND |A_bot| > 0.1, upper triangle):
  green solid  = both cohorts same sign   (8/9)
  red dashed   = sign disagrees           (1/9: Bacil–Fuso)
  blue = facilitation (+), red = competition (-)

Run from repo root:
    python scripts/figures/fig_botelho_backbone.py
"""

import json
import numpy as np
import networkx as nx
import matplotlib as _mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import thesis_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

_ROOT = Path(__file__).resolve().parents[2]
RES   = _ROOT / "results" / "botelho_validation"
OUT_PDF = _ROOT / "results" / "fig_botelho_backbone.pdf"
OUT_PNG = _ROOT / "results" / "fig_botelho_backbone.png"

THETA = 0.1
FAC   = "#2166ac"   # facilitation blue
COMP  = "#b2182b"   # competition red
AGREE_ALPHA  = 0.92
DISAG_ALPHA  = 0.55


def main():
    thesis_style.use()
    _mpl.rcParams["text.usetex"] = False

    A_dk  = np.array(json.load(open(RES / "dieckow_A_noprior_comparison.json"))["A_dieckow_noprior"])
    A_bot = np.array(json.load(open(RES / "botelho_A_comparison_noprior.json"))["A_botelho"])

    N = len(GUILD_ORDER)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    strong = []
    for i, j in pairs:
        adk, abot = A_dk[i, j], A_bot[i, j]
        if abs(adk) > THETA and abs(abot) > THETA:
            agree = (np.sign(adk) == np.sign(abot))
            strong.append((i, j, adk, abot, bool(agree)))

    nodes = sorted({i for i, j, *_ in strong} | {j for i, j, *_ in strong})

    # ── layout: circular with Actinobacteria at top ────────────────────────
    n_nodes = len(nodes)
    angles  = {n: 2 * np.pi * k / n_nodes for k, n in enumerate(nodes)}
    # put Actinobacteria at top (angle = pi/2)
    act_idx = nodes.index(GUILD_ORDER.index("Actinobacteria")) if GUILD_ORDER.index("Actinobacteria") in nodes else 0
    offset  = np.pi / 2 - angles[nodes[act_idx]]
    pos = {n: (np.cos(angles[n] + offset), np.sin(angles[n] + offset)) for n in nodes}

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.08, wspace=0.12)

    # ── Panel A: network ───────────────────────────────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(A) Cross-cohort backbone (theta = 0.1)", fontsize=8.5, pad=4)

    # draw edges
    for i, j, adk, abot, agree in strong:
        xi, yi = pos[i]
        xj, yj = pos[j]
        sign_dk = np.sign(adk)
        color   = FAC if sign_dk > 0 else COMP
        lw      = 1.2 + 2.5 * min(abs(adk), 2.0) / 2.0
        ls      = "-" if agree else "--"
        alpha   = AGREE_ALPHA if agree else DISAG_ALPHA
        ax.plot([xi, xj], [yi, yj], color=color, lw=lw, ls=ls,
                alpha=alpha, solid_capstyle="round", zorder=1)

    # draw nodes
    node_size = 420
    for n in nodes:
        x, y = pos[n]
        g = GUILD_ORDER[n]
        ax.scatter(x, y, s=node_size, color=GUILD_COLORS[g],
                   edgecolors="k", linewidths=0.8, zorder=3)
        # label outside node
        lx = x * 1.22
        ly = y * 1.22
        ax.text(lx, ly, GUILD_SHORT[g], ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=4)

    # legend
    legend_els = [
        Line2D([0], [0], color=FAC,  lw=2, label="Facilitation (+)"),
        Line2D([0], [0], color=COMP, lw=2, label="Competition (−)"),
        Line2D([0], [0], color="gray", lw=2, ls="-",  label="Agree (8/9)"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="Disagree (1/9)"),
    ]
    ax.legend(handles=legend_els, fontsize=6.5, loc="lower left",
              framealpha=0.85, bbox_to_anchor=(-0.02, -0.02))

    # ── Panel B: dot-plot of A values per strong pair ─────────────────────
    ax2 = axes[1]
    ax2.set_title("(B) Edge magnitudes", fontsize=8.5, pad=4)

    y_labels = []
    for k, (i, j, adk, abot, agree) in enumerate(strong):
        y = len(strong) - k - 1
        gi = GUILD_SHORT[GUILD_ORDER[i]]
        gj = GUILD_SHORT[GUILD_ORDER[j]]
        label = f"{gi}–{gj}"
        y_labels.append(label)
        color = "gray" if agree else "#e08000"
        ax2.plot([adk, abot], [y, y], color=color, lw=1.0, alpha=0.7)
        ax2.scatter(adk,  y, marker="o", s=30, color=FAC if adk > 0 else COMP,
                    zorder=3, label="Dieckow" if k == 0 else "")
        ax2.scatter(abot, y, marker="s", s=30, color=FAC if abot > 0 else COMP,
                    zorder=3, label="Botelho" if k == 0 else "", alpha=0.7)

    ax2.axvline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax2.set_yticks(range(len(strong)))
    ax2.set_yticklabels(y_labels[::-1], fontsize=6.5)
    ax2.set_xlabel("$A_{ij}$ value", fontsize=8)
    ax2.tick_params(labelsize=7)

    dot_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=5, label="Dieckow"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=5, label="Botelho"),
    ]
    ax2.legend(handles=dot_legend, fontsize=6.5, loc="lower right")

    fig.suptitle(
        f"Dieckow × Botelho concordant backbone  (8/9 strong pairs agree, $p\\approx0.02$)",
        fontsize=9, y=0.97)

    fig.savefig(OUT_PDF, dpi=150)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
