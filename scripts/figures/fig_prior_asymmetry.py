# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_prior_asymmetry.py — Prior sign agreement: health vs dysbiosis asymmetry.

Data source: results/kegg_sign_summary.json
  SA_map_pct      : MAP posterior sign agreement WITHOUT kegg prior (no-prior baseline)
  SA_kegg_map_pct : MAP posterior sign agreement WITH kegg prior (1000p kegg run)

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_prior_asymmetry.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from thesis_style import use, clean_ax, PALETTE

ROOT = pathlib.Path(__file__).parents[2]
SRC  = ROOT / "results" / "kegg_sign_summary.json"
OUT  = ROOT / "results" / "fig_prior_asymmetry"

CONDS = ["CS", "CH", "DS", "DH"]
LABELS = ["CS\n(commensal\nstatic)", "CH\n(commensal\nHOBIC)", "DS\n(dysbiotic\nstatic)", "DH\n(dysbiotic\nHOBIC)"]

# Colours: health=teal, dysbiosis=coral
C_HEALTH  = PALETTE["health"]
C_DYSBIO  = PALETTE["dysbiosis"]
C_NOPRIOR = PALETTE["prior_off"]

def main():
    data = json.loads(SRC.read_text())["conditions"]

    sa_noprior = np.array([data[c]["SA_map_pct"]      for c in CONDS])
    sa_prior   = np.array([data[c]["SA_kegg_map_pct"] for c in CONDS])
    delta      = sa_prior - sa_noprior

    fig, ax = plt.subplots(figsize=use(width_frac=0.72, aspect=0.72))

    x = np.arange(len(CONDS))
    w = 0.35

    bar_colors = [C_HEALTH, C_HEALTH, C_DYSBIO, C_DYSBIO]

    b1 = ax.bar(x - w/2, sa_noprior, w,
                color=C_NOPRIOR, edgecolor="white", linewidth=0.5,
                label="without metabolic prior")
    b2 = ax.bar(x + w/2, sa_prior, w,
                color=bar_colors, edgecolor="white", linewidth=0.5,
                label="with metabolic prior")

    # chance line
    ax.axhline(50, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(3.55, 51.5, "chance", fontsize=7, va="bottom", ha="right", color="0.35")

    # delta annotations
    for i, (xp, d) in enumerate(zip(x, delta)):
        sign = "+" if d >= 0 else ""
        color = "#1a5276" if d > 0 else "#7b241c"
        ax.text(xp + w/2, sa_prior[i] + 1.5, f"${sign}{d:.0f}$pp",
                ha="center", va="bottom", fontsize=7, color=color)

    # health / dysbiosis divider
    ax.axvline(1.5, color="0.5", linewidth=0.6, linestyle=":")
    ax.text(0.5, 97, "health", ha="center", va="top", fontsize=8,
            color=C_HEALTH, fontweight="bold")
    ax.text(2.5, 97, "dysbiosis", ha="center", va="top", fontsize=8,
            color=C_DYSBIO, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=7.5)
    ax.set_ylabel(r"Sign agreement (\%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 3.6)
    clean_ax(ax)

    leg = ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9,
                    edgecolor="0.8", handlelength=1.2, handletextpad=0.5)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}")
        print(f"saved → {OUT}.{ext}")

if __name__ == "__main__":
    main()
