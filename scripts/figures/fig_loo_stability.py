# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_loo_stability.py — Fig 8: LOO-CV A matrix stability heatmaps.

Two panels:
  Left  — sign consistency (fraction of 10 LOO folds with same sign as MAP)
  Right — coefficient of variation (A_std / |A_mean|), log-scaled

Both shown as symmetric 10×10 heatmaps (upper triangle only, diagonal blank).
Pairs with prior != 0 are annotated with a dot.

Run from repo root:
    python scripts/figures/fig_loo_stability.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib as _mpl
from thesis_style import use
import pub_style as _ps; _ps.apply()
from guild_replicator_dieckow import GUILD_SHORT, GUILD_ORDER

REPO = Path(__file__).parents[2]
STATS = REPO / "results" / "dieckow_cr" / "loo_stability_stats.csv"
OUT   = REPO / "results" / "fig_loo_stability"

N = 10
LABELS = [GUILD_SHORT[g] for g in GUILD_ORDER[:N]]


def load_stats(path):
    sc  = np.full((N, N), np.nan)
    cv  = np.full((N, N), np.nan)
    pri = np.zeros((N, N))
    with open(path) as f:
        for row in csv.DictReader(f):
            i, j = int(row["i"]), int(row["j"])
            sc[i, j] = sc[j, i] = float(row["sign_consistency"])
            cv[i, j] = cv[j, i] = float(row["CV"])
            pri[i, j] = pri[j, i] = float(row["prior"])
    return sc, cv, pri


def mask_lower(mat):
    """Return mat with lower triangle (including diagonal) set to NaN."""
    m = mat.copy()
    idx = np.tril_indices(N)
    m[idx] = np.nan
    return m


def draw_heatmap(ax, data, cmap, norm, title, cbar_label):
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(range(N)); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(LABELS, fontsize=7)
    ax.set_title(title, fontsize=8, pad=4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cbar_label)
    return im


def main():
    figsize = use(width_frac=1.0, aspect=0.42)
    # usetex=True set by use() — requires TL2025 in PATH
    sc, cv, pri = load_stats(STATS)

    sc_up  = mask_lower(sc)
    cv_up  = mask_lower(cv)
    pri_up = mask_lower(pri)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: sign consistency
    norm_sc = mcolors.Normalize(vmin=0.5, vmax=1.0)
    draw_heatmap(axes[0], sc_up, "RdYlGn", norm_sc,
                 "Sign consistency", "Fraction of folds")

    # Right: CV (log scale, clip at 0.01–5)
    cv_plot = np.clip(cv_up, 0.01, 5.0)
    norm_cv = mcolors.LogNorm(vmin=0.01, vmax=5.0)
    draw_heatmap(axes[1], cv_plot, "viridis_r", norm_cv,
                 r"Coefficient of variation  $\sigma/|\bar{A}|$", "CV (log scale)")

    # Annotate pairs with non-zero prior
    for ax, data in [(axes[0], sc_up), (axes[1], cv_plot)]:
        for i in range(N):
            for j in range(i + 1, N):
                if pri_up[i, j] != 0 and not np.isnan(data[i, j]):
                    ax.plot(j, i, ".", color="black", ms=3, zorder=5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"Saved {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
