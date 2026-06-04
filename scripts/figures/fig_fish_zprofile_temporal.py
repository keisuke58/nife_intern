# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_fish_zprofile_temporal.py — FISH depth profiles across days (A).

Species composition at each depth layer, averaged over x-y, for CH and DH
conditions on days 1, 6, 10, 15, 21. Shows how biofilm stratifies over time.

Input:  results/fish_3d/ic/phi_xyz_{CH,DH}_d{1,6,10,15,21}.npy  (128,128,Nz,5)
Output: results/fish_zprofile_temporal.{pdf,png}

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_fish_zprofile_temporal.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from thesis_style import use, clean_ax, SP_COLORS

ROOT  = pathlib.Path(__file__).parents[2]
IC    = ROOT / "results" / "fish_3d" / "ic"
OUT   = ROOT / "results" / "fig_fish_zprofile_temporal"

SHORT  = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
COLORS = SP_COLORS
DAYS   = [1, 6, 10, 15, 21]
CONDS  = ['CH', 'DH']
CLABEL = {'CH': 'Commensal HOBIC (CH)', 'DH': 'Dysbiotic HOBIC (DH)'}

Z_UM_PER_SLICE = 2.0  # µm per z-slice (≈ confocal step)


def load_zprofile(cond, day):
    """Return relative composition profile (Nz, 5) and depth axis (µm)."""
    f = IC / f"phi_xyz_{cond}_d{day}.npy"
    if not f.exists():
        return None, None
    arr = np.load(f)                      # (128, 128, Nz, 5)
    mean = arr.mean(axis=(0, 1))          # (Nz, 5)
    total = mean.sum(axis=1, keepdims=True)
    rel = mean / np.where(total > 1e-6, total, 1.0)
    nz = arr.shape[2]
    depth = np.arange(nz) * Z_UM_PER_SLICE
    return rel, depth


# Visual emphasis: Fn (3) and Pg (4) are the depth-separation story
KEY_SP   = {3, 4}          # Fn, Pg
LW_KEY   = 1.8
LW_OTHER = 0.75
ALPHA_OTHER = 0.45


def main():
    fig, axes = plt.subplots(
        2, len(DAYS),
        figsize=use(width_frac=1.0, aspect=0.52),
        sharey=False,
    )

    for row, cond in enumerate(CONDS):
        for col, day in enumerate(DAYS):
            ax = axes[row, col]
            rel, depth = load_zprofile(cond, day)

            if rel is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            else:
                for sp in range(5):
                    is_key = sp in KEY_SP
                    ax.plot(rel[:, sp], depth, color=COLORS[sp],
                            linewidth=LW_KEY if is_key else LW_OTHER,
                            alpha=1.0 if is_key else ALPHA_OTHER,
                            zorder=3 if is_key else 2)
                ax.set_xlim(0, 0.55)
                ax.set_ylim(depth[-1], 0)  # surface at top, per-panel depth

            clean_ax(ax)

            if row == 0:
                ax.set_title(f"Day {day}", fontsize=8)
            if col == 0:
                ax.set_ylabel(r"Depth ($\mu$m)", fontsize=8)
                ax.text(-0.45, 0.5, CLABEL[cond], transform=ax.transAxes,
                        va="center", ha="center", fontsize=8,
                        fontweight="bold", rotation=90)
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel("Rel. abundance", fontsize=7.5)
            ax.set_xticks([0, 0.25, 0.5])

    # legend: key species first, then others
    order = [3, 4, 0, 1, 2]   # Fn, Pg first
    handles = [mlines.Line2D([], [], color=COLORS[i],
                             lw=LW_KEY if i in KEY_SP else LW_OTHER,
                             alpha=1.0 if i in KEY_SP else 0.7,
                             label=SHORT[i])
               for i in order]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7.5, framealpha=0.9, edgecolor="0.8",
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}")
        print(f"saved → {OUT}.{ext}")


if __name__ == "__main__":
    main()
