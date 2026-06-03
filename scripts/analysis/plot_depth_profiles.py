#!/usr/bin/env python3
"""
plot_depth_profiles.py — polished depth-profile figures from the extracted
z-profile CSV (no .lif re-decode needed).

Reads results/diffusion_fit/zprofiles_all_ti.csv (columns: condition, day, z_um,
+ 5 species intensities) and renders three views, all with physical µm depth axis
(surface at top, deep at bottom) and Times-family fonts:

  1. <stem>_grid.png   — condition × day line grid, µm ticks + "surface→deep" arrow
  2. <stem>_overlay.png— CH (solid) vs DH (dashed) on one panel per day, to read
                         the Pg deep-sinking in dysbiosis directly
  3. <stem>_stacked.png— per-depth stacked-area composition (each depth's 5-species
                         fractions sum to 1), condition × day grid

Run from the repo root:
    python scripts/analysis/plot_depth_profiles.py
    python scripts/analysis/plot_depth_profiles.py --csv results/diffusion_fit/zprofiles_all_ti.csv
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
})

HERE = Path(__file__).resolve().parents[2]
SPECIES = ["S.oralis", "A.naeslundii", "Vd/Vp", "F.nucleatum", "P.gingivalis"]
SHORT = {"S.oralis": "S.o", "A.naeslundii": "A.n", "Vd/Vp": "Vd/Vp",
         "F.nucleatum": "F.n", "P.gingivalis": "P.g"}
COL = {"S.oralis": "#1f77b4", "A.naeslundii": "#2ca02c", "Vd/Vp": "#bcbd22",
       "F.nucleatum": "#9467bd", "P.gingivalis": "#d62728"}


def panel_fractions(sub):
    """Return (z_um sorted, frac matrix (nz,5) per-z renormalised among 5 species)."""
    sub = sub.sort_values("z_um")
    z = sub["z_um"].to_numpy()
    M = sub[SPECIES].to_numpy(float)
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return z, M / s


def _depth_arrow(ax, zmax):
    """Add a 'surface → deep' annotation arrow down the left of the panel."""
    ax.annotate("", xy=(-0.16, 0.02), xytext=(-0.16, 0.98),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6))
    ax.text(-0.215, 0.5, "surface → deep", rotation=90, va="center",
            ha="center", transform=ax.transAxes, fontsize=10, color="#444")


def fig_grid(data, conds, days, out):
    nr, nc = len(conds), len(days)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 3.4 * nr), squeeze=False)
    for ri, cond in enumerate(conds):
        for ci, day in enumerate(days):
            ax = axes[ri][ci]
            sub = data[(data.condition == cond) & (data.day == day)]
            if sub.empty:
                ax.axis("off"); continue
            z, frac = panel_fractions(sub)
            for si, sp in enumerate(SPECIES):
                ax.plot(frac[:, si], z, color=COL[sp], lw=2.2, label=SHORT[sp])
            ax.set_title(f"{cond} day {day}", fontsize=13)
            ax.set_xlim(0, 1); ax.set_ylim(z.max(), 0)        # surface at top
            if ri == nr - 1:
                ax.set_xlabel("fraction", fontsize=12)
            if ci == 0:
                ax.set_ylabel(f"{cond}\ndepth z (µm)", fontsize=12)
                _depth_arrow(ax, z.max())
    axes[0][nc - 1].legend(fontsize=10, loc="upper right", framealpha=0.9)
    fig.suptitle("FISH depth profiles — per-z renormalised fractions (µm depth)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def fig_overlay(data, days, out):
    """CH (solid) vs DH (dashed) overlaid, one panel per day."""
    nc = len(days)
    fig, axes = plt.subplots(1, nc, figsize=(2.9 * nc, 4.0), squeeze=False)
    for ci, day in enumerate(days):
        ax = axes[0][ci]
        zmax = 0.0
        for cond, ls in (("CH", "-"), ("DH", "--")):
            sub = data[(data.condition == cond) & (data.day == day)]
            if sub.empty:
                continue
            z, frac = panel_fractions(sub)
            zmax = max(zmax, z.max())
            for si, sp in enumerate(SPECIES):
                ax.plot(frac[:, si], z, color=COL[sp], lw=2.0, ls=ls)
        ax.set_title(f"day {day}", fontsize=13)
        ax.set_xlim(0, 1); ax.set_ylim(zmax, 0)
        if ci == 0:
            ax.set_ylabel("depth z (µm)", fontsize=12)
            _depth_arrow(ax, zmax)
        ax.set_xlabel("fraction", fontsize=12)
    # legend: species colours + CH/DH linestyle
    from matplotlib.lines import Line2D
    sp_handles = [Line2D([0], [0], color=COL[sp], lw=2.4, label=SHORT[sp]) for sp in SPECIES]
    ls_handles = [Line2D([0], [0], color="#444", lw=2.0, ls="-", label="CH (commensal)"),
                  Line2D([0], [0], color="#444", lw=2.0, ls="--", label="DH (dysbiotic)")]
    fig.legend(handles=sp_handles + ls_handles, loc="lower center",
               ncol=7, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("CH vs DH depth profiles — P. gingivalis sinks deeper in dysbiosis",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def fig_stacked(data, conds, days, out):
    """Per-depth stacked-area composition (fractions sum to 1 at each depth)."""
    nr, nc = len(conds), len(days)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 3.4 * nr), squeeze=False)
    for ri, cond in enumerate(conds):
        for ci, day in enumerate(days):
            ax = axes[ri][ci]
            sub = data[(data.condition == cond) & (data.day == day)]
            if sub.empty:
                ax.axis("off"); continue
            z, frac = panel_fractions(sub)
            left = np.zeros_like(z)
            for si, sp in enumerate(SPECIES):
                ax.fill_betweenx(z, left, left + frac[:, si], color=COL[sp],
                                 label=SHORT[sp], lw=0)
                left = left + frac[:, si]
            ax.set_title(f"{cond} day {day}", fontsize=13)
            ax.set_xlim(0, 1); ax.set_ylim(z.max(), 0)
            if ri == nr - 1:
                ax.set_xlabel("composition", fontsize=12)
            if ci == 0:
                ax.set_ylabel(f"{cond}\ndepth z (µm)", fontsize=12)
                _depth_arrow(ax, z.max())
    axes[0][nc - 1].legend(fontsize=9, loc="upper right", framealpha=0.95)
    fig.suptitle("FISH depth composition — per-depth stacked 5-species fractions",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/diffusion_fit/zprofiles_all_ti.csv")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.is_absolute():
        csv = HERE / csv
    data = pd.read_csv(csv)
    conds = [c for c in ("CH", "DH") if c in set(data.condition)]
    days = sorted(data.day.unique())
    stem = csv.with_suffix("")

    fig_grid(data, conds, days, Path(f"{stem}_grid.png"))
    fig_overlay(data, days, Path(f"{stem}_overlay.png"))
    fig_stacked(data, conds, days, Path(f"{stem}_stacked.png"))


if __name__ == "__main__":
    main()
