#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
replot_fish_3d.py — regenerate the two 3-D FISH summary figures from the cached
per-FOV metrics CSV (results/fish_3d/fish_3d_lateral_metrics.csv), with the unified
thesis style. This avoids re-decoding the 84 .lif FOVs (heavy); the plotted
quantities (FnPg Manders M1, lateral CV) are already in the CSV.

Run with TeX Live 2025 on PATH (usetex). Outputs the same PNG names as
fish_3d_batch.py plus vector PDFs.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thesis_style import use as thesis_style

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "results" / "fish_3d"
CSV = OUT / "fish_3d_lateral_metrics.csv"
SHORT = ["S.o", "A.n", "Vd/Vp", "F.n", "P.g"]
COND = [("CH", "#1f77b4"), ("DH", "#d62728")]


def main():
    df = pd.read_csv(CSV)
    g = df.groupby(["cond", "day"]).mean(numeric_only=True).reset_index()

    # ── Fig A: Fn-Pg lateral vs 3-D co-localisation over time ──────────────────
    fig, ax = plt.subplots(1, 2, figsize=thesis_style(1.0, aspect=0.42))
    for cond, c in COND:
        sub = g[g.cond == cond].sort_values("day")
        if len(sub):
            ax[0].plot(sub.day, sub.FnPg_patch_M1, "o-", color=c, lw=1.6, label=cond)
            ax[1].plot(sub.day, sub.FnPg_3d_M1, "o-", color=c, lw=1.6, label=cond)
    ax[0].set_title(r"\textit{Fn}--\textit{Pg} lateral patch (xy)")
    ax[1].set_title(r"\textit{Fn}--\textit{Pg} 3-D (voxels)")
    for a in ax:
        a.set_xlabel("day")
        a.set_ylabel(r"Manders $M_1$ (\textit{Pg} with \textit{Fn})")
        a.legend(); a.grid(ls=":", alpha=0.6)
    fig.suptitle(r"\textit{Fn}--\textit{Pg} co-localisation over time --- CH vs DH (84 FOV)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fish_3d_fnpg_coloc.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fish_3d_fnpg_coloc.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Fig B: lateral heterogeneity (mean species CV) over time ───────────────
    cols = [f"latCV_{s}" for s in SHORT]
    g["latCV_mean"] = g[cols].mean(axis=1)
    fig2, ax2 = plt.subplots(figsize=thesis_style(0.8, aspect=0.64))
    for cond, c in COND:
        sub = g[g.cond == cond].sort_values("day")
        if len(sub):
            ax2.plot(sub.day, sub.latCV_mean, "o-", color=c, lw=1.8, label=cond)
    ax2.set_xlabel("day")
    ax2.set_ylabel("lateral heterogeneity (mean CV of xy projection)")
    ax2.set_title("Lateral patchiness over time --- CH vs DH")
    ax2.legend(); ax2.grid(ls=":", alpha=0.6)
    fig2.tight_layout()
    fig2.savefig(OUT / "fish_3d_lateral_heterogeneity.png", dpi=300, bbox_inches="tight")
    fig2.savefig(OUT / "fish_3d_lateral_heterogeneity.pdf", bbox_inches="tight")
    plt.close(fig2)

    print("wrote fish_3d_fnpg_coloc.{png,pdf}, fish_3d_lateral_heterogeneity.{png,pdf}")


if __name__ == "__main__":
    main()
