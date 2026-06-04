#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
Dieckow 10-guild gLV interaction matrix — thesis figure.
Best model: gLV alpha=0.25 (fit_guild.json, RMSE=0.0495).
Sign-prior borders match dieckow_A_matrix.pdf (Hamilton version).

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/plot_dieckow_glv_Amatrix_thesis.py

Outputs: results/dieckow_cr/dieckow_glv_Amatrix_thesis.{pdf,png}
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from thesis_style import use as thesis_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_SHORT
from build_net_flow_expanded import build_net_flow_expanded

LOO_GLOB = Path(__file__).resolve().parents[2] / "results/dieckow_cr"
OUT_DIR  = Path(__file__).resolve().parents[2] / "results/dieckow_cr"

COLOR_AGREE = "#2ca02c"   # green  — sign matches prior
COLOR_DIS   = "#d62728"   # red    — sign discordant


def main():
    import glob as _glob
    # Best model: gLV alpha=0.25 L1+L2 (no AGORA), mean LOO-RMSE=0.0490
    fold_files = sorted(_glob.glob(str(LOO_GLOB / "loo_glv_a0p25_fold*.json")))
    if not fold_files:
        raise FileNotFoundError("No LOO fold files found")
    fold_data = [json.load(open(f)) for f in fold_files]
    A    = np.mean([np.array(d["A"]) for d in fold_data], axis=0)
    rmse = float(np.mean([d["rmse"] for d in fold_data]))
    guilds = fold_data[0].get("guilds", GUILD_ORDER)
    short  = [GUILD_SHORT.get(g, g[:6]) for g in guilds]
    n      = len(short)

    # Sign prior (L1+L2 only) for border colours
    from build_net_flow_expanded import net_flow_glv
    F = np.array(net_flow_glv(use_agora=False, competition_weight=0.25))
    constrained = (np.sign(F) != 0) & (~np.eye(n, dtype=bool))
    agree = constrained & (np.sign(F) == np.sign(A))

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=thesis_style(0.70, aspect=1.05))

    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    im   = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=7)
    ax.tick_params(length=0)
    ax.set_xlabel("Source guild")
    ax.set_ylabel("Target guild")
    ax.set_title(
        f"Interaction matrix $A$ --- gLV ($\\alpha=0.25$, L1+L2, mean of 10 LOO folds)\n"
        f"Mean LOO-RMSE = {rmse:.4f}",
        fontsize=9,
    )

    # Cell value annotations + sign-prior border boxes
    half = 0.45
    for i in range(n):
        for j in range(n):
            if constrained[i, j]:
                ec = COLOR_AGREE if agree[i, j] else COLOR_DIS
                lw = 1.5
                rect = plt.Rectangle(
                    (j - half, i - half), 2 * half, 2 * half,
                    linewidth=lw, edgecolor=ec, facecolor="none",
                    transform=ax.transData, clip_on=True,
                )
                ax.add_patch(rect)
            # text: only for cells with |A_ij| > 0.05
            val = A[i, j]
            if abs(val) > 0.05:
                thresh = 0.55 * max(0.5, np.abs(A).max() * 0.95)
                tc = "white" if abs(val) > thresh else "#333333"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=tc)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, shrink=0.8)
    cb.set_label(r"$A_{ij}$")
    cb.ax.tick_params(labelsize=7)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="none", edgecolor=COLOR_AGREE, linewidth=1.5,
              label="Sign $\\checkmark$ (metabolic prior)"),
        Patch(facecolor="none", edgecolor=COLOR_DIS, linewidth=1.5,
              label="Sign $\\times$ (discordant)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), fontsize=7, ncol=2,
              framealpha=0.9, handlelength=1.2, borderpad=0.6)

    out_pdf = OUT_DIR / "dieckow_glv_Amatrix_thesis.pdf"
    out_png = OUT_DIR / "dieckow_glv_Amatrix_thesis.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
