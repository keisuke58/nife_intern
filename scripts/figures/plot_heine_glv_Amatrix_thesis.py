#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
Heine 5-species gLV interaction matrices — thesis figure.
Parallel to heine_posterior_figs.py fig_heatmap() but uses the gLV MAP fit.

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/plot_heine_glv_Amatrix_thesis.py

Outputs: results/heine2025/heine_glv_Amatrix_thesis.{pdf,png}
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from thesis_style import use as thesis_style

RESULT_JSON = Path(__file__).resolve().parents[2] / "results/heine2025/fit_glv_heine.json"
OUT_DIR     = Path(__file__).resolve().parents[2] / "results/heine2025"

CONDS      = ["CS", "CH", "DS", "DH"]
COND_TITLE = {"CS": "CS", "CH": "CH", "DS": "DS", "DH": "DH"}
SP         = ["S.o", "A.n", "Vd", "F.n", "P.g"]


def main():
    res = json.load(open(RESULT_JSON))

    fig, axes = plt.subplots(1, 4, figsize=thesis_style(1.0, aspect=0.30))
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    im = None
    for ax, ck in zip(axes, CONDS):
        A = np.array(res[ck]["A"])
        rmse = res[ck]["rmse"]
        im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")
        ax.set_xticks(range(5))
        ax.set_xticklabels(SP, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(5))
        ax.set_yticklabels(SP if ck == "CS" else [], fontsize=7)
        ax.set_title(f"{COND_TITLE[ck]}\n(RMSE = {rmse:.3f})", fontsize=8)
        ax.tick_params(length=0)
        # Cell value annotations
        vmax_abs = np.abs(A).max()
        thresh = 0.55 * max(0.5, vmax_abs * 0.95)
        for i in range(5):
            for j in range(5):
                val = A[i, j]
                if abs(val) > 0.04:
                    tc = "white" if abs(val) > thresh else "#333333"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=5.0, color=tc)

    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cb.set_label(r"$A_{ij}$")
    fig.suptitle(r"gLV MAP interaction matrices $\mathbf{A}$ (Heine 2025)")

    out_pdf = OUT_DIR / "heine_glv_Amatrix_thesis.pdf"
    out_png = OUT_DIR / "heine_glv_Amatrix_thesis.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
