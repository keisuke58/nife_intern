#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
UMAP comparison: Hamilton posterior clouds + gLV MAP overlay.

Both are projected to the same 15-dim space (symmetric entries of A).
For gLV (asymmetric), each off-diagonal pair (i,j) is averaged: (A_ij+A_ji)/2.

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/plot_heine_umap_comparison.py

Outputs: results/heine2025/heine_umap_comparison.{pdf,png}
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from paper_data import paper_5sp_samples, paper_5sp_theta
from thesis_style import use as thesis_style

CONDS   = ["CS", "CH", "DS", "DH"]
COLORS  = {"CS": "#1f77b4", "CH": "#2ca02c", "DS": "#ff7f0e", "DH": "#d62728"}
OUT_DIR = Path(__file__).resolve().parents[2] / "results/heine2025"
GLV_JSON = Path(__file__).resolve().parents[2] / "results/heine2025/fit_glv_heine.json"

# 15 unique symmetric entries: (i,j) with i<=j, row-major
UPPER_IDX = [(i, j) for i in range(5) for j in range(i, 5)]  # 15 pairs

# Hamilton theta -> 15-dim symmetric vector (same ordering as UPPER_IDX)
A_IDX_THETA = {
    (0, 0): 0,  (0, 1): 1,  (1, 1): 2,  (2, 2): 5,  (2, 3): 6,
    (3, 3): 7,  (0, 2): 10, (0, 3): 11, (1, 2): 12, (1, 3): 13,
    (4, 4): 14, (0, 4): 16, (1, 4): 17, (2, 4): 18, (3, 4): 19,
}


def samples_to_vec(samples):
    """Hamilton posterior samples (N, 20) -> (N, 15) symmetric entries."""
    idxs = [A_IDX_THETA[ij] for ij in UPPER_IDX]
    return samples[:, idxs]


def glv_A_to_vec(A):
    """gLV asymmetric A (5,5) -> 15-dim by averaging off-diagonal pairs."""
    A = np.array(A)
    vec = np.array([(A[i, j] + A[j, i]) / 2 if i != j else A[i, i]
                    for i, j in UPPER_IDX])
    return vec


def main():
    import umap as umap_lib

    n_sub = 1200
    rng = np.random.default_rng(42)

    # ── Hamilton posterior vectors ───────────────────────────────────────────────
    H_vecs, H_labs = [], []
    H_map_vecs = {}
    for ck in CONDS:
        s = paper_5sp_samples(ck)
        sub = s[rng.choice(s.shape[0], size=min(n_sub, s.shape[0]), replace=False)]
        H_vecs.append(samples_to_vec(sub))
        H_labs += [ck] * len(sub)
        H_map_vecs[ck] = samples_to_vec(paper_5sp_theta(ck)[np.newaxis, :])[0]

    H_X = np.vstack(H_vecs)

    # ── gLV MAP vectors ──────────────────────────────────────────────────────────
    glv = json.load(open(GLV_JSON))
    G_vecs = {ck: glv_A_to_vec(glv[ck]["A"]) for ck in CONDS}
    G_rmse = {ck: glv[ck]["rmse"] for ck in CONDS}

    # ── Fit UMAP on Hamilton posterior only, then transform MAP points ───────────
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                            random_state=42)
    emb_H = reducer.fit_transform(H_X)
    emb_H_map = np.vstack([reducer.transform(H_map_vecs[ck][np.newaxis, :])
                            for ck in CONDS])
    emb_G_map = np.vstack([reducer.transform(G_vecs[ck][np.newaxis, :])
                            for ck in CONDS])

    H_labs = np.array(H_labs)

    # ── Figure ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=thesis_style(0.62, aspect=0.9))

    # Hamilton posterior clouds
    for ck in CONDS:
        m = H_labs == ck
        ax.scatter(emb_H[m, 0], emb_H[m, 1], s=2, alpha=0.35,
                   color=COLORS[ck], rasterized=True, label=f"{ck} (Ham.)")

    # Hamilton MAP stars (same colour, larger, filled)
    for k, ck in enumerate(CONDS):
        ax.scatter(emb_H_map[k, 0], emb_H_map[k, 1], s=80,
                   marker="*", color=COLORS[ck], edgecolors="k",
                   linewidths=0.5, zorder=5)

    # gLV MAP crosses (same colour, X marker)
    for k, ck in enumerate(CONDS):
        ax.scatter(emb_G_map[k, 0], emb_G_map[k, 1], s=60,
                   marker="X", color=COLORS[ck], edgecolors="k",
                   linewidths=0.5, zorder=5,
                   label=f"{ck} gLV (RMSE={G_rmse[ck]:.3f})")
        # thin dashed line from Hamilton MAP to gLV MAP
        ax.plot([emb_H_map[k, 0], emb_G_map[k, 0]],
                [emb_H_map[k, 1], emb_G_map[k, 1]],
                color=COLORS[ck], lw=0.8, ls="--", alpha=0.7, zorder=4)

    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.set_title(r"UMAP of $\mathbf{A}$ (Hamilton posterior + gLV MAP)", fontsize=9)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="grey", markersize=5, lw=0,
               alpha=0.6, label="Hamilton posterior"),
        Line2D([0], [0], marker="*", color="grey", markersize=8, lw=0,
               markeredgecolor="k", markeredgewidth=0.4,
               label="Hamilton MAP"),
        Line2D([0], [0], marker="X", color="grey", markersize=7, lw=0,
               markeredgecolor="k", markeredgewidth=0.4,
               label="gLV MAP"),
    ]
    for ck in CONDS:
        legend_elements.append(
            Line2D([0], [0], marker="s", color=COLORS[ck], markersize=6,
                   lw=0, label=ck)
        )
    ax.legend(handles=legend_elements, fontsize=6.5,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.9, borderpad=0.6)

    out_pdf = OUT_DIR / "heine_umap_comparison.pdf"
    out_png = OUT_DIR / "heine_umap_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
