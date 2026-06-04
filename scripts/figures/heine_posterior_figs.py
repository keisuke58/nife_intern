#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
heine_posterior_figs.py — regenerate the Heine 5-species posterior figures
(interaction-matrix heatmaps, 15-entry posterior violins, UMAP embedding) DIRECTLY
from the canonical posterior (paper_data.py: 10000-particle Phase-2 TMCMC), with the
unified thesis style. This bypasses the original Tmcmc202601 figure scripts, whose
config.json/run-dir paths were archived (data-path rot); only samples.npy /
theta_MAP.json (which survive) are needed — no JAX, no ODE integration.

Run with TeX Live 2025 on PATH (usetex):
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/figures/heine_posterior_figs.py

Outputs (results/heine_repro/): heatmap_A_4cond.{pdf,png}, posterior_violin.{pdf,png},
umap_A_3d.{pdf,png}.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from paper_data import paper_5sp_samples, paper_5sp_theta
from thesis_style import use as thesis_style

OUT = Path(__file__).resolve().parents[2] / "results" / "heine_repro"
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ["CS", "CH", "DS", "DH"]
COND_TITLE = {"CS": "CS", "CH": "CH", "DS": "DS", "DH": "DH"}
SP = ["S.o", "A.n", "Vd", "F.n", "P.g"]

# theta(20) -> A(5x5 symmetric); index of each unique A entry (cf. hamilton_ode_jax)
A_IDX = {  # (i,j) -> theta index (i<=j)
    (0, 0): 0, (0, 1): 1, (1, 1): 2, (2, 2): 5, (2, 3): 6, (3, 3): 7,
    (0, 2): 10, (0, 3): 11, (1, 2): 12, (1, 3): 13, (4, 4): 14,
    (0, 4): 16, (1, 4): 17, (2, 4): 18, (3, 4): 19,
}
# the 15 unique interaction entries, ordered diag then off-diag, with labels
ENTRIES = [((i, j), f"{SP[i]}\\,{SP[j]}" if i != j else f"{SP[i]}") for (i, j) in
           sorted(A_IDX, key=lambda ij: (ij[0] != ij[1], ij))]


def theta_to_A(theta):
    A = np.zeros((5, 5))
    for (i, j), k in A_IDX.items():
        A[i, j] = A[j, i] = theta[k]
    return A


# ── Fig: MAP interaction-matrix heatmaps (1x4) ────────────────────────────────

def fig_heatmap():
    fig, axes = plt.subplots(1, 4, figsize=thesis_style(1.0, aspect=0.30))
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    im = None
    for ax, ck in zip(axes, CONDS):
        A = theta_to_A(paper_5sp_theta(ck))
        im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="equal")
        ax.set_xticks(range(5)); ax.set_xticklabels(SP, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(5)); ax.set_yticklabels(SP if ck == "CS" else [], fontsize=7)
        ax.set_title(COND_TITLE[ck])
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cb.set_label(r"$A_{ij}$")
    fig.suptitle(r"MAP interaction matrices $\mathbf{A}$ (Phase~2, $N_p=10{,}000$)")
    fig.savefig(OUT / "heatmap_A_4cond.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "heatmap_A_4cond.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote heatmap_A_4cond.{pdf,png}")


# ── Fig: posterior violins of the 15 interaction entries (4 rows) ─────────────

def fig_violin(n_sub=1500):
    colors = {"CS": "#1f77b4", "CH": "#2ca02c", "DS": "#ff7f0e", "DH": "#d62728"}
    fig, axes = plt.subplots(4, 1, figsize=thesis_style(1.0, aspect=1.05), sharex=True)
    labels = [lab for _, lab in ENTRIES]
    idxs = [A_IDX[ij] for ij, _ in ENTRIES]
    rng = np.random.default_rng(0)
    for ax, ck in zip(axes, CONDS):
        s = paper_5sp_samples(ck)
        sub = s[rng.choice(s.shape[0], size=min(n_sub, s.shape[0]), replace=False)]
        data = [sub[:, k] for k in idxs]
        parts = ax.violinplot(data, positions=range(len(idxs)), showextrema=False, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor(colors[ck]); b.set_alpha(0.6); b.set_edgecolor("none")
        th = paper_5sp_theta(ck)
        ax.plot(range(len(idxs)), [th[k] for k in idxs], "*", color="k", ms=4, zorder=5)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_ylabel(ck, rotation=0, ha="right", va="center")
        ax.set_ylim(-6, 8)
    axes[-1].set_xticks(range(len(labels)))
    axes[-1].set_xticklabels([f"${l}$" for l in labels], rotation=60, ha="right", fontsize=6)
    fig.suptitle(r"Posterior of the 15 interaction entries (stars: MAP)")
    fig.savefig(OUT / "posterior_violin.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "posterior_violin.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote posterior_violin.{pdf,png}")


# ── Fig: UMAP 3D embedding of A-matrix posterior samples ──────────────────────

def fig_umap(n_sub=1200):
    import umap
    colors = {"CS": "#1f77b4", "CH": "#2ca02c", "DS": "#ff7f0e", "DH": "#d62728"}
    idxs = list(A_IDX.values())  # 15 unique A entries
    rng = np.random.default_rng(1)
    X, lab = [], []
    for ck in CONDS:
        s = paper_5sp_samples(ck)
        sub = s[rng.choice(s.shape[0], size=min(n_sub, s.shape[0]), replace=False)][:, idxs]
        X.append(sub); lab += [ck] * len(sub)
    X = np.vstack(X); lab = np.array(lab)
    emb = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                    random_state=42).fit_transform(X)
    fig = plt.figure(figsize=thesis_style(0.62, aspect=0.9))
    ax = fig.add_subplot(111, projection="3d")
    for ck in CONDS:
        m = lab == ck
        ax.scatter(emb[m, 0], emb[m, 1], emb[m, 2], s=2, alpha=0.5,
                   color=colors[ck], label=ck)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    ax.legend(markerscale=3, loc="upper left", fontsize=7)
    fig.suptitle(r"UMAP of posterior $\mathbf{A}$ samples")
    fig.savefig(OUT / "umap_A_3d.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "umap_A_3d.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote umap_A_3d.{pdf,png}")


if __name__ == "__main__":
    fig_heatmap()
    fig_violin()
    fig_umap()
    print(f"\nAll outputs in {OUT}")
