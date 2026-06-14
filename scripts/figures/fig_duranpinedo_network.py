# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_duranpinedo_network.py — Cross-cohort eigenvector centrality comparison.

Compare guild-level eigenvector centrality of the Duran-Pinedo A matrix (5 seeds,
prior-free gLV) vs the Dieckow A matrix (LOO consensus, no prior).

Panels:
  Left  — side-by-side bar chart per guild (Dieckow vs Duran-Pinedo)
  Right — scatter: Dieckow centrality vs Duran-Pinedo centrality, with Spearman r

Also prints the centrality rank correlation and top-3 concordant guilds.

Run from repo root:
    python scripts/figures/fig_duranpinedo_network.py
"""

import glob
import json
import numpy as np
import networkx as nx
import matplotlib as _mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

import thesis_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

_ROOT = Path(__file__).resolve().parents[2]
RES_DK  = _ROOT / "results" / "dieckow_cr"
RES_BOT = _ROOT / "results" / "duranpinedo_validation"
OUT_PDF = _ROOT / "results" / "fig_duranpinedo_network.pdf"
OUT_PNG = _ROOT / "results" / "fig_duranpinedo_network.png"


def eigenvector_centrality(A):
    """Symmetrised |A|, eigenvector centrality (numpy solver)."""
    N = A.shape[0]
    Asym = 0.5 * (np.abs(A) + np.abs(A.T))
    np.fill_diagonal(Asym, 0.0)
    G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            if Asym[i, j] > 0:
                G.add_edge(i, j, weight=float(Asym[i, j]))
    eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    return np.array([eig.get(i, 0.0) for i in range(N)])


def load_dieckow_centrality():
    net = json.loads((RES_DK / "network_analysis.json").read_text())
    return np.array(net["A_centrality_symmetric"]["eigenvector"])


def load_duranpinedo_centrality():
    seed_files = sorted(glob.glob(str(RES_BOT / "A_dk_noprior_seed*.npy")))
    As = np.stack([np.load(f) for f in seed_files])   # (5, 10, 10)
    A_mean = As.mean(axis=0)
    return eigenvector_centrality(A_mean), As


def main():
    thesis_style.use()
    _mpl.rcParams["text.usetex"] = False

    eig_dk  = load_dieckow_centrality()
    eig_bot, As_bot = load_duranpinedo_centrality()

    # per-seed centralities for error bars
    eig_bot_seeds = np.array([eigenvector_centrality(A) for A in As_bot])
    eig_bot_sd = eig_bot_seeds.std(axis=0)

    # rank correlation
    rho, pval = spearmanr(eig_dk, eig_bot)

    short = [GUILD_SHORT[g] for g in GUILD_ORDER]
    colors = [GUILD_COLORS[g] for g in GUILD_ORDER]

    # sort by Dieckow centrality for display
    order = np.argsort(eig_dk)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.97, top=0.88, bottom=0.22)

    # ── Left: grouped bar chart ────────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(GUILD_ORDER))
    w = 0.38
    bars_dk  = ax.bar(x - w/2, eig_dk[order],  w, color=[colors[i] for i in order],
                      alpha=0.85, label="Dieckow", edgecolor="k", linewidth=0.4)
    bars_bot = ax.bar(x + w/2, eig_bot[order], w, color=[colors[i] for i in order],
                      alpha=0.40, hatch="//", label="Duran-Pinedo",
                      edgecolor="k", linewidth=0.4,
                      yerr=eig_bot_sd[order], error_kw=dict(elinewidth=0.8, capsize=2))

    ax.set_xticks(x)
    ax.set_xticklabels([short[i] for i in order], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Eigenvector centrality", fontsize=9)
    ax.set_title("Guild centrality by cohort", fontsize=9, pad=4)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, None)

    # ── Right: scatter ────────────────────────────────────────────────────────
    ax2 = axes[1]
    for i, g in enumerate(GUILD_ORDER):
        ax2.errorbar(eig_dk[i], eig_bot[i], yerr=eig_bot_sd[i],
                     fmt="o", color=colors[i], ms=6,
                     elinewidth=0.8, capsize=2, zorder=3)
        ax2.annotate(short[i], (eig_dk[i], eig_bot[i]),
                     textcoords="offset points", xytext=(4, 2),
                     fontsize=6.5, color=colors[i])

    # identity line (perfect agreement)
    lim_max = max(eig_dk.max(), eig_bot.max()) * 1.1
    ax2.plot([0, lim_max], [0, lim_max], "k--", lw=0.7, alpha=0.4, label="y = x")
    ax2.set_xlabel("Dieckow centrality", fontsize=9)
    ax2.set_ylabel("Duran-Pinedo centrality", fontsize=9)
    ax2.set_title(f"Cross-cohort centrality\n$\\rho={rho:.2f}$, $p={pval:.3f}$",
                  fontsize=9, pad=4)
    ax2.tick_params(labelsize=7)
    ax2.set_xlim(-0.02, lim_max)
    ax2.set_ylim(-0.02, lim_max)

    fig.suptitle("Duran-Pinedo vs Dieckow: eigenvector centrality (prior-free gLV)",
                 fontsize=9, y=0.97)

    fig.savefig(OUT_PDF, dpi=150)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PDF}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\nSpearman rho={rho:.3f}  p={pval:.4f}")
    print(f"\n{'Guild':22s}  {'Dieckow':>8s}  {'Duran-Pinedo':>8s}  {'rank-DK':>7s}  {'rank-BOT':>8s}")
    rk_dk  = (eig_dk.argsort()[::-1].argsort() + 1)
    rk_bot = (eig_bot.argsort()[::-1].argsort() + 1)
    for i, g in enumerate(GUILD_ORDER):
        print(f"  {g:22s}  {eig_dk[i]:8.3f}  {eig_bot[i]:8.3f}  {rk_dk[i]:>7d}  {rk_bot[i]:>8d}")


if __name__ == "__main__":
    main()
