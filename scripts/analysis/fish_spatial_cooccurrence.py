# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fish_spatial_cooccurrence.py — Spatial co-occurrence vs A-matrix sign (B).

For each species pair (i,j), compare:
  - Predicted interaction sign: sgn(A_ij + A_ji) from Heine TMCMC posterior
  - Observed spatial correlation: Pearson r between phi_xyz[:,:,z,i] and
    phi_xyz[:,:,z,j], averaged over depth and days

Output:
  results/fish_cooccurrence_vs_Amatrix.{pdf,png}
  results/fish_cooccurrence_stats.json

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/analysis/fish_spatial_cooccurrence.py
"""

import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

from thesis_style import use, clean_ax
from paper_data import paper_5sp_theta, paper_5sp_samples
from attractor_analysis import theta_to_matrices

ROOT  = pathlib.Path(__file__).parents[2]
IC    = ROOT / "results" / "fish_3d" / "ic"
OUT_FIG  = ROOT / "results" / "fish_cooccurrence_vs_Amatrix"
OUT_JSON = ROOT / "results" / "fish_cooccurrence_stats.json"

SHORT  = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
DAYS   = [1, 6, 10, 15, 21]
CONDS  = ['CH', 'DH']

N_SP   = 5
PAIRS  = [(i, j) for i in range(N_SP) for j in range(i+1, N_SP)]   # 10 pairs


def pearson_by_depth(arr):
    """arr: (128, 128, Nz, 5). Return (Nz, 10) Pearson r per pair per depth."""
    nz = arr.shape[2]
    r_z = np.full((nz, len(PAIRS)), np.nan)
    for k, (i, j) in enumerate(PAIRS):
        xi = arr[:, :, :, i].reshape(128 * 128, nz)   # (pixels, Nz)
        xj = arr[:, :, :, j].reshape(128 * 128, nz)
        for z in range(nz):
            x, y = xi[:, z], xj[:, z]
            # only compute where at least one is nonzero
            mask = (x > 0) | (y > 0)
            if mask.sum() > 10:
                r_z[z, k], _ = pearsonr(x[mask], y[mask])
    return r_z


def load_posterior_A(cond):
    """Return mean A (5,5) and per-pair sign consistency across posterior samples."""
    samples = paper_5sp_samples(cond)           # (10000, 20)
    A_samples = np.array([theta_to_matrices(s)[0] for s in samples])  # (10000,5,5)
    A_mean = A_samples.mean(axis=0)
    return A_mean, A_samples


def main():
    # ── posterior A matrices ──────────────────────────────────────────────────
    A_mean = {}
    A_samp = {}
    for cond in CONDS:
        A_mean[cond], A_samp[cond] = load_posterior_A(cond)

    # ── spatial co-occurrence by depth ────────────────────────────────────────
    # corr_data[cond][pair_k] = list of Pearson r values across (day, z-slices)
    corr_data = {cond: {k: [] for k in range(len(PAIRS))} for cond in CONDS}

    for cond in CONDS:
        for day in DAYS:
            f = IC / f"phi_xyz_{cond}_d{day}.npy"
            if not f.exists():
                continue
            arr = np.load(f)                     # (128, 128, Nz, 5)
            r_z = pearson_by_depth(arr)          # (Nz, 10)
            for k in range(len(PAIRS)):
                valid = r_z[:, k][~np.isnan(r_z[:, k])]
                corr_data[cond][k].extend(valid.tolist())

    # ── summary stats ─────────────────────────────────────────────────────────
    stats = {}
    for cond in CONDS:
        A = A_mean[cond]
        stats[cond] = []
        for k, (i, j) in enumerate(PAIRS):
            sym   = A[i, j] + A[j, i]           # symmetrised interaction
            pred_sign = int(np.sign(sym))
            r_vals = np.array(corr_data[cond][k])
            mean_r = float(np.nanmean(r_vals)) if len(r_vals) else np.nan
            agree  = (pred_sign == np.sign(mean_r)) if (pred_sign != 0 and not np.isnan(mean_r)) else None
            stats[cond].append({
                "pair": f"{SHORT[i]}-{SHORT[j]}",
                "i": i, "j": j,
                "A_sym": float(sym),
                "pred_sign": pred_sign,
                "mean_r": mean_r,
                "n_observations": len(r_vals),
                "sign_agree": bool(agree) if agree is not None else None,
            })

    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"stats → {OUT_JSON}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.42),
                             sharey=True)

    for ax, cond in zip(axes, CONDS):
        rows = stats[cond]
        x    = np.arange(len(PAIRS))
        mean_r_vals = [r["mean_r"] for r in rows]
        agree_flags = [r["sign_agree"] for r in rows]

        bar_colors = []
        for r, a in zip(rows, agree_flags):
            if a is True:
                bar_colors.append("#2196A6" if r["pred_sign"] > 0 else "#D95F02")
            elif a is False:
                bar_colors.append("#e74c3c")
            else:
                bar_colors.append("#bdbdbd")

        bars = ax.bar(x, mean_r_vals, color=bar_colors,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([r["pair"] for r in rows],
                            rotation=45, ha="right", fontsize=6.5)
        ax.set_title(cond, fontsize=9)
        clean_ax(ax)
        ax.set_ylim(-0.5, 0.5)

        n_ag = sum(1 for a in agree_flags if a is True)
        n_tot = sum(1 for a in agree_flags if a is not None)
        ax.text(0.97, 0.97, f"agree {n_ag}/{n_tot}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)

    axes[0].set_ylabel("Mean Pearson $r$ (spatial)", fontsize=8)

    legend_handles = [
        mpatches.Patch(color="#2196A6", label="predicted cooperative, observed $r>0$"),
        mpatches.Patch(color="#D95F02", label="predicted competitive, observed $r<0$"),
        mpatches.Patch(color="#e74c3c", label="sign mismatch"),
        mpatches.Patch(color="#bdbdbd", label="predicted sign = 0"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=6.5, framealpha=0.9, edgecolor="0.8",
               bbox_to_anchor=(0.5, -0.12))

    fig.suptitle(r"Spatial co-occurrence vs.\ predicted interaction sign",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_FIG}.{ext}", bbox_inches="tight")
        print(f"saved → {OUT_FIG}.{ext}")

    # ── console summary ───────────────────────────────────────────────────────
    for cond in CONDS:
        rows = stats[cond]
        n_ag  = sum(1 for r in rows if r["sign_agree"] is True)
        n_dis = sum(1 for r in rows if r["sign_agree"] is False)
        n_tot = n_ag + n_dis
        print(f"\n{cond}: {n_ag}/{n_tot} pairs sign-agree "
              f"({100*n_ag/n_tot:.0f}% if n_tot>0 else '?')")
        for r in rows:
            flag = "✓" if r["sign_agree"] else ("✗" if r["sign_agree"] is False else "—")
            print(f"  {flag} {r['pair']:12s}  A_sym={r['A_sym']:+.3f}  "
                  f"mean_r={r['mean_r']:+.3f}  n={r['n_observations']}")


if __name__ == "__main__":
    main()
