#!/usr/bin/env python3
"""
Community type (CT) analysis of Dieckow 10 patients.
Clusters patients by mean 10-guild composition (Szafranski 2025 preprint style).

Output:
  results/dieckow_otu/community_types.pdf + .png
  results/dieckow_otu/community_types.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from guild_replicator_dieckow import GUILD_ORDER, N_G

PHI_NPY = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
OUT_DIR = Path(__file__).parent / 'results' / 'dieckow_otu'

PATIENTS = list('ABCDEFGHKL')
SHORT    = [g[:6] for g in GUILD_ORDER]
CT_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

GUILD_COLORS = [
    '#8B4513', '#2ca02c', '#ff7f0e', '#1f77b4',
    '#17becf', '#bcbd22', '#ffbb78', '#aec7e8',
    '#98df8a', '#999999',
]


def best_k(X, Z, k_range=(2, 4)):
    """Pick k with highest silhouette score."""
    scores = {}
    for k in range(k_range[0], k_range[1] + 1):
        labels = fcluster(Z, k, criterion='maxclust')
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X, labels)
    return max(scores, key=scores.get), scores


def main():
    phi = np.load(PHI_NPY)          # (10, 3, 10)
    # Feature: mean across weeks → (10, 10)
    X = phi.mean(axis=1)

    # Hierarchical clustering (Ward, Bray-Curtis-like = Euclidean on compositions)
    dist = pdist(X, metric='euclidean')
    Z    = linkage(dist, method='ward')

    k_best, sil_scores = best_k(X, Z)
    labels = fcluster(Z, k_best, criterion='maxclust')
    print(f'Best k={k_best}  silhouette scores: {sil_scores}')
    for p, ct in zip(PATIENTS, labels):
        print(f'  Patient {p} → CT{ct}')

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            height_ratios=[1, 1.4])

    # Panel a: dendrogram
    ax_dend = fig.add_subplot(gs[0, 0])
    dend = dendrogram(Z, labels=PATIENTS, ax=ax_dend,
                      color_threshold=Z[-(k_best-1), 2],
                      above_threshold_color='#888888')
    ax_dend.set_title('a  Patient dendrogram (Ward / Euclidean)', fontsize=10, loc='left')
    ax_dend.set_ylabel('Distance')
    ax_dend.spines['top'].set_visible(False)
    ax_dend.spines['right'].set_visible(False)

    # Panel b: silhouette scores
    ax_sil = fig.add_subplot(gs[0, 1])
    ks = sorted(sil_scores)
    ax_sil.bar(ks, [sil_scores[k] for k in ks],
               color=[CT_COLORS[k-2] for k in ks], edgecolor='k', linewidth=0.5)
    ax_sil.axvline(k_best, color='red', lw=1.5, ls='--', label=f'k={k_best} selected')
    ax_sil.set_xlabel('Number of clusters k')
    ax_sil.set_ylabel('Silhouette score')
    ax_sil.set_title('b  Silhouette score vs k', fontsize=10, loc='left')
    ax_sil.legend(fontsize=8)
    ax_sil.spines['top'].set_visible(False)
    ax_sil.spines['right'].set_visible(False)

    # Panel c: guild composition per CT (stacked bar, mean ± range)
    ax_ct = fig.add_subplot(gs[1, 0])
    ct_ids = sorted(set(labels))
    ct_means = []
    for ct in ct_ids:
        mask = (labels == ct)
        ct_means.append(X[mask].mean(axis=0))   # (10,)
    ct_means = np.array(ct_means)   # (k, 10)

    bottom = np.zeros(len(ct_ids))
    for g, (col, name) in enumerate(zip(GUILD_COLORS, GUILD_ORDER)):
        ax_ct.bar([f'CT{ct}' for ct in ct_ids], ct_means[:, g],
                  bottom=bottom, color=col, label=name[:12])
        bottom += ct_means[:, g]
    ax_ct.set_ylim(0, 1.05)
    ax_ct.set_ylabel('Mean relative abundance')
    ax_ct.set_title('c  Guild composition per community type', fontsize=10, loc='left')
    ax_ct.legend(loc='upper right', fontsize=6, ncol=2, frameon=False)
    ax_ct.spines['top'].set_visible(False)
    ax_ct.spines['right'].set_visible(False)

    # Panel d: per-patient stacked bars sorted by CT
    ax_pat = fig.add_subplot(gs[1, 1])
    order  = np.argsort(labels)   # sort patients by CT
    sorted_patients = [PATIENTS[i] for i in order]
    sorted_labels   = labels[order]
    sorted_X        = X[order]

    bottom = np.zeros(len(PATIENTS))
    for g, (col, name) in enumerate(zip(GUILD_COLORS, GUILD_ORDER)):
        ax_pat.bar(range(len(PATIENTS)), sorted_X[:, g],
                   bottom=bottom, color=col, width=0.7)
        bottom += sorted_X[:, g]

    ax_pat.set_xticks(range(len(PATIENTS)))
    ax_pat.set_xticklabels(
        [f'{p}\n(CT{l})' for p, l in zip(sorted_patients, sorted_labels)],
        fontsize=8)
    ax_pat.set_ylim(0, 1.05)
    ax_pat.set_ylabel('Mean relative abundance')
    ax_pat.set_title('d  Per-patient composition (sorted by CT)', fontsize=10, loc='left')

    # CT colour spans
    prev_ct = sorted_labels[0]
    span_start = 0
    for idx, ct in enumerate(sorted_labels):
        if ct != prev_ct or idx == len(sorted_labels) - 1:
            end = idx if ct != prev_ct else idx + 1
            ax_pat.axvspan(span_start - 0.4, end - 0.6,
                           color=CT_COLORS[prev_ct - 1], alpha=0.12, zorder=0)
            span_start = idx
            prev_ct = ct
    ax_pat.spines['top'].set_visible(False)
    ax_pat.spines['right'].set_visible(False)

    fig.suptitle('Dieckow 2024 — Community type analysis (10-guild, Ward clustering)',
                 fontsize=11)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'community_types.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f'Saved: {out}')
    plt.close(fig)

    # Save JSON
    out_json = OUT_DIR / 'community_types.json'
    with open(out_json, 'w') as f:
        json.dump({
            'k': int(k_best),
            'silhouette_scores': {str(k): float(v) for k, v in sil_scores.items()},
            'patient_ct': {p: int(ct) for p, ct in zip(PATIENTS, labels)},
            'ct_mean_composition': {
                f'CT{ct}': {g: float(ct_means[i, j])
                             for j, g in enumerate(GUILD_ORDER)}
                for i, ct in enumerate(ct_ids)
            },
        }, f, indent=2)
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    main()
