#!/usr/bin/env python3
"""
Plot week1→2→3 relative abundance trajectories for all 10 Dieckow patients.
Input:  results/dieckow_otu/phi_obs_raw.npy  (10, 3, 5)
Output: results/dieckow_otu/timeseries.pdf + .png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'lines.linewidth': 1.8,
    'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

PHI_NPY  = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_obs_raw.npy'
OUT_DIR  = Path(__file__).parent / 'results' / 'dieckow_otu'

PATIENTS = list('ABCDEFGHKL')
WEEKS    = [1, 2, 3]
LABELS   = ['So', 'An', 'Vd', 'Fn', 'Pg']
COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
FULL_NAMES = {
    'So': 'Streptococcus',
    'An': 'Actinomyces',
    'Vd': 'Veillonella',
    'Fn': 'Fusobacterium',
    'Pg': 'Porphyromonas',
}


def main():
    phi = np.load(PHI_NPY)   # (10, 3, 5)

    fig, axes = plt.subplots(2, 5, figsize=(12, 4.8), sharey=True)
    axes = axes.ravel()

    for i, (p, ax) in enumerate(zip(PATIENTS, axes)):
        for k, (sp, col) in enumerate(zip(LABELS, COLORS)):
            ax.plot(WEEKS, phi[i, :, k], 'o-', color=col, lw=1.8,
                    ms=5, label=FULL_NAMES[sp] if i == 0 else None)
        ax.set_title(f'Patient {p}', fontsize=10, fontweight='bold')
        ax.set_xticks(WEEKS)
        ax.set_xticklabels(['W1', 'W2', 'W3'], fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0.7, 3.3)
        if i % 5 == 0:
            ax.set_ylabel('Relative abundance', fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.legend(loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle('Dieckow 2024 — 5-species relative abundance (16S PacBio)',
                 fontsize=11, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'timeseries.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=300)
        print(f'Saved: {out}')

    plt.close(fig)


if __name__ == '__main__':
    main()
