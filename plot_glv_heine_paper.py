#!/usr/bin/env python3
"""Paper-quality figure: gLV fit to Heine 2025 (trajectories + A-matrix heatmaps)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── paper rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.titlesize':    9,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'axes.linewidth':    0.7,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'lines.linewidth':   1.5,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ── paths & constants ─────────────────────────────────────────────────────────
DATA_CSV    = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_replicates.csv')
RESULT_JSON = Path('/home/nishioka/IKM_Hiwi/nife/results/heine2025/fit_glv_heine.json')
OUT_DIR     = Path('/home/nishioka/IKM_Hiwi/nife/results/heine2025')

DAYS   = [1, 3, 6, 10, 15, 21]
N_SP   = 5
SHORT  = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']

# Heine paper colours (≈ blue, green, amber, purple, red)
COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']

CONDITIONS = [
    ('Commensal', 'Static',  'CS', 'Commensal static'),
    ('Commensal', 'HOBIC',   'CH', 'Commensal HOBIC'),
    ('Dysbiotic', 'Static',  'DS', 'Dysbiotic static'),
    ('Dysbiotic', 'HOBIC',   'DH', 'Dysbiotic HOBIC'),
]
SPECIES = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',
                  'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula',
                  'F. nucleatum', 'P. gingivalis_W83'],
}
PANEL_LETTERS = list('abcdefgh')


# ── ODE helpers ───────────────────────────────────────────────────────────────
def replicator_rhs(t, phi, A, b):
    phi = np.maximum(phi, 1e-10)
    phi = phi / phi.sum()
    f    = A @ phi + b
    fbar = phi @ f
    return phi * (f - fbar)


def integrate_glv(A, b, phi0, t_eval):
    sol = solve_ivp(replicator_rhs, [t_eval[0], t_eval[-1]], phi0,
                    t_eval=t_eval, args=(A, b), method='RK45',
                    rtol=1e-6, atol=1e-9, max_step=0.5)
    traj = np.maximum(sol.y.T, 0)
    return traj / traj.sum(axis=1, keepdims=True)


def load_phi(df, condition, cultivation):
    sp_list = SPECIES[condition]
    mask    = (df['condition'] == condition) & (df['cultivation'] == cultivation)
    sub     = df[mask]
    med = np.zeros((len(DAYS), N_SP))
    q1  = np.zeros((len(DAYS), N_SP))
    q3  = np.zeros((len(DAYS), N_SP))
    for j, sp in enumerate(sp_list):
        sp_sub = sub[sub['species'] == sp]
        for i, day in enumerate(DAYS):
            vals = sp_sub[sp_sub['day'] == day]['distribution_pct'].values
            if len(vals):
                med[i, j] = np.median(vals)
                q1[i, j]  = np.percentile(vals, 25)
                q3[i, j]  = np.percentile(vals, 75)
    for arr in [med, q1, q3]:
        rs = arr.sum(axis=1, keepdims=True)
        arr /= np.where(rs > 0, rs, 1.0)
    return med, q1, q3


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    res = json.load(open(RESULT_JSON))
    df  = pd.read_csv(DATA_CSV)
    t_fine = np.linspace(1, 21, 300)

    # Figure: 2 rows (trajectory, heatmap) × 4 cols + colourbar col
    fig = plt.figure(figsize=(7.0, 4.8))
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            height_ratios=[1.6, 1.0],
                            hspace=0.55, wspace=0.38)

    panel_idx = 0
    axes_traj = []
    axes_heat = []

    for col, (condition, cultivation, label, title) in enumerate(CONDITIONS):
        r   = res[label]
        A   = np.array(r['A'])
        b   = np.array(r['b'])
        med, q1, q3 = load_phi(df, condition, cultivation)

        # ── trajectory panel ──────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        pred = integrate_glv(A, b, med[0], t_fine)

        for j in range(N_SP):
            # IQR shading
            ax.fill_between(DAYS,
                            np.maximum(q1[:, j], 0),
                            np.maximum(q3[:, j], 0),
                            color=COLORS[j], alpha=0.18, linewidth=0)
            # Observed median dots
            ax.plot(DAYS, med[:, j], 'o', color=COLORS[j],
                    ms=3.5, zorder=5, markeredgewidth=0.4,
                    markeredgecolor='white')
            # Predicted line
            ax.plot(t_fine, pred[:, j], color=COLORS[j], lw=1.6)

        ax.set_xlim(0, 23)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xticks(DAYS)
        ax.set_xticklabels([str(d) for d in DAYS])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0', '0.5', '1'] if col == 0 else [])
        if col == 0:
            ax.set_ylabel('Relative abundance')
        ax.set_xlabel('Day' if col == 1 or col == 2 else '')
        ax.set_title(f'{title}\nRMSE = {r["rmse"]:.4f}', fontsize=8.5)

        # Panel letter
        ax.text(-0.18, 1.08, PANEL_LETTERS[panel_idx], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
        panel_idx += 1
        axes_traj.append(ax)

        # ── A-matrix heatmap ──────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, col])
        vmax = max(0.5, np.abs(A).max() * 0.95)
        im   = ax2.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')

        ax2.set_xticks(range(N_SP))
        ax2.set_xticklabels(SHORT, fontsize=6.5)
        ax2.set_yticks(range(N_SP))
        ax2.set_yticklabels(SHORT if col == 0 else [], fontsize=6.5)
        ax2.tick_params(length=0)
        ax2.set_title(f'Interaction matrix A', fontsize=7.5, pad=3)

        # Cell annotations
        thresh = 0.55 * vmax
        for i in range(N_SP):
            for j in range(N_SP):
                c = 'white' if abs(A[i, j]) > thresh else '#333333'
                ax2.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                         fontsize=5.5, color=c)

        # Colorbar only on last column
        if col == 3:
            cbar = plt.colorbar(im, ax=ax2, fraction=0.055, pad=0.06, shrink=0.9)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label('$A_{ij}$', fontsize=7)

        ax2.text(-0.18, 1.12, PANEL_LETTERS[panel_idx], transform=ax2.transAxes,
                 fontsize=10, fontweight='bold', va='top')
        panel_idx += 1
        axes_heat.append(ax2)

    # ── legend (trajectory panels, attached to rightmost) ────────────────────
    legend_elements = [
        Line2D([0], [0], color=COLORS[j], lw=1.6, label=SHORT[j])
        for j in range(N_SP)
    ] + [
        Line2D([0], [0], marker='o', color='gray', lw=0, ms=3.5,
               label='Observed (IQR)'),
    ]
    axes_traj[-1].legend(handles=legend_elements, loc='upper right',
                         frameon=False, fontsize=6.5, handlelength=1.4)

    out_pdf = OUT_DIR / 'glv_heine_fit_paper.pdf'
    out_png = OUT_DIR / 'glv_heine_fit_paper.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
