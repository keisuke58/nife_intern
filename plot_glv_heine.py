#!/usr/bin/env python3
"""Plot gLV fit results for Heine 2025: trajectories + A-matrix heatmaps."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_CSV    = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_replicates.csv')
RESULT_JSON = Path('/home/nishioka/IKM_Hiwi/nife/results/heine2025/fit_glv_heine.json')
OUT_DIR     = Path('/home/nishioka/IKM_Hiwi/nife/results/heine2025')

DAYS    = [1, 3, 6, 10, 15, 21]
N_SP    = 5
SHORT   = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
COLORS  = ['#2196F3', '#4CAF50', '#FFC107', '#9C27B0', '#F44336']
LABELS  = ['CS', 'CH', 'DS', 'DH']
COND_TITLES = ['Commensal Static', 'Commensal HOBIC', 'Dysbiotic Static', 'Dysbiotic HOBIC']
SPECIES = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',  'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula', 'F. nucleatum', 'P. gingivalis_W83'],
}
CONDITIONS = [
    ('Commensal', 'Static',  'CS'),
    ('Commensal', 'HOBIC',   'CH'),
    ('Dysbiotic', 'Static',  'DS'),
    ('Dysbiotic', 'HOBIC',   'DH'),
]


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
    traj = sol.y.T
    traj = np.maximum(traj, 0)
    traj = traj / traj.sum(axis=1, keepdims=True)
    return traj


def load_phi(df, condition, cultivation):
    sp_list = SPECIES[condition]
    mask    = (df['condition'] == condition) & (df['cultivation'] == cultivation)
    sub     = df[mask]
    phi_med = np.zeros((len(DAYS), N_SP))
    phi_q1  = np.zeros((len(DAYS), N_SP))
    phi_q3  = np.zeros((len(DAYS), N_SP))
    for j, sp in enumerate(sp_list):
        sp_sub = sub[sub['species'] == sp]
        for i, day in enumerate(DAYS):
            vals = sp_sub[sp_sub['day'] == day]['distribution_pct'].values
            if len(vals) > 0:
                phi_med[i, j] = np.median(vals)
                phi_q1[i, j]  = np.percentile(vals, 25)
                phi_q3[i, j]  = np.percentile(vals, 75)
    for arr in [phi_med, phi_q1, phi_q3]:
        rs = arr.sum(axis=1, keepdims=True)
        arr /= np.where(rs > 0, rs, 1.0)
    return phi_med, phi_q1, phi_q3


def main():
    res = json.load(open(RESULT_JSON))
    df  = pd.read_csv(DATA_CSV)

    t_fine = np.linspace(1, 21, 200)

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            height_ratios=[2, 2, 1.4],
                            hspace=0.45, wspace=0.32)

    for col, (condition, cultivation, label) in enumerate(CONDITIONS):
        r   = res[label]
        A   = np.array(r['A'])
        b   = np.array(r['b'])
        phi_med, phi_q1, phi_q3 = load_phi(df, condition, cultivation)

        # ── row 0: stacked area (observed) ──
        ax0 = fig.add_subplot(gs[0, col])
        phi_stack = phi_med.copy()
        bottom = np.zeros(len(DAYS))
        for j in range(N_SP):
            ax0.bar(DAYS, phi_stack[:, j], bottom=bottom,
                    color=COLORS[j], alpha=0.75, width=1.5, label=SHORT[j])
            bottom += phi_stack[:, j]
        ax0.set_xlim(0, 22); ax0.set_ylim(0, 1)
        ax0.set_title(f'{label} — observed', fontsize=10, fontweight='bold')
        ax0.set_ylabel('fraction' if col == 0 else '')
        ax0.set_xticks(DAYS)
        if col == 0:
            ax0.legend(fontsize=7, loc='upper right', framealpha=0.7)

        # ── row 1: trajectory (predicted vs observed) ──
        ax1 = fig.add_subplot(gs[1, col])
        pred_fine = integrate_glv(A, b, phi_med[0], t_fine)
        for j in range(N_SP):
            ax1.plot(t_fine, pred_fine[:, j], color=COLORS[j], lw=1.8)
            yerr_lo = np.maximum(phi_med[:, j] - phi_q1[:, j], 0)
            yerr_hi = np.maximum(phi_q3[:, j] - phi_med[:, j], 0)
            ax1.errorbar(DAYS, phi_med[:, j], yerr=[yerr_lo, yerr_hi],
                         fmt='o', color=COLORS[j], ms=4, lw=1, capsize=2, alpha=0.85)
        ax1.set_xlim(0, 22); ax1.set_ylim(-0.02, 1.05)
        ax1.set_title(f'{label} — fit  RMSE={r["rmse"]:.4f}', fontsize=10)
        ax1.set_xlabel('day'); ax1.set_ylabel('fraction' if col == 0 else '')
        ax1.set_xticks(DAYS)

        # ── row 2: A-matrix heatmap ──
        ax2 = fig.add_subplot(gs[2, col])
        vmax = max(0.5, np.abs(A).max())
        im   = ax2.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax2.set_xticks(range(N_SP)); ax2.set_xticklabels(SHORT, fontsize=7)
        ax2.set_yticks(range(N_SP)); ax2.set_yticklabels(SHORT, fontsize=7)
        ax2.set_title(f'{label} — A matrix', fontsize=9)
        for i in range(N_SP):
            for j in range(N_SP):
                ax2.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                         fontsize=6, color='k' if abs(A[i,j]) < 0.6 * vmax else 'w')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle('gLV fit — Heine 2025 (5-species, 4 conditions)', fontsize=13, y=1.01)

    out = OUT_DIR / 'glv_heine_fit.pdf'
    fig.savefig(out, bbox_inches='tight')
    out_png = OUT_DIR / 'glv_heine_fit.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
