#!/usr/bin/env python3
"""
species_importance_analysis.py — Species-level importance for nife paper.

Analyses (all 4 conditions CS/CH/DS/DH):

  1. Interaction importance   : rank |A_ij| — which species pair matters most
  2. Species knockout         : remove species i (φ_i ≡ 0), simulate forward
                                → Bray-Curtis divergence from full model
  3. Productivity impact      : Σ_i b_i φ_i(t) — growth-weighted community output
                                knockout vs full model
  4. Volume growth contribution: ∫ φ_i(t) dt / ∫ φ_total(t) dt
                                 + knockout effect on total volume

All figures saved to results/species_importance/.

Usage:
    python species_importance_analysis.py
    python species_importance_analysis.py --cond DH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
})
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

HERE     = Path(__file__).resolve().parent
GLV_FIT  = HERE / 'results' / 'heine2025' / 'fit_glv_heine.json'
DATA_CSV = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data'
                '/fig3_species_distribution_replicates.csv')
OUT_DIR  = HERE / 'results' / 'species_importance'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT   = ['S.o', 'A.n', 'Vd/Vp', 'F.n', 'P.g']
COLORS  = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
N_SP    = 5
T_END   = 21.0   # days
T_EVAL  = np.linspace(0, T_END, 300)
CONDITIONS = ['CS', 'CH', 'DS', 'DH']
COND_LABELS = {'CS': 'Commensal Static', 'CH': 'Commensal HOBIC',
               'DS': 'Dysbiotic Static', 'DH': 'Dysbiotic HOBIC'}
COND_META = {
    'CS': ('Commensal', 'Static'),
    'CH': ('Commensal', 'HOBIC'),
    'DS': ('Dysbiotic', 'Static'),
    'DH': ('Dysbiotic', 'HOBIC'),
}
SPECIES_MAP = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',  'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula', 'F. nucleatum', 'P. gingivalis_W83'],
}


# ── gLV ODE ───────────────────────────────────────────────────────────────

def glv_rhs(t, phi, A, b, mask):
    """gLV: dφ_i/dt = φ_i (b_i + Σ_j A_ij φ_j),  mask zeros out species i."""
    phi = np.maximum(phi * mask, 0)
    dphi = phi * (b + A @ phi)
    return dphi * mask


def simulate_glv(A, b, phi0, mask=None, t_eval=T_EVAL):
    """Simulate gLV ODE. Returns (T, N_SP) trajectory."""
    if mask is None:
        mask = np.ones(N_SP)
    phi0_masked = phi0 * mask
    sol = solve_ivp(glv_rhs, [0, T_EVAL[-1]], phi0_masked,
                    args=(A, b, mask), t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y.T   # (T, N_SP)


def bray_curtis(p, q):
    """Bray-Curtis dissimilarity between two composition vectors."""
    return np.sum(np.abs(p - q)) / (np.sum(p) + np.sum(q) + 1e-12)


# ── Load data ─────────────────────────────────────────────────────────────

def load_params(cond_tag):
    d = json.loads(GLV_FIT.read_text())
    A = np.array(d[cond_tag]['A'])
    b = np.array(d[cond_tag]['b'])
    return A, b


def get_phi0(cond_tag):
    """Day-1 median observed composition from Heine data."""
    condition, cultivation = COND_META[cond_tag]
    sp_names = SPECIES_MAP[condition]
    df  = pd.read_csv(DATA_CSV)
    sub = df[(df['condition'] == condition) & (df['cultivation'] == cultivation)
             & (df['day'] == 1)]
    phi0 = np.zeros(N_SP)
    for s, name in enumerate(sp_names):
        vals = sub[sub['species'] == name]['distribution_pct'].values
        phi0[s] = np.median(vals) if len(vals) else 0.0
    total = phi0.sum()
    return phi0 / total if total > 1e-12 else np.ones(N_SP) / N_SP


# ── Analysis 1: Interaction importance ───────────────────────────────────

def plot_interaction_importance(results_by_cond, out_path=None):
    """Heatmap of |A_ij| per condition + mean rank."""
    fig, axes = plt.subplots(1, len(results_by_cond), figsize=(3.2 * len(results_by_cond), 3.0))
    if len(results_by_cond) == 1:
        axes = [axes]

    for ax, (tag, res) in zip(axes, results_by_cond.items()):
        A = res['A']
        im = ax.imshow(np.abs(A), cmap='YlOrRd', vmin=0)
        ax.set_xticks(range(N_SP)); ax.set_xticklabels(SHORT, fontsize=8, rotation=45)
        ax.set_yticks(range(N_SP)); ax.set_yticklabels(SHORT, fontsize=8)
        ax.set_title(f'{tag}\n{COND_LABELS[tag]}', fontsize=8)
        ax.set_xlabel('source j', fontsize=7)
        ax.set_ylabel('target i', fontsize=7)
        for i in range(N_SP):
            for j in range(N_SP):
                ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if abs(A[i,j]) < 0.5*np.abs(A).max() else 'white')
        fig.colorbar(im, ax=ax, fraction=0.046, label='|A_ij|')

    fig.suptitle('Interaction matrix |A_ij| — species pair importance', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def rank_interactions(A, top_n=5):
    """Return top N (i,j) pairs by |A_ij|, excluding diagonal."""
    pairs = []
    for i in range(N_SP):
        for j in range(N_SP):
            if i != j:
                pairs.append((abs(A[i, j]), i, j, A[i, j]))
    pairs.sort(reverse=True)
    return pairs[:top_n]


# ── Analysis 2+3+4: Knockout ──────────────────────────────────────────────

def run_knockout(A, b, phi0):
    """
    For each species i, simulate with φ_i ≡ 0.
    Returns dict with full trajectory + per-knockout results.
    """
    full_traj = simulate_glv(A, b, phi0)   # (T, N_SP)

    knockouts = {}
    for i in range(N_SP):
        mask = np.ones(N_SP)
        mask[i] = 0.0
        traj = simulate_glv(A, b, phi0, mask=mask)
        knockouts[i] = traj
    return full_traj, knockouts


def compute_knockout_metrics(full_traj, knockouts, b):
    """
    For each knockout:
      - bc_div   : mean Bray-Curtis divergence from full model over time
      - prod_drop: fractional drop in Σ b_i φ_i (productivity)
      - vol_drop : fractional drop in ∫ φ_total dt (volume growth proxy)
    """
    metrics = {}
    nf        = len(full_traj)
    full_prod = np.maximum(full_traj @ b, 0)
    full_vol  = np.trapezoid(full_traj.sum(axis=1), T_EVAL[:nf])

    for i, traj in knockouts.items():
        n       = min(nf, len(traj))
        bc_vals = np.array([bray_curtis(full_traj[t], traj[t]) for t in range(n)])
        ko_prod = np.maximum(traj[:n] @ b, 0)
        ko_vol  = np.trapezoid(traj[:n].sum(axis=1), T_EVAL[:n])
        fp_mean = full_prod[:n].mean()

        metrics[i] = {
            'bc_mean':   float(bc_vals.mean()),
            'bc_max':    float(bc_vals.max()),
            'prod_drop': float((fp_mean - ko_prod.mean()) / (fp_mean + 1e-12)),
            'vol_drop':  float((full_vol - ko_vol) / (full_vol + 1e-12)),
        }
    return metrics


def compute_volume_contributions(full_traj):
    """
    Species-specific contribution to total volume growth.
    Returns (N_SP,) array summing to 1.
    """
    nf = len(full_traj)
    integrals = np.array([np.trapezoid(full_traj[:, i], T_EVAL[:nf]) for i in range(N_SP)])
    total     = integrals.sum()
    return integrals / (total + 1e-12), integrals


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_knockout_summary(all_metrics, out_path=None):
    """
    Bar chart: per-species knockout effect on BC divergence,
    productivity drop, and volume drop — for each condition.
    """
    n_cond = len(all_metrics)
    fig, axes = plt.subplots(3, n_cond, figsize=(3.0 * n_cond, 7.0), sharey='row')

    metric_keys   = ['bc_mean', 'prod_drop', 'vol_drop']
    metric_labels = ['Bray-Curtis divergence\n(from full model)',
                     'Productivity drop\n(Σ b_i φ_i)',
                     'Volume growth drop\n(∫ φ_total dt)']

    for col, (tag, metrics) in enumerate(all_metrics.items()):
        for row, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            ax  = axes[row, col]
            vals = [metrics[i][key] for i in range(N_SP)]
            bars = ax.bar(range(N_SP), vals, color=COLORS, alpha=0.85)
            ax.set_xticks(range(N_SP))
            ax.set_xticklabels(SHORT, rotation=45, fontsize=8)
            ax.axhline(0, color='k', lw=0.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == 0:
                ax.set_title(f'{tag}\n{COND_LABELS[tag]}', fontsize=9)
            # annotate values
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=6)

    fig.suptitle('Effect of species knockout', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_volume_contributions(all_contribs, out_path=None):
    """Stacked bar: species fractional contribution to ∫ φ_total dt."""
    n_cond = len(all_contribs)
    fig, ax = plt.subplots(figsize=(2.0 * n_cond, 3.5))
    x      = np.arange(n_cond)
    bottom = np.zeros(n_cond)
    tags   = list(all_contribs.keys())

    for s in range(N_SP):
        vals = np.array([all_contribs[tag][0][s] for tag in tags])
        ax.bar(x, vals, bottom=bottom, color=COLORS[s], label=SHORT[s], alpha=0.85)
        for xi, (v, b_) in enumerate(zip(vals, bottom)):
            if v > 0.04:
                ax.text(xi, b_ + v/2, f'{v:.0%}', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        bottom += vals

    ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=10)
    ax.set_ylabel('Fractional contribution to ∫ φ_total dt', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('Species contribution to biofilm volume growth', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_trajectory_comparison(full_traj, knockouts, tag, out_path=None):
    """Per-species knockout: full vs knockout trajectory for most impactful species."""
    fig, axes = plt.subplots(N_SP, N_SP, figsize=(2.2 * N_SP, 2.2 * N_SP))
    for ko_sp in range(N_SP):
        for obs_sp in range(N_SP):
            ax = axes[ko_sp, obs_sp]
            ko_traj = knockouts[ko_sp]
            n = min(len(full_traj), len(ko_traj))
            ax.plot(T_EVAL[:len(full_traj)], full_traj[:, obs_sp], color=COLORS[obs_sp], lw=1.5, label='Full')
            ax.plot(T_EVAL[:n], ko_traj[:n, obs_sp], '--', color=COLORS[obs_sp],
                    lw=1.5, alpha=0.7, label=f'KO {SHORT[ko_sp]}')
            ax.set_ylim(0, 1)
            if ko_sp == N_SP - 1:
                ax.set_xlabel('Day', fontsize=7)
            if obs_sp == 0:
                ax.set_ylabel(f'KO {SHORT[ko_sp]}', fontsize=7, color=COLORS[ko_sp])
            if ko_sp == 0:
                ax.set_title(SHORT[obs_sp], fontsize=8, color=COLORS[obs_sp])
            ax.tick_params(labelsize=6)
    fig.suptitle(f'{tag} — knockout trajectories (row=removed species, col=observed species)',
                 fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def print_summary(tag, A, b, metrics, contribs):
    print(f'\n{"="*50}')
    print(f'{tag} ({COND_LABELS[tag]})')
    print(f'{"="*50}')
    print('\nTop 5 interactions by |A_ij|:')
    for rank, (mag, i, j, val) in enumerate(rank_interactions(A), 1):
        sign = '+' if val > 0 else '-'
        print(f'  {rank}. {SHORT[j]} → {SHORT[i]}  A={val:+.3f}  ({sign})')
    print('\nSpecies knockout effects:')
    print(f'  {"Species":<10} {"BC div":>8} {"Prod drop":>10} {"Vol drop":>9}')
    for i in range(N_SP):
        m = metrics[i]
        print(f'  {SHORT[i]:<10} {m["bc_mean"]:8.4f} {m["prod_drop"]:10.4f} {m["vol_drop"]:9.4f}')
    print('\nVolume growth contributions:')
    frac, integ = contribs
    for i in range(N_SP):
        print(f'  {SHORT[i]:<10} {frac[i]:.1%}  (integral={integ[i]:.2f})')


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond', choices=['CS','CH','DS','DH','all'], default='all')
    args = parser.parse_args()

    conds = CONDITIONS if args.cond == 'all' else [args.cond]

    results_by_cond = {}
    all_metrics     = {}
    all_contribs    = {}

    for tag in conds:
        A, b   = load_params(tag)
        phi0   = get_phi0(tag)

        full_traj, knockouts = run_knockout(A, b, phi0)
        metrics  = compute_knockout_metrics(full_traj, knockouts, b)
        contribs = compute_volume_contributions(full_traj)

        results_by_cond[tag] = {'A': A, 'b': b}
        all_metrics[tag]     = metrics
        all_contribs[tag]    = contribs

        print_summary(tag, A, b, metrics, contribs)

        plot_trajectory_comparison(
            full_traj, knockouts, tag,
            out_path=OUT_DIR / f'trajectories_ko_{tag}.png'
        )

    # Summary plots across conditions
    plot_interaction_importance(
        results_by_cond,
        out_path=OUT_DIR / 'interaction_importance.png'
    )
    plot_knockout_summary(
        all_metrics,
        out_path=OUT_DIR / 'knockout_summary.png'
    )
    plot_volume_contributions(
        all_contribs,
        out_path=OUT_DIR / 'volume_contributions.png'
    )

    # Save JSON
    summary = {}
    for tag in conds:
        A, b = load_params(tag)
        summary[tag] = {
            'top_interactions': [
                {'source': SHORT[j], 'target': SHORT[i], 'A_ij': float(val)}
                for _, i, j, val in rank_interactions(A)
            ],
            'knockout_metrics': {
                SHORT[i]: all_metrics[tag][i] for i in range(N_SP)
            },
            'volume_contributions': {
                SHORT[i]: float(all_contribs[tag][0][i]) for i in range(N_SP)
            },
        }
    (OUT_DIR / 'species_importance_summary.json').write_text(
        json.dumps(summary, indent=2)
    )
    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
