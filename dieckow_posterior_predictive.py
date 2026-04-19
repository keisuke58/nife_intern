#!/usr/bin/env python3
"""
dieckow_posterior_predictive.py — Posterior predictive check
  Use Dieckow 1000p posterior samples to predict Szafranski 127-sample
  composition distribution. Quantifies model transferability from
  longitudinal (Dieckow peri-implant) to cross-sectional (Szafranski
  peri-implant) data.

Method:
  For each Szafranski sample phi_obs:
    For each posterior sample theta_k (k=1..N_post):
      Run ODE 20 weeks from phi_obs → phi_pred_k
    → posterior predictive distribution: {phi_pred_k}
    → summary: median, 90% CI per species

  Key outputs:
    1. Coverage: fraction of phi_obs within 90% CI → model calibration
    2. Predictive RMSE per diagnosis (PIH/PIM/PI)
    3. Uncertainty maps: CI width vs phi_obs
    4. Comparison: in vitro (Heine DH) vs in vivo (Dieckow) predictive spread

Usage:
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
    python3 dieckow_posterior_predictive.py [--n-post 100]
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/core')
from hamilton_ode_jax_nsp import simulate_0d_nsp

print(f'JAX devices: {jax.devices()}')

# ── Constants ─────────────────────────────────────────────────────────────────
SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP = 5; N_A = 15; N_STEPS = 2500; DT = 1e-4; C = 25.0; ALPHA = 100.0

FITS_DIR   = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
OUT_DIR    = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_postpred')
GMM_CSV    = Path('/home/nishioka/IKM_Hiwi/nife/results/gmm_attractor_analysis.csv')
HEINE_RUNS = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SP_COLORS    = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#264653']
DIAG_COLORS  = {'PIH': '#1565C0', 'PIM': '#2E7D32', 'PI': '#B71C1C'}

def run_ode_endpoint(theta20, phi0, n_weeks=20):
    t = jnp.array(theta20, dtype=jnp.float64)
    phi = jnp.array(phi0,   dtype=jnp.float64)
    for _ in range(n_weeks):
        traj = simulate_0d_nsp(t, n_sp=N_SP, n_steps=N_STEPS, dt=DT,
                                phi_init=phi, c_const=C, alpha_const=ALPHA)
        phi = traj[-1] / jnp.maximum(traj[-1].sum(), 1e-12)
    return np.array(phi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-post', type=int, default=100,
                        help='Number of posterior samples to use (default 100)')
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    import pandas as pd
    d1 = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    samples = np.array(d1['samples'])   # (1000, 65)
    gmm     = pd.read_csv(GMM_CSV)

    phi_cols = ['phi0_So', 'phi0_An', 'phi0_Vd', 'phi0_Fn', 'phi0_Pg']
    phi_obs  = gmm[phi_cols].values.astype(np.float64)   # (127, 5)
    diag     = gmm['diagnosis'].values

    N_szaf   = len(phi_obs)
    N_post   = min(args.n_post, len(samples))
    rng      = np.random.default_rng(42)
    post_idx = rng.choice(len(samples), N_post, replace=False)

    print(f'Szafranski samples: {N_szaf}')
    print(f'Posterior samples:  {N_post}')

    # Average b across all patients (exclude patient F)
    theta_full = np.array(d1['theta_map'])
    b_avg = np.zeros(N_SP)
    for i, p in enumerate(PATIENTS):
        if p != 'F':
            b_avg += theta_full[N_A + i*N_SP: N_A + (i+1)*N_SP]
    b_avg /= (len(PATIENTS) - 1)

    # ── Posterior predictive: run each Szafranski sample through N_post ODEs ──
    print(f'\nRunning {N_szaf} × {N_post} posterior ODE evaluations...')
    # pred_endpoints shape: (N_szaf, N_post, 5)
    pred_endpoints = np.zeros((N_szaf, N_post, N_SP))

    for ki, pi in enumerate(post_idx):
        A_utri = samples[pi, :N_A]
        theta20 = np.concatenate([A_utri, b_avg])
        for si in range(N_szaf):
            phi0 = np.clip(phi_obs[si], 1e-6, 1.0)
            phi0 /= phi0.sum()
            pred_endpoints[si, ki] = run_ode_endpoint(theta20, phi0)
        if (ki + 1) % 20 == 0:
            print(f'  {ki+1}/{N_post} posterior samples done')

    # Save raw predictions
    np.save(OUT_DIR / 'pred_endpoints.npy', pred_endpoints)
    print(f'Saved: {OUT_DIR}/pred_endpoints.npy')

    # ── Summary statistics ────────────────────────────────────────────────────
    pred_median = np.median(pred_endpoints, axis=1)   # (127, 5)
    pred_lo     = np.percentile(pred_endpoints, 5,  axis=1)  # 90% CI
    pred_hi     = np.percentile(pred_endpoints, 95, axis=1)

    # Coverage: fraction of phi_obs within 90% CI
    in_ci = (phi_obs >= pred_lo) & (phi_obs <= pred_hi)   # (127, 5)
    coverage_per_sp = in_ci.mean(axis=0)
    coverage_total  = in_ci.mean()
    print(f'\n90% CI coverage:')
    for sp, c in zip(SHORT, coverage_per_sp):
        print(f'  {sp}: {c:.1%}')
    print(f'  Overall: {coverage_total:.1%}')

    # RMSE per diagnosis
    rmse_per_diag = {}
    for d in ['PIH', 'PIM', 'PI']:
        mask = diag == d
        if mask.sum() == 0:
            continue
        rmse = np.sqrt(np.mean((pred_median[mask] - phi_obs[mask])**2))
        rmse_per_diag[d] = float(rmse)
        print(f'  Predictive RMSE [{d}]: {rmse:.4f}  (n={mask.sum()})')
    rmse_overall = np.sqrt(np.mean((pred_median - phi_obs)**2))
    print(f'  Predictive RMSE [ALL]: {rmse_overall:.4f}')

    # ── Figure 1: Predicted vs observed per species (scatter + CI) ───────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for sp, ax in enumerate(axes):
        for d, col in DIAG_COLORS.items():
            mask = diag == d
            ax.errorbar(
                phi_obs[mask, sp],
                pred_median[mask, sp],
                yerr=[pred_median[mask, sp] - pred_lo[mask, sp],
                      pred_hi[mask, sp] - pred_median[mask, sp]],
                fmt='o', color=col, alpha=0.5, ms=4, lw=0.8, label=d,
                capsize=2,
            )
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel(f'Observed {SHORT[sp]}', fontsize=10)
        ax.set_ylabel(f'Predicted {SHORT[sp]}' if sp == 0 else '', fontsize=10)
        ax.set_title(f'{SHORT[sp]}\ncov={coverage_per_sp[sp]:.0%}', fontsize=10)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if sp == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f'Posterior predictive check: Dieckow θ → Szafranski 127 samples\n'
                 f'N_post={N_post}, Overall RMSE={rmse_overall:.4f}, Coverage={coverage_total:.0%}',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'postpred_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/postpred_scatter.png')

    # ── Figure 2: CI width distribution by diagnosis ──────────────────────────
    ci_width = pred_hi - pred_lo   # (127, 5)
    fig2, axes2 = plt.subplots(1, 5, figsize=(16, 4))
    for sp, ax in enumerate(axes2):
        data  = [ci_width[diag == d, sp] for d in ['PIH', 'PIM', 'PI']]
        parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True, widths=0.7)
        for body, col in zip(parts['bodies'], DIAG_COLORS.values()):
            body.set_facecolor(col); body.set_alpha(0.6)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['PIH', 'PIM', 'PI'], fontsize=9)
        ax.set_title(f'{SHORT[sp]}', fontsize=10)
        ax.set_ylabel('90% CI width' if sp == 0 else '', fontsize=10)
        ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Posterior predictive uncertainty: 90% CI width by diagnosis\n'
                 f'N_post={N_post}', fontsize=11)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'postpred_ci_width.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/postpred_ci_width.png')

    # ── Figure 3: RMSE per diagnosis bar ─────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    diags = list(rmse_per_diag.keys())
    vals  = [rmse_per_diag[d] for d in diags]
    bars  = ax3.bar(diags, vals, color=[DIAG_COLORS[d] for d in diags], alpha=0.85)
    ax3.axhline(rmse_overall, color='black', ls='--', lw=1.5,
                label=f'Overall={rmse_overall:.4f}')
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    ax3.set_ylabel('Predictive RMSE', fontsize=11)
    ax3.set_title(f'Dieckow→Szafranski predictive RMSE by diagnosis\n(N_post={N_post})',
                  fontsize=11)
    ax3.legend(fontsize=10); ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_ylim(0, max(vals) * 1.3)
    plt.tight_layout()
    fig3.savefig(OUT_DIR / 'postpred_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/postpred_rmse.png')

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        'n_szafranski': N_szaf,
        'n_posterior':  N_post,
        'rmse_overall': float(rmse_overall),
        'rmse_per_diag': rmse_per_diag,
        'coverage_90pct_total': float(coverage_total),
        'coverage_90pct_per_sp': dict(zip(SHORT, coverage_per_sp.tolist())),
    }
    with open(OUT_DIR / 'postpred_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {OUT_DIR}/postpred_summary.json')
    print(f'\nAll outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()
