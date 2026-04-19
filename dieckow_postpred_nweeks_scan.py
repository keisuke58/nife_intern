#!/usr/bin/env python3
"""
dieckow_postpred_nweeks_scan.py
  n_weeks ∈ [1, 3, 5, 10] で posterior predictive を走らせて
  RMSE・Coverage vs n_weeks を 2 パネルで可視化する。
"""
import os, sys, json, functools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/core')
from hamilton_ode_jax_nsp import simulate_0d_nsp

print(f'JAX devices: {jax.devices()}')

SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP = 5; N_A = 15; N_STEPS = 2500; DT = 1e-4; C = 25.0; ALPHA = 100.0
N_POST   = 20
N_WEEKS_LIST = [1, 3, 5, 10]

FITS_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
GMM_CSV  = Path('/home/nishioka/IKM_Hiwi/nife/results/gmm_attractor_analysis.csv')
OUT_DIR  = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_postpred_scan')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIAG_COLORS = {'PIH': '#1565C0', 'PIM': '#2E7D32', 'PI': '#B71C1C'}


# ── JIT-compiled batched ODE (static n_weeks via closure) ─────────────────────
@functools.cache
def _get_batch_fn(n_weeks: int):
    def _single(theta20, phi0):
        phi = phi0
        for _ in range(n_weeks):
            traj = simulate_0d_nsp(theta20, n_sp=N_SP, n_steps=N_STEPS, dt=DT,
                                    phi_init=phi, c_const=C, alpha_const=ALPHA)
            phi = traj[-1] / jnp.maximum(traj[-1].sum(), 1e-12)
        return phi
    return jax.jit(jax.vmap(_single, in_axes=(None, 0)))


def run_scan(phi_obs_jax, phi_obs_np, diag, post_thetas):
    """Run all n_weeks and return per-condition results dict."""
    results = {}

    for n_weeks in N_WEEKS_LIST:
        import time
        print(f'\n── n_weeks={n_weeks} ──')
        batch_fn = _get_batch_fn(n_weeks)
        N_szaf   = phi_obs_np.shape[0]
        pred     = np.zeros((N_szaf, len(post_thetas), N_SP))

        for ki, theta20 in enumerate(post_thetas):
            t0 = time.time()
            pred[:, ki] = np.array(batch_fn(theta20, phi_obs_jax))
            elapsed = time.time() - t0
            if ki == 0:
                print(f'  compiled+ran in {elapsed:.1f}s')
            elif (ki + 1) % 10 == 0:
                print(f'  {ki+1}/{len(post_thetas)} ({elapsed:.2f}s/batch)')

        pred_med = np.median(pred, axis=1)
        pred_lo  = np.percentile(pred, 5,  axis=1)
        pred_hi  = np.percentile(pred, 95, axis=1)

        in_ci    = (phi_obs_np >= pred_lo) & (phi_obs_np <= pred_hi)
        cov_total = in_ci.mean()
        cov_sp    = in_ci.mean(axis=0)

        rmse_all  = float(np.sqrt(np.mean((pred_med - phi_obs_np)**2)))
        rmse_diag = {}
        for d in ['PIH', 'PIM', 'PI']:
            mask = diag == d
            if mask.any():
                rmse_diag[d] = float(np.sqrt(np.mean((pred_med[mask] - phi_obs_np[mask])**2)))

        print(f'  RMSE={rmse_all:.4f}, Coverage={cov_total:.1%}')
        results[n_weeks] = dict(
            rmse_all=rmse_all, rmse_diag=rmse_diag,
            coverage=float(cov_total), coverage_sp=cov_sp.tolist(),
            pred_med=pred_med, pred_lo=pred_lo, pred_hi=pred_hi,
        )

    return results


def plot_results(results):
    weeks = N_WEEKS_LIST
    rmse_all = [results[w]['rmse_all'] for w in weeks]
    cov_all  = [results[w]['coverage'] * 100 for w in weeks]

    rmse_by_diag = {d: [results[w]['rmse_diag'].get(d, np.nan) for w in weeks]
                    for d in ['PIH', 'PIM', 'PI']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: RMSE vs n_weeks
    ax1.plot(weeks, rmse_all, 'ko-', lw=2, ms=8, label='All', zorder=5)
    for d, col in DIAG_COLORS.items():
        ax1.plot(weeks, rmse_by_diag[d], 'o--', color=col, lw=1.5, ms=6, label=d)
    ax1.set_xlabel('ODE weeks (n_weeks)', fontsize=12)
    ax1.set_ylabel('Predictive RMSE', fontsize=12)
    ax1.set_title('RMSE vs prediction horizon\n(Dieckow θ → Szafranski 127 samples)', fontsize=11)
    ax1.set_xticks(weeks)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Coverage vs n_weeks
    ax2.plot(weeks, cov_all, 'rs-', lw=2, ms=8)
    ax2.axhline(90, color='gray', ls='--', lw=1, label='Ideal 90%')
    ax2.set_xlabel('ODE weeks (n_weeks)', fontsize=12)
    ax2.set_ylabel('90% CI Coverage (%)', fontsize=12)
    ax2.set_title('Posterior predictive coverage\nvs prediction horizon', fontsize=11)
    ax2.set_xticks(weeks)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Attractor collapse diagnosis (N_post={N_POST})', fontsize=13, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'nweeks_scan.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out}')


def main():
    import pandas as pd

    d1      = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    samples = np.array(d1['samples'])
    gmm     = pd.read_csv(GMM_CSV)

    phi_cols = ['phi0_So', 'phi0_An', 'phi0_Vd', 'phi0_Fn', 'phi0_Pg']
    phi_obs  = gmm[phi_cols].values.astype(np.float64)
    diag     = gmm['diagnosis'].values

    phi_norm = np.clip(phi_obs, 1e-6, 1.0)
    phi_norm = phi_norm / phi_norm.sum(axis=1, keepdims=True)
    phi_jax  = jnp.array(phi_norm)

    rng      = np.random.default_rng(42)
    post_idx = rng.choice(len(samples), N_POST, replace=False)

    theta_full = np.array(d1['theta_map'])
    b_avg = np.zeros(N_SP)
    for i, p in enumerate(PATIENTS):
        if p != 'F':
            b_avg += theta_full[N_A + i*N_SP: N_A + (i+1)*N_SP]
    b_avg /= (len(PATIENTS) - 1)

    post_thetas = [jnp.array(np.concatenate([samples[pi, :N_A], b_avg]))
                   for pi in post_idx]

    print(f'Szafranski: {len(phi_obs)} samples, N_post={N_POST}')
    results = run_scan(phi_jax, phi_obs, diag, post_thetas)

    # Save summary JSON
    summary = {str(w): {k: v for k, v in r.items()
                        if not isinstance(v, np.ndarray)}
               for w, r in results.items()}
    with open(OUT_DIR / 'nweeks_scan_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {OUT_DIR}/nweeks_scan_summary.json')

    plot_results(results)
    print('Done.')


if __name__ == '__main__':
    main()
