#!/usr/bin/env python3
"""
estimate_b_szafranski.py — Inverse problem: per-patient b estimation

Given fixed A (Dieckow MAP), find b_i for each Szafranski sample such that:
  ODE(A, b_i, phi_obs_i, 1 week) ≈ phi_obs_i  (quasi-fixed-point)

Loss: ||phi_pred - phi_obs||² + λ||log_b - log_b0||²
Optimizer: Adam (optax), 300 steps, vmap over all 127 samples simultaneously.
Memory: N_STEPS=500 (instead of 2500) during optimization to fit in 24GB GPU.
Final RMSE is evaluated with N_STEPS=2500 for accuracy.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import jax
import jax.numpy as jnp
import optax
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/core')
from hamilton_ode_jax_nsp import simulate_0d_nsp

print(f'JAX devices: {jax.devices()}')

SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP = 5; N_A = 15; DT = 1e-4; C = 25.0; ALPHA = 100.0

FITS_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
GMM_CSV  = Path('/home/nishioka/IKM_Hiwi/nife/results/gmm_attractor_analysis.csv')
OUT_DIR  = Path('/home/nishioka/IKM_Hiwi/nife/results/b_szafranski')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIAG_COLORS = {'PIH': '#1565C0', 'PIM': '#2E7D32', 'PI': '#B71C1C'}
N_STEPS_OPT = 300
LR          = 0.05
LAMBDA_REG  = 0.01


def _make_vg(n_steps: int):
    """Build JIT+vmap value_and_grad with n_steps baked in as a closure constant."""
    def _run_ode(theta, phi0):
        traj = simulate_0d_nsp(theta, n_sp=N_SP, n_steps=n_steps, dt=DT,
                                phi_init=phi0, c_const=C, alpha_const=ALPHA)
        return traj[-1] / jnp.maximum(traj[-1].sum(), 1e-12)

    def _loss(log_b, A_utri, phi_obs_i, log_b0):
        theta    = jnp.concatenate([A_utri, jnp.exp(log_b)])
        phi_pred = _run_ode(theta, phi_obs_i)
        return (jnp.sum((phi_pred - phi_obs_i) ** 2)
                + LAMBDA_REG * jnp.sum((log_b - log_b0) ** 2))

    return jax.jit(
        jax.vmap(
            jax.value_and_grad(_loss, argnums=0),
            in_axes=(0, None, 0, None)
        )
    )


def optimize_b(phi_obs_jax, A_utri, b0, n_steps_opt=500):
    import time
    N      = phi_obs_jax.shape[0]
    log_b0 = jnp.log(b0)
    log_b  = jnp.tile(log_b0, (N, 1))  # (N, 5)

    vg_fn     = _make_vg(n_steps_opt)
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(log_b)
    loss_history = []

    print(f'  {N} samples × {N_SP} params, {N_STEPS_OPT} steps (n_steps={n_steps_opt})...')
    for step in range(N_STEPS_OPT):
        t0 = time.time()
        vals, grads = vg_fn(log_b, A_utri, phi_obs_jax, log_b0)
        updates, opt_state = optimizer.update(grads, opt_state)
        log_b = optax.apply_updates(log_b, updates)
        mean_loss = float(jnp.mean(vals))
        loss_history.append(mean_loss)
        if step == 0:
            print(f'  step 1: loss={mean_loss:.6f}  (compile: {time.time()-t0:.1f}s)')
        if (step + 1) % 50 == 0:
            print(f'  step {step+1}: loss={mean_loss:.6f}')

    return np.array(jnp.exp(log_b)), np.array(loss_history)


def eval_rmse(b_hat, phi_obs_jax, A_utri, n_steps=2500):
    """Evaluate fixed-point RMSE with full n_steps accuracy."""
    b_jax = jnp.array(b_hat)
    def _run(b, phi):
        return simulate_0d_nsp(jnp.concatenate([A_utri, b]), n_sp=N_SP,
                               n_steps=n_steps, dt=DT, phi_init=phi,
                               c_const=C, alpha_const=ALPHA)[-1]
    pred_raw = jax.jit(jax.vmap(_run, in_axes=(0, 0)))(b_jax, phi_obs_jax)
    pred = pred_raw / jnp.maximum(pred_raw.sum(axis=1, keepdims=True), 1e-12)
    return float(jnp.sqrt(jnp.mean((pred - phi_obs_jax) ** 2)))


def plot_results(b_hat, diag, loss_history, b0):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: convergence
    ax = axes[0, 0]
    ax.semilogy(loss_history, lw=1.5)
    ax.set_xlabel('Adam step'); ax.set_ylabel('Mean loss (log)')
    ax.set_title('Optimization convergence'); ax.grid(True, alpha=0.3)

    # Panel 2: b̂ grouped bar by diagnosis
    ax = axes[0, 1]
    x, w = np.arange(N_SP), 0.25
    for xi, (d, col) in enumerate(DIAG_COLORS.items()):
        mask = diag == d
        med  = np.median(b_hat[mask], axis=0)
        q1   = np.percentile(b_hat[mask], 25, axis=0)
        q3   = np.percentile(b_hat[mask], 75, axis=0)
        ax.bar(x + (xi - 1) * w, med, w * 0.9, color=col, alpha=0.8,
               label=d, yerr=[med - q1, q3 - med], capsize=3)
    ax.axhline(float(b0.mean()), color='k', ls='--', lw=1, label='Dieckow b̄')
    ax.set_xticks(x); ax.set_xticklabels(SHORT)
    ax.set_ylabel('b̂'); ax.set_title('b̂ median ± IQR by diagnosis')
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)

    # Panel 3: PCA of log b̂
    ax = axes[0, 2]
    pca  = PCA(n_components=2)
    b_pc = pca.fit_transform(np.log(b_hat + 1e-6))
    for d, col in DIAG_COLORS.items():
        mask = diag == d
        ax.scatter(b_pc[mask, 0], b_pc[mask, 1], c=col, alpha=0.7, s=40,
                   label=f'{d} (n={mask.sum()})')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
    ax.set_title('PCA of log b̂'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panels 4-6: strip plots for So, Fn, Pg (most informative for peri-implantitis)
    for si, (sp_i, ax) in enumerate(zip([0, 3, 4], axes[1])):
        for xi, (d, col) in enumerate(DIAG_COLORS.items()):
            vals   = b_hat[diag == d, sp_i]
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(xi + jitter, vals, c=col, alpha=0.5, s=18)
            ax.plot([xi - 0.2, xi + 0.2], [np.median(vals)] * 2, 'k-', lw=2)
        ax.axhline(float(b0[sp_i]), color='gray', ls='--', lw=1, label='Dieckow b̄')
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['PIH', 'PIM', 'PI'])
        ax.set_title(f'b̂_{SHORT[sp_i]}')
        ax.set_ylabel('b̂' if si == 0 else '')
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Inverse b̂ (fixed A from Dieckow MAP, quasi-fixed-point loss)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'b_hat_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/b_hat_analysis.png')


def main():
    import pandas as pd

    d1   = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    gmm  = pd.read_csv(GMM_CSV)

    phi_cols = ['phi0_So', 'phi0_An', 'phi0_Vd', 'phi0_Fn', 'phi0_Pg']
    phi_obs  = gmm[phi_cols].values.astype(np.float64)
    diag     = gmm['diagnosis'].values

    phi_norm = np.clip(phi_obs, 1e-6, 1.0)
    phi_norm /= phi_norm.sum(axis=1, keepdims=True)
    phi_jax  = jnp.array(phi_norm)

    theta_map = np.array(d1['theta_map'])
    A_utri    = jnp.array(theta_map[:N_A])
    b_avg     = np.zeros(N_SP)
    for i, p in enumerate(PATIENTS):
        if p != 'F':
            b_avg += theta_map[N_A + i*N_SP: N_A + (i+1)*N_SP]
    b_avg /= (len(PATIENTS) - 1)
    b0 = jnp.array(b_avg)

    print(f'Szafranski: {len(phi_obs)} samples')
    print(f'Dieckow mean b: {dict(zip(SHORT, np.round(b_avg, 2)))}')

    b_hat, loss_history = optimize_b(phi_jax, A_utri, b0)

    np.save(OUT_DIR / 'b_hat.npy', b_hat)
    np.save(OUT_DIR / 'loss_history.npy', loss_history)

    rmse_after = eval_rmse(b_hat, phi_jax, A_utri)
    print(f'\nFixed-point RMSE after opt (n_steps=2500): {rmse_after:.4f}')
    print('(cf. forward n_weeks=1 RMSE=0.0836 with Dieckow mean b)')

    print('\n── b̂ medians by diagnosis ──')
    for d in ['PIH', 'PIM', 'PI']:
        mask = diag == d
        med  = np.median(b_hat[mask], axis=0)
        print(f'  {d} (n={mask.sum()}): ' +
              ', '.join(f'{s}={v:.2f}' for s, v in zip(SHORT, med)))

    plot_results(b_hat, diag, loss_history, b0)

    summary = {
        'n_samples': int(len(phi_obs)),
        'n_steps_opt': N_STEPS_OPT, 'lr': LR,
        'lambda_reg': LAMBDA_REG,
        'final_loss': float(loss_history[-1]),
        'rmse_after_opt': rmse_after,
        'dieckow_b0': dict(zip(SHORT, b_avg.tolist())),
        'b_medians_by_diag': {
            d: dict(zip(SHORT, np.median(b_hat[diag == d], axis=0).tolist()))
            for d in ['PIH', 'PIM', 'PI']
        }
    }
    with open(OUT_DIR / 'b_hat_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nAll outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()
