#!/usr/bin/env python3
"""
estimate_Ab_szafranski.py — Joint per-patient A+b inverse problem

Fix nothing: jointly optimize A_i (15 params) + b_i (5 params) per patient.
Loss: ||ODE(A_i, b_i, φ_obs, 1week) − φ_obs||² + λ_A||A_i−A0||² + λ_b||log_b_i−log_b0||²

Memory: n_steps=300 for backprop (20 params × 127 samples, ~12GB est.)
Final RMSE evaluated with n_steps=2500.
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
OUT_DIR  = Path('/home/nishioka/IKM_Hiwi/nife/results/Ab_szafranski')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIAG_COLORS = {'PIH': '#1565C0', 'PIM': '#2E7D32', 'PI': '#B71C1C'}
N_STEPS_OPT = 500
LR          = 0.02
LAMBDA_A    = 0.1   # stronger reg for A (15 params, harder to identify)
LAMBDA_B    = 0.01
N_OPT_STEPS = 400


def _make_vg(n_steps: int, A0: jnp.ndarray, log_b0: jnp.ndarray):
    """JIT+vmap value_and_grad with all constants baked in."""
    def _run(A_utri, log_b, phi0):
        theta = jnp.concatenate([A_utri, jnp.exp(log_b)])
        traj  = simulate_0d_nsp(theta, n_sp=N_SP, n_steps=n_steps, dt=DT,
                                 phi_init=phi0, c_const=C, alpha_const=ALPHA)
        return traj[-1] / jnp.maximum(traj[-1].sum(), 1e-12)

    def _loss(params, phi_obs_i):
        A_utri = params[:N_A]
        log_b  = params[N_A:]
        phi_pred = _run(A_utri, log_b, phi_obs_i)
        return (jnp.sum((phi_pred - phi_obs_i) ** 2)
                + LAMBDA_A * jnp.sum((A_utri - A0) ** 2)
                + LAMBDA_B * jnp.sum((log_b - log_b0) ** 2))

    return jax.jit(
        jax.vmap(
            jax.value_and_grad(_loss, argnums=0),
            in_axes=(0, 0)
        )
    )


def optimize_Ab(phi_obs_jax, A0, log_b0):
    import time
    N     = phi_obs_jax.shape[0]
    n_par = N_A + N_SP

    # Init: tile Dieckow MAP A + mean log_b
    params0 = jnp.concatenate([A0, log_b0])
    params  = jnp.tile(params0, (N, 1))  # (N, 20)

    vg_fn     = _make_vg(N_STEPS_OPT, A0, log_b0)
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)
    history   = []

    print(f'  {N} samples × {n_par} params, {N_OPT_STEPS} steps (n_steps={N_STEPS_OPT})...')
    for step in range(N_OPT_STEPS):
        t0 = time.time()
        vals, grads = vg_fn(params, phi_obs_jax)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ml = float(jnp.mean(vals))
        history.append(ml)
        if step == 0:
            print(f'  step 1: loss={ml:.6f}  (compile: {time.time()-t0:.1f}s)')
        if (step + 1) % 50 == 0:
            print(f'  step {step+1}: loss={ml:.6f}')

    A_hat     = np.array(params[:, :N_A])
    b_hat     = np.array(jnp.exp(params[:, N_A:]))
    return A_hat, b_hat, np.array(history)


def eval_rmse(A_hat, b_hat, phi_obs_jax, n_steps=2500):
    A_jax = jnp.array(A_hat)
    b_jax = jnp.array(b_hat)
    def _run(A_utri, b, phi):
        traj = simulate_0d_nsp(jnp.concatenate([A_utri, b]), n_sp=N_SP,
                               n_steps=n_steps, dt=DT, phi_init=phi,
                               c_const=C, alpha_const=ALPHA)
        raw = traj[-1]
        return raw / jnp.maximum(raw.sum(), 1e-12)
    pred = jax.jit(jax.vmap(_run, in_axes=(0, 0, 0)))(A_jax, b_jax, phi_obs_jax)
    return float(jnp.sqrt(jnp.mean((pred - phi_obs_jax) ** 2)))


def plot_results(A_hat, b_hat, diag, history, A0, b0):
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Convergence
    axes[0, 0].semilogy(history, lw=1.5)
    axes[0, 0].set(xlabel='Adam step', ylabel='Mean loss (log)',
                   title='A+b joint optimization'); axes[0, 0].grid(True, alpha=0.3)

    # b̂ comparison
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
    ax.set(ylabel='b̂', title='b̂ by diagnosis (joint A+b opt)')
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)

    # PCA of [log_A, log_b]
    ax = axes[0, 2]
    feats = np.concatenate([A_hat, np.log(b_hat + 1e-6)], axis=1)
    pca   = PCA(n_components=2)
    pc    = pca.fit_transform(feats)
    for d, col in DIAG_COLORS.items():
        mask = diag == d
        ax.scatter(pc[mask, 0], pc[mask, 1], c=col, alpha=0.7, s=40,
                   label=f'{d} (n={mask.sum()})')
    ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.0%})',
           ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.0%})',
           title='PCA of [A, log b̂]')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # A matrix shift per diagnosis
    ax = axes[1, 0]
    for d, col in DIAG_COLORS.items():
        mask = diag == d
        delta_A = np.mean(A_hat[mask], axis=0) - np.array(A0)
        ax.bar(np.arange(N_A) + list(DIAG_COLORS.keys()).index(d) * 0.25,
               delta_A, 0.25, color=col, alpha=0.7, label=d)
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='A param index', ylabel='ΔA (vs Dieckow MAP)',
           title='A shift by diagnosis'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Strip plots b̂ So, Fn, Pg
    rng = np.random.default_rng(0)
    for si, (sp_i, ax) in enumerate(zip([0, 3, 4], axes[1, 1:])):
        for xi, (d, col) in enumerate(DIAG_COLORS.items()):
            vals   = b_hat[diag == d, sp_i]
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(xi + jitter, vals, c=col, alpha=0.5, s=18)
            ax.plot([xi - 0.2, xi + 0.2], [np.median(vals)] * 2, 'k-', lw=2)
        ax.axhline(float(b0[sp_i]), color='gray', ls='--', lw=1)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['PIH', 'PIM', 'PI'])
        ax.set_title(f'b̂_{SHORT[sp_i]} (joint opt)')
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Joint A+b̂ inverse (fixed-point loss, Dieckow prior)', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'Ab_hat_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/Ab_hat_analysis.png')


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
    A0        = jnp.array(theta_map[:N_A])
    b_avg     = np.zeros(N_SP)
    for i, p in enumerate(PATIENTS):
        if p != 'F':
            b_avg += theta_map[N_A + i*N_SP: N_A + (i+1)*N_SP]
    b_avg /= (len(PATIENTS) - 1)
    b0     = jnp.array(b_avg)
    log_b0 = jnp.log(b0)

    print(f'Szafranski: {len(phi_obs)} samples')
    print(f'Joint opt: {N_A} A params + {N_SP} b params = {N_A+N_SP} per patient')

    A_hat, b_hat, history = optimize_Ab(phi_jax, A0, log_b0)

    np.save(OUT_DIR / 'A_hat.npy', A_hat)
    np.save(OUT_DIR / 'b_hat.npy', b_hat)
    np.save(OUT_DIR / 'loss_history.npy', history)

    rmse = eval_rmse(A_hat, b_hat, phi_jax)
    print(f'\nFixed-point RMSE (n_steps=2500): {rmse:.4f}')
    print('(cf. b-only RMSE=0.0753, Dieckow mean RMSE=0.0836)')

    print('\n── b̂ medians by diagnosis ──')
    for d in DIAGS:
        mask = diag == d
        med  = np.median(b_hat[mask], axis=0)
        print(f'  {d} (n={mask.sum()}): ' + ', '.join(f'{s}={v:.2f}' for s, v in zip(SHORT, med)))

    plot_results(A_hat, b_hat, diag, history, A0, b0)

    summary = {
        'n_samples': int(len(phi_obs)),
        'n_steps_opt': N_STEPS_OPT, 'lr': LR,
        'lambda_A': LAMBDA_A, 'lambda_b': LAMBDA_B,
        'rmse_Ab_joint': rmse,
        'rmse_b_only': 0.0753,
        'rmse_baseline': 0.0836,
    }
    with open(OUT_DIR / 'Ab_hat_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nAll outputs → {OUT_DIR}')


DIAGS = ['PIH', 'PIM', 'PI']

if __name__ == '__main__':
    main()
