#!/usr/bin/env python3
"""
estimate_Ab_dieckow_timeseries.py — Per-patient A+b from 3-week time-series

Uses Dieckow 10-patient × 3-week composition data.
IC = week 1, predict week 2 (1w ODE) and week 3 (2w ODE).
Loss = Σ_{t∈{2,3}} ||phi_pred(t) - phi_obs(t)||² + λ_A||A−A0||² + λ_b||log_b−log_b0||²

Result: per-patient A_i (identifiable from dynamics), then compare to Szafranski b-only.
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
N_STEPS_1W = 2500  # ODE steps per week (matches TMCMC calibration)

FITS_DIR   = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
OBS_JSON   = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_obs_matrix_5sp.json')
OUT_DIR    = Path('/home/nishioka/IKM_Hiwi/nife/results/Ab_dieckow_timeseries')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# n_steps per week during backprop — keep low to stay under XLA 1024-buffer limit
# (2 chained weeks × vmap10 patients → 1040 buffers at 500 steps → capped at 200)
N_STEPS_OPT = 200
LR          = 0.01
LAMBDA_A    = 0.01
LAMBDA_B    = 0.01
N_OPT_STEPS = 600


def _make_single_vg(n_steps_per_week: int, A0: jnp.ndarray, log_b0: jnp.ndarray):
    """Split week2/week3 losses into separate JIT calls to stay under XLA 1024-buffer limit.

    Two chained ODE value_and_grads exceed 1024 XLA kernel args on CUDA 12.4.
    By differentiating through each week independently (phi_w2 treated as constant
    for the week3 gradient), each JIT call sees only one ODE's backward pass.
    """

    def _run_week(theta, phi_start):
        traj = simulate_0d_nsp(theta, n_sp=N_SP, n_steps=n_steps_per_week, dt=DT,
                               phi_init=phi_start, c_const=C, alpha_const=ALPHA)
        raw = traj[-1]
        return raw / jnp.maximum(raw.sum(), 1e-12)

    # Forward-only: get phi_w2 to use as fixed IC for week3 gradient
    run_week_fwd = jax.jit(_run_week)

    def _loss_w2(params, phi_obs_i):
        A_utri = params[:N_A]; log_b = params[N_A:]
        theta = jnp.concatenate([A_utri, jnp.exp(log_b)])
        phi_w2 = _run_week(theta, phi_obs_i[0])
        return (jnp.sum((phi_w2 - phi_obs_i[1]) ** 2)
                + LAMBDA_A * jnp.sum((A_utri - A0) ** 2)
                + LAMBDA_B * jnp.sum((log_b - log_b0) ** 2))

    def _loss_w3(params, phi_w2_fixed, phi_obs_3):
        A_utri = params[:N_A]; log_b = params[N_A:]
        theta = jnp.concatenate([A_utri, jnp.exp(log_b)])
        phi_w3 = _run_week(theta, phi_w2_fixed)
        return jnp.sum((phi_w3 - phi_obs_3) ** 2)

    vg_w2 = jax.jit(jax.value_and_grad(_loss_w2, argnums=0))
    vg_w3 = jax.jit(jax.value_and_grad(_loss_w3, argnums=0))

    def combined_vg(params, phi_obs_i):
        theta = jnp.concatenate([params[:N_A], jnp.exp(params[N_A:])])
        phi_w2 = run_week_fwd(theta, phi_obs_i[0])   # fixed IC for week3
        v2, g2 = vg_w2(params, phi_obs_i)
        v3, g3 = vg_w3(params, phi_w2, phi_obs_i[2])
        return v2 + v3, g2 + g3

    return combined_vg


def load_dieckow_timeseries():
    """Load obs matrix → (N_patients, 3, 5) normalized phi."""
    raw = json.load(open(OBS_JSON))
    phi_list = []
    valid_patients = []
    for p in PATIENTS:
        obs_p = np.array(raw['obs'][p])  # (5, 3): [species, week]
        phi_w = obs_p.T                   # (3, 5): [week, species]
        # Skip if any week is all zeros or NaN
        if np.any(np.isnan(phi_w)) or np.any(phi_w.sum(axis=1) < 0.01):
            print(f'  Skip patient {p} (missing weeks)')
            continue
        phi_w = np.clip(phi_w, 1e-6, 1.0)
        phi_w = phi_w / phi_w.sum(axis=1, keepdims=True)
        phi_list.append(phi_w)
        valid_patients.append(p)
    return np.array(phi_list), valid_patients  # (N, 3, 5)


def optimize_Ab_timeseries(phi_obs, A0, log_b0):
    import time
    N       = phi_obs.shape[0]
    phi_jax = jnp.array(phi_obs)
    params0 = jnp.concatenate([A0, jnp.exp(log_b0)])
    params  = [params0] * N  # list of per-patient params

    single_vg = _make_single_vg(N_STEPS_OPT, A0, log_b0)
    optimizers  = [optax.adam(LR) for _ in range(N)]
    opt_states  = [opt.init(params[i]) for i, opt in enumerate(optimizers)]
    history     = []

    print(f'  {N} patients × {N_A+N_SP} params, {N_OPT_STEPS} steps'
          f' (n_steps/week={N_STEPS_OPT}, sequential per patient)...')
    for step in range(N_OPT_STEPS):
        t0 = time.time()
        vals_list, grads_list = [], []
        for i in range(N):
            v, g = single_vg(params[i], phi_jax[i])
            vals_list.append(float(v))
            grads_list.append(g)
            updates, opt_states[i] = optimizers[i].update(g, opt_states[i])
            params[i] = optax.apply_updates(params[i], updates)
        ml = float(np.mean(vals_list))
        history.append(ml)
        if step == 0:
            print(f'  step 1: loss={ml:.6f}  (compile: {time.time()-t0:.1f}s)')
        if (step + 1) % 100 == 0:
            print(f'  step {step+1}: loss={ml:.6f}')

    params_np = np.array([np.array(p) for p in params])
    A_hat = params_np[:, :N_A]
    b_hat = np.exp(params_np[:, N_A:])
    return A_hat, b_hat, np.array(history)


def eval_timeseries_rmse(A_hat, b_hat, phi_obs, n_steps=2500):
    """Evaluate RMSE on week-2 and week-3 predictions."""
    phi_jax = jnp.array(phi_obs)
    A_jax   = jnp.array(A_hat)
    b_jax   = jnp.array(b_hat)

    def _predict(A_utri, b, phi_obs_i):
        theta = jnp.concatenate([A_utri, b])
        phi_ic = phi_obs_i[0]
        def _run(phi_start):
            traj = simulate_0d_nsp(theta, n_sp=N_SP, n_steps=n_steps, dt=DT,
                                   phi_init=phi_start, c_const=C, alpha_const=ALPHA)
            raw = traj[-1]
            return raw / jnp.maximum(raw.sum(), 1e-12)
        phi_w2 = _run(phi_ic)
        phi_w3 = _run(phi_w2)
        return jnp.stack([phi_w2, phi_w3])  # (2, 5)

    pred = jax.jit(jax.vmap(_predict, in_axes=(0, 0, 0)))(A_jax, b_jax, phi_jax)
    obs_w23 = phi_obs[:, 1:, :]  # (N, 2, 5)
    return float(jnp.sqrt(jnp.mean((pred - obs_w23) ** 2)))


def plot_results(A_hat, b_hat, patients, history, A0, b0_np, phi_obs):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Convergence
    axes[0, 0].semilogy(history, lw=1.5)
    axes[0, 0].set(xlabel='Adam step', ylabel='Loss (log)',
                   title='Time-series A+b convergence')
    axes[0, 0].grid(True, alpha=0.3)

    # A matrix shift per patient
    ax = axes[0, 1]
    delta_A = A_hat - np.array(A0)
    im = ax.imshow(delta_A, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(delta_A).max(), vmax=np.abs(delta_A).max())
    ax.set_xticks(range(N_A)); ax.set_xlabel('A param index')
    ax.set_yticks(range(len(patients))); ax.set_yticklabels(patients)
    ax.set_title('ΔA per patient (vs Dieckow MAP)')
    plt.colorbar(im, ax=ax)

    # b̂ per patient
    ax = axes[0, 2]
    x = np.arange(N_SP)
    for pi, p in enumerate(patients):
        ax.plot(x, b_hat[pi], 'o-', ms=5, lw=1, label=p, alpha=0.8)
    ax.plot(x, b0_np, 'k--', lw=2, label='MAP mean', zorder=10)
    ax.set_xticks(x); ax.set_xticklabels(SHORT)
    ax.set(ylabel='b̂', title='b̂ per Dieckow patient (time-series fit)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Predicted vs observed week 2 & 3 per patient
    A_jax = jnp.array(A_hat); b_jax = jnp.array(b_hat)
    colors = plt.cm.tab10(np.linspace(0, 1, len(patients)))
    for wi, wlabel in enumerate(['Week 2', 'Week 3']):
        ax = axes[1, wi]
        for pi, (p, col) in enumerate(zip(patients, colors)):
            theta = np.concatenate([A_hat[pi], b_hat[pi]])
            phi_ic = phi_obs[pi, 0]
            from hamilton_ode_jax_nsp import simulate_0d_nsp as sim
            phi_cur = phi_ic
            for _ in range(wi + 1):
                traj = sim(jnp.array(theta), n_sp=N_SP, n_steps=2500, dt=DT,
                           phi_init=jnp.array(phi_cur), c_const=C, alpha_const=ALPHA)
                raw = np.array(traj[-1])
                phi_cur = raw / max(raw.sum(), 1e-12)
            obs_w = phi_obs[pi, wi + 1]
            ax.scatter(obs_w, phi_cur, c=[col], s=30, alpha=0.8, label=p)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlabel=f'Observed {wlabel}', ylabel='Predicted',
               title=f'{wlabel}: predicted vs observed', xlim=(-0.05, 1.05),
               ylim=(-0.05, 1.05))
        ax.grid(True, alpha=0.3)

    # A matrix comparison: MAP vs patient mean
    ax = axes[1, 2]
    A_mean = A_hat.mean(axis=0)
    ax.bar(range(N_A), A_mean - np.array(A0), color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='A param index', ylabel='ΔA (patient mean vs MAP)',
           title='Mean A shift from time-series fit')
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Per-patient A+b from Dieckow 3-week time-series', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'Ab_timeseries_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/Ab_timeseries_analysis.png')


def main():
    d1 = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))

    theta_map = np.array(d1['theta_map'])
    A0        = jnp.array(theta_map[:N_A])
    b_avg     = np.zeros(N_SP)
    for i, p in enumerate(PATIENTS):
        if p != 'F':
            b_avg += theta_map[N_A + i*N_SP: N_A + (i+1)*N_SP]
    b_avg /= (len(PATIENTS) - 1)
    b0     = jnp.array(b_avg)
    log_b0 = jnp.log(b0)

    phi_obs, valid_patients = load_dieckow_timeseries()
    print(f'Dieckow time-series: {len(valid_patients)} patients, shape={phi_obs.shape}')
    print(f'Patients: {valid_patients}')

    A_hat, b_hat, history = optimize_Ab_timeseries(phi_obs, A0, log_b0)

    np.save(OUT_DIR / 'A_hat.npy', A_hat)
    np.save(OUT_DIR / 'b_hat.npy', b_hat)
    np.save(OUT_DIR / 'loss_history.npy', history)
    np.save(OUT_DIR / 'phi_obs.npy', phi_obs)

    rmse = eval_timeseries_rmse(A_hat, b_hat, phi_obs)
    print(f'\nTime-series RMSE (weeks 2+3, n_steps=2500): {rmse:.4f}')

    # Baseline: Dieckow MAP predicting weeks 2 & 3
    phi_obs_jax = jnp.array(phi_obs)
    def _pred_map(phi_obs_i):
        theta = jnp.concatenate([A0, b0])
        phi_cur = phi_obs_i[0]
        preds = []
        for _ in range(2):
            traj = simulate_0d_nsp(theta, n_sp=N_SP, n_steps=2500, dt=DT,
                                   phi_init=phi_cur, c_const=C, alpha_const=ALPHA)
            raw = traj[-1] / jnp.maximum(traj[-1].sum(), 1e-12)
            preds.append(raw)
            phi_cur = raw
        return jnp.stack(preds)
    pred_map = jax.jit(jax.vmap(_pred_map))(phi_obs_jax)
    rmse_map = float(jnp.sqrt(jnp.mean((pred_map - phi_obs_jax[:, 1:]) ** 2)))
    print(f'Baseline RMSE (MAP theta, weeks 2+3):  {rmse_map:.4f}')

    print('\n── b̂ per patient ──')
    for pi, p in enumerate(valid_patients):
        print(f'  {p}: ' + ', '.join(f'{s}={v:.2f}' for s, v in zip(SHORT, b_hat[pi])))

    plot_results(A_hat, b_hat, valid_patients, history, A0, b_avg, phi_obs)

    summary = {
        'valid_patients': valid_patients,
        'n_steps_opt': N_STEPS_OPT, 'lr': LR,
        'lambda_A': LAMBDA_A, 'lambda_b': LAMBDA_B,
        'rmse_timeseries': rmse,
        'rmse_map_baseline': rmse_map,
        'b_hat': {p: dict(zip(SHORT, b_hat[pi].tolist()))
                  for pi, p in enumerate(valid_patients)},
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nAll outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()
