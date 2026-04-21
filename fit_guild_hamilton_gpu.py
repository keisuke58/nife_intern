#!/usr/bin/env python3
"""
10-guild Hamilton ODE fit — GPU + JAX vmap + Adam.

Runs simulate_0d_nsp(n_sp=10) for all 10 patients in parallel via vmap.
JAX autodiff (grad) through the Hamilton ODE → Adam optimiser.

Usage (on vancouver01):
  CUDA_VISIBLE_DEVICES=1 python3 fit_guild_hamilton_gpu.py

Output: results/dieckow_cr/fit_guild_hamilton.json
"""

import json, sys, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
from hamilton_ode_jax_nsp import simulate_0d_nsp

PHI_NPY     = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SP    = 10
N_A     = N_SP * (N_SP + 1) // 2   # 55 upper-triangle params (symmetric A)
N_STEPS = 200
LAMBDA  = 1e-3
N_P     = 10

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
PATIENTS = list('ABCDEFGHKL')


# ── Single-patient equilibrium (differentiable) ───────────────────────────────

@jit
def eq_phi_one(A_upper, b_p, phi_init):
    """Hamilton ODE to equilibrium → normalised phibar for one patient/week."""
    theta    = jnp.concatenate([A_upper, b_p])
    phibar   = simulate_0d_nsp(theta, n_sp=N_SP, n_steps=N_STEPS,
                                dt=1e-4, phi_init=phi_init,
                                c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]
    s  = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(N_SP) / N_SP)


# ── Batched over patients via vmap ────────────────────────────────────────────

@jit
def loss_fn(A_upper, b_all, phi_obs):
    """RMSE over W2+W3 for all patients + L2 reg on A."""
    def patient_loss(b_p, phi_p):
        phi_W2 = eq_phi_one(A_upper, b_p, phi_p[0])
        phi_W3 = eq_phi_one(A_upper, b_p, phi_W2)
        return (jnp.sum((phi_W2 - phi_p[1]) ** 2) +
                jnp.sum((phi_W3 - phi_p[2]) ** 2))

    sq  = vmap(patient_loss)(b_all, phi_obs).mean() / (2 * N_SP)
    reg = LAMBDA * jnp.sum(A_upper ** 2)
    return jnp.sqrt(sq) + reg


grad_fn = jit(grad(loss_fn, argnums=(0, 1)))


# ── Adam ─────────────────────────────────────────────────────────────────────

def adam_step(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m_new = tuple(b1 * mi + (1 - b1) * gi for mi, gi in zip(m, grads))
    v_new = tuple(b2 * vi + (1 - b2) * gi**2 for vi, gi in zip(v, grads))
    mh = tuple(mi / (1 - b1**t) for mi in m_new)
    vh = tuple(vi / (1 - b2**t) for vi in v_new)
    p_new = tuple(pi - lr * mhi / (jnp.sqrt(vhi) + eps)
                  for pi, mhi, vhi in zip(params, mh, vh))
    return p_new, m_new, v_new


def apply_diag_constraint(A_upper):
    """Keep diagonal entries (A_ii) ≤ 0."""
    diag_idx = []
    idx = 0
    for j in range(N_SP):
        for i in range(j + 1):
            if i == j:
                diag_idx.append(idx)
            idx += 1
    for k in diag_idx:
        A_upper = A_upper.at[k].set(jnp.minimum(A_upper[k], 0.0))
    return A_upper


def rmse_pure(A_upper, b_all, phi_obs):
    """RMSE without regularisation."""
    def patient_mse(b_p, phi_p):
        phi_W2 = eq_phi_one(A_upper, b_p, phi_p[0])
        phi_W3 = eq_phi_one(A_upper, b_p, phi_W2)
        return (jnp.sum((phi_W2 - phi_p[1]) ** 2) +
                jnp.sum((phi_W3 - phi_p[2]) ** 2))
    sq = vmap(patient_mse)(b_all, phi_obs).mean() / (2 * N_SP)
    return float(jnp.sqrt(sq))


def default_A_upper():
    A = -0.1 * np.eye(N_SP)
    upper = []
    for j in range(N_SP):
        for i in range(j + 1):
            upper.append(A[i, j])
    return jnp.array(upper)


def main():
    print(f'JAX devices: {jax.devices()}', flush=True)

    phi_obs = jnp.array(np.load(PHI_NPY))
    print(f'phi_obs: {phi_obs.shape}', flush=True)

    # Warm start from gLV result (symmetrise A)
    prev = RESULTS_DIR / 'fit_guild.json'
    if prev.exists():
        d     = json.load(open(prev))
        A_glv = np.array(d['A'])
        A_sym = (A_glv + A_glv.T) / 2.0
        A_upper = []
        for j in range(N_SP):
            for i in range(j + 1):
                A_upper.append(A_sym[i, j])
        A_upper = jnp.array(A_upper)
        b_all   = jnp.array(d['b_all'])
        print('Warm start: gLV → symmetrised A', flush=True)
    else:
        A_upper = default_A_upper()
        b_all   = jnp.full((N_P, N_SP), 0.1)

    # Adam state
    m = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))
    v = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))

    print('\nWarm-up JIT compile (may take 10-30 min)...', flush=True)
    t0 = time.time()
    _ = loss_fn(A_upper, b_all, phi_obs)
    print(f'First forward done in {time.time()-t0:.1f}s', flush=True)

    # Compile grad
    print('Compiling grad...', flush=True)
    _ = grad_fn(A_upper, b_all, phi_obs)
    print(f'Grad compiled in {time.time()-t0:.1f}s', flush=True)

    best_loss = float(loss_fn(A_upper, b_all, phi_obs))
    best_A, best_b = A_upper, b_all

    N_EPOCHS = 3000
    LR = 1e-3

    for epoch in range(1, N_EPOCHS + 1):
        gA, gb = grad_fn(A_upper, b_all, phi_obs)
        (A_upper, b_all), m, v = adam_step(
            (A_upper, b_all), (gA, gb), m, v, epoch, lr=LR)
        A_upper = apply_diag_constraint(A_upper)

        if epoch % 100 == 0 or epoch == 1:
            val = float(loss_fn(A_upper, b_all, phi_obs))
            print(f'  epoch {epoch:5d}  loss={val:.5f}  ({time.time()-t0:.1f}s)',
                  flush=True)
            if val < best_loss:
                best_loss = val
                best_A, best_b = A_upper, b_all

    A_upper, b_all = best_A, best_b
    rmse = rmse_pure(A_upper, b_all, phi_obs)
    print(f'\nFinal RMSE: {rmse:.5f}  ({time.time()-t0:.1f}s)', flush=True)

    # Reconstruct full symmetric A
    A_full = np.zeros((N_SP, N_SP))
    idx = 0
    for j in range(N_SP):
        for i in range(j + 1):
            v_ = float(A_upper[idx])
            A_full[i, j] = v_
            A_full[j, i] = v_
            idx += 1

    print('\nEffective A matrix:', flush=True)
    header = '         ' + ''.join(f'{g[:4]:>8s}' for g in GUILD_ORDER)
    print(header, flush=True)
    for i, name in enumerate(GUILD_ORDER):
        row = ''.join(f'{A_full[i,j]:8.3f}' for j in range(N_SP))
        print(f'  {name[:8]:8s} {row}', flush=True)

    out = RESULTS_DIR / 'fit_guild_hamilton.json'
    json.dump(dict(
        A=A_full.tolist(), A_upper=np.array(A_upper).tolist(),
        b_all=np.array(b_all).tolist(), rmse=rmse,
        guilds=GUILD_ORDER, patients=PATIENTS,
        n_steps=N_STEPS, lambda_reg=LAMBDA,
        message=f'Hamilton ODE GPU vmap Adam n_steps={N_STEPS} lr={LR}',
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
