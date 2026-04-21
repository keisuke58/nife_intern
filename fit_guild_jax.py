#!/usr/bin/env python3
"""
10-guild gLV fit using JAX autodiff + Adam optimiser.
Analytical gradients → orders of magnitude faster than finite-difference L-BFGS-B.

Output: results/dieckow_cr/fit_guild.json  (overwrites if RMSE improves)
"""

import json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

jax.config.update('jax_enable_x64', True)

PHI_NPY    = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
N_G      = 10
PATIENTS = list('ABCDEFGHKL')
DT       = 1.0          # 1 week
N_STEPS  = 20           # Euler steps per week
LAMBDA   = 1e-3         # reduced from 0.01


# ── JAX replicator (Euler, JIT-compiled) ──────────────────────────────────────

@jit
def euler_step(phi, b, A):
    """One Euler micro-step of replicator ODE."""
    dt = DT / N_STEPS
    f     = b + A @ phi
    fmean = jnp.dot(phi, f)
    dphi  = phi * (f - fmean)
    phi1  = jnp.clip(phi + dt * dphi, 0.0, None)
    return phi1 / (phi1.sum() + 1e-12)


@jit
def integrate_week(phi0, b, A):
    """Integrate one week with N_STEPS Euler steps."""
    def step(phi, _):
        return euler_step(phi, b, A), None
    phi1, _ = jax.lax.scan(step, phi0, None, length=N_STEPS)
    return phi1


@jit
def predict_patient(phi0, b, A):
    """Predict weeks 2 and 3 from week 1."""
    phi2 = integrate_week(phi0, b, A)
    phi3 = integrate_week(phi2, b, A)
    return phi2, phi3


@jit
def loss_fn(A, b_all, phi_obs):
    """Mean squared error over W2+W3 predictions + L2 reg on A."""
    def patient_mse(phi_p, b_p):
        phi2, phi3 = predict_patient(phi_p[0], b_p, A)
        return jnp.sum((phi2 - phi_p[1])**2) + jnp.sum((phi3 - phi_p[2])**2)
    mse = vmap(patient_mse)(phi_obs, b_all).mean() / (2 * N_G)
    reg = LAMBDA * jnp.sum(A ** 2)
    return jnp.sqrt(mse) + reg


grad_fn = jit(grad(loss_fn, argnums=(0, 1)))


# ── Adam optimiser (manual) ───────────────────────────────────────────────────

def adam_update(params, grads, m, v, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    m_new = tuple(b1 * mi + (1 - b1) * gi for mi, gi in zip(m, grads))
    v_new = tuple(b2 * vi + (1 - b2) * gi**2 for vi, gi in zip(v, grads))
    m_hat = tuple(mi / (1 - b1**t) for mi in m_new)
    v_hat = tuple(vi / (1 - b2**t) for vi in v_new)
    params_new = tuple(
        pi - lr * mh / (jnp.sqrt(vh) + eps)
        for pi, mh, vh in zip(params, m_hat, v_hat)
    )
    return params_new, m_new, v_new


def apply_constraints(A):
    """Keep diagonal ≤ 0 (self-limitation)."""
    diag_clipped = jnp.minimum(jnp.diag(A), 0.0)
    return A.at[jnp.arange(N_G), jnp.arange(N_G)].set(diag_clipped)


def main():
    phi_obs = jnp.array(np.load(PHI_NPY))   # (10, 3, 10)
    n_p     = phi_obs.shape[0]
    print(f'phi_obs: {phi_obs.shape}  λ={LAMBDA}', flush=True)

    # Warm-start from previous L-BFGS-B result if available
    prev = RESULTS_DIR / 'fit_guild.json'
    if prev.exists():
        d = json.load(open(prev))
        A     = jnp.array(d['A'])
        b_all = jnp.array(d['b_all'])
        print(f'Warm start from previous fit  RMSE={d["rmse"]:.5f}', flush=True)
    else:
        A     = jnp.zeros((N_G, N_G)).at[jnp.arange(N_G), jnp.arange(N_G)].set(-0.1)
        b_all = jnp.full((n_p, N_G), 0.1)

    # Adam state
    m = (jnp.zeros_like(A), jnp.zeros_like(b_all))
    v = (jnp.zeros_like(A), jnp.zeros_like(b_all))

    best_loss  = float(loss_fn(A, b_all, phi_obs))
    best_A     = A
    best_b_all = b_all

    t0 = time.time()
    N_EPOCHS = 5000
    LR = 3e-3

    for epoch in range(1, N_EPOCHS + 1):
        gA, gb = grad_fn(A, b_all, phi_obs)
        (A, b_all), m, v = adam_update((A, b_all), (gA, gb), m, v, epoch, lr=LR)
        A = apply_constraints(A)

        if epoch % 200 == 0 or epoch == 1:
            val = float(loss_fn(A, b_all, phi_obs))
            print(f'  epoch {epoch:5d}  loss={val:.5f}  ({time.time()-t0:.1f}s)', flush=True)
            if val < best_loss:
                best_loss  = val
                best_A     = A.copy()
                best_b_all = b_all.copy()

    A, b_all = best_A, best_b_all

    # Compute pure RMSE (no regularisation)
    def rmse_pure(A, b_all, phi_obs):
        def patient_mse(phi_p, b_p):
            phi2, phi3 = predict_patient(phi_p[0], b_p, A)
            return jnp.sum((phi2 - phi_p[1])**2) + jnp.sum((phi3 - phi_p[2])**2)
        mse = vmap(patient_mse)(phi_obs, b_all).mean() / (2 * N_G)
        return float(jnp.sqrt(mse))

    rmse_final = rmse_pure(A, b_all, phi_obs)
    print(f'\nFinal RMSE (no reg): {rmse_final:.5f}  ({time.time()-t0:.1f}s total)', flush=True)

    # Print A matrix
    A_np = np.array(A)
    print('\nEffective A matrix:', flush=True)
    header = '         ' + ''.join(f'{g[:4]:>8s}' for g in GUILD_ORDER)
    print(header, flush=True)
    for i, name in enumerate(GUILD_ORDER):
        row = ''.join(f'{A_np[i,j]:8.3f}' for j in range(N_G))
        print(f'  {name[:8]:8s} {row}', flush=True)

    out = RESULTS_DIR / 'fit_guild.json'
    json.dump(dict(
        A=A_np.tolist(), b_all=np.array(b_all).tolist(),
        rmse=rmse_final, guilds=GUILD_ORDER,
        patients=PATIENTS, success=True,
        message=f'JAX Adam {N_EPOCHS} epochs lr={LR} λ={LAMBDA}',
        n_calls=N_EPOCHS, lambda_reg=LAMBDA,
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
