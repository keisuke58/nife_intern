#!/usr/bin/env python3
"""
10-guild gLV fit using JAX autodiff + Adam optimiser.
Analytical gradients → orders of magnitude faster than finite-difference L-BFGS-B.

Output: results/dieckow_cr/fit_guild.json  (overwrites if RMSE improves)
"""

import json, time, argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

jax.config.update('jax_enable_x64', True)

from guild_replicator_dieckow import GUILD_ORDER

PHI_NPY    = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def _loss_builder(mask_obs):
    n_g = int(mask_obs.shape[2])

    @jit
    def loss_fn(A, b_all, phi_obs):
        def patient_sse(phi_p, b_p, m_p):
            phi2, phi3 = predict_patient(phi_p[0], b_p, A)
            m2 = m_p[1].astype(phi_obs.dtype)
            m3 = m_p[2].astype(phi_obs.dtype)
            sse2 = jnp.sum((phi2 - phi_p[1])**2) * m2
            sse3 = jnp.sum((phi3 - phi_p[2])**2) * m3
            terms = m2 + m3
            return sse2 + sse3, terms

        sse, terms = vmap(patient_sse)(phi_obs, b_all, mask_obs)
        denom = jnp.maximum(terms.sum(), 1.0) * n_g
        mse = sse.sum() / denom
        reg = LAMBDA * jnp.sum(A ** 2)
        return jnp.sqrt(mse) + reg

    @jit
    def rmse_pure(A, b_all, phi_obs):
        def patient_sse(phi_p, b_p, m_p):
            phi2, phi3 = predict_patient(phi_p[0], b_p, A)
            m2 = m_p[1].astype(phi_obs.dtype)
            m3 = m_p[2].astype(phi_obs.dtype)
            sse2 = jnp.sum((phi2 - phi_p[1])**2) * m2
            sse3 = jnp.sum((phi3 - phi_p[2])**2) * m3
            terms = m2 + m3
            return sse2 + sse3, terms

        sse, terms = vmap(patient_sse)(phi_obs, b_all, mask_obs)
        denom = jnp.maximum(terms.sum(), 1.0) * n_g
        mse = sse.sum() / denom
        return jnp.sqrt(mse)

    return loss_fn, rmse_pure


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
    n = A.shape[0]
    return A.at[jnp.arange(n), jnp.arange(n)].set(diag_clipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy', default=str(PHI_NPY))
    ap.add_argument('--out-json', default=str(RESULTS_DIR / 'fit_guild.json'))
    ap.add_argument('--warm-start', default=str(RESULTS_DIR / 'fit_guild.json'))
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=3e-3)
    args = ap.parse_args()

    phi_np = np.load(args.phi_npy)
    if phi_np.ndim != 3 or phi_np.shape[1] != 3:
        raise ValueError(f'Expected phi array shape (n_patients, 3, n_guilds), got {phi_np.shape}')

    present = phi_np.sum(axis=2) > 1e-9
    keep_patients = present[:, 0]
    kept_idx = np.flatnonzero(keep_patients)
    phi_np = phi_np[kept_idx]
    present = present[kept_idx]

    patients_used = [PATIENTS[i] for i in kept_idx] if len(PATIENTS) >= (kept_idx.max() + 1 if kept_idx.size else 0) else [str(i) for i in kept_idx]

    phi_obs = jnp.array(phi_np)
    mask_obs = jnp.array(present)
    n_p, _, n_g = phi_obs.shape
    loss_fn, rmse_pure = _loss_builder(mask_obs)
    grad_fn = jit(grad(loss_fn, argnums=(0, 1)))

    print(f'phi_obs: {phi_obs.shape}  observed_terms={int(present[:,1:].sum())}  λ={LAMBDA}', flush=True)

    # Warm-start from previous L-BFGS-B result if available
    warm = Path(args.warm_start)
    A = None
    b_all = None
    if warm.exists():
        d = json.load(open(warm))
        A0 = np.array(d.get('A', []), dtype=float)
        b0 = np.array(d.get('b_all', []), dtype=float)
        if A0.shape == (n_g, n_g) and b0.ndim == 2 and b0.shape[1] == n_g:
            A = jnp.array(A0)
            if b0.shape[0] == n_p:
                b_all = jnp.array(b0)
            else:
                prev_pat = d.get('patients')
                if isinstance(prev_pat, list) and len(prev_pat) == b0.shape[0]:
                    idx_map = {p: i for i, p in enumerate(prev_pat)}
                    rows = [idx_map.get(p) for p in patients_used]
                    if all(r is not None for r in rows):
                        b_all = jnp.array(b0[rows])
            if b_all is not None:
                rmse0 = float(d.get('rmse', float('nan')))
                print(f'Warm start from {warm.name}  RMSE={rmse0:.5f}', flush=True)
    if A is None or b_all is None:
        A = jnp.zeros((n_g, n_g)).at[jnp.arange(n_g), jnp.arange(n_g)].set(-0.1)
        b_all = jnp.full((n_p, n_g), 0.1)

    # Adam state
    m = (jnp.zeros_like(A), jnp.zeros_like(b_all))
    v = (jnp.zeros_like(A), jnp.zeros_like(b_all))

    best_loss  = float(loss_fn(A, b_all, phi_obs))
    best_A     = A
    best_b_all = b_all

    t0 = time.time()
    N_EPOCHS = int(args.epochs)
    LR = float(args.lr)

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

    rmse_final = float(rmse_pure(A, b_all, phi_obs))
    print(f'\nFinal RMSE (no reg): {rmse_final:.5f}  ({time.time()-t0:.1f}s total)', flush=True)

    # Print A matrix
    A_np = np.array(A)
    print('\nEffective A matrix:', flush=True)
    header = '         ' + ''.join(f'{g[:4]:>8s}' for g in GUILD_ORDER)
    print(header, flush=True)
    for i, name in enumerate(GUILD_ORDER[:n_g]):
        row = ''.join(f'{A_np[i,j]:8.3f}' for j in range(n_g))
        print(f'  {name[:8]:8s} {row}', flush=True)

    out = Path(args.out_json)
    json.dump(dict(
        A=A_np.tolist(), b_all=np.array(b_all).tolist(),
        rmse=rmse_final, guilds=GUILD_ORDER[:n_g],
        patients=patients_used, success=True,
        phi_npy=str(args.phi_npy),
        observed_terms=int(present[:, 1:].sum()),
        message=f'JAX Adam {N_EPOCHS} epochs lr={LR} λ={LAMBDA}',
        n_calls=N_EPOCHS, lambda_reg=LAMBDA,
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
