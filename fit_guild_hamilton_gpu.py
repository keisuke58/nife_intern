#!/usr/bin/env python3
"""
Class-level Hamilton ODE fit — GPU + JAX vmap + Adam.

Runs simulate_0d_nsp(n_sp=N) for all patients in parallel via vmap.
JAX autodiff (grad) through the Hamilton ODE → Adam optimiser.

Usage (on vancouver01):
  CUDA_VISIBLE_DEVICES=1 python3 fit_guild_hamilton_gpu.py

Output: results/dieckow_cr/fit_guild_hamilton.json
"""

import json, sys, time, argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
jax.config.update('jax_enable_x64', True)

from guild_replicator_dieckow import GUILD_ORDER

_repo_root = Path(__file__).resolve().parents[1]
_hamilton_path = _repo_root / 'Tmcmc202601' / 'data_5species' / 'main'
sys.path.insert(0, str(_hamilton_path))
from hamilton_ode_jax_nsp import simulate_0d_nsp

def build_fns(n_sp, n_steps, lambda_reg):
    @jit
    def eq_phi_one(A_upper, b_p, phi_init):
        theta  = jnp.concatenate([A_upper, b_p])
        s0 = jnp.sum(phi_init)
        phi0 = jnp.where(s0 > 1e-12, phi_init / s0, jnp.ones(n_sp) / n_sp)
        phibar = simulate_0d_nsp(
            theta, n_sp=n_sp, n_steps=n_steps, dt=1e-4, phi_init=phi0,
            c_const=25.0, alpha_const=100.0
        )
        eq = phibar[-1]
        s  = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    @jit
    def loss_fn(A_upper, b_all, phi_obs, present_mask):
        def patient_terms(b_p, phi_p, m_p):
            phi_W2 = eq_phi_one(A_upper, b_p, phi_p[0])
            phi_W3 = eq_phi_one(A_upper, b_p, phi_W2)
            m2 = m_p[1]
            m3 = m_p[2]
            sq = m2 * jnp.sum((phi_W2 - phi_p[1]) ** 2) + m3 * jnp.sum((phi_W3 - phi_p[2]) ** 2)
            cnt = (m2 + m3) * n_sp
            return sq, cnt

        sq_all, cnt_all = vmap(patient_terms)(b_all, phi_obs, present_mask)
        sq = jnp.sum(sq_all)
        cnt = jnp.sum(cnt_all)
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        reg = lambda_reg * jnp.sum(A_upper ** 2)
        return rmse + reg

    grad_fn = jit(grad(loss_fn, argnums=(0, 1)))

    def rmse_pure(A_upper, b_all, phi_obs, present_mask):
        def patient_terms(b_p, phi_p, m_p):
            phi_W2 = eq_phi_one(A_upper, b_p, phi_p[0])
            phi_W3 = eq_phi_one(A_upper, b_p, phi_W2)
            m2 = m_p[1]
            m3 = m_p[2]
            sq = m2 * jnp.sum((phi_W2 - phi_p[1]) ** 2) + m3 * jnp.sum((phi_W3 - phi_p[2]) ** 2)
            cnt = (m2 + m3) * n_sp
            return sq, cnt

        sq_all, cnt_all = vmap(patient_terms)(b_all, phi_obs, present_mask)
        sq = float(jnp.sum(sq_all))
        cnt = float(jnp.sum(cnt_all))
        return float(np.sqrt(sq / cnt)) if cnt > 0 else float('nan')

    return eq_phi_one, loss_fn, grad_fn, rmse_pure


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
    n_upper = int(A_upper.shape[0])
    n_sp = int((np.sqrt(8 * n_upper + 1) - 1) / 2)
    for j in range(n_sp):
        for i in range(j + 1):
            if i == j:
                diag_idx.append(idx)
            idx += 1
    for k in diag_idx:
        A_upper = A_upper.at[k].set(jnp.minimum(A_upper[k], 0.0))
    return A_upper


def default_A_upper(n_sp):
    A = -0.1 * np.eye(n_sp)
    upper = []
    for j in range(n_sp):
        for i in range(j + 1):
            upper.append(A[i, j])
    return jnp.array(upper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy', default=str(Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'))
    ap.add_argument('--out-json', default=str(Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_gpu.json'))
    ap.add_argument('--warm-start-json', default=str(Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild.json'))
    ap.add_argument('--n-steps', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lambda-reg', type=float, default=1e-3)
    ap.add_argument('--log-every', type=int, default=100)
    args = ap.parse_args()

    print(f'JAX devices: {jax.devices()}', flush=True)

    phi_all = np.load(args.phi_npy)
    if phi_all.ndim != 3 or phi_all.shape[1] != 3:
        raise ValueError(f'Expected phi shape (n_patients,3,n_guilds), got {phi_all.shape}')
    n_p, _, n_sp = phi_all.shape
    guilds = GUILD_ORDER[:n_sp]

    present = (phi_all.sum(axis=2) > 1e-12).astype(np.float64)
    keep = present[:, 0] > 0.0
    phi_all = phi_all[keep, :, :]
    present = present[keep, :]
    patients = [p for k, p in zip(keep.tolist(), list('ABCDEFGHKL')) if k] if n_p == 10 else [str(i) for i in range(phi_all.shape[0])]

    phi_obs = jnp.array(phi_all)
    present_mask = jnp.array(present)
    print(f'phi_obs: {phi_obs.shape} (kept {phi_obs.shape[0]} / {n_p} patients)', flush=True)

    eq_phi_one, loss_fn, grad_fn, rmse_pure = build_fns(n_sp=n_sp, n_steps=args.n_steps, lambda_reg=args.lambda_reg)

    warm = Path(args.warm_start_json)
    if warm.exists():
        d = json.load(open(warm))
        A0 = np.array(d['A'])
        if A0.shape[0] != n_sp:
            A0 = A0[:n_sp, :n_sp]
        A_sym = (A0 + A0.T) / 2.0
        A_upper = []
        for j in range(n_sp):
            for i in range(j + 1):
                A_upper.append(A_sym[i, j])
        A_upper = jnp.array(A_upper)
        b0 = np.array(d['b_all'])
        b0 = b0[: n_p, : n_sp]
        b0 = b0[keep, :]
        b_all = jnp.array(b0)
        print('Warm start: gLV → symmetrised A', flush=True)
    else:
        A_upper = default_A_upper(n_sp)
        b_all   = jnp.full((phi_obs.shape[0], n_sp), 0.1)

    m = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))
    v = (jnp.zeros_like(A_upper), jnp.zeros_like(b_all))

    print('\nWarm-up JIT compile...', flush=True)
    t0 = time.time()
    _ = loss_fn(A_upper, b_all, phi_obs, present_mask)
    print(f'First forward done in {time.time()-t0:.1f}s', flush=True)

    print('Compiling grad...', flush=True)
    _ = grad_fn(A_upper, b_all, phi_obs, present_mask)
    print(f'Grad compiled in {time.time()-t0:.1f}s', flush=True)

    best_loss = float(loss_fn(A_upper, b_all, phi_obs, present_mask))
    best_A, best_b = A_upper, b_all

    for epoch in range(1, args.epochs + 1):
        gA, gb = grad_fn(A_upper, b_all, phi_obs, present_mask)
        (A_upper, b_all), m, v = adam_step(
            (A_upper, b_all), (gA, gb), m, v, epoch, lr=args.lr
        )
        A_upper = apply_diag_constraint(A_upper)

        if epoch % args.log_every == 0 or epoch == 1:
            val = float(loss_fn(A_upper, b_all, phi_obs, present_mask))
            print(f'  epoch {epoch:5d}  loss={val:.5f}  ({time.time()-t0:.1f}s)', flush=True)
            if val < best_loss:
                best_loss = val
                best_A, best_b = A_upper, b_all

    A_upper, b_all = best_A, best_b
    rmse = rmse_pure(A_upper, b_all, phi_obs, present_mask)
    print(f'\nFinal RMSE: {rmse:.5f}  ({time.time()-t0:.1f}s)', flush=True)

    A_full = np.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            v_ = float(A_upper[idx])
            A_full[i, j] = v_
            A_full[j, i] = v_
            idx += 1

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(
        A=A_full.tolist(),
        A_upper=np.array(A_upper).tolist(),
        b_all=np.array(b_all).tolist(),
        rmse=rmse,
        guilds=guilds,
        patients=patients,
        n_steps=args.n_steps,
        lambda_reg=args.lambda_reg,
        message=f'Hamilton ODE GPU vmap Adam n_steps={args.n_steps} lr={args.lr}',
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
