#!/usr/bin/env python3
"""
hamilton_steadystate_jax.py — JAX-differentiable steady-state solver for Hamilton replicator.

Instead of integrating the ODE for nsteps, find the fixed point φ* where φ̇=0:
    φ̇_i = φ_i (b_i + Σ_j A_ij φ_j − f̄) = 0,  Σ_i φ_i = 1

Two solvers:
  1. damped_iteration(A, b, phi0, n_iter=200): damped replicator iterations (differentiable)
  2. newton_ss(A, b, phi0, n_iter=30): Newton on the interior simplex equations (differentiable)

Both are JAX-jittable and support jax.grad through them.
"""
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


# ── Replicator fitness ────────────────────────────────────────────────────────
def _fitness(phi, A, b):
    """Per-guild fitness f_i = b_i + Σ_j A_ij φ_j."""
    return b + A @ phi


def _mean_fitness(phi, A, b):
    return jnp.dot(phi, _fitness(phi, A, b))


# ── Solver 1: Damped replicator iteration ─────────────────────────────────────
def damped_iteration(A, b, phi0, n_iter=300, dt=0.02):
    """
    Damped discrete replicator map:
        φ_i^{t+1} ∝ φ_i^t · exp(dt · f_i(φ^t))
    Converges to the interior ESS when one exists.
    Fully JAX-differentiable via lax.fori_loop → lax.scan.
    """
    def step(phi, _):
        f    = _fitness(phi, A, b)
        phi2 = phi * jnp.exp(dt * f)
        return phi2 / phi2.sum(), None

    phi_final, _ = jax.lax.scan(step, phi0, None, length=n_iter)
    return phi_final


# ── Solver 2: Newton on simplex interior ─────────────────────────────────────
def newton_ss(A, b, phi0, n_iter=40, tol=1e-10, lam=1e-6):
    """
    Newton's method on the system:
        g(φ) = 0,  g_i = φ_i (f_i - f̄),  i < n_sp
        φ_{n_sp-1} = 1 - Σ_{i<n_sp-1} φ_i   (simplex constraint)

    Returns φ* in the probability simplex.
    Not perfectly differentiable through convergence, but good enough for
    warm-started gradient descent (outer loop uses damped_iteration).
    """
    n = phi0.shape[0]

    def body(carry, _):
        phi, converged = carry
        # Enforce simplex
        phi = jnp.clip(phi, 1e-12, None)
        phi = phi / phi.sum()

        f    = _fitness(phi, A, b)
        fbar = jnp.dot(phi, f)
        g    = phi * (f - fbar)           # residual

        # Jacobian of g w.r.t. φ (n×n)
        J = jax.jacobian(lambda p: p * (_fitness(p, A, b) - jnp.dot(p, _fitness(p, A, b))))(phi)
        # Tikhonov regularization for stability
        J_reg = J + lam * jnp.eye(n)

        delta, _ = jnp.linalg.lstsq(J_reg, -g, rcond=None)
        phi_new  = phi + delta
        phi_new  = jnp.clip(phi_new, 1e-12, None)
        phi_new  = phi_new / phi_new.sum()

        new_conv = converged | (jnp.max(jnp.abs(g)) < tol)
        phi_out  = jnp.where(new_conv, phi, phi_new)
        return (phi_out, new_conv), None

    (phi_final, _), _ = jax.lax.scan(body, (phi0, False), None, length=n_iter)
    return phi_final


# ── Recommended solver for fitting: damped + refine ──────────────────────────
def predict_ss(A, b, phi0, n_damp=400, dt=0.02):
    """
    Predict steady state from initial condition phi0.
    Uses damped iteration only (fully differentiable for L-BFGS-B).
    Returns φ* ∈ Δ.
    """
    return damped_iteration(A, b, phi0, n_iter=n_damp, dt=dt)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import numpy as np
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from guild_replicator_dieckow import N_G, GUILD_SHORT_LIST

    rng = np.random.default_rng(0)
    A_np = rng.normal(0, 0.3, (N_G, N_G))
    np.fill_diagonal(A_np, -0.5)
    b_np = rng.normal(0.1, 0.05, N_G)

    A = jnp.array(A_np)
    b = jnp.array(b_np)
    phi0 = jnp.ones(N_G) / N_G

    print('Testing damped_iteration...', flush=True)
    ss_damp = jax.jit(damped_iteration)(A, b, phi0)
    print(f'  φ* (damp): {np.array(ss_damp).round(3)}')

    # Check residual
    f    = b + A @ ss_damp
    fbar = jnp.dot(ss_damp, f)
    resid = ss_damp * (f - fbar)
    print(f'  max|g| = {float(jnp.max(jnp.abs(resid))):.2e}')

    print('Testing gradient (auto-diff through damped_iteration)...', flush=True)
    loss = lambda A, b: jnp.sum((damped_iteration(A, b, phi0) - 0.1) ** 2)
    gA, gb = jax.jit(jax.grad(loss, argnums=(0, 1)))(A, b)
    print(f'  grad norm A={float(jnp.linalg.norm(gA)):.4f}  b={float(jnp.linalg.norm(gb)):.4f}')
    print('OK')
