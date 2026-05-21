"""
hamilton_ode_jax_nsp_ift.py — NSP ODE with Implicit Function Theorem (IFT) gradient.

Instead of differentiating through the Newton iterations (slow to compile due to
nested lax.scan + jacfwd), use the Implicit Function Theorem:

  At the Newton solution g* where G(g*, g_prev, θ) = 0:
    dg*/dθ = -(∂G/∂g*)^{-1} × (∂G/∂θ)

The custom_vjp backward involves only:
  1. J = ∂G/∂g*  (one jacfwd on residual, 22 tangents)
  2. Solve J^T v = g_bar  (22×22 linear system)
  3. ∂G/∂g_prev  (one jacfwd, 22 tangents)
  4. ∂G/∂A_upper (one jacfwd, n_A = n_sp*(n_sp+1)//2 tangents)
  5. ∂G/∂b       (one jacfwd, n_sp tangents)

newton_ift takes A_upper (upper triangle) directly to avoid double-counting
that occurs when chain-ruling through _upper_to_full.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hamilton_ode_jax_nsp import _residual, _newton_step, _make_initial_state


def _upper_to_full(A_upper, n_sp):
    """Reconstruct symmetric A matrix from upper triangle (column-major)."""
    A = jnp.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A = A.at[i, j].set(A_upper[idx])
            A = A.at[j, i].set(A_upper[idx])
            idx += 1
    return A


def make_ift_simulate(n_sp, n_steps, dt, c_const=25.0, alpha_const=100.0):
    """
    Create a differentiable NSP simulation function using IFT backward.

    Returns:
        simulate_single(A_upper, b_diag, phi_init) -> (n_sp,) equilibrium composition
        Differentiable w.r.t. A_upper and b_diag via IFT custom_vjp on Newton step.

    Note: newton_ift takes A_upper directly (not the full A matrix) so that
    the gradient w.r.t. A_upper is computed correctly inside _bwd without
    going through _upper_to_full's chain rule.
    """
    active_mask = jnp.ones(n_sp, dtype=jnp.int64)
    n_A = n_sp * (n_sp + 1) // 2

    # Fixed parameters captured in Python closure (not JAX-traced)
    _fixed = {
        "n_sp": n_sp,
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(n_sp, dtype=jnp.float64),
        "EtaPhi": jnp.ones(n_sp, dtype=jnp.float64),
        "c": c_const,
        "alpha": alpha_const,
        "K_hill": jnp.array(0.0, dtype=jnp.float64),
        "n_hill": jnp.array(4.0, dtype=jnp.float64),
        "hill_gate_species": jnp.array((-1, 0), dtype=jnp.int32),
        "active_mask": active_mask,
    }

    @jax.custom_vjp
    def newton_ift(g_prev, A_upper, b_diag):
        """Newton step with IFT backward. Takes A_upper (upper triangle)."""
        A = _upper_to_full(A_upper, n_sp)
        params = {**_fixed, "A": A, "b_diag": b_diag}
        return _newton_step(g_prev, params)

    def _fwd(g_prev, A_upper, b_diag):
        A = _upper_to_full(A_upper, n_sp)
        params = {**_fixed, "A": A, "b_diag": b_diag}
        g_new = _newton_step(g_prev, params)
        return g_new, (g_prev, g_new, A, A_upper, b_diag)

    def _bwd(res, g_bar):
        g_prev, g_new, A, A_upper, b_diag = res
        params = {**_fixed, "A": A, "b_diag": b_diag}

        # ── IFT backward ─────────────────────────────────────────────────────
        # J = ∂G/∂g_new (Newton Jacobian at converged solution)
        J = jax.jacfwd(lambda g: _residual(g, g_prev, params))(g_new)

        # Adjoint: solve J^T v = g_bar
        v = jnp.linalg.solve(J.T, g_bar)

        # IFT: x_bar = -(∂G/∂x)^T v  for each input x.
        # Note the negative sign: v solves J^T v = g_bar where J = ∂G/∂g_new,
        # but the cotangent for other inputs has an additional minus sign.

        # Cotangent for g_prev: -(∂G/∂g_prev)^T @ v
        J_gp = jax.jacfwd(lambda gp: _residual(g_new, gp, params))(g_prev)
        g_prev_bar = -(J_gp.T @ v)

        # Cotangent for A_upper: -(∂G/∂A_upper)^T @ v
        # J_Au shape: (n_state, n_A)
        J_Au = jax.jacfwd(
            lambda au: _residual(
                g_new, g_prev, {**params, "A": _upper_to_full(au, n_sp)}
            )
        )(A_upper)
        A_upper_bar = -(J_Au.T @ v)   # (n_A,)

        # Cotangent for b_diag: -(∂G/∂b_diag)^T @ v
        J_b = jax.jacfwd(
            lambda b: _residual(g_new, g_prev, {**params, "b_diag": b})
        )(b_diag)
        b_bar = -(J_b.T @ v)

        return g_prev_bar, A_upper_bar, b_bar

    newton_ift.defvjp(_fwd, _bwd)

    def simulate_single(A_upper, b_diag, phi_init):
        """
        Run NSP ODE for n_steps and return final normalized composition.

        Args:
            A_upper: (n_A,) upper triangle of symmetric A matrix
            b_diag:  (n_sp,) growth rates
            phi_init: (n_sp,) initial composition (will be normalized)
        Returns:
            (n_sp,) predicted equilibrium composition
        """
        g0 = _make_initial_state(phi_init, n_sp, active_mask, psi_init=0.999)

        def body(g, _):
            return newton_ift(g, A_upper, b_diag), None

        g_T, _ = jax.lax.scan(body, g0, None, length=n_steps)

        phi = g_T[0:n_sp]
        psi = g_T[n_sp + 1 : 2 * n_sp + 1]
        eq = phi * psi
        s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    return simulate_single
