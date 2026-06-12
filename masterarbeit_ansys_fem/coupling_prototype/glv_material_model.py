"""gLV (replicator) material model — comparison baseline against the NSP model.

Same Heine fit (A, b) as `nsp_material_model.py`, but the *replicator* gLV dynamics from the
canonical `guild_replicator_dieckow.py`:

    f = b + A @ phi ;  dphi/dt = phi * (f - phi·f)        (conserves sum(phi)=1)

A Gauss-point call here is one explicit RK4 step of this ODE. STATEV = phi (5), vs the NSP model's
12-dim state — so gLV is the cheaper, lower-state constitutive option. Useful to show the FEM
coupling is model-agnostic: swap NSP<->gLV behind the same evaluator contract.
"""
from __future__ import annotations
import json
from pathlib import Path

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

NIFE_ROOT = Path('/home/nishioka/IKM_Hiwi/nife')
GLV_FIT = NIFE_ROOT / 'results' / 'heine2025' / 'fit_glv_heine.json'

N_SP = 5


def load_Ab(cond_tag: str = "DH"):
    d = json.loads(GLV_FIT.read_text())
    A = jnp.array(d[cond_tag]['A'], dtype=jnp.float64)
    b = jnp.array(d[cond_tag]['b'], dtype=jnp.float64)
    return A, b


def replicator_rhs(phi, A, b):
    f = b + A @ phi
    return phi * (f - phi @ f)


def rk4_step(phi, A, b, dt):
    k1 = replicator_rhs(phi, A, b)
    k2 = replicator_rhs(phi + 0.5 * dt * k1, A, b)
    k3 = replicator_rhs(phi + 0.5 * dt * k2, A, b)
    k4 = replicator_rhs(phi + dt * k3, A, b)
    phi_new = phi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return jnp.clip(phi_new, 1e-10, 1.0)


def make_evaluator(cond_tag: str = "DH", dt: float = 1e-3):
    A, b = load_Ab(cond_tag)

    def step(phi):
        return rk4_step(phi, A, b, dt)

    return jax.jit(step), jax.jit(jax.jacfwd(step)), (A, b)
