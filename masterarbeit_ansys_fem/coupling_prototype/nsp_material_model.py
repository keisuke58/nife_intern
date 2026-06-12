"""Real material model — the Heine 2025 NSP (Hamilton) constitutive response.

This is the actual model to invoke at each Gauss point, replacing the placeholder in
`material_model.py`. It is the SAME per-node reaction that `nsp_pde_1d_heine.py` vmaps over its
z-nodes — i.e. the spatial PDE already does "invoke the NSP model at each node", which is exactly
"invoke the material model at each Gauss point" in FEM terms. The FEM job generalises that
node-loop from a 1-D method-of-lines grid to 3-D element Gauss points.

State / contract (continuum-biofilm framing):
  * internal state g (the STATEV / SDV vector), dim 2N+2 = 12 for N=5 species:
        g = [phi_1..phi_5, phi_0, psi_1..psi_5, gamma]
    phi = relative abundance (composition), psi = per-species "live/activity" field,
    phi_0 = free fraction, gamma = Lagrange multiplier enforcing sum(phi)+phi_0 = 1.
  * evaluate_state(g_prev, params) -> (g_new, J)
        g_new = one implicit-Euler NSP step (6 inner Newton iters)
        J     = d g_new / d g_prev  (the consistent tangent the OUTER FEM Newton needs)

NOTE on physical role: NSP returns a *composition / reaction* update, not a mechanical stress.
Embedding it at a Gauss point supplies a source / internal-variable evolution; the genuine
modelling step for the thesis is the phi -> stress (or phi -> growth/swelling) coupling that closes
the mechanics. This module verifies the reaction map and its tangent are well-posed for the
implicit FEM solve — the prerequisite for either coupling.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# NSP core (canonical, from the TMCMC 5-species project)
NSP_PATH = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
sys.path.insert(0, str(NSP_PATH))
from hamilton_ode_jax_nsp import _newton_step, _make_initial_state  # noqa: E402

NIFE_ROOT = Path('/home/nishioka/IKM_Hiwi/nife')
GLV_FIT = NIFE_ROOT / 'results' / 'heine2025' / 'fit_glv_heine.json'

N_SP = 5
N_STATE = 2 * N_SP + 2  # 12

# Same fixed NSP constants the 0-D LOO fits / the 1-D PDE use.
NSP_PARAMS_BASE = {
    "n_sp": N_SP,
    "dt_h": 1e-3,
    "Kp1": 1e-4,
    "Eta": jnp.ones(N_SP, dtype=jnp.float64),
    "EtaPhi": jnp.ones(N_SP, dtype=jnp.float64),
    "c": 25.0,
    "alpha": 100.0,
    "K_hill": jnp.array(0.0, dtype=jnp.float64),
    "n_hill": jnp.array(4.0, dtype=jnp.float64),
    "hill_gate_species": jnp.array([-1, 0], dtype=jnp.int32),
    "active_mask": jnp.ones(N_SP, dtype=jnp.int64),
}


def load_params(cond_tag: str = "DH") -> dict:
    """Params dict (incl. fitted A, b) for a Heine condition: CS/CH/DS/DH."""
    d = json.loads(GLV_FIT.read_text())
    A = jnp.array(d[cond_tag]['A'], dtype=jnp.float64)
    b = jnp.array(d[cond_tag]['b'], dtype=jnp.float64)
    return {**NSP_PARAMS_BASE, "A": A, "b_diag": b}


def make_initial_state(phi, psi_init: float = 0.999):
    """Build a 12-dim NSP state from a 5-vector composition phi."""
    return _make_initial_state(jnp.asarray(phi, dtype=jnp.float64),
                               N_SP, NSP_PARAMS_BASE["active_mask"], psi_init=psi_init)


def make_evaluator(params: dict):
    """Return (step, tangent) closures: g_prev -> g_new and g_prev -> d g_new/d g_prev."""
    def step(g_prev):
        return _newton_step(g_prev, params)
    step_jit = jax.jit(step)
    tangent_jit = jax.jit(jax.jacfwd(step))
    return step_jit, tangent_jit
