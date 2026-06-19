"""Two-way mechano-biology coupling: NSP composition ↔ growth eigenstrain ↔ stress feedback.

Extends nsp_mechanics_model (one-way: φ_Pg → growth → stress) with a stress-feedback channel
inspired by the C2 Python prototype (extensions/c2_twoway_coupling.py):

    K(σ) = 1 / (1 + (|σ_eq| / σ_c)^2)            (compressive vM stress lowers carrying capacity)
    growth_eigenstrain = β * φ_Pg * K(σ_prev)    (effective growth rate is throttled where σ is high)

σ_prev is supplied by the Abaqus UMAT from the *previous converged increment* via an extended wire
protocol (umat_socket_twoway.f / umat_server_twoway.py). This makes the loop closed:

    σ(t)  →  K(σ)  →  effective growth  →  ε_growth(t+Δt)  →  σ(t+Δt)

The composition (NSP) tangent is unchanged; only the mechanical effective driver is modulated. The
mechanical tangent stays DDSDDE = C (the eigenstrain is still independent of the current mechanical
strain after the within-step lookup).

Contract (same shape as nsp_mechanics_model.evaluate, plus σ_prev):

    stress(6), ddsdde(6,6), statev_new = evaluate_twoway(strain, dstrain, statev, props, sigma_prev)

props extends {"E","nu","beta","cond","phi_init"} with
    "sigma_c"      ─ carrying-capacity scale [stress units]
    "feedback"     ─ True/False switch; False reproduces the one-way headline byte-for-byte
"""
from __future__ import annotations

import numpy as np

import nsp_material_model as nm
from material_model import elastic_tangent

PG_INDEX = 4
I_VOL = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def von_mises(sig6: np.ndarray) -> float:
    """vM stress from a Voigt 6-vector (σ11,σ22,σ33,σ12,σ13,σ23)."""
    s = np.asarray(sig6, dtype=float)
    return float(np.sqrt(0.5 * ((s[0] - s[1])**2 + (s[1] - s[2])**2 + (s[2] - s[0])**2)
                         + 3.0 * (s[3]**2 + s[4]**2 + s[5]**2)))


def carrying_capacity(sigma_prev: np.ndarray, sigma_c: float) -> float:
    """K(σ) ∈ (0, 1] — 1 at zero stress, drops as |σ_vM|/σ_c grows."""
    if sigma_c <= 0:
        return 1.0
    return 1.0 / (1.0 + (von_mises(sigma_prev) / sigma_c) ** 2)


def evaluate_twoway(strain: np.ndarray, dstrain: np.ndarray, statev: np.ndarray,
                    props: dict, sigma_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-way NSP-mechanics constitutive evaluation; falls back to the one-way path if
    props["feedback"] is False or σ_prev is all zeros (first increment)."""
    E = float(props["E"]); nu = float(props["nu"]); beta = float(props["beta"])
    feedback = bool(props.get("feedback", True))
    sigma_c = float(props.get("sigma_c", 1.0))

    # advance the NSP state by one reaction step (unchanged from the one-way model)
    statev = np.asarray(statev, dtype=float).copy()
    if not np.any(statev):
        statev = nm.init_state(props.get("cond", "DH"), props.get("phi_init", None))
    statev = nm.newton_step(statev, props.get("cond", "DH"))

    phi_pg = statev[PG_INDEX]

    # stress-feedback factor K(σ_prev)
    if feedback and np.any(sigma_prev):
        K = carrying_capacity(sigma_prev, sigma_c)
    else:
        K = 1.0

    eps_growth = beta * phi_pg * K * I_VOL
    eps_mech = np.asarray(strain, dtype=float) + np.asarray(dstrain, dtype=float) - eps_growth
    C = elastic_tangent(E, nu)
    stress = C @ eps_mech
    return stress, C, statev
