"""Python constitutive model — the object the FEM solver calls at each Gauss point.

This is the ONE function ANSYS UserMat / Abaqus UMAT must invoke per integration point:

    stress, ddsdde, state = evaluate(strain, dstrain, state, props)

Conventions (match Abaqus UMAT / ANSYS USERMAT so the bridge is a 1:1 mapping):
  * 3-D, Voigt order (11, 22, 33, 12, 13, 23)
  * ENGINEERING shear strain (gamma_ij = 2*eps_ij)
  * `strain`  : total strain at start of increment, shape (6,)
  * `dstrain` : strain increment, shape (6,)
  * `state`   : solution-dependent state variables (SDV / STATEV), shape (nstatev,)
  * returns Cauchy stress (6,), consistent tangent DDSDDE (6, 6), updated state

>>> PLACEHOLDER LAW <<<
The body below is small-strain isotropic linear elasticity, used ONLY so the cross-language
plumbing and the tangent check can be exercised today. Replace `evaluate` with the real
Hamilton-principle biofilm material model from the publication — keep this signature unchanged
and the UMAT/UserMat bridge needs no edits.
"""
from __future__ import annotations
import numpy as np

NSTATEV = 1  # bump when the real model carries internal variables (e.g. growth, damage)


def elastic_tangent(E: float, nu: float) -> np.ndarray:
    """Isotropic linear-elastic DDSDDE in Abaqus Voigt order, engineering shear."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu  # engineering shear -> factor is mu, not 2*mu
    return C


def evaluate(strain, dstrain, state, props):
    """Constitutive update at one Gauss point. See module docstring for the contract.

    props = {"E": Young's modulus, "nu": Poisson ratio}  (placeholder law)
    """
    strain = np.asarray(strain, float)
    dstrain = np.asarray(dstrain, float)
    state = np.asarray(state, float) if state is not None else np.zeros(NSTATEV)

    # --- replace this block with the real material model -------------------
    C = elastic_tangent(props["E"], props["nu"])
    total = strain + dstrain
    stress = C @ total
    ddsdde = C
    state = state  # no evolution in the placeholder
    # -----------------------------------------------------------------------
    return stress, ddsdde, state
