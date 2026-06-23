"""Finite-strain biofilm material model: multiplicative growth split F = Fe . Fg.

Replaces the small-strain version (nsp_mechanics_model.py): the CLSM-calibrated depth growth
(eps_zz ~ 1.2-2.3, lambda_z ~ 2.2-2.4) is far outside the small-strain regime, where the additive
strain split is simply *invalid* (not objective). NOTE the confined-stress *magnitude* is similar
in both (~15-20 mu) — you genuinely cannot confine ~140% growth cheaply, in either formulation —
but only the multiplicative split is kinematically correct (proper J=det(Fe), frame-indifferent)
and remains well-posed as a constitutive law inside the FEM Newton loop.

Kinematics:
  * total deformation gradient F (from the FEM)
  * growth part Fg = diag(1, 1, lambda_z),  lambda_z = 1 + max(0, beta_depth*phi_Pg + intercept)
    (substratum-normal growth; lateral FOV fixed — see calibrate_beta.py)
  * elastic part Fe = F . Fg^{-1}  carries the stress
Constitutive law on Fe: compressible neo-Hookean (Bonet & Wood),
    sigma = (mu/J)(b - I) + (lambda/J) ln(J) I ,   b = Fe Fe^T,  J = det(Fe)
Stress scales with mu, so results are reported as sigma/mu (modulus-agnostic; biofilm mu ~ 10-1e3 Pa).

Two boundary cases at a single Gauss point bracket the physics:
  * confined  (F = I): growth fully restrained -> large but FINITE (bounded) compressive stress,
    unlike the diverging small-strain estimate.
  * free-top  (substrate-bonded laterally, traction-free top): solve F_zz so sigma_zz = 0. With
    z-only growth this is nearly stress-free -> explains how biofilms thicken without failing.
    Real residual stress needs spatial growth gradients -> the FEM job, not a single point.
"""
from __future__ import annotations
import numpy as np

import nsp_material_model as nm
from nsp_mechanics_model import _calib, PG_INDEX, NSTATEV, _nsp_step_for

I3 = np.eye(3)


def lame(E, nu):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lam


def cauchy_neo_hookean(Fe, E, nu):
    mu, lam = lame(E, nu)
    J = np.linalg.det(Fe)
    b = Fe @ Fe.T
    return (mu / J) * (b - I3) + (lam / J) * np.log(J) * I3


def growth_gradient(phi_pg, cond):
    beta_d, intcpt = _calib(cond)
    lam_z = 1.0 + max(0.0, beta_d * phi_pg + intcpt)
    Fg = np.diag([1.0, 1.0, lam_z])
    return Fg, lam_z


def stress_from_F(F, phi_pg, props):
    """Cauchy stress (3x3) for total F given current Pg fraction."""
    Fg, lam_z = growth_gradient(phi_pg, props.get("cond", "DH"))
    Fe = F @ np.linalg.inv(Fg)
    return cauchy_neo_hookean(Fe, props["E"], props["nu"]), lam_z


def relax_free_top(phi_pg, props, tol=1e-10):
    """Substrate-bonded laterally (F_xx=F_yy=1), traction-free top: solve F_zz s.t. sigma_zz=0."""
    Fzz = 1.0
    for _ in range(50):
        F = np.diag([1.0, 1.0, Fzz])
        s, _ = stress_from_F(F, phi_pg, props)
        # numerical derivative d sigma_zz / d Fzz
        h = 1e-7
        F2 = np.diag([1.0, 1.0, Fzz + h])
        s2, _ = stress_from_F(F2, phi_pg, props)
        dsig = (s2[2, 2] - s[2, 2]) / h
        step = -s[2, 2] / dsig
        Fzz += step
        if abs(step) < tol:
            break
    F = np.diag([1.0, 1.0, Fzz])
    s, lam_z = stress_from_F(F, phi_pg, props)
    return s, Fzz, lam_z


def spatial_tangent_neo_hookean(Fe, E, nu):
    """Consistent spatial (Eulerian) tangent for compressible neo-Hookean.

    Spatial elasticity tensor (Bonet & Wood §6.5, push-forward of material tangent):
      c_ijkl = (lam/J)*δ_ij*δ_kl + ((mu - lam*ln(J))/J)*(δ_ik*δ_jl + δ_il*δ_jk)

    Returned as DDSDDE (6×6) in Abaqus Voigt order (11,22,33,12,13,23), engineering shear.
    At J→1: reduces exactly to elastic_tangent(E, nu) (linear isotropic limit). ✓
    """
    mu_e, lam_e = lame(E, nu)
    J = np.linalg.det(Fe)
    c_lam = lam_e / J
    c_mu  = (mu_e - lam_e * np.log(J)) / J   # effective shear modulus; = mu at J=1
    _v = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    C = np.zeros((6, 6))
    for I, (i, j) in enumerate(_v):
        for Jj, (k, l) in enumerate(_v):
            C[I, Jj] = (c_lam * (i == j) * (k == l)
                        + c_mu * ((i == k) * (j == l) + (i == l) * (j == k)))
    return C


def stress_and_tangent_from_F(F, phi_pg, props):
    """Cauchy stress + consistent spatial tangent for total F and current Pg fraction."""
    Fg, lam_z = growth_gradient(phi_pg, props.get("cond", "DH"))
    Fe = F @ np.linalg.inv(Fg)
    sigma   = cauchy_neo_hookean(Fe, props["E"], props["nu"])
    ddsdde  = spatial_tangent_neo_hookean(Fe, props["E"], props["nu"])
    return sigma, ddsdde, lam_z


def advance(statev, props):
    """One Gauss-point step: advance NSP composition, return updated state + phi_Pg."""
    statev = np.asarray(statev, float)
    if statev.size != NSTATEV or not np.any(statev):
        statev = np.asarray(nm.make_initial_state(props.get("phi_init", np.full(5, 0.2))), float)
    g_new = np.asarray(_nsp_step_for(props.get("cond", "DH"))(statev), float)
    return g_new, g_new[PG_INDEX]
