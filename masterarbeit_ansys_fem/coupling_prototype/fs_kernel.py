"""Pure-numpy finite-strain kernel (NO JAX) — used by the tabulated solve-time server.

Same neo-Hookean F=Fe.Fg math as nsp_mechanics_fs.py, but with zero JAX import so the constitutive
server stays lightweight (the surrogate promotion path). Growth coefficients come from the CLSM
calibration JSON; the NSP composition enters only as a precomputed phi_Pg.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

I3 = np.eye(3)
_CALIB_FILE = Path(__file__).resolve().parent / 'beta_calibration.json'


def calib(cond: str):
    try:
        d = json.loads(_CALIB_FILE.read_text())[cond]
        return d["beta_depth"], d["intercept"]
    except Exception:
        return 17.0, -1.34


def lame(E, nu):
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def cauchy_neo_hookean(Fe, E, nu):
    mu, lam = lame(E, nu)
    J = np.linalg.det(Fe)
    b = Fe @ Fe.T
    return (mu / J) * (b - I3) + (lam / J) * np.log(J) * I3


def material_jacobian_nh(Fe, E, nu):
    """Consistent Jaumann-rate material Jacobian DDSDDE (6x6, Abaqus Voigt, engineering shear)
    for sigma = (mu/J)(b-I)+(lam/J)lnJ I. Effective moduli: lam'=lam/J, mu'=(mu-lam lnJ)/J."""
    mu, lam = lame(E, nu)
    J = np.linalg.det(Fe)
    lam_e = lam / J
    mu_e = (mu - lam * np.log(J)) / J
    C = np.zeros((6, 6))
    C[:3, :3] = lam_e
    C[0, 0] = C[1, 1] = C[2, 2] = lam_e + 2.0 * mu_e
    C[3, 3] = C[4, 4] = C[5, 5] = mu_e
    return C


def elastic_part(F, Fg):
    return F @ np.linalg.inv(Fg)


def growth_gradient(phi_pg, cond):
    bd, ic = calib(cond)
    lam_z = 1.0 + max(0.0, bd * phi_pg + ic)
    return np.diag([1.0, 1.0, lam_z]), lam_z


def stress_from_F(F, phi_pg, E, nu, cond):
    Fg, lam_z = growth_gradient(phi_pg, cond)
    Fe = F @ np.linalg.inv(Fg)
    return cauchy_neo_hookean(Fe, E, nu), lam_z


# --- isotropic volumetric growth (for the laterally-confined column / residual stress) ---
def growth_gradient_iso(theta):
    """Isotropic volumetric growth: Fg = theta^(1/3) I, theta = volume growth factor (>=1)."""
    s = theta ** (1.0 / 3.0)
    return np.diag([s, s, s]), s


def stress_from_F_iso(F, theta, E, nu):
    Fg, s = growth_gradient_iso(theta)
    Fe = F @ np.linalg.inv(Fg)
    return cauchy_neo_hookean(Fe, E, nu), s


def growth_gradient_depth(phi_pg, gamma, intercept=0.0):
    """Anisotropic substratum-normal growth: Fg = diag(1,1,lambda_z), lambda_z = 1+max(0,gamma*phi+ic)."""
    lam_z = 1.0 + max(0.0, gamma * phi_pg + intercept)
    return np.diag([1.0, 1.0, lam_z]), lam_z


def stress_and_tangent(F, Fg, E, nu):
    """Cauchy stress (3x3) + consistent Jacobian (6x6) for a given total F and growth Fg."""
    Fe = F @ np.linalg.inv(Fg)
    return cauchy_neo_hookean(Fe, E, nu), material_jacobian_nh(Fe, E, nu)
