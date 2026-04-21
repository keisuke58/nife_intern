#!/usr/bin/env python3
"""
Consumer-resource ODE model for Dieckow 5-species implant biofilm.

Metabolites (quasi-steady-state):
  M_LA   : lactic acid   — produced by So, An; consumed by Vd
  M_MK   : menaquinone   — produced by An, Vd; consumed by Pg
  M_H2O2 : H2O2          — produced by So; inhibits Pg (IS_INHIBITED_BY)

QSS:
  M_LA*   = (p_LA_So*φ_So + p_LA_An*φ_An) / (c_LA_Vd*φ_Vd + δ_LA)
  M_MK*   = (p_MK_An*φ_An + p_MK_Vd*φ_Vd) / (c_MK_Pg*φ_Pg + δ_MK)
  M_H2O2* = p_H2O2*φ_So / δ_H2O2

Effective fitness:
  f_So  = b_So
  f_An  = b_An
  f_Vd  = b_Vd + α_LA  * M_LA*
  f_Fn  = b_Fn
  f_Pg  = b_Pg + α_MK  * M_MK* - α_H2O2 * M_H2O2*

Replicator:
  dφ_i/dt = φ_i*(f_i - <f>)

Shared params (13):
  [p_LA_So, p_LA_An, c_LA_Vd, log_δ_LA,
   p_MK_An, p_MK_Vd, c_MK_Pg, log_δ_MK,
   p_H2O2,  log_δ_H2O2,
   α_LA, α_MK, α_H2O2]

Patient-specific params (5 per patient):
  [b_So, b_An, b_Vd, b_Fn, b_Pg]

Source: Dieckow Supplementary File 1 (microbe-metabolite interactions)
"""

import numpy as np
from scipy.integrate import odeint

N_SP   = 5          # So, An, Vd, Fn, Pg
N_MET  = 3          # LA, MK, H2O2
N_CR   = 13         # shared CR params
N_B    = N_SP       # patient-specific growth rates
IDX_SO, IDX_AN, IDX_VD, IDX_FN, IDX_PG = 0, 1, 2, 3, 4

GENERA = ['Streptococcus', 'Actinomyces', 'Veillonella', 'Fusobacterium', 'Porphyromonas']
LABELS = ['So', 'An', 'Vd', 'Fn', 'Pg']

# ----- parameter unpacking -----

def unpack_cr(theta_cr):
    """Unpack 13 shared CR params."""
    (p_LA_So, p_LA_An, c_LA_Vd, log_dLA,
     p_MK_An, p_MK_Vd, c_MK_Pg, log_dMK,
     p_H2O2,  log_dH2O2,
     a_LA, a_MK, a_H2O2) = theta_cr
    return dict(
        p_LA_So=p_LA_So, p_LA_An=p_LA_An,
        c_LA_Vd=c_LA_Vd, dLA=np.exp(log_dLA),
        p_MK_An=p_MK_An, p_MK_Vd=p_MK_Vd,
        c_MK_Pg=c_MK_Pg, dMK=np.exp(log_dMK),
        p_H2O2=p_H2O2,   dH2O2=np.exp(log_dH2O2),
        a_LA=a_LA, a_MK=a_MK, a_H2O2=a_H2O2,
    )

def qss_metabolites(phi, p):
    """QSS concentrations for LA, MK, H2O2 (all ≥ 0)."""
    phi_So, phi_An, phi_Vd, phi_Fn, phi_Pg = phi
    denom_LA = max(p['c_LA_Vd'] * phi_Vd + p['dLA'], 1e-9)
    denom_MK = max(p['c_MK_Pg'] * phi_Pg + p['dMK'], 1e-9)
    M_LA   = max((p['p_LA_So']*phi_So + p['p_LA_An']*phi_An) / denom_LA, 0.0)
    M_MK   = max((p['p_MK_An']*phi_An + p['p_MK_Vd']*phi_Vd) / denom_MK, 0.0)
    M_H2O2 = max(p['p_H2O2'] * phi_So / max(p['dH2O2'], 1e-9), 0.0)
    return M_LA, M_MK, M_H2O2

def fitness_vector(phi, b, p):
    """Effective fitness f_i for each species."""
    M_LA, M_MK, M_H2O2 = qss_metabolites(phi, p)
    f = np.array(b, dtype=float)
    f[IDX_VD] += p['a_LA']   * M_LA
    f[IDX_PG] += p['a_MK']   * M_MK - p['a_H2O2'] * M_H2O2
    return f

def cr_rhs(phi, t, b, p):
    phi = np.clip(phi, 0.0, 1.0)
    phi = phi / max(phi.sum(), 1e-9)
    f   = fitness_vector(phi, b, p)
    return phi * (f - phi @ f)

def simulate_cr(b, theta_cr, t_weeks=(1, 2, 3), phi0=None):
    """
    Integrate CR ODE from week 0 to weeks in t_weeks.
    Returns phi array shape (len(t_weeks), N_SP).
    """
    p = unpack_cr(theta_cr)
    if phi0 is None:
        phi0 = np.ones(N_SP) / N_SP
    t_span = np.linspace(0, max(t_weeks), 500)
    sol = odeint(cr_rhs, phi0, t_span, args=(b, p), rtol=1e-6, atol=1e-8)
    sol = np.clip(sol, 0.0, None)
    sol = sol / sol.sum(axis=1, keepdims=True)
    out = []
    for tw in t_weeks:
        idx = np.searchsorted(t_span, tw)
        out.append(sol[idx])
    return np.array(out)  # (n_weeks, N_SP)

def rmse_cr(theta_cr, b_all, phi_obs, t_weeks=(1, 2, 3)):
    """
    phi_obs: (n_patients, n_weeks, N_SP), b_all: (n_patients, N_SP)
    Returns scalar RMSE over all valid (non-NaN) observations.
    """
    errs = []
    for k, (b, obs) in enumerate(zip(b_all, phi_obs)):
        phi_hat = simulate_cr(b, theta_cr, t_weeks)
        mask = ~np.isnan(obs)
        if mask.any():
            errs.append((phi_hat[mask] - obs[mask]) ** 2)
    return float(np.sqrt(np.concatenate(errs).mean())) if errs else np.nan

# ----- effective A matrix from CR params -----

def effective_A_from_cr(theta_cr):
    """
    Compute the 5×5 effective interaction matrix implied by the CR model
    at φ = (0.2, 0.2, 0.2, 0.2, 0.2) (equal composition reference).
    A_ij = d(f_i)/d(φ_j) at reference.
    """
    p  = unpack_cr(theta_cr)
    phi_ref = np.ones(N_SP) * 0.2
    A  = np.zeros((N_SP, N_SP))
    phi_So, phi_An, phi_Vd, phi_Fn, phi_Pg = phi_ref

    dLA   = max(p['c_LA_Vd'] * phi_Vd + p['dLA'], 1e-9)
    dMK   = max(p['c_MK_Pg'] * phi_Pg + p['dMK'], 1e-9)

    # ∂f_Vd / ∂φ_So  = α_LA * p_LA_So / denom_LA
    A[IDX_VD, IDX_SO] = p['a_LA'] * p['p_LA_So'] / dLA
    A[IDX_VD, IDX_AN] = p['a_LA'] * p['p_LA_An'] / dLA
    A[IDX_VD, IDX_VD] = -p['a_LA'] * (p['p_LA_So']*phi_So + p['p_LA_An']*phi_An) * p['c_LA_Vd'] / dLA**2

    A[IDX_PG, IDX_AN] = p['a_MK'] * p['p_MK_An'] / dMK
    A[IDX_PG, IDX_VD] = p['a_MK'] * p['p_MK_Vd'] / dMK
    A[IDX_PG, IDX_PG] = -p['a_MK'] * (p['p_MK_An']*phi_An + p['p_MK_Vd']*phi_Vd) * p['c_MK_Pg'] / dMK**2
    A[IDX_PG, IDX_SO] = -p['a_H2O2'] * p['p_H2O2'] / max(p['dH2O2'], 1e-9)
    return A

# ----- default initial parameter guess -----

def default_theta_cr():
    """Return a reasonable starting point for the 13 CR params."""
    return np.array([
        1.0,  1.0,  1.0, np.log(0.5),   # p_LA_So, p_LA_An, c_LA_Vd, log_δ_LA
        1.0,  1.0,  1.0, np.log(0.5),   # p_MK_An, p_MK_Vd, c_MK_Pg, log_δ_MK
        1.0,       np.log(0.5),          # p_H2O2, log_δ_H2O2
        1.0,  1.0,  1.0,                 # α_LA, α_MK, α_H2O2
    ], dtype=float)
