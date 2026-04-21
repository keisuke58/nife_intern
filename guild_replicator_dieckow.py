#!/usr/bin/env python3
"""
10-guild generalized Lotka-Volterra replicator model for Dieckow data.

ODE (replicator form, conserves Σφ_i = 1):
  dφ_i/dt = φ_i * (f_i(φ) - <f>(φ))
  f_i(φ)  = b_i + Σ_j A_ij * φ_j
  <f>      = Σ_i φ_i * f_i

Parameters:
  A      : (N_G, N_G) shared interaction matrix
  b_all  : (n_patients, N_G) patient-specific intrinsic rates

phi_obs: (n_patients, 3, N_G)  — weeks 1, 2, 3
RMSE integrates W1→W2 and W2→W3 (Δt = 1 week each).
"""

import numpy as np
from scipy.integrate import solve_ivp

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
N_G = len(GUILD_ORDER)   # 10

DT = 1.0   # 1 week per interval


def replicator_rhs(t, phi, b, A):
    f   = b + A @ phi
    fmean = phi @ f
    return phi * (f - fmean)


def integrate_step(phi0, b, A, dt=DT):
    """Integrate replicator ODE for one time step; clip to [0,1] and renormalise."""
    sol = solve_ivp(replicator_rhs, [0, dt], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8, dense_output=False)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


def predict_trajectory(phi0, b, A):
    """Predict weeks 2 and 3 starting from week 1."""
    phi2 = integrate_step(phi0, b, A)
    phi3 = integrate_step(phi2, b, A)
    return phi2, phi3


def rmse_guild(A, b_all, phi_obs):
    """Mean RMSE over all patients for W2 and W3 predictions."""
    n_p = phi_obs.shape[0]
    sq_sum = 0.0
    count  = 0
    for i in range(n_p):
        phi2_pred, phi3_pred = predict_trajectory(phi_obs[i, 0], b_all[i], A)
        sq_sum += np.sum((phi2_pred - phi_obs[i, 1])**2)
        sq_sum += np.sum((phi3_pred - phi_obs[i, 2])**2)
        count  += 2 * N_G
    return np.sqrt(sq_sum / count)


# ---- pack / unpack ----
N_A = N_G * N_G   # 100 A entries


def pack(A, b_all):
    return np.concatenate([A.ravel(), b_all.ravel()])


def unpack(theta, n_p):
    A     = theta[:N_A].reshape(N_G, N_G)
    b_all = theta[N_A:].reshape(n_p, N_G)
    return A, b_all


def default_A():
    """Initialise A: small negative diagonal (self-limitation), zeros off-diagonal."""
    A = np.zeros((N_G, N_G))
    np.fill_diagonal(A, -0.1)
    return A
