#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))

"""
Gaussian moment-closure fitting for replicator-form gLV on Heine 2025 data.

Motivation: ZapienCampos2024 (PLoS Biol, DOI 10.1371/journal.pbio.3002913) —
tracking 1st + 2nd statistical moments jointly improves identifiability of A.

For each condition (CS/CH/DS/DH):
  1. Deterministic fit (A, b) → AIC_det
  2. Moment-closure fit (A, b, D) tracking μ(t) + C(t) → AIC_moments
  3. Print Δ_AIC = AIC_moments - AIC_det (negative ⟹ moment-closure is better)

Outputs:
  results/fit_stochastic_glv_moments.json
  results/fig_stochastic_glv_moments.{pdf,png}
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis_style import use as thesis_style

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_CSV = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"
                "/fig3_species_distribution_replicates.csv")
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DAYS = [1, 3, 6, 10, 15, 21]
N_SP = 5
N_DAYS = len(DAYS)
N_PARAMS_DET = N_SP * N_SP + N_SP  # 30

CONDITIONS = [
    ("Commensal", "Static", "CS"),
    ("Commensal", "HOBIC",  "CH"),
    ("Dysbiotic", "Static", "DS"),
    ("Dysbiotic", "HOBIC",  "DH"),
]
SPECIES = {
    "Commensal": ["S. oralis", "A. naeslundii", "V. dispar",  "F. nucleatum", "P. gingivalis_20709"],
    "Dysbiotic": ["S. oralis", "A. naeslundii", "V. parvula", "F. nucleatum", "P. gingivalis_W83"],
}
SHORT = ["So", "An", "Vd/Vp", "Fn", "Pg"]
N_RESTARTS = 2
ALPHA_COV = 0.1   # covariance loss weight


# ---------------------------------------------------------------------------
# ODE helpers
# ---------------------------------------------------------------------------

def replicator_rhs(t, phi, A, b):
    phi = np.maximum(phi, 1e-10)
    phi = phi / phi.sum()
    f = A @ phi + b
    fbar = phi @ f
    return phi * (f - fbar)


def jacobian_replicator(phi, A, b):
    """Jacobian J_ij of replicator-form gLV at state phi."""
    phi = np.maximum(phi, 1e-10)
    phi = phi / phi.sum()
    f = A @ phi + b
    fbar = phi @ f
    # J_ij = phi_i*(A_ij - sum_k phi_k*A_kj) + delta_ij*(f_i - fbar)
    J = phi[:, None] * (A - (phi @ A)[None, :]) + np.diag(f - fbar)
    return J


def integrate_det(A, b, phi0):
    """Integrate deterministic trajectory; returns (N_DAYS, N_SP) array."""
    sol = solve_ivp(replicator_rhs, [DAYS[0], DAYS[-1]], phi0,
                    t_eval=DAYS, args=(A, b),
                    method="RK45", rtol=1e-6, atol=1e-9, max_step=0.5)
    traj = sol.y.T
    traj = np.maximum(traj, 0)
    traj = traj / traj.sum(axis=1, keepdims=True)
    return traj


def moment_rhs(t, state, A, b, D):
    """
    Joint ODE for (μ, C) using linearised moment closure.
      dμ/dt  = replicator_rhs(μ)          (first-order approx)
      dC/dt  = J C + C Jᵀ + diag(D)
    state: flat vector [μ (5), C_flat (25)]
    """
    mu = state[:N_SP]
    C = state[N_SP:].reshape(N_SP, N_SP)

    dmu = replicator_rhs(t, mu, A, b)
    J = jacobian_replicator(mu, A, b)
    dC = J @ C + C @ J.T + np.diag(D)

    return np.concatenate([dmu, dC.ravel()])


def integrate_moments(A, b, D, phi0, C0):
    """Returns (mu_traj, C_traj) of shapes (N_DAYS, N_SP) and (N_DAYS, N_SP, N_SP)."""
    state0 = np.concatenate([phi0, C0.ravel()])
    sol = solve_ivp(moment_rhs, [DAYS[0], DAYS[-1]], state0,
                    t_eval=DAYS, args=(A, b, D),
                    method="RK45", rtol=1e-5, atol=1e-8, max_step=0.5)
    mu_traj = sol.y[:N_SP, :].T
    mu_traj = np.maximum(mu_traj, 0)
    mu_traj = mu_traj / mu_traj.sum(axis=1, keepdims=True)
    C_traj = sol.y[N_SP:, :].T.reshape(N_DAYS, N_SP, N_SP)
    return mu_traj, C_traj


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition(df, condition, cultivation):
    """
    Returns:
      mu_obs  : (N_DAYS, N_SP) mean of replicates (fraction, not %)
      C_obs   : (N_DAYS, N_SP, N_SP) empirical covariance (or NaN-filled if < 2 reps)
      n_reps  : (N_DAYS,) number of replicates per day
    """
    sp_list = SPECIES[condition]
    sub = df[(df["condition"] == condition) & (df["cultivation"] == cultivation)]
    mu_obs = np.zeros((N_DAYS, N_SP))
    C_obs = np.full((N_DAYS, N_SP, N_SP), np.nan)
    n_reps = np.zeros(N_DAYS, dtype=int)

    for ti, day in enumerate(DAYS):
        day_data = sub[sub["day"] == day]
        # matrix shape: (n_replicates, N_SP)
        reps = np.array([
            day_data[day_data["species"] == sp]["distribution_pct"].values
            for sp in sp_list
        ], dtype=float).T  # (n_rep, N_SP)
        if reps.shape[0] == 0:
            continue
        # Normalise to fractions
        row_sums = reps.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        reps = reps / row_sums
        n_reps[ti] = reps.shape[0]
        mu_obs[ti] = reps.mean(axis=0)
        if reps.shape[0] >= 2:
            C_obs[ti] = np.cov(reps.T)
    return mu_obs, C_obs, n_reps


# ---------------------------------------------------------------------------
# Deterministic fit
# ---------------------------------------------------------------------------

def rmse_det(x, mu_obs):
    A = x[:N_SP * N_SP].reshape(N_SP, N_SP)
    b = x[N_SP * N_SP:]
    try:
        pred = integrate_det(A, b, mu_obs[0])
        return float(np.sqrt(np.mean((pred - mu_obs) ** 2)))
    except Exception:
        return 1.0


def fit_deterministic(mu_obs, rng):
    best_res = None
    for _ in range(N_RESTARTS):
        x0 = rng.normal(0, 0.5, N_PARAMS_DET)
        res = minimize(rmse_det, x0, args=(mu_obs,),
                       method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    A = best_res.x[:N_SP * N_SP].reshape(N_SP, N_SP)
    b = best_res.x[N_SP * N_SP:]
    rmse = best_res.fun
    return A, b, rmse


# ---------------------------------------------------------------------------
# Moment-closure fit
# ---------------------------------------------------------------------------

def loss_moments(x, mu_obs, C_obs, C0):
    A = x[:N_SP * N_SP].reshape(N_SP, N_SP)
    b = x[N_SP * N_SP: N_PARAMS_DET]
    D = np.abs(x[N_PARAMS_DET:])  # enforce non-negative
    try:
        mu_pred, C_pred = integrate_moments(A, b, D, mu_obs[0], C0)
    except Exception:
        return 1.0 + ALPHA_COV

    rmse_mu = float(np.sqrt(np.mean((mu_pred - mu_obs) ** 2)))

    # Covariance loss: only at timepoints with valid empirical covariance
    cov_errs = []
    for ti in range(N_DAYS):
        if not np.any(np.isnan(C_obs[ti])):
            cov_errs.append(np.mean((C_pred[ti] - C_obs[ti]) ** 2))
    rmse_cov = float(np.sqrt(np.mean(cov_errs))) if cov_errs else 0.0

    return rmse_mu + ALPHA_COV * rmse_cov


def fit_moments(mu_obs, C_obs, A_det, b_det, rng):
    # Initial covariance: empirical at day 1 (if available), else zeros
    C0 = C_obs[0] if not np.any(np.isnan(C_obs[0])) else np.zeros((N_SP, N_SP))
    C0 = np.maximum(C0, 0)  # ensure PSD approx

    best_res = None
    for k in range(N_RESTARTS):
        if k == 0:
            # warm-start from deterministic solution
            x0_ab = np.concatenate([A_det.ravel(), b_det])
        else:
            x0_ab = rng.normal(0, 0.5, N_PARAMS_DET)
        x0_D = np.abs(rng.normal(0, 0.01, N_SP))
        x0 = np.concatenate([x0_ab, x0_D])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(loss_moments, x0,
                           args=(mu_obs, C_obs, C0),
                           method="L-BFGS-B",
                           bounds=([(None, None)] * N_PARAMS_DET +
                                   [(0.0, None)] * N_SP),
                           options={"maxiter": 800, "ftol": 1e-12})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    A = best_res.x[:N_SP * N_SP].reshape(N_SP, N_SP)
    b = best_res.x[N_SP * N_SP: N_PARAMS_DET]
    D = np.abs(best_res.x[N_PARAMS_DET:])

    # Recompute individual RMSE components
    C0_use = C_obs[0] if not np.any(np.isnan(C_obs[0])) else np.zeros((N_SP, N_SP))
    C0_use = np.maximum(C0_use, 0)
    try:
        mu_pred, C_pred = integrate_moments(A, b, D, mu_obs[0], C0_use)
        rmse_mu = float(np.sqrt(np.mean((mu_pred - mu_obs) ** 2)))
        cov_errs = []
        for ti in range(N_DAYS):
            if not np.any(np.isnan(C_obs[ti])):
                cov_errs.append(np.mean((C_pred[ti] - C_obs[ti]) ** 2))
        rmse_cov = float(np.sqrt(np.mean(cov_errs))) if cov_errs else 0.0
    except Exception:
        rmse_mu = best_res.fun
        rmse_cov = 0.0
        mu_pred = np.zeros_like(mu_obs)
        C_pred = np.zeros((N_DAYS, N_SP, N_SP))

    return A, b, D, rmse_mu, rmse_cov, mu_pred, C_pred


# ---------------------------------------------------------------------------
# AIC helpers
# ---------------------------------------------------------------------------

def aic(n_obs, rmse, n_params):
    return n_obs * np.log(rmse ** 2 + 1e-30) + 2 * n_params


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COLORS = ["#377eb8", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3"]

def plot_condition(axes_mu, axes_cov, label, mu_obs, C_obs,
                   mu_det, mu_mom, C_mom):
    days_arr = np.array(DAYS)
    # --- mean trajectories ---
    for i in range(N_SP):
        col = COLORS[i]
        axes_mu.plot(days_arr, mu_obs[:, i],  "o",   color=col, ms=3)
        axes_mu.plot(days_arr, mu_det[:, i],  "--",  color=col, lw=1.0, alpha=0.7)
        axes_mu.plot(days_arr, mu_mom[:, i],  "-",   color=col, lw=1.5,
                     label=SHORT[i] if i == 0 else "_")
    axes_mu.set_title(label, fontsize=8)
    axes_mu.set_ylabel("Fraction", fontsize=7)
    axes_mu.set_ylim(0, 1)
    axes_mu.tick_params(labelsize=6)

    # --- diagonal variance ---
    for i in range(N_SP):
        col = COLORS[i]
        var_obs = np.array([C_obs[ti, i, i] if not np.isnan(C_obs[ti, i, i]) else np.nan
                            for ti in range(N_DAYS)])
        var_mom = np.array([C_mom[ti, i, i] for ti in range(N_DAYS)])
        axes_cov.plot(days_arr, var_obs, "o",  color=col, ms=3)
        axes_cov.plot(days_arr, var_mom, "-",  color=col, lw=1.2)
    axes_cov.set_ylabel("Var (diag C)", fontsize=7)
    axes_cov.set_xlabel("Day", fontsize=7)
    axes_cov.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)
    df = pd.read_csv(DATA_CSV)

    fig_size = thesis_style(width_frac=1.0, aspect=0.6)
    fig, axes = plt.subplots(2, 4, figsize=fig_size)

    results = {}

    for col_idx, (condition, cultivation, label) in enumerate(CONDITIONS):
        print(f"\n=== {label} ({condition}, {cultivation}) ===")
        mu_obs, C_obs, n_reps = load_condition(df, condition, cultivation)
        print(f"  Replicates per day: {n_reps}")

        # --- deterministic fit ---
        A_det, b_det, rmse_det_val = fit_deterministic(mu_obs, rng)
        n_obs_mu = N_DAYS * N_SP
        aic_det = aic(n_obs_mu, rmse_det_val, N_PARAMS_DET)
        mu_det = integrate_det(A_det, b_det, mu_obs[0])
        print(f"  Det  RMSE_mu={rmse_det_val:.4f}  AIC={aic_det:.2f}")

        # --- moment-closure fit ---
        A_mom, b_mom, D_mom, rmse_mu_mom, rmse_cov_mom, mu_mom, C_mom = \
            fit_moments(mu_obs, C_obs, A_det, b_det, rng)
        aic_mom = aic(n_obs_mu, rmse_mu_mom, N_PARAMS_DET + N_SP)
        delta_aic = aic_mom - aic_det
        print(f"  Mom  RMSE_mu={rmse_mu_mom:.4f}  RMSE_cov={rmse_cov_mom:.4f}"
              f"  AIC={aic_mom:.2f}  ΔAIC={delta_aic:.2f}"
              f"  {'(moment-closure better)' if delta_aic < 0 else '(det better)'}")

        results[label] = dict(
            rmse_det=rmse_det_val,
            rmse_moments=rmse_mu_mom,
            rmse_cov=rmse_cov_mom,
            aic_det=aic_det,
            aic_moments=aic_mom,
            delta_aic=delta_aic,
            A_det=A_det.tolist(),
            b_det=b_det.tolist(),
            A_moments=A_mom.tolist(),
            b_moments=b_mom.tolist(),
            D_moments=D_mom.tolist(),
        )

        plot_condition(axes[0, col_idx], axes[1, col_idx], label,
                       mu_obs, C_obs, mu_det, mu_mom, C_mom)

    # Legend on first mu-panel
    for i, sp in enumerate(SHORT):
        axes[0, 0].plot([], [], color=COLORS[i], label=sp)
    axes[0, 0].legend(fontsize=5, loc="upper right")
    # Dashed = det, solid = moment-closure annotation
    axes[0, 0].plot([], [], "k--", lw=1.0, label="det")
    axes[0, 0].plot([], [], "k-",  lw=1.5, label="mom")
    axes[0, 0].legend(fontsize=5, loc="upper right", ncol=2)

    fig.suptitle("Moment-closure gLV fit — Heine 2025 (5 species)", fontsize=8)
    fig.tight_layout()

    pdf_path = RESULTS_DIR / "fig_stochastic_glv_moments.pdf"
    png_path = RESULTS_DIR / "fig_stochastic_glv_moments.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {pdf_path}")

    json_path = RESULTS_DIR / "fit_stochastic_glv_moments.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved: {json_path}")

    # Summary table
    print("\n--- AIC summary ---")
    print(f"{'Cond':<6}  {'AIC_det':>10}  {'AIC_mom':>10}  {'ΔAIC':>8}")
    for label, r in results.items():
        print(f"{label:<6}  {r['aic_det']:>10.2f}  {r['aic_moments']:>10.2f}"
              f"  {r['delta_aic']:>8.2f}")


if __name__ == "__main__":
    main()
