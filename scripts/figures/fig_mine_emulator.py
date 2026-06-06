#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
fig_mine_emulator.py — MINE-style (MCMC Informed Neural Emulator) for the
Heine 5-species Hamilton ODE.

Following Haario et al. 2026 (arXiv 2603.10987 / Haario2026MINE):
"Instead of sampling network weights, we introduce the model-parameter
distribution as an input to network training via MCMC. In this way, the
surrogate achieves the same uncertainty quantification as the underlying
physical model, but with substantially reduced computation time."

Here the TMCMC 10 000-particle posterior (results/ultimate_10000p/) already
provides posterior-informed parameter draws.  We:
  1. Subsample N_TRAIN=3000 draws from each attractor posterior.
  2. Forward-simulate the Hamilton ODE for each draw → trajectory (6 × 5).
  3. Train a MLP regressor: θ (20-dim) → RMSE scalar.
  4. Evaluate on held-out N_TEST=500 draws.
  5. Plot parity (true RMSE vs predicted RMSE), speedup bar.

The emulator can replace ODE calls during LOO-style evaluation, enabling
rapid sensitivity analysis over the posterior.

Outputs:
  results/fig_mine_emulator.{pdf,png}
  results/mine_emulator_data.npz    (theta, rmse_true, rmse_pred per cond)

Run:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/fig_mine_emulator.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from paper_data import paper_5sp_samples
from thesis_style import use as thesis_style

# ── Heine data loading ────────────────────────────────────────────────────────
DATA_CSV = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species"
                "/experiment_data/fig3_species_distribution_replicates.csv")
DAYS = [1, 3, 6, 10, 15, 21]
N_SP = 5
CONDITIONS_MAP = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}
SPECIES_MAP = {
    "Commensal": ["S. oralis", "A. naeslundii", "V. dispar",
                  "F. nucleatum", "P. gingivalis_20709"],
    "Dysbiotic": ["S. oralis", "A. naeslundii", "V. parvula",
                  "F. nucleatum", "P. gingivalis_W83"],
}

# ── theta parameterisation ────────────────────────────────────────────────────
A_IDX = {
    (0, 0): 0,  (0, 1): 1,  (1, 1): 2,
    (2, 2): 5,  (2, 3): 6,  (3, 3): 7,
    (0, 2): 10, (0, 3): 11, (1, 2): 12, (1, 3): 13, (4, 4): 14,
    (0, 4): 16, (1, 4): 17, (2, 4): 18, (3, 4): 19,
}
B_IDX = [3, 4, 8, 9, 15]   # b_0 … b_4

OUT = Path(__file__).resolve().parents[2] / "results"
N_TRAIN = 3000
N_TEST  = 500
RNG_SEED = 42


def load_phi_obs(df: pd.DataFrame, cond_key: str) -> np.ndarray:
    cond_str, cult_str = CONDITIONS_MAP[cond_key]
    sp_list = SPECIES_MAP[cond_str]
    mask = (df["condition"] == cond_str) & (df["cultivation"] == cult_str)
    sub = df[mask]
    phi = np.zeros((len(DAYS), N_SP))
    for j, sp in enumerate(sp_list):
        sp_sub = sub[sub["species"] == sp]
        for i, day in enumerate(DAYS):
            vals = sp_sub[sp_sub["day"] == day]["distribution_pct"].values
            phi[i, j] = np.median(vals) if len(vals) > 0 else 0.0
    row_sums = phi.sum(axis=1, keepdims=True)
    phi = phi / np.where(row_sums > 0, row_sums, 1.0)
    return phi


def theta_to_Ab(theta: np.ndarray):
    A = np.zeros((N_SP, N_SP))
    for (i, j), k in A_IDX.items():
        A[i, j] = A[j, i] = theta[k]
    b = theta[B_IDX]
    return A, b


def integrate_hamilton(A: np.ndarray, b: np.ndarray,
                       phi0: np.ndarray) -> np.ndarray:
    def rhs(t, phi):
        phi = np.maximum(phi, 1e-10)
        phi = phi / phi.sum()
        f = A @ phi + b
        fbar = phi @ f
        return phi * (f - fbar)

    sol = solve_ivp(rhs, [DAYS[0], DAYS[-1]], phi0.copy(),
                    t_eval=DAYS, method="RK45",
                    rtol=1e-5, atol=1e-8, max_step=2.0)
    if not sol.success:
        return np.full((len(DAYS), N_SP), np.nan)
    traj = sol.y.T
    traj = np.maximum(traj, 0.0)
    row_sums = traj.sum(axis=1, keepdims=True)
    traj = traj / np.where(row_sums > 0, row_sums, 1.0)
    return traj


def rmse_from_theta(theta: np.ndarray, phi_obs: np.ndarray) -> float:
    A, b = theta_to_Ab(theta)
    traj = integrate_hamilton(A, b, phi_obs[0])
    if np.any(np.isnan(traj)):
        return np.nan
    return float(np.sqrt(np.mean((traj - phi_obs) ** 2)))


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading Heine data …")
df = pd.read_csv(DATA_CSV)
phi_obs_all = {ck: load_phi_obs(df, ck) for ck in CONDITIONS_MAP}

CONDS = ["CS", "CH", "DS", "DH"]
COND_COLOR = {"CS": "#2166ac", "CH": "#74add1", "DS": "#d73027", "DH": "#a50026"}

rng = np.random.default_rng(RNG_SEED)

all_results = {}
for cond in CONDS:
    print(f"\n── {cond} ──")
    samples = paper_5sp_samples(cond)
    phi_obs = phi_obs_all[cond]

    idx_all = rng.permutation(len(samples))
    idx_train = idx_all[:N_TRAIN]
    idx_test  = idx_all[N_TRAIN:N_TRAIN + N_TEST]

    # Forward simulate training set
    print(f"  Simulating {N_TRAIN} training samples …", end=" ", flush=True)
    t0 = time.perf_counter()
    X_train = samples[idx_train]
    y_train = np.array([rmse_from_theta(th, phi_obs) for th in X_train])
    ode_time_train = time.perf_counter() - t0
    valid = np.isfinite(y_train)
    X_train, y_train = X_train[valid], y_train[valid]
    print(f"done in {ode_time_train:.1f}s  ({valid.sum()} valid)")

    # Forward simulate test set
    print(f"  Simulating {N_TEST}  test     samples …", end=" ", flush=True)
    t0 = time.perf_counter()
    X_test = samples[idx_test]
    y_test = np.array([rmse_from_theta(th, phi_obs) for th in X_test])
    ode_time_test = time.perf_counter() - t0
    valid_t = np.isfinite(y_test)
    X_test, y_test = X_test[valid_t], y_test[valid_t]
    print(f"done in {ode_time_test:.1f}s")

    # Compute trajectories for training set (30-dim output: 6 tp × 5 sp)
    print(f"  Computing trajectories (train) …", end=" ", flush=True)
    A_train_traj = np.zeros((len(X_train), len(DAYS) * N_SP))
    for s_idx, theta in enumerate(X_train):
        A, b = theta_to_Ab(theta)
        traj = integrate_hamilton(A, b, phi_obs[0])
        if np.any(np.isnan(traj)):
            A_train_traj[s_idx] = phi_obs.ravel()   # fallback: obs
        else:
            A_train_traj[s_idx] = traj.ravel()
    print("done")

    print(f"  Computing trajectories (test)  …", end=" ", flush=True)
    A_test_traj = np.zeros((len(X_test), len(DAYS) * N_SP))
    for s_idx, theta in enumerate(X_test):
        A, b = theta_to_Ab(theta)
        traj = integrate_hamilton(A, b, phi_obs[0])
        if np.any(np.isnan(traj)):
            A_test_traj[s_idx] = phi_obs.ravel()
        else:
            A_test_traj[s_idx] = traj.ravel()
    print("done")

    # Scale & train MLP on trajectory output
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(A_train_traj)
    X_tr_s = scaler_X.transform(X_train)
    X_te_s = scaler_X.transform(X_test)
    y_tr_s = scaler_y.transform(A_train_traj)

    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                       activation="relu", max_iter=800,
                       random_state=RNG_SEED, verbose=False,
                       early_stopping=True, validation_fraction=0.1)
    print(f"  Training MLP (θ→traj) …", end=" ", flush=True)
    t0 = time.perf_counter()
    mlp.fit(X_tr_s, y_tr_s)
    nn_time_train = time.perf_counter() - t0
    print(f"done in {nn_time_train:.1f}s")

    # Predict trajectories then compute RMSE from predicted trajectory
    t0 = time.perf_counter()
    y_pred_s = mlp.predict(X_te_s)
    nn_time_pred = time.perf_counter() - t0
    y_pred_traj = scaler_y.inverse_transform(y_pred_s)
    y_pred_traj = np.clip(y_pred_traj, 0, 1)

    # RMSE of emulator trajectory vs true ODE trajectory
    diff = (y_pred_traj - A_test_traj).reshape(len(X_test), len(DAYS), N_SP)
    y_test_rmse = np.sqrt(np.mean((A_test_traj - phi_obs.ravel())**2, axis=1))
    y_pred_rmse = np.sqrt(np.mean(diff**2, axis=(1,2)))
    y_test = y_test_rmse   # redefine as RMSE for plotting parity
    y_pred = y_pred_rmse

    r2 = r2_score(A_test_traj.ravel(), y_pred_traj.ravel())
    rmse_emul = float(np.sqrt(np.mean((y_pred_traj - A_test_traj)**2)))
    ode_per_eval = ode_time_test / len(y_test) * 1000   # ms
    nn_per_eval  = nn_time_pred  / len(y_test) * 1e6    # µs
    speedup = ode_per_eval * 1000 / nn_per_eval if nn_per_eval > 0 else np.inf

    print(f"  R²={r2:.4f}  RMSE_emul={rmse_emul:.5f}")
    print(f"  ODE: {ode_per_eval:.2f} ms/call,  NN: {nn_per_eval:.2f} µs/call"
          f"  → ×{speedup:.0f} speedup")

    all_results[cond] = dict(
        X_test=X_test, y_test=y_test, y_pred=y_pred,
        r2=r2, rmse_emul=rmse_emul,
        ode_ms=ode_per_eval, nn_us=nn_per_eval, speedup=speedup,
        ode_time_train=ode_time_train, nn_time_train=nn_time_train,
    )

# ── Figure ─────────────────────────────────────────────────────────────────────
thesis_style()
fig, axes = plt.subplots(2, 4, figsize=thesis_style(1.0, aspect=0.55))

for col, cond in enumerate(CONDS):
    res = all_results[cond]
    color = COND_COLOR[cond]

    # Top row: parity plot (true vs predicted RMSE)
    ax = axes[0, col]
    ax.scatter(res["y_test"], res["y_pred"], s=4, alpha=0.4,
               color=color, rasterized=True)
    lo = min(res["y_test"].min(), res["y_pred"].min())
    hi = max(res["y_test"].max(), res["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("ODE RMSE" if col == 0 else "")
    ax.set_ylabel("NN RMSE" if col == 0 else "")
    ax.set_title(f"{cond}\n$R^2$={res['r2']:.3f}", fontsize=8)
    ax.tick_params(labelsize=7)

    # Bottom row: speedup & timing bar
    ax2 = axes[1, col]
    labels = ["ODE (ms)", "NN (µs×0.001)"]
    heights = [res["ode_ms"], res["nn_us"] * 0.001]
    bars = ax2.bar(labels, heights, color=[color, "gray"], width=0.5)
    ax2.set_ylabel("Time / call" if col == 0 else "")
    ax2.set_title(f"×{res['speedup']:.0f}", fontsize=9)
    ax2.tick_params(axis="x", labelsize=7)

axes[0, 0].set_ylabel("Predicted RMSE (NN)")
axes[1, 0].set_ylabel("Normalized time (ms)")

fig.suptitle("MINE neural emulator — Heine 5-species Hamilton ODE\n"
             "(Haario et al. 2026, arXiv 2603.10987)", fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_mine_emulator.{ext}", dpi=300, bbox_inches="tight")
print(f"\nSaved → results/fig_mine_emulator.{{pdf,png}}")

# Save data
np.savez(OUT / "mine_emulator_data.npz",
         **{f"{c}_{k}": v for c, res in all_results.items()
            for k, v in res.items() if isinstance(v, np.ndarray)})
