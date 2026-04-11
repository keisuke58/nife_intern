"""
run_posterior_predictive.py — Posterior predictive checks for COMETS Monod calibration
=======================================================================================
B) Full-trajectory CI bands  — 5 species per condition, 90 % posterior predictive
A) Cross-condition RMSE matrix  — 2×2 (train × eval) normalised RMSE heatmap

Loads
-----
  pipeline_results/tmcmc_monod_{diseased,healthy}/samples.npz
  pipeline_results/oed_result.json
  pipeline_results/oed_healthy/oed_result.json

Outputs
-------
  tmcmc_monod_diseased/posterior_predictive.png
  tmcmc_monod_healthy/posterior_predictive.png
  cross_prediction/rmse_matrix.png
  cross_prediction/rmse_matrix.json

Usage
-----
    python comets/run_posterior_predictive.py \\
        --out comets/pipeline_results \\
        --n_pp_draws 100 \\
        --n_cross_draws 200 \\
        --workers 4 \\
        --seed 0

PBS: see run_posterior_predictive.sh
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Simulation engine (shared with run_oed.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))   # put nife/ on sys.path

from comets.run_oed import (            # noqa: E402
    _simulate,
    _apply_theta,
    THETA_NOMINAL,
    PARAM_NAMES,
    N_PARAMS,
    SPECIES_ORDER,
)

SP_COLORS = {
    "So": "#1f77b4", "An": "#ff7f0e", "Vp": "#2ca02c",
    "Fn": "#d62728", "Pg": "#9467bd",
}

# ---------------------------------------------------------------------------
# Module-level simulation helper (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _sim_full(args: tuple) -> np.ndarray:
    """Simulate and return full trajectory (max_cycles, 5)."""
    log_theta, condition, max_cycles, dt = args
    theta = np.exp(log_theta)
    return _simulate(_apply_theta(theta), condition, max_cycles, dt)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_samples(results_dir: Path, condition: str) -> np.ndarray:
    """Return log_theta_samples (N, 10) from saved npz."""
    npz = np.load(results_dir / f"tmcmc_monod_{condition}" / "samples.npz")
    return npz["log_theta_samples"]


def _load_oed(results_dir: Path, condition: str) -> tuple[list[int], list[float]]:
    """Return (selected_cycles, selected_times_h) from OED JSON."""
    if condition == "healthy":
        path = results_dir / "oed_healthy" / "oed_result.json"
    else:
        path = results_dir / "oed_result.json"
    with open(path) as f:
        oed = json.load(f)
    return oed["selected_cycles"], oed["selected_times_h"]


def _subsample_idx(n_total: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_total, size=min(n_draws, n_total), replace=False)


# ---------------------------------------------------------------------------
# B) Full-trajectory posterior predictive
# ---------------------------------------------------------------------------

_FULL_CYCLES = 8000   # 80 h at dt=0.01


def plot_posterior_predictive(
    log_theta_samples: np.ndarray,
    condition: str,
    selected_cycles: list[int],
    selected_times_h: list[float],
    out_path: Path,
    dt: float = 0.01,
    n_draws: int = 100,
    workers: int = 1,
    seed: int = 0,
) -> None:
    """
    Draw n_draws samples from the posterior, simulate each to _FULL_CYCLES,
    and plot 90 % CI bands against the noiseless truth.
    """
    rng = np.random.default_rng(seed)
    idx = _subsample_idx(len(log_theta_samples), n_draws, rng)
    draws = log_theta_samples[idx]                     # (n_draws, 10)

    # Noiseless truth trajectory (at THETA_NOMINAL)
    bm_truth = _simulate(_apply_theta(THETA_NOMINAL), condition, _FULL_CYCLES, dt)
    t_h = np.arange(_FULL_CYCLES) * dt

    # Parallel simulation of posterior draws
    print(f"  [PP] simulating {len(draws)} draws for {condition} …", flush=True)
    t0 = time.time()
    args = [(d, condition, _FULL_CYCLES, dt) for d in draws]
    if workers > 1:
        with Pool(workers) as pool:
            trajs = np.array(pool.map(_sim_full, args))   # (n_draws, 8000, 5)
    else:
        trajs = np.array([_sim_full(a) for a in args])
    print(f"  [PP] done in {time.time()-t0:.1f}s", flush=True)

    q5  = np.percentile(trajs, 5,  axis=0)   # (8000, 5)
    q50 = np.percentile(trajs, 50, axis=0)
    q95 = np.percentile(trajs, 95, axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for j, (sp, ax) in enumerate(zip(SPECIES_ORDER, axes)):
        col = SP_COLORS[sp]
        ax.fill_between(t_h, q5[:, j], q95[:, j], alpha=0.25, color=col)
        ax.plot(t_h, q50[:, j],   color=col, lw=1.5, ls="-",  label="posterior median")
        ax.plot(t_h, bm_truth[:, j], color="black", lw=1.0, ls="--", alpha=0.8, label="truth (θ_nom)")
        # Mark OED observation times
        obs_bm = bm_truth[selected_cycles, j]
        ax.scatter(selected_times_h, obs_bm, color="red", s=30, zorder=6,
                   label="OED timepoints" if j == 0 else "")
        ax.set_title(sp, fontsize=10)
        ax.set_xlabel("Time (h)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Biomass (g)", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"Posterior predictive — {condition}  "
        f"({len(draws)} draws, 90 % CI shaded)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# A) Cross-condition RMSE matrix
# ---------------------------------------------------------------------------

def _nrmse_draws(
    log_theta_samples: np.ndarray,
    cond_eval: str,
    cycles_eval: list[int],
    dt: float,
    n_draws: int,
    workers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Evaluate log_theta_samples on cond_eval at its OED timepoints.

    Returns
    -------
    nrmse : (n_draws,) — per-draw mean normalised RMSE across species
            NRMSE_s = sqrt(mean_t[(pred-truth)^2]) / mean_t[truth]
            overall  = mean_s[NRMSE_s]
    """
    idx = _subsample_idx(len(log_theta_samples), n_draws, rng)
    draws = log_theta_samples[idx]

    max_cycles_eval = max(cycles_eval) + 1
    bm_truth = _simulate(_apply_theta(THETA_NOMINAL), cond_eval, max_cycles_eval, dt)
    y_truth = bm_truth[cycles_eval, :]                 # (n_obs, 5)

    args = [(d, cond_eval, max_cycles_eval, dt) for d in draws]
    if workers > 1:
        with Pool(workers) as pool:
            trajs = pool.map(_sim_full, args)           # list of (max_cycles_eval, 5)
    else:
        trajs = [_sim_full(a) for a in args]

    nrmse_per_draw = []
    for traj in trajs:
        y_pred = np.asarray(traj)[cycles_eval, :]      # (n_obs, 5)
        mean_truth = np.maximum(np.mean(y_truth, axis=0), 1e-15)
        nrmse_sp = np.sqrt(np.mean((y_pred - y_truth) ** 2, axis=0)) / mean_truth
        nrmse_per_draw.append(float(np.mean(nrmse_sp)))

    return np.array(nrmse_per_draw)


def run_cross_prediction(
    log_samples: dict[str, np.ndarray],   # {"diseased": …, "healthy": …}
    oed_cycles: dict[str, list[int]],
    out_dir: Path,
    dt: float = 0.01,
    n_draws: int = 200,
    workers: int = 1,
    seed: int = 0,
) -> dict:
    """
    Compute 2×2 NRMSE matrix: rows = train condition, cols = eval condition.
    Also saves a heatmap and JSON.
    """
    conditions = ["diseased", "healthy"]
    rng = np.random.default_rng(seed)

    matrix_median: list[list[float]] = []
    matrix_mean:   list[list[float]] = []
    raw: dict[str, dict[str, list[float]]] = {c: {} for c in conditions}

    for cond_train in conditions:
        row_med, row_mean = [], []
        for cond_eval in conditions:
            tag = f"{cond_train}→{cond_eval}"
            print(f"  [cross] {tag}: {n_draws} draws …", flush=True)
            t0 = time.time()
            nrmse = _nrmse_draws(
                log_samples[cond_train],
                cond_eval,
                oed_cycles[cond_eval],
                dt, n_draws, workers, rng,
            )
            print(f"    done {time.time()-t0:.1f}s  median NRMSE={np.median(nrmse):.4f}")
            raw[cond_train][cond_eval] = nrmse.tolist()
            row_med.append(float(np.median(nrmse)))
            row_mean.append(float(np.mean(nrmse)))
        matrix_median.append(row_med)
        matrix_mean.append(row_mean)

    # ---- Heatmap ----
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = np.array(matrix_median)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(mat, cmap="viridis_r", vmin=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["diseased", "healthy"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["diseased", "healthy"])
    ax.set_xlabel("Eval condition"); ax.set_ylabel("Train condition")
    ax.set_title("Median NRMSE (2×2 cross-prediction)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() * 0.5 else "black",
                    fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="NRMSE")

    # Right: box plots (4 conditions)
    ax2 = axes[1]
    labels, data = [], []
    for cond_train in conditions:
        for cond_eval in conditions:
            arrow = "→"
            labels.append(f"{cond_train[:3]}{arrow}{cond_eval[:3]}")
            data.append(raw[cond_train][cond_eval])
    bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True, notch=False)
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
    ax2.set_ylabel("NRMSE (per draw)")
    ax2.set_title("NRMSE distribution (n_draws per cell)")
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("Cross-condition posterior predictive RMSE", fontsize=12)
    plt.tight_layout()
    png_path = out_dir / "rmse_matrix.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {png_path}")

    # ---- JSON ----
    result = {
        "conditions": conditions,
        "n_draws": n_draws,
        "dt": dt,
        "matrix_median_nrmse": matrix_median,
        "matrix_mean_nrmse": matrix_mean,
        "raw_nrmse": raw,
    }
    json_path = out_dir / "rmse_matrix.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved: {json_path}")
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Posterior predictive checks for COMETS Monod calibration"
    )
    ap.add_argument("--out",           default="comets/pipeline_results",
                    help="Root results directory (same as run_tmcmc_monod.py --out)")
    ap.add_argument("--n_pp_draws",    type=int, default=100,
                    help="Posterior draws for full-trajectory CI plots (B)")
    ap.add_argument("--n_cross_draws", type=int, default=200,
                    help="Posterior draws per cell for cross-prediction (A)")
    ap.add_argument("--workers",       type=int, default=1,
                    help="Parallel workers (Pool)")
    ap.add_argument("--dt",            type=float, default=0.01,
                    help="Time step (h) — must match OED / TMCMC")
    ap.add_argument("--seed",          type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    rng_seed = args.seed

    # Load posteriors
    print("Loading posteriors …")
    log_samples: dict[str, np.ndarray] = {}
    oed_cycles:  dict[str, list[int]]  = {}
    oed_times:   dict[str, list[float]]= {}

    for cond in ("diseased", "healthy"):
        log_samples[cond] = _load_samples(out_root, cond)
        oed_cycles[cond], oed_times[cond] = _load_oed(out_root, cond)
        print(f"  {cond}: {log_samples[cond].shape[0]} samples, "
              f"OED times={[f'{t:.1f}h' for t in oed_times[cond]]}")

    # ---- B: Posterior predictive CI plots ----
    print("\n=== B: Posterior predictive (full trajectory) ===")
    for cond in ("diseased", "healthy"):
        out_pp = out_root / f"tmcmc_monod_{cond}" / "posterior_predictive.png"
        plot_posterior_predictive(
            log_samples[cond], cond,
            oed_cycles[cond], oed_times[cond],
            out_path=out_pp,
            dt=args.dt,
            n_draws=args.n_pp_draws,
            workers=args.workers,
            seed=rng_seed,
        )

    # ---- A: Cross-condition RMSE matrix ----
    print("\n=== A: Cross-condition RMSE matrix ===")
    cross_out = out_root / "cross_prediction"
    cross_result = run_cross_prediction(
        log_samples, oed_cycles,
        out_dir=cross_out,
        dt=args.dt,
        n_draws=args.n_cross_draws,
        workers=args.workers,
        seed=rng_seed,
    )

    # Summary
    print("\n=== Summary: Cross-prediction median NRMSE ===")
    conds = cross_result["conditions"]
    mat   = cross_result["matrix_median_nrmse"]
    header = "            " + "  ".join(f"{c:>10}" for c in conds)
    print(header)
    for i, cond_train in enumerate(conds):
        row = "  ".join(f"{mat[i][j]:>10.4f}" for j in range(len(conds)))
        print(f"  {cond_train:<10}  {row}")
    print("\nDone.")


if __name__ == "__main__":
    main()
