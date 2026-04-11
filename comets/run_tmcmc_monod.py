"""
run_tmcmc_monod.py — Bayesian calibration of COMETS Monod parameters via TMCMC
================================================================================
Twin experiment: synthetic observations at OED-selected timepoints with
proportional noise.  TMCMC (Ching & Chen 2007) in log-parameter space
recovers posterior over 10 Monod kinetic parameters.

Parameters calibrated
---------------------
theta[0:5]  = mu_max per species  (So, An, Vp, Fn, Pg)
theta[5:10] = Km_primary per species

Usage
-----
    python comets/run_tmcmc_monod.py \\
        --condition diseased \\
        --n_particles 500 \\
        --n_mcmc_steps 3 \\
        --workers 12 \\
        --out comets/pipeline_results

PBS: see run_tmcmc_monod.sh
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

# ---------------------------------------------------------------------------
# Prior: uniform in log-space (= log-uniform in theta-space)
# ---------------------------------------------------------------------------
PRIOR_LB_FRAC = 0.15   # theta >= theta_nominal * 0.15
PRIOR_UB_FRAC = 6.0    # theta <= theta_nominal * 6.0

LOG_LB = np.log(THETA_NOMINAL * PRIOR_LB_FRAC)
LOG_UB = np.log(THETA_NOMINAL * PRIOR_UB_FRAC)


def _log_prior_batch(log_thetas: np.ndarray) -> np.ndarray:
    """Vectorised: returns 0.0 or -inf per row."""
    in_bounds = np.all((log_thetas >= LOG_LB) & (log_thetas <= LOG_UB), axis=1)
    return np.where(in_bounds, 0.0, -np.inf)


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(LOG_LB, LOG_UB, size=(n, N_PARAMS))


# ---------------------------------------------------------------------------
# Log-likelihood  (picklable class for multiprocessing)
# ---------------------------------------------------------------------------

class LogLikelihood:
    """
    Gaussian likelihood with proportional noise.

    sigma_s(t) = noise_cv * |y_obs_s(t)|
    log L(theta) = -0.5 * sum[ (X_pred - y_obs)^2 / sigma^2 + log(2*pi*sigma^2) ]
    """

    def __init__(
        self,
        y_obs: np.ndarray,
        selected_cycles: list[int],
        condition: str,
        max_cycles: int,
        dt: float,
        noise_cv: float,
    ):
        self.y_obs = y_obs               # (n_obs, 5)
        self.sel = selected_cycles
        self.condition = condition
        self.max_cycles = max_cycles
        self.dt = dt
        self.sig2 = (noise_cv * np.maximum(np.abs(y_obs), 1e-12)) ** 2
        self.log_norm = 0.5 * np.sum(np.log(2.0 * np.pi * self.sig2))

    def __call__(self, log_theta: np.ndarray) -> float:
        if np.any(log_theta < LOG_LB) or np.any(log_theta > LOG_UB):
            return -np.inf
        try:
            theta = np.exp(log_theta)
            monod = _apply_theta(theta)
            bm = _simulate(monod, self.condition, self.max_cycles, self.dt)
            bm_pred = bm[self.sel, :]               # (n_obs, 5)
            ll = (-0.5 * np.sum((bm_pred - self.y_obs) ** 2 / self.sig2)
                  - self.log_norm)
            return float(ll)
        except Exception:
            return -1e30


# ---------------------------------------------------------------------------
# Twin experiment
# ---------------------------------------------------------------------------

def make_observations(
    condition: str,
    selected_cycles: list[int],
    max_cycles: int,
    dt: float,
    noise_cv: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate with THETA_NOMINAL, add proportional noise at selected_cycles.

    Returns
    -------
    y_obs    : (n_obs, 5) noisy observations
    bm_truth : (max_cycles, 5) noiseless truth trajectory
    """
    rng = np.random.default_rng(seed)
    bm_truth = _simulate(_apply_theta(THETA_NOMINAL), condition, max_cycles, dt)
    y_obs = bm_truth[selected_cycles, :].copy()
    y_obs *= 1.0 + noise_cv * rng.standard_normal(y_obs.shape)
    y_obs = np.maximum(y_obs, 1e-15)
    return y_obs, bm_truth


# ---------------------------------------------------------------------------
# TMCMC helpers
# ---------------------------------------------------------------------------

def _find_dbeta(log_likes: np.ndarray, beta: float, cov_target: float) -> float:
    """Binary search for Delta_beta such that CoV(w_i = L^Delta_beta) <= cov_target."""
    def _cov(db: float) -> float:
        lw = db * log_likes - (db * log_likes).max()
        w = np.exp(lw)
        s = w.sum()
        if s <= 0.0:
            return np.inf
        w /= s
        return float(np.std(w) / (w.mean() + 1e-300))

    if _cov(1.0 - beta) <= cov_target:
        return 1.0 - beta           # single step to beta=1

    lo, hi = 0.0, 1.0 - beta
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        (lo if _cov(mid) <= cov_target else hi).__class__   # trick to avoid if
        if _cov(mid) <= cov_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break
    return max(lo, 1e-6)


def _batch_eval(log_like_fn, particles: np.ndarray, pool) -> np.ndarray:
    """Evaluate log_like for each row of particles, parallel if pool given."""
    if pool is not None:
        return np.array(pool.map(log_like_fn, particles))
    return np.array([log_like_fn(p) for p in particles])


def _mcmc_step(
    particles: np.ndarray,
    log_likes: np.ndarray,
    log_like_fn,
    pool,
    n_steps: int,
    beta: float,
    cov_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """n_steps MH random-walk steps in log-space; proposals evaluated in parallel."""
    N, d = particles.shape

    # Proposal covariance from ensemble
    C = np.cov(particles.T) * cov_scale ** 2
    try:
        L_chol = np.linalg.cholesky(C + 1e-10 * np.eye(d))
    except np.linalg.LinAlgError:
        L_chol = np.diag(np.sqrt(np.maximum(np.diag(C), 1e-12)))

    lp_cur = _log_prior_batch(particles)
    n_accept = 0

    for _ in range(n_steps):
        proposals = particles + rng.standard_normal((N, d)) @ L_chol.T
        new_ll = _batch_eval(log_like_fn, proposals, pool)
        lp_prop = _log_prior_batch(proposals)

        log_alpha = beta * (new_ll - log_likes) + (lp_prop - lp_cur)
        accept = np.log(rng.uniform(size=N)) < log_alpha
        particles = np.where(accept[:, None], proposals, particles)
        log_likes = np.where(accept, new_ll, log_likes)
        lp_cur = np.where(accept, lp_prop, lp_cur)
        n_accept += int(accept.sum())

    return particles, log_likes, n_accept / (N * n_steps)


# ---------------------------------------------------------------------------
# TMCMC main
# ---------------------------------------------------------------------------

def run_tmcmc(
    log_like_fn: LogLikelihood,
    n_particles: int = 500,
    n_mcmc_steps: int = 3,
    cov_target: float = 1.0,
    cov_scale: float = 0.4,
    seed: int = 42,
    n_workers: int = 1,
    verbose: bool = True,
) -> dict:
    """
    TMCMC sampler (Ching & Chen 2007).

    Returns
    -------
    dict:  theta_samples, log_theta_samples, log_likes,
           betas, log_evidence, stage_info
    """
    rng = np.random.default_rng(seed)
    pool = Pool(n_workers) if n_workers > 1 else None

    try:
        # Stage 0: sample from prior
        particles = sample_prior(n_particles, rng)
        t0 = time.time()
        log_likes = _batch_eval(log_like_fn, particles, pool)
        if verbose:
            frac = np.isfinite(log_likes).mean()
            print(f"  Stage  0 | β=0.0000  prior eval {time.time()-t0:.1f}s  "
                  f"finite={frac:.2f}")

        beta = 0.0
        stage_info: list[dict] = []
        log_evidence = 0.0

        stage = 1
        while beta < 1.0 - 1e-8:
            t0 = time.time()

            # Replace -inf with very negative value for weight computation
            ll_safe = np.where(np.isfinite(log_likes), log_likes, -1e300)

            dbeta = _find_dbeta(ll_safe, beta, cov_target)
            dbeta = min(dbeta, 1.0 - beta)
            beta_new = beta + dbeta

            # Importance weights
            lw = dbeta * ll_safe
            lw -= lw.max()
            w = np.exp(lw)
            log_evidence += float(np.log(w.mean() + 1e-300))
            w_norm = w / (w.sum() + 1e-300)

            # Resample
            idx = rng.choice(n_particles, size=n_particles, p=w_norm)
            particles = particles[idx]
            log_likes = log_likes[idx]

            # MCMC perturbation
            particles, log_likes, ar = _mcmc_step(
                particles, log_likes, log_like_fn, pool,
                n_mcmc_steps, beta_new, cov_scale, rng,
            )

            beta = beta_new
            dt_s = time.time() - t0
            info = {
                "stage": stage,
                "beta": float(beta),
                "dbeta": float(dbeta),
                "accept_rate": float(ar),
                "log_evidence_cum": float(log_evidence),
                "dt_s": float(dt_s),
            }
            stage_info.append(info)

            if verbose:
                print(f"  Stage {stage:3d} | β={beta:.4f}  Δβ={dbeta:.5f}  "
                      f"AR={ar:.3f}  logEv={log_evidence:.2f}  dt={dt_s:.1f}s")

            stage += 1

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    theta_samples = np.exp(particles)
    return {
        "log_theta_samples": particles,
        "theta_samples": theta_samples,
        "log_likes": log_likes,
        "betas": [s["beta"] for s in stage_info],
        "log_evidence": float(log_evidence),
        "stage_info": stage_info,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

SP_COLORS = {
    "So": "#1f77b4", "An": "#ff7f0e", "Vp": "#2ca02c",
    "Fn": "#d62728", "Pg": "#9467bd",
}
_PARAM_COLORS = (
    [SP_COLORS[sp] for sp in SPECIES_ORDER]
    + [SP_COLORS[sp] for sp in SPECIES_ORDER]
)


def plot_recovery(
    theta_samples: np.ndarray,
    out_path: Path,
    condition: str,
) -> None:
    """2×5 grid of posterior histograms; true value and 90% CI highlighted."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()

    n_covered = 0
    for k, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
        samp = theta_samples[:, k]
        true_val = THETA_NOMINAL[k]
        q5, q50, q95 = np.percentile(samp, [5, 50, 95])
        covered = (q5 <= true_val <= q95)
        n_covered += int(covered)

        bins = np.linspace(samp.min(), samp.max(), 35)
        ax.hist(samp, bins=bins, color=_PARAM_COLORS[k], alpha=0.72, density=True)
        ax.axvline(true_val, color="red", lw=2.0, label="true")
        ax.axvline(q50, color="navy", lw=1.5, ls="--", label="median")
        ax.axvspan(q5, q95, alpha=0.14, color="gray", label="90% CI")
        ax.set_title(f"{name}\ntrue={true_val:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#f0fff0" if covered else "#fff0f0")

    axes[0].legend(fontsize=7)
    fig.suptitle(
        f"TMCMC posterior recovery — {condition}  "
        f"(90% CI covers {n_covered}/{N_PARAMS})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_stage_diagnostics(stage_info: list[dict], out_path: Path, condition: str) -> None:
    """beta schedule, acceptance rate, and per-stage runtime."""
    stages = [s["stage"] for s in stage_info]
    betas = [s["beta"] for s in stage_info]
    ars = [s["accept_rate"] for s in stage_info]
    dts = [s["dt_s"] for s in stage_info]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(stages, betas, "o-", color="#2ca02c", ms=5)
    axes[0].axhline(1.0, ls="--", color="gray", lw=1)
    axes[0].set(xlabel="Stage", ylabel="β", title="Temperature schedule")

    axes[1].plot(stages, ars, "s-", color="#d62728", ms=5)
    axes[1].axhline(0.234, ls="--", color="gray", lw=1, label="optimal 0.234")
    axes[1].set(xlabel="Stage", ylabel="Acceptance rate", title="MCMC acceptance")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1)

    axes[2].plot(stages, dts, "^-", color="#9467bd", ms=5)
    axes[2].set(xlabel="Stage", ylabel="Wall time (s)", title="Per-stage runtime")

    fig.suptitle(f"TMCMC diagnostics — {condition}", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    result: dict,
    out_dir: Path,
    condition: str,
    selected_cycles: list[int],
    selected_times_h: list[float],
    noise_cv: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "samples.npz",
        theta_samples=result["theta_samples"],
        log_theta_samples=result["log_theta_samples"],
        log_likes=result["log_likes"],
        theta_nominal=THETA_NOMINAL,
        param_names=np.array(PARAM_NAMES),
    )

    q5, q50, q95 = np.percentile(result["theta_samples"], [5, 50, 95], axis=0)
    summary = {
        "condition": condition,
        "n_params": N_PARAMS,
        "n_particles": int(len(result["theta_samples"])),
        "n_stages": len(result["stage_info"]),
        "log_evidence": float(result["log_evidence"]),
        "noise_cv": float(noise_cv),
        "selected_cycles": list(selected_cycles),
        "selected_times_h": list(selected_times_h),
        "posterior_summary": [
            {
                "name": PARAM_NAMES[k],
                "true": float(THETA_NOMINAL[k]),
                "q5": float(q5[k]),
                "q50": float(q50[k]),
                "q95": float(q95[k]),
                "covered": bool(q5[k] <= THETA_NOMINAL[k] <= q95[k]),
            }
            for k in range(N_PARAMS)
        ],
        "stage_info": result["stage_info"],
    }

    json_path = out_dir / "tmcmc_result.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {json_path}")

    n_covered = sum(s["covered"] for s in summary["posterior_summary"])
    print(f"  90% CI coverage: {n_covered}/{N_PARAMS}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="TMCMC Bayesian calibration of COMETS Monod parameters"
    )
    ap.add_argument("--condition", default="diseased", choices=["healthy", "diseased"])
    ap.add_argument("--n_particles", type=int, default=500)
    ap.add_argument("--n_mcmc_steps", type=int, default=3)
    ap.add_argument("--cov_scale", type=float, default=0.4,
                    help="Proposal covariance scale factor")
    ap.add_argument("--noise_cv", type=float, default=0.10,
                    help="Proportional noise CV for twin experiment")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers (Pool)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dt", type=float, default=0.01,
                    help="Time step (h) — must match OED")
    ap.add_argument("--out", type=str, default="comets/pipeline_results")
    ap.add_argument("--oed_json", type=str, default=None,
                    help="Override path to OED result JSON")
    args = ap.parse_args()

    out_root = Path(args.out)

    # Locate OED result JSON
    if args.oed_json:
        oed_path = Path(args.oed_json)
    elif args.condition == "healthy":
        oed_path = out_root / "oed_healthy" / "oed_result.json"
    else:
        oed_path = out_root / "oed_result.json"

    if oed_path.exists():
        with open(oed_path) as f:
            oed = json.load(f)
        selected_cycles: list[int] = oed["selected_cycles"]
        selected_times_h: list[float] = oed["selected_times_h"]
        print(f"  OED timepoints: {[f'{t:.1f}h' for t in selected_times_h]}")
    else:
        print(f"  OED result not found at {oed_path} — using fallback timepoints")
        max_cyc_fallback = 6000
        selected_cycles = list(range(800, max_cyc_fallback, 1000))[:6]
        selected_times_h = [c * args.dt for c in selected_cycles]

    max_cycles = max(selected_cycles) + 1   # only simulate up to last observation

    out_dir = out_root / f"tmcmc_monod_{args.condition}"

    print(f"\n=== TMCMC Monod calibration: {args.condition} ===")
    print(f"  n_particles={args.n_particles}  n_mcmc_steps={args.n_mcmc_steps}  "
          f"workers={args.workers}  seed={args.seed}")
    print(f"  max_cycles={max_cycles} ({max_cycles * args.dt:.1f} h)  "
          f"n_obs={len(selected_cycles)}  noise_cv={args.noise_cv}")

    # Twin experiment data
    y_obs, bm_truth = make_observations(
        args.condition, selected_cycles, max_cycles, args.dt,
        args.noise_cv, seed=0,
    )
    print(f"  y_obs shape: {y_obs.shape}  (n_obs × 5 species)")

    # Log-likelihood object
    log_like_fn = LogLikelihood(
        y_obs, selected_cycles, args.condition, max_cycles, args.dt, args.noise_cv,
    )

    # Run TMCMC
    t_start = time.time()
    result = run_tmcmc(
        log_like_fn,
        n_particles=args.n_particles,
        n_mcmc_steps=args.n_mcmc_steps,
        cov_target=1.0,
        cov_scale=args.cov_scale,
        seed=args.seed,
        n_workers=args.workers,
        verbose=True,
    )
    elapsed = time.time() - t_start
    n_stages = len(result["stage_info"])
    print(f"\n  Done: {elapsed:.1f}s ({elapsed/60:.1f} min), {n_stages} stages, "
          f"log_evidence={result['log_evidence']:.2f}")

    # Coverage report
    theta_s = result["theta_samples"]
    q5, q50, q95 = np.percentile(theta_s, [5, 50, 95], axis=0)
    for k in range(N_PARAMS):
        covered = q5[k] <= THETA_NOMINAL[k] <= q95[k]
        sym = "ok" if covered else "!!"
        print(f"  [{sym}] {PARAM_NAMES[k]:15s}  true={THETA_NOMINAL[k]:.4f}  "
              f"[{q5[k]:.4f}, {q95[k]:.4f}]")

    # Save and plot
    save_results(
        result, out_dir, args.condition,
        selected_cycles, selected_times_h, args.noise_cv,
    )
    plot_recovery(theta_s, out_dir / "recovery_plot.png", args.condition)
    plot_stage_diagnostics(
        result["stage_info"], out_dir / "stage_diagnostics.png", args.condition,
    )


if __name__ == "__main__":
    main()
