"""
run_oed.py — D-Optimal Experimental Design for COMETS Monod parameters
=======================================================================

Finds the measurement timepoints that maximise det(Fisher information matrix)
for 10 Monod kinetic parameters (μ_max × 5 + Km_primary × 5).

Usage
-----
    python run_oed.py [--condition healthy|diseased] [--n_obs 6] [--out DIR]

Outputs
-------
  pipeline_results/oed_sensitivity.png   — per-parameter sensitivity heatmap
  pipeline_results/oed_selected.png      — selected timepoints on DI trajectory
  pipeline_results/oed_result.json       — selected times, det(F), param names
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Monod parameters — mirrored from oral_biofilm.py (keep in sync)
# ---------------------------------------------------------------------------

SPECIES_ORDER = ["So", "An", "Vp", "Fn", "Pg"]

MONOD_PARAMS_BASE: dict[str, dict] = {
    "So": {
        "mu_max": 0.50,
        "uptake": {"glc_D[e]": (8.0, 0.05, 0.10)},
        "multi": "sum",
        "o2_inhibit": False,
        "secretion": {"lac_L[e]": 1.8},
        "primary_sub": "glc_D[e]",
    },
    "An": {
        "mu_max": 0.35,
        "uptake": {"glc_D[e]": (6.0, 0.08, 0.08)},
        "multi": "sum",
        "o2_inhibit": False,
        "secretion": {"lac_L[e]": 1.2, "succ[e]": 0.3},
        "primary_sub": "glc_D[e]",
    },
    "Vp": {
        "mu_max": 0.40,
        "uptake": {"lac_L[e]": (10.0, 0.15, 0.07)},
        "multi": "sum",
        "o2_inhibit": True,
        "secretion": {},
        "primary_sub": "lac_L[e]",
    },
    "Fn": {
        "mu_max": 0.32,
        "uptake": {"glc_D[e]": (4.0, 0.12, 0.09), "lac_L[e]": (5.0, 0.18, 0.08)},
        "multi": "sum",
        "o2_inhibit": True,
        "secretion": {},
        "primary_sub": "glc_D[e]",
    },
    "Pg": {
        "mu_max": 0.20,
        "uptake": {"succ[e]": (3.0, 0.08, 0.10), "pheme[e]": (0.5, 0.005, 0.12)},
        "multi": "product",
        "o2_inhibit": True,
        "secretion": {},
        "primary_sub": "succ[e]",
    },
}

MEDIA_HEALTHY = {
    "glc_D[e]": 0.20, "o2[e]": 0.50, "lac_L[e]": 0.05,
    "nh4[e]": 10.0, "pi[e]": 10.0, "h2o[e]": 1000.0, "ca2[e]": 2.0, "mg2[e]": 1.0,
}
MEDIA_DISEASED = {
    "glc_D[e]": 0.05, "lac_L[e]": 0.20, "succ[e]": 0.10, "pheme[e]": 0.50,
    "nh4[e]": 10.0, "pi[e]": 10.0, "h2o[e]": 1000.0, "ca2[e]": 2.0, "mg2[e]": 1.0,
}
INIT_FRACTIONS = {
    "healthy":  {"So": 0.40, "An": 0.20, "Vp": 0.20, "Fn": 0.15, "Pg": 0.05},
    "diseased": {"So": 0.10, "An": 0.10, "Vp": 0.10, "Fn": 0.35, "Pg": 0.35},
}
TOTAL_INIT_BIOMASS = 1e-4   # g

# Primary substrate key for each species (for Km parameterisation)
PRIMARY_SUB = {
    "So": "glc_D[e]",
    "An": "glc_D[e]",
    "Vp": "lac_L[e]",
    "Fn": "glc_D[e]",
    "Pg": "succ[e]",
}

# θ layout:  [0:5] = μ_max per species,  [5:10] = Km_primary per species
PARAM_NAMES = (
    ["μ_max_" + sp for sp in SPECIES_ORDER]
    + ["Km_" + sp for sp in SPECIES_ORDER]
)
N_PARAMS = len(PARAM_NAMES)   # 10

THETA_NOMINAL = np.array([
    MONOD_PARAMS_BASE[sp]["mu_max"] for sp in SPECIES_ORDER
] + [
    MONOD_PARAMS_BASE[sp]["uptake"][PRIMARY_SUB[sp]][1] for sp in SPECIES_ORDER
])

# ---------------------------------------------------------------------------
# Standalone Monod dFBA simulation
# ---------------------------------------------------------------------------

def _simulate(
    monod: dict,
    condition: str,
    max_cycles: int,
    dt: float,
    K_total: float = 0.01,
    o2_inhibit_factor: float = 2.0,
) -> np.ndarray:
    """Run dFBA with given monod parameters.

    Returns
    -------
    bm : ndarray shape (max_cycles, 5)  — biomass for each species
    """
    from collections import defaultdict

    media_init = MEDIA_HEALTHY if condition == "healthy" else MEDIA_DISEASED
    media = dict(media_init)
    tracked = frozenset(media.keys())

    fracs = INIT_FRACTIONS[condition]
    biomass = {sp: TOTAL_INIT_BIOMASS * fracs[sp] for sp in SPECIES_ORDER}

    bm = np.zeros((max_cycles, 5))

    for cycle in range(max_cycles):
        bm[cycle] = [biomass[sp] for sp in SPECIES_ORDER]

        delta_media: dict = defaultdict(float)
        total_bm = sum(biomass.values())
        logistic = max(0.0, 1.0 - total_bm / K_total)
        o2_now = media.get("o2[e]", 0.0)

        for sp_key in SPECIES_ORDER:
            bm_val = biomass[sp_key]
            if bm_val < 1e-15:
                continue
            p = monod[sp_key]

            q_sub: dict = {}
            for sub_key, (q_max, Km, Y) in p["uptake"].items():
                conc = media.get(sub_key, 0.0)
                q_sub[sub_key] = q_max * conc / (Km + conc + 1e-15)

            if p["multi"] == "product":
                mu = p["mu_max"]
                for sub_key, (q_max, Km, Y) in p["uptake"].items():
                    conc = media.get(sub_key, 0.0)
                    mu *= conc / (Km + conc + 1e-15)
            else:
                mu = sum(
                    q_sub[sub_key] * Y
                    for sub_key, (_, _, Y) in p["uptake"].items()
                )
                mu = min(mu, p["mu_max"])

            if p["o2_inhibit"] and o2_now > 0.0:
                mu *= 1.0 / (1.0 + o2_inhibit_factor * o2_now / (0.01 + o2_now))

            biomass[sp_key] = max(bm_val * np.exp(mu * dt * logistic), 1e-15)

            primary = p.get("primary_sub")
            for sub_key in p["uptake"]:
                if sub_key in tracked:
                    delta_media[sub_key] -= q_sub[sub_key] * bm_val * dt
            if primary and primary in q_sub:
                for sec_key, stoich in p.get("secretion", {}).items():
                    if sec_key in tracked:
                        delta_media[sec_key] += stoich * q_sub[primary] * bm_val * dt

        for met in tracked:
            media[met] = max(0.0, media[met] + delta_media.get(met, 0.0))

    return bm   # (max_cycles, 5)


def _apply_theta(theta: np.ndarray) -> dict:
    """Return deep-copied MONOD_PARAMS with θ applied."""
    p = copy.deepcopy(MONOD_PARAMS_BASE)
    for i, sp in enumerate(SPECIES_ORDER):
        p[sp]["mu_max"] = float(theta[i])
    for j, sp in enumerate(SPECIES_ORDER):
        sub_key = PRIMARY_SUB[sp]
        old = p[sp]["uptake"][sub_key]
        p[sp]["uptake"][sub_key] = (old[0], float(theta[5 + j]), old[2])
    return p


# ---------------------------------------------------------------------------
# Sensitivity and Fisher information
# ---------------------------------------------------------------------------

def compute_jacobian(
    theta: np.ndarray,
    condition: str,
    max_cycles: int,
    dt: float,
    delta_frac: float = 0.01,
) -> np.ndarray:
    """Finite-difference Jacobian.

    Returns
    -------
    J : ndarray shape (max_cycles, 5, N_PARAMS)
        J[t, s, k] = ∂X_s(t) / ∂θ_k
    """
    bm0 = _simulate(_apply_theta(theta), condition, max_cycles, dt)
    J = np.zeros((max_cycles, 5, N_PARAMS))
    for k in range(N_PARAMS):
        dtheta = theta.copy()
        delta = max(abs(theta[k]) * delta_frac, 1e-8)
        dtheta[k] += delta
        bm_k = _simulate(_apply_theta(dtheta), condition, max_cycles, dt)
        J[:, :, k] = (bm_k - bm0) / delta
    return J


def compute_fisher_at_times(J: np.ndarray, t_indices: list[int], noise_cv: float = 0.10) -> np.ndarray:
    """Build Fisher information matrix for a set of measurement times.

    Noise: proportional to biomass (coefficient of variation = noise_cv).
    F = Σ_t  J(t)^T · Σ^{-1} · J(t)
    where Σ = diag((noise_cv * X(t))^2).

    Parameters
    ----------
    J : (max_cycles, 5, N_PARAMS)
    t_indices : list of cycle indices to include
    """
    bm_ref = None  # will infer noise from J scale if needed
    F = np.zeros((N_PARAMS, N_PARAMS))
    for t in t_indices:
        Jt = J[t]        # (5, N_PARAMS)
        # noise variance: use absolute scale proxy σ² = 1 (unit-less sensitivity)
        # For proportional noise we'd need bm at t; approximate from J norms
        F += Jt.T @ Jt
    return F


def greedy_d_optimal(
    J: np.ndarray,
    bm_baseline: np.ndarray,
    t_candidates: np.ndarray,
    n_obs: int,
    noise_cv: float = 0.10,
) -> tuple[list[int], list[float]]:
    """Greedy D-optimal selection: add one timepoint at a time maximising det(F).

    Parameters
    ----------
    J : (max_cycles, 5, N_PARAMS) — Jacobian
    bm_baseline : (max_cycles, 5) — biomass for noise normalisation
    t_candidates : array of candidate cycle indices
    n_obs : number of timepoints to select

    Returns
    -------
    selected : list of cycle indices (length n_obs)
    logdets  : list of log(det(F)) after each addition
    """
    eps = 1e-30
    selected: list[int] = []
    logdets: list[float] = []
    F_cur = np.zeros((N_PARAMS, N_PARAMS))

    for _ in range(n_obs):
        best_t = None
        best_ld = -np.inf
        for t in t_candidates:
            if t in selected:
                continue
            Jt = J[t]            # (5, N_PARAMS)
            bm_t = bm_baseline[t]
            # proportional noise: σ_s = noise_cv * X_s  (floor 1e-8)
            sig2 = np.maximum(noise_cv * bm_t, 1e-8) ** 2
            weighted_Jt = Jt / sig2[:, None]  # (5, N_PARAMS)  — each row scaled
            F_new = F_cur + Jt.T @ weighted_Jt
            # log-det via eigenvalues (numerically stable)
            eigvals = np.linalg.eigvalsh(F_new)
            ld = np.sum(np.log(np.maximum(eigvals, eps)))
            if ld > best_ld:
                best_ld = ld
                best_t = t
                best_Jt = Jt
                best_wJt = weighted_Jt

        selected.append(best_t)
        logdets.append(best_ld)
        F_cur += best_Jt.T @ best_wJt

    return selected, logdets


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SP_COLORS = {"So": "#1f77b4", "An": "#ff7f0e", "Vp": "#2ca02c",
             "Fn": "#d62728", "Pg": "#9467bd"}


def _compute_di(bm: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy (DI) from (n, 5) biomass array."""
    total = bm.sum(axis=1, keepdims=True)
    phi = bm / np.maximum(total, 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(phi > 1e-15, phi * np.log(phi), 0.0).sum(axis=1)
    H_max = np.log(bm.shape[1])
    return H / H_max


def plot_sensitivity(J: np.ndarray, times_h: np.ndarray, out_path: Path) -> None:
    """Heatmap of |∂X/∂θ| per species aggregated by L2 norm over species."""
    # Aggregate over species dimension: sens[t, k] = ||J[t, :, k]||
    sens = np.linalg.norm(J, axis=1)   # (max_cycles, N_PARAMS)
    # Normalise each parameter by its own max for visibility
    sens_norm = sens / (sens.max(axis=0, keepdims=True) + 1e-30)

    # Subsample to ~200 timepoints for display
    step = max(1, len(times_h) // 200)
    t_disp = times_h[::step]
    s_disp = sens_norm[::step].T   # (N_PARAMS, n_disp)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    im = ax.imshow(s_disp, aspect="auto", origin="upper",
                   extent=[t_disp[0], t_disp[-1], N_PARAMS - 0.5, -0.5],
                   cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(N_PARAMS))
    ax.set_yticklabels(PARAM_NAMES, fontsize=9)
    ax.set_xlabel("Time (h)")
    ax.set_title("Normalised sensitivity |∂X/∂θ| — D-optimal timepoint selection")
    plt.colorbar(im, ax=ax, label="Norm. sensitivity")

    # Bottom panel: overall sensitivity (sum over params)
    total_sens = s_disp.sum(axis=0)
    axes[1].fill_between(t_disp, total_sens, alpha=0.7, color="steelblue")
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Σ sensitivity")
    axes[1].set_xlim(t_disp[0], t_disp[-1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_selected(
    bm_baseline: np.ndarray,
    di_baseline: np.ndarray,
    times_h: np.ndarray,
    selected_cycles: list[int],
    logdets: list[float],
    condition: str,
    out_path: Path,
) -> None:
    """Plot: DI trajectory + biomass + selected timepoints + det(F) curve."""
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # --- Panel A: DI + selected timepoints ---
    ax_di = fig.add_subplot(gs[0, 0])
    ax_di.plot(times_h, di_baseline, "k-", lw=1.5, label="DI")
    for rank, cyc in enumerate(selected_cycles):
        ax_di.axvline(times_h[cyc], color="red", lw=1.2,
                      alpha=0.8, ls="--", label=f"t*{rank+1}" if rank < 3 else None)
        ax_di.text(times_h[cyc], di_baseline.max() * 1.02,
                   f"#{rank+1}", ha="center", fontsize=7, color="red")
    ax_di.set_xlabel("Time (h)")
    ax_di.set_ylabel("DI")
    ax_di.set_title(f"DI trajectory ({condition})\nSelected timepoints (red dashed)")
    ax_di.legend(fontsize=7, ncol=2)

    # --- Panel B: Biomass timecourses ---
    ax_bm = fig.add_subplot(gs[0, 1])
    for i, sp in enumerate(SPECIES_ORDER):
        ax_bm.semilogy(times_h, bm_baseline[:, i], label=sp,
                       color=SP_COLORS[sp], lw=1.2)
    for cyc in selected_cycles:
        ax_bm.axvline(times_h[cyc], color="red", lw=0.8, alpha=0.5, ls="--")
    ax_bm.set_xlabel("Time (h)")
    ax_bm.set_ylabel("Biomass (g)")
    ax_bm.set_title("Species dynamics")
    ax_bm.legend(fontsize=8)

    # --- Panel C: det(F) vs number of observations ---
    ax_det = fig.add_subplot(gs[1, 0])
    n_sel = list(range(1, len(logdets) + 1))
    ax_det.plot(n_sel, logdets, "o-", color="navy", lw=1.5)
    ax_det.set_xlabel("Number of observations")
    ax_det.set_ylabel("log det(F)")
    ax_det.set_title("D-optimality gain (greedy)")
    ax_det.set_xticks(n_sel)

    # --- Panel D: Selected timepoints summary table ---
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.axis("off")
    rows = [[f"#{i+1}", f"{times_h[cyc]:.1f} h", f"{logdets[i]:.2f}"]
            for i, cyc in enumerate(selected_cycles)]
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=["Rank", "Time (h)", "log det(F)"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax_tbl.set_title("Optimal measurement times")

    fig.suptitle(
        f"OED: D-Optimal timepoints for Monod parameter identification\n"
        f"Condition: {condition}  |  {len(selected_cycles)} observations  |  {N_PARAMS} parameters",
        fontsize=11,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D-Optimal OED for COMETS Monod params")
    parser.add_argument("--condition", default="diseased", choices=["healthy", "diseased"])
    parser.add_argument("--n_obs", type=int, default=6, help="Number of timepoints to select")
    parser.add_argument("--max_cycles", type=int, default=8000)
    parser.add_argument("--dt", type=float, default=0.01, help="timestep in hours")
    parser.add_argument("--n_candidates", type=int, default=40,
                        help="Number of candidate timepoints (log-spaced)")
    parser.add_argument("--noise_cv", type=float, default=0.10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "pipeline_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_h = args.max_cycles * args.dt
    times_h = np.arange(args.max_cycles) * args.dt

    print(f"[OED] condition={args.condition}, max_cycles={args.max_cycles}, "
          f"dt={args.dt}h, total={total_h:.1f}h")

    # 1. Baseline simulation
    print("[OED] Running baseline simulation ...")
    bm_baseline = _simulate(
        copy.deepcopy(MONOD_PARAMS_BASE), args.condition, args.max_cycles, args.dt
    )
    di_baseline = _compute_di(bm_baseline)

    # 2. Compute Jacobian (finite difference)
    print(f"[OED] Computing Jacobian ({N_PARAMS} parameter perturbations) ...")
    J = compute_jacobian(THETA_NOMINAL, args.condition, args.max_cycles, args.dt)
    print(f"  J shape: {J.shape},  max |J|: {np.abs(J).max():.3e}")

    # 3. Candidate timepoints: log-spaced from ~1h to end (avoid t=0 where all equal)
    t_start_h = 0.5
    t_candidates_h = np.logspace(
        np.log10(t_start_h), np.log10(total_h * 0.98), args.n_candidates
    )
    t_candidates_idx = np.unique(
        np.clip((t_candidates_h / args.dt).astype(int), 0, args.max_cycles - 1)
    ).tolist()
    print(f"  Candidate timepoints: {len(t_candidates_idx)} "
          f"({times_h[t_candidates_idx[0]]:.1f}h — {times_h[t_candidates_idx[-1]]:.1f}h)")

    # 4. Greedy D-optimal selection
    print(f"[OED] Greedy D-optimal selection (n_obs={args.n_obs}) ...")
    selected_cycles, logdets = greedy_d_optimal(
        J, bm_baseline, t_candidates_idx, args.n_obs, noise_cv=args.noise_cv
    )
    selected_times_h = [float(times_h[c]) for c in selected_cycles]
    print(f"  Selected times (h): {[f'{t:.2f}' for t in selected_times_h]}")
    print(f"  log det(F) progression: {[f'{ld:.2f}' for ld in logdets]}")

    # 5. Plots
    print("[OED] Generating plots ...")
    plot_sensitivity(J, times_h, out_dir / "oed_sensitivity.png")
    plot_selected(
        bm_baseline, di_baseline, times_h,
        selected_cycles, logdets,
        args.condition, out_dir / "oed_selected.png",
    )

    # 6. Save result JSON
    result = {
        "condition": args.condition,
        "n_params": N_PARAMS,
        "param_names": PARAM_NAMES,
        "theta_nominal": THETA_NOMINAL.tolist(),
        "n_obs": args.n_obs,
        "selected_cycles": selected_cycles,
        "selected_times_h": selected_times_h,
        "logdet_F_progression": logdets,
        "logdet_F_final": logdets[-1] if logdets else None,
        "noise_cv": args.noise_cv,
    }
    json_path = out_dir / "oed_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path}")

    print("[OED] Done.")


if __name__ == "__main__":
    main()
