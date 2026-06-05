#!/usr/bin/env python3
"""
dieckow_ode_continuous.py
=========================
Take the *best Dieckow 5-species fit* (TMCMC 1000-particle posterior over a
shared symmetric A + per-patient b) and turn the 3-week discrete observations
into continuous-time per-patient predictions with calibrated uncertainty.

The fit produces a posterior over θ = (A_utri(15), b_A(5), …, b_L(5)) = (65,).
Each posterior sample, integrated forward from each patient's W1 observation,
gives one φ_i(t) trajectory.  Stack samples → median trajectory + 5/95 % CI
band.  Then derive several continuous-time quantities the 3-week sampling cannot
see directly:

  Case A — Posterior bundle
    Per-patient × per-genus φ_i(t) on a fine grid (every ~4 min of model time),
    with 90 % credible band from N posterior samples (default 200; --full-posterior
    uses all 1000).  Replaces the MAP point estimate with calibrated uncertainty.

  Case B — Forward extrapolation to steady state ("predicted basin")
    Integrate from W3 out to t=84 days (12 weeks) under each sample.  Final
    composition = patient-specific predicted attractor.  Compare patients to
    see which "basin" each one is heading into without imposing external Heine
    in-vitro centroids (the in-vitro / in-vivo offset is real — see
    ANALYSIS_NOTES §1).

  Case C — Cumulative pathogen exposure
    ∫₀²¹ φ_Pg(t) dt per patient (median ± 90 % CI from the bundle).  Continuous
    integral over the 3-week window — a sharper risk metric than the W3 snapshot
    alone, because it accumulates the whole trajectory.

Compute budget (CPU, NumPy fallback):
  10 patients × N samples × 3 weeks × 2500 steps each.
    N=10  (MAP-only):     ~1   s
    N=200 (default):      ~30  s
    N=1000 (--full):      ~2-3 min
On GPU with JAX vmap (cluster, fifawc/vancouver01), each is ~10x faster.

Outputs (default --out-dir results/dieckow_ode_continuous/):
  - traj_<patient>.csv             long-format trajectories with CI per genus
  - extrap_steady_state.csv        per-patient predicted steady state (mean ± CI)
  - cumulative_exposure.csv        per-patient ∫₀²¹ φ_Pg(t) dt and similar
  - fig_posterior_bundle.{pdf,png} 10-patient grid, 5 genera per panel
  - fig_extrap_steady.{pdf,png}    bar chart of predicted basins per patient
  - fig_cumulative_pg.{pdf,png}    ranked cumulative Pg exposure with CI
"""
import sys as _sys, pathlib as _pathlib  # noqa  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import canonical simulator + data loader from dieckow_hamilton_fit.
# These imports are the same ones the original fit script uses, so the
# integrated dynamics here are byte-identical to what produced the saved fit.
try:
    from scripts.analysis.dieckow_hamilton_fit import (
        build_obs_matrix,
        extract_patient_theta,
        simulate_patient,
        PATIENT_ORDER, GENERA_5, N_SP, N_STEPS, DT,
        C_CONST, ALPHA_CONST,
        N_A_PARAMS,
    )
    _HAS_FIT_MODULE = True
except Exception as _e:
    _HAS_FIT_MODULE = False
    _import_error = _e

# JAX simulate_0d_nsp (returns full trajectory) — needed for the fine grid.
try:
    import jax.numpy as jnp
    from hamilton_ode_jax_nsp import simulate_0d_nsp
    _HAS_JAX_SIM = True
except Exception as _e:
    _HAS_JAX_SIM = False
    _jax_error = _e

_HERE = Path(__file__).resolve().parents[2]
DEFAULT_FIT = _HERE / "results" / "dieckow_fits" / "fit_joint_5sp_1000p.json"
DEFAULT_OUT = _HERE / "results" / "dieckow_ode_continuous"

GENUS_COLORS = {
    "Streptococcus":  "#2ca02c",
    "Actinomyces":    "#9467bd",
    "Veillonella":    "#1f77b4",
    "Fusobacterium":  "#ff7f0e",
    "Porphyromonas":  "#d62728",
}


# ── Integration helpers ───────────────────────────────────────────────────────

def simulate_patient_continuous(theta_20, phi_init, n_weeks=3):
    """Integrate one patient for n_weeks and return the full fine-grid trajectory.

    Chains n_weeks calls to simulate_0d_nsp (each returns the full per-step
    trajectory), giving a (n_weeks * N_STEPS, N_SP) array. Model-time per week
    is N_STEPS * DT (= 0.25 model units); we map step → day linearly so the
    output day grid is [0, 21] with n_weeks * N_STEPS points (each ~4 min).
    """
    if not _HAS_JAX_SIM:
        raise RuntimeError(f"hamilton_ode_jax_nsp not importable: {_jax_error}")
    theta_jax = jnp.array(theta_20, dtype=jnp.float64)
    phi_cur = jnp.array(phi_init, dtype=jnp.float64)
    trajs = []
    for _ in range(n_weeks):
        traj = simulate_0d_nsp(
            theta_jax, n_sp=N_SP, n_steps=N_STEPS, dt=DT,
            phi_init=phi_cur,
            c_const=C_CONST, alpha_const=ALPHA_CONST,
        )
        traj_np = np.asarray(traj, dtype=np.float64)
        # Re-normalise per row (gLV replicator should preserve Σ=1, but float
        # error and clipping can drift).
        s = traj_np.sum(axis=1, keepdims=True)
        traj_np = traj_np / np.maximum(s, 1e-12)
        trajs.append(traj_np)
        # Continue from this week's final state.
        phi_cur = jnp.array(traj_np[-1])
    full = np.concatenate(trajs, axis=0)                       # (n_weeks*N_STEPS, N_SP)
    n_steps_total = full.shape[0]
    t_day = np.linspace(0.0, n_weeks * 7.0, n_steps_total)     # 1 week = 7 day
    return t_day, full


def subsample_samples(samples, n, rng):
    """Take a random subsample of posterior samples (no replacement)."""
    samples = np.asarray(samples)
    if n >= len(samples):
        return samples
    idx = rng.choice(len(samples), size=n, replace=False)
    return samples[idx]


def bundle_per_patient(theta_samples, phi_init, n_weeks=3):
    """Integrate every sample for one patient. Returns (n_samples, T, N_SP)."""
    out = []
    t_day = None
    for theta in theta_samples:
        theta_p = np.concatenate([theta[:N_A_PARAMS], theta[-N_SP:]])  # placeholder
        # NB: caller passes already-extracted per-patient 20-vector via theta_samples,
        # so just use the raw theta_20 here.
        t, traj = simulate_patient_continuous(theta, phi_init, n_weeks=n_weeks)
        t_day = t
        out.append(traj)
    return t_day, np.stack(out, axis=0)


def extract_patient_samples(samples, patient_idx):
    """From joint (n_samples, 65) → per-patient (n_samples, 20) = [A_utri(15), b(5)]."""
    samples = np.asarray(samples)
    A_utri = samples[:, :N_A_PARAMS]
    b_start = N_A_PARAMS + patient_idx * N_SP
    b = samples[:, b_start: b_start + N_SP]
    return np.concatenate([A_utri, b], axis=1)


# ── Case A: posterior bundle, per patient ────────────────────────────────────

def case_A_posterior_bundles(samples, obs, n_weeks, out_dir):
    """Per patient: trajectory bundle → CSV (median + CI) + summary figure."""
    print("\n[A] Posterior bundle trajectories")
    all_long = []
    bundles = {}   # patient -> (t_day, (n_samples, T, N_SP))
    for p_idx, patient in enumerate(PATIENT_ORDER):
        if patient not in obs:
            continue
        phi_obs = obs[patient]                          # (N_SP, weeks)
        phi_init = phi_obs[:, 0]
        theta_p_samples = extract_patient_samples(samples, p_idx)
        t_day, bundle = bundle_per_patient(theta_p_samples, phi_init, n_weeks=n_weeks)
        bundles[patient] = (t_day, bundle)
        med = np.median(bundle, axis=0)                 # (T, N_SP)
        q05 = np.percentile(bundle, 5,  axis=0)
        q95 = np.percentile(bundle, 95, axis=0)
        # long format
        rows = []
        for k, t in enumerate(t_day):
            for j, g in enumerate(GENERA_5):
                rows.append({
                    "patient": patient, "t_day": t, "genus": g,
                    "phi_median": med[k, j], "phi_q05": q05[k, j], "phi_q95": q95[k, j],
                })
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"traj_{patient}.csv", index=False)
        all_long.append(df)
        print(f"  patient {patient}: bundle ({bundle.shape[0]} samples × {bundle.shape[1]} steps)"
              f" → traj_{patient}.csv")

    # 10-patient grid figure
    n_pat = len(bundles)
    ncol = 5
    nrow = (n_pat + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax_idx, (patient, (t_day, bundle)) in enumerate(bundles.items()):
        ax = axes[ax_idx // ncol, ax_idx % ncol]
        med = np.median(bundle, axis=0)
        q05 = np.percentile(bundle, 5,  axis=0)
        q95 = np.percentile(bundle, 95, axis=0)
        phi_obs = obs[patient]
        weeks_obs = phi_obs.shape[1]
        days_obs = np.array([0.0, 7.0, 14.0][:weeks_obs])  # W1, W2, W3 day positions
        for j, g in enumerate(GENERA_5):
            col = GENUS_COLORS[g]
            ax.fill_between(t_day, q05[:, j], q95[:, j], color=col, alpha=0.18,
                            linewidth=0)
            ax.plot(t_day, med[:, j], color=col, lw=1.4, label=g if ax_idx == 0 else None)
            ax.plot(days_obs, phi_obs[j, :weeks_obs], "o", color=col, ms=4,
                    mec="black", mew=0.5)
        ax.set_title(f"Patient {patient}", fontsize=10)
        ax.set_xlim(0, n_weeks * 7)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 7, 14, 21])
    # Hide unused panels
    for k in range(len(bundles), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    # Common labels and legend
    for ax in axes[-1, :]:
        ax.set_xlabel("time [day]")
    for ax in axes[:, 0]:
        ax.set_ylabel("φ")
    handles = [plt.Line2D([0], [0], color=GENUS_COLORS[g], lw=2, label=g) for g in GENERA_5]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Dieckow 5-species posterior trajectory bundles "
                 f"(median + 5-95 % CI, N={bundle.shape[0]} samples)", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_dir / "fig_posterior_bundle.pdf")
    fig.savefig(out_dir / "fig_posterior_bundle.png", dpi=200)
    plt.close(fig)
    print(f"  wrote fig_posterior_bundle.{{pdf,png}}")
    return bundles


# ── Case B: forward extrapolation to predicted steady state ──────────────────

def case_B_extrapolate_to_steady(samples, obs, extrap_weeks, out_dir):
    """For each patient, integrate to t = extrap_weeks*7 days starting from W3.

    Returns per-patient distribution of the long-time composition (the
    "predicted basin"). Uses every sample (no further subsampling) so the
    uncertainty mass is fully propagated to the basin estimate.
    """
    print(f"\n[B] Forward extrapolation to t = {extrap_weeks*7} days (predicted basin)")
    rows = []
    basin_phi = {}
    for p_idx, patient in enumerate(PATIENT_ORDER):
        if patient not in obs:
            continue
        phi_obs = obs[patient]
        phi_init_W3 = phi_obs[:, -1]                    # start from last observed week
        theta_p_samples = extract_patient_samples(samples, p_idx)
        ss = []
        for theta in theta_p_samples:
            _, traj = simulate_patient_continuous(theta, phi_init_W3, n_weeks=extrap_weeks)
            ss.append(traj[-1])
        ss = np.stack(ss)                                # (n_samples, N_SP)
        basin_phi[patient] = ss
        med = np.median(ss, axis=0)
        q05 = np.percentile(ss, 5,  axis=0)
        q95 = np.percentile(ss, 95, axis=0)
        for j, g in enumerate(GENERA_5):
            rows.append({
                "patient": patient, "genus": g,
                "phi_ss_median": med[j], "phi_ss_q05": q05[j], "phi_ss_q95": q95[j],
            })
        print(f"  patient {patient}: steady-state median ="
              f" {dict(zip(GENERA_5, np.round(med, 3)))}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "extrap_steady_state.csv", index=False)

    # Stacked bar chart of predicted steady-state composition per patient
    patients = list(basin_phi)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(patients)), 4))
    bottoms = np.zeros(len(patients))
    for j, g in enumerate(GENERA_5):
        vals = np.array([np.median(basin_phi[p], axis=0)[j] for p in patients])
        ax.bar(patients, vals, bottom=bottoms, color=GENUS_COLORS[g], label=g,
               edgecolor="white", lw=0.5)
        bottoms += vals
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("φ (predicted steady state)")
    ax.set_xlabel("patient")
    ax.set_title(f"Forward-extrapolated steady state (median over posterior, "
                 f"t = {extrap_weeks*7} day)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5,
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_extrap_steady.pdf")
    fig.savefig(out_dir / "fig_extrap_steady.png", dpi=200)
    plt.close(fig)
    print(f"  wrote extrap_steady_state.csv + fig_extrap_steady.{{pdf,png}}")
    return basin_phi


# ── Case C: cumulative Pg exposure ───────────────────────────────────────────

def case_C_cumulative_exposure(bundles, out_dir):
    """∫₀²¹ φ_genus(t) dt per patient and per genus, with 90 % CI from bundles."""
    print("\n[C] Cumulative exposure (∫ φ_genus dt over the 3-week window)")
    rows = []
    for patient, (t_day, bundle) in bundles.items():
        # Restrict to the observed 3-week window [0, 21].
        m = t_day <= 21.0
        t = t_day[m]
        b = bundle[:, m, :]                              # (n_samples, T_obs, N_SP)
        # Per-sample trapezoidal integral.
        cum_per_sample = np.trapz(b, t, axis=1)          # (n_samples, N_SP)
        for j, g in enumerate(GENERA_5):
            c = cum_per_sample[:, j]
            rows.append({
                "patient": patient, "genus": g,
                "cum_median": float(np.median(c)),
                "cum_q05":    float(np.percentile(c, 5)),
                "cum_q95":    float(np.percentile(c, 95)),
                "cum_mean":   float(c.mean()),
                "cum_std":    float(c.std()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cumulative_exposure.csv", index=False)

    # Figure: ranked Pg cumulative exposure with CI
    pg = df[df.genus == "Porphyromonas"].sort_values("cum_median", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(pg)), 4))
    yerr = np.vstack([pg["cum_median"] - pg["cum_q05"], pg["cum_q95"] - pg["cum_median"]])
    ax.bar(pg["patient"], pg["cum_median"], yerr=yerr,
           color=GENUS_COLORS["Porphyromonas"], capsize=4,
           edgecolor="black", lw=0.6, alpha=0.85)
    ax.set_ylabel(r"$\int_0^{21}\,\varphi_{Pg}\,d t$  [day]")
    ax.set_xlabel("patient (ranked)")
    ax.set_title("Cumulative P. gingivalis exposure over 3 weeks (median ± 5–95 % CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cumulative_pg.pdf")
    fig.savefig(out_dir / "fig_cumulative_pg.png", dpi=200)
    plt.close(fig)
    print(f"  wrote cumulative_exposure.csv + fig_cumulative_pg.{{pdf,png}}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument("--fit", type=Path, default=DEFAULT_FIT,
                   help="Path to fit_joint_5sp_1000p.json.")
    p.add_argument("--n-samples", type=int, default=200,
                   help="Posterior subsample for case A (default 200; --full-posterior overrides).")
    p.add_argument("--full-posterior", action="store_true",
                   help="Use all posterior samples (~5x compute).")
    p.add_argument("--extrap-weeks", type=int, default=9,
                   help="Forward-extrapolation horizon for case B (weeks). Default 9 (≈ 84 day).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for posterior subsampling.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="Output directory.")
    p.add_argument("--skip-extrap", action="store_true",
                   help="Skip case B (the slow part if --full-posterior).")
    args = p.parse_args()

    if not _HAS_FIT_MODULE:
        raise RuntimeError(
            f"Could not import dieckow_hamilton_fit (needs JAX + cluster paths "
            f"to dieckow_taxonomy/). Original error: {_import_error}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit = json.load(open(args.fit))
    theta_map = np.asarray(fit["theta_map"])
    samples_all = np.asarray(fit["samples"])
    print(f"Loaded fit: theta_map shape={theta_map.shape},"
          f" samples shape={samples_all.shape}, RMSE={fit['rmse']:.4f}")

    rng = np.random.default_rng(args.seed)
    n_keep = samples_all.shape[0] if args.full_posterior else min(args.n_samples, samples_all.shape[0])
    samples = subsample_samples(samples_all, n_keep, rng)
    print(f"Using {len(samples)} posterior samples")

    obs = build_obs_matrix()
    print(f"Loaded observations for {len(obs)} patients")

    bundles = case_A_posterior_bundles(samples, obs, n_weeks=3, out_dir=args.out_dir)
    case_C_cumulative_exposure(bundles, out_dir=args.out_dir)

    if not args.skip_extrap:
        # Case B uses a smaller subsample by default to keep extrapolation cheap.
        ss_samples = subsample_samples(samples, min(100, len(samples)), rng)
        case_B_extrapolate_to_steady(ss_samples, obs,
                                     extrap_weeks=args.extrap_weeks,
                                     out_dir=args.out_dir)

    print("\ndone.")


if __name__ == "__main__":
    main()
