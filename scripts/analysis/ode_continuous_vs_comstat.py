#!/usr/bin/env python3
"""
ode_continuous_vs_comstat.py
============================
Use the gLV ODE MAP fit (Heine 2025 attractors) as a continuous-time interpolant
and combine it with the discrete COMSTAT (FISH .lif) measurements.

The gLV ODE produces phi_i(t) on any t we ask for; COMSTAT gives us
coverage_i(d) and biovolume_i(d) only on the days a FISH stack was acquired
(CH/DH: Day 1, 6, 10, 15, 21).  We can:

  (1) Densify time.  Integrate the ODE on a fine grid (default dt=0.05 day,
      421 points over 21 days) — this gives phi_i(t) for ANY t.
  (2) Predict biovolume continuously.  Linearly interpolate measured
      total_biovolume(d) across days, then multiply by phi_i(t):
          B_i_pred(t) = phi_i(t) * total_biovolume_interp(t)
      → continuous per-species biovolume curve (filled-in days included).
  (3) Spatial-occupancy ratio R_i(t) = coverage_i_interp(t) / phi_i(t).
      R > 1 : the species spreads thin over the surface (low volume, wide footprint).
      R < 1 : the species piles up in tall colonies (high volume, narrow footprint).
      Tracking R_i(t) tells us whether a species changes its spatial strategy
      over time — e.g. P. gingivalis falling R in DH = sinking into deep colonies.
  (4) Cumulative exposure ∫₀^t phi_i(τ) dτ and ∫₀^t coverage_i(τ) dτ.
      Useful for the OED / intervention-timing arguments.

Outputs (per condition, default CH and DH):
  - results/biofilmQ_comstat/ode_continuous_<COND>.csv  long format
  - results/biofilmQ_comstat/fig_ode_continuous_<COND>.{pdf,png}

This is fit-free at runtime (the gLV A,b were already estimated). One ODE solve
per condition, ~10 ms each; trivial compute.
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
from scipy.integrate import solve_ivp

# Heine 2025 → COMSTAT species-name mapping (Heine fit uses pretty names with
# spaces/strain tags; COMSTAT CSV uses the compact decoder labels).
HEINE_TO_COMSTAT = {
    "S. oralis":            "S.oralis",
    "A. naeslundii":        "A.naeslundii",
    "V. dispar":            "Vd/Vp",
    "F. nucleatum":         "F.nucleatum",
    "P. gingivalis_20709":  "P.gingivalis",
}

SPECIES_COLORS = {
    "S.oralis":     "#2ca02c",
    "A.naeslundii": "#9467bd",
    "Vd/Vp":        "#1f77b4",
    "F.nucleatum":  "#ff7f0e",
    "P.gingivalis": "#d62728",
}

_HERE = Path(__file__).resolve().parents[2]
FIT_PATH = _HERE / "results" / "heine2025" / "fit_glv_heine.json"
COMSTAT_CSV = _HERE / "results" / "biofilmQ_comstat" / "comstat_metrics.csv"
OUT_DIR = _HERE / "results" / "biofilmQ_comstat"


def replicator_rhs(t, phi, b, A):
    """gLV replicator form: dφ_i/dt = φ_i (b_i + Σ A_ij φ_j − ⟨f⟩)."""
    f = b + A @ phi
    return phi * (f - phi @ f)


def integrate_phi(A, b, phi0, t_grid):
    """Integrate the replicator ODE on the requested time grid (Radau, stiff-safe)."""
    sol = solve_ivp(
        replicator_rhs, (t_grid[0], t_grid[-1]), phi0, args=(b, A),
        method="Radau", t_eval=t_grid, rtol=1e-8, atol=1e-10,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")
    phi = np.clip(sol.y, 0.0, None)
    return phi / phi.sum(axis=0, keepdims=True)


def aggregate_comstat(df_cond):
    """Per-(day, species) median + IQR for coverage and biovolume.

    Median is used (matches comstat_stats.csv Mann-Whitney convention) and is
    robust to FOV outliers. Returns one row per (day, species).
    """
    g = df_cond.groupby(["day", "species"])
    out = g.agg(
        coverage_med=("coverage", "median"),
        coverage_q25=("coverage", lambda s: np.percentile(s, 25)),
        coverage_q75=("coverage", lambda s: np.percentile(s, 75)),
        biovol_med=("biovolume_um3", "median"),
        biovol_q25=("biovolume_um3", lambda s: np.percentile(s, 25)),
        biovol_q75=("biovolume_um3", lambda s: np.percentile(s, 75)),
        n_fov=("coverage", "size"),
    ).reset_index()
    return out


def interp_per_species(agg, species_list, t_grid):
    """Linear-interpolate median coverage & biovolume on the fine grid (per species).

    Coverage is per-species; total_biovolume is per-day (sum over species in the
    median sense → use sum of per-species medians).  Outside observed range we
    clip-extrapolate (hold first/last value) rather than extrapolate linearly.
    """
    out = {}
    days_obs = sorted(agg["day"].unique())
    # total biovolume at each observed day = sum of per-species median biovolume
    tot_biovol_obs = np.array([
        agg[agg.day == d]["biovol_med"].sum() for d in days_obs
    ])
    tot_biovol_t = np.interp(t_grid, days_obs, tot_biovol_obs,
                             left=tot_biovol_obs[0], right=tot_biovol_obs[-1])
    out["total_biovol_t"] = tot_biovol_t
    out["days_obs"] = np.array(days_obs)
    out["tot_biovol_obs"] = tot_biovol_obs

    for sp in species_list:
        sub = agg[agg.species == sp].sort_values("day")
        if sub.empty:
            out[sp] = None
            continue
        d = sub["day"].to_numpy()
        cov = sub["coverage_med"].to_numpy()
        bv = sub["biovol_med"].to_numpy()
        cov_t = np.interp(t_grid, d, cov, left=cov[0], right=cov[-1])
        bv_t = np.interp(t_grid, d, bv, left=bv[0], right=bv[-1])
        out[sp] = {
            "days_obs":      d,
            "coverage_obs":  cov,
            "biovol_obs":    bv,
            "coverage_q25":  sub["coverage_q25"].to_numpy(),
            "coverage_q75":  sub["coverage_q75"].to_numpy(),
            "biovol_q25":    sub["biovol_q25"].to_numpy(),
            "biovol_q75":    sub["biovol_q75"].to_numpy(),
            "n_fov":         sub["n_fov"].to_numpy(),
            "coverage_t":    cov_t,
            "biovol_t":      bv_t,
        }
    return out


def build_long_table(cond, t_grid, phi_t, heine_species, interp):
    """Produce the long-format CSV: one row per (t, species)."""
    rows = []
    for i, sp_h in enumerate(heine_species):
        sp_c = HEINE_TO_COMSTAT[sp_h]
        m = interp.get(sp_c)
        phi = phi_t[i]
        # cumulative ∫₀^t phi dτ via trapezoidal
        cum_phi = np.concatenate([[0.0], np.cumsum(0.5 * (phi[:-1] + phi[1:]) * np.diff(t_grid))])
        if m is None:
            cov_t = np.full_like(phi, np.nan)
            bv_t = np.full_like(phi, np.nan)
            R = np.full_like(phi, np.nan)
            cum_cov = np.full_like(phi, np.nan)
        else:
            cov_t = m["coverage_t"]
            bv_t = m["biovol_t"]
            # R = coverage / phi; guard small phi
            R = np.where(phi > 1e-4, cov_t / phi, np.nan)
            cum_cov = np.concatenate([[0.0], np.cumsum(0.5 * (cov_t[:-1] + cov_t[1:]) * np.diff(t_grid))])
        # predicted continuous biovolume: phi(t) * total_biovolume(t)
        bv_pred = phi * interp["total_biovol_t"]
        for k, t in enumerate(t_grid):
            rows.append({
                "condition":           cond,
                "t_day":               t,
                "species":             sp_c,
                "phi_ode":             phi[k],
                "coverage_interp":     cov_t[k],
                "biovol_interp_um3":   bv_t[k],
                "biovol_pred_um3":     bv_pred[k],
                "R_spatial":           R[k],
                "phi_integral":        cum_phi[k],
                "coverage_integral":   cum_cov[k],
            })
    return pd.DataFrame(rows)


def make_figure(cond, t_grid, phi_t, heine_species, interp, out_path_pdf, out_path_png):
    """3-column figure: φ(t)+coverage scatter | R_i(t) | cumulative ∫ φ dτ."""
    n_sp = len(heine_species)
    fig, axes = plt.subplots(
        n_sp, 3, figsize=(12.5, 2.0 * n_sp),
        sharex=True, gridspec_kw={"hspace": 0.25, "wspace": 0.30},
    )

    for i, sp_h in enumerate(heine_species):
        sp_c = HEINE_TO_COMSTAT[sp_h]
        col = SPECIES_COLORS.get(sp_c, "#444444")
        phi = phi_t[i]
        m = interp.get(sp_c)

        # ── Col 1: phi(t) ODE + COMSTAT coverage scatter
        ax = axes[i, 0]
        ax.plot(t_grid, phi, color=col, lw=2.0, label=r"$\varphi_i(t)$ ODE")
        if m is not None:
            ax2 = ax.twinx()
            yerr = np.vstack([m["coverage_obs"] - m["coverage_q25"],
                              m["coverage_q75"] - m["coverage_obs"]])
            ax2.errorbar(m["days_obs"], m["coverage_obs"], yerr=yerr,
                         fmt="o", ms=4, color=col, alpha=0.55,
                         mec="black", mew=0.4, capsize=2, label="coverage (FISH)")
            ax2.set_ylabel("coverage", fontsize=8, color=col)
            ax2.tick_params(axis="y", labelsize=7, colors=col)
            ax2.set_ylim(bottom=0)
        ax.set_ylabel(sp_c + r"  $\varphi$", fontsize=9)
        ax.set_ylim(0, max(0.05, phi.max() * 1.15))
        if i == 0:
            ax.set_title(r"$\varphi_i(t)$ (ODE) + coverage (FISH)", fontsize=10)

        # ── Col 2: R_i(t) = coverage / phi
        ax = axes[i, 1]
        if m is not None:
            R = np.where(phi > 1e-3, m["coverage_t"] / phi, np.nan)
            ax.plot(t_grid, R, color=col, lw=2.0)
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_ylim(0.05, 50)
        ax.set_ylabel(r"$R_i = $cov$/\varphi$", fontsize=9)
        if i == 0:
            ax.set_title(r"Spatial-occupancy ratio $R_i(t)$", fontsize=10)

        # ── Col 3: cumulative ∫ phi dτ
        ax = axes[i, 2]
        cum_phi = np.concatenate([[0.0], np.cumsum(0.5 * (phi[:-1] + phi[1:]) * np.diff(t_grid))])
        ax.plot(t_grid, cum_phi, color=col, lw=2.0)
        ax.fill_between(t_grid, 0, cum_phi, color=col, alpha=0.15)
        ax.set_ylabel(r"$\int_0^t \varphi_i\,d\tau$", fontsize=9)
        if i == 0:
            ax.set_title(r"Cumulative exposure", fontsize=10)

    for ax in axes[-1, :]:
        ax.set_xlabel("time [day]", fontsize=10)

    fig.suptitle(
        f"gLV ODE continuous prediction × COMSTAT discrete observations — {cond}",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path_pdf)
    fig.savefig(out_path_png, dpi=200)
    plt.close(fig)


def run_condition(cond, fit, comstat_df, t_grid, out_dir):
    if cond not in fit:
        print(f"  [skip] {cond}: not in fit JSON")
        return
    sub = comstat_df[comstat_df["condition"] == cond]
    if sub.empty:
        print(f"  [skip] {cond}: no COMSTAT rows")
        return
    c = fit[cond]
    A = np.array(c["A"])
    b = np.array(c["b"])
    heine_species = c["species"]
    n = len(heine_species)
    # initial composition: median over Day-1 FOVs, per Heine species order
    day1 = sub[sub["day"] == 1].groupby("species")["biovolume_um3"].median()
    cov1 = sub[sub["day"] == 1].groupby("species")["coverage"].median()
    # use coverage at Day 1 as a proxy for composition init (then normalise);
    # gLV is composition, not biovolume, so coverage_obs/Σcov is the right proxy.
    phi0_comstat = np.array([cov1.get(HEINE_TO_COMSTAT[sp], 1.0 / n) for sp in heine_species])
    phi0 = phi0_comstat / phi0_comstat.sum()
    print(f"  [{cond}] phi0 (from Day 1 coverage):", dict(zip(heine_species, np.round(phi0, 3))))

    phi_t = integrate_phi(A, b, phi0, t_grid)        # (n_species, n_time)
    agg = aggregate_comstat(sub)
    interp = interp_per_species(agg, [HEINE_TO_COMSTAT[s] for s in heine_species], t_grid)

    df_long = build_long_table(cond, t_grid, phi_t, heine_species, interp)
    csv_path = out_dir / f"ode_continuous_{cond}.csv"
    df_long.to_csv(csv_path, index=False)
    print(f"  [{cond}] wrote {csv_path}  ({len(df_long)} rows)")

    pdf_path = out_dir / f"fig_ode_continuous_{cond}.pdf"
    png_path = out_dir / f"fig_ode_continuous_{cond}.png"
    make_figure(cond, t_grid, phi_t, heine_species, interp, pdf_path, png_path)
    print(f"  [{cond}] wrote {pdf_path.name} / {png_path.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument("--conditions", nargs="+", default=["CH", "DH"],
                   choices=["CS", "CH", "DS", "DH"],
                   help="Heine attractors to process (default: CH DH — those with COMSTAT data).")
    p.add_argument("--dt", type=float, default=0.05,
                   help="ODE time step [day] (default 0.05 → 421 points over 21 days).")
    p.add_argument("--t-end", type=float, default=21.0,
                   help="Integration end time [day] (default 21).")
    p.add_argument("--fit", type=Path, default=FIT_PATH,
                   help="Path to fit_glv_heine.json.")
    p.add_argument("--comstat", type=Path, default=COMSTAT_CSV,
                   help="Path to comstat_metrics.csv.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Output directory.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit = json.load(open(args.fit))
    comstat_df = pd.read_csv(args.comstat)

    t_grid = np.arange(0.0, args.t_end + args.dt / 2, args.dt)
    print(f"Time grid: t ∈ [0, {args.t_end}] day, dt={args.dt} → {len(t_grid)} points")
    print(f"Conditions: {args.conditions}")

    for cond in args.conditions:
        run_condition(cond, fit, comstat_df, t_grid, args.out_dir)

    print("done.")


if __name__ == "__main__":
    main()
