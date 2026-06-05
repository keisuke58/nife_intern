#!/usr/bin/env python3
"""
heine_ode_continuous.py
=======================
Use the gLV MAP fit (Heine 2025 attractors) to produce a continuous-time ODE
trajectory for all four conditions (CS / CH / DS / DH) and overlay the discrete
replicate observations from fig3_species_distribution_replicates.csv.

The MAP fit gives A (5×5 asymm) + b (5) per condition.  We integrate on a fine
dt=0.05 day grid from day 1 to day 21 (initial condition = median composition on
day 1), yielding φ_i(t) for any t.

Outputs (default: results/heine2025/):
  fig_ode_continuous_4cond.{pdf,png}  — 4×5 panel grid
  fig_ode_continuous_<COND>.{pdf,png} — per-condition 1×5 panel
  ode_continuous_<COND>.csv           — long-format continuous trajectory

Run:
    python scripts/analysis/heine_ode_continuous.py
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

_HERE = Path(__file__).resolve().parents[2]

DATA_CSV  = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"
                 "/fig3_species_distribution_replicates.csv")
FIT_JSON  = _HERE / "results" / "heine2025" / "fit_glv_heine.json"
OUT_DIR   = _HERE / "results" / "heine2025"

DAYS_OBS  = [1, 3, 6, 10, 15, 21]

CONDITIONS = [
    ("Commensal", "Static", "CS"),
    ("Commensal", "HOBIC",  "CH"),
    ("Dysbiotic", "Static", "DS"),
    ("Dysbiotic", "HOBIC",  "DH"),
]
SPECIES_MAP = {
    "Commensal": ["S. oralis", "A. naeslundii", "V. dispar",  "F. nucleatum", "P. gingivalis_20709"],
    "Dysbiotic": ["S. oralis", "A. naeslundii", "V. parvula", "F. nucleatum", "P. gingivalis_W83"],
}
SHORT_NAMES = {
    "S. oralis":            "S. oralis",
    "A. naeslundii":        "A. naeslundii",
    "V. dispar":            "V. dispar",
    "V. parvula":           "V. parvula",
    "F. nucleatum":         "F. nucleatum",
    "P. gingivalis_20709":  "P. gingivalis (20709)",
    "P. gingivalis_W83":    "P. gingivalis (W83)",
}
SP_COLORS = {
    "S. oralis":            "#2ca02c",
    "A. naeslundii":        "#9467bd",
    "V. dispar":            "#1f77b4",
    "V. parvula":           "#1f77b4",
    "F. nucleatum":         "#ff7f0e",
    "P. gingivalis_20709":  "#d62728",
    "P. gingivalis_W83":    "#d62728",
}


# ── ODE helpers ───────────────────────────────────────────────────────────────

def replicator_rhs(t, phi, A, b):
    """gLV replicator: dφ_i/dt = φ_i (b_i + Σ A_ij φ_j − ⟨f⟩)."""
    phi = np.maximum(phi, 0.0)
    f   = A @ phi + b
    return phi * (f - phi @ f)


def integrate_phi(A, b, phi0, t_grid):
    sol = solve_ivp(
        replicator_rhs, (t_grid[0], t_grid[-1]), phi0,
        args=(A, b), t_eval=t_grid,
        method="Radau", rtol=1e-8, atol=1e-10,
    )
    phi_t = np.maximum(sol.y.T, 0.0)
    row_sums = phi_t.sum(axis=1, keepdims=True)
    return phi_t / np.where(row_sums > 0, row_sums, 1.0)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_phi_obs(df, condition_str, cultivation_str, species_list):
    """Return (n_days, n_sp) median phi from the replicate CSV, on DAYS_OBS."""
    mask = (df["condition"] == condition_str) & (df["cultivation"] == cultivation_str)
    sub  = df[mask]
    phi  = np.zeros((len(DAYS_OBS), len(species_list)))
    for j, sp in enumerate(species_list):
        sp_sub = sub[sub["species"] == sp]
        for i, day in enumerate(DAYS_OBS):
            vals = sp_sub[sp_sub["day"] == day]["distribution_pct"].values
            phi[i, j] = np.median(vals) / 100.0 if len(vals) > 0 else 0.0
    row_sums = phi.sum(axis=1, keepdims=True)
    return phi / np.where(row_sums > 0, row_sums, 1.0)


def load_phi_replicates(df, condition_str, cultivation_str, species_list):
    """Return dict[species] -> (n_replicates_per_day, n_days) arrays for scatter."""
    mask = (df["condition"] == condition_str) & (df["cultivation"] == cultivation_str)
    sub  = df[mask]
    result = {}
    for sp in species_list:
        sp_sub = sub[sub["species"] == sp]
        days_arr, vals_arr = [], []
        for day in DAYS_OBS:
            vals = sp_sub[sp_sub["day"] == day]["distribution_pct"].values / 100.0
            for v in vals:
                days_arr.append(day)
                vals_arr.append(v)
        result[sp] = (np.array(days_arr), np.array(vals_arr))
    return result


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csv(cond_key, t_grid, phi_t, species_list, out_dir):
    rows = []
    for k, t in enumerate(t_grid):
        for j, sp in enumerate(species_list):
            rows.append({"condition": cond_key, "t_day": t, "species": sp,
                         "phi": phi_t[k, j]})
    df = pd.DataFrame(rows)
    path = out_dir / f"ode_continuous_{cond_key}.csv"
    df.to_csv(path, index=False)
    print(f"  [{cond_key}] wrote {path.name}  ({len(df)} rows)")
    return df


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_condition_axes(axes_row, cond_key, t_grid, phi_t, species_list,
                         phi_obs, replicates, show_xlabel=True):
    """Fill one row of a subplot grid (one axis per species)."""
    for j, sp in enumerate(species_list):
        ax = axes_row[j]
        col = SP_COLORS[sp]

        # continuous ODE line
        ax.plot(t_grid, phi_t[:, j], color=col, lw=2.0, label="ODE")

        # replicate scatter
        days_r, vals_r = replicates[sp]
        ax.scatter(days_r, vals_r, color=col, s=16, alpha=0.55,
                   zorder=3, edgecolors="none", label="replicates")

        # median overlay
        ax.plot(DAYS_OBS, phi_obs[:, j], "o", color=col, ms=5,
                mec="black", mew=0.6, zorder=4, label="median")

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 22)
        ax.set_xticks([1, 6, 10, 15, 21])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(SHORT_NAMES[sp], fontsize=9, pad=3)
        if show_xlabel:
            ax.set_xlabel("day", fontsize=8)
        ax.tick_params(labelsize=7)

    axes_row[0].set_ylabel(f"{cond_key}\n" + r"$\varphi_i$", fontsize=9)


def make_per_condition_fig(cond_key, t_grid, phi_t, species_list,
                           phi_obs, replicates, out_dir):
    fig, axes = plt.subplots(1, len(species_list),
                             figsize=(3.0 * len(species_list), 3.0), sharey=True)
    _plot_condition_axes(axes, cond_key, t_grid, phi_t, species_list,
                         phi_obs, replicates, show_xlabel=True)
    # legend on first axis
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.6)
    fig.suptitle(
        f"gLV ODE continuous prediction vs Heine 2025 replicates — {cond_key}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"fig_ode_continuous_{cond_key}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{cond_key}] wrote fig_ode_continuous_{cond_key}.pdf/png")


def make_4cond_fig(results, out_dir):
    """4-row (conditions) × 5-col (species) summary figure."""
    n_sp   = 5
    n_cond = len(results)
    fig, all_axes = plt.subplots(
        n_cond, n_sp,
        figsize=(3.0 * n_sp, 2.8 * n_cond),
        sharey=True, sharex=True,
    )
    for row, (cond_key, data) in enumerate(results.items()):
        axes_row = all_axes[row]
        _plot_condition_axes(
            axes_row, cond_key,
            data["t_grid"], data["phi_t"], data["species"],
            data["phi_obs"], data["replicates"],
            show_xlabel=(row == n_cond - 1),
        )
    # column headers (species names) on top row
    for j, sp in enumerate(results[list(results.keys())[0]]["species"]):
        all_axes[0][j].set_title(SHORT_NAMES[sp], fontsize=9, pad=3)

    fig.suptitle(
        "gLV ODE continuous trajectories vs Heine 2025 replicates (CS / CH / DS / DH)",
        fontsize=12, y=1.002,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"fig_ode_continuous_4cond.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [all] wrote fig_ode_continuous_4cond.pdf/png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument("--conditions", nargs="+", default=["CS", "CH", "DS", "DH"],
                   choices=["CS", "CH", "DS", "DH"])
    p.add_argument("--dt",    type=float, default=0.05,
                   help="ODE time step [day] (default 0.05)")
    p.add_argument("--t-end", type=float, default=21.0,
                   help="Integration end time [day] (default 21)")
    p.add_argument("--fit",     type=Path, default=FIT_JSON)
    p.add_argument("--data",    type=Path, default=DATA_CSV)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fit = json.load(open(args.fit))
    df  = pd.read_csv(args.data)

    t_grid = np.arange(0.0, args.t_end + args.dt / 2, args.dt)
    # shift to start at day 1 (obs starts day 1, not day 0)
    t_grid = t_grid + 1.0
    print(f"Time grid: t ∈ [{t_grid[0]:.1f}, {t_grid[-1]:.1f}] day, "
          f"dt={args.dt} → {len(t_grid)} points")

    cond_lookup = {key: (cstr, cult) for cstr, cult, key in CONDITIONS}
    results = {}

    for cond_key in args.conditions:
        if cond_key not in fit:
            print(f"  [skip] {cond_key}: not in fit JSON")
            continue
        c          = fit[cond_key]
        A          = np.array(c["A"])
        b          = np.array(c["b"])
        species    = c["species"]
        cond_str, cult_str = cond_lookup[cond_key]
        sp_list_full = SPECIES_MAP["Commensal" if cond_str == "Commensal" else "Dysbiotic"]

        phi_obs   = load_phi_obs(df, cond_str, cult_str, sp_list_full)
        replicates = load_phi_replicates(df, cond_str, cult_str, sp_list_full)

        phi0 = phi_obs[0]        # day-1 median as initial condition
        phi_t = integrate_phi(A, b, phi0, t_grid)

        print(f"\n[{cond_key}] RMSE(MAP)={c['rmse']:.4f}")
        print(f"  phi0 (day 1 median): {dict(zip(species, np.round(phi0, 3)))}")

        save_csv(cond_key, t_grid, phi_t, sp_list_full, args.out_dir)
        make_per_condition_fig(cond_key, t_grid, phi_t, sp_list_full,
                               phi_obs, replicates, args.out_dir)

        results[cond_key] = {
            "t_grid":     t_grid,
            "phi_t":      phi_t,
            "phi_obs":    phi_obs,
            "replicates": replicates,
            "species":    sp_list_full,
        }

    if len(results) > 1:
        make_4cond_fig(results, args.out_dir)

    print("\ndone.")


if __name__ == "__main__":
    main()
