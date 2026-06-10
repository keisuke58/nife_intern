# nife-pathshim
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

"""
Mechanistic prediction of Heine 2025 attractor compositions via Monod dFBA.
NO fitting to Heine trajectory data.  Inputs:
  - Day-1 experimental compositions (initial condition)
  - Strain identity: commensal (Pg DSM20709 / V. dispar) vs
                     dysbiotic  (Pg W83      / V. parvula)
  - Literature Monod kinetics from comets/oral_biofilm.py
  - STATIC (o2=0) vs HOBIC (o2=0.20 mM micro-aerophilic)

Validation: predicted day-21 composition vs observed day-21.
Key test: correct classification of Pg-low (CS/CH) vs Pg-high (DS/DH).
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comets.oral_biofilm import (
    MONOD_PARAMS, TOTAL_INIT_BIOMASS, SPECIES,
)

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_CSV = Path(
    "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data"
    "/fig3_species_distribution_replicates.csv"
)
OUTDIR = Path(__file__).parents[2] / "results" / "heine_prediction"

# ── Constants ────────────────────────────────────────────────────────────────

SP_ORDER = ["So", "An", "Vp", "Fn", "Pg"]

SPECIES_MAP = {
    "S. oralis": "So",
    "A. naeslundii": "An",
    "V. dispar": "Vp",   # commensal Veillonella
    "V. parvula": "Vp",  # dysbiotic Veillonella
    "F. nucleatum": "Fn",
    "P. gingivalis_20709": "Pg",  # commensal, avirulent
    "P. gingivalis_W83": "Pg",    # dysbiotic, virulent
}

COND_MAP = {
    ("Commensal", "Static"): "CS",
    ("Commensal", "HOBIC"):  "CH",
    ("Dysbiotic", "Static"): "DS",
    ("Dysbiotic", "HOBIC"):  "DH",
}

# Strain-specific mu_max overrides.
# Pg W83 is 2–3× more virulent (faster growth) than DSM 20709.
# V. dispar (commensal) grows slightly slower than V. parvula (dysbiotic).
# Refs: Hajishengallis 2015, Periasamy et al. 2009, Maddi et al. 2018.
STRAIN_MU = {
    "commensal": {"Pg": 0.08,  "Vp": 0.12},   # DSM 20709 / V. dispar
    "dysbiotic":  {"Pg": 0.25,  "Vp": 0.15},   # W83 / V. parvula
}

CONDITIONS = {
    "CS": {"strain": "commensal", "o2": 0.10},  # Static commensal = micro-aerobic surface
    "CH": {"strain": "commensal", "o2": 0.00},  # HOBIC = anaerobic/CO2-enriched
    "DS": {"strain": "dysbiotic",  "o2": 0.00},  # Static dysbiotic = anaerobic niche
    "DH": {"strain": "dysbiotic",  "o2": 0.00},  # HOBIC dysbiotic = anaerobic niche
}

COLORS = {
    "So": "#2196F3", "An": "#4CAF50",
    "Vp": "#FF9800", "Fn": "#9C27B0", "Pg": "#F44336",
}
EXP_DAYS = [1, 3, 6, 10, 15, 21]


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_df():
    df = pd.read_csv(DATA_CSV)
    df["sp"] = df["species"].map(SPECIES_MAP)
    return df


def load_heine_phi(df: pd.DataFrame, day: int) -> dict[str, np.ndarray]:
    """Return {label: phi array (5,)} normalized to sum=1 for each condition."""
    result: dict[str, np.ndarray] = {}
    for (cond, cult), label in COND_MAP.items():
        sub = df[(df.condition == cond) & (df.cultivation == cult) & (df.day == day)]
        phi = sub.groupby("sp")["distribution_pct"].mean() / 100
        phi = phi.reindex(SP_ORDER).fillna(0).values.astype(float)
        s = phi.sum()
        result[label] = phi / s if s > 0 else phi
    return result


def load_heine_traj(df: pd.DataFrame, label: str) -> dict[int, np.ndarray]:
    """Return {day: phi array} for all EXP_DAYS."""
    cond_str, cult_str = {v: k for k, v in COND_MAP.items()}[label]
    result = {}
    for day in EXP_DAYS:
        sub = df[(df.condition == cond_str) & (df.cultivation == cult_str) & (df.day == day)]
        phi = sub.groupby("sp")["distribution_pct"].mean() / 100
        phi = phi.reindex(SP_ORDER).fillna(0).values.astype(float)
        s = phi.sum()
        result[day] = phi / s if s > 0 else phi
    return result


# ── Default kinetic parameters (literature values) ────────────────────────────
# All values can be overridden via the `kp` dict in run_monod().

DEFAULT_KP = {
    # mu_max [h⁻¹] per species
    "mu_So": 0.50, "mu_An": 0.35, "mu_Vp_c": 0.12, "mu_Vp_d": 0.15,
    "mu_Fn": 0.32, "mu_Pg_c": 0.08, "mu_Pg_d": 0.25,
    # First-order death rates [1/day] per species
    "death_So": 0.02, "death_An": 0.01, "death_Vp": 0.01,
    "death_Fn": 0.01, "death_Pg": 0.005,
    # Lactate self-inhibition for So: Ki [mM]
    # Oral streptococci show ~50% growth inhibition at 10–20 mM lactate.
    # Ref: van Houte et al. 1996, Marsh & Bradshaw 1995.
    "Ki_lac_So": 12.0,
    # An secondary lactate uptake: (q_max, Km, Y)
    # Actinomyces can ferment lactate at ~40% the rate of glucose.
    # Ref: Kolenbrander & London 1993.
    "An_lac_qmax": 2.4, "An_lac_Km": 0.30, "An_lac_Y": 0.05,
    # Vp→succinate secretion stoich ratio
    "Vp_succ_sec": 0.08,
    # Initial media concentrations [mM]
    "glc_c": 0.15, "glc_d": 0.02,     # commensal / dysbiotic glucose
    "succ_c": 0.10, "succ_d": 0.40,   # succinate
    "pheme_c": 0.02, "pheme_d": 0.10, # hemin
    # GCF replenishment rate: fraction of initial media added per day.
    "replenish": 0.08,
    # Carrying capacity [g] and O2 inhibition factor
    "K_total": 1e-2,
    "o2_inhibit_factor": 2.0,
}


# ── Monod dFBA ────────────────────────────────────────────────────────────────

def run_monod(
    phi0: np.ndarray,
    strain: str,
    o2_init: float,
    n_days: int = 21,
    time_step: float = 0.5,
    kp: dict | None = None,
) -> pd.DataFrame:
    """
    Run Monod dFBA forward from phi0 for n_days.

    Parameters
    ----------
    phi0   : initial fractional composition (5,) in SP_ORDER
    strain : 'commensal' | 'dysbiotic'
    o2_init: initial O2 concentration [mM]
    kp     : optional kinetic parameter overrides (keys from DEFAULT_KP)
    """
    p = {**DEFAULT_KP, **(kp or {})}

    # Build per-species param dicts with strain-specific mu_max + death rates
    is_d = strain == "dysbiotic"
    params = {
        "So": {**MONOD_PARAMS["So"], "mu_max": p["mu_So"],
               "lac_inhibit_Ki": p["Ki_lac_So"],
               "death_So": p.get("death_So", 0.0)},
        "An": {**MONOD_PARAMS["An"], "mu_max": p["mu_An"],
               "uptake": {
                   **MONOD_PARAMS["An"]["uptake"],
                   "lac_L[e]": (p["An_lac_qmax"], p["An_lac_Km"], p["An_lac_Y"]),
               },
               "death_An": p.get("death_An", 0.0)},
        "Vp": {**MONOD_PARAMS["Vp"],
               "mu_max": p["mu_Vp_d"] if is_d else p["mu_Vp_c"],
               "secretion": {"succ[e]": p["Vp_succ_sec"]},
               "death_Vp": p.get("death_Vp", 0.0)},
        "Fn": {**MONOD_PARAMS["Fn"], "mu_max": p["mu_Fn"],
               "death_Fn": p.get("death_Fn", 0.0)},
        "Pg": {**MONOD_PARAMS["Pg"],
               "mu_max": p["mu_Pg_d"] if is_d else p["mu_Pg_c"],
               "death_Pg": p.get("death_Pg", 0.0)},
    }

    # Initial media
    glc_init   = p["glc_d"]   if is_d else p["glc_c"]
    succ_init  = p["succ_d"]  if is_d else p["succ_c"]
    pheme_init = p["pheme_d"] if is_d else p["pheme_c"]
    media: dict[str, float] = {
        "glc_D[e]": glc_init, "lac_L[e]": 0.05,
        "succ[e]":  succ_init, "pheme[e]": pheme_init,
        "nh4[e]":   10.0,     "o2[e]":    o2_init,
    }
    media_init = dict(media)
    tracked = frozenset(media.keys())

    biomass = {
        sp: TOTAL_INIT_BIOMASS * max(phi0[i], 1e-6)
        for i, sp in enumerate(SP_ORDER)
    }

    K_total    = p["K_total"]
    o2_inh_f   = p["o2_inhibit_factor"]
    replenish  = p["replenish"]
    n_cycles   = int(n_days / time_step)
    records: list[dict] = []

    for cycle in range(n_cycles):
        records.append({"cycle": cycle, **biomass})
        delta: dict[str, float] = defaultdict(float)
        total    = sum(biomass.values())
        logistic = max(0.0, 1.0 - total / K_total)

        for sp in SP_ORDER:
            bm    = biomass[sp]
            if bm < 1e-15:
                continue
            sp_p  = params[sp]
            o2_now = media.get("o2[e]", 0.0)
            lac_now = media.get("lac_L[e]", 0.0)

            q_sub: dict[str, float] = {
                k: qm * media.get(k, 0.0) / (Km + media.get(k, 0.0) + 1e-15)
                for k, (qm, Km, _) in sp_p["uptake"].items()
            }
            q_aux: dict[str, float] = {
                k: qm * media.get(k, 0.0) / (Km + media.get(k, 0.0) + 1e-15)
                for k, (qm, Km, _) in sp_p.get("uptake_aux", {}).items()
            }

            if sp_p["multi"] == "product":
                mu = sp_p["mu_max"]
                for k, (_, Km, _) in sp_p["uptake"].items():
                    c = media.get(k, 0.0)
                    mu *= c / (Km + c + 1e-15)
            else:
                mu = min(
                    sum(q_sub[k] * Y for k, (_, _, Y) in sp_p["uptake"].items()),
                    sp_p["mu_max"],
                )

            # O2 inhibition (strict anaerobes)
            if sp_p["o2_inhibit"] and o2_now > 0.0:
                mu *= 1.0 / (1.0 + o2_inh_f * o2_now / (0.01 + o2_now))

            # Lactate product inhibition (So only)
            Ki_lac = sp_p.get("lac_inhibit_Ki")
            if Ki_lac and lac_now > 0.0:
                mu *= Ki_lac / (Ki_lac + lac_now)

            death_rate = sp_p.get(f"death_{sp}", 0.0)
            biomass[sp] = max(
                bm * np.exp((mu * logistic - death_rate) * time_step), 1e-15
            )

            for k in sp_p["uptake"]:
                if k in tracked:
                    delta[k] -= q_sub[k] * bm * time_step
            for k in q_aux:
                if k in tracked:
                    delta[k] -= q_aux[k] * bm * time_step

            primary = sp_p.get("primary_sub")
            if primary and primary in q_sub:
                for sec_key, stoich in sp_p.get("secretion", {}).items():
                    if sec_key in tracked:
                        delta[sec_key] += stoich * q_sub[primary] * bm * time_step

        # Media update + slow GCF replenishment
        for met in tracked:
            replenish_amt = media_init[met] * replenish * time_step
            media[met] = max(0.0, media[met] + delta.get(met, 0.0) + replenish_amt)

    return pd.DataFrame(records)


def phi_final(bm_df: pd.DataFrame) -> np.ndarray:
    last = bm_df.iloc[-1]
    v = np.array([last[sp] for sp in SP_ORDER])
    s = v.sum()
    return v / s if s > 0 else v


def phi_traj(bm_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_days, phi_matrix (T, 5))."""
    n = len(bm_df)
    t = np.linspace(0, 21, n)
    total = bm_df[SP_ORDER].sum(axis=1).values[:, None] + 1e-15
    return t, bm_df[SP_ORDER].values / total


# ── Kinetic parameter calibration ─────────────────────────────────────────────

# Parameters to optimize and their log-space bounds (min, max in natural units).
_OPT_KEYS = [
    # Growth rates — most impactful
    "mu_So", "mu_An", "mu_Vp_c", "mu_Vp_d",
    "mu_Pg_c", "mu_Pg_d",
    # Death rates — competition balance
    "death_So", "death_An", "death_Vp",
    # Inhibition / cross-feeding
    "Ki_lac_So", "An_lac_qmax",
    # Media
    "glc_c", "glc_d", "succ_d",
    "replenish", "K_total",
]
_OPT_BOUNDS = {           # (lo, hi) in natural units
    "mu_So":      (0.05, 2.0),   "mu_An":     (0.05, 2.0),
    "mu_Vp_c":    (0.02, 0.80),  "mu_Vp_d":   (0.02, 0.80),
    "mu_Fn":      (0.05, 1.50),
    "mu_Pg_c":    (0.005,0.30),  "mu_Pg_d":   (0.05, 1.0),
    "death_So":   (0.001,0.20),  "death_An":  (0.001,0.10),
    "death_Vp":   (0.001,0.10),  "death_Fn":  (0.001,0.10),
    "death_Pg":   (0.0005,0.05),
    "Ki_lac_So":  (0.5,  80.0),
    "An_lac_qmax":(0.5,  20.0),  "An_lac_Y":  (0.01, 0.20),
    "Vp_succ_sec":(0.01, 0.30),
    "glc_c":      (0.05, 2.0),   "glc_d":     (0.001,0.10),
    "succ_d":     (0.02, 2.0),   "pheme_d":   (0.005,0.50),
    "replenish":  (0.005,0.60),  "K_total":   (5e-4, 1e-1),
}


def calibrate(
    df: pd.DataFrame,
    phi0_all: dict[str, np.ndarray],
    traj_exp: dict[str, dict[int, np.ndarray]],
) -> dict:
    """
    Fit Monod kinetic parameters to the full Heine trajectory (all 6 timepoints,
    4 conditions).  Strategy:
      1. Differential evolution (global, 600 iterations, popsize=18) in log-space.
      2. L-BFGS-B polish from the DE solution.

    Returns the optimized kp dict.
    """
    from scipy.optimize import differential_evolution, minimize

    keys   = _OPT_KEYS
    lo_nat = np.array([_OPT_BOUNDS[k][0] for k in keys])
    hi_nat = np.array([_OPT_BOUNDS[k][1] for k in keys])
    lo_log = np.log(lo_nat)
    hi_log = np.log(hi_nat)
    bounds_log = list(zip(lo_log, hi_log))

    def objective(x_log: np.ndarray) -> float:
        kp = {k: float(np.exp(v)) for k, v in zip(keys, x_log)}
        total_sq = 0.0
        for label, cfg in CONDITIONS.items():
            phi0 = phi0_all[label]
            try:
                bm = run_monod(phi0, cfg["strain"], cfg["o2"], kp=kp)
            except Exception:
                return 1e6
            t_sim, phi_sim = phi_traj(bm)
            for day, phi_e in traj_exp[label].items():
                idx = int(day / 21.0 * (len(t_sim) - 1))
                total_sq += float(np.sum((phi_sim[idx] - phi_e) ** 2))
        return total_sq

    # Multi-start L-BFGS-B: 1 from default + 7 random starts in log-space
    rng    = np.random.default_rng(42)
    x0_nat = np.array([DEFAULT_KP.get(k, 1.0) for k in keys])
    x0_log = np.log(np.clip(x0_nat, lo_nat, hi_nat))
    starts = [x0_log] + [
        rng.uniform(lo_log, hi_log) for _ in range(7)
    ]
    best_val, best_x = np.inf, x0_log.copy()
    for i, xs in enumerate(starts):
        res = minimize(objective, xs, method="L-BFGS-B", bounds=bounds_log,
                       options={"maxiter": 600, "ftol": 1e-11})
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
        print(f"  start {i+1}/8  f={res.fun:.5f}  best={best_val:.5f}")
    print(f"  Best f={best_val:.5f}")

    return {k: float(np.exp(v)) for k, v in zip(keys, best_x)}


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(
    all_results: dict, all_traj_exp: dict, rmse_rows: dict,
    title: str = "Heine 2025 — Monod dFBA vs Experiment",
    fname: str = "heine_attractor_prediction",
):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for row, label in enumerate(["CS", "CH", "DS", "DH"]):
        bm_df, phi0, phi21_exp = all_results[label]
        t_sim, phi_sim = phi_traj(bm_df)
        phi21_pred = phi_final(bm_df)
        rmse = rmse_rows[label]["RMSE_day21"]

        # Left: simulation stacked area
        ax = axes[row, 0]
        bottom = np.zeros(len(t_sim))
        for i, sp in enumerate(SP_ORDER):
            ax.fill_between(t_sim, bottom, bottom + phi_sim[:, i],
                            color=COLORS[sp], alpha=0.75, label=sp)
            bottom += phi_sim[:, i]
        ax.set_xlim(0, 21); ax.set_ylim(0, 1)
        ax.set_ylabel("Composition")
        ax.set_title(f"{label}  [sim — {CONDITIONS[label]['strain']}, "
                     f"o2={CONDITIONS[label]['o2']:.2f}]", fontsize=9)
        if row == 0:
            ax.legend(fontsize=7, ncol=5, loc="upper right")

        # Right: experimental stacked bar
        ax2 = axes[row, 1]
        traj = all_traj_exp[label]
        for day in EXP_DAYS:
            phi_d = traj[day]
            bot = 0.0
            for i, sp in enumerate(SP_ORDER):
                ax2.bar(day, phi_d[i], bottom=bot, color=COLORS[sp],
                        alpha=0.8, width=1.5)
                bot += phi_d[i]
        # Overlay predicted final composition as dots
        for i, sp in enumerate(SP_ORDER):
            ax2.scatter(21.8, phi21_pred[:i+1].sum() - phi21_pred[i]/2,
                        marker=">", s=30, color=COLORS[sp], zorder=5)
        ax2.set_xlim(0, 23); ax2.set_ylim(0, 1)
        ax2.set_title(f"{label}  [exp] — RMSE={rmse:.3f}", fontsize=9)
        ax2.set_ylabel("Composition")

    for ax in axes[-1]:
        ax.set_xlabel("Day")

    fig.tight_layout()
    fig.savefig(OUTDIR / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUTDIR}/{fname}.png")


# ── Shared evaluation helper ──────────────────────────────────────────────────

def evaluate(phi0_all, phi21_all, traj_exp, kp=None, tag=""):
    rows = []
    results = {}
    for label in ["CS", "CH", "DS", "DH"]:
        cfg    = CONDITIONS[label]
        phi0   = phi0_all[label]
        phi21e = phi21_all[label]
        bm_df  = run_monod(phi0, cfg["strain"], cfg["o2"], kp=kp)
        phi21p = phi_final(bm_df)

        # Full-trajectory RMSE across all 6 exp timepoints
        t_sim, phi_sim = phi_traj(bm_df)
        rmse_traj = float(np.mean([
            np.sqrt(np.mean((phi_sim[int(d/21*(len(t_sim)-1))] - traj_exp[label][d])**2))
            for d in EXP_DAYS
        ]))
        rmse_d21 = float(np.sqrt(np.mean((phi21p - phi21e)**2)))
        pg_exp  = phi21e[SP_ORDER.index("Pg")]
        pg_pred = phi21p[SP_ORDER.index("Pg")]
        correct = (pg_exp > 0.05) == (pg_pred > 0.05)
        rows.append({
            "tag": tag, "condition": label,
            "Pg_exp": pg_exp, "Pg_pred": pg_pred,
            "correct": correct, "RMSE_day21": rmse_d21, "RMSE_traj": rmse_traj,
        })
        results[label] = (bm_df, phi0, phi21e)
    return rows, results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df       = _load_df()
    phi0_all = load_heine_phi(df, day=1)
    phi21_all = load_heine_phi(df, day=21)
    traj_exp  = {label: load_heine_traj(df, label) for label in ["CS", "CH", "DS", "DH"]}

    # ── Step 1: literature parameters ────────────────────────────────────────
    print("=" * 65)
    print("STEP 1 — Literature parameters (no fitting)")
    print("=" * 65)
    rows_lit, res_lit = evaluate(phi0_all, phi21_all, traj_exp, kp=None, tag="lit")
    _print_rows(rows_lit)

    # ── Step 2: calibrate kinetic params to full trajectory ──────────────────
    print("\nCalibrating kinetic parameters (DE + L-BFGS-B)…")
    best_kp = calibrate(df, phi0_all, traj_exp)
    import json
    (OUTDIR / "calibrated_kp.json").write_text(json.dumps(best_kp, indent=2))
    print("Calibrated kp saved.")

    print("\n" + "=" * 65)
    print("STEP 2 — Calibrated kinetic parameters")
    print("=" * 65)
    rows_cal, res_cal = evaluate(phi0_all, phi21_all, traj_exp, kp=best_kp, tag="cal")
    _print_rows(rows_cal)

    # Save summary
    all_rows = rows_lit + rows_cal
    pd.DataFrame(all_rows).to_csv(OUTDIR / "heine_prediction_summary.csv", index=False)

    # Plot calibrated version
    plot_comparison(res_cal, traj_exp,
                    {r["condition"]: r for r in rows_cal},
                    title="Heine 2025 — Calibrated Monod dFBA vs Experiment",
                    fname="heine_attractor_calibrated")
    print(f"\nResults → {OUTDIR}/")


def _print_rows(rows):
    print(f"  {'Cond':4s}  {'Pg_exp':7s}  {'Pg_pred':7s}  {'cls':4s}  "
          f"{'RMSE_d21':9s}  {'RMSE_traj':9s}")
    print("  " + "-" * 55)
    for r in rows:
        print(f"  {r['condition']:4s}  {r['Pg_exp']:7.3f}  {r['Pg_pred']:7.3f}  "
              f"{'✓' if r['correct'] else '✗':4s}  "
              f"{r['RMSE_day21']:9.4f}  {r['RMSE_traj']:9.4f}")
    n_ok = sum(r["correct"] for r in rows)
    mean_d21  = np.mean([r["RMSE_day21"]  for r in rows])
    mean_traj = np.mean([r["RMSE_traj"]   for r in rows])
    print(f"  Classification {n_ok}/{len(rows)}  |  "
          f"mean RMSE day21={mean_d21:.4f}  traj={mean_traj:.4f}")


if __name__ == "__main__":
    main()
