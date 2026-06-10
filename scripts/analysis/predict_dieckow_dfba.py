# nife-pathshim
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))   # allow sibling-script imports

"""
Cross-system prediction: Heine-calibrated Monod dFBA → Dieckow in-vivo validation.

Two prediction modes
---------------------
A) Personalised:   week-1 patient composition → simulate 14 days → predict week-3.
   Tests in vitro → in vivo transferability of calibrated kinetics.

B) Attractor-only: clinical label (CT1=commensal / CT2=dysbiotic) only, no patient
   initial composition.  Standard IC taken from Heine day-1 means.
   Tests whether strain identity alone predicts week-3 community state.

Input
-----
- results/dieckow_otu/phi_guild.npy        : (10, 3, 10) guild fractions
- results/dieckow_otu/community_types.json : CT1/CT2 labels per patient
- results/heine_prediction/calibrated_kp.json : Heine-calibrated Monod parameters
- Heine day-1 CSV (for Plan B standard ICs)

Output
------
- results/dieckow_prediction/planA_summary.csv
- results/dieckow_prediction/planB_summary.csv
- results/dieckow_prediction/dieckow_prediction.png
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comets.oral_biofilm import MONOD_PARAMS, TOTAL_INIT_BIOMASS

# Import shared Monod engine and helpers from Heine script
from predict_heine_attractors_dfba import (
    run_monod, phi_final, phi_traj,
    _load_df, load_heine_phi,
    CONDITIONS, SP_ORDER, DEFAULT_KP,
    COLORS,
)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parents[2]
PHI_NPY = ROOT / "results" / "dieckow_otu" / "phi_guild.npy"
CT_JSON = ROOT / "results" / "dieckow_otu" / "community_types.json"
KP_JSON = ROOT / "results" / "heine_prediction" / "calibrated_kp.json"
OUTDIR  = ROOT / "results" / "dieckow_prediction"

# ── Dieckow guild → 5-species mapping ────────────────────────────────────────
# Guild order from guild_replicator_dieckow.py: (0=Actinobacteria, 1=Bacilli,
# 2=Bacteroidia, 3=Betaproteobacteria, 4=Clostridia, 5=Coriobacteriia,
# 6=Fusobacteriia, 7=Gammaproteobacteria, 8=... 9=Negativicutes)
# Best-available mapping (coarse; Bacteroidia includes all Bacteroidia, not just Pg):
GUILD_TO_SP = {
    1: "So",   # Bacilli (Streptococcus spp.)
    0: "An",   # Actinobacteria (Actinomyces spp.)
    2: "Pg",   # Bacteroidia ← proxy for Porphyromonas/Prevotella
    6: "Fn",   # Fusobacteriia
    9: "Vp",   # Negativicutes (Veillonella spp.)
}

PATIENTS = list("ABCDEFGHKL")  # 10 patients, alphabetical


def load_dieckow() -> tuple[np.ndarray, dict, dict]:
    """Return (phi_guild (10,3,10), patient_ct dict, calibrated_kp dict)."""
    phi = np.load(PHI_NPY)
    ct  = json.loads(CT_JSON.read_text())["patient_ct"]
    kp  = json.loads(KP_JSON.read_text())
    return phi, ct, kp


def guild_to_5sp(phi_guild: np.ndarray) -> np.ndarray:
    """Extract 5 species from 10-guild array and renormalise to sum=1.

    SP_ORDER = ["So","An","Vp","Fn","Pg"]
    Guild indices: So=1(Bacilli), An=0(Actinobacteria), Vp=9(Negativicutes),
                   Fn=6(Fusobacteriia), Pg=2(Bacteroidia).
    """
    v = np.array([
        phi_guild[1],   # So
        phi_guild[0],   # An
        phi_guild[9],   # Vp
        phi_guild[6],   # Fn
        phi_guild[2],   # Pg
    ])
    s = v.sum()
    return v / s if s > 0 else v


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ── Plan A: personalised week-1 → week-3 prediction ─────────────────────────

def plan_a(phi, patient_ct, kp):
    rows = []
    for i, pat in enumerate(PATIENTS):
        ct  = patient_ct[pat]
        strain = "commensal" if ct == 1 else "dysbiotic"

        phi0_guild = phi[i, 0, :]   # week-1
        phi3_guild = phi[i, 2, :]   # week-3 (target)

        phi0 = guild_to_5sp(phi0_guild)   # initial condition
        phi3 = guild_to_5sp(phi3_guild)   # observed week-3

        bm  = run_monod(phi0, strain, o2_init=0.00, n_days=14, kp=kp)
        pred = phi_final(bm)

        rows.append({
            "patient": pat, "CT": ct, "strain": strain,
            **{f"obs_{sp}":  float(phi3[j])  for j, sp in enumerate(SP_ORDER)},
            **{f"pred_{sp}": float(pred[j]) for j, sp in enumerate(SP_ORDER)},
            "RMSE_wk3": rmse(pred, phi3),
            "Pg_obs": float(phi3[SP_ORDER.index("Pg")]),
            "Pg_pred": float(pred[SP_ORDER.index("Pg")]),
            "Pg_correct": (phi3[SP_ORDER.index("Pg")] > 0.05)
                          == (pred[SP_ORDER.index("Pg")] > 0.05),
        })
    return pd.DataFrame(rows)


# ── Plan B: attractor-only (strain label → standard IC → simulate) ───────────

def _heine_day1_ic() -> dict[str, np.ndarray]:
    """Load Heine day-1 mean compositions for commensal and dysbiotic conditions."""
    try:
        df = _load_df()
        phi0_all = load_heine_phi(df, day=1)
        # commensal → average of CS and CH; dysbiotic → average of DS and DH
        phi_com = (phi0_all["CS"] + phi0_all["CH"]) / 2
        phi_dys = (phi0_all["DS"] + phi0_all["DH"]) / 2
    except Exception:
        # Fallback: generic oral-biofilm initial states
        # So-dominant commensal vs Pg-enriched dysbiotic
        phi_com = np.array([0.60, 0.25, 0.08, 0.05, 0.02])   # So An Vp Fn Pg
        phi_dys = np.array([0.40, 0.15, 0.25, 0.10, 0.10])
    return {"commensal": phi_com, "dysbiotic": phi_dys}


def plan_b(phi, patient_ct, kp):
    std_ic = _heine_day1_ic()
    rows = []
    for i, pat in enumerate(PATIENTS):
        ct  = patient_ct[pat]
        strain = "commensal" if ct == 1 else "dysbiotic"

        phi3_guild = phi[i, 2, :]   # week-3 observed
        phi3 = guild_to_5sp(phi3_guild)

        phi0 = std_ic[strain]   # standard IC — no patient data used

        bm   = run_monod(phi0, strain, o2_init=0.00, n_days=21, kp=kp)
        pred = phi_final(bm)

        rows.append({
            "patient": pat, "CT": ct, "strain": strain,
            **{f"obs_{sp}":  float(phi3[j])  for j, sp in enumerate(SP_ORDER)},
            **{f"pred_{sp}": float(pred[j]) for j, sp in enumerate(SP_ORDER)},
            "RMSE_wk3": rmse(pred, phi3),
            "Pg_obs": float(phi3[SP_ORDER.index("Pg")]),
            "Pg_pred": float(pred[SP_ORDER.index("Pg")]),
            "Pg_correct": (phi3[SP_ORDER.index("Pg")] > 0.05)
                          == (pred[SP_ORDER.index("Pg")] > 0.05),
        })
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(df_a: pd.DataFrame, df_b: pd.DataFrame):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Dieckow in-vivo — Monod dFBA cross-system prediction\n"
                 "(kinetics calibrated on Heine 2025 in-vitro only)",
                 fontsize=10, fontweight="bold")

    x = np.arange(len(PATIENTS))
    width = 0.35

    for row_idx, (df, plan_label) in enumerate([(df_a, "Plan A: personalised"),
                                                (df_b, "Plan B: attractor-only")]):
        # Left: Pg comparison (observed vs predicted)
        ax = axes[row_idx, 0]
        ax.bar(x - width/2, df["Pg_obs"],  width, label="Observed wk-3",
               color="#ccc", edgecolor="k", linewidth=0.5)
        ax.bar(x + width/2, df["Pg_pred"], width, label="Predicted",
               color="#F44336", alpha=0.7, edgecolor="k", linewidth=0.5)
        for j, pat in enumerate(PATIENTS):
            ok = df.loc[df.patient == pat, "Pg_correct"].values[0]
            ax.text(j, max(df.loc[df.patient == pat, "Pg_obs"].values[0],
                            df.loc[df.patient == pat, "Pg_pred"].values[0]) + 0.01,
                    "✓" if ok else "✗", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(PATIENTS, fontsize=8)
        ax.set_ylabel("Pg fraction")
        ct_markers = ["▲" if patient_ct_from_df(df, p) == 2 else "●"
                      for p in PATIENTS]
        ax.set_title(f"{plan_label} — Pg fraction\n"
                     f"(▲=CT2/dysbiotic, ●=CT1/commensal)", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 0.5)

        # Right: per-species stacked bar obs vs pred
        ax2 = axes[row_idx, 1]
        obs_mat  = df[[f"obs_{sp}"  for sp in SP_ORDER]].values
        pred_mat = df[[f"pred_{sp}" for sp in SP_ORDER]].values
        bot_o = np.zeros(len(PATIENTS))
        bot_p = np.zeros(len(PATIENTS))
        for j, sp in enumerate(SP_ORDER):
            ax2.bar(x - width/2, obs_mat[:, j],  width, bottom=bot_o,
                    color=COLORS[sp], alpha=0.9, linewidth=0)
            ax2.bar(x + width/2, pred_mat[:, j], width, bottom=bot_p,
                    color=COLORS[sp], alpha=0.5, linewidth=0, hatch="//")
            bot_o += obs_mat[:, j]; bot_p += pred_mat[:, j]
        ax2.set_xticks(x); ax2.set_xticklabels(PATIENTS, fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Composition (solid=obs, hatch=pred)")
        mean_rmse = df["RMSE_wk3"].mean()
        n_ok = df["Pg_correct"].sum()
        ax2.set_title(f"{plan_label}\nmean RMSE={mean_rmse:.3f}  "
                      f"Pg-class {n_ok}/{len(df)}", fontsize=9)
        # Legend patches
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=COLORS[sp], label=sp) for sp in SP_ORDER]
        ax2.legend(handles=patches, fontsize=7, ncol=5, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTDIR / "dieckow_prediction.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "dieckow_prediction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUTDIR}/dieckow_prediction.png")


def patient_ct_from_df(df, pat):
    return df.loc[df.patient == pat, "CT"].values[0]


# ── Main ─────────────────────────────────────────────────────────────────────

def _print_df(df: pd.DataFrame, label: str):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'Pat':3s}  {'CT':3s}  {'Pg_obs':7s}  {'Pg_pred':7s}  "
          f"{'Pg_cls':6s}  {'RMSE_wk3':9s}")
    print("  " + "-" * 55)
    for _, r in df.iterrows():
        print(f"  {r.patient:3s}  CT{int(r.CT)}  {r.Pg_obs:7.3f}  {r.Pg_pred:7.3f}  "
              f"{'✓' if r.Pg_correct else '✗':6s}  {r.RMSE_wk3:9.4f}")
    n_ok = df.Pg_correct.sum()
    print(f"  Pg classification: {n_ok}/{len(df)}  |  "
          f"mean RMSE={df.RMSE_wk3.mean():.4f}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    phi, patient_ct, kp = load_dieckow()
    print(f"Loaded {len(PATIENTS)} patients, calibrated kp from Heine.")

    print("\nRunning Plan A  (week-1 IC → simulate 14 days → predict week-3)…")
    df_a = plan_a(phi, patient_ct, kp)

    print("Running Plan B  (CT label + Heine day-1 IC → simulate 21 days → predict week-3)…")
    df_b = plan_b(phi, patient_ct, kp)

    _print_df(df_a, "PLAN A — Personalised (in-vivo wk-1 IC)")
    _print_df(df_b, "PLAN B — Attractor-only (Heine IC, CT label only)")

    df_a.to_csv(OUTDIR / "planA_summary.csv", index=False)
    df_b.to_csv(OUTDIR / "planB_summary.csv", index=False)
    print(f"\n  CSVs saved → {OUTDIR}/")

    print("\nPlotting…")
    plot_results(df_a, df_b)
    print("Done.")


if __name__ == "__main__":
    main()
