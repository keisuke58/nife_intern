# nife-pathshim
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

"""
fig_monod_prediction.py — Publication figure: mechanistic Monod dFBA prediction.

Two-panel layout
----------------
(A) Heine 2025 in-vitro validation: predicted vs observed day-21 composition
    for 4 conditions (CS / CH / DS / DH).  Uses calibrated kinetic parameters.
(B) Dieckow 2024 in-vivo cross-system validation: Pg fraction (observed week-3
    vs predicted from CT label only, Plan B).  10/10 classification shown.

Run with TeX Live on PATH:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_monod_prediction.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from thesis_style import use, PALETTE, clean_ax, SP_COLORS, SP_SHORT
from predict_heine_attractors_dfba import (
    run_monod, phi_final, CONDITIONS, SP_ORDER,
    _load_df, load_heine_phi,
)

ROOT   = Path(__file__).parents[2]
KP_JSON = ROOT / "results" / "heine_prediction" / "calibrated_kp.json"
DPLAN_B = ROOT / "results" / "dieckow_prediction" / "planB_summary.csv"
PHI_NPY = ROOT / "results" / "dieckow_otu" / "phi_guild.npy"
CT_JSON = ROOT / "results" / "dieckow_otu" / "community_types.json"
OUTDIR  = ROOT / "results" / "dieckow_prediction"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Species colours & labels (thesis_style SP_COLORS order: So An Vp Fn Pg) ──
SP_LABELS = ["\\textit{S.\\ oralis}", "\\textit{A.\\ naeslundii}",
             "\\textit{Veillonella}", "\\textit{F.\\ nucleatum}",
             "\\textit{P.\\ gingivalis}"]

COND_LABELS = {
    "CS": "CS\n(comm., static)",
    "CH": "CH\n(comm., HOBIC)",
    "DS": "DS\n(dysb., static)",
    "DH": "DH\n(dysb., HOBIC)",
}
COND_COLOR = {
    "CS": PALETTE["cs"], "CH": PALETTE["ch"],
    "DS": PALETTE["ds"], "DH": PALETTE["dh"],
}
PATIENTS = list("ABCDEFGHKL")


def guild_to_5sp(g):
    v = np.array([g[1], g[0], g[9], g[6], g[2]])  # So An Vp Fn Pg
    s = v.sum(); return v / s if s > 0 else v


def run_heine_panel(kp):
    """Return obs and pred phi-vectors for each condition."""
    df = _load_df()
    phi0_all  = load_heine_phi(df, day=1)
    phi21_all = load_heine_phi(df, day=21)
    pred = {}
    for label, cfg in CONDITIONS.items():
        bm = run_monod(phi0_all[label], cfg["strain"], cfg["o2"], n_days=21, kp=kp)
        pred[label] = phi_final(bm)
    return phi21_all, pred


def make_figure():
    kp = json.loads(KP_JSON.read_text())
    phi21_obs, phi21_pred = run_heine_panel(kp)

    # Dieckow Plan B
    df_b = pd.read_csv(DPLAN_B)
    phi  = np.load(PHI_NPY)
    ct   = json.loads(CT_JSON.read_text())["patient_ct"]

    figsize = use(width_frac=1.0, aspect=0.46)
    fig, (ax_h, ax_d) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [5, 6], "wspace": 0.42},
    )

    # ── Panel A: Heine stacked bars obs / pred ────────────────────────────────
    conditions = ["CS", "CH", "DS", "DH"]
    n_cond  = len(conditions)
    width   = 0.32
    gap     = 0.08
    group_w = 2 * width + gap
    x_centers = np.arange(n_cond) * (group_w + 0.25)

    for ci, cond in enumerate(conditions):
        x_obs  = x_centers[ci] - width / 2 - gap / 2
        x_pred = x_centers[ci] + width / 2 + gap / 2
        obs  = phi21_obs[cond]
        pred = phi21_pred[cond]
        for arr, xpos, alpha in [(obs, x_obs, 0.9), (pred, x_pred, 0.55)]:
            bot = 0.0
            for si in range(len(SP_ORDER)):
                ax_h.bar(xpos, arr[si], width, bottom=bot,
                         color=SP_COLORS[si], alpha=alpha,
                         linewidth=0.3, edgecolor="white")
                bot += arr[si]
        # Classification tick
        pg_obs  = phi21_obs[cond][SP_ORDER.index("Pg")]
        pg_pred = phi21_pred[cond][SP_ORDER.index("Pg")]
        ok = (pg_obs > 0.05) == (pg_pred > 0.05)
        ax_h.text(x_centers[ci], 1.05, r"$\checkmark$" if ok else r"$\times$",
                  ha="center", va="bottom", fontsize=7,
                  color="#1a7431" if ok else "#c0392b")

    ax_h.set_xticks(x_centers)
    ax_h.set_xticklabels(
        ["CS\n(comm.)", "CH\n(comm.)", "DS\n(dysb.)", "DH\n(dysb.)"],
        fontsize=7, linespacing=1.3)
    ax_h.set_ylabel("Relative abundance")
    ax_h.set_ylim(0, 1.13)
    ax_h.set_xlim(x_centers[0] - 0.45, x_centers[-1] + 0.45)
    ax_h.set_title(r"\textbf{(A)} Heine 2025 in-vitro (day~21)", fontsize=8,
                   pad=28)
    clean_ax(ax_h)

    # obs / pred + species legend — above the title
    obs_patch  = mpatches.Patch(fc="dimgrey", alpha=0.9, label="Obs.")
    pred_patch = mpatches.Patch(fc="dimgrey", alpha=0.45, label="Pred.")
    sp_patches = [mpatches.Patch(fc=SP_COLORS[i], label=SP_SHORT[i])
                  for i in range(len(SP_ORDER))]
    ax_h.legend(handles=[obs_patch, pred_patch] + sp_patches,
                fontsize=6, ncol=7, loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                handlelength=0.8, handletextpad=0.3, columnspacing=0.4,
                framealpha=0.95, edgecolor="0.8")

    # ── Panel B: Dieckow Pg fraction per patient ──────────────────────────────
    x = np.arange(len(PATIENTS))
    w = 0.32

    pg_obs_all  = []
    pg_pred_all = []
    ct_labels   = []
    correct_all = []
    for pat in PATIENTS:
        row = df_b[df_b.patient == pat].iloc[0]
        pg_obs_all.append(row["Pg_obs"])
        pg_pred_all.append(row["Pg_pred"])
        ct_labels.append(ct[pat])
        correct_all.append(bool(row["Pg_correct"]))

    for j, (pat, pg_o, pg_p, ctl, ok) in enumerate(
            zip(PATIENTS, pg_obs_all, pg_pred_all, ct_labels, correct_all)):
        c_obs  = PALETTE["health"] if ctl == 1 else PALETTE["dysbiosis"]
        ax_d.bar(x[j] - w / 2, pg_o, w, color=c_obs,      alpha=0.88,
                 linewidth=0.3, edgecolor="white")
        ax_d.bar(x[j] + w / 2, pg_p, w, color=c_obs,      alpha=0.42,
                 linewidth=0.3, edgecolor="white", hatch="///")
        top = max(pg_o, pg_p)
        ax_d.text(x[j], top + 0.012,
                  r"$\checkmark$" if ok else r"$\times$",
                  ha="center", va="bottom", fontsize=6.5,
                  color="#1a7431" if ok else "#c0392b")

    # Pg>0.05 classification threshold
    ax_d.axhline(0.05, color="0.4", ls="--", lw=0.7, label=r"$P_g = 0.05$")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(PATIENTS, fontsize=7)
    ax_d.set_xlabel("Patient")
    ax_d.set_ylabel(r"\textit{P.\,gingivalis} fraction (week 3)")
    ax_d.set_ylim(0, 0.52)
    ax_d.set_title(r"\textbf{(B)} Dieckow 2024 in-vivo (CT label only)", fontsize=8)
    clean_ax(ax_d)

    # Legend for panel B
    h_pat = [
        mpatches.Patch(fc=PALETTE["health"],    alpha=0.88, label="CT1 (commensal)"),
        mpatches.Patch(fc=PALETTE["dysbiosis"], alpha=0.88, label="CT2 (dysbiotic)"),
        mpatches.Patch(fc="dimgrey", alpha=0.88, label="Observed"),
        mpatches.Patch(fc="dimgrey", alpha=0.42, hatch="///", label="Predicted"),
    ]
    ax_d.legend(handles=h_pat, fontsize=6, loc="upper right",
                handlelength=1.0, handletextpad=0.4,
                framealpha=0.9, edgecolor="0.8")

    fig.tight_layout()
    pdf_path = OUTDIR / "fig_monod_prediction.pdf"
    png_path = OUTDIR / "fig_monod_prediction.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    make_figure()
