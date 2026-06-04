#!/usr/bin/env python3
"""
gLV fit figure matching the Hamilton fig2_phi_transposed format.

Layout: 5 rows (species) × 4 cols (conditions).
Each panel: replicate IQR box plots + gLV MAP trajectory.

Output: results/heine2025/glv_heine_fit_thesis.{pdf,png}
Usage:  python scripts/figures/plot_glv_heine_thesis.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from thesis_style import use as _ts_use
_ts_use()

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 11,
    "axes.titleweight": "bold", "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 8.5, "lines.linewidth": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.2,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parents[2]
DATA_CSV    = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species"
                   "/experiment_data/fig3_species_distribution_replicates.csv")
RESULT_JSON = _HERE / "results/heine2025/fit_glv_heine.json"
OUT_DIR     = _HERE / "results/heine2025"

DAYS = [1, 3, 6, 10, 15, 21]
N_SP = 5

# ── Metadata ──────────────────────────────────────────────────────────────────
CONDS = ["CS", "CH", "DS", "DH"]

CK_MAP = {
    "CS": ("Commensal", "Static"), "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"), "DH": ("Dysbiotic", "HOBIC"),
}

SPECIES_LISTS = {
    "Commensal": ["S. oralis", "A. naeslundii", "V. dispar",
                  "F. nucleatum", "P. gingivalis_20709"],
    "Dysbiotic":  ["S. oralis", "A. naeslundii", "V. parvula",
                  "F. nucleatum", "P. gingivalis_W83"],
}

# species index mapping for replicate data
REP_SP_MAP = {
    "S. oralis": 0, "A. naeslundii": 1,
    "V. dispar": 2, "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4, "P. gingivalis_W83": 4,
}

SPECIES_ITALIC = [
    r"$\it{S.\ oralis}$",
    r"$\it{A.\ naeslundii}$",
    r"$\it{Veillonella}$ spp.",
    r"$\it{F.\ nucleatum}$",
    r"$\it{P.\ gingivalis}$",
]

# Per-condition intra-panel strain labels (rows 2 and 4 vary by condition)
STRAIN_LABEL = {
    2: {"CS": r"$\it{V.\ dispar}$",      "CH": r"$\it{V.\ dispar}$",
        "DS": r"$\it{V.\ parvula}$",     "DH": r"$\it{V.\ parvula}$"},
    4: {"CS": r"$\it{P.\ gingivalis}$ ATCC 20709",
        "CH": r"$\it{P.\ gingivalis}$ ATCC 20709",
        "DS": r"$\it{P.\ gingivalis}$ W83",
        "DH": r"$\it{P.\ gingivalis}$ W83"},
}

COND_LABELS = {
    "CS": r"$\bf{Commensal}$" + "\n" + r"$\bf{Static}$",
    "CH": r"$\bf{Commensal}$" + "\n" + r"$\bf{HOBIC}$",
    "DS": r"$\bf{Dysbiotic}$" + "\n" + r"$\bf{Static}$",
    "DH": r"$\bf{Dysbiotic}$" + "\n" + r"$\bf{HOBIC}$",
}

SP_COLORS = {
    "C": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#E07070"],
    "D": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#8B0000"],
}


# ── ODE ───────────────────────────────────────────────────────────────────────
def _rhs(t, phi, A, b):
    phi = np.maximum(phi, 1e-10); phi /= phi.sum()
    f = A @ phi + b
    return phi * (f - phi @ f)


def integrate(A, b, phi0, t_eval):
    sol = solve_ivp(_rhs, [t_eval[0], t_eval[-1]], phi0,
                    t_eval=t_eval, args=(A, b), method="RK45",
                    rtol=1e-6, atol=1e-9, max_step=0.5)
    traj = np.maximum(sol.y.T, 0)
    return traj / traj.sum(axis=1, keepdims=True)


def median_phi(rep_df, condition, cultivation):
    sp_list = SPECIES_LISTS[condition]
    phi = np.zeros((len(DAYS), N_SP))
    for j, sp in enumerate(sp_list):
        for i, day in enumerate(DAYS):
            vals = rep_df.loc[
                (rep_df["condition"] == condition) &
                (rep_df["cultivation"] == cultivation) &
                (rep_df["species"] == sp) &
                (rep_df["day"] == day), "distribution_pct"
            ].values
            phi[i, j] = np.median(vals) if len(vals) else 0.0
    rs = phi.sum(axis=1, keepdims=True)
    return phi / np.where(rs > 0, rs, 1.0)


# ── Figure ────────────────────────────────────────────────────────────────────
def main():
    fit = json.loads(RESULT_JSON.read_text())
    rep_df = pd.read_csv(DATA_CSV)
    t_fine = np.linspace(1, 21, 2501)

    fig, axes = plt.subplots(5, 4, figsize=(7.5, 9.5), sharex=True, sharey=True)

    for col, ck in enumerate(CONDS):
        cond, cult = CK_MAP[ck]
        A = np.array(fit[ck]["A"])
        b = np.array(fit[ck]["b"])
        rmse = fit[ck]["rmse"]
        colors = SP_COLORS["C" if ck.startswith("C") else "D"]

        phi_med = median_phi(rep_df, cond, cult)
        phi_map = integrate(A, b, phi_med[0], t_fine)

        cond_rep = rep_df[(rep_df["condition"] == cond) & (rep_df["cultivation"] == cult)]

        for row in range(5):
            ax = axes[row, col]

            # Replicate box plots
            sp_names = [s for s, idx in REP_SP_MAP.items() if idx == row]
            boxes, positions = [], []
            for day in DAYS:
                vals = cond_rep.loc[
                    (cond_rep["day"] == day) & (cond_rep["species"].isin(sp_names)),
                    "distribution_pct"
                ].values / 100.0
                if len(vals):
                    boxes.append(vals); positions.append(day)
            if boxes:
                bx = ax.boxplot(
                    boxes, positions=positions, widths=1.8,
                    patch_artist=True, showfliers=False, manage_ticks=False, zorder=1,
                    medianprops=dict(color="k", linewidth=0.8),
                    whiskerprops=dict(color="gray", linewidth=0.5),
                    capprops=dict(color="gray", linewidth=0.5),
                    boxprops=dict(linewidth=0.4),
                )
                for p in bx["boxes"]:
                    p.set_facecolor(colors[row]); p.set_alpha(0.15)
                    p.set_edgecolor(colors[row])

            # MAP line + median dots
            ax.plot(t_fine, phi_map[:, row], color=colors[row], lw=1.5, zorder=4)
            ax.scatter(DAYS, phi_med[:, row], s=20, color=colors[row],
                       edgecolors="k", linewidths=0.4, zorder=5)

            ax.set_xlim(0, 22.5); ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(DAYS); ax.grid(True)

            # Column title (top row only)
            if row == 0:
                ax.set_title(
                    COND_LABELS[ck] + "\n" + r"$\mathrm{RMSE} = " + f"{rmse:.3f}" + r"$",
                    fontsize=11, pad=6,
                )

            # Row y-label (left column only)
            if col == 0:
                ylabel = (SPECIES_ITALIC[row] + r"$\quad \varphi_i\;[-]$"
                          if row < 4 else r"$\varphi_i\;[-]$")
                ax.set_ylabel(ylabel, fontsize=10)

            # Intra-panel strain label (V and Pg rows)
            if row in STRAIN_LABEL and ck in STRAIN_LABEL[row]:
                ax.text(0.5, 0.92, STRAIN_LABEL[row][ck],
                        transform=ax.transAxes, fontsize=7.5,
                        ha="center", va="top", fontstyle="italic",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

            # X label (bottom row only)
            if row == 4:
                ax.set_xlabel("Day", fontsize=12)

    # Top legend
    _lc = "#2166AC"
    handles = [
        Line2D([0], [0], color=_lc, lw=1.5, label=r"gLV MAP $\varphi_i$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_lc,
               markeredgecolor="k", markersize=5, label="Exp. median"),
        Patch(facecolor=_lc, alpha=0.15, edgecolor=_lc, linewidth=0.5,
              label="Replicate IQR"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.5,
               framealpha=0.95, handlelength=1.5, handletextpad=0.5,
               borderpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(h_pad=0.4, w_pad=0.3, rect=[0, 0, 1, 0.965])

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"glv_heine_fit_thesis.{ext}"
        fig.savefig(out)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
