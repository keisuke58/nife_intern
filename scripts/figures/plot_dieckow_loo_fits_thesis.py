#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
plot_dieckow_loo_fits_thesis.py
Dieckow 10-patient in-vivo LOO predictions vs observations — thesis figure.
Best model: gLV alpha=0.25, L1+L2 prior (mean LOO-RMSE = 0.0490).

Layout: 2 rows × 5 cols (patients A-E top, F-L bottom).
Each panel: bars = observed phi (W1/W2/W3), × = LOO-predicted phi.

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/plot_dieckow_loo_fits_thesis.py

Output: results/dieckow_cr/dieckow_loo_fits_thesis.{pdf,png}
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from thesis_style import use as thesis_style
from guild_replicator_dieckow import (
    GUILD_ORDER, GUILD_SHORT, GUILD_COLORS, predict_trajectory
)

ROOT    = Path(__file__).resolve().parents[2]
PHI_NPY = ROOT / "results/dieckow_otu/phi_guild.npy"
LOO_DIR = ROOT / "results/dieckow_cr"
OUT     = LOO_DIR / "dieckow_loo_fits_thesis"

PATIENTS = list("ABCDEFGHKL")  # 10 patients (I/J unused in Dieckow)
WEEKS    = [1, 2, 3]

SHORT = [GUILD_SHORT[g] for g in GUILD_ORDER]
COLORS = [GUILD_COLORS[g] for g in GUILD_ORDER]
N_G = len(GUILD_ORDER)

# ── load observed data ─────────────────────────────────────────────────────────
phi_obs = np.load(PHI_NPY)   # (10, 3, 10)
n_p = phi_obs.shape[0]

# ── load LOO fold results ──────────────────────────────────────────────────────
fold_files = sorted(glob.glob(str(LOO_DIR / "loo_glv_a0p25_fold*.json")))
if not fold_files:
    raise FileNotFoundError("No LOO fold files found")

# build dict: patient_id → (A, b_ho, rmse, hold_idx)
folds = {}
for f in fold_files:
    d = json.load(open(f))
    folds[d["patient"]] = d

# ── figure ─────────────────────────────────────────────────────────────────────
ncols, nrows = 5, 2
fig, axes = plt.subplots(nrows, ncols,
                         figsize=thesis_style(1.0, aspect=0.50),
                         sharey=True)
axes = axes.flatten()

bar_w = 0.22
x_base = np.arange(N_G)

for ax_i, pat in enumerate(PATIENTS[:10]):
    ax = axes[ax_i]
    if pat not in folds:
        ax.set_visible(False)
        continue

    d     = folds[pat]
    hi    = d["hold_idx"]
    A     = np.array(d["A"])
    b_ho  = np.array(d["b_ho"])
    rmse  = d["rmse"]

    obs_w1 = phi_obs[hi, 0]   # shape (N_G,)
    obs_w2 = phi_obs[hi, 1]
    obs_w3 = phi_obs[hi, 2]

    # LOO prediction: predict W2, W3 from W1
    pred_w2, pred_w3 = predict_trajectory(obs_w1, b_ho, A)

    # bars: observed W1, W2, W3
    offsets = [-bar_w, 0, bar_w]
    for wi, (obs, off) in enumerate(zip([obs_w1, obs_w2, obs_w3], offsets)):
        alpha = 0.55 + 0.15 * wi
        ax.bar(x_base + off, obs, bar_w,
               color=COLORS, alpha=alpha, linewidth=0)

    # × marks: predicted W2 and W3
    for pred, off, mk in [(pred_w2, 0, "x"), (pred_w3, bar_w, "x")]:
        ax.scatter(x_base + off, pred, marker=mk, s=18,
                   color="black", linewidths=0.9, zorder=5)

    ax.set_title(f"Patient {pat}", fontsize=7)
    ax.text(0.97, 0.95, f"RMSE={rmse:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="0.4")
    ax.set_xticks(x_base)
    ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=5.5)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# hide unused axes
for ax in axes[len(PATIENTS):]:
    ax.set_visible(False)

# y-axis label on leftmost
axes[0].set_ylabel(r"$\phi$ fraction", fontsize=7)
axes[5].set_ylabel(r"$\phi$ fraction", fontsize=7)

# shared legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
leg_handles = (
    [Patch(facecolor="0.6", alpha=a, label=f"W{w+1} obs")
     for w, a in enumerate([0.55, 0.70, 0.85])] +
    [Line2D([0],[0], marker="x", color="black", linestyle="none",
            markersize=5, label="W2/W3 pred (LOO)")]
)
fig.legend(handles=leg_handles, loc="lower center",
           ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.02),
           frameon=True, framealpha=0.9)

fig.suptitle(
    r"LOO predictions: gLV ($\alpha=0.25$, L1+L2 prior) --- 10 Dieckow patients",
    fontsize=8, y=1.01)
fig.tight_layout(rect=(0, 0.05, 1, 1))

for ext in ("pdf", "png"):
    kw = {"dpi": 300} if ext == "png" else {}
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print(f"Saved: {OUT}.pdf")
