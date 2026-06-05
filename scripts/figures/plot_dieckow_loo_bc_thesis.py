#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
Dieckow LOO comparison figure — thesis fig 4.5.
Left:  LOO-RMSE for all model variants (from fold files + table values).
Right: LOO-BC for the 3 variants where it was computed (table values),
       with null baselines dashed.

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/plot_dieckow_loo_bc_thesis.py

Output: results/dieckow_cr/dieckow_loo_bc.{pdf,png}
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from thesis_style import use as thesis_style

CR = Path(__file__).resolve().parents[2] / "results/dieckow_cr"

# ── LOO-RMSE: compute from fold files where available, else from table ─────────
def fold_rmse(pattern):
    files = sorted(glob.glob(str(CR / pattern)))
    if not files:
        return None, None
    vals = [json.load(open(f))["rmse"] for f in files]
    return float(np.mean(vals)), float(np.std(vals))

rmse_glv_a025, std_glv_a025 = fold_rmse("loo_glv_a0p25_fold*.json")   # best gLV L1+L2
rmse_glv_a000, std_glv_a000 = fold_rmse("loo_glv_a0p0_fold*.json")    # gLV free

# Remaining values from table (tab:dieckow_loo) — no fold files available
RMSE_MODELS = [
    # (label, rmse, std, color, bold)
    ("Hamilton\n+MICOM",     0.0502, 0.014, "#5b9bd5", False),
    ("Hamilton\nL1+L2",      0.0516, 0.013, "#70ad47", False),
    ("gLV+MICOM\n$\\alpha$=0.25", 0.0516, 0.015, "#9dc3e6", False),
    ("gLV $\\alpha$=0.25\nL1+L2 $\\bigstar$",
                             rmse_glv_a025 or 0.0490, std_glv_a025 or 0.016, "#1f4e79", True),
    ("MDSINE2",              0.0552, 0.020, "#bfbfbf", False),
    ("Hamilton\n+AGORA",     0.0562, 0.015, "#a9d18e", False),
    ("gLV free",             rmse_glv_a000 or 0.0588, std_glv_a000 or 0.018, "#d9d9d9", False),
    ("Hamilton\nno prior",   0.0595, 0.020, "#aeaaaa", False),
    ("NSP IFT",              0.0906, 0.030, "#c00000", False),
]

# ── LOO-BC: from tab:dieckow_loo (only 3 variants have BC values) ──────────────
BC_MODELS = [
    ("Hamilton\n+AGORA",  0.147, "#a9d18e"),
    ("gLV free",          0.154, "#d9d9d9"),
    ("Hamilton\nL1+L2",   0.155, "#70ad47"),
]
BC_NULL_PERSISTENCE = 0.281
BC_NULL_GRANDMEAN   = 0.254

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=thesis_style(1.0, aspect=0.42))

# Left: RMSE
ax = axes[0]
labels = [m[0] for m in RMSE_MODELS]
rmses  = [m[1] for m in RMSE_MODELS]
stds   = [m[2] for m in RMSE_MODELS]
colors = [m[3] for m in RMSE_MODELS]
bolds  = [m[4] for m in RMSE_MODELS]
x = np.arange(len(RMSE_MODELS))
bars = ax.bar(x, rmses, color=colors, edgecolor="white", linewidth=0.5,
              yerr=stds, capsize=2.5,
              error_kw={"lw": 0.8, "capthick": 0.8})
for bar, bold in zip(bars, bolds):
    if bold:
        bar.set_edgecolor("gold"); bar.set_linewidth(2.0)
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=4.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=5.0)
ax.set_ylabel("LOO-RMSE (mean $\\pm$ SD)", fontsize=7)
ax.set_title("(A) LOO-RMSE by model\n(lower = better)", fontsize=7)
ax.tick_params(axis="both", labelsize=5.5)

# Right: BC
ax = axes[1]
x2 = np.arange(len(BC_MODELS))
bc_labs  = [m[0] for m in BC_MODELS]
bc_vals  = [m[1] for m in BC_MODELS]
bc_cols  = [m[2] for m in BC_MODELS]
bars2 = ax.bar(x2, bc_vals, color=bc_cols, edgecolor="white", linewidth=0.5)
ax.axhline(BC_NULL_PERSISTENCE, color="#d62728", ls="--", lw=1.0,
           label=f"Persistence null ({BC_NULL_PERSISTENCE})")
ax.axhline(BC_NULL_GRANDMEAN,   color="#ff7f0e", ls="--", lw=1.0,
           label=f"Grand-mean null ({BC_NULL_GRANDMEAN})")
for bar, val in zip(bars2, bc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)
ax.set_xticks(x2)
ax.set_xticklabels(bc_labs, fontsize=6.0)
ax.set_ylabel("LOO Bray--Curtis (lower = better)", fontsize=7)
ax.set_title("(B) LOO Bray--Curtis\n(3 variants computed)", fontsize=7)
ax.legend(fontsize=5.5, loc="upper right")
ax.tick_params(axis="both", labelsize=5.5)
ax.set_ylim(0, 0.32)

fig.tight_layout(w_pad=2.0)

out_pdf = CR / "dieckow_loo_bc.pdf"
out_png = CR / "dieckow_loo_bc.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_pdf}")
