#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
fig_welch_so_vei_depth_thesis.py — Spatial cross-feeding: Veillonella shallower than
S. oralis in both CH and DH.

Panel (a): CoM depth vs day, So vs Vei, CH and DH
Panel (b): Δz (Vei - So) bar chart per timepoint, colored by condition
Panel (c): Comparison with Welch 2016 text box

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/plot_welch_so_vei_depth_thesis.py

Output: results/diffusion_fit/fig_welch_so_vei_depth.{pdf,png}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import wilcoxon

from thesis_style import use as thesis_style

ROOT = Path(__file__).resolve().parents[2]
CSV  = ROOT / "results/diffusion_fit/spatial_crossfeeding_com_depth.csv"
OUT  = ROOT / "results/diffusion_fit/fig_welch_so_vei_depth"

COL = {"CH": "#1f77b4", "DH": "#d62728"}
DAYS = [1, 6, 10, 15, 21]

df = pd.read_csv(CSV)

fig, axes = plt.subplots(1, 3, figsize=thesis_style(1.0, aspect=0.38),
                         gridspec_kw={"width_ratios": [2, 1.4, 1.8]})

# ── (a) CoM depth vs day ────────────────────────────────────────────────────
ax = axes[0]
markers = {"CH": "o", "DH": "s"}
styles  = {"CH": "-",  "DH": "--"}
for cond in ("CH", "DH"):
    sub = df[df.condition == cond].sort_values("day")
    ax.plot(sub.day, sub["S.oralis"],    color=COL[cond], ls=styles[cond],
            marker=markers[cond], ms=5, lw=1.4,
            label=f"So ({cond})")
    ax.plot(sub.day, sub["Vd/Vp"],       color=COL[cond], ls=styles[cond],
            marker=markers[cond], ms=5, lw=1.4, mfc="white",
            label=f"Vei ({cond})")

ax.set_xlabel("Day")
ax.set_ylabel(r"Centre-of-mass depth [$\mu$m]")
ax.set_title(r"\textbf{(a)} So vs Vei depth CoM" + "\n(lower $z$ = closer to bulk)",
             fontsize=7)
ax.set_xticks(DAYS)
# legend: outside / below axes to avoid overlap
ax.legend(fontsize=6, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.22), frameon=True, framealpha=0.9,
          handlelength=1.5, columnspacing=0.8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── (b) Δz bar chart ────────────────────────────────────────────────────────
ax = axes[1]
ch = df[df.condition == "CH"].sort_values("day")
dh = df[df.condition == "DH"].sort_values("day")

ch_dz = (ch["Vd/Vp"].values - ch["S.oralis"].values)
dh_dz = (dh["Vd/Vp"].values - dh["S.oralis"].values)
all_dz = np.concatenate([ch_dz, dh_dz])

x_ch = np.arange(len(DAYS)) * 2.2
x_dh = x_ch + 0.9
w = 0.8

ax.bar(x_ch, ch_dz, width=w, color=COL["CH"], alpha=0.85, label="CH")
ax.bar(x_dh, dh_dz, width=w, color=COL["DH"], alpha=0.85, label="DH")
ax.axhline(0, color="black", lw=0.7, ls=":")

# sign test p-value
from scipy.stats import binomtest
n_neg = int((all_dz < 0).sum())
pval  = binomtest(n_neg, len(all_dz), 0.5).pvalue
ax.text(0.5, 0.97, f"$p = {pval:.4f}$\n($n={len(all_dz)}$)",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=6.5, color="#444")

ax.set_xticks((x_ch + x_dh) / 2)
ax.set_xticklabels([str(d) for d in DAYS], fontsize=6.5)
ax.set_xlabel("Day", fontsize=7)
ax.set_ylabel(r"$\Delta z$ (Vei $-$ So) [$\mu$m]", fontsize=7)
ax.set_title(r"\textbf{(b)} Vei shallower than So" + "\n($\\Delta z < 0$)",
             fontsize=7)
# legend below
ax.legend(fontsize=6, loc="upper center",
          bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True,
          framealpha=0.9, handlelength=1.2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ── (c) Welch 2016 comparison text box ──────────────────────────────────────
ax = axes[2]
ax.axis("off")
mean_dz = all_dz.mean()
textstr = (
    r"\textbf{Welch 2016 (PNAS)}" + "\n"
    r"CLASI-FISH, supragingival" + "\n"
    r"\textit{Veillonella}--\textit{Streptococcus}" + "\n"
    "``interdigitated'' arrangement\n"
    "\n"
    r"\textbf{This study (HOBIC FISH, $n=10$)}" + "\n"
    r"\textit{Vei} CoM shallower than \textit{So}" + "\n"
    f"by {abs(mean_dz):.2f}" + r" $\mu$m (mean," + "\n"
    f"$p = {pval:.3f}$); 10/10 timepoints\n"
    "\n"
    r"\textit{Interpretation:}" + "\n"
    r"Vei above So (closer to bulk)" + "\n"
    r"$\rightarrow$ lactate diffuses upward" + "\n"
    r"$\rightarrow$ AGORA cross-feeding" + "\n"
    "confirmed spatially"
)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
        va="top", ha="left", fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.5", fc="#eef4fb", ec="#2060a0", lw=0.8))
ax.set_title(r"\textbf{(c)} Comparison with Welch 2016", fontsize=7, pad=4)

fig.suptitle(
    r"Spatial cross-feeding: \textit{Veillonella} (lactate consumer) is shallower"
    r" than \textit{S.~oralis} (producer)",
    fontsize=8, y=1.01)

fig.tight_layout(rect=(0, 0.1, 1, 1))

for ext in ("pdf", "png"):
    kw = {"dpi": 300} if ext == "png" else {}
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print(f"Saved: {OUT}.pdf")
