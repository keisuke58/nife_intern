#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
fig_sigma_stability.py — Interaction heterogeneity and marginal stability
analysis across the four Heine 2025 attractor states (CS / CH / DS / DH).

Draws on: Pasqualini et al. 2024 (eLife 105948 / Pasqualini2024) who showed
that *healthy* microbiomes exhibit higher interaction-strength heterogeneity
(σ_A) than dysbiotic ones, and that marginal stability (λ_max closer to 0)
correlates with disease state.

We reproduce that analysis on the Heine 5-species Hamilton TMCMC posterior
(10 000 particles per attractor) to test whether the same signature holds
for the peri-implant biofilm model.

Metrics (per posterior sample):
  σ_A     = std( off-diagonal A[i,j] )           — heterogeneity
  μ_A     = mean( |off-diagonal A[i,j]| )         — overall interaction scale
  λ_max   = max real eigenvalue of A              — marginal stability proxy
  ρ_A     = σ_A / μ_A                            — relative dispersion (CV)

Outputs: results/fig_sigma_stability.{pdf,png}
         results/fig_sigma_stability_data.npz    (σ, λ, μ arrays per cond)

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
        scripts/figures/fig_sigma_stability.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

from paper_data import paper_5sp_samples
from thesis_style import use as thesis_style

# ── theta → A parameterisation (cf. heine_posterior_figs.py) ─────────────────
A_IDX = {
    (0, 0): 0,  (0, 1): 1,  (1, 1): 2,
    (2, 2): 5,  (2, 3): 6,  (3, 3): 7,
    (0, 2): 10, (0, 3): 11, (1, 2): 12, (1, 3): 13, (4, 4): 14,
    (0, 4): 16, (1, 4): 17, (2, 4): 18, (3, 4): 19,
}
OFFDIAG_KEYS = [(i, j) for (i, j) in A_IDX if i != j]   # 10 unique off-diag pairs

CONDS      = ["CS", "CH", "DS", "DH"]
COND_LABEL = {"CS": "CS", "CH": "CH", "DS": "DS", "DH": "DH"}
COND_COLOR = {"CS": "#2166ac", "CH": "#74add1", "DS": "#d73027", "DH": "#a50026"}
N_SP       = 5
SP_LABELS  = ["S.o", "A.n", "Vd", "F.n", "P.g"]

OUT = Path(__file__).resolve().parents[2] / "results"


def theta_to_A(theta: np.ndarray) -> np.ndarray:
    A = np.zeros((N_SP, N_SP))
    for (i, j), k in A_IDX.items():
        A[i, j] = A[j, i] = theta[k]
    return A


def compute_posterior_stats(samples: np.ndarray) -> dict:
    """Vectorised computation of σ_A, μ_A, λ_max, ρ_A per sample."""
    n = len(samples)
    sigma_A   = np.empty(n)
    mu_A      = np.empty(n)
    lambda_max = np.empty(n)

    for idx, theta in enumerate(samples):
        A = theta_to_A(theta)
        od = np.array([A[i, j] for (i, j) in OFFDIAG_KEYS])
        sigma_A[idx]    = od.std()
        mu_A[idx]       = np.abs(od).mean()
        lambda_max[idx] = np.linalg.eigvalsh(A).max()   # symmetric → real eigvals

    rho_A = np.where(mu_A > 1e-12, sigma_A / mu_A, np.nan)
    return {"sigma": sigma_A, "mu": mu_A, "lambda_max": lambda_max, "rho": rho_A}


# ── Load and compute ──────────────────────────────────────────────────────────
print("Loading TMCMC posterior samples …")
data = {}
for cond in CONDS:
    samp = paper_5sp_samples(cond)
    data[cond] = compute_posterior_stats(samp)
    print(f"  {cond}: n={len(samp)}, σ_A={data[cond]['sigma'].mean():.3f},"
          f" λ_max={data[cond]['lambda_max'].mean():.3f}")

# Save raw arrays
np.savez(OUT / "fig_sigma_stability_data.npz",
         **{f"{cond}_{k}": v
            for cond in CONDS for k, v in data[cond].items()})

# ── Statistical tests (Mann–Whitney U; CS vs DH, CH vs DS) ───────────────────
pairs = [("CS", "DH"), ("CH", "DS")]
tests = {}
for metric in ("sigma", "mu", "lambda_max"):
    tests[metric] = {}
    for (c1, c2) in pairs:
        u, p = stats.mannwhitneyu(data[c1][metric], data[c2][metric],
                                  alternative="two-sided")
        tests[metric][(c1, c2)] = p
        print(f"  {metric:12s}  {c1} vs {c2}:  p={p:.2e}")

# MINE R² per condition (from fig_mine_emulator run)
MINE_R2 = {"CS": 0.916, "CH": 0.957, "DS": 0.958, "DH": 0.787}


def _add_significance(ax, x1, x2, y_bracket, h_frac, stars, color="k"):
    """Draw a bracket + stars above a violin plot without escaping the axes."""
    y2 = y_bracket + h_frac
    ax.plot([x1, x1, x2, x2], [y_bracket, y2, y2, y_bracket],
            color=color, lw=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y2, stars,
            ha="center", va="bottom", fontsize=7, color=color)


# ── Figure (1×4) ───────────────────────────────────────────────────────────────
thesis_style()
fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))

metrics = [
    ("sigma",      r"$\sigma_A$", False),
    ("mu",         r"$\mu_A$",    False),
    ("lambda_max", r"$\lambda_{\max}$", True),
]

for ax, (key, ylabel, add_zero) in zip(axes[:3], metrics):
    vals = [data[c][key] for c in CONDS]

    parts = ax.violinplot(vals, positions=range(4), showmedians=True,
                          showextrema=False)
    for body, cond in zip(parts["bodies"], CONDS):
        body.set_facecolor(COND_COLOR[cond])
        body.set_alpha(0.75)
    parts["cmedians"].set_color("k")
    parts["cmedians"].set_linewidth(1.5)

    if add_zero:
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)

    ax.set_xticks(range(4))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
    ax.set_ylabel(ylabel)

    # significance in axes fraction space — immune to ylim/KDE issues
    for (c1, c2), yline, ytxt in [(("CH", "DS"), 0.91, 0.93),
                                   (("CS", "DH"), 0.97, 0.99)]:
        p = tests[key].get((c1, c2))
        if p is None:
            p = tests[key].get((c2, c1))
        if p is None:
            continue
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        x1f = (CONDS.index(c1) + 0.5) / 4.0
        x2f = (CONDS.index(c2) + 0.5) / 4.0
        xmid = (x1f + x2f) / 2.0
        # horizontal line at yline in axes fraction
        ax.plot([x1f, x2f], [yline, yline], transform=ax.transAxes,
                color="k", lw=0.8, clip_on=False)
        ax.text(xmid, ytxt, stars, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7, color="k", clip_on=False)

# Panel 4: σ_A vs MINE R² scatter — ties both analyses together
ax4 = axes[3]
sigma_med = {c: np.median(data[c]["sigma"]) for c in CONDS}
sigma_iqr = {c: (np.percentile(data[c]["sigma"], 75) -
                 np.percentile(data[c]["sigma"], 25)) for c in CONDS}
for c in CONDS:
    ax4.errorbar(sigma_med[c], MINE_R2[c],
                 xerr=sigma_iqr[c] / 2,
                 fmt="o", color=COND_COLOR[c], ms=8, capsize=3, label=c)
    ax4.annotate(c, (sigma_med[c], MINE_R2[c]),
                 xytext=(4, 4), textcoords="offset points", fontsize=8)
# Pearson r
xs = [sigma_med[c] for c in CONDS]
ys = [MINE_R2[c] for c in CONDS]
r_val = np.corrcoef(xs, ys)[0, 1]
ax4.set_xlabel(r"$\sigma_A$ (posterior heterogeneity)")
ax4.set_ylabel(r"MINE $R^2$ (emulator accuracy)")
ax4.set_title(f"Posterior width vs emulability\n$r={r_val:.2f}$")
ax4.set_ylim(0.7, 1.0)

axes[0].set_title(r"Interaction heterogeneity ($\sigma_A$)")
axes[1].set_title(r"Interaction strength ($\mu_A$)")
axes[2].set_title(r"Marginal stability ($\lambda_{\max}$)")

fig.suptitle(r"Heine 2025 — TMCMC posterior ($N_p=10{,}000$) $\times$ MINE emulator",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.93])

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_sigma_stability.{ext}", dpi=300, bbox_inches="tight")
print(f"Saved → results/fig_sigma_stability.{{pdf,png}}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\nMedian ± IQR summary:")
for cond in CONDS:
    for key in ("sigma", "lambda_max"):
        arr = data[cond][key]
        med = np.median(arr)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        print(f"  {cond}  {key:12s}: {med:.4f} ± {iqr:.4f}")
