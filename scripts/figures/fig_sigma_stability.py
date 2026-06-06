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

# ── Figure ─────────────────────────────────────────────────────────────────────
thesis_style()
fig, axes = plt.subplots(1, 3, figsize=thesis_style(1.0, aspect=0.30))

metrics = [
    ("sigma",      r"$\sigma_A$ (off-diag std)",    False),
    ("mu",         r"$\mu_A$ (mean $|A_{ij}|$)",     False),
    ("lambda_max", r"$\lambda_{\max}$ (spectral)",    True),
]

for ax, (key, ylabel, add_zero) in zip(axes, metrics):
    vals = [data[c][key] for c in CONDS]
    parts = ax.violinplot(vals, positions=range(4), showmedians=True,
                          showextrema=False)
    for body, cond in zip(parts["bodies"], CONDS):
        body.set_facecolor(COND_COLOR[cond])
        body.set_alpha(0.75)
    parts["cmedians"].set_color("k")
    parts["cmedians"].set_linewidth(1.5)

    if add_zero:
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

    ax.set_xticks(range(4))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
    ax.set_ylabel(ylabel)

    # Annotate p-values
    for (c1, c2) in pairs:
        p = tests[key].get((c1, c2), None)
        if p is None:
            continue
        x1, x2 = CONDS.index(c1), CONDS.index(c2)
        ymax = max(np.percentile(data[c1][key], 97.5),
                   np.percentile(data[c2][key], 97.5))
        h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05 if ax.get_ylim()[1] != 0 else 0.1
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.annotate("", xy=(x2, ymax + h), xytext=(x1, ymax + h),
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.8))
        ax.text((x1 + x2) / 2, ymax + h * 1.1, stars,
                ha="center", va="bottom", fontsize=7)

axes[0].set_title("Interaction heterogeneity")
axes[1].set_title("Interaction strength scale")
axes[2].set_title("Marginal stability")

fig.suptitle(r"Heine 2025 attractor states — TMCMC posterior ($N_p=10{,}000$)"
             "\n(cf. Pasqualini et al. 2024)", fontsize=8)
fig.tight_layout()

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
