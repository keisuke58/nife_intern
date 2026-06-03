#!/usr/bin/env python3
"""
make_pipeline_overview.py — publication-grade "data flow / pipeline overview"
figure for the oral-biofilm modelling project, built from REAL data:

  (1) input          : mean 10-guild composition over the Dieckow cohort
                       (results/dieckow_otu/phi_guild.npy, 10 patients x 3 weeks x 10 guilds)
  (2) sign prior     : AGORA metabolic net-flow sign matrix  P = sgn(F)
  (3) inference      : replicator/Hamilton ODE + one-sided sign-prior hinge
  (4) interaction    : the fitted 10x10 interaction matrix A-hat
  (5) validation     : LOO-CV RMSE across models + the four ODE attractors
  ((2),(4),(r),(SA) read from results/dieckow_cr/fit_glv_hamilton_kegg_expanded_agora_w1p0.json)

Five stage-panels left->right, joined by flow arrows. Vector PDF + PNG.
Run from the repo root:  python scripts/figures/make_pipeline_overview.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from guild_replicator_dieckow import GUILD_COLORS, GUILD_SHORT

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})

HERE = Path(__file__).resolve().parents[2]
FIT = HERE / "results" / "dieckow_cr" / "fit_glv_hamilton_kegg_expanded_agora_w1p0.json"
PHI = HERE / "results" / "dieckow_otu" / "phi_guild.npy"
OUTDIR = HERE / "results" / "figures"
NAVY = "#1a3a5c"


def short(g):
    return GUILD_SHORT.get(g, g[:5])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = json.loads(FIT.read_text())
    guilds = d["guilds"]
    A = np.array(d["A"], float)
    P = np.sign(np.array(d["net_flow"], float))
    r, SA, npairs = d["r"], d["sign_agreement"], d["n_constrained_pairs"]
    phi = np.load(PHI)                       # (10 patients, 3 weeks, 10 guilds)
    phi_mean = phi.mean(axis=0)              # (3 weeks, 10 guilds)
    weeks = np.arange(1, phi.shape[1] + 1)
    labels = [short(g) for g in guilds]
    colors = [GUILD_COLORS.get(g, "#999999") for g in guilds]

    fig = plt.figure(figsize=(16.0, 6.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1.0, 1.12, 1.0, 1.12],
                          left=0.045, right=0.988, top=0.80, bottom=0.13, wspace=0.58)
    ax1, ax2, ax3, ax4, ax5 = [fig.add_subplot(gs[0, i]) for i in range(5)]

    # ---- (1) input: mean guild composition over weeks (stacked area) ----
    cum = np.zeros_like(weeks, float)
    order = np.argsort(-phi_mean.mean(0))    # most abundant guild at bottom
    for k in order:
        ax1.fill_between(weeks, cum, cum + phi_mean[:, k], color=colors[k],
                         label=labels[k], lw=0.4, edgecolor="white")
        cum = cum + phi_mean[:, k]
    ax1.set_xlim(weeks.min(), weeks.max()); ax1.set_ylim(0, 1)
    ax1.set_xticks(weeks); ax1.set_xlabel("week", fontsize=11)
    ax1.set_ylabel("guild fraction $\\varphi$", fontsize=11)
    ax1.set_title("1.  16S → guild $\\varphi$\n(10 patients × 3 wk × 10)",
                  fontsize=12.5, color=NAVY)
    ax1.tick_params(labelsize=9)
    ax1.legend(fontsize=6.4, ncol=2, loc="lower center", framealpha=0.9,
               handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))

    # ---- (2) AGORA sign prior P = sgn(F) ----
    im2 = ax2.imshow(P, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax2.set_title("2.  AGORA sign prior\n$P=\\mathrm{sgn}(F)$", fontsize=12.5, color=NAVY)
    _grid_ticks(ax2, labels)
    ax2.text(0.5, -0.20, f"{int((P>0).sum())} cross-feeding constraints (all $+$)",
             transform=ax2.transAxes, ha="center", fontsize=8.6, color="#444")

    # ---- (3) inference: equation card ----
    ax3.axis("off")
    card = FancyBboxPatch((0.04, 0.16), 0.92, 0.62, transform=ax3.transAxes,
                          boxstyle="round,pad=0.02,rounding_size=0.04",
                          fc="#eef3f8", ec=NAVY, lw=1.2, zorder=1)
    ax3.add_patch(card)
    ax3.set_title("3.  inference\ngLV / Hamilton + sign prior", fontsize=12.5, color=NAVY)
    ax3.text(0.5, 0.66, r"$\dot{\varphi}_i=\varphi_i\,[(A\varphi+b)_i-\varphi^{\!\top}(A\varphi+b)]$",
             transform=ax3.transAxes, ha="center", va="center", fontsize=11.5)
    ax3.text(0.5, 0.46, r"$\min\ \frac{1}{2\sigma^2}\sum_t\|\varphi^{obs}_t-\varphi^{pred}_t\|^2$",
             transform=ax3.transAxes, ha="center", va="center", fontsize=10.5)
    ax3.text(0.5, 0.30, r"$+\ W\!\!\sum_{(i,j)\in\mathcal{M}}[-P_{ij}A_{ij}]_+$",
             transform=ax3.transAxes, ha="center", va="center", fontsize=10.5, color="#b22222")
    ax3.text(0.5, 0.10, "TMCMC / L-BFGS-B", transform=ax3.transAxes,
             ha="center", va="center", fontsize=8.6, color="#444", style="italic")

    # ---- (4) fitted interaction matrix A-hat ----
    vlim = np.abs(A).max()
    im4 = ax4.imshow(A, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")
    ax4.set_title("4.  interaction matrix $\\hat{A}$", fontsize=12.5, color=NAVY)
    _grid_ticks(ax4, labels)
    cb = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("$A_{ij}$  (− suppress / + promote)", fontsize=8)
    ax4.text(0.5, -0.20, f"$r={r:.2f}$,  sign agreement {SA},  {npairs} pairs",
             transform=ax4.transAxes, ha="center", fontsize=8.6, color="#444")

    # ---- (5) validation: LOO-RMSE bars + attractors ----
    models = ["gLV\nfree", "L1+L2", "AGORA\nW=1.0"]
    rmse = [0.0455, 0.0516, 0.0504]
    bcol = ["#9aa7b4", "#6f9fc8", NAVY]
    xb = np.arange(len(models))
    bars = ax5.bar(xb, rmse, color=bcol, width=0.66, zorder=3)
    ax5.set_ylim(0.040, 0.054)
    ax5.set_xticks(xb); ax5.set_xticklabels(models, fontsize=8.6)
    ax5.set_ylabel("LOO-CV RMSE", fontsize=11)
    ax5.set_title("5.  validation + attractors", fontsize=12.5, color=NAVY)
    ax5.tick_params(labelsize=8.5)
    ax5.grid(axis="y", ls=":", lw=0.6, color="#cccccc", zorder=0)
    for x, v in zip(xb, rmse):
        ax5.text(x, v + 0.0004, f"{v:.4f}", ha="center", fontsize=7.8)
    ax5.text(0.5, -0.30, "attractors:  CS · CH · DS · DH\n(commensal/dysbiotic × static/HOBIC)",
             transform=ax5.transAxes, ha="center", fontsize=8.6, color="#444")

    # ---- flow arrows between stage panels ----
    fig.canvas.draw()
    axes = [ax1, ax2, ax3, ax4, ax5]
    y = 0.46
    for a, b in zip(axes[:-1], axes[1:]):
        x0 = a.get_position().x1
        x1 = b.get_position().x0
        arr = FancyArrowPatch((x0 + 0.004, y), (x1 - 0.004, y),
                              transform=fig.transFigure, mutation_scale=20,
                              arrowstyle="-|>", color=NAVY, lw=2.2, zorder=50)
        fig.add_artist(arr)

    fig.suptitle("Oral-biofilm community-dynamics pipeline  —  "
                 "from longitudinal 16S to a metabolism-constrained interaction model",
                 fontsize=15, color=NAVY, y=0.965, weight="bold")

    png = OUTDIR / "pipeline_overview_pub.png"
    pdf = OUTDIR / "pipeline_overview_pub.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", png)
    print("wrote", pdf)


def _grid_ticks(ax, labels):
    n = len(labels)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.0)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0)


if __name__ == "__main__":
    main()
