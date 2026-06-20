"""Optimal intervention timing — is there a treatment window before the vicious-cycle tipping point?

The mechano-biological cascade (fig_periimplantitis_rankl_opg.py) already carries intervention hooks:
debridement (dysbiosis drive B -> B*tx_fac after tx_week) and anti-RANKL (block_rankl). Because the
loop is bistable, the clinically decisive question is not *whether* but *when*: an intervention applied
before the trajectory crosses the point of no return arrests bone loss; the same intervention applied
later cannot. This script maps that window.

Outputs:
  (A) bone-loss trajectories for a progressor under early / late / no debridement.
  (B) final 36-mo loss vs intervention time -> the latest treatment time that still keeps loss < 2 mm.
  (C) per-patient treatment window vs dysbiosis severity (GDI): sicker patients must be treated sooner.
Pure ODE (no FEM). Run: python masterarbeit_ansys_fem/extensions/intervention_timing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from fig_periimplantitis_rankl_opg import (  # noqa: E402  reuse the calibrated cascade
    amplification, integrate, load_patients, WEEKS,
)

LOSS_OK = 2.0          # peri-implantitis threshold [mm]
TX_FAC = 0.15          # debridement: residual dysbiosis drive after treatment


def final_loss(B, A, tx_week=None, tx_fac=TX_FAC, block_rankl=0.0):
    _, out = integrate(B, A, tx_week=tx_week, tx_fac=tx_fac, block_rankl=block_rankl)
    return out[-1, 5]


def treatment_window(B, A, grid):
    """Latest debridement time (weeks) that still keeps 36-mo loss < LOSS_OK; nan if untreatable/never-needed."""
    losses = np.array([final_loss(B, A, tx_week=tw) for tw in grid])
    ok = grid[losses < LOSS_OK]
    if losses[-1] < LOSS_OK and len(ok) == len(grid):
        return np.inf           # never progresses even untreated
    return float(ok.max()) if len(ok) else 0.0


def main() -> int:
    A = amplification()
    pats, B, gdi = load_patients()
    grid = np.linspace(0, WEEKS * 0.9, 40)

    # pick a clear progressor (high dysbiosis drive)
    ip = int(np.argmax(B)); Bp = B[ip]
    print(f"progressor: {pats[ip]}  B={Bp:.2f}  GDI={gdi[ip]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # (A) trajectories: no tx / early / late
    t, _ = integrate(Bp, A); t_wk = np.linspace(0, WEEKS, len(t))
    for tw, lab, col in [(None, "no treatment", "#c0392b"),
                         (24, "treat @ 6 mo (early)", "#27ae60"),
                         (72, "treat @ 18 mo (late)", "#e67e22")]:
        _, out = integrate(Bp, A, tx_week=tw)
        axes[0].plot(t_wk / 4.345, out[:, 5], lw=2, color=col, label=lab)
        if tw is not None:
            axes[0].axvline(tw / 4.345, color=col, ls=":", lw=1.0)
    axes[0].axhline(LOSS_OK, color="0.5", ls="--", lw=1.0, label=f"{LOSS_OK} mm threshold")
    axes[0].set_xlabel("time [months]"); axes[0].set_ylabel("marginal bone loss $L$ [mm]")
    axes[0].set_title("(A) timing decides the outcome", fontsize=11, loc="left")
    axes[0].legend(fontsize=8); axes[0].grid(True, lw=0.4, alpha=0.4)

    # (B) final loss vs intervention time -> the window
    losses = np.array([final_loss(Bp, A, tx_week=tw) for tw in grid])
    axes[1].plot(grid / 4.345, losses, "o-", color="#2c3e50", lw=2)
    axes[1].axhline(LOSS_OK, color="0.5", ls="--", lw=1.0)
    win = treatment_window(Bp, A, grid)
    if np.isfinite(win):
        axes[1].axvline(win / 4.345, color="#c0392b", lw=1.5,
                        label=f"latest safe treatment ≈ {win/4.345:.0f} mo")
        axes[1].axvspan(0, win / 4.345, color="#27ae60", alpha=0.12)
    axes[1].set_xlabel("intervention time [months]"); axes[1].set_ylabel("final 36-mo loss [mm]")
    axes[1].set_title("(B) the treatment window", fontsize=11, loc="left")
    axes[1].legend(fontsize=8); axes[1].grid(True, lw=0.4, alpha=0.4)

    # (C) per-patient window vs GDI severity
    wins = np.array([treatment_window(b, A, grid) for b in B])
    prog = np.isfinite(wins)                              # patients that ever need treatment
    axes[2].scatter(gdi[prog], wins[prog] / 4.345, s=55, c="#c0392b", edgecolor="k", lw=0.4)
    n_never = int((~prog).sum())
    axes[2].set_xlabel("dysbiosis severity GDI"); axes[2].set_ylabel("latest safe treatment time [months]")
    axes[2].set_title(f"(C) sicker → treat sooner\n({n_never} patients never cross untreated)",
                      fontsize=10, loc="left")
    axes[2].grid(True, lw=0.4, alpha=0.4)
    if prog.sum() >= 3:
        from scipy.stats import spearmanr
        rho, pv = spearmanr(gdi[prog], wins[prog])
        axes[2].annotate(f"Spearman ρ={rho:+.2f}, p={pv:.2f}", xy=(0.95, 0.95),
                         xycoords="axes fraction", ha="right", va="top", fontsize=8)

    fig.suptitle("Intervention timing: a treatment window before the dysbiosis tipping point (model)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_intervention_timing.{ext}", dpi=200)

    print(f"latest safe debridement for the progressor ≈ {win/4.345:.0f} months "
          f"(treat after this and 36-mo loss exceeds {LOSS_OK} mm)")
    print(f"{n_never}/{len(B)} patients never cross {LOSS_OK} mm untreated")
    print(f"wrote {(OUT / 'fig_intervention_timing.pdf').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
