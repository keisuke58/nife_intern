"""Spatial vicious cycle (1-D) — why peri-implant bone loss spreads as a saucer.

The 0-D model says *whether* the loop tips; it cannot say *where* the defect goes. The clinical
pattern is "saucerisation": loss begins at the implant neck and spreads laterally/apically as a
crater. The mechanism proposed here is a stress-concentration travelling front: as bone is resorbed,
the peak crestal stress relocates to the *current bone margin* (a notch concentrates stress at its
tip), so the overload that drives Frost resorption tracks the receding margin — a self-propagating
front, not a uniform thinning.

Minimal 1-D reaction–diffusion–mechanics model along the crest coordinate x (x=0 at the implant):
    inflammation   dI/dt = D ∇²I + kI·src(x) − gI·I            (biofilm/cytokine spread from the sulcus)
    bone loss      dL/dt = kL·I·relu(σ(x) − σ_thr)·(1 − L)     (Frost overload, logistic-capped)
    stress         σ(x)  = σ_base + σ_amp·exp(−(x − margin)²/w²)   (concentration AT the bone margin)
where margin(t) is the furthest extent of significant loss. This is an illustrative mechanism model
(not geometry-calibrated); it demonstrates that the saucer emerges from margin-tracking overload.

Run: python masterarbeit_ansys_fem/extensions/spatial_vicious_cycle.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)


def simulate():
    X, nx = 6.0, 240                     # crest coordinate [mm]
    x = np.linspace(0, X, nx); dx = x[1] - x[0]
    T, dt = 60.0, 0.002                  # months
    nt = int(T / dt)
    D, kI, gI = 0.04, 1.4, 0.6           # inflammation diffusion / source / resolution
    kL, sthr = 0.9, 1.0                  # resorption gain / Frost overload threshold (normalised)
    s_base, s_amp, w = 0.5, 1.6, 0.45    # baseline / margin-concentration amplitude / width
    src = np.exp(-(x / 0.5) ** 2)        # dysbiosis source at the sulcus/neck (x≈0)

    I = np.zeros(nx); L = np.zeros(nx)
    snaps_t = [5, 15, 30, 55]; snaps = {}
    kymo = []; tk = []
    margin_hist = []
    for k in range(nt):
        t = k * dt
        margin = x[np.where(L > 0.5)[0].max()] if np.any(L > 0.5) else 0.0
        sigma = s_base + s_amp * np.exp(-((x - margin) / w) ** 2)
        lap = np.empty_like(I)
        lap[1:-1] = (I[2:] - 2 * I[1:-1] + I[:-2]) / dx ** 2
        lap[0] = lap[1]; lap[-1] = lap[-2]
        I += dt * (D * lap + kI * src - gI * I)
        I = np.clip(I, 0, None)
        L += dt * (kL * I * np.maximum(sigma - sthr, 0.0) * (1 - L))
        L = np.clip(L, 0, 1)
        if k % (nt // 240) == 0:
            kymo.append(L.copy()); tk.append(t); margin_hist.append(margin)
        for st in snaps_t:
            if abs(t - st) < dt:
                snaps[st] = (L.copy(), sigma.copy())
    return x, np.array(kymo), np.array(tk), np.array(margin_hist), snaps, sthr


def main() -> int:
    x, kymo, tk, margin, snaps, sthr = simulate()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # (A) kymograph: bone-loss front propagating laterally over time
    im = axes[0].pcolormesh(x, tk, kymo, cmap="inferno", shading="auto", vmin=0, vmax=1)
    axes[0].plot(margin, tk, color="#3498db", lw=1.6, label="bone margin")
    axes[0].set_xlabel("crest coordinate $x$ [mm] (0 = implant)"); axes[0].set_ylabel("time [months]")
    axes[0].set_title("(A) loss propagates as a front (saucer)", fontsize=11, loc="left")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=axes[0], label="bone loss $L$")

    # (B) snapshots: stress concentration tracks the receding margin
    cmap = plt.get_cmap("viridis")
    sts = sorted(snaps)
    for i, st in enumerate(sts):
        L, sig = snaps[st]
        c = cmap(i / max(len(sts) - 1, 1))
        axes[1].plot(x, L, color=c, lw=2, label=f"L @ {st} mo")
        axes[1].plot(x, (sig - sig.min()) / (sig.max() - sig.min()), color=c, lw=1.0, ls=":")
    axes[1].set_xlabel("crest coordinate $x$ [mm]"); axes[1].set_ylabel("bone loss $L$  /  stress (dotted)")
    axes[1].set_title("(B) overload (dotted) rides the margin", fontsize=11, loc="left")
    axes[1].legend(fontsize=7.5); axes[1].grid(True, lw=0.4, alpha=0.4)

    # (C) saucer depth (margin position) vs time — front kinetics
    axes[2].plot(tk, margin, color="#c0392b", lw=2.2)
    axes[2].set_xlabel("time [months]"); axes[2].set_ylabel("saucer extent (margin $x$) [mm]")
    axes[2].set_title("(C) saucer growth kinetics", fontsize=11, loc="left")
    axes[2].grid(True, lw=0.4, alpha=0.4)

    fig.suptitle("Spatial vicious cycle (1-D): saucerisation as a margin-tracking overload front (model)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_spatial_vicious_cycle.{ext}", dpi=200)

    print(f"saucer extent reached {margin.max():.2f} mm at {tk[-1]:.0f} months "
          f"(front speed ≈ {margin.max()/tk[-1]*12:.2f} mm/yr)")
    print(f"wrote {(OUT / 'fig_spatial_vicious_cycle.pdf').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
