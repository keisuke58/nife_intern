"""h-convergence figure for the crowned tier2b assembly.

Reads hconv_results.jsonl (produced by run_crown_hconvergence.sh) and produces:

  - results/biofilmQ_comstat/...  (no, this is FEM-side: writes alongside the jsonl)
    hconv_summary.csv             : LC × {crown,generic} × {C3D4,C3D10} table
    fig_crown_hconvergence.{pdf,png}
        Panel A: crestal p95 [MPa] vs LC (log-log), crown + generic, C3D4 + C3D10
        Panel B: same as A but with Richardson-extrapolated p95∞ as a horizontal line
                 per (kind, order); inset prints the apparent convergence order p̂.
        Panel C: moment-arm ratio (crown / generic) vs LC for both orders — the
                 thesis headline quantity. Reports its h-stability margin.

Richardson extrapolation (on the finest two LCs of each series):
    p95∞ ≈ p95_fine + (p95_fine - p95_coarse) / (r^p̂ - 1)
with refinement ratio r = LC_coarse / LC_fine and apparent order p̂ inferred from a
3-grid sequence (Roache 1994) when available, else falls back to p=2 (the theoretical
order for smooth quadratic elements on a regular problem; conservative for C3D4).

Usage:  python fig_crown_hconvergence.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JSONL = HERE / "hconv_results.jsonl"
CSV = HERE / "hconv_summary.csv"
PDF = HERE / "fig_crown_hconvergence.pdf"
PNG = HERE / "fig_crown_hconvergence.png"


def apparent_order(p_grid: np.ndarray, h_grid: np.ndarray) -> float:
    """Roache 3-grid apparent order p̂ = log((p1-p2)/(p2-p3)) / log(r), assuming uniform r.
    Returns NaN if data are noisy / non-monotone."""
    if len(p_grid) < 3:
        return float("nan")
    # Sort by h descending (coarse first)
    idx = np.argsort(-h_grid)
    p = p_grid[idx]
    h = h_grid[idx]
    r1 = h[0] / h[1]
    r2 = h[1] / h[2]
    if abs(r1 - r2) / r1 > 0.30:
        return float("nan")          # refinement not (roughly) uniform
    eps23 = p[1] - p[2]
    eps12 = p[0] - p[1]
    if eps12 == 0 or eps23 == 0:
        return float("nan")
    ratio = eps12 / eps23
    if ratio <= 0:                    # non-monotone → invalid
        return float("nan")
    return math.log(ratio) / math.log(r1)


def richardson(p_fine: float, p_coarse: float, r: float, p_hat: float) -> float:
    return p_fine + (p_fine - p_coarse) / (r ** p_hat - 1.0)


def main() -> int:
    if not JSONL.exists():
        print(f"missing {JSONL.name} — run run_crown_hconvergence.sh first")
        return 1
    rows = [json.loads(L) for L in JSONL.read_text().splitlines() if L.strip()]
    df = pd.DataFrame(rows).sort_values(["kind", "order", "lc"], ascending=[True, True, False])
    df.to_csv(CSV, index=False)
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    colors = {"crown": "#d62728", "generic": "#1f77b4"}
    markers = {"C3D4": "o", "C3D10": "s"}

    # ── Panel A & B: p95 vs LC ────────────────────────────────────────────────
    p_hat_table = []
    extra_table = []
    for kind in ("crown", "generic"):
        for order in ("C3D4", "C3D10"):
            sub = df[(df["kind"] == kind) & (df["order"] == order)].sort_values("lc", ascending=False)
            if sub.empty:
                continue
            h = sub["lc"].to_numpy()
            p = sub["crestal_p95_MPa"].to_numpy()
            label = f"{kind} {order}"
            for ax in (axes[0], axes[1]):
                ax.plot(h, p, marker=markers[order], color=colors[kind],
                        lw=1.4, ms=7, mfc="white" if order == "C3D4" else colors[kind],
                        label=label)
            p_hat = apparent_order(p, h)
            r = h[-2] / h[-1] if len(h) >= 2 else float("nan")
            p_inf = (richardson(p[-1], p[-2], r, p_hat if not math.isnan(p_hat) else 2.0)
                     if len(p) >= 2 else float("nan"))
            p_hat_table.append({"kind": kind, "order": order, "p_hat": p_hat,
                                "p95_finest": float(p[-1]), "p95_inf": p_inf, "lc_finest": float(h[-1])})
            if not math.isnan(p_inf):
                axes[1].axhline(p_inf, color=colors[kind], lw=0.8,
                                ls="--" if order == "C3D4" else ":")
            # absolute residual from extrapolated infinite-grid value
            if not math.isnan(p_inf):
                extra_table.append({"kind": kind, "order": order,
                                    "rel_err_at_LC04_pct": 100.0 * abs(
                                        float(sub[sub["lc"].round(2) == 0.40]["crestal_p95_MPa"].mean())
                                        - p_inf) / p_inf
                                    if (sub["lc"].round(2) == 0.40).any() else float("nan")})

    for ax, ttl in zip(axes[:2], (r"(A) crestal $p_{95}$ vs $h$",
                                  r"(B) Richardson extrapolation (dashed/dotted = $p_{95}^\infty$)")):
        ax.set_xscale("log"); ax.set_xlabel("crown tet edge LC [mm]")
        ax.set_ylabel(r"crestal peri-implant bone $p_{95}$ [MPa]")
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.grid(True, which="both", lw=0.4, alpha=0.4)
        ax.invert_xaxis()
        ax.legend(fontsize=8, loc="best", frameon=True)

    # ── Panel C: moment-arm ratio crown/generic vs LC ─────────────────────────
    ax = axes[2]
    for order in ("C3D4", "C3D10"):
        sub_c = df[(df["kind"] == "crown") & (df["order"] == order)].sort_values("lc", ascending=False)
        sub_g = df[(df["kind"] == "generic") & (df["order"] == order)].sort_values("lc", ascending=False)
        if sub_c.empty or sub_g.empty:
            continue
        # align on LC
        m = pd.merge(sub_c[["lc", "crestal_p95_MPa"]].rename(columns={"crestal_p95_MPa": "p_c"}),
                     sub_g[["lc", "crestal_p95_MPa"]].rename(columns={"crestal_p95_MPa": "p_g"}),
                     on="lc")
        ratio = m["p_c"] / m["p_g"]
        ax.plot(m["lc"], ratio, marker=markers[order], color="#2ca02c",
                lw=1.4, ms=7, mfc="white" if order == "C3D4" else "#2ca02c",
                label=f"crown/generic {order}")
        if len(ratio) >= 2:
            ax.axhline(ratio.iloc[-1], color="#2ca02c", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("crown tet edge LC [mm]")
    ax.set_ylabel(r"moment-arm ratio  $p_{95}^{\rm crown}/p_{95}^{\rm generic}$")
    ax.set_title(r"(C) moment-arm ratio $h$-stability", fontsize=11, loc="left")
    ax.invert_xaxis(); ax.grid(True, which="both", lw=0.4, alpha=0.4)
    ax.axhline(1.5, color="0.5", lw=0.6, ls="-")
    ax.text(0.99, 0.92, "thesis headline ×1.5",
            transform=ax.transAxes, ha="right", fontsize=8, color="0.4")
    ax.legend(fontsize=8, loc="lower left", frameon=True)

    fig.suptitle("Crown assembly h-convergence — thesis-metric crestal $p_{95}$", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PDF); fig.savefig(PNG, dpi=200)
    print(f"wrote {PDF.name} / {PNG.name}")

    if p_hat_table:
        ph = pd.DataFrame(p_hat_table)
        print("\nRichardson extrapolation (3-grid Roache p̂, else p=2 fallback):")
        print(ph.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
