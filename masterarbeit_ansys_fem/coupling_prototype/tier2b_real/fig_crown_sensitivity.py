"""Tornado plot for the crowned headline material-OAT sensitivity sweep.

Reads sens_results.jsonl (produced by run_crown_sensitivity.sh) plus the baseline crestal-p95 (read
either from hconv_results.jsonl at LC=0.40 C3D4 or from the legacy crestal_p95_thesis_metric.py output)
and produces, per kind (crown / generic):

  fig_crown_sensitivity.{pdf,png}
      Two-panel tornado (crown left, generic right). Each row is a perturbed material; bar length is
      the Δp95 (low → high) deviation from baseline. Rows are sorted by absolute bar length so the
      driver material is on top. Numbers next to each bar are the relative change (%) per E-doubling.

  sens_summary.csv
      Wide table material × {baseline, low, high, Δlow, Δhigh, sensitivity_pct_per_doubling}.

The "sensitivity per E-doubling" is the average of |p95(0.5x) - p95(1.0x)| / p95(1.0x) and
|p95(2.0x) - p95(1.0x)| / p95(1.0x), i.e. the typical % swing of the crestal p95 when that material's E
varies by a factor of 2 in either direction. The material with the largest sensitivity is the
one to refine measurements/modelling for first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SENS_JSONL = HERE / "sens_results.jsonl"
HCONV_JSONL = HERE / "hconv_results.jsonl"
CSV = HERE / "sens_summary.csv"
PDF = HERE / "fig_crown_sensitivity.pdf"
PNG = HERE / "fig_crown_sensitivity.png"


def load_baseline() -> dict[str, float]:
    """Return {kind: baseline_p95} for crown/generic at headline LC=0.40 C3D4.

    Prefers hconv_results.jsonl LC=0.40 C3D4 entries; otherwise falls back to the values from the
    canonical audit note (C3D4 headline: crown=27.0, generic=17.2 MPa)."""
    if HCONV_JSONL.exists():
        rows = [json.loads(L) for L in HCONV_JSONL.read_text().splitlines() if L.strip()]
        base = {}
        for r in rows:
            if abs(r["lc"] - 0.40) < 1e-6 and r["order"] == "C3D4":
                base[r["kind"]] = r["crestal_p95_MPa"]
        if {"crown", "generic"} <= base.keys():
            return base
    return {"crown": 27.0, "generic": 17.2}      # audit note fallback


def main() -> int:
    if not SENS_JSONL.exists():
        print(f"missing {SENS_JSONL.name} — run run_crown_sensitivity.sh first")
        return 1

    rows = [json.loads(L) for L in SENS_JSONL.read_text().splitlines() if L.strip()]
    df = pd.DataFrame(rows)
    base = load_baseline()
    print(f"baseline (headline LC=0.40 C3D4): {base}")

    summary = []
    for kind in ("crown", "generic"):
        b = base[kind]
        for mat, sub in df[df.kind == kind].groupby("mat"):
            lo = float(sub[sub.mul.round(2) == 0.50]["crestal_p95_MPa"].mean())
            hi = float(sub[sub.mul.round(2) == 2.00]["crestal_p95_MPa"].mean())
            d_lo = lo - b; d_hi = hi - b
            sens = 0.5 * (abs(d_lo) + abs(d_hi)) / b * 100.0
            summary.append({"kind": kind, "mat": mat, "p95_base": b,
                            "p95_low": lo, "p95_high": hi,
                            "delta_low": d_lo, "delta_high": d_hi,
                            "sens_pct_per_doubling": sens})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(CSV, index=False)
    print(sdf.to_string(index=False))

    # ── Tornado ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=False)
    bar_color = {"crown": "#d62728", "generic": "#1f77b4"}
    for ax, kind in zip(axes, ("crown", "generic")):
        sk = sdf[sdf.kind == kind].copy()
        sk["abs_swing"] = np.maximum(np.abs(sk.delta_low), np.abs(sk.delta_high))
        sk = sk.sort_values("abs_swing")
        y = np.arange(len(sk))
        # draw bars from low → high (sign-preserving so direction is visible)
        for i, (_, row) in enumerate(sk.iterrows()):
            ax.barh(i, row["delta_high"], color=bar_color[kind], alpha=0.85,
                    edgecolor="black", lw=0.5, label="E × 2.0" if i == 0 else "")
            ax.barh(i, row["delta_low"], color=bar_color[kind], alpha=0.45,
                    edgecolor="black", lw=0.5, hatch="//",
                    label="E × 0.5" if i == 0 else "")
            ax.text(max(row["delta_low"], row["delta_high"]) + 0.3, i,
                    f"{row['sens_pct_per_doubling']:.1f} %", va="center", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels(sk.mat)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel(r"$\Delta p_{95}$ [MPa] vs. baseline")
        ax.set_title(f"{kind}  (baseline = {sdf[sdf.kind==kind]['p95_base'].iloc[0]:.1f} MPa)",
                     fontsize=11)
        ax.legend(fontsize=8, loc="lower right", frameon=True)
        ax.grid(True, axis="x", lw=0.4, alpha=0.4)

    fig.suptitle("Material-parameter OAT sensitivity — crestal $p_{95}$ swing per E-doubling",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PDF); fig.savefig(PNG, dpi=200)
    print(f"wrote {PDF.name} / {PNG.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
