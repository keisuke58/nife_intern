"""Bridge-regression PASS/FAIL figure (Stage A → B/C consistency check).

Reads bridge_regression_results.jsonl (produced by run_bridge_regression.sh) and renders a one-page
summary suitable for inclusion in the thesis as an audit log: each B/C-test row shows headline value,
regression value, relative error, and a coloured PASS/FAIL chip.

The figure is deliberately compact (single column, no caption needed) — the headline message is
"Stage A precision audit did not invalidate any cited B/C result" or, if any row fails, "row X has
shifted beyond the 5 % audit tolerance and the cited number needs a one-line update".

Usage:  python fig_bridge_regression.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
JSONL = HERE / "bridge_regression_results.jsonl"
PDF = HERE / "fig_bridge_regression.pdf"
PNG = HERE / "fig_bridge_regression.png"
CSV = HERE / "bridge_regression_summary.csv"
TOL = 0.05      # 5 % audit tolerance


def main() -> int:
    if not JSONL.exists():
        print(f"missing {JSONL.name} — run run_bridge_regression.sh first")
        return 1
    rows = [json.loads(L) for L in JSONL.read_text().splitlines() if L.strip()]
    df = pd.DataFrame(rows)
    df["rel_err_pct"] = df["rel_err"] * 100.0
    df.to_csv(CSV, index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.0, 0.55 + 0.40 * len(df)))
    ax.axis("off")
    headers = ["test", "headline", "regression", "Δ %", "status"]
    cell_text = []
    cell_colors = []
    for _, r in df.iterrows():
        chip = "PASS" if r["pass"] else "FAIL"
        cell_text.append([str(r["test"]), f"{r['headline']:.4g}", f"{r['observed']:.4g}",
                          f"{r['rel_err_pct']:+.2f}", chip])
        row_color = "#d8efd0" if r["pass"] else "#fbd5d5"
        cell_colors.append([row_color] * len(headers))

    table = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                     cellLoc="center", colColours=["#dddddd"] * len(headers), loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.4)

    overall = "PASS" if df["pass"].all() else "FAIL"
    fig.suptitle(f"Bridge regression after Stage A — {overall}  "
                 f"({df['pass'].sum()}/{len(df)} within ±{TOL*100:.0f} %)",
                 fontsize=12, y=0.99 if len(df) > 3 else 0.96)

    fig.tight_layout()
    fig.savefig(PDF); fig.savefig(PNG, dpi=200)
    print(f"wrote {PDF.name} / {PNG.name}")
    return 0 if df["pass"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
