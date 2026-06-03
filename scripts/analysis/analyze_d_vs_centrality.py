#!/usr/bin/env python3
"""
analyze_d_vs_centrality.py — does a species' spatial diffusivity relate to its
role in the ecological (gLV) interaction matrix?

For each condition it loads the 5x5 gLV A matrix (nsp_pde_1d_heine.load_A_b) and
the fitted diffusivities (D_fit_<cond>.json), computes per-species interaction
strengths (out = Σ_j|A_ij|, in = Σ_j|A_ji|, total) and correlates them with D_i.
Hypothesis: "central" species (high interaction strength) diffuse less.

NB: only 5 species → correlations are illustrative, not powered. Uses whatever
D_fit_<cond>.json is current (preliminary until the converged sweep result is
promoted). Run from the repo root.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsp_pde_1d_heine import load_A_b

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})
HERE = Path(__file__).resolve().parents[2]
FITDIR = HERE / "results" / "diffusion_fit"
SHORT = ["S.o", "A.n", "Vd/Vp", "F.n", "P.g"]
COLS = ["#1f77b4", "#2ca02c", "#bcbd22", "#9467bd", "#d62728"]


def main():
    conds = ["CH", "DH"]
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.6), squeeze=False)
    for ci, cond in enumerate(conds):
        jf = FITDIR / f"D_fit_{cond}.json"
        if not jf.exists():
            print(f"  [skip] {jf.name} missing"); continue
        D = np.array(json.loads(jf.read_text())["D_fit"])
        A, _ = load_A_b(cond)
        A = np.asarray(A)
        off = A - np.diag(np.diag(A))
        out_str = np.abs(off).sum(axis=1)     # Σ_j|A_ij|  (influence on others)
        in_str = np.abs(off).sum(axis=0)      # Σ_j|A_ji|  (influenced by others)
        total = out_str + in_str
        r = np.corrcoef(total, D)[0, 1]
        ax = axes[0][ci]
        for k in range(len(SHORT)):
            ax.scatter(total[k], D[k], s=90, color=COLS[k], zorder=3)
            ax.annotate(SHORT[k], (total[k], D[k]), fontsize=10,
                        xytext=(5, 4), textcoords="offset points")
        # trend line
        if np.ptp(total) > 0:
            m, q = np.polyfit(total, D, 1)
            xs = np.array([total.min(), total.max()])
            ax.plot(xs, m * xs + q, "--", color="#888", lw=1.5, zorder=2)
        ax.set_title(f"{cond}: D vs gLV interaction strength  (r={r:+.2f})", fontsize=12)
        ax.set_xlabel("total |A| interaction strength (in+out)", fontsize=11)
        if ci == 0:
            ax.set_ylabel("fitted diffusivity D", fontsize=11)
        print(f"  {cond}: corr(total|A|, D) = {r:+.3f}   "
              f"D={np.round(D,4)}  total|A|={np.round(total,3)}")
    fig.suptitle("Spatial diffusivity vs ecological centrality (5 species — illustrative)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FITDIR / "d_vs_centrality.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
