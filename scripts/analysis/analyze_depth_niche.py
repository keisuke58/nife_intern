#!/usr/bin/env python3
"""
analyze_depth_niche.py — depth ecology of the HOBIC FISH biofilms.

From the z-profile CSV (depth × 5 species, per condition/day) it computes, with
no fitting required:

  (1) Depth niche — each species' centre-of-mass depth (intensity-weighted mean
      z, µm) over time, CH vs DH. Tests e.g. whether P. gingivalis sinks into the
      anaerobic deep layers as dysbiosis develops.   -> depth_niche.png
  (2) CH-vs-DH divergence — per normalised-depth × day Bray-Curtis distance
      between the commensal and dysbiotic community fractions: when/where does
      dysbiosis diverge?                              -> ch_dh_divergence.png

Run from the repo root:
    python scripts/analysis/analyze_depth_niche.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})

HERE = Path(__file__).resolve().parents[2]
FITDIR = HERE / "results" / "diffusion_fit"
CSV = FITDIR / "zprofiles_all_ti.csv"
SP = ["S.oralis", "A.naeslundii", "Vd/Vp", "F.nucleatum", "P.gingivalis"]
COL = {"S.oralis": "#1f77b4", "A.naeslundii": "#2ca02c", "Vd/Vp": "#bcbd22",
       "F.nucleatum": "#9467bd", "P.gingivalis": "#d62728"}


def fractions(df):
    M = df[SP].values.astype(float)
    s = M.sum(1, keepdims=True); s[s == 0] = 1.0
    return M / s


def com_depth(df):
    """Intensity-weighted centre-of-mass depth (µm) per species."""
    g = df.sort_values("z_um"); z = g.z_um.values
    out = {}
    for sp in SP:
        w = np.maximum(g[sp].values.astype(float), 0)
        out[sp] = float((w * z).sum() / w.sum()) if w.sum() > 0 else np.nan
    return out


def main():
    df = pd.read_csv(CSV)
    conds = ["CH", "DH"]
    days = sorted(df.day.unique())

    # ---- (1) depth niche: CoM depth vs day, per species, CH vs DH ----
    rows = []
    fig1, ax = plt.subplots(figsize=(7.2, 5.0))
    for cond, ls, mk in [("CH", "-", "o"), ("DH", "--", "s")]:
        for sp in SP:
            ys = []
            for d in days:
                sub = df[(df.condition == cond) & (df.day == d)]
                cz = com_depth(sub)[sp] if len(sub) else np.nan
                ys.append(cz); rows.append({"cond": cond, "day": d, "species": sp, "com_depth_um": cz})
            ax.plot(days, ys, ls, marker=mk, color=COL[sp], lw=2, ms=5,
                    label=(sp if cond == "CH" else None))
    ax.invert_yaxis()
    ax.set_xlabel("day", fontsize=12); ax.set_ylabel("centre-of-mass depth (µm)", fontsize=12)
    ax.set_title("Depth niche over time — CH (solid) vs DH (dashed)", fontsize=13)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=COL[s], lw=2, label=s) for s in SP]
    h += [Line2D([0], [0], color="k", lw=2, ls="-", marker="o", label="CH"),
          Line2D([0], [0], color="k", lw=2, ls="--", marker="s", label="DH")]
    ax.legend(handles=h, fontsize=8, ncol=2, loc="best")
    fig1.tight_layout(); fig1.savefig(FITDIR / "depth_niche.png", dpi=140, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(FITDIR / "depth_niche.csv", index=False)

    # ---- (2) CH-vs-DH Bray-Curtis divergence per normalised depth × day ----
    ngrid = int(df[(df.condition == "CH") & (df.day == days[0])].shape[0])
    BC = np.full((ngrid, len(days)), np.nan)
    for j, d in enumerate(days):
        ch = df[(df.condition == "CH") & (df.day == d)].sort_values("z_um")
        dh = df[(df.condition == "DH") & (df.day == d)].sort_values("z_um")
        if len(ch) != ngrid or len(dh) != ngrid:
            continue
        fch, fdh = fractions(ch), fractions(dh)
        BC[:, j] = 0.5 * np.abs(fch - fdh).sum(axis=1)   # Bray-Curtis on fractions, 0..1

    fig2, (axt, axh) = plt.subplots(2, 1, figsize=(6.8, 6.2),
                                    gridspec_kw={"height_ratios": [1, 3]}, sharex=True)
    axt.plot(range(len(days)), np.nanmean(BC, axis=0), "-o", color="#222", lw=2)
    axt.set_ylabel("mean BC", fontsize=11); axt.set_ylim(0, max(0.3, np.nanmax(BC) * 1.05))
    axt.set_title("CH vs DH community divergence (Bray-Curtis on depth fractions)", fontsize=13)
    im = axh.imshow(BC, aspect="auto", cmap="magma", vmin=0, vmax=np.nanmax(BC),
                    extent=[-0.5, len(days) - 0.5, 1, 0])
    axh.set_xticks(range(len(days))); axh.set_xticklabels(days)
    axh.set_xlabel("day", fontsize=12); axh.set_ylabel("normalised depth\n(0 = surface)", fontsize=12)
    cb = fig2.colorbar(im, ax=axh, fraction=0.046, pad=0.04); cb.set_label("Bray-Curtis", fontsize=10)
    fig2.tight_layout(); fig2.savefig(FITDIR / "ch_dh_divergence.png", dpi=140, bbox_inches="tight")

    # ---- console summary ----
    print("=== depth niche: P.gingivalis CoM depth (µm), CH vs DH ===")
    for d in days:
        chz = com_depth(df[(df.condition == "CH") & (df.day == d)])["P.gingivalis"]
        dhz = com_depth(df[(df.condition == "DH") & (df.day == d)])["P.gingivalis"]
        print(f"  day {d:>2}:  CH {chz:5.1f}   DH {dhz:5.1f}   (Δ {dhz - chz:+.1f} µm)")
    print("\n=== CH-vs-DH divergence (depth-averaged Bray-Curtis) per day ===")
    for j, d in enumerate(days):
        print(f"  day {d:>2}: {np.nanmean(BC[:, j]):.3f}")
    print(f"\nwrote depth_niche.png / .csv and ch_dh_divergence.png to {FITDIR}")


if __name__ == "__main__":
    main()
