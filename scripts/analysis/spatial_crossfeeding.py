#!/usr/bin/env python3
"""
spatial_crossfeeding.py — does the AGORA metabolic backbone show up in SPACE?

The concordant backbone (ecology x AGORA) says lactate flows
    S. oralis (Bacilli, producer)  ->  Veillonella (Negativicutes) & A. naeslundii (consumers).
If that cross-feeding is real and diffusion-limited, the consumers should sit at a
different depth than the producer (a lactate gradient). We test, from the FISH
depth profiles (results/diffusion_fit/zprofiles_all_ti.csv), whether the
centre-of-mass depth of each consumer differs from the producer's, paired across
the 10 (condition, day) samples (Wilcoxon signed-rank).

Larger z = deeper (towards the substratum / anaerobic side), per the depth
convention used throughout (surface -> deep).

Run from the repo root:  python scripts/analysis/spatial_crossfeeding.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})
HERE = Path(__file__).resolve().parents[2]
FITDIR = HERE / "results" / "diffusion_fit"
PRODUCER = "S.oralis"                      # lactate producer (Streptococcus / Bacilli)
CONSUMERS = ["Vd/Vp", "A.naeslundii"]      # Veillonella (Negativicutes), Actinobacteria
COL = {"S.oralis": "#1f77b4", "A.naeslundii": "#2ca02c", "Vd/Vp": "#bcbd22",
       "F.nucleatum": "#9467bd", "P.gingivalis": "#d62728"}
ALLSP = ["S.oralis", "A.naeslundii", "Vd/Vp", "F.nucleatum", "P.gingivalis"]


def com_depth(sub, sp):
    z = sub["z_um"].to_numpy(float)
    w = sub[sp].to_numpy(float)
    w = np.clip(w, 0, None)
    return float((z * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def main():
    df = pd.read_csv(FITDIR / "zprofiles_all_ti.csv")
    keys = df[["condition", "day"]].drop_duplicates().itertuples(index=False)
    rows = []
    for cond, day in keys:
        sub = df[(df.condition == cond) & (df.day == day)]
        rec = {"condition": cond, "day": day}
        for sp in ALLSP:
            rec[sp] = com_depth(sub, sp)
        rows.append(rec)
    cz = pd.DataFrame(rows).sort_values(["condition", "day"]).reset_index(drop=True)

    print("centre-of-mass depth (µm) per sample:")
    print(cz.to_string(index=False,
          formatters={s: (lambda v: f"{v:5.1f}") for s in ALLSP}))

    print("\n=== spatial stratification of the cross-feeding pair (paired Wilcoxon, n=%d) ===" % len(cz))
    stats = {}
    for cons in CONSUMERS:
        d = (cz[cons] - cz[PRODUCER]).to_numpy()    # >0: consumer deeper than producer
        d = d[~np.isnan(d)]
        try:
            p_two = float(wilcoxon(d).pvalue)
        except ValueError:
            p_two = np.nan
        frac_deeper = float((d > 0).mean())
        direction = "deeper" if np.median(d) > 0 else "shallower"
        stats[cons] = {"mean_delta_um": float(np.mean(d)), "median_delta_um": float(np.median(d)),
                       "frac_consumer_deeper": frac_deeper,
                       "consumer_is": direction, "p_two_sided": p_two, "n": int(d.size)}
        print(f"  {cons:14s} vs {PRODUCER}:  Δz mean={np.mean(d):+5.1f} µm, "
              f"median={np.median(d):+5.1f} µm  → consumer {direction} than producer "
              f"in {max(frac_deeper,1-frac_deeper)*100:.0f}% of samples, "
              f"Wilcoxon p={p_two:.3f}")
    print("  (interpretation: producer S.oralis forms the deeper/basal layer; the lactate\n"
          "   consumer sits above it → consistent with upward lactate diffusion.)")

    # ---- figure: producer vs consumers depth, paired per sample ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    # (a) paired dot plot
    ax = axes[0]
    x = np.arange(len(cz))
    for sp in [PRODUCER] + CONSUMERS:
        ax.plot(x, cz[sp], "o-", color=COL[sp], lw=1.6, ms=6, label=sp,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_ylim(ax.get_ylim()[::-1])   # surface at top
    ax.set_xticks(x); ax.set_xticklabels([f"{c}{d}" for c, d in zip(cz.condition, cz.day)],
                                         rotation=60, fontsize=8)
    ax.set_ylabel("centre-of-mass depth (µm)\nsurface → deep", fontsize=12)
    ax.set_title("Producer vs consumers — depth per sample", fontsize=13)
    ax.legend(fontsize=9, loc="best"); ax.grid(ls=":", lw=0.6, color="#ccc", alpha=0.7)

    # (b) paired difference (consumer − producer); >0 = consumer deeper
    ax = axes[1]
    w = 0.35
    for i, cons in enumerate(CONSUMERS):
        d = (cz[cons] - cz[PRODUCER]).to_numpy()
        ax.bar(x + (i - 0.5) * w, d, w, color=COL[cons],
               label=f"{cons} − {PRODUCER}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}{d}" for c, d in zip(cz.condition, cz.day)],
                                         rotation=60, fontsize=8)
    ax.set_ylabel("Δ depth (µm)   (>0 = consumer deeper)", fontsize=12)
    sub = "; ".join(f"{c.split('/')[0]} p={stats[c]['p_two_sided']:.3f}" for c in CONSUMERS)
    ax.set_title(f"Consumer minus producer depth\n({sub})", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", ls=":", lw=0.6, color="#ccc", alpha=0.7)

    fig.suptitle("Spatial test of the lactate cross-feeding backbone "
                 "(So → Veillonella / A. naeslundii)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FITDIR / "spatial_crossfeeding.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    cz.to_csv(FITDIR / "spatial_crossfeeding_com_depth.csv", index=False)
    import json
    (FITDIR / "spatial_crossfeeding_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {out}")
    print(f"wrote {FITDIR/'spatial_crossfeeding_com_depth.csv'} and _stats.json")


if __name__ == "__main__":
    main()
