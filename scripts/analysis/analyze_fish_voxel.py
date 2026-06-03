#!/usr/bin/env python3
"""
analyze_fish_voxel.py — voxel-level ecology from the HOBIC FISH .lif (one pass).

Decodes every Ti FOV to the 5 species once, then produces:

  (3b) F.nucleatum–P.gingivalis co-localisation per (condition, day): Pearson over
       voxels + Manders M1 (fraction of Pg signal sitting where Fn is present).
       Tests the bridging hypothesis (Fn associates with the pathogen more in DH).
                                                       -> fn_pg_coloc.png
  (7)  Bootstrap CI on the P.gingivalis depth-fraction profile (resample FOVs):
       honest uncertainty — DH late days have only 2 FOVs -> very wide bands.
                                                       -> bootstrap_ci_pg.png
  (4') True total biomass per species (∑ decoded voxel intensity) per (cond, day).
                                                       -> printed / voxel_biomass.csv

Heavy (loads full volumes). Run from the repo root:
    python scripts/analysis/analyze_fish_voxel.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_root))                       # repo root (fish_decode)
_sys.path.insert(0, str(_root / "scripts" / "pde"))   # lif_to_zprofiles helpers

import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from readlif.reader import LifFile
from fish_decode import SPECIES_ORDER, channel_luts, decode, norm_scalars
from lif_to_zprofiles import (series_volume, condition_from_filename,
                              series_day, series_substrate, parse_day)

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})

HERE = Path(__file__).resolve().parents[2]
FITDIR = HERE / "results" / "diffusion_fit"
COL = {"S.oralis": "#1f77b4", "A.naeslundii": "#2ca02c", "Vd/Vp": "#bcbd22",
       "F.nucleatum": "#9467bd", "P.gingivalis": "#d62728"}
NGRID = 40


def fov_metrics(img, luts):
    """Return (zprofile_grid(40,5), coloc dict, biomass dict) for one FOV (Ti)."""
    vol = series_volume(img)                       # (C, Z, Y, X)
    sp = decode(vol, luts, norm_scalars(vol, luts))  # {species:(Z,Y,X)}
    nz = vol.shape[1]
    grid = np.linspace(0, 1, NGRID)
    prof = np.zeros((NGRID, len(SPECIES_ORDER)))
    biomass = {}
    for k, name in enumerate(SPECIES_ORDER):
        a = sp.get(name)
        if a is None:
            continue
        zp = a.reshape(nz, -1).mean(axis=1)        # xy-mean per z
        prof[:, k] = np.interp(grid, np.linspace(0, 1, nz), zp)
        biomass[name] = float(a.sum())
    fn, pg = sp.get("F.nucleatum"), sp.get("P.gingivalis")
    coloc = {}
    if fn is not None and pg is not None:
        f, p = fn.ravel(), pg.ravel()
        if f.std() > 0 and p.std() > 0:
            coloc["pearson"] = float(np.corrcoef(f, p)[0, 1])
        thr = 0.1                                   # Fn "present" threshold (normalised)
        denom = float(pg.sum())
        coloc["manders_pg_on_fn"] = float(pg[fn > thr].sum() / denom) if denom > 0 else np.nan
    return prof, coloc, biomass


def main():
    files = sorted(glob.glob(str(HERE / "HOBIC FISH" / "*.lif")))
    profiles = defaultdict(list)     # (cond,day) -> list of (40,5)
    coloc_rows, biomass = [], defaultdict(lambda: defaultdict(float))
    for path in files:
        cond = condition_from_filename(path)
        if not cond:
            continue
        lif = LifFile(path)
        imgs = list(lif.get_iter_image())
        luts = channel_luts(lif, imgs[0].channels)
        fday = parse_day(Path(path).stem, None)
        for img in imgs:
            if series_substrate(img.name) == "glass":   # Ti + unlabelled only
                continue
            day = series_day(img.name, path, fday)
            prof, coloc, bm = fov_metrics(img, luts)
            profiles[(cond, day)].append(prof)
            if "pearson" in coloc:
                coloc_rows.append({"cond": cond, "day": day, **coloc})
            for sp, v in bm.items():
                biomass[(cond, day)][sp] += v
        print(f"  {Path(path).name}: done ({cond})", flush=True)

    cdf = pd.DataFrame(coloc_rows)
    cdf.to_csv(FITDIR / "fn_pg_coloc.csv", index=False)
    days = sorted({d for _, d in profiles})

    # ---- (3b) Fn-Pg colocalisation: Manders M1 CH vs DH over day ----
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for cond, mk, ls in [("CH", "o", "-"), ("DH", "s", "--")]:
        m, e = [], []
        for d in days:
            g = cdf[(cdf.cond == cond) & (cdf.day == d)]["manders_pg_on_fn"].dropna()
            m.append(g.mean() if len(g) else np.nan)
            e.append(g.std() if len(g) > 1 else 0.0)
        ax.errorbar(days, m, yerr=e, fmt=mk, ls=ls, color="#444" if cond == "CH" else "#c0392b",
                    lw=2, ms=6, capsize=3, label=cond)
    ax.set_xlabel("day", fontsize=12)
    ax.set_ylabel("Manders M1: fraction of Pg signal\nco-localised with F.nucleatum", fontsize=11)
    ax.set_title("F.nucleatum–P.gingivalis co-localisation (bridging) — CH vs DH", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(FITDIR / "fn_pg_coloc.png", dpi=140, bbox_inches="tight")

    # ---- (7) bootstrap CI on Pg depth-fraction profile ----
    rng = np.random.default_rng(0)
    pg_idx = SPECIES_ORDER.index("P.gingivalis")
    grid = np.linspace(0, 1, NGRID)
    fig2, axes = plt.subplots(1, len(days), figsize=(2.7 * len(days), 4.0), squeeze=False, sharey=True)
    for j, d in enumerate(days):
        ax = axes[0][j]
        for cond, c in [("CH", "#1f77b4"), ("DH", "#d62728")]:
            P = profiles.get((cond, d), [])
            if not P:
                continue
            P = np.stack(P)                                  # (nfov,40,5)
            fr = P / np.clip(P.sum(axis=2, keepdims=True), 1e-9, None)
            pg = fr[:, :, pg_idx]                            # (nfov,40)
            n = pg.shape[0]
            boot = pg[rng.integers(0, n, size=(2000, n))].mean(axis=1)   # (2000,40)
            lo, mid, hi = np.percentile(boot, [5, 50, 95], axis=0)
            ax.plot(mid, grid, color=c, lw=2, label=f"{cond} (n={n})")
            ax.fill_betweenx(grid, lo, hi, color=c, alpha=0.2)
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_title(f"day {d}", fontsize=12)
        ax.set_xlabel("Pg fraction", fontsize=11)
        if j == 0:
            ax.set_ylabel("normalised depth (0=surface)", fontsize=11)
        ax.legend(fontsize=8, loc="best")
    fig2.suptitle("P.gingivalis depth fraction — FOV-bootstrap 90% CI (DH late days: n=2 → wide)",
                  fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    fig2.savefig(FITDIR / "bootstrap_ci_pg.png", dpi=140, bbox_inches="tight")

    # ---- (4') true total biomass ----
    brows = []
    for (cond, d), bm in sorted(biomass.items()):
        r = {"cond": cond, "day": d, **{k: bm.get(k, 0.0) for k in SPECIES_ORDER}}
        r["total"] = sum(bm.values()); brows.append(r)
    pd.DataFrame(brows).to_csv(FITDIR / "voxel_biomass.csv", index=False)

    print("\n=== Fn-Pg Manders M1 (mean) CH vs DH ===")
    for d in days:
        ch = cdf[(cdf.cond == "CH") & (cdf.day == d)]["manders_pg_on_fn"].mean()
        dh = cdf[(cdf.cond == "DH") & (cdf.day == d)]["manders_pg_on_fn"].mean()
        print(f"  day {d:>2}: CH {ch:.3f}  DH {dh:.3f}")
    print(f"\nwrote fn_pg_coloc.png/.csv, bootstrap_ci_pg.png, voxel_biomass.csv to {FITDIR}")


if __name__ == "__main__":
    main()
