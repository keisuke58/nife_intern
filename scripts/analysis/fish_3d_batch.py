#!/usr/bin/env python3
"""
fish_3d_batch.py — batch 3-D (x,y,z) extraction over ALL HOBIC .lif FOVs.

For every titanium FOV across all .lif files it decodes the 5 species per voxel
(no xy-averaging) and measures the LATERAL structure that the depth pipeline drops:
  - lateral heterogeneity per species  = CV of the xy (top-down) projection
  - Fn–Pg lateral patch co-localisation = Manders M1 on the xy projections
  - Fn–Pg 3-D co-localisation           = Manders M1 on the full voxels
  - per-species voxel biomass
Results are pooled by (condition, day) → CH vs DH lateral time-evolution figures,
and one downsampled phi(x,y,z,5) per (condition, day) is saved as a 3-D PDE
initial state (results/fish_3d/ic/).

Memory-safe: one FOV at a time, decoded at a stride for speed (--decode-ds).
Run from repo root:  python scripts/analysis/fish_3d_batch.py [--decode-ds 2]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_root)); _sys.path.insert(0, str(_root / "scripts" / "pde"))

import argparse, gc, glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from readlif.reader import LifFile

from fish_decode import SPECIES_ORDER, SPECIES_RGB, channel_luts, decode, norm_scalars
from lif_to_zprofiles import (series_volume, condition_from_filename, series_day,
                              series_substrate, parse_day)

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "results" / "fish_3d"
ICDIR = OUT / "ic"
SHORT = {"S.oralis": "S.o", "A.naeslundii": "A.n", "Vd/Vp": "Vd/Vp",
         "F.nucleatum": "F.n", "P.gingivalis": "P.g"}


def manders_m1(a, b, thr_frac=0.02):
    """fraction of a-signal that overlaps b (b above thr_frac*max(b))."""
    a = np.clip(a, 0, None); b = np.clip(b, 0, None)
    tot = a.sum()
    if tot <= 0:
        return np.nan
    mask = b > (thr_frac * b.max() if b.max() > 0 else 0)
    return float(a[mask].sum() / tot)


def cv(x):
    x = np.clip(x, 0, None)
    m = x.mean()
    return float(x.std() / m) if m > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode-ds", type=int, default=2, dest="dds",
                    help="xy stride applied to the stack before decode (speed; 2 => 512x512)")
    ap.add_argument("--ic-ds", type=int, default=8, dest="ic_ds",
                    help="extra xy downsample for the saved IC arrays")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); ICDIR.mkdir(parents=True, exist_ok=True)
    s = args.dds

    rows = []
    zrows = []                                  # per-FOV depth profiles phi_sp(z) (the A-track extension)
    ic_saved = set()
    files = sorted(glob.glob(str(HERE / "HOBIC FISH" / "*.lif")))
    for path in files:
        cond = condition_from_filename(path)
        if not cond:
            continue
        try:
            lif = LifFile(path); imgs = list(lif.get_iter_image())
        except Exception as e:
            print(f"  [skip] {Path(path).name}: {e}", flush=True); continue
        luts = channel_luts(lif, imgs[0].channels)
        fday = parse_day(Path(path).stem, None)
        for fi, img in enumerate(imgs):
            if series_substrate(img.name) == "glass":
                continue
            day = series_day(img.name, path, fday)
            try:
                vol = series_volume(img)            # (C,Z,Y,X)
                if s > 1:
                    vol = vol[:, :, ::s, ::s]
                dec = decode(vol, luts, norm_scalars(vol, luts))   # {sp:(Z,Y,X)}
            except Exception as e:
                print(f"  [skip FOV] {Path(path).name}#{fi}: {e}", flush=True); continue
            rec = {"cond": cond, "day": int(day), "file": Path(path).stem, "fov": fi}
            xy = {sp: dec[sp].mean(axis=0) for sp in SPECIES_ORDER}     # (Y,X) lateral
            for sp in SPECIES_ORDER:
                rec[f"latCV_{SHORT[sp]}"] = cv(xy[sp])
                rec[f"biomass_{SHORT[sp]}"] = float(np.clip(dec[sp], 0, None).sum())
            rec["FnPg_patch_M1"] = manders_m1(xy["P.gingivalis"], xy["F.nucleatum"])  # lateral
            rec["FnPg_3d_M1"] = manders_m1(dec["P.gingivalis"], dec["F.nucleatum"])    # 3D
            rows.append(rec)

            # --- per-FOV depth profile: xy-mean occupancy per z slice, per species (A-track) ---
            zprof = {sp: np.clip(dec[sp], 0, None).mean(axis=(1, 2)) for sp in SPECIES_ORDER}  # each (Z,)
            tot_z = np.sum([zprof[sp] for sp in SPECIES_ORDER], axis=0) + 1e-12
            nz = len(tot_z)
            for zi in range(nz):
                zr = {"cond": cond, "day": int(day), "file": Path(path).stem, "fov": fi,
                      "z_idx": zi, "z_frac": zi / max(1, nz - 1)}
                for sp in SPECIES_ORDER:
                    zr[SHORT[sp]] = float(zprof[sp][zi] / tot_z[zi])      # occupancy fraction at depth z
                zrows.append(zr)

            key = (cond, int(day))
            if key not in ic_saved:                # save one IC per (cond,day)
                d2 = max(1, args.ic_ds // max(1, s))
                arr = np.stack([dec[sp][:, ::d2, ::d2] for sp in SPECIES_ORDER], axis=-1)
                arr = np.transpose(arr, (2, 1, 0, 3)).astype(np.float32)   # (X,Y,Z,5)
                np.save(ICDIR / f"phi_xyz_{cond}_d{int(day)}.npy", arr)
                ic_saved.add(key)
            del vol, dec, xy; gc.collect()
        print(f"  {Path(path).name}: done ({cond})", flush=True)

    df = pd.DataFrame(rows).sort_values(["cond", "day", "fov"]).reset_index(drop=True)
    df.to_csv(OUT / "fish_3d_lateral_metrics.csv", index=False)
    print(f"\nwrote {OUT/'fish_3d_lateral_metrics.csv'}  ({len(df)} FOVs)")

    zdf = pd.DataFrame(zrows).sort_values(["cond", "day", "fov", "z_idx"]).reset_index(drop=True)
    zdf.to_csv(OUT / "zprofiles_per_fov.csv", index=False)
    print(f"wrote {OUT/'zprofiles_per_fov.csv'}  ({zdf['fov'].count()} z-rows, "
          f"{len(zdf.groupby(['cond','day','file','fov']))} FOV profiles)")
    print(df.groupby(["cond", "day"]).size().to_string())

    # ---- pooled (cond,day) means ----
    g = df.groupby(["cond", "day"]).mean(numeric_only=True).reset_index()

    # Fig A: Fn-Pg lateral patch coloc + 3D coloc over time, CH vs DH
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for cond, c in [("CH", "#1f77b4"), ("DH", "#d62728")]:
        sub = g[g.cond == cond].sort_values("day")
        if len(sub):
            ax[0].plot(sub.day, sub.FnPg_patch_M1, "o-", color=c, lw=2, label=cond)
            ax[1].plot(sub.day, sub.FnPg_3d_M1, "o-", color=c, lw=2, label=cond)
    ax[0].set_title("Fn–Pg LATERAL patch co-localisation (xy)", fontsize=12)
    ax[1].set_title("Fn–Pg 3-D co-localisation (voxels)", fontsize=12)
    for a in ax:
        a.set_xlabel("day"); a.set_ylabel("Manders $M_1$ (Pg with Fn)"); a.legend(); a.grid(ls=":", alpha=0.6)
    fig.suptitle("Fn–Pg co-localisation over time — CH vs DH (all FOVs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fish_3d_fnpg_coloc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig B: lateral heterogeneity (mean species CV) over time, CH vs DH
    cols = [f"latCV_{SHORT[sp]}" for sp in SPECIES_ORDER]
    g["latCV_mean"] = g[cols].mean(axis=1)
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
    for cond, c in [("CH", "#1f77b4"), ("DH", "#d62728")]:
        sub = g[g.cond == cond].sort_values("day")
        if len(sub):
            ax2.plot(sub.day, sub.latCV_mean, "o-", color=c, lw=2.2, label=cond)
    ax2.set_xlabel("day"); ax2.set_ylabel("lateral heterogeneity  (mean CV of xy projection)")
    ax2.set_title("Lateral patchiness over time — CH vs DH", fontsize=13)
    ax2.legend(); ax2.grid(ls=":", alpha=0.6)
    fig2.tight_layout(); fig2.savefig(OUT / "fish_3d_lateral_heterogeneity.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    print(f"wrote fish_3d_fnpg_coloc.png, fish_3d_lateral_heterogeneity.png")
    print(f"saved {len(ic_saved)} IC arrays to {ICDIR}")


if __name__ == "__main__":
    main()
