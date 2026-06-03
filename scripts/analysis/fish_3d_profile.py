#!/usr/bin/env python3
"""
fish_3d_profile.py — full 3-D (x, y, z) species structure from a HOBIC FISH .lif
FOV, WITHOUT the xy-averaging that lif_to_zprofiles.py applies.

Per-voxel decode (fish_decode) gives φ_i(z, y, x) for the 5 species; we then show
the lateral (x, y) structure that the depth-only pipeline averages away:
  (1) xy top-down projection per species   (the lateral profile)
  (2) xz side projection per species        (the depth profile, for comparison)
  (3) an orthoview (xy / xz / yz) for one species (default P. gingivalis)
  (4) 1-D mean profiles along x, y, z per species
and saves a downsampled φ(x, y, z, 5) array (usable as a 3-D PDE initial state).

Run from the repo root, e.g.:
  python scripts/analysis/fish_3d_profile.py                       # first DH Ti FOV
  python scripts/analysis/fish_3d_profile.py --file "HOBIC FISH/241018_*.lif" --series 0
  python scripts/analysis/fish_3d_profile.py --ortho P.gingivalis --downsample 4
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_root = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_root))
_sys.path.insert(0, str(_root / "scripts" / "pde"))   # lif_to_zprofiles helpers

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from readlif.reader import LifFile

from fish_decode import SPECIES_ORDER, SPECIES_RGB, channel_luts, decode, norm_scalars
from lif_to_zprofiles import (series_volume, condition_from_filename, series_day,
                              series_substrate, parse_day, voxel_um)

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "results" / "fish_3d"
SHORT = {"S.oralis": "S.o", "A.naeslundii": "A.n", "Vd/Vp": "Vd/Vp",
         "F.nucleatum": "F.n", "P.gingivalis": "P.g"}


def species_cmap(sp):
    from matplotlib.colors import LinearSegmentedColormap
    r, g, b = SPECIES_RGB[sp]
    return LinearSegmentedColormap.from_list(sp, [(0, 0, 0), (r, g, b)])


def pick_default_file():
    files = sorted(glob.glob(str(HERE / "HOBIC FISH" / "*.lif")))
    for p in files:                      # prefer a dysbiotic (DH) file
        if condition_from_filename(p) == "DH":
            return p
    return files[0] if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="path/glob to a .lif (default: first DH file)")
    ap.add_argument("--series", type=int, default=0, help="FOV index within the file")
    ap.add_argument("--ortho", default="P.gingivalis", help="species for the orthoview")
    ap.add_argument("--downsample", type=int, default=4, help="xy downsample factor for the saved .npy")
    args = ap.parse_args()

    path = args.file or pick_default_file()
    if path and ("*" in path or "?" in path):
        path = sorted(glob.glob(path))[0]
    if not path:
        raise SystemExit("no .lif file found")
    path = str(path)
    cond = condition_from_filename(path) or "?"
    lif = LifFile(path)
    imgs = list(lif.get_iter_image())
    luts = channel_luts(lif, imgs[0].channels)
    img = imgs[args.series]
    day = series_day(img.name, path, parse_day(Path(path).stem, None))
    dxr, dyr, dzr = voxel_um(img)        # µm per voxel (x, y, z)
    dxr, dyr, dzr = abs(dxr), abs(dyr), abs(dzr)   # sign-agnostic extents (depth +ve)

    vol = series_volume(img)             # (C, Z, Y, X)
    dec = decode(vol, luts, norm_scalars(vol, luts))   # {species: (Z, Y, X)}
    Z, Y, X = next(iter(dec.values())).shape
    tag = f"{cond}_d{day}_s{args.series}"
    print(f"file={Path(path).name}  FOV={args.series}  cond={cond} day={day}")
    print(f"  voxel grid (Z,Y,X)=({Z},{Y},{X})  voxel µm (x,y,z)=({dxr:.3f},{dyr:.3f},{dzr:.3f})")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- (1)+(2) xy top-down and xz side projections per species ----
    fig, axes = plt.subplots(2, len(SPECIES_ORDER), figsize=(2.7 * len(SPECIES_ORDER), 5.6))
    for k, sp in enumerate(SPECIES_ORDER):
        V = dec[sp]                       # (Z, Y, X)
        xy = V.mean(axis=0)               # (Y, X)  top-down (lateral profile)
        xz = V.mean(axis=1)               # (Z, X)  side (depth profile)
        cm = species_cmap(sp)
        axes[0, k].imshow(xy, cmap=cm, origin="lower",
                          extent=[0, X * dxr, 0, Y * dyr], aspect="equal")
        axes[0, k].set_title(SHORT[sp], color=SPECIES_RGB[sp], fontsize=12)
        axes[1, k].imshow(xz, cmap=cm, origin="upper",
                          extent=[0, X * dxr, Z * dzr, 0], aspect="auto")
        if k == 0:
            axes[0, k].set_ylabel("y (µm)\n[xy top-down]", fontsize=10)
            axes[1, k].set_ylabel("z depth (µm)\n[xz side]", fontsize=10)
        axes[1, k].set_xlabel("x (µm)", fontsize=9)
    fig.suptitle(f"FISH 3-D structure — {cond} day {day} (FOV {args.series})   "
                 f"top: lateral xy · bottom: depth xz", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"fish3d_proj_{tag}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- (3) orthoview for one species ----
    sp = args.ortho if args.ortho in dec else "P.gingivalis"
    V = dec[sp]
    cm = species_cmap(sp)
    fo, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ax[0].imshow(V.mean(0), cmap=cm, origin="lower", extent=[0, X*dxr, 0, Y*dyr], aspect="equal")
    ax[0].set_title("xy (top-down)"); ax[0].set_xlabel("x (µm)"); ax[0].set_ylabel("y (µm)")
    ax[1].imshow(V.mean(1), cmap=cm, origin="upper", extent=[0, X*dxr, Z*dzr, 0], aspect="auto")
    ax[1].set_title("xz (side)"); ax[1].set_xlabel("x (µm)"); ax[1].set_ylabel("z depth (µm)")
    ax[2].imshow(V.mean(2), cmap=cm, origin="upper", extent=[0, Y*dyr, Z*dzr, 0], aspect="auto")
    ax[2].set_title("yz (side)"); ax[2].set_xlabel("y (µm)"); ax[2].set_ylabel("z depth (µm)")
    fo.suptitle(f"{sp} orthoview — {cond} day {day}", color=SPECIES_RGB[sp], fontsize=13)
    fo.tight_layout(rect=(0, 0, 1, 0.95))
    fo.savefig(OUT / f"fish3d_ortho_{SHORT[sp].replace('/','')}_{tag}.png", dpi=170, bbox_inches="tight")
    plt.close(fo)

    # ---- (4) 1-D mean profiles along x, y, z ----
    fp, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    xs, ys, zs = np.arange(X)*dxr, np.arange(Y)*dyr, np.arange(Z)*dzr
    for sp in SPECIES_ORDER:
        V = dec[sp]; c = SPECIES_RGB[sp]
        ax[0].plot(xs, V.mean(axis=(0, 1)), color=c, lw=1.8, label=SHORT[sp])
        ax[1].plot(ys, V.mean(axis=(0, 2)), color=c, lw=1.8)
        ax[2].plot(zs, V.mean(axis=(1, 2)), color=c, lw=1.8)
    ax[0].set_xlabel("x (µm)"); ax[0].set_title("lateral profile along x"); ax[0].set_ylabel("mean φ")
    ax[1].set_xlabel("y (µm)"); ax[1].set_title("lateral profile along y")
    ax[2].set_xlabel("z depth (µm)"); ax[2].set_title("depth profile along z")
    ax[0].legend(fontsize=8, ncol=2)
    fp.suptitle(f"FISH per-axis profiles — {cond} day {day} (lateral x,y were averaged away before)", fontsize=12)
    fp.tight_layout(rect=(0, 0, 1, 0.95))
    fp.savefig(OUT / f"fish3d_axisprofiles_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close(fp)

    # ---- save a downsampled φ(x,y,z,5) array (usable as a 3-D PDE initial state) ----
    ds = max(1, args.downsample)
    arr = np.stack([dec[sp][:, ::ds, ::ds] for sp in SPECIES_ORDER], axis=-1)  # (Z, Yd, Xd, 5)
    arr = np.transpose(arr, (2, 1, 0, 3))    # (Xd, Yd, Z, 5) to match the 3-D PDE (x,y,z)
    np.save(OUT / f"phi_xyz_{tag}.npy", arr.astype(np.float32))
    print(f"  wrote fish3d_proj/ortho/axisprofiles PNGs and phi_xyz_{tag}.npy {arr.shape} to {OUT}")


if __name__ == "__main__":
    main()
