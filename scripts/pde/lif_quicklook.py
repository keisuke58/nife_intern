#!/usr/bin/env python3
"""
lif_quicklook.py — Headless visualiser for Leica .lif confocal stacks.

The cluster has no X display, so Fiji/napari GUIs are unavailable. This renders
PNG quick-looks straight from readlif that you can open in VSCode:

  * --mode overview (default): one PNG per .lif. For every series, a row of
    per-channel max-intensity projections (each in its Leica LUT colour) plus a
    composite RGB. Good first look / channel→species sanity check.
  * --mode montage --series N: a z-slice montage (every slice as a small tile,
    composite RGB) for one series — to eyeball biofilm depth structure.

Channel LUT order is read from the .lif XML (e.g. Blue/Yellow/Green/Red).

Usage
-----
    python lif_quicklook.py "HOBIC FISH/220518_HOBIC22_5Spezies_FISH_Tag1.lif"
    python lif_quicklook.py FILE.lif --mode montage --series 1
    python lif_quicklook.py "HOBIC FISH"/*.lif --out figures/lif_quicklook
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from readlif.reader import LifFile

from fish_decode import SPECIES_ORDER, SPECIES_RGB, channel_luts as fd_luts
from fish_decode import decode as fd_decode, norm_scalars as fd_norm

# Leica LUT name -> (R,G,B) tint applied to that channel's grey intensities.
LUT_RGB = {
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.2, 0.4, 1.0),
    "Yellow": (1.0, 0.85, 0.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Gray": (1.0, 1.0, 1.0),
    "Grey": (1.0, 1.0, 1.0),
}


def channel_luts(lif: LifFile, n_ch: int) -> list[str]:
    """Per-channel LUT names from the .lif XML, falling back to defaults."""
    try:
        root = ET.fromstring(lif.xml_header)
        names = [cd.get("LUTName") for cd in root.iter("ChannelDescription")]
        names = [n for n in names if n]
        if len(names) >= n_ch:
            return names[:n_ch]
    except Exception:
        pass
    return (["Gray", "Green", "Red", "Blue"] * n_ch)[:n_ch]


def _norm(plane: np.ndarray, p_lo=1.0, p_hi=99.7) -> np.ndarray:
    """Percentile-stretch a 2D array to [0,1] for display."""
    plane = plane.astype(np.float32)
    lo, hi = np.percentile(plane, [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((plane - lo) / (hi - lo), 0, 1)


def series_stack(img) -> np.ndarray:
    """Return a (C, Z, Y, X) float array for one readlif series."""
    nz = img.dims.z or 1
    nc = img.channels or 1
    out = []
    for c in range(nc):
        zs = [np.asarray(img.get_frame(z=z, t=0, c=c)) for z in range(nz)]
        out.append(np.stack(zs, axis=0))
    return np.stack(out, axis=0)  # (C, Z, Y, X)


def composite_rgb(planes2d: list[np.ndarray], luts: list[str]) -> np.ndarray:
    """Additively blend per-channel (already 0-1) planes into one RGB image."""
    h, w = planes2d[0].shape
    rgb = np.zeros((h, w, 3), np.float32)
    for plane, lut in zip(planes2d, luts):
        tint = np.array(LUT_RGB.get(lut, (1, 1, 1)), np.float32)
        rgb += plane[..., None] * tint[None, None, :]
    return np.clip(rgb, 0, 1)


def overview(lif: LifFile, stem: str, outdir: Path) -> Path:
    series = list(lif.get_iter_image())
    luts = channel_luts(lif, series[0].channels or 1)
    nrows = len(series)
    ncols = (series[0].channels or 1) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows),
                             squeeze=False)
    for r, img in enumerate(series):
        stack = series_stack(img)            # (C,Z,Y,X)
        mips = [_norm(stack[c].max(0)) for c in range(stack.shape[0])]
        for c, mip in enumerate(mips):
            ax = axes[r][c]
            tint = np.array(LUT_RGB.get(luts[c], (1, 1, 1)), np.float32)
            ax.imshow(mip[..., None] * tint[None, None, :])
            if r == 0:
                ax.set_title(f"ch{c}: {luts[c]}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"[{r}] {img.name}\nz={img.dims.z}", fontsize=8)
        ax = axes[r][ncols - 1]
        ax.imshow(composite_rgb(mips, luts))
        if r == 0:
            ax.set_title("composite (MIP)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(stem, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = outdir / f"{stem}__overview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def montage(lif: LifFile, stem: str, sidx: int, outdir: Path) -> Path:
    img = list(lif.get_iter_image())[sidx]
    luts = channel_luts(lif, img.channels or 1)
    stack = series_stack(img)                # (C,Z,Y,X)
    nz = stack.shape[1]
    ncols = int(np.ceil(np.sqrt(nz)))
    nrows = int(np.ceil(nz / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.7 * nrows),
                             squeeze=False)
    for z in range(nrows * ncols):
        ax = axes[z // ncols][z % ncols]
        ax.set_xticks([]); ax.set_yticks([])
        if z < nz:
            planes = [_norm(stack[c, z]) for c in range(stack.shape[0])]
            ax.imshow(composite_rgb(planes, luts))
            ax.set_title(f"z={z}", fontsize=7)
        else:
            ax.axis("off")
    fig.suptitle(f"{stem} — series [{sidx}] {img.name} (z-montage, composite)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = outdir / f"{stem}__series{sidx}_zmontage.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def species_overview(lif: LifFile, stem: str, outdir: Path) -> Path:
    """Per series, decode the 4 channels into the 5 species (F.nucleatum = blue∩red)
    and show a MIP per species in its analysis colour + a 5-species composite."""
    series = list(lif.get_iter_image())
    luts = fd_luts(lif, series[0].channels or 1)
    ncols = len(SPECIES_ORDER) + 1
    fig, axes = plt.subplots(len(series), ncols,
                             figsize=(2.4 * ncols, 2.4 * len(series)), squeeze=False)
    for r, img in enumerate(series):
        stack = series_stack(img)                          # (C,Z,Y,X) raw
        sp_vol = fd_decode(stack, luts, fd_norm(stack, luts))  # voxel-wise decode
        mips = {}
        for c, sp in enumerate(SPECIES_ORDER):
            vol = sp_vol.get(sp)
            mip = _norm(vol.max(0)) if vol is not None else np.zeros(stack.shape[-2:], np.float32)
            mips[sp] = mip
            ax = axes[r][c]
            ax.imshow(mip[..., None] * np.array(SPECIES_RGB[sp], np.float32)[None, None, :])
            if r == 0:
                ax.set_title(sp, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"[{r}] {img.name}\nz={img.dims.z}", fontsize=8)
        comp = np.zeros(stack.shape[-2:] + (3,), np.float32)
        for sp in SPECIES_ORDER:
            comp += mips[sp][..., None] * np.array(SPECIES_RGB[sp], np.float32)[None, None, :]
        ax = axes[r][ncols - 1]
        ax.imshow(np.clip(comp, 0, 1))
        if r == 0:
            ax.set_title("5-species composite", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{stem} — species decode (Fn = blue∩red)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = outdir / f"{stem}__species.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lif", nargs="+", help="one or more .lif files")
    ap.add_argument("--mode", choices=["overview", "species", "montage"], default="overview")
    ap.add_argument("--series", type=int, default=0, help="series index for montage")
    ap.add_argument("--out", default="figures/lif_quicklook", help="output dir")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for path in args.lif:
        p = Path(path)
        lif = LifFile(str(p))
        stem = p.stem
        if args.mode == "overview":
            out = overview(lif, stem, outdir)
        elif args.mode == "species":
            out = species_overview(lif, stem, outdir)
        else:
            out = montage(lif, stem, args.series, outdir)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
