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

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from readlif.reader import LifFile

from fish_decode import SPECIES_ORDER, SPECIES_RGB, channel_luts as fd_luts
from fish_decode import decode as fd_decode, norm_scalars as fd_norm


def _setup_style() -> None:
    """Times-family text, larger fonts (image annotations match the EN docs)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",          # ∩ etc. render even with a Times text font
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "figure.titlesize": 24,
    })


def px_per_um(img) -> float | None:
    """Pixels per µm in x (from the readlif scale); None if unavailable."""
    s = getattr(img, "scale", None)
    if s is not None and len(s) > 0 and s[0]:
        return float(s[0])
    return None


def _nice_bar_um(width_um: float) -> float:
    """A round scale-bar length (1/2/5 × 10ⁿ) close to ~1/5 of the field width."""
    target = max(width_um / 5.0, 1e-6)
    mag = 10 ** math.floor(math.log10(target))
    for m in (1, 2, 5, 10):
        if m * mag >= target:
            return m * mag
    return 10 * mag


def add_scalebar(ax, ppu: float | None, w_px: int, h_px: int) -> None:
    """Draw a white µm scale bar (black halo) bottom-right of an image axis."""
    if not ppu or ppu <= 0:
        return
    bar_um = _nice_bar_um(w_px / ppu)
    bar_px = bar_um * ppu
    x1 = w_px * 0.97
    x0 = x1 - bar_px
    y = h_px * 0.95
    ax.plot([x0, x1], [y, y], color="white", lw=6, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=9, foreground="black")])
    ax.text((x0 + x1) / 2, y - h_px * 0.02, f"{bar_um:g} µm",
            color="white", ha="center", va="bottom", fontsize=17, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")])

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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    for r, img in enumerate(series):
        stack = series_stack(img)            # (C,Z,Y,X)
        ppu = px_per_um(img)
        h_px, w_px = stack.shape[-2:]
        mips = [_norm(stack[c].max(0)) for c in range(stack.shape[0])]
        for c, mip in enumerate(mips):
            ax = axes[r][c]
            tint = np.array(LUT_RGB.get(luts[c], (1, 1, 1)), np.float32)
            ax.imshow(mip[..., None] * tint[None, None, :])
            if r == 0:
                ax.set_title(f"ch{c}: {luts[c]}", fontsize=19)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"[{r}] {img.name}\nz={img.dims.z}", fontsize=15)
        ax = axes[r][ncols - 1]
        ax.imshow(composite_rgb(mips, luts))
        add_scalebar(ax, ppu, w_px, h_px)
        if r == 0:
            ax.set_title("composite (MIP)", fontsize=19)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(stem, fontsize=23)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = outdir / f"{stem}__overview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def montage(lif: LifFile, stem: str, sidx: int, outdir: Path) -> Path:
    img = list(lif.get_iter_image())[sidx]
    luts = channel_luts(lif, img.channels or 1)
    stack = series_stack(img)                # (C,Z,Y,X)
    ppu = px_per_um(img)
    dz_um = (1.0 / img.scale[2]) if (getattr(img, "scale", None) and len(img.scale) > 2
                                     and img.scale[2]) else None
    h_px, w_px = stack.shape[-2:]
    nz = stack.shape[1]
    ncols = int(np.ceil(np.sqrt(nz)))
    nrows = int(np.ceil(nz / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows),
                             squeeze=False)
    for z in range(nrows * ncols):
        ax = axes[z // ncols][z % ncols]
        ax.set_xticks([]); ax.set_yticks([])
        if z < nz:
            planes = [_norm(stack[c, z]) for c in range(stack.shape[0])]
            ax.imshow(composite_rgb(planes, luts))
            zlab = f"z={z}" + (f" ({z * dz_um:.0f} µm)" if dz_um else "")
            ax.set_title(zlab, fontsize=15)
            if z == 0:
                add_scalebar(ax, ppu, w_px, h_px)
        else:
            ax.axis("off")
    fig.suptitle(f"{stem} — series [{sidx}] {img.name} (z-montage, composite)",
                 fontsize=19)
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
                             figsize=(3.2 * ncols, 3.2 * len(series)), squeeze=False)
    for r, img in enumerate(series):
        stack = series_stack(img)                          # (C,Z,Y,X) raw
        ppu = px_per_um(img)
        h_px, w_px = stack.shape[-2:]
        sp_vol = fd_decode(stack, luts, fd_norm(stack, luts))  # voxel-wise decode
        mips = {}
        for c, sp in enumerate(SPECIES_ORDER):
            vol = sp_vol.get(sp)
            mip = _norm(vol.max(0)) if vol is not None else np.zeros(stack.shape[-2:], np.float32)
            mips[sp] = mip
            ax = axes[r][c]
            ax.imshow(mip[..., None] * np.array(SPECIES_RGB[sp], np.float32)[None, None, :])
            if r == 0:
                ax.set_title(sp, fontsize=20)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"[{r}] {img.name}\nz={img.dims.z}", fontsize=15)
        comp = np.zeros(stack.shape[-2:] + (3,), np.float32)
        for sp in SPECIES_ORDER:
            comp += mips[sp][..., None] * np.array(SPECIES_RGB[sp], np.float32)[None, None, :]
        ax = axes[r][ncols - 1]
        ax.imshow(np.clip(comp, 0, 1))
        add_scalebar(ax, ppu, w_px, h_px)
        if r == 0:
            ax.set_title("5-species composite", fontsize=20)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(rf"{stem} — species decode (Fn = blue $\cap$ red)", fontsize=23)
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

    _setup_style()
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
