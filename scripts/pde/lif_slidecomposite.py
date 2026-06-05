#!/usr/bin/env python3
"""
lif_slidecomposite.py — slide-ready FISH composite panels from .lif stacks.

Outputs per LIF file:
  --mode gallery  :  best N FOVs as composites in a row (16:9 friendly)
  --mode feature  :  one FOV: xy MIP + xz depth cross-section side by side
  --mode both     :  both (default)

Usage
-----
    python scripts/pde/lif_slidecomposite.py "HOBIC FISH/220518_HOBIC22_5Spezies_FISH_Tag1.lif"
    python scripts/pde/lif_slidecomposite.py "HOBIC FISH"/*.lif --n-panel 3 --out figures/lif_slides
    python scripts/pde/lif_slidecomposite.py FILE.lif --mode feature --series 2
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))      # scripts/pde/ for lif_quicklook

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from readlif.reader import LifFile

from fish_decode import SPECIES_ORDER, SPECIES_RGB, channel_luts as fd_luts
from fish_decode import decode as fd_decode, norm_scalars as fd_norm
from lif_quicklook import series_stack, _norm, add_scalebar, _setup_style, px_per_um


SPECIES_LABEL = {
    'S.oralis':     'S. oralis',
    'A.naeslundii': 'A. naeslundii',
    'Vd/Vp':        'Vd/Vp',
    'F.nucleatum':  'F. nucleatum',
    'P.gingivalis': 'P. gingivalis',
}


def _sp_rgb(sp) -> np.ndarray:
    return np.array(SPECIES_RGB[sp], np.float32)


def _composite_mip(sp_vol: dict, axis: int = 0) -> np.ndarray:
    """5-species composite MIP along `axis`. Returns HxWx3 float [0,1]."""
    shape = next(v for v in sp_vol.values() if v is not None).shape
    out_shape = list(shape)
    out_shape.pop(axis)
    comp = np.zeros(out_shape + [3], np.float32)
    for sp in SPECIES_ORDER:
        vol = sp_vol.get(sp)
        if vol is None:
            continue
        mip = _norm(vol.max(axis))
        comp += mip[..., None] * _sp_rgb(sp)[None, None, :]
    return np.clip(comp, 0, 1)


def _best_series(all_series, luts, n: int):
    """Pick n series with highest mean composite signal (most biofilm = most beautiful)."""
    scored = []
    for img in all_series:
        stack = series_stack(img)
        sp_vol = fd_decode(stack, luts, fd_norm(stack, luts))
        mean_sig = np.mean([
            _norm(v.max(0)).mean()
            for v in sp_vol.values() if v is not None
        ])
        scored.append((mean_sig, img))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [img for _, img in scored[:n]]


def _species_legend(ax, loc="lower right", fontsize=11):
    handles = [
        plt.Line2D([0], [0], color=SPECIES_RGB[sp], lw=5,
                   label=SPECIES_LABEL[sp])
        for sp in SPECIES_ORDER
    ]
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.75,
              edgecolor="white", labelcolor="white",
              facecolor="#111111")


# ── Gallery ──────────────────────────────────────────────────────────────────

def render_gallery(lif: LifFile, stem: str, outdir: Path,
                   n_panel: int = 3, dpi: int = 150) -> Path:
    """Row of n_panel best-signal FOV composites, wide/slide-friendly."""
    all_series = list(lif.get_iter_image())
    luts = fd_luts(lif, all_series[0].channels or 1)
    top = _best_series(all_series, luts, n_panel)

    fig, axes = plt.subplots(1, n_panel,
                              figsize=(5.5 * n_panel, 5.5),
                              squeeze=False)
    for idx, img in enumerate(top):
        ax = axes[0][idx]
        stack = series_stack(img)
        sp_vol = fd_decode(stack, luts, fd_norm(stack, luts))
        comp = _composite_mip(sp_vol, axis=0)   # xy MIP
        ax.imshow(comp, interpolation="bilinear")
        ppu = px_per_um(img)
        add_scalebar(ax, ppu, stack.shape[-1], stack.shape[-2])
        ax.set_xticks([]); ax.set_yticks([])
        name = str(img.name)[:28] + ("…" if len(str(img.name)) > 28 else "")
        ax.set_title(name, fontsize=13, color="white",
                     bbox=dict(facecolor="black", alpha=0.45, pad=2))
        if idx == n_panel - 1:
            _species_legend(ax, loc="upper right", fontsize=9)
    fig.patch.set_facecolor("black")
    fig.suptitle(f"{stem} — FISH 5-species composite (max-intensity projection)",
                 fontsize=15, color="white")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / f"{stem}__slide_gallery.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"wrote  {out}")
    return out


# ── Feature (xy + xz) ────────────────────────────────────────────────────────

def render_feature(lif: LifFile, stem: str, outdir: Path,
                   sidx: int | None = None, dpi: int = 200) -> Path:
    """Feature panel: large xy composite + xz depth cross-section."""
    all_series = list(lif.get_iter_image())
    luts = fd_luts(lif, all_series[0].channels or 1)
    img = all_series[sidx] if sidx is not None else _best_series(all_series, luts, 1)[0]

    stack = series_stack(img)          # (C, Z, Y, X)
    sp_vol = fd_decode(stack, luts, fd_norm(stack, luts))
    ppu = px_per_um(img)
    nz, ny, nx = stack.shape[1:]

    xy_comp = _composite_mip(sp_vol, axis=0)   # (Y, X, 3) — top-down
    xz_comp = _composite_mip(sp_vol, axis=1)   # (Z, X, 3) — depth cross-section

    # physical aspect ratio for xz panel
    try:
        sc = getattr(img, "scale", None) or []
        dz_um = (1.0 / sc[2]) if len(sc) > 2 and sc[2] and sc[2] > 0 else None
        dx_um = (1.0 / ppu) if ppu and ppu > 0 else None
        xz_aspect = (dz_um / dx_um) if (dz_um and dx_um and dx_um > 0) else 1.0
        if not (np.isfinite(xz_aspect) and xz_aspect > 0):
            xz_aspect = 1.0
    except Exception:
        xz_aspect = 1.0

    fig = plt.figure(figsize=(14, 6), facecolor="black")
    # left: square-ish xy panel; right: xz panel (depth ~ narrower)
    ax1 = fig.add_axes([0.02, 0.08, 0.54, 0.84])
    ax2 = fig.add_axes([0.60, 0.08, 0.38, 0.84])

    ax1.imshow(xy_comp, interpolation="bilinear")
    add_scalebar(ax1, ppu, nx, ny)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Top-down view  (xy, MIP)", fontsize=15, color="white", pad=6)
    _species_legend(ax1, loc="lower right", fontsize=10)

    ax2.imshow(xz_comp, aspect="auto", interpolation="bilinear")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Depth cross-section  (xz, MIP)", fontsize=15, color="white", pad=6)
    # annotate physical depth if scale is available
    if ppu and getattr(img, "scale", None) and len(img.scale) > 2 and img.scale[2] and img.scale[2] > 0:
        dz_um = 1.0 / img.scale[2]
        total_z = nz * dz_um
        ax2.set_ylabel(f"depth  (0 – {total_z:.0f} µm)", fontsize=12, color="white")
    ax2.set_xlabel("x  (µm)" if ppu else "x (pixels)", fontsize=12, color="white")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(0.5)

    name = str(img.name)
    fig.suptitle(f"{stem}  ·  {name}", fontsize=14, color="white", y=0.99)

    out = outdir / f"{stem}__slide_feature.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"wrote  {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lif", nargs="+", help=".lif file(s)")
    ap.add_argument("--mode", choices=["gallery", "feature", "both"], default="both")
    ap.add_argument("--n-panel", type=int, default=3,
                    help="FOVs shown in gallery mode (default 3)")
    ap.add_argument("--series", type=int, default=None,
                    help="force a specific series index for feature mode")
    ap.add_argument("--out", default="figures/lif_slides",
                    help="output directory (default figures/lif_slides)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    _setup_style()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for path in args.lif:
        p = Path(path)
        lif = LifFile(str(p))
        stem = p.stem
        if args.mode in ("gallery", "both"):
            render_gallery(lif, stem, outdir, args.n_panel, args.dpi)
        if args.mode in ("feature", "both"):
            render_feature(lif, stem, outdir, args.series, args.dpi)


if __name__ == "__main__":
    main()
