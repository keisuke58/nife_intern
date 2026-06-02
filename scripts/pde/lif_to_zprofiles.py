#!/usr/bin/env python3
"""
lif_to_zprofiles.py — Convert a Leica .lif confocal stack into the inputs the
spatial-PDE diffusion fit consumes (Heine 2025 CLSM/FISH 3D data).

Two modes
---------
1. --inspect : open the .lif and print its structure (series, dims x/y/z/t,
   channel count, voxel size in µm). RUN THIS FIRST when the file arrives —
   it tells us the channel→species mapping we need for export.

2. export (default) : for each selected series, reduce the (x,y) plane per
   z-slice to a mean intensity → depth profile per channel, map channels to the
   5 species, and write a WIDE CSV with columns:
       condition, day, z_um, S.oralis, A.naeslundii, Vd/Vp, F.nucleatum, P.gingivalis
   which fit_diffusion_clsm.py's load_zprofile_csv() reads directly.
   Optionally also dumps the full 3D voxel array (--full3d) for the 3D model /
   cross-diffusion identifiability.

Channel→species mapping
-----------------------
Each FISH channel ≈ one species. Pass it explicitly, e.g.:
    --channel-map "0:S.oralis,1:A.naeslundii,2:Vd/Vp,3:F.nucleatum,4:P.gingivalis"
(species labels must match SPECIES_LABELS so the downstream loader's substring
match works.)

Usage
-----
    python lif_to_zprofiles.py data.lif --inspect
    python lif_to_zprofiles.py data.lif --condition DH --day 6 \
        --out results/diffusion_fit/zprofiles_DH.csv --full3d
    # (channel map defaults to the confirmed Heine order; override with --channel-map)

NOTE: readlif declares numpy>=2 but works on the env's numpy 1.26 (import-tested).
If a real file misbehaves, isolate this preprocessing in its own venv.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from readlif.reader import LifFile
except ImportError:
    sys.exit("readlif not installed.  pip install --user readlif")

from fish_decode import (SPECIES_ORDER, channel_luts as fd_luts,
                         color_index as fd_color_index,
                         decode as fd_decode, norm_scalars as fd_norm)

# Must match fit_diffusion_clsm.SPECIES_LABELS (substring match downstream).
# Labels are STRAIN-AGNOSTIC: the strain is fixed by the condition tag, not the
# channel. Veillonella = V. dispar in CS/CH (commensal), V. parvula in DS/DH
# (dysbiotic); P. gingivalis = ATCC 20709 in CS/CH, W83 in DS/DH. So always map
# the Veillonella channel to 'Vd/Vp' and the Pg channel to 'P.gingivalis'; the
# --condition tag on each CSV row carries the strain identity downstream.
SPECIES_LABELS = ['S.oralis', 'A.naeslundii', 'Vd/Vp', 'F.nucleatum', 'P.gingivalis']

# The Heine .lif has 4 detector channels for 5 species: F.nucleatum is dual-
# labelled (Alexa405 + Alexa647) so it appears in BOTH blue and red. There is NO
# 1:1 channel→species map and NO purple channel. The decode lives in fish_decode:
#   Blue=S.oralis+F.nucleatum, Green=A.naeslundii, Yellow=Vd/Vp, Red=P.gingivalis+F.nucleatum
#   ⇒ Fn=blue∩red, So=blue−Fn, Pg=red−Fn  (see fish_decode.py / Heine Table S5).
# DEFAULT_CHANNEL_MAP is kept ONLY for the --channel-map RAW override path (debug
# / non-Heine files); the default export uses the colocalisation decode instead.
DEFAULT_CHANNEL_MAP = {
    0: 'S.oralis', 1: 'A.naeslundii', 2: 'Vd/Vp', 3: 'F.nucleatum', 4: 'P.gingivalis',
}


# ── helpers ──────────────────────────────────────────────────────────────────

def voxel_um(img):
    """Return (dx, dy, dz) in µm per pixel from readlif scale (px per µm); None if absent."""
    out = []
    scale = getattr(img, 'scale', None)
    for k in range(3):
        s = scale[k] if scale is not None and len(scale) > k else None
        out.append(1.0 / s if (s not in (None, 0)) else None)
    return tuple(out)


def parse_day(name, fallback):
    """Read a day number out of a series/file name like 'Tag1' (German), 'Day6',
    'd_15', 't10'.  'tag'/'day' are matched before bare 't'/'d' to avoid spurious
    hits on dates etc."""
    m = re.search(r'(?:tag|day|t|d)[ _]?(\d+)', str(name), flags=re.IGNORECASE)
    return int(m.group(1)) if m else fallback


# ── Heine e-mail conventions (Nils Heine 2026-05-26, via Szafranski) ──────────
# HOBIC22 = commensal model, HOBIC24 = dysbiotic model. These are flow-chamber
# (HOBIC) stainings, so commensal→CH, dysbiotic→DH. Day 0 = inoculation. The day
# is normally at the END of the file name (…Tag1, …Tag15); BUT for HOBIC24 after
# Day 1 a single file holds several days, and the sample day is the FIRST number
# of each *image (series)* name → group by series-name leading integer instead.

def condition_from_filename(path):
    """'HOBIC22'→'CH' (commensal), 'HOBIC24'→'DH' (dysbiotic); None if unknown."""
    n = Path(path).name.upper()
    if 'HOBIC22' in n:
        return 'CH'
    if 'HOBIC24' in n:
        return 'DH'
    return None


def is_hobic24(path):
    return 'HOBIC24' in Path(path).name.upper()


def leading_int(name):
    """First integer token in a string, e.g. '6 3' → 6, '92 1' → 92; None if none."""
    m = re.match(r'\s*(\d+)', str(name))
    return int(m.group(1)) if m else None


def series_day(img_name, path, file_day):
    """Day for one series, following the Heine naming convention for this file.
    HOBIC24: leading integer of the series name (multiple days per file).
    Otherwise: the file-level day (…TagN), with a series-name parse as backup."""
    if is_hobic24(path):
        d = leading_int(img_name)
        if d is not None:
            return d
    return parse_day(img_name, file_day)


# Species ↔ analysis-graph colour (mirrors COLORS in nsp_pde_1d_heine.py /
# guild scripts). The Heine paper figure uses the SAME colour scheme, so the
# .lif channel LUT colour → figure legend colour → species.
SPECIES_GRAPH_COLOR = {
    'S.oralis':     '#1f77b4 (blue)',
    'A.naeslundii': '#2ca02c (green)',
    'Vd/Vp':        'yellow in FISH figure',
    'F.nucleatum':  'purple in FISH figure',
    'P.gingivalis': '#d62728 (red)',
}


def dump_channel_meta(lif: LifFile):
    """Best-effort: pull channel LUT / fluorophore names from the LIF XML so the
    channels can be matched to the Heine paper figure by colour."""
    try:
        root = lif.xml_root
    except Exception as e:                       # pragma: no cover
        print(f'  (could not read XML metadata: {e})')
        return
    tags  = ('ChannelDescription', 'WideFieldChannelInfo', 'ChannelInfo', 'ChannelScalingInfo')
    attrs = ('LUTName', 'FluoCubeName', 'DyeName', 'ChannelName', 'Name',
             'ContrastingMethodName', 'ChannelTag')
    hits = []
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        if tag in tags:
            picked = {a: el.attrib[a] for a in attrs if a in el.attrib}
            if picked:
                hits.append((tag, picked))
    if hits:
        print('  channel metadata from XML (match LUT colour to the paper figure):')
        for tag, picked in hits:
            print(f'     {tag}: {picked}')
    else:
        print('  (no LUT/fluorophore tags found in XML — match channel order to the figure manually)')


def inspect(lif: LifFile):
    print(f'LifFile with {len(list(lif.get_iter_image()))} image series\n')
    for i, img in enumerate(lif.get_iter_image()):
        d = img.dims                      # namedtuple: x, y, z, t, m
        dx, dy, dz = voxel_um(img)
        print(f'[{i}] name={img.name!r}')
        print(f'     dims: x={d.x} y={d.y} z={d.z} t={getattr(d,"t",1)} '
              f'channels={img.channels}')
        print(f'     voxel µm: dx={dx} dy={dy} dz={dz}')
        print(f'     parsed day guess: {parse_day(img.name, None)}')
        print()
    dump_channel_meta(lif)
    print('\nspecies ↔ analysis-graph colour (same scheme as Heine paper figure):')
    for sp, col in SPECIES_GRAPH_COLOR.items():
        print(f'     {sp:14s} {col}')


# ── export: z-profile (xy-mean per slice) ─────────────────────────────────────

def series_zprofile(img, t=0):
    """Return (nz, n_channels) array of xy-mean intensity per z-slice."""
    nz = img.dims.z
    prof = np.zeros((nz, img.channels), dtype=np.float64)
    for c in range(img.channels):
        for z in range(nz):
            frame = np.asarray(img.get_frame(z=z, t=t, c=c), dtype=np.float64)
            prof[z, c] = frame.mean()
    return prof


def series_volume(img, t=0):
    """Return full 3D stack (n_channels, nz, ny, nx)."""
    nz = img.dims.z
    vol = None
    for c in range(img.channels):
        planes = [np.asarray(img.get_frame(z=z, t=t, c=c)) for z in range(nz)]
        stack = np.stack(planes, axis=0)               # (nz, ny, nx)
        if vol is None:
            vol = np.zeros((img.channels,) + stack.shape, dtype=stack.dtype)
        vol[c] = stack
    return vol


def species_zprofile(img, luts, t=0):
    """(nz, len(SPECIES_ORDER)) xy-mean intensity per z-slice, AFTER the voxel-wise
    5-species decode (F.nucleatum = blue∩red). Colocalisation must precede the
    xy-average — min-of-means ≠ mean-of-mins — so we decode the full 3D volume."""
    vol  = series_volume(img, t=t)                 # (n_ch, nz, ny, nx)
    sp   = fd_decode(vol, luts, fd_norm(vol, luts))  # {species: (nz,ny,nx)}
    nz   = vol.shape[1]
    prof = np.zeros((nz, len(SPECIES_ORDER)), dtype=np.float64)
    for k, name in enumerate(SPECIES_ORDER):
        a = sp.get(name)
        prof[:, k] = a.reshape(nz, -1).mean(axis=1) if a is not None else 0.0
    return prof


def parse_channel_map(s):
    """'0:S.oralis,1:A.naeslundii' -> {0:'S.oralis', 1:'A.naeslundii'}."""
    out = {}
    for part in s.split(','):
        idx, name = part.split(':')
        out[int(idx)] = name.strip()
    return out


# NOTE: the old colour→species map (blue=S.o, red=Pg, purple=F.n, …) was REMOVED.
# It silently mis-decoded the dual-labelled F.nucleatum, double-counting it into
# S.oralis (blue) and P.gingivalis (red) and dropping it entirely (no purple ch).
# Channel→species now goes through fish_decode (Fn=blue∩red). See fish_decode.py.


def averaged_profile(profiles, cmap, n_grid=40):
    """Interpolate each FOV's z-profile onto a common [0,1] grid and average.
    profiles: list of (nz, n_channels). Returns (grid, {species: (n_grid,)})."""
    grid = np.linspace(0, 1, n_grid)
    per_sp = {sp: [] for sp in cmap.values()}
    for prof in profiles:
        z_src = np.linspace(0, 1, prof.shape[0])
        for c, sp in cmap.items():
            per_sp[sp].append(np.interp(grid, z_src, prof[:, c]))
    avg = {sp: np.mean(np.stack(v), axis=0) for sp, v in per_sp.items()}
    return grid, avg


def _plot_depth(avg_for_plot, out_csv):
    """Quick depth-profile figure (per-z renormalised fractions), one panel per
    (condition, day). Keys may be a bare day or a (condition, day) tuple."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {'S.oralis': '#1f77b4', 'A.naeslundii': '#2ca02c', 'Vd/Vp': '#bcbd22',
           'F.nucleatum': '#9467bd', 'P.gingivalis': '#d62728'}
    keys = sorted(avg_for_plot)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.4), squeeze=False)
    for j, key in enumerate(keys):
        ax = axes[0][j]
        grid, avg, depth = avg_for_plot[key]
        sp_list = list(avg)
        M = np.stack([avg[sp] for sp in sp_list], axis=1)      # (n_grid, nsp)
        s = M.sum(axis=1, keepdims=True); s[s == 0] = 1.0
        frac = M / s
        z_um = grid * depth
        for si, sp in enumerate(sp_list):
            ax.plot(frac[:, si], z_um, color=COL.get(sp, 'k'), lw=2, label=sp)
        title = f'{key[0]} d{key[1]}' if isinstance(key, tuple) else f'Day {key}'
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('fraction'); ax.set_xlim(0, 1); ax.invert_yaxis()
        if j == 0:
            ax.set_ylabel('depth z (µm)')
    axes[0][-1].legend(fontsize=6, loc='best')
    fig.suptitle('FISH depth profiles (FOV-averaged, per-z renormalised)', fontsize=10)
    fig.tight_layout()
    p = Path(out_csv).with_suffix('.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f'Saved plot: {p}')


def out_path_default(args):
    """Output CSV path: --out wins; for a single file name by its stem; for several
    files of one condition use zprofiles_<COND>_merged.csv; else zprofiles_merged.csv."""
    if args.out:
        return args.out
    paths = args.lif
    if len(paths) == 1:
        return f'results/diffusion_fit/zprofiles_{Path(paths[0]).stem}.csv'
    conds = {args.condition or condition_from_filename(p) for p in paths}
    conds.discard(None)
    if len(conds) == 1:
        return f'results/diffusion_fit/zprofiles_{next(iter(conds))}_merged.csv'
    return 'results/diffusion_fit/zprofiles_merged.csv'


def main():
    ap = argparse.ArgumentParser(description='Leica .lif → z-profile CSV for diffusion fit')
    ap.add_argument('lif', type=str, nargs='+',
                    help='one or more .lif files; replicates are POOLED at the FOV '
                         'level per (condition, day) into one merged CSV')
    ap.add_argument('--inspect', action='store_true', help='print structure and exit')
    ap.add_argument('--series', type=str, default='all',
                    help='comma-separated series indices to export, or "all"')
    ap.add_argument('--channel-map', type=str, default=None,
                    help='e.g. "0:S.oralis,1:A.naeslundii,2:Vd/Vp,3:F.nucleatum,4:P.gingivalis"; '
                         'if omitted, uses the confirmed DEFAULT_CHANNEL_MAP (canonical S,A,V,F,P order)')
    ap.add_argument('--condition', type=str, default=None,
                    help='CS/CH/DS/DH tag; if omitted, auto from filename '
                         '(HOBIC22→CH commensal, HOBIC24→DH dysbiotic)')
    ap.add_argument('--day', type=int, default=None, help='day value (fallback if not in series/file name)')
    ap.add_argument('--out', type=str, default=None,
                    help='output CSV path; default results/diffusion_fit/zprofiles_<stem>.csv')
    ap.add_argument('--n-grid', type=int, default=40, help='depth grid points for the averaged profile')
    ap.add_argument('--full3d', action='store_true', help='also dump full 3D voxel npy per series')
    args = ap.parse_args()

    if args.inspect:
        for path in args.lif:
            print(f'### {path}')
            inspect(LifFile(path))
            print()
        return

    # decode/cmap depend only on --channel-map, not the file → decide once
    if args.channel_map:
        # RAW 1:1 override (debug / non-Heine files): one channel = one species,
        # NO colocalisation decode. F.nucleatum will be wrong if it is dual-labelled.
        cmap = parse_channel_map(args.channel_map)
        use_decode = False
        print(f'  RAW channel map (explicit, decode OFF): {cmap}')
    else:
        # canonical Heine 4ch→5sp decode: Fn=blue∩red, So=blue−Fn, Pg=red−Fn
        cmap = {k: sp for k, sp in enumerate(SPECIES_ORDER)}
        use_decode = True
        print('  5-species decode ON: An=green, Vd/Vp=yellow, Fn=blue∩red, So=blue−Fn, Pg=red−Fn')
    missing = [sp for sp in SPECIES_LABELS if sp not in cmap.values()]
    if missing:
        print(f'  [note] species NOT present: {missing}')

    from collections import defaultdict
    # pool FOV profiles across ALL input files, keyed by (condition, day) — this is
    # how Day-1 replicates from different experiment dates get merged.
    prof_pool, depth_pool = defaultdict(list), defaultdict(list)
    for path in args.lif:
        lif = LifFile(path)
        condition = args.condition or condition_from_filename(path)
        if not condition:
            sys.exit(f'{Path(path).name}: cannot derive condition (need HOBIC22/HOBIC24); '
                     'pass --condition explicitly.')
        images = list(lif.get_iter_image())
        luts = fd_luts(lif, images[0].channels)
        if use_decode:
            miss = [c for c in ('blue', 'green', 'yellow', 'red') if c not in fd_color_index(luts)]
            if miss:
                print(f'  [warn] {Path(path).name}: missing LUT colours {miss}; decode degraded')
        sel = (list(range(len(images))) if args.series == 'all'
               else [int(x) for x in args.series.split(',')])
        file_day = parse_day(Path(path).stem, args.day)
        print(f'  {Path(path).name}: condition={condition} (auto={args.condition is None}), '
              f'LUTs={luts}, {len(sel)} FOV(s)')
        for i in sel:
            img = images[i]
            day = series_day(img.name, path, file_day)
            if day is None:
                sys.exit(f'{Path(path).name} series [{i}] {img.name!r}: no day; pass --day')
            _, _, dz = voxel_um(img)
            dz = abs(dz) if dz else 1.0
            prof = species_zprofile(img, luts) if use_decode else series_zprofile(img)
            prof_pool[(condition, day)].append(prof)
            depth_pool[(condition, day)].append(prof.shape[0] * dz)
            if args.full3d:
                vol = series_volume(img)
                vp = Path(out_path_default(args)).with_name(
                    f'vol3d_{condition}_day{day}_{Path(path).stem}_s{i}.npy')
                np.save(vp, vol)
                print(f'    saved 3D volume: {vp}  shape={vol.shape} (c,z,y,x)')

    out_path = out_path_default(args)

    rows, avg_for_plot = [], {}
    for (condition, day), profs in sorted(prof_pool.items()):
        grid, avg = averaged_profile(profs, cmap, args.n_grid)
        depth_um  = float(np.mean(depth_pool[(condition, day)]))
        avg_for_plot[(condition, day)] = (grid, avg, depth_um)
        for k in range(args.n_grid):
            row = {'condition': condition, 'day': day, 'z_um': float(grid[k] * depth_um)}
            for sp, y in avg.items():
                row[sp] = float(y[k])
            rows.append(row)
        print(f'  → {condition} day {day}: pooled {len(profs)} FOV(s) across files, '
              f'depth≈{depth_um:.1f} µm')

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df)} rows → {out_path}')
    print(f'Columns: {list(df.columns)}')
    try:
        _plot_depth(avg_for_plot, out_path)
    except Exception as e:
        print(f'  [warn] plot failed: {e}')


if __name__ == '__main__':
    main()
