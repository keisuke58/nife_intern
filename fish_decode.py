#!/usr/bin/env python3
"""
fish_decode.py — canonical 4-channel → 5-species decode for the Heine 2025
HOBIC FISH .lif stacks. Single source of truth shared by lif_quicklook.py and
lif_to_zprofiles.py.

Why this exists
---------------
The .lif has only **4 detector channels** but the biofilm has **5 species**.
Per Heine et al. 2025 (Front. Oral Health 1649419), Table S5 + Methods §2.6,
F. nucleatum is targeted by TWO probes with the same 16S sequence but different
dyes (FUS664-blau = Alexa 405, FUS664-rot = Alexa 647) → it lights up in BOTH
the blue and the red channel ("co-localized blue and red fluorescence").

    LUT / laser / emission        contributing species
    Blue   405 nm  413–477 nm  →  S. oralis  +  F. nucleatum
    Green  488 nm  509–576 nm  →  A. naeslundii          (clean)
    Yellow 552 nm  576–648 nm  →  V. dispar / V. parvula (clean)
    Red    638 nm  648–777 nm  →  P. gingivalis  +  F. nucleatum

Decoding (per voxel, NOT on xy-averaged profiles — min-of-means ≠ mean-of-mins):
    F. nucleatum  = min(blue, red)          # the part present in both channels
    S. oralis     = max(blue - F.nucleatum, 0)
    P. gingivalis = max(red  - F.nucleatum, 0)
    A. naeslundii = green
    V. dispar/parv= yellow

CAVEAT: the two F.nucleatum dyes/PMT gains are not identical, so min() after
per-channel normalisation is a first-order unmixing heuristic, not a calibrated
quantity. It correctly stops F.nucleatum being double-counted into So/Pg, which
is what the old colour=species map got wrong. Refine with a proper threshold /
spectral unmixing if the fit needs it.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

# canonical species order (matches GUILD-style S,A,V,F,P and SPECIES_LABELS)
SPECIES_ORDER = ['S.oralis', 'A.naeslundii', 'Vd/Vp', 'F.nucleatum', 'P.gingivalis']

# analysis-graph colours (same scheme as the Heine figure / pde scripts)
SPECIES_HEX = {
    'S.oralis':     '#1f77b4',
    'A.naeslundii': '#2ca02c',
    'Vd/Vp':        '#bcbd22',
    'F.nucleatum':  '#9467bd',
    'P.gingivalis': '#d62728',
}


def _hex2rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


SPECIES_RGB = {sp: _hex2rgb(h) for sp, h in SPECIES_HEX.items()}

# raw FISH channels (by LUT colour) that the decode needs
RAW_COLORS = ('blue', 'green', 'yellow', 'red')


def channel_luts(lif, n_ch: int) -> list:
    """LUT colour name of each of the first image's `n_ch` channels, in order."""
    root = ET.fromstring(lif.xml_header)
    luts = [el.attrib['LUTName'] for el in root.iter()
            if el.tag.split('}')[-1] == 'ChannelDescription' and 'LUTName' in el.attrib]
    return luts[:n_ch]


def color_index(luts: list) -> dict:
    """{'blue':ch_idx, 'green':..., 'yellow':..., 'red':...} from LUT names."""
    return {l.strip().lower(): c for c, l in enumerate(luts)}


def norm_scalars(stack: np.ndarray, luts: list, mode: str = "p995") -> dict:
    """Per-raw-colour normalisation scalar so channels are comparable for min().

    stack: (C, ...) raw intensities. mode 'p995' = 99.5th percentile per channel
    (robust, good for 8-bit FISH); 'max' = dtype/array max; 'bit8' = 255.
    """
    ci = color_index(luts)
    out = {}
    for col in RAW_COLORS:
        if col not in ci:
            continue
        a = stack[ci[col]]
        if mode == "bit8":
            s = 255.0
        elif mode == "max":
            s = float(a.max())
        else:
            s = float(np.percentile(a, 99.5))
        out[col] = s if s > 0 else 1.0
    return out


def decode(stack: np.ndarray, luts: list, scalars: dict | None = None) -> dict:
    """Decode a (C, ...) raw channel stack into {species: (...) float array}.

    `scalars` (from norm_scalars) divides each raw colour channel before the
    colocalisation; pass None to skip normalisation (raw units). Spatial shape
    is preserved, so this works on a 2D plane, a 2D MIP, or a 3D (Z,Y,X) volume.
    """
    ci = color_index(luts)

    def chan(col):
        if col not in ci:
            return None
        a = stack[ci[col]].astype(np.float32)
        if scalars is not None and col in scalars:
            a = a / scalars[col]
        return a

    blue, green, yellow, red = (chan(c) for c in RAW_COLORS)

    out = {}
    if blue is not None and red is not None:
        fn = np.minimum(blue, red)
        out['F.nucleatum'] = fn
        out['S.oralis'] = np.clip(blue - fn, 0, None)
        out['P.gingivalis'] = np.clip(red - fn, 0, None)
    else:                                   # degenerate file: fall back to raw
        out['F.nucleatum'] = None
        out['S.oralis'] = blue
        out['P.gingivalis'] = red
    out['A.naeslundii'] = green
    out['Vd/Vp'] = yellow
    return out
