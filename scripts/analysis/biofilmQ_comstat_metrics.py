#!/usr/bin/env python3
"""
biofilmQ_comstat_metrics.py
===========================
Compute COMSTAT / BiofilmQ-equivalent structural metrics from Heine HOBIC FISH .lif files.

Metrics per FOV per species (analogous to BiofilmQ cube analysis):
  - biovolume      [µm³]  — voxels above threshold × voxel volume
  - mean_thickness [µm]   — mean biofilm height per xy column
  - roughness_Ra   [-]    — normalised mean absolute thickness deviation
  - coverage       [0-1]  — fraction of xy area occupied by biofilm

Channel decoding uses fish_decode.py (canonical 4ch → 5sp, Fn = blue ∩ red).
Pixel scale: xy = 0.5544 µm/px, z = 0.5003 µm/step (from .lif metadata).

Output:
  results/biofilmQ_comstat/comstat_metrics.csv
  results/biofilmQ_comstat/fig_comstat_comparison.pdf/.png

Usage:
  python scripts/analysis/biofilmQ_comstat_metrics.py
  python scripts/analysis/biofilmQ_comstat_metrics.py --threshold 15
"""
import sys as _sys, pathlib as _pathlib  # noqa  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from readlif.reader import LifFile
from scipy.ndimage import label as nd_label


import fish_decode as fd
import thesis_style

thesis_style.use()
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "lines.linewidth": 1.2, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parents[2]
LIF_DIR = _HERE / "HOBIC FISH"
OUT_DIR = _HERE / "results" / "biofilmQ_comstat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# Pixel/voxel size (from readlif scale, confirmed in fish_pair_correlation.py)
XY_UM = 0.5544   # µm per pixel (x = y)
Z_UM  = 0.5003   # µm per z-step

VOXEL_VOL = XY_UM * XY_UM * Z_UM   # µm³

SPECIES = fd.SPECIES_ORDER   # ['S.oralis','A.naeslundii','Vd/Vp','F.nucleatum','P.gingivalis']

SP_COLORS = {
    "S.oralis":      "#2166AC",
    "A.naeslundii":  "#1B7837",
    "Vd/Vp":         "#DAA520",
    "F.nucleatum":   "#7B3294",
    "P.gingivalis":  "#8B0000",
}
SP_SHORT = {
    "S.oralis": "So", "A.naeslundii": "An", "Vd/Vp": "Vd/Vp",
    "F.nucleatum": "Fn", "P.gingivalis": "Pg",
}


# ── Helper: condition + day from filename ─────────────────────────────────────
def parse_lif_meta(lif_path: Path) -> tuple:
    """Return (condition, day) from .lif filename.
    HOBIC22 = CH (commensal), HOBIC24 = DH (dysbiotic).
    """
    name = lif_path.stem.upper()
    cond = "CH" if "HOBIC22" in name else ("DH" if "HOBIC24" in name else "??")
    m = re.search(r"TAG[_\s]?(\d+)", name)
    day = int(m.group(1)) if m else -1
    return cond, day


# ── Core metrics ──────────────────────────────────────────────────────────────
def comstat_metrics(vol: np.ndarray, threshold: float) -> dict:
    """Compute COMSTAT metrics from a 3-D (z, y, x) volume of one species.

    Parameters
    ----------
    vol       : float array (z, y, x), normalised 0–1
    threshold : intensity threshold (0–1) to define "biofilm present"

    Returns dict with keys: biovolume, mean_thickness, roughness_Ra, coverage.
    """
    binary = vol > threshold                  # (z, y, x) bool
    nz, ny, nx = binary.shape

    # Surface coverage: fraction of xy positions that have any biofilm
    col_has_bio = binary.any(axis=0)          # (ny, nx)
    coverage = col_has_bio.mean()

    # Thickness per xy column = number of z-slices above threshold × Z_UM
    thickness_map = binary.sum(axis=0) * Z_UM  # (ny, nx), µm

    # Mean thickness (only over columns that have biofilm)
    occupied = col_has_bio
    mean_thickness = thickness_map[occupied].mean() if occupied.any() else 0.0

    # Roughness Ra = mean(|T_i - T_mean|) / T_mean  (Heydorn et al. 2000)
    if mean_thickness > 0:
        roughness_Ra = float(np.abs(thickness_map[occupied] - mean_thickness).mean() / mean_thickness)
    else:
        roughness_Ra = np.nan

    # Biovolume = occupied voxels × voxel volume
    biovolume = float(binary.sum()) * VOXEL_VOL   # µm³

    return {
        "biovolume_um3":    biovolume,
        "mean_thickness_um": float(mean_thickness),
        "roughness_Ra":      roughness_Ra,
        "coverage":          float(coverage),
    }


# ── Per-FOV processing ────────────────────────────────────────────────────────
def process_lif(lif_path, threshold):
    """Return list of per-FOV per-species metric dicts."""
    cond, day = parse_lif_meta(lif_path)
    lif = LifFile(str(lif_path))
    results = []

    for series_idx in range(len(lif.image_list)):
        im = lif.get_image(series_idx)
        fov_name = im.name

        # Load 4-channel stack: shape (C, Z, Y, X) → pass to decode
        try:
            stack = np.array([
                np.stack([np.array(im.get_frame(z=z, t=0, c=c))
                          for z in range(im.dims.z)], axis=0)
                for c in range(im.channels)
            ], dtype=np.float32)   # (C, Z, Y, X)
        except Exception as e:
            warnings.warn(f"{lif_path.name} series {series_idx}: read error {e}")
            continue

        # Decode 4 channels → 5 species using fish_decode
        luts = fd.channel_luts(lif, im.channels)
        scalars = fd.norm_scalars(stack, luts)
        decoded = fd.decode(stack, luts, scalars)   # dict {species: (Z,Y,X)}

        for sp in SPECIES:
            vol = decoded.get(sp)
            if vol is None:
                continue

            # Determine threshold
            if threshold is not None:
                thr = threshold
            else:
                # Otsu on the max-projection for speed
                proj = vol.max(axis=0)
                try:
                    # Otsu threshold via histogram
                    hist, edges = np.histogram(proj[proj > 0].ravel(), bins=256)
                    bin_centers = 0.5 * (edges[:-1] + edges[1:])
                    total = hist.sum()
                    if total == 0:
                        raise ValueError
                    w0 = np.cumsum(hist) / total
                    w1 = 1 - w0
                    mu0 = np.cumsum(hist * bin_centers) / (np.cumsum(hist) + 1e-9)
                    mu1 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / (np.sum(hist) - np.cumsum(hist) + 1e-9)
                    inter_var = w0 * w1 * (mu0 - mu1) ** 2
                    thr = float(bin_centers[np.argmax(inter_var)])
                except Exception:
                    thr = 0.1

            metrics = comstat_metrics(vol, thr)
            metrics.update({
                "file":       lif_path.name,
                "series":     series_idx,
                "fov":        fov_name,
                "condition":  cond,
                "day":        day,
                "species":    sp,
                "threshold":  thr,
            })
            results.append(metrics)

    return results


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(df: pd.DataFrame, out_stem: Path):
    metrics_plot = [
        ("biovolume_um3",     r"Biovolume [µm$^3$]"),
        ("mean_thickness_um", "Mean thickness [µm]"),
        ("roughness_Ra",      r"Roughness $R_a$ [–]"),
        ("coverage",          "Surface coverage [0–1]"),
    ]
    conds = ["CH", "DH"]
    cond_colors = {"CH": "#4393C3", "DH": "#D6604D"}

    fig, axes = plt.subplots(len(SPECIES), len(metrics_plot),
                              figsize=(9, 8), sharex=False)

    for row, sp in enumerate(SPECIES):
        sub = df[df["species"] == sp]
        for col, (metric, ylabel) in enumerate(metrics_plot):
            ax = axes[row, col]
            for cond in conds:
                vals = sub[sub["condition"] == cond][metric].dropna().values
                if len(vals) == 0:
                    continue
                # box + strip
                bp = ax.boxplot([vals], positions=[conds.index(cond)],
                                widths=0.4, patch_artist=True, showfliers=False,
                                medianprops=dict(color="k", linewidth=1.0),
                                boxprops=dict(facecolor=cond_colors[cond], alpha=0.6,
                                              linewidth=0.6),
                                whiskerprops=dict(linewidth=0.6),
                                capprops=dict(linewidth=0.6))
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(conds.index(cond) + jitter, vals,
                           s=8, color=cond_colors[cond], alpha=0.6, linewidths=0)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(conds, fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col == 0:
                ax.set_ylabel(SP_SHORT[sp], fontsize=8, rotation=0,
                              labelpad=28, va="center")
            if row == 0:
                ax.set_title(ylabel, fontsize=8, pad=4)

    fig.suptitle("COMSTAT metrics: CH vs DH (per FOV, per species)", y=1.01)
    fig.tight_layout(h_pad=0.4, w_pad=0.5)

    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p)
        print(f"Saved: {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lif-dir", type=Path, default=LIF_DIR,
                    help="Directory containing .lif files (default: HOBIC FISH/)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Fixed intensity threshold 0–1 (default: per-channel Otsu)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "comstat_metrics.csv")
    args = ap.parse_args()

    lif_files = sorted(args.lif_dir.glob("*.lif"))
    if not lif_files:
        print(f"No .lif files found in {args.lif_dir}")
        return

    print(f"Found {len(lif_files)} .lif files")
    all_rows = []
    for lif_path in lif_files:
        cond, day = parse_lif_meta(lif_path)
        print(f"  {lif_path.name}  ({cond}, day {day}) ...", end=" ", flush=True)
        rows = process_lif(lif_path, args.threshold)
        all_rows.extend(rows)
        print(f"{len(rows)//len(SPECIES)} FOVs")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nMetrics saved: {args.out}")
    print(df.groupby(["condition", "species"])[
        ["biovolume_um3", "mean_thickness_um", "roughness_Ra", "coverage"]
    ].median().round(3).to_string())

    out_fig = OUT_DIR / "fig_comstat_comparison"
    make_figure(df, out_fig)


if __name__ == "__main__":
    main()
