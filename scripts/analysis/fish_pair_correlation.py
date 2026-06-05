# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fish_pair_correlation.py — Pair cross-correlation g(r) for FISH voxel data.

Computes the pair cross-correlation function g(r) between two species (default:
Fn index-3 vs Pg index-4) at each day for CH and DH conditions, following
Valm et al. 2012 (PLOS ONE, PMC3360060):
  g(r) > 1  → co-aggregation at distance r
  g(r) = 1  → random (Poisson) distribution
  g(r) < 1  → mutual avoidance / exclusion

Implementation:
  For each voxel pair at distance r ± dr/2, count how often one is Fn-dominated
  and the other is Pg-dominated relative to the expectation under independence.
  Uses a fast random-subsample approach (N_pairs random pairs per bin) rather
  than exhaustive enumeration over 128³ voxel pairs.

Input:  results/fish_3d/ic/phi_xyz_{CH,DH}_d{day}.npy  (128,128,Nz,5)
Output: results/fish_pair_correlation.{pdf,png,json}

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/analysis/fish_pair_correlation.py
"""

import json
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from thesis_style import use, clean_ax, SP_COLORS

ROOT   = pathlib.Path(__file__).parents[2]
IC_DIR = ROOT / "results" / "fish_3d" / "ic"
OUT    = ROOT / "results" / "fish_pair_correlation"

# Pair to compute g(r) for
SP_A, SP_B = 3, 4         # Fn=3, Pg=4
SP_NAMES   = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
COLORS     = SP_COLORS

DAYS   = [1, 6, 10, 15, 21]
CONDS  = ['CH', 'DH']
Z_UM   = 2.0               # µm per voxel (z)
XY_UM  = 0.5               # µm per voxel (x,y) — confocal lateral
R_MAX  = 25.0              # µm — max distance
DR     = 2.0               # µm — bin width
N_SAMPLE = 80_000          # random voxel pairs to sample per bin estimate
THRESH = 0.15              # fraction threshold to call a voxel "species-dominated"
RNG    = np.random.default_rng(42)


def load_voxels(cond, day):
    """Return (x,y,z) coordinates in µm and composition for each voxel."""
    f = IC_DIR / f"phi_xyz_{cond}_d{day}.npy"
    if not f.exists():
        return None
    phi = np.load(f)                        # (Nx, Ny, Nz, 5)
    Nx, Ny, Nz, _ = phi.shape
    # Build coordinate arrays
    xi = np.arange(Nx) * XY_UM
    yi = np.arange(Ny) * XY_UM
    zi = np.arange(Nz) * Z_UM
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')  # each (Nx,Ny,Nz)
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)   # (N,3)
    comp   = phi.reshape(-1, 5)                                      # (N,5)
    # Normalise to relative composition per voxel
    total  = comp.sum(axis=1, keepdims=True)
    comp   = comp / np.where(total > 1e-6, total, 1.0)
    return coords, comp


def pair_cross_correlation(coords, comp, sp_a, sp_b,
                           r_max=R_MAX, dr=DR, n_sample=N_SAMPLE, thresh=THRESH):
    """Estimate g(r) between species sp_a and sp_b.

    Uses random-pair subsampling: draw N pairs uniformly at random from all
    voxels, bin by their Euclidean distance, compare observed co-occurrence
    rate to the marginal (independence) expectation.
    """
    N = coords.shape[0]
    r_edges = np.arange(0, r_max + dr, dr)
    r_mids  = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins  = len(r_mids)

    # Indicator masks: is voxel "dominated" by sp_a or sp_b?
    mask_a = comp[:, sp_a] > thresh
    mask_b = comp[:, sp_b] > thresh

    # Global fractions (for independence baseline)
    frac_a = mask_a.mean()
    frac_b = mask_b.mean()

    g = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    # Sample random pairs, compute distances, accumulate per bin
    n_pairs_total = n_sample * n_bins
    idx_i = RNG.integers(0, N, size=n_pairs_total)
    idx_j = RNG.integers(0, N, size=n_pairs_total)
    same  = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % N

    diffs = coords[idx_j] - coords[idx_i]
    dists = np.sqrt((diffs**2).sum(axis=1))

    # Joint indicator: i is sp_a and j is sp_b (or vice versa)
    joint = (mask_a[idx_i] & mask_b[idx_j]) | (mask_b[idx_i] & mask_a[idx_j])

    bin_ids = np.digitize(dists, r_edges) - 1
    valid   = (bin_ids >= 0) & (bin_ids < n_bins)

    for b in range(n_bins):
        sel = valid & (bin_ids == b)
        n_sel = sel.sum()
        if n_sel < 10:
            continue
        counts[b]  = n_sel
        obs_rate   = joint[sel].mean()
        # Under independence: P(i=A, j=B or i=B, j=A) = 2 * frac_a * frac_b
        expected   = 2.0 * frac_a * frac_b
        g[b] = obs_rate / expected if expected > 1e-9 else np.nan

    return r_mids, g, counts


def bootstrap_ci(coords, comp, sp_a, sp_b, n_boot=200, ci=0.95, **kwargs):
    """Bootstrap 95% CI for g(r) by resampling voxels with replacement."""
    N = coords.shape[0]
    r_mids, g_obs, _ = pair_cross_correlation(coords, comp, sp_a, sp_b, **kwargs)
    boot_gs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, N, size=N)
        _, g_b, _ = pair_cross_correlation(
            coords[idx], comp[idx], sp_a, sp_b, **kwargs)
        boot_gs.append(g_b)
    boot_gs = np.array(boot_gs)          # (n_boot, n_bins)
    lo = np.nanpercentile(boot_gs, (1 - ci) / 2 * 100, axis=0)
    hi = np.nanpercentile(boot_gs, (1 + ci) / 2 * 100, axis=0)
    return r_mids, g_obs, lo, hi


def main():
    results = {}
    N_BOOT = 100   # bootstrap resamples per (cond, day)

    fig, axes = plt.subplots(1, 2, figsize=use(width_frac=0.9, aspect=0.55),
                             sharey=True)

    day_cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(DAYS)))

    for col, cond in enumerate(CONDS):
        ax  = axes[col]
        res = []

        for i_day, day in enumerate(DAYS):
            data = load_voxels(cond, day)
            if data is None:
                continue
            coords, comp = data
            r_mids, g, lo, hi = bootstrap_ci(
                coords, comp, SP_A, SP_B, n_boot=N_BOOT)
            res.append({'day': day, 'r': r_mids.tolist(), 'g': g.tolist(),
                        'lo': lo.tolist(), 'hi': hi.tolist()})

            # Light smoothing over 2 bins for presentation
            def smooth(arr):
                out = arr.copy()
                fin = ~np.isnan(arr)
                if fin.sum() > 3:
                    tmp = np.where(fin, arr, np.nanmean(arr[fin]))
                    out = uniform_filter1d(tmp, size=2)
                    out[~fin] = np.nan
                return out

            g_s  = smooth(g)
            lo_s = smooth(lo)
            hi_s = smooth(hi)
            valid = ~np.isnan(g_s)
            dcol = day_cmap[i_day]
            ax.plot(r_mids[valid], g_s[valid],
                    color=dcol, lw=1.3, label=f'Day {day}')
            ax.fill_between(r_mids[valid], lo_s[valid], hi_s[valid],
                            color=dcol, alpha=0.18)

        ax.axhline(1.0, color='0.4', ls='--', lw=0.8, label='Random (g=1)')
        ax.set_xlim(0, R_MAX)
        ax.set_xlabel(r'Distance $r$ (µm)', fontsize=8)
        panel_label = r'\textbf{(a)} ' if col == 0 else r'\textbf{(b)} '
        cond_label  = 'Commensal HOBIC (CH)' if cond == 'CH' else 'Dysbiotic HOBIC (DH)'
        ax.set_title(panel_label + cond_label, fontsize=8, pad=3)
        clean_ax(ax)
        if col == 0:
            ax.set_ylabel(fr'$g(r)$ [Fn–Pg pair cross-correlation]', fontsize=8)

        results[cond] = res

    # Legend (day colors + reference line)
    day_handles = [mlines.Line2D([], [], color=day_cmap[i], lw=1.3,
                                 label=f'Day {d}')
                   for i, d in enumerate(DAYS)]
    ref_handle  = mlines.Line2D([], [], color='0.4', ls='--', lw=0.8,
                                label='Random (g=1)')
    axes[1].legend(handles=day_handles + [ref_handle],
                   fontsize=7, framealpha=0.9, loc='upper right')

    # Annotation
    fig.text(0.5, -0.04,
             fr'Pair cross-correlation between {SP_NAMES[SP_A]} and {SP_NAMES[SP_B]} '
             fr'(threshold $\phi_i > {THRESH}$, {N_SAMPLE:,} random pairs per bin). '
             r'$g(r)>1$: co-aggregation; $g(r)<1$: avoidance. '
             r'Valm et al.\ 2012.',
             ha='center', fontsize=6, color='0.4')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', bbox_inches='tight')
        print(f'saved → {OUT}.{ext}')

    with open(f'{OUT}.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'saved → {OUT}.json')


if __name__ == '__main__':
    main()
