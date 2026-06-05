# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_pde_ch_profile.py — PDE CH depth-profile figure for thesis.

Side-by-side:
  (a) FISH z-profile (mean over x,y from phi_xyz_CH_d1.npy)
  (b) Model z-profile: each species smoothed with Gaussian (sigma propto D_fit)
      as an illustrative proxy for diffusive smoothing.

Label: "illustrative (single-day fit)"

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_pde_ch_profile.py
"""

import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from thesis_style import use, clean_ax

ROOT = pathlib.Path(__file__).parents[2]
OUT  = ROOT / "results" / "fig_pde_ch_profile"

# paths
FIT_JSON = ROOT / "results" / "diffusion_fit" / "D_fit_CH.json"
NPY_FILE = ROOT / "results" / "fish_3d" / "ic" / "phi_xyz_CH_d1.npy"

# Species display names and colors (So, An, Vd/Vp, Fn, Pg)
SP_NAMES  = ['S.o.', 'A.n.', 'Vd/Vp', 'F.n.', 'P.g.']
SP_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']

# Scale factor: map D (physical, ~1e-5 to ~7e-3) to Gaussian sigma in pixels.
# D_min ~1e-5 → sigma~0.5px (no smoothing); D_max ~7e-3 → sigma~5px (visible)
D_REF_MIN = 1e-5
D_REF_MAX = 7e-3
SIGMA_MIN  = 0.3
SIGMA_MAX  = 5.0


def d_to_sigma(d):
    """Map diffusion coefficient to Gaussian smoothing sigma (px)."""
    d_clipped = max(D_REF_MIN, min(D_REF_MAX, d))
    t = (np.log10(d_clipped) - np.log10(D_REF_MIN)) / \
        (np.log10(D_REF_MAX) - np.log10(D_REF_MIN))
    return SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN)


def main():
    # --- load data ---
    fit = json.load(open(FIT_JSON))
    D_fit    = fit['D_fit']      # list of 5 floats
    sp_names = fit['D_species']  # ['S.o', 'A.n', 'Vd/Vp', 'F.n', 'P.g']

    phi = np.load(NPY_FILE)      # (128, 128, Nz, 5) float32
    # z-profile: mean over x, y
    zprof = phi.mean(axis=(0, 1))   # (Nz, 5)
    Nz = zprof.shape[0]
    z_um = np.arange(Nz) * 2.0     # 2 µm / voxel

    # model z-profile: Gaussian-smoothed FISH as diffusion proxy
    zprof_model = np.zeros_like(zprof)
    for sp in range(5):
        sigma = d_to_sigma(D_fit[sp])
        zprof_model[:, sp] = gaussian_filter1d(zprof[:, sp], sigma=sigma)

    # normalise each species independently to [0,1] for display clarity
    def norm_col(arr):
        out = arr.copy()
        for sp in range(5):
            mx = out[:, sp].max()
            if mx > 0:
                out[:, sp] /= mx
        return out

    zprof_n = norm_col(zprof)
    zprof_m = norm_col(zprof_model)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=use(width_frac=0.65, aspect=0.75),
                             sharey=True)

    titles = [r'\textbf{(a)} FISH z-profile (observed)',
              r'\textbf{(b)} Model z-profile (illus.)']
    data   = [zprof_n, zprof_m]

    for ax, d, title in zip(axes, data, titles):
        for sp in range(5):
            ax.plot(d[:, sp], z_um, color=SP_COLORS[sp],
                    lw=1.1, label=SP_NAMES[sp])
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlabel('Rel.\ abundance (norm.)', fontsize=7.5)
        ax.invert_yaxis()
        clean_ax(ax)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel(r'Depth ($\mu$m from surface)', fontsize=7.5)
    axes[1].legend(fontsize=6.5, loc='lower right', framealpha=0.9,
                   edgecolor='0.8', handlelength=1.2)

    # footnote
    fig.text(0.5, -0.03,
             r'\textit{Illustrative: model smoothing uses Gaussian proxy for '
             r'diffusion (single-day fit, CH attractor)}',
             ha='center', fontsize=6, color='0.4')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(str(OUT) + '.' + ext)
        print('saved ->', str(OUT) + '.' + ext)

    print('\nD_fit (CH):', ['{:.2e}'.format(d) for d in D_fit])
    print('u_fit:', fit['u_fit'])
    print('loss:', fit['loss'])


if __name__ == '__main__':
    main()
