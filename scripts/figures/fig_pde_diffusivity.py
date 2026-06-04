# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_pde_diffusivity.py — Thesis figure: 3D-PINN inferred diffusivities CH vs DH.

Two panels:
  (a) Grouped bar chart: D_i (µm²/day) per species for CH and DH
  (b) Ratio D_i(DH) / D_i(CH) — which species change most between conditions

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_pde_diffusivity.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from thesis_style import use, clean_ax, SP_COLORS

ROOT  = pathlib.Path(__file__).parents[2]
OUT   = ROOT / "results" / "fig_pde_diffusivity"

SHORT  = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
COLORS = SP_COLORS
Z_MAX  = {'CH': 64.0, 'DH': 79.0}
T_MAX  = 21.0

PINN3D = {
    'CH': ROOT / 'results' / 'pinn_3d' / 'pinn3d_D_fit_CH.json',
    'DH': ROOT / 'results' / 'pinn_3d' / 'pinn3d_D_fit_DH.json',
}


def load_D_phys(cond):
    d = json.loads(PINN3D[cond].read_text())
    z = Z_MAX[cond]; t = T_MAX
    D = np.array([v * z**2 / t for v in d['D_fit']])
    u = d['u_fit'] * z / t
    return D, u, d['final_loss_total']


def main():
    D_CH, u_CH, loss_CH = load_D_phys('CH')
    D_DH, u_DH, loss_DH = load_D_phys('DH')

    fig, axes = plt.subplots(1, 2, figsize=use(width_frac=0.85, aspect=0.5))

    # --- (a) grouped bar chart ---
    ax = axes[0]
    x    = np.arange(5)
    w    = 0.35
    bars_ch = ax.bar(x - w/2, D_CH, width=w, color=[COLORS[i] for i in range(5)],
                     alpha=0.55, label='CH', edgecolor='k', linewidth=0.4)
    bars_dh = ax.bar(x + w/2, D_DH, width=w, color=[COLORS[i] for i in range(5)],
                     alpha=1.0,  label='DH', edgecolor='k', linewidth=0.4,
                     hatch='//')

    # Highlight Pg in DH (lowest)
    pg_idx = 4
    ax.annotate(r'$D_{Pg}$ min (DH)',
                xy=(x[pg_idx] + w/2, D_DH[pg_idx]),
                xytext=(x[pg_idx] + w/2 + 0.5, D_DH[pg_idx] + 0.8),
                fontsize=6.5, color='0.2',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3'))

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=8)
    ax.set_ylabel(r'$D_i$ (µm² day$^{-1}$)', fontsize=8)
    ax.set_title(r'\textbf{(a)} Species diffusivities', fontsize=8, pad=3)
    clean_ax(ax)

    # Legend
    patch_ch = mpatches.Patch(facecolor='0.6', alpha=0.55,
                               edgecolor='k', linewidth=0.4, label='CH')
    patch_dh = mpatches.Patch(facecolor='0.6', alpha=1.0, hatch='//',
                               edgecolor='k', linewidth=0.4, label='DH')
    ax.legend(handles=[patch_ch, patch_dh], fontsize=7, framealpha=0.9)

    # --- (b) ratio DH/CH ---
    ax2 = axes[1]
    ratio = D_DH / D_CH
    bar_colors = [COLORS[i] for i in range(5)]
    ax2.barh(x, ratio, color=bar_colors, edgecolor='k', linewidth=0.4, alpha=0.85)
    ax2.axvline(1.0, color='0.4', ls='--', lw=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(SHORT, fontsize=8)
    ax2.set_xlabel(r'$D_i^{\mathrm{DH}} / D_i^{\mathrm{CH}}$', fontsize=8)
    ax2.set_title(r'\textbf{(b)} DH/CH diffusivity ratio', fontsize=8, pad=3)
    ax2.invert_yaxis()
    clean_ax(ax2)
    fn_idx = 3
    # Annotate Pg as unique: only species with D_DH < D_CH
    ax2.annotate(r'$D_{Pg}^{\mathrm{DH}} < D_{Pg}^{\mathrm{CH}}$' + '\n(Pg stays deep)',
                 xy=(ratio[pg_idx], x[pg_idx]),
                 xytext=(ratio[pg_idx] + 0.35, x[pg_idx] + 0.6),
                 fontsize=6.5, color='0.2',
                 arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3'))
    # Annotate Fn as largest ratio
    ax2.annotate('Fn: largest\nDH/CH increase',
                 xy=(ratio[fn_idx], x[fn_idx]),
                 xytext=(ratio[fn_idx] - 1.0, x[fn_idx] - 0.8),
                 fontsize=6.5, color='0.2',
                 arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3'))

    # Caption info
    fig.text(0.5, -0.03,
             fr'3D-PINN inverse fit (3000 epochs). '
             fr'CH loss={loss_CH:.4f}, DH loss={loss_DH:.4f}. '
             r'$u_{\mathrm{CH}}$=' + f'{u_CH:+.2f},' +
             r' $u_{\mathrm{DH}}$=' + f'{u_DH:+.2f} µm day$^{{-1}}$.',
             ha='center', fontsize=6, color='0.4')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', bbox_inches='tight')
        print(f'saved → {OUT}.{ext}')


if __name__ == '__main__':
    main()
