#!/usr/bin/env python3
"""
plot_A_biological_interpretation.py
  Fig 6: Annotated A matrix heatmap (3 regimes)
  Fig 7: A_ij vs |F_ij| scatter + top-pairs bar chart
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

CR  = _here / 'results' / 'dieckow_cr'
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

d   = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
A   = np.array(d['A'])
nf  = np.array(d['net_flow'])
nfs = (nf + nf.T) / 2

# Unambiguous 5-char abbreviations
ABBR = ['Actin', 'Bacil', 'Bctrd', 'β-Pro', 'Clost', 'Corio',
        'Fusob', 'γ-Pro', 'Negat', 'Other']

# Regime classification (consistent with analyse_loo_stability.py)
REGIME_COLORS = {'data+prior aligned':    '#c0392b',
                 'prior-constrained muted': '#e67e22',
                 'data-driven no prior':    '#2980b9',
                 'variable':                '#95a5a6'}

def classify_regime(i, j, A, nfs):
    fval = abs(nfs[i, j])
    aval = abs(A[i, j])
    if fval > 0 and aval > 0.15:
        return 'data+prior aligned'
    elif fval > 0 and aval <= 0.15:
        return 'prior-constrained muted'
    else:
        return 'data-driven no prior'

# Known literature interactions
KNOWN = {
    (GUILD_ORDER.index('Actinobacteria'), GUILD_ORDER.index('Bacilli')):
        'co-agg.',
    (GUILD_ORDER.index('Bacilli'), GUILD_ORDER.index('Betaproteobacteria')):
        'co-agg.',
    (GUILD_ORDER.index('Actinobacteria'), GUILD_ORDER.index('Betaproteobacteria')):
        'co-agg.',
    (GUILD_ORDER.index('Actinobacteria'), GUILD_ORDER.index('Negativicutes')):
        'lactate',
    (GUILD_ORDER.index('Actinobacteria'), GUILD_ORDER.index('Bacteroidia')):
        'AA supply',
    (GUILD_ORDER.index('Bacteroidia'), GUILD_ORDER.index('Fusobacteriia')):
        'syntrophy',
    (GUILD_ORDER.index('Bacilli'), GUILD_ORDER.index('Negativicutes')):
        'lactate',
}


# ── Fig 6: Annotated heatmap ─────────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(7.5, 6.8))

vmax = max(abs(A[~np.eye(N_G, dtype=bool)]).max(), 0.5)
im = ax6.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
cbar = plt.colorbar(im, ax=ax6, shrink=0.82, label=r'$A_{ij}$ (MAP)')
cbar.ax.tick_params(labelsize=8)

ax6.set_xticks(range(N_G)); ax6.set_yticks(range(N_G))
ax6.set_xticklabels(ABBR, fontsize=8.5, rotation=45, ha='right')
ax6.set_yticklabels(ABBR, fontsize=8.5)
ax6.set_title(r'Interaction matrix $A$ (AGORA $W$=1.0 MAP)', fontsize=10, pad=8)

# Literature annotations on cells
for (i, j), label in KNOWN.items():
    for (ri, rj) in [(i, j), (j, i)]:
        fc = 'white' if abs(A[ri, rj]) > vmax * 0.5 else '#333'
        ax6.text(rj, ri, label, ha='center', va='center',
                 fontsize=5.5, color=fc, fontweight='bold',
                 fontfamily='sans-serif')

# Mark unconstrained (data-driven) pairs with dotted border
for i in range(N_G):
    for j in range(N_G):
        if i != j and nfs[i, j] == 0:
            ax6.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                          fill=False, edgecolor='#888', lw=0.7, ls=':'))

# Regime legend
legend_patches = [
    mpatches.Patch(facecolor=REGIME_COLORS['data+prior aligned'],   alpha=0.8, label='Data+prior aligned'),
    mpatches.Patch(facecolor=REGIME_COLORS['prior-constrained muted'], alpha=0.8, label='Prior-constrained muted'),
    mpatches.Patch(facecolor=REGIME_COLORS['data-driven no prior'],  alpha=0.8, label='Data-driven (dotted border)'),
]
ax6.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, -0.12),
           ncol=1, fontsize=7.5, framealpha=0.9)

fig6.tight_layout()
for ext in ('png', 'pdf'):
    out = FIG / f'fig_A_heatmap_annotated.{ext}'
    fig6.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


# ── Fig 7: A_ij vs |F_ij| scatter + top-pairs bar chart ─────────────────────
fig7, axes7 = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel A: scatter
ax = axes7[0]
pairs_list = []
for i in range(N_G):
    for j in range(i+1, N_G):
        regime = classify_regime(i, j, A, nfs)
        pairs_list.append((i, j, A[i,j], abs(nfs[i,j]), regime))

for regime_name, color in REGIME_COLORS.items():
    sub = [(i,j,av,fv) for (i,j,av,fv,rg) in pairs_list if rg == regime_name]
    if not sub: continue
    fvals = [s[3] for s in sub]
    avals = [s[2] for s in sub]
    ax.scatter(fvals, avals, color=color, s=55, alpha=0.8,
               label=regime_name.replace(' muted','').replace(' no prior',''),
               edgecolors='none', zorder=3)

# Annotate top pairs
for (i, j, aval, fval, regime) in pairs_list:
    if abs(aval) > 0.30:
        ax.annotate(f'{ABBR[i]}↔{ABBR[j]}', (fval, aval),
                    fontsize=6.5, xytext=(4, 1), textcoords='offset points',
                    va='center')

ax.axhline(0, color='k', lw=0.6, ls='--')
ax.set_xlabel(r'Prior strength $|F_{ij}|$')
ax.set_ylabel(r'Fitted $A_{ij}$ (MAP)')
ax.set_title(r'(A) Prior strength vs fitted interaction', fontsize=10)
leg_handles = [
    mpatches.Patch(color=REGIME_COLORS['data+prior aligned'],     label='Data+prior aligned'),
    mpatches.Patch(color=REGIME_COLORS['prior-constrained muted'], label='Prior-constrained muted'),
    mpatches.Patch(color=REGIME_COLORS['data-driven no prior'],    label='Data-driven (no prior)'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='k',
               markersize=6, label='Constrained pair'),
    plt.Line2D([0],[0], marker='x', color='k', markersize=6, label='Unconstrained pair'),
]
ax.legend(handles=leg_handles[:3], fontsize=7.5, loc='upper right')

# Panel B: horizontal bar chart — top 16 pairs by |A_ij|
ax2 = axes7[1]
pairs_sorted = sorted(pairs_list, key=lambda x: abs(x[2]), reverse=True)[:16]
pairs_sorted_asc = sorted(pairs_sorted, key=lambda x: x[2])

vals   = [p[2] for p in pairs_sorted_asc]
labs   = [f'{ABBR[p[0]]}↔{ABBR[p[1]]}' for p in pairs_sorted_asc]
fmags  = [p[3] for p in pairs_sorted_asc]
regimes= [p[4] for p in pairs_sorted_asc]
colors = [REGIME_COLORS[rg] for rg in regimes]
hatch  = ['' if f > 0 else '///' for f in fmags]

bars = ax2.barh(range(len(vals)), vals, color=colors, alpha=0.82, height=0.7)
for bar, h in zip(bars, hatch):
    bar.set_hatch(h)

ax2.set_yticks(range(len(labs)))
ax2.set_yticklabels(labs, fontsize=8)
ax2.axvline(0, color='k', lw=0.6)
ax2.set_xlabel(r'$A_{ij}$ (MAP)')
ax2.set_title(r'(B) Top 16 interactions by $|A_{ij}|$' + '\n(/// = no metabolic prior)', fontsize=10)

# |F| annotation on bars
for k, (v, fmag) in enumerate(zip(vals, fmags)):
    if fmag > 0:
        xpos = v + 0.02 * np.sign(v)
        ha = 'left' if v > 0 else 'right'
        ax2.text(xpos, k, f'|F|={fmag:.1f}', va='center', ha=ha,
                 fontsize=6, color='#555')

fig7.suptitle(r'AGORA $W$=1.0 MAP: $A$ matrix biological interpretation',
              fontsize=11, y=1.01)
fig7.tight_layout()
for ext in ('png', 'pdf'):
    out = FIG / f'fig_A_biological_interpretation.{ext}'
    fig7.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
