#!/usr/bin/env python3
"""
plot_A_biological_interpretation.py
  MAP A matrix biological interpretation:
  1. A_ij vs |F_ij| scatter — prior strength vs fitted interaction magnitude
  2. Annotated heatmap — top pairs labelled
  3. Bar chart — top facilitation/competition pairs with prior flag
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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
sp  = np.sign(nfs)

GS   = [g[:8] for g in GUILD_ORDER]
GFULL = GUILD_ORDER

# Literature-known interactions (oral biofilm)
KNOWN = {
    ('Actinobacteria', 'Bacilli'):        'Co-aggregation\n(Kolenbrander 1993)',
    ('Bacilli', 'Betaproteobacteria'):    'Co-aggregation\n(Kolenbrander 2000)',
    ('Actinobacteria', 'Betaproteobacteria'): 'Co-colonisers\n(early biofilm)',
    ('Actinobacteria', 'Negativicutes'):  'Lactate cross-feeding\n(Mikx & van der Hoeven 1975)',
    ('Fusobacteriia', 'Bacteroidia'):     'Syntrophy\n(Kolenbrander 2010)',
    ('Bacilli', 'Negativicutes'):         'Lactate→propionate\n(Mikx & van der Hoeven 1975)',
}
KNOWN_IDX = {(GFULL.index(a), GFULL.index(b)): v for (a, b), v in KNOWN.items()
             if a in GFULL and b in GFULL}

# ── Figure 1: A_ij vs |F_ij| scatter ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.rcParams.update({'font.size': 9})

ax = axes[0]
for i in range(N_G):
    for j in range(i+1, N_G):
        aval = A[i, j]
        fval = abs(nfs[i, j])
        has_prior = fval > 0
        c = '#e74c3c' if aval > 0.05 else '#2980b9' if aval < -0.05 else '#95a5a6'
        m = 'o' if has_prior else 'x'
        ax.scatter(fval, aval, color=c, marker=m, s=60, alpha=0.8, zorder=3)
        # Annotate top pairs
        key = (i, j) if (i, j) in KNOWN_IDX else (j, i) if (j, i) in KNOWN_IDX else None
        if abs(aval) > 0.3:
            label = f'{GS[i]}\n↔{GS[j]}'
            ax.annotate(label, (fval, aval), fontsize=6.5,
                        xytext=(4, 0), textcoords='offset points', va='center')

ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('Prior strength |F_ij|', fontsize=9)
ax.set_ylabel('Fitted A_ij', fontsize=9)
ax.set_title('Prior strength vs fitted interaction\n(○=constrained, ×=unconstrained)', fontsize=9)
leg = [mpatches.Patch(color='#e74c3c', label='Facilitation (A>0.05)'),
       mpatches.Patch(color='#2980b9', label='Competition (A<−0.05)'),
       mpatches.Patch(color='#95a5a6', label='Near-zero')]
ax.legend(handles=leg, fontsize=7, loc='upper right')

# ── Figure 2: Bar chart of top ±10 off-diagonal pairs ───────────────────────
ax2 = axes[1]
pairs = [(A[i,j], i, j, abs(nfs[i,j])) for i in range(N_G) for j in range(i+1, N_G)]
pairs.sort(key=lambda x: abs(x[0]), reverse=True)
top = pairs[:16]

vals  = [p[0] for p in top]
labs  = [f'{GS[p[1]]}↔{GS[p[2]]}' for p in top]
fmags = [p[3] for p in top]
has_p = [f > 0 for f in fmags]
colors = ['#e74c3c' if v > 0 else '#2980b9' for v in vals]
hatch  = ['' if hp else '///' for hp in has_p]

bars = ax2.barh(range(len(vals)), vals, color=colors, alpha=0.8)
for bar, h in zip(bars, hatch):
    bar.set_hatch(h)

ax2.set_yticks(range(len(labs)))
ax2.set_yticklabels(labs, fontsize=7.5)
ax2.axvline(0, color='k', lw=0.5)
ax2.set_xlabel('A_ij (MAP)', fontsize=9)
ax2.set_title('Top interactions by |A_ij|\n(/// = no metabolic prior)', fontsize=9)

# Prior-strength annotation
for k, (v, fmag) in enumerate(zip(vals, fmags)):
    if fmag > 0:
        ax2.text(v + 0.01 * np.sign(v), k, f'|F|={fmag:.1f}', va='center',
                 ha='left' if v > 0 else 'right', fontsize=6, color='#555')

fig.suptitle('AGORA W=1.0 MAP: A matrix biological interpretation', fontsize=10)
fig.tight_layout()
out = FIG / 'fig_A_biological_interpretation.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')

# ── Figure 3: Annotated heatmap ──────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 7))
im = ax3.imshow(A, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
plt.colorbar(im, ax=ax3, shrink=0.8, label='A_ij (MAP)')
ax3.set_xticks(range(N_G)); ax3.set_yticks(range(N_G))
ax3.set_xticklabels(GS, fontsize=7, rotation=45, ha='right')
ax3.set_yticklabels(GS, fontsize=7)
ax3.set_title('A matrix (AGORA W=1.0 MAP)\nBiological annotations', fontsize=9)

ANNOT = {
    (GFULL.index('Actinobacteria'), GFULL.index('Bacilli')):
        'co-agg.',
    (GFULL.index('Bacilli'), GFULL.index('Betaproteobacteria')):
        'co-agg.',
    (GFULL.index('Actinobacteria'), GFULL.index('Negativicutes')):
        'lactate',
    (GFULL.index('Actinobacteria'), GFULL.index('Bacteroidia')):
        'AA supply',
    (GFULL.index('Bacteroidia'), GFULL.index('Fusobacteriia')):
        'syntrophy',
}
for (i, j), label in ANNOT.items():
    for (ri, rj) in [(i, j), (j, i)]:
        ax3.text(rj, ri, label, ha='center', va='center', fontsize=5.5,
                 color='white' if abs(A[ri, rj]) > 0.3 else 'black',
                 fontweight='bold')

# Mark unconstrained pairs
for i in range(N_G):
    for j in range(N_G):
        if i != j and nfs[i, j] == 0:
            ax3.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                          fill=False, edgecolor='grey', lw=0.8, ls=':'))

fig3.tight_layout()
out3 = FIG / 'fig_A_heatmap_annotated.png'
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f'Saved {out3}')

# ── Print summary ─────────────────────────────────────────────────────────────
print('\n=== Biological summary ===')
print('\nData+prior aligned (|A|>0.3, |F|>1):')
for v, i, j, fmag in pairs:
    if abs(v) > 0.3 and fmag > 1.0:
        lit = KNOWN_IDX.get((i,j), KNOWN_IDX.get((j,i), '—'))
        print(f'  {GFULL[i]:20s} ↔ {GFULL[j]:20s}  A={v:+.3f}  |F|={fmag:.1f}  lit="{lit}"')

print('\nPrior-constrained, data-small (|A|<0.01, |F|>1):')
for v, i, j, fmag in pairs:
    if abs(v) < 0.01 and fmag > 1.0:
        lit = KNOWN_IDX.get((i,j), KNOWN_IDX.get((j,i), '—'))
        print(f'  {GFULL[i]:20s} ↔ {GFULL[j]:20s}  A={v:+.4f}  |F|={fmag:.1f}  lit="{lit}"')

print('\nData-driven, no prior (|A|>0.05, |F|=0):')
for v, i, j, fmag in pairs:
    if abs(v) > 0.05 and fmag == 0.0:
        print(f'  {GFULL[i]:20s} ↔ {GFULL[j]:20s}  A={v:+.3f}  (no prior)')
