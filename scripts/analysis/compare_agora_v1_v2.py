#!/usr/bin/env python3
"""
Compare AGORA v1 / v2 / MICOM sign priors.
Prints sign matrices and generates a comparison figure.

MICOM mode: community FBA cooperative tradeoff (Diener 2020).
  - fraction=0.5 (Diener 2020 default)
  - flux magnitude weighting (not binary count)
  - toxins harm all co-occurring guilds
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/nife')
from build_net_flow_expanded import build_net_flow_expanded
from guild_replicator_dieckow import GUILD_SHORT_LIST

OUT = Path('/home/nishioka/IKM_Hiwi/nife/results')
N = 10
SHORT = GUILD_SHORT_LIST[:N]

print('Building v1 (symmetric)...', flush=True)
nf_v1_sym = build_net_flow_expanded(use_agora=True, agora_medium='v1',
                                     symmetrize=True, verbose=True)
print('Building v2 (symmetric)...', flush=True)
nf_v2_sym = build_net_flow_expanded(use_agora=True, agora_medium='v2',
                                     symmetrize=True, verbose=True)
print('Building MICOM perfect (symmetric)...', flush=True)
nf_micom_sym = build_net_flow_expanded(use_agora=True, agora_medium='micom',
                                        micom_fraction=0.5,
                                        symmetrize=True, verbose=True)
print('Building v1 (directed)...', flush=True)
nf_v1_dir = build_net_flow_expanded(use_agora=True, agora_medium='v1',
                                     symmetrize=False, verbose=True)
print('Building v2 (directed)...', flush=True)
nf_v2_dir = build_net_flow_expanded(use_agora=True, agora_medium='v2',
                                     symmetrize=False, verbose=True)
print('Building MICOM perfect (directed)...', flush=True)
nf_micom_dir = build_net_flow_expanded(use_agora=True, agora_medium='micom',
                                        micom_fraction=0.5,
                                        symmetrize=False, verbose=True)

# L1+L2 baseline
nf_l12_sym = build_net_flow_expanded(use_agora=False, symmetrize=True, verbose=False)
nf_l12_dir = build_net_flow_expanded(use_agora=False, symmetrize=False, verbose=False)

def make_cat(nf_base, nf_full):
    cat = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            vb, vf = nf_base[i,j], nf_full[i,j]
            if   vb != 0 and vf != 0: cat[i,j] = 1 if vb > 0 else 2
            elif vb != 0 and vf == 0: cat[i,j] = 6   # cancelled
            elif vb == 0 and vf != 0: cat[i,j] = 3 if vf > 0 else 4
            else:                      cat[i,j] = 5
    return cat

cat_v1_sym   = make_cat(nf_l12_sym, nf_v1_sym)
cat_v2_sym   = make_cat(nf_l12_sym, nf_v2_sym)
cat_micom_sym = make_cat(nf_l12_sym, nf_micom_sym)
cat_v1_dir   = make_cat(nf_l12_dir, nf_v1_dir)
cat_v2_dir   = make_cat(nf_l12_dir, nf_v2_dir)
cat_micom_dir = make_cat(nf_l12_dir, nf_micom_dir)

def count_stats(cat, sym=True):
    stats = {}
    for c in range(7):
        if sym:
            # upper triangle only
            mask = np.triu(np.ones((N,N), bool), k=1)
        else:
            mask = ~np.eye(N, dtype=bool)
        stats[c] = int((cat[mask] == c).sum())
    return stats

print('\n=== Symmetric (Hamilton) ===')
for name, cat in [('v1', cat_v1_sym), ('v2', cat_v2_sym), ('micom_perf', cat_micom_sym)]:
    s = count_stats(cat, sym=True)
    print(f'  {name}: L12+={s[1]} L12-={s[2]} AGORA+={s[3]} AGORA-={s[4]} '
          f'cancelled={s[6]} free={s[5]}  '
          f'total_constrained={s[1]+s[2]+s[3]+s[4]}/45')

print('\n=== Directed (gLV) ===')
for name, cat in [('v1', cat_v1_dir), ('v2', cat_v2_dir), ('micom_perf', cat_micom_dir)]:
    s = count_stats(cat, sym=False)
    print(f'  {name}: L12+={s[1]} L12-={s[2]} AGORA+={s[3]} AGORA-={s[4]} '
          f'cancelled={s[6]} free={s[5]}  '
          f'total_constrained={s[1]+s[2]+s[3]+s[4]}/90')

# ── Diff matrices ────────────────────────────────────────────────────────────
diff_v1_v2_sym    = (cat_v1_sym   != cat_v2_sym).astype(int)
diff_v1_micom_sym = (cat_v1_sym   != cat_micom_sym).astype(int)
diff_v1_v2_dir    = (cat_v1_dir   != cat_v2_dir).astype(int)
diff_v1_micom_dir = (cat_v1_dir   != cat_micom_dir).astype(int)

print(f'\nDifferences v1 vs v2   sym: {diff_v1_v2_sym.sum()} cells changed')
print(f'Differences v1 vs micom sym: {diff_v1_micom_sym.sum()} cells changed')
print(f'Differences v1 vs v2   dir: {diff_v1_v2_dir.sum()} cells changed')
print(f'Differences v1 vs micom dir: {diff_v1_micom_dir.sum()} cells changed')

print('\n-- Symmetric: v1 vs micom_perf changed pairs --')
for i in range(N):
    for j in range(i+1, N):
        if cat_v1_sym[i,j] != cat_micom_sym[i,j]:
            print(f'  {SHORT[i]} ↔ {SHORT[j]}: v1={cat_v1_sym[i,j]} → micom={cat_micom_sym[i,j]}')

print('\n-- Directed: v1 vs micom_perf changed pairs --')
for i in range(N):
    for j in range(N):
        if i == j: continue
        if cat_v1_dir[i,j] != cat_micom_dir[i,j]:
            print(f'  {SHORT[j]}→{SHORT[i]}: v1={cat_v1_dir[i,j]} → micom={cat_micom_dir[i,j]}')

# ── Figure: 2×2 comparison ──────────────────────────────────────────────────
COLORS = ['#555555','#1a7340','#1a3d7a','#72c78a','#72a4e0','#d9c97a','#e8a030']
CMAP   = ListedColormap(COLORS)
SIGNS  = {1:'+', 2:'−', 3:'+', 4:'−', 5:'?', 6:'✕'}
TXTC   = {1:'white',2:'white',3:'#111',4:'#111',5:'#999',6:'white'}

def draw(ax, cat, title, sym=True):
    ax.imshow(cat, cmap=CMAP, vmin=0, vmax=6, aspect='equal')
    for k in range(N+1):
        ax.axhline(k-.5, color='white', lw=.5)
        ax.axvline(k-.5, color='white', lw=.5)
    ax.plot([-.5,N-.5],[-.5,N-.5],'k--',lw=1.1,alpha=.4,zorder=5)
    for i in range(N):
        for j in range(N):
            c = cat[i,j]
            if c == 0:
                ax.text(j,i,'●',ha='center',va='center',fontsize=7,color='white',zorder=6)
            elif c == 5:
                ax.text(j,i,'?',ha='center',va='center',fontsize=8,color='#999',zorder=6)
            elif c in SIGNS:
                ax.text(j,i,SIGNS[c],ha='center',va='center',
                        fontsize=10,fontweight='bold',color=TXTC[c],zorder=6)
    ax.set_xticks(range(N)); ax.set_xticklabels(SHORT,rotation=45,ha='right',fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(SHORT,fontsize=8)
    ax.set_xlabel('Source $j$', fontsize=9); ax.set_ylabel('Target $i$', fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight='bold', loc='left', pad=5)
    for k in [9]:
        ax.add_patch(plt.Rectangle((k-.5,-.5),1,N,fill=False,edgecolor='#cc2222',
                     lw=1.1,ls=':',zorder=7,alpha=.7))
        ax.add_patch(plt.Rectangle((-.5,k-.5),N,1,fill=False,edgecolor='#cc2222',
                     lw=1.1,ls=':',zorder=7,alpha=.7))
    for ri,ci in [(2,6),(6,2)]:
        ax.add_patch(plt.Rectangle((ci-.5,ri-.5),1,1,fill=False,
                     edgecolor='#e000e0',lw=1.6,zorder=8))

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.patch.set_facecolor('white')

sv1   = count_stats(cat_v1_sym)
sv2   = count_stats(cat_v2_sym)
sm    = count_stats(cat_micom_sym)
dv1   = count_stats(cat_v1_dir, sym=False)
dv2   = count_stats(cat_v2_dir, sym=False)
dm    = count_stats(cat_micom_dir, sym=False)

draw(axes[0][0], cat_v1_sym,
     f'A1  AGORA v1 symmetric\n'
     f'AGORA+:{sv1[3]}  AGORA-:{sv1[4]}  cancelled:{sv1[6]}  free:{sv1[5]}/45')
draw(axes[0][1], cat_v2_sym,
     f'A2  AGORA v2 symmetric (realistic saliva)\n'
     f'AGORA+:{sv2[3]}  AGORA-:{sv2[4]}  cancelled:{sv2[6]}  free:{sv2[5]}/45')
draw(axes[0][2], cat_micom_sym,
     f'A3  MICOM perfect symmetric (fraction=0.5)\n'
     f'AGORA+:{sm[3]}  AGORA-:{sm[4]}  cancelled:{sm[6]}  free:{sm[5]}/45')
draw(axes[1][0], cat_v1_dir,
     f'B1  AGORA v1 directed\n'
     f'AGORA+:{dv1[3]}  AGORA-:{dv1[4]}  cancelled:{dv1[6]}  free:{dv1[5]}/90', sym=False)
draw(axes[1][1], cat_v2_dir,
     f'B2  AGORA v2 directed (realistic saliva)\n'
     f'AGORA+:{dv2[3]}  AGORA-:{dv2[4]}  cancelled:{dv2[6]}  free:{dv2[5]}/90', sym=False)
draw(axes[1][2], cat_micom_dir,
     f'B3  MICOM perfect directed (fraction=0.5)\n'
     f'AGORA+:{dm[3]}  AGORA-:{dm[4]}  cancelled:{dm[6]}  free:{dm[5]}/90', sym=False)

patches = [
    mpatches.Patch(color='#1a7340', label='L1+L2 positive (+)'),
    mpatches.Patch(color='#1a3d7a', label='L1+L2 negative (−)'),
    mpatches.Patch(color='#72c78a', label='AGORA added (+)'),
    mpatches.Patch(color='#72a4e0', label='AGORA added (−)'),
    mpatches.Patch(color='#e8a030', label='L1+L2 cancelled by AGORA (✕)'),
    mpatches.Patch(color='#d9c97a', label='Unconstrained (?)'),
    mpatches.Patch(color='#555555', label='Self-limitation (diagonal)'),
]
fig.legend(handles=patches, fontsize=9, ncol=4,
           loc='lower center', bbox_to_anchor=(0.5, 0.01),
           framealpha=0.95, edgecolor='#cccccc')

fig.suptitle('AGORA Sign Prior Comparison: v1 / v2 / MICOM-perfect\n'
             'v1: blood-plasma scale  |  v2: realistic saliva  |  '
             'MICOM: community FBA τ=0.5, flux-weighted, toxin→all-guilds',
             fontsize=12, fontweight='bold')

fig.tight_layout(rect=[0, 0.07, 1, 0.94])
fig.savefig(OUT / 'fig_agora_v1_v2_micom_comparison.png', dpi=150, bbox_inches='tight')
fig.savefig(OUT / 'fig_agora_v1_v2_micom_comparison.pdf', dpi=150, bbox_inches='tight')
print('\nSaved fig_agora_v1_v2_micom_comparison.{png,pdf}')
