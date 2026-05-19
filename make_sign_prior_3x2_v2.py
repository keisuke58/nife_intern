"""
3×2 sign prior figure — AGORA v2 (realistic saliva concentrations).
  Row 1: L1+L2                | sym | directed
  Row 2: L1+L2+AGORA v2       | sym | directed
  Row 3: Unconstrained         | sym | directed
"""
import numpy as np, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/nife')
from build_net_flow_expanded import build_net_flow_expanded

OUT = Path('/home/nishioka/IKM_Hiwi/nife/results')

SHORT = ['Actin.','Bacil.','Bact.','β-Prot.','Clost.',
         'Coriob.','Fusob.','γ-Prot.','Negat.','Other']
N = 10

print('Loading matrices...', flush=True)
nf_l12_sym  = build_net_flow_expanded(use_agora=False, symmetrize=True,  verbose=False)
nf_l12_dir  = build_net_flow_expanded(use_agora=False, symmetrize=False, verbose=False)
nf_ag2_sym  = build_net_flow_expanded(use_agora=True, agora_medium='v2', symmetrize=True,  verbose=True)
nf_ag2_dir  = build_net_flow_expanded(use_agora=True, agora_medium='v2', symmetrize=False, verbose=True)
print('Done.', flush=True)

def make_cat(nf_base, nf_full):
    cat = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            vb, vf = nf_base[i,j], nf_full[i,j]
            if   vb != 0 and vf != 0: cat[i,j] = 1 if vb > 0 else 2
            elif vb != 0 and vf == 0: cat[i,j] = 6
            elif vb == 0 and vf != 0: cat[i,j] = 3 if vf > 0 else 4
            else:                      cat[i,j] = 5
    return cat

cat_l12_sym = make_cat(nf_l12_sym, nf_l12_sym)
cat_l12_dir = make_cat(nf_l12_dir, nf_l12_dir)
cat_ag_sym  = make_cat(nf_l12_sym, nf_ag2_sym)
cat_ag_dir  = make_cat(nf_l12_dir, nf_ag2_dir)

def make_uc_cat(cat_full):
    out = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:      out[i,j] = 0
            elif cat_full[i,j] == 5: out[i,j] = 5
            else:            out[i,j] = 7
    return out

cat_uc_sym_inv = make_uc_cat(cat_ag_sym)
cat_uc_dir_inv = make_uc_cat(cat_ag_dir)

# 0=diag,1=L12+,2=L12-,3=AGORA+,4=AGORA-,5=free,6=cancelled,7=dimmed_constrained
COLORS = ['#555555','#1a7340','#1a3d7a','#72c78a','#72a4e0',
          '#d9c97a','#e8a030','#d0d0d0']
CMAP   = ListedColormap(COLORS)
SIGNS  = {1:'+', 2:'−', 3:'+', 4:'−', 5:'?', 6:'✕'}
TXTC   = {1:'white',2:'white',3:'#111',4:'#111',5:'#999',6:'white',7:''}

def count(cat, ids, sym=True):
    s = 0
    for i in range(N):
        for j in range(N):
            if sym and i >= j: continue
            if not sym and i == j: continue
            if cat[i,j] in ids: s += 1
    return s

def draw(ax, cat, title, sym=True, fs=8.5):
    ax.imshow(cat, cmap=CMAP, vmin=0, vmax=7, aspect='equal')
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
                ax.text(j,i,'?',ha='center',va='center',fontsize=8,
                        color='#999',fontweight='normal',zorder=6)
            elif c in SIGNS and c != 5:
                ax.text(j,i,SIGNS[c],ha='center',va='center',
                        fontsize=10,fontweight='bold',color=TXTC[c],zorder=6)
    ax.set_xticks(range(N)); ax.set_xticklabels(SHORT,rotation=45,ha='right',fontsize=fs)
    ax.set_yticks(range(N)); ax.set_yticklabels(SHORT,fontsize=fs)
    ax.set_xlabel('Source $j$', fontsize=9)
    ax.set_ylabel('Target $i$', fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight='bold', loc='left', pad=6)
    for k in [9]:
        ax.add_patch(plt.Rectangle((k-.5,-.5),1,N,fill=False,
                     edgecolor='#cc2222',lw=1.1,ls=':',zorder=7,alpha=.7))
        ax.add_patch(plt.Rectangle((-.5,k-.5),N,1,fill=False,
                     edgecolor='#cc2222',lw=1.1,ls=':',zorder=7,alpha=.7))
    ax.text(9,-.9,'†',ha='center',fontsize=8,color='#cc2222')
    for ri,ci in [(2,6),(6,2)]:
        ax.add_patch(plt.Rectangle((ci-.5,ri-.5),1,1,fill=False,
                     edgecolor='#e000e0',lw=1.6,zorder=8))

fig = plt.figure(figsize=(16, 22))
fig.patch.set_facecolor('white')
gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.22,
              left=0.07, right=0.97, top=0.95, bottom=0.07)

axes = [[fig.add_subplot(gs[r,c]) for c in range(2)] for r in range(3)]

for r, lbl in enumerate(['L1+L2','L1+L2 + AGORA v2','Unconstrained']):
    fig.text(0.01, 0.95 - r*0.328, lbl, va='center', ha='left',
             fontsize=11, fontweight='bold', rotation=90, color='#333')

for c, lbl in enumerate(['Symmetric  ($A_{ij}=A_{ji}$, Hamilton)',
                          'Directed / Asymmetric  ($A_{ij}\\neq A_{ji}$, gLV)']):
    fig.text(0.07 + c*0.46, 0.965, lbl, ha='left', va='bottom',
             fontsize=10.5, fontweight='bold', color='#1a3d7a' if c==0 else '#1a7340')

# Row 0: L1+L2
draw(axes[0][0], cat_l12_sym,
     f'A1  L1+L2  symmetric\n'
     f'{count(cat_l12_sym,[1,2])}/45 constrained  (+:{count(cat_l12_sym,[1])}  −:{count(cat_l12_sym,[2])})')
draw(axes[0][1], cat_l12_dir,
     f'A2  L1+L2  directed\n'
     f'{count(cat_l12_dir,[1,2],False)}/90 constrained  (+:{count(cat_l12_dir,[1],False)}  −:{count(cat_l12_dir,[2],False)})', sym=False)

# Row 1: L1+L2 + AGORA v2
draw(axes[1][0], cat_ag_sym,
     f'B1  L1+L2 + AGORA v2  symmetric\n'
     f'{count(cat_ag_sym,[1,2,3,4])}/45  '
     f'AGORA+:{count(cat_ag_sym,[3])}  AGORA−:{count(cat_ag_sym,[4])}  '
     f'cancelled(✕):{count(cat_ag_sym,[6])}  free:{count(cat_ag_sym,[5])}')
draw(axes[1][1], cat_ag_dir,
     f'B2  L1+L2 + AGORA v2  directed\n'
     f'{count(cat_ag_dir,[1,2,3,4],False)}/90  '
     f'AGORA+:{count(cat_ag_dir,[3],False)}  AGORA−:{count(cat_ag_dir,[4],False)}  '
     f'cancelled:{count(cat_ag_dir,[6],False)}  free:{count(cat_ag_dir,[5],False)}', sym=False)

# Row 2: Unconstrained
n_uc_s = count(cat_ag_sym,[5]); n_uc_d = count(cat_ag_dir,[5],False)
draw(axes[2][0], cat_uc_sym_inv,
     f'C1  Unconstrained  symmetric\n'
     f'{n_uc_s}/45 pairs  (? = no sign prior)', sym=True)
draw(axes[2][1], cat_uc_dir_inv,
     f'C2  Unconstrained  directed\n'
     f'{n_uc_d}/90 directed pairs  (? = no sign prior)', sym=False)

patches = [
    mpatches.Patch(color='#1a7340', label='L1+L2 positive (+)'),
    mpatches.Patch(color='#1a3d7a', label='L1+L2 negative (−)'),
    mpatches.Patch(color='#72c78a', label='AGORA v2 added (+)  cross-feeding'),
    mpatches.Patch(color='#72a4e0', label='AGORA v2 added (−)  competition'),
    mpatches.Patch(color='#e8a030', label='L1+L2 cancelled by AGORA (✕)'),
    mpatches.Patch(color='#d9c97a', label='Unconstrained (?)'),
    mpatches.Patch(color='#d0d0d0', label='Constrained (dimmed in Row C)'),
    mpatches.Patch(color='#555555', label='Self-limitation (diagonal)'),
    mpatches.Patch(facecolor='white', edgecolor='#cc2222', ls=':',
                   label='† Other: excluded from AGORA'),
    mpatches.Patch(facecolor='white', edgecolor='#e000e0',
                   label='▪ Bact–Fusob: competition signal in v2'),
]
fig.legend(handles=patches, fontsize=8.5, ncol=5,
           loc='lower center', bbox_to_anchor=(0.5, 0.01),
           framealpha=0.95, edgecolor='#cccccc')

fig.suptitle('Sign Prior Coverage — AGORA v2 (realistic saliva, MacArthur competition)\n'
             'Symmetric vs Directed  (Dieckow 2024, 10 guilds)',
             fontsize=13, fontweight='bold')

for ext in ('png','pdf'):
    fig.savefig(OUT / f'fig_sign_prior_3x2_v2.{ext}', dpi=150, bbox_inches='tight')
print('Saved fig_sign_prior_3x2_v2.{png,pdf}')
