#!/usr/bin/env python3
"""
Publication-quality figure generation for dieckow_analysis paper.
Outputs to dieckow_paper/figures/ at 300 DPI.

Figures generated:
  fig1_A_matrix.png            — AGORA W=1.0 interaction matrix (heatmap)
  fig2_loo_comparison.png      — LOO-RMSE per patient + scatter
  fig3_joshi_attractor.png     — External validation GDI violin
  fig4_b_pca.png               — b-vector PCA (CT1/CT2 corrected labels)
  fig5_A_biological.png        — Interaction regime scatter + bar
  fig6_agora_weight_sweep.png  — AGORA weight sensitivity (3 panels)
  fig7_loo_stability.png       — A matrix sign consistency heatmap
  fig8_all_models_rmse_bc.png  — LOO RMSE + BC model comparison
"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sys as _sys
_sys.path.insert(0, '/home/nishioka/IKM_Hiwi/nife')   # for thesis_style
from thesis_style import use as thesis_style, TEXTWIDTH_IN

# ── Paths ─────────────────────────────────────────────────────────────────────
CR   = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr')
OTU  = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_otu')
DEST = Path(__file__).parent / 'figures'
DEST.mkdir(exist_ok=True)

# ── Global style (Nature/Science journal quality) ─────────────────────────────
NAVY  = '#1a3a5c'
BLUE  = '#2166ac'
RED   = '#d6604d'
GREEN = '#4dac26'
GREY  = '#888888'
LGREY = '#dddddd'

# Unified thesis style (usetex / lmodern, 9 pt, body-matched). Aesthetic extras
# (spines off, tick widths) are layered on top without touching the font settings.
thesis_style()
plt.rcParams.update({
    'axes.titleweight':   'bold',
    'legend.framealpha':  0.9,
    'legend.edgecolor':   LGREY,
    'figure.dpi':         150,
    'savefig.pad_inches': 0.05,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
})

# ── Data loading ──────────────────────────────────────────────────────────────
PATIENTS = list('ABCDEFGHKL')
CT_MAP   = json.load(open(OTU / 'community_types.json'))['patient_ct']  # {pat: 1 or 2}

d_agora   = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
A_agora   = np.array(d_agora['A'])
b_all     = np.array(d_agora['b_all'])   # (10,10)
net_flow  = np.array(d_agora['net_flow'])
GUILDS    = d_agora['guilds']
G_SHORT   = ['Actin.','Bacil.','Bact.','β-Prot.','Clost.',
              'Corio.','Fusob.','γ-Prot.','Negat.','Other']

def load_loo_folds(pattern):
    """Load per-fold RMSE dicts."""
    out = {}
    for i, p in enumerate(PATIENTS):
        f = CR / pattern.format(i)
        if f.exists():
            d = json.load(open(f))
            out[p] = d.get('rmse', None)
    return out

agora_pp = load_loo_folds('loo_expanded_agora_w1p0_fold{}.json')
l12_pp   = load_loo_folds('loo_expanded_a0p0_l12_fold{}.json')
free_pp  = load_loo_folds('loo_expanded_a0p0_fold{}.json')

def save(fig, name):
    path = DEST / name
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    pdf_path = path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved {path.name}  ({path.stat().st_size//1024} KB)  +PDF')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — A matrix heatmap (AGORA W=1.0, full 10-patient fit)
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 1: A matrix heatmap...')

net_sym = (net_flow + net_flow.T) / 2.0
sp_sym  = np.sign(net_sym)

# classify regimes
mag = np.abs(A_agora)
wt  = np.abs(net_sym)
regime = np.full((10, 10), 'data_driven', dtype=object)
for i in range(10):
    for j in range(10):
        if i == j:
            regime[i, j] = 'diagonal'
        elif wt[i, j] > 1.0 and mag[i, j] > 0.3:
            regime[i, j] = 'aligned'
        elif wt[i, j] > 1.0 and mag[i, j] <= 0.3:
            regime[i, j] = 'muted'
        elif wt[i, j] == 0.0:
            regime[i, j] = 'data_driven'
        else:
            regime[i, j] = 'weak_prior'

fig, ax = plt.subplots(figsize=(6.0, 5.0))

vmax = 2.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(A_agora, cmap='RdBu_r', norm=norm, aspect='equal')

ax.set_xticks(range(10))
ax.set_xticklabels(G_SHORT, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(10))
ax.set_yticklabels(G_SHORT, fontsize=9)

# cell annotations: value + sign indicator
for i in range(10):
    for j in range(10):
        val = A_agora[i, j]
        txt = f'{val:.2f}'
        # background brightness
        bg_dark = abs(val) > 1.0
        color = 'white' if bg_dark else 'black'
        ax.text(j, i, txt, ha='center', va='center',
                fontsize=6.5, color=color, fontweight='normal')

        # sign indicator box (only for constrained pairs, off-diagonal)
        if i != j and wt[i, j] > 0.5:
            correct = (sp_sym[i, j] * val > 0)
            ec = '#2ecc71' if correct else '#e74c3c'
            rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                  linewidth=2.0, edgecolor=ec,
                                  facecolor='none', zorder=5)
            ax.add_patch(rect)

cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label('Interaction strength $A_{ij}$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Regime legend
from matplotlib.lines import Line2D
legend_els = [
    mpatches.Patch(facecolor='none', edgecolor='#2ecc71', linewidth=2,
                   label='Sign ✓ (metabolic prior)'),
    mpatches.Patch(facecolor='none', edgecolor='#e74c3c', linewidth=2,
                   label='Sign ✗ (discordant)'),
]
ax.legend(handles=legend_els, loc='lower right', fontsize=8,
          bbox_to_anchor=(1.0, -0.22), ncol=2)

ax.set_title('Interaction matrix $A$ — Hamilton + AGORA W=1.0\n'
             f'Training RMSE = {d_agora["rmse"]:.4f}  |  '
             f'Pearson $r$ = {d_agora["r"]:.3f}  |  SA = {d_agora["sign_agreement"]}',
             fontsize=10, pad=8)

ax.tick_params(length=0)
fig.tight_layout()
save(fig, 'fig1_A_matrix.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — LOO-RMSE comparison: L1+L2 vs AGORA W=1.0
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 2: LOO comparison...')

pats    = [p for p in PATIENTS if p in agora_pp and p in l12_pp]
ag_v    = [agora_pp[p] for p in pats]
l12_v   = [l12_pp[p]   for p in pats]
free_v  = [free_pp.get(p, None) for p in pats]
delta   = [a - b for a, b in zip(ag_v, l12_v)]
ct_col  = [BLUE if CT_MAP[p] == 1 else RED for p in pats]
ct_lbl  = [f'CT{CT_MAP[p]}' for p in pats]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

# ── Panel A: grouped bar ──────────────────────────────────────────────────────
ax = axes[0]
x = np.arange(len(pats))
w = 0.32
b1 = ax.bar(x - w/2, l12_v,  w, color='#6baed6', label='L1+L2 (34 pairs)',
            edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, ag_v,   w, color=NAVY,      label='AGORA W=1.0 (35 pairs)',
            edgecolor='white', linewidth=0.5)

mean_l12  = np.mean(l12_v)
mean_ag   = np.mean(ag_v)
ax.axhline(mean_l12, color='#6baed6', ls='--', lw=1.2, alpha=0.8)
ax.axhline(mean_ag,  color=NAVY,      ls='--', lw=1.2, alpha=0.8)

# CT annotation above bars
for i, p in enumerate(pats):
    ymax = max(l12_v[i], ag_v[i])
    ax.text(i, ymax + 0.003, ct_lbl[i], ha='center', va='bottom',
            fontsize=7.5, color=ct_col[i], fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(pats)
ax.set_ylabel('LOO-RMSE')
ax.set_xlabel('Patient')
ax.set_ylim(0, max(max(l12_v), max(ag_v)) * 1.18)
ax.legend(loc='upper left', fontsize=8)
ax.set_title(f'(A)  Per-patient LOO-RMSE\n'
             f'L1+L2 mean = {mean_l12:.4f}  |  AGORA mean = {mean_ag:.4f}  '
             f'(−{(mean_l12-mean_ag)/mean_l12*100:.1f}\%)',
             fontsize=10)

# significance brackets for improved patients
for i, p in enumerate(pats):
    if delta[i] < -0.002:
        y = max(l12_v[i], ag_v[i]) + 0.0065
        ax.plot([i-w/2, i+w/2], [y, y], color=GREEN, lw=1.2)
        ax.text(i, y+0.001, '↓', ha='center', va='bottom',
                color=GREEN, fontsize=7)

# ── Panel B: scatter L1+L2 vs AGORA, coloured by delta ────────────────────────
ax = axes[1]
lim = max(max(l12_v), max(ag_v)) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, zorder=1)

norm_delta = plt.Normalize(-0.007, 0.005)
cmap_delta = plt.cm.RdYlGn_r

for p, l, a, d in zip(pats, l12_v, ag_v, delta):
    c = cmap_delta(norm_delta(d))
    ax.scatter(l, a, color=c, s=70, zorder=5, edgecolors='white', linewidths=0.6)
    ax.annotate(p, (l, a), xytext=(5, 3), textcoords='offset points',
                fontsize=8.5, fontweight='bold',
                color=BLUE if CT_MAP[p] == 1 else RED)

sm = plt.cm.ScalarMappable(cmap=cmap_delta, norm=norm_delta)
cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.03)
cbar.set_label('ΔRMSE (AGORA − L1+L2)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('L1+L2 LOO-RMSE')
ax.set_ylabel('AGORA W=1.0 LOO-RMSE')
ax.set_title('(B)  AGORA vs L1+L2 (below diagonal = improved)\n'
             '7/10 patients improved  |  paired $t$-test $p$=0.08',
             fontsize=10)

# CT legend
ax.scatter([], [], color=BLUE, s=55, label='CT1 (commensal)')
ax.scatter([], [], color=RED,  s=55, label='CT2 (dysbiotic)')
ax.legend(fontsize=8)

fig.tight_layout(w_pad=2.5)
save(fig, 'fig2_loo_comparison.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Joshi attractor (simplified: 1 panel, neutral b)
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 3: Joshi attractor...')

joshi = pd.read_csv(CR / 'joshi_attractor_results.csv')
diag_order = ['Health', 'Mucositis', 'Peri-implantitis']
diag_short = {'Health': 'Health\n(n=56)', 'Mucositis': 'Mucositis\n(n=39)',
               'Peri-implantitis': 'Peri-impl.\n(n=32)'}
diag_col   = {'Health': '#2166ac', 'Mucositis': '#f4a582', 'Peri-implantitis': '#d6604d'}

fig, ax = plt.subplots(figsize=(5.0, 5.0))

positions = [0, 1, 2]
for pos, diag in zip(positions, diag_order):
    subset = joshi[joshi['diagnosis'] == diag]['gdi_raw_log_ratio'].values
    col = diag_col[diag]

    # violin
    vp = ax.violinplot(subset, positions=[pos], widths=0.6,
                       showmedians=False, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor(col)
        body.set_alpha(0.55)
        body.set_edgecolor('none')

    # jitter strip
    jx = np.random.normal(pos, 0.07, size=len(subset))
    ax.scatter(jx, subset, color=col, s=18, alpha=0.75, zorder=4,
               edgecolors='white', linewidths=0.3)

    # median bar
    med = np.median(subset)
    ax.plot([pos - 0.18, pos + 0.18], [med, med], color='black',
            lw=2.0, zorder=6)

    # mean annotation. For Health, anchor just above the x-axis (axes-fraction
    # y via get_xaxis_transform); for the others, label the mean in place.
    mn = np.mean(subset)
    if pos == 0:
        ax.text(pos, 0.02, f'mean={mn:.2f}', ha='center', va='bottom',
                fontsize=8, color=col, fontweight='bold',
                transform=ax.get_xaxis_transform())
    else:
        ax.text(pos, mn, f'mean={mn:.2f}', ha='center', va='top', fontsize=8,
                color=col, fontweight='bold', transform=ax.transData)

ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5,
           label='GDI = 0 (equal dysbiotic/commensal)')

# stats annotations
H_pi_pval = stats.mannwhitneyu(
    joshi[joshi['diagnosis']=='Health']['gdi_raw_log_ratio'],
    joshi[joshi['diagnosis']=='Peri-implantitis']['gdi_raw_log_ratio'],
    alternative='less').pvalue
kw_stat, kw_p = stats.kruskal(
    *[joshi[joshi['diagnosis']==d]['gdi_raw_log_ratio'].values for d in diag_order])

ax.text(0.97, 0.97,
        f'Kruskal–Wallis $p$ < 0.001\n'
        f'Health vs PI:  $p$ < 0.001\n'
        f'Spearman $\\rho$ = 0.37  ($p$ < 0.001)',
        ha='right', va='top', transform=ax.transAxes,
        fontsize=8.5, bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                ec=LGREY, lw=0.8))

ax.set_xticks(positions)
ax.set_xticklabels([diag_short[d] for d in diag_order], fontsize=9)
ax.set_ylabel('Guild Dysbiosis Index (GDI)\n'
              r'$\log(\phi_{\rm Fuso}+\phi_{\rm Bact}+\phi_{\rm Clost})$'
              r'$-\log(\phi_{\rm Bac}+\phi_{\rm Act}+\phi_{\rm Neg})$',
              fontsize=9)
ax.set_title('External validation: peri-implant dysbiosis prediction\n'
             'Dieckow $A$ matrix → Joshi et al. 2025 (n=127)',
             fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.spines['bottom'].set_visible(True)
fig.tight_layout()
save(fig, 'fig3_joshi_attractor.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — b-vector PCA (corrected: CT1/CT2 labels, biplot)
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 4: b-vector PCA (corrected CT labels)...')

# b_all: rows = patients (ABCDEFGHKL), cols = guilds
ct_labels = [CT_MAP[p] for p in PATIENTS]
ct_colors_per_pat = [BLUE if c == 1 else RED for c in ct_labels]

pca = PCA(n_components=2)
b_pc = pca.fit_transform(b_all)
loadings = pca.components_.T  # (10, 2)
ev = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(5.5, 5.0))

# Scatter patients
for i, p in enumerate(PATIENTS):
    c  = BLUE if CT_MAP[p] == 1 else RED
    mk = 'o' if CT_MAP[p] == 1 else 's'
    ax.scatter(b_pc[i, 0], b_pc[i, 1], color=c, marker=mk,
               s=90, zorder=5, edgecolors='white', linewidths=0.8)
    ax.text(b_pc[i, 0] + 0.03, b_pc[i, 1] + 0.04, p,
            fontsize=9, fontweight='bold', color=c)

# Biplot arrows (top 6 loadings by PC1 magnitude)
scale = 0.6 * max(np.abs(b_pc).max(axis=0)[:2])
order = np.argsort(-np.abs(loadings[:, 0]))[:7]
for gi in order:
    lx, ly = loadings[gi, 0] * scale, loadings[gi, 1] * scale
    ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=GREY,
                                lw=1.2, mutation_scale=10))
    ax.text(lx * 1.12, ly * 1.12, G_SHORT[gi],
            ha='center', va='center', fontsize=7.5, color=GREY)

ax.axhline(0, color=LGREY, lw=0.6)
ax.axvline(0, color=LGREY, lw=0.6)

# Legend
handles = [
    mpatches.Patch(color=BLUE, label='CT1 (commensal)'),
    mpatches.Patch(color=RED,  label='CT2 (dysbiotic)'),
]
ax.legend(handles=handles, fontsize=9, loc='upper right')

ax.set_xlabel(f'PC1  ({ev[0]*100:.1f}\% explained variance)', fontsize=10)
ax.set_ylabel(f'PC2  ({ev[1]*100:.1f}\% explained variance)', fontsize=10)
ax.set_title('PCA of patient-specific growth vectors $\\hat{b}^{(p)}$\n'
             '(CT1/CT2 separation without label supervision)',
             fontsize=10)
fig.tight_layout()
save(fig, 'fig4_b_pca.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — A matrix biological interpretation (regime scatter + ranked bar)
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 5: A matrix biological interpretation...')

# build per-pair data
pair_data = []
for i in range(10):
    for j in range(i+1, 10):
        pair_data.append({
            'label': f'{G_SHORT[i]}–{G_SHORT[j]}',
            'A_ij':  A_agora[i, j],
            'F_ij':  abs(net_sym[i, j]),
            'regime': regime[i, j],
        })
df = pd.DataFrame(pair_data)

regime_color = {
    'aligned':    '#d73027',   # red
    'muted':      '#4575b4',   # blue
    'data_driven':'#636363',   # grey
    'weak_prior': '#fc8d59',   # orange
}
regime_label = {
    'aligned':    'Data + prior aligned (n=13)',
    'muted':      'Prior-constrained, muted (n=22)',
    'data_driven':'Data-driven, no prior (n=10)',
    'weak_prior': 'Weak prior',
}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8),
                          gridspec_kw={'width_ratios': [1, 1.3]})

# ── Panel A: |F_ij| vs |A_ij| scatter ─────────────────────────────────────────
ax = axes[0]
for reg, grp in df.groupby('regime'):
    if reg == 'diagonal': continue
    ax.scatter(grp['F_ij'], np.abs(grp['A_ij']),
               color=regime_color.get(reg, GREY),
               label=regime_label.get(reg, reg),
               s=50, alpha=0.85, edgecolors='white', linewidths=0.5, zorder=4)

ax.set_xlabel('Prior stiffness $|F_{ij}|$', fontsize=10)
ax.set_ylabel('Fitted magnitude $|A_{ij}|$', fontsize=10)
ax.set_title('(A)  Interaction regime classification', fontsize=10)
ax.legend(fontsize=7.5, loc='upper right')

# annotate selected pairs
highlights = {
    'Bacil.–β-Prot.': ('+1.83', 'Bacil.–β-Prot.\n+1.83'),
    'Actin.–Bacil.':  ('+1.61', 'Actin.–Bacil.\n+1.61'),
    'Actin.–β-Prot.': ('+1.67', 'Actin.–β-Prot.\n+1.67'),
}
for _, row in df.iterrows():
    if row['label'] in highlights:
        ax.annotate(highlights[row['label']][1],
                    xy=(row['F_ij'], abs(row['A_ij'])),
                    xytext=(8, -15), textcoords='offset points',
                    fontsize=6.5, color='#d73027',
                    arrowprops=dict(arrowstyle='-', color='#d73027', lw=0.7))

# ── Panel B: ranked bar of |A_ij| off-diagonal ────────────────────────────────
ax = axes[1]
# top 20 pairs by |A|
top = df.nlargest(20, 'A_ij').copy()
colors_bar = [regime_color.get(r, GREY) for r in top['regime']]
y = np.arange(len(top))
bars = ax.barh(y, top['A_ij'], color=colors_bar, alpha=0.85,
               edgecolor='white', linewidth=0.4)
ax.set_yticks(y)
ax.set_yticklabels(top['label'], fontsize=7.5)
ax.set_xlabel('Fitted $A_{ij}$', fontsize=10)
ax.set_title('(B)  Top interactions ranked by $A_{ij}$', fontsize=10)
ax.axvline(0, color='black', lw=0.6)

# regime colour legend
handles2 = [mpatches.Patch(color=v, label=regime_label[k])
            for k, v in regime_color.items() if k in df['regime'].values]
ax.legend(handles=handles2, fontsize=7, loc='lower right')

fig.tight_layout(w_pad=2.5)
save(fig, 'fig5_A_biological.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — AGORA weight sensitivity (3 panels)
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 6: AGORA weight sweep...')

sweep_configs = [
    ('L1+L2',     'fit_glv_hamilton_kegg_expanded.json',           None,    34),
    ('W=0.5',     'fit_glv_hamilton_kegg_expanded_agora_w0p5.json', 0.5,   35),
    ('W=1.0★',   'fit_glv_hamilton_kegg_expanded_agora_w1p0.json', 1.0,   35),
    ('W=1.5',     'fit_glv_hamilton_kegg_expanded_agora_w1p5.json', 1.5,   35),
    ('W=2.0',     'fit_glv_hamilton_kegg_expanded_agora_w2p0.json', 2.0,   35),
    ('Mac c=0.01','fit_glv_hamilton_kegg_expanded_mac_c0p010.json', 'mac', 70),
    ('Mac c=0.05','fit_glv_hamilton_kegg_expanded_mac_c0p050.json', 'mac', 70),
    ('Mac c=0.10','fit_glv_hamilton_kegg_expanded_mac_c0p100.json', 'mac', 70),
]

sweep = []
for label, fname, w, n_pairs in sweep_configs:
    f = CR / fname
    if not f.exists():
        continue
    d = json.load(open(f))
    sa_str = d.get('sign_agreement', '0/0')
    sa_num = int(sa_str.split('/')[0]) if '/' in str(sa_str) else 0
    sa_den = int(sa_str.split('/')[1]) if '/' in str(sa_str) else n_pairs
    sweep.append({
        'label':  label,
        'rmse':   d['rmse'],
        'r':      d.get('r', 0),
        'sa':     sa_num / sa_den * 100,
        'is_mac': w == 'mac',
        'is_opt': label == 'W=1.0★',
    })

labels_sw = [s['label'] for s in sweep]
rmse_sw   = [s['rmse']  for s in sweep]
r_sw      = [s['r']     for s in sweep]
sa_sw     = [s['sa']    for s in sweep]
colors_sw = ['#969696' if s['is_mac'] else
             (NAVY      if s['is_opt'] else '#6baed6')
             for s in sweep]

x = np.arange(len(sweep))
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

def sweep_bar(ax, values, ylabel, title, panel, fmt='{:.3f}', ylim=None):
    bars = ax.bar(x, values, color=colors_sw, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sw, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'({panel})  {title}', fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    # value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                fmt.format(val), ha='center', va='bottom', fontsize=7.5)
    # highlight optimal
    opt_idx = next(i for i, s in enumerate(sweep) if s['is_opt'])
    axes[0].bar(opt_idx, values[opt_idx],
                color=NAVY, edgecolor='gold', linewidth=2.0) if ax == axes[0] else None

sweep_bar(axes[0], rmse_sw, 'Training RMSE', 'Training RMSE', 'A',
          fmt='{:.4f}', ylim=(0.050, 0.082))
sweep_bar(axes[1], r_sw,    'Pearson $r$',   'Pearson $r$',   'B',
          fmt='{:.3f}', ylim=(0.88, 0.97))
sweep_bar(axes[2], sa_sw,   'Sign agreement (\%)', 'Sign agreement', 'C',
          fmt='{:.0f}\%', ylim=(0, 115))

# MacArthur grey patch legend
handles_sw = [
    mpatches.Patch(color=NAVY,     label='AGORA (optimal W=1.0)'),
    mpatches.Patch(color='#6baed6',label='AGORA (other weights)'),
    mpatches.Patch(color='#969696',label='MacArthur Gaussian'),
]
axes[2].legend(handles=handles_sw, fontsize=8, loc='upper right')

# Phase transition annotation
axes[2].annotate('Phase\ntransition', xy=(2, 100), xytext=(3.2, 108),
                 fontsize=7.5, color='#d73027', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#d73027', lw=1.0))

fig.tight_layout(w_pad=2.0)
save(fig, 'fig6_agora_weight_sweep.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — LOO A-matrix sign consistency heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 7: LOO sign consistency...')

# Load all 10 fold A matrices for AGORA W=1.0
fold_As = []
for i in range(10):
    f = CR / f'loo_expanded_agora_w1p0_fold{i}.json'
    if f.exists():
        d = json.load(open(f))
        fold_As.append(np.array(d['A']))

if fold_As:
    fold_As = np.stack(fold_As)   # (10, 10, 10)
    full_A  = A_agora

    # sign consistency per pair
    sc = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i == j:
                sc[i, j] = np.nan
            else:
                full_sign = np.sign(full_A[i, j]) if abs(full_A[i, j]) > 1e-6 else 0
                fold_signs = np.sign(fold_As[:, i, j])
                if full_sign == 0:
                    sc[i, j] = np.nan
                else:
                    sc[i, j] = np.mean(fold_signs == full_sign)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8),
                              gridspec_kw={'width_ratios': [1, 1.2]})

    # ── Heatmap ───────────────────────────────────────────────────────────────
    ax = axes[0]
    cmap_sc = plt.cm.YlGn
    cmap_sc.set_bad('white')
    im = ax.imshow(sc, cmap=cmap_sc, vmin=0.6, vmax=1.0, aspect='equal')
    ax.set_xticks(range(10)); ax.set_xticklabels(G_SHORT, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(10)); ax.set_yticklabels(G_SHORT, fontsize=8)
    for i in range(10):
        for j in range(10):
            if not np.isnan(sc[i, j]):
                c = 'white' if sc[i, j] > 0.87 else 'black'
                ax.text(j, i, f'{sc[i,j]:.1f}', ha='center', va='center',
                        fontsize=6.5, color=c)
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label('Sign consistency (10 folds)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title('(A)  Sign consistency across LOO folds\n'
                 '(all 45 pairs: SC ≥ 0.70)', fontsize=10)
    ax.tick_params(length=0)

    # ── Box-plot of |A_ij| distributions per regime ────────────────────────
    ax = axes[1]
    # select top 12 pairs by |full_A|, off-diagonal
    pair_vals = []
    for i in range(10):
        for j in range(i+1, 10):
            if abs(full_A[i, j]) > 0.0:
                pair_vals.append((abs(full_A[i,j]), i, j))
    pair_vals.sort(reverse=True)
    top12 = pair_vals[:12]

    box_data  = []
    box_labs  = []
    box_colors= []
    for _, i, j in top12:
        vals = fold_As[:, i, j]
        box_data.append(vals)
        box_labs.append(f'{G_SHORT[i]}\n{G_SHORT[j]}')
        box_colors.append(regime_color.get(regime[i, j], GREY))

    bp = ax.boxplot(box_data, vert=False, patch_artist=True,
                    medianprops=dict(color='black', lw=1.5),
                    whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.set_yticks(range(1, len(box_labs)+1))
    ax.set_yticklabels(box_labs, fontsize=7.5)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('$A_{ij}$ across 10 LOO folds', fontsize=10)
    ax.set_title('(B)  Top-12 pair magnitude distributions\n(coloured by regime)', fontsize=10)

    handles_r = [mpatches.Patch(color=regime_color[k], alpha=0.7, label=regime_label[k])
                 for k in ['aligned', 'muted', 'data_driven'] if k in regime_color]
    ax.legend(handles=handles_r, fontsize=7.5, loc='lower right')

    fig.tight_layout(w_pad=2.5)
    save(fig, 'fig7_loo_stability.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — All models LOO RMSE + BC comparison (corrected: LOO BC, not training)
# ═══════════════════════════════════════════════════════════════════════════════
print('Fig 8: All models RMSE + BC...')

# Load LOO summaries
models_data = []

def load_summary(fname, label, color, is_best=False):
    f = CR / fname
    if not f.exists():
        return
    d = json.load(open(f))
    rmse = d.get('loo_rmse_mean', d.get('mean', None))
    std  = d.get('loo_rmse_std',  d.get('std',  None))
    bc   = d.get('loo_bc_mean',   None)
    models_data.append({'label': label, 'rmse': rmse, 'std': std,
                        'bc': bc, 'color': color, 'best': is_best})

load_summary('loo_expanded_agora_w1p0_summary.json', 'AGORA\nW=1.0★', NAVY, True)
load_summary('loo_expanded_summary.json',             'L1+L2',          '#6baed6')

# add gLV free from loo_cv files
for fname, lbl, col in [
    ('loo_cv_glv_pure.json',       'gLV\nfree',      '#bdbdbd'),
    ('loo_cv_hamilton.json',       'Hamilton\nno prior', '#969696'),
]:
    f = CR / fname
    if f.exists():
        d = json.load(open(f))
        pp = d.get('per_patient', [])
        if isinstance(pp, dict):
            vals = [v for v in pp.values() if v is not None]
        else:
            vals = [r['rmse'] for r in pp if r.get('rmse') is not None]
        if vals:
            models_data.append({'label': lbl,
                                'rmse': np.mean(vals),
                                'std': np.std(vals),
                                'bc': None,
                                'color': col,
                                'best': False})

if not models_data:
    print('  No model data found — skipping Fig 8')
else:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # sort by RMSE
    models_data.sort(key=lambda x: (x['rmse'] or 99))
    labs   = [m['label'] for m in models_data]
    rmses  = [m['rmse']  for m in models_data]
    stds   = [m['std'] or 0 for m in models_data]
    bcs    = [m['bc']   for m in models_data]
    cols   = [m['color'] for m in models_data]
    x = np.arange(len(models_data))

    # RMSE bar
    ax = axes[0]
    bars = ax.bar(x, rmses, color=cols, edgecolor='white', linewidth=0.5,
                  yerr=stds, capsize=3, error_kw={'lw': 1, 'capthick': 1})
    # highlight best
    best_i = next((i for i, m in enumerate(models_data) if m['best']), 0)
    bars[best_i].set_edgecolor('gold')
    bars[best_i].set_linewidth(2.5)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel('LOO-RMSE (mean ± SD)', fontsize=10)
    ax.set_title('(A)  LOO-RMSE by model\n(lower = better)', fontsize=10)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # BC bar (if available)
    bc_models = [m for m in models_data if m['bc'] is not None]
    ax = axes[1]
    if bc_models:
        bc_labs  = [m['label'] for m in bc_models]
        bc_vals  = [m['bc']    for m in bc_models]
        bc_cols  = [m['color'] for m in bc_models]
        x2 = np.arange(len(bc_models))
        bars2 = ax.bar(x2, bc_vals, color=bc_cols, edgecolor='white', linewidth=0.5)
        ax.axhline(0.281, color='#d62728', ls='--', lw=1.2, label='Persistence null (0.281)')
        ax.axhline(0.254, color='#ff7f0e', ls='--', lw=1.2, label='Grand mean null (0.254)')
        ax.set_xticks(x2); ax.set_xticklabels(bc_labs, fontsize=8.5)
        ax.set_ylabel('LOO Bray-Curtis', fontsize=10)
        ax.set_title('(B)  LOO Bray-Curtis by model\n(lower = better)', fontsize=10)
        ax.legend(fontsize=8)
        for bar, val in zip(bars2, bc_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'LOO BC data not available\nin summary files',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_visible(True)

    fig.tight_layout(w_pad=2.5)
    save(fig, 'fig8_all_models_rmse_bc.png')

print('\nAll figures done.')
print(f'Output: {DEST}')
