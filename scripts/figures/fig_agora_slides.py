# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_agora_slides.py — Slide-optimised versions of AGORA figures.

Generates higher-fontsize, slide-aspect-ratio versions of:
  fig2_agora_pipeline_slide.png   — pipeline flowchart (Slide 8)
  fig_agora_weight_sensitivity_slide.png — W-sweep RMSE (Slide 15)

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_agora_slides.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap

ROOT   = pathlib.Path(__file__).parents[2]
OUT    = ROOT / 'results'
CR_DIR = ROOT / 'results' / 'dieckow_cr'

SLIDE_DPI = 300

# ──────────────────────────────────────────────────────────────────────────────
# Shared style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})


# ──────────────────────────────────────────────────────────────────────────────
# Fig A: pipeline flow (Slide 8)
# ──────────────────────────────────────────────────────────────────────────────
def make_pipeline_slide():
    from build_net_flow_expanded import build_net_flow_expanded, net_flow_glv
    from guild_replicator_dieckow import GUILD_ORDER

    GUILD_ABBR = {
        'Actinobacteria': 'Act', 'Bacilli': 'Bct',
        'Bacteroidia': 'Bac', 'Betaproteobacteria': 'Bpr',
        'Clostridia': 'Clo', 'Coriobacteriia': 'Cor',
        'Fusobacteriia': 'Fus', 'Gammaproteobacteria': 'Gpr',
        'Negativicutes': 'Neg', 'Other': 'Oth',
    }

    F_l12  = np.array(build_net_flow_expanded(use_agora=False))
    F_full = np.array(net_flow_glv())
    abbrs  = [GUILD_ABBR.get(g, g[:3]) for g in GUILD_ORDER]

    # Slide: wider panels, bigger fonts
    fig = plt.figure(figsize=(14, 5.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.0, 1.2],
                           wspace=0.38, left=0.03, right=0.97,
                           top=0.93, bottom=0.10)

    # ── Panel A: pipeline boxes ─────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title('(A)  AGORA2 FBA pipeline', fontsize=13,
                 fontweight='bold', loc='left', pad=6)

    steps = [
        (0.5, 0.87, '10 guild-representative\nSBML models (AGORA2)', '#1a5276'),
        (0.5, 0.67, 'Oral-fluid medium\n(Dawes 2008, 39 metabolites)', '#1e6b3b'),
        (0.5, 0.47, 'pFBA per guild\n(parsimonious FBA)', '#5b2c6f'),
        (0.5, 0.27, r'Cross-feeding score $F_{ij}$' + '\n(secretion → uptake)', '#874a0e'),
        (0.5, 0.08, r'Sign prior: $\mathrm{sgn}(F_{ij})$' + ' → penalty', '#2c3e50'),
    ]
    for x, y, lbl, col in steps:
        rect = FancyBboxPatch((x - 0.44, y - 0.085), 0.88, 0.17,
                              boxstyle='round,pad=0.03', lw=0.8,
                              edgecolor='#ddd', facecolor=col, alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=10,
                color='white', fontweight='bold', zorder=4, linespacing=1.3)
    for i in range(len(steps) - 1):
        y0 = steps[i][1]   - 0.085
        y1 = steps[i+1][1] + 0.085
        ax.annotate('', xy=(0.5, y1), xytext=(0.5, y0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.4,
                                   mutation_scale=14))

    # ── Panel B: bar chart ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    ax.set_title('(B)  Constrained pairs by prior layer', fontsize=13,
                 fontweight='bold', loc='left', pad=6)

    n_l1l2 = int(np.sum(F_l12 != 0))
    n_full = int(np.sum(F_full != 0))
    n_l3   = n_full - n_l1l2
    counts = [10, n_l1l2, n_full]
    layers = ['L1\n(Exp.)', 'L1+L2\n(Pred.)', 'L1+L2+L3\n(AGORA2)']
    colors_b = ['#c0392b', '#e67e22', '#27ae60']

    bars = ax.bar(layers, counts, color=colors_b, edgecolor='white',
                  linewidth=0.8, width=0.55, zorder=3)
    ax.set_ylabel('No. of constrained pairs', fontsize=12)
    ax.set_ylim(0, n_full * 1.28)
    ax.grid(axis='y', ls='--', lw=0.6, alpha=0.4, zorder=0)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, str(v),
                ha='center', va='bottom', fontsize=13, fontweight='bold', color='#222')
    ax.annotate('', xy=(2, n_full), xytext=(2, n_l1l2),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
    ax.text(2.3, (n_full + n_l1l2)/2, f'+{n_l3} pairs\n(AGORA)',
            va='center', fontsize=10, color='#1a5e20', fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel C: sign matrix ───────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    ax.set_title('(C)  Sign prior matrix  sgn(F[i,j])', fontsize=13,
                 fontweight='bold', loc='left', pad=6)

    S     = np.sign(F_full)
    S_l12 = np.sign(F_l12)
    cmap3 = ListedColormap(['#2471a3', '#f5f5f5', '#1e8449'])
    ax.imshow(S, cmap=cmap3, vmin=-1.5, vmax=1.5, aspect='auto')

    for i in range(10):
        for j in range(10):
            if S[i, j] != 0 and S_l12[i, j] == 0:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, hatch='///', edgecolor='white',
                             lw=0, alpha=0.55, zorder=3))

    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(abbrs, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(abbrs, fontsize=10)
    ax.set_xlabel('Source guild $j$', fontsize=12)
    ax.set_ylabel('Target guild $i$', fontsize=12)

    legend_el = [
        mpatches.Patch(facecolor='#1e8449', label='Cross-feeding (+)'),
        mpatches.Patch(facecolor='#2471a3', label='Inhibition (−)'),
        mpatches.Patch(facecolor='#f5f5f5', edgecolor='#aaa', label='Unconstrained'),
        mpatches.Patch(facecolor='#aaa', hatch='///', label='L3 (AGORA) only'),
    ]
    ax.legend(handles=legend_el, fontsize=10, loc='lower right',
              framealpha=0.92, edgecolor='#ccc', bbox_to_anchor=(1.02, -0.04))

    out = OUT / 'fig2_agora_pipeline_slide.png'
    fig.savefig(out, dpi=SLIDE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'saved → {out}')


# ──────────────────────────────────────────────────────────────────────────────
# Fig B: W-sweep RMSE (Slide 15) — two-panel, clean
# ──────────────────────────────────────────────────────────────────────────────
def make_weight_sweep_slide():
    """Recreate the W-sweep figure from fit_glv_hamilton_kegg_expanded_agora_w*.json."""
    WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0]
    GREEN   = '#27ae60'
    BLUE    = '#2980b9'
    GREY    = '#7f8c8d'
    RED     = '#d62728'

    def fit_file(w):
        if w == 0.0:
            return CR_DIR / 'fit_glv_hamilton_kegg_expanded.json'
        tag = str(w).replace('.', 'p')
        return CR_DIR / f'fit_glv_hamilton_kegg_expanded_agora_w{tag}.json'

    def loo_rmse(pattern):
        files = sorted(CR_DIR.glob(f'{pattern}_fold*.json'))
        return np.mean([json.loads(f.read_text())['rmse'] for f in files]) if files else float('nan')

    def loo_bc(pattern):
        files = sorted(CR_DIR.glob(f'{pattern}_fold*.json'))
        if not files:
            return float('nan')
        vals = [json.loads(f.read_text()).get('bray_curtis', float('nan')) for f in files]
        return np.nanmean(vals)

    # Training RMSE and SA vs W
    rmse_train, sa_vals = [], []
    for w in WEIGHTS:
        f = fit_file(w)
        if f.exists():
            d = json.loads(f.read_text())
            rmse_train.append(d['rmse'])
            sa_str = d.get('sign_agreement', '0/1')
            num, den = map(int, sa_str.split('/'))
            sa_vals.append(num / den * 100 if den else float('nan'))
        else:
            rmse_train.append(float('nan'))
            sa_vals.append(float('nan'))

    # LOO reference lines
    loo_free  = loo_rmse('loo_glv_a0p0')
    loo_l12   = loo_rmse('loo_expanded_a0p0')
    loo_agora = loo_rmse('loo_glv_agora_a0p0')

    # per-patient at W=1.0
    fold_rmses = {}
    for fold in range(10):
        f = CR_DIR / f'loo_glv_agora_a0p0_fold{fold}.json'
        if f.exists():
            fold_rmses[fold] = json.loads(f.read_text())['rmse']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(wspace=0.40, left=0.07, right=0.97, top=0.90, bottom=0.13)

    # ── Left: SA + RMSE vs W ───────────────────────────────────────
    ax = axes[0]
    W_arr = np.array(WEIGHTS)
    sa_arr = np.array(sa_vals)
    rm_arr = np.array(rmse_train)

    ax.plot(W_arr, sa_arr, 'o-', color=BLUE, lw=2.0, ms=7, label='Sign agreement (SA %)')
    ax2l = ax.twinx()
    ax2l.plot(W_arr, rm_arr, 's--', color=GREEN, lw=1.8, ms=7, label='Training RMSE')

    # LOO baselines on right axis
    if not np.isnan(loo_free):
        ax2l.axhline(loo_free,  ls=':',  color=GREY,     lw=1.3, label=f'LOO free  ({loo_free:.4f})')
    if not np.isnan(loo_agora):
        ax2l.axhline(loo_agora, ls='-.', color='#1f77b4', lw=1.3, label=f'LOO AGORA W=1.0  ({loo_agora:.4f})')

    # Mark W=1.0 phase transition
    if 1.0 in WEIGHTS:
        idx10 = WEIGHTS.index(1.0)
        ax.plot(1.0, sa_arr[idx10], 'o', color=RED, ms=12, zorder=5)
        ax.axvline(1.0, color=RED, lw=0.9, ls='--', alpha=0.45)

    ax.set_xlabel('AGORA weight $W$', fontsize=13)
    ax.set_ylabel('Sign agreement SA (%)', fontsize=12, color=BLUE)
    ax2l.set_ylabel('RMSE', fontsize=12, color=GREEN)
    ax.set_xticks(WEIGHTS)
    ax.tick_params(labelsize=11)
    ax2l.tick_params(labelsize=11)
    ax.set_title('(A)  W sweep: phase transition at $W=1.0$', fontsize=13,
                 fontweight='bold', loc='left', pad=5)
    # combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2l.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, framealpha=0.9,
              loc='lower right')
    ax.spines[['top']].set_visible(False)

    # ── Right: per-patient RMSE at W=1.0 (LOO) ───────────────────
    ax3 = axes[1]
    if fold_rmses:
        folds  = sorted(fold_rmses)
        vals   = [fold_rmses[f] for f in folds]
        mean_v = np.mean(vals)
        colors = [GREEN if v <= mean_v else RED for v in vals]
        labels = [chr(65 + f) for f in folds]

        ax3.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.6, alpha=0.88)
        ax3.axhline(mean_v, ls='--', color=GREY, lw=1.3,
                    label=f'Mean ({mean_v:.4f})')
        ax3.set_xlabel('Patient', fontsize=13)
        ax3.set_ylabel('LOO-RMSE', fontsize=12)
        ax3.set_title('(B)  Per-patient LOO-RMSE  (W = 1.0)', fontsize=13,
                      fontweight='bold', loc='left', pad=5)
        ax3.legend(fontsize=11, framealpha=0.9)
        ax3.tick_params(labelsize=11)
        ax3.spines[['top', 'right']].set_visible(False)
    else:
        ax3.text(0.5, 0.5, 'No LOO fold data', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, color=GREY)

    out = OUT / 'fig_agora_weight_sensitivity_slide.png'
    fig.savefig(out, dpi=SLIDE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'saved → {out}')


if __name__ == '__main__':
    print('Generating slide-optimised AGORA figures...')
    make_pipeline_slide()
    make_weight_sweep_slide()
    print('Done.')
