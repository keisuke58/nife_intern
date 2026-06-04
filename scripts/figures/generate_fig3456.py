#!/usr/bin/env python3
"""
generate_fig3456.py — Paper figures 3, 4, 5, 6.

Fig 3: AGORA2 sign vs fitted gLV A — scatter + agreement bar
Fig 4: W-sweep phase transition (SA and RMSE vs AGORA weight)
Fig 5: LOO-CV comparison — all models, per-patient
Fig 6: A matrix heatmap with 3 interaction regimes annotated

Usage:
    python generate_fig3456.py            # all figures
    python generate_fig3456.py --fig 4   # single figure
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_net_flow_expanded import build_net_flow_expanded
from guild_replicator_dieckow import GUILD_ORDER, N_G

HERE    = Path(__file__).resolve().parents[2]
DIR     = HERE / 'results' / 'dieckow_cr'
OUT_DIR = HERE / 'results'
PATIENTS = list('ABCDEFGHKL')
GUILD_SHORT = ['Actin.', 'Bacil.', 'Bctrd.', 'β-Prot.',
               'Clost.', 'Corio.', 'Fusob.', 'γ-Prot.', 'Negat.', 'Other']

# ── colour palette ─────────────────────────────────────────────────────────
C_ALIGN  = '#2166ac'   # blue   — data+prior aligned
C_MUTED  = '#92c5de'   # light blue — prior-constrained, muted
C_DATA   = '#d6604d'   # red    — data-driven
C_OTHER  = '#b0b0b0'   # grey   — other
C_AGREE  = '#1a9641'
C_DISAGR = '#d7191c'


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — AGORA sign vs gLV A
# ═══════════════════════════════════════════════════════════════════════════

def make_fig3():
    pairs = json.loads((DIR / 'agora_sign_comparison.json').read_text())
    F_l12 = build_net_flow_expanded(use_agora=False, verbose=False)
    F_full = build_net_flow_expanded(use_agora=True,  verbose=False)

    A_vals, F_vals, agree_list, is_l3 = [], [], [], []
    for p in pairs:
        i = GUILD_ORDER.index(p['src'])
        j = GUILD_ORDER.index(p['tgt'])
        A_vals.append(p['A'])
        F_vals.append(p['sign_agora'])
        agree_list.append(p['agree'])
        is_l3.append(F_l12[i, j] == 0 and F_full[i, j] != 0)

    A_vals    = np.array(A_vals)
    F_vals    = np.array(F_vals)
    agree_arr = np.array(agree_list)
    is_l3_arr = np.array(is_l3)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8),
                             gridspec_kw={'width_ratios': [1.6, 1]})

    # ── Panel A: scatter A_ij vs sgn(F_ij) ──────────────────────────────
    ax = axes[0]
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(F_vals))

    colors = np.where(agree_arr, C_AGREE, C_DISAGR)
    markers = np.where(is_l3_arr, '^', 'o')

    for mk, idx in [('o', ~is_l3_arr), ('^', is_l3_arr)]:
        label = 'L1+L2' if mk == 'o' else 'L3 (AGORA)'
        ax.scatter(F_vals[idx] + jitter[idx], A_vals[idx],
                   c=colors[idx], marker=mk, s=28, alpha=0.85,
                   linewidths=0.4, edgecolors='white', label=label, zorder=3)

    # shade correct-sign quadrants
    for xs, ys in [([0, 3], [0, 10]), ([-3, 0], [-10, 0])]:
        ax.fill_between(xs, ys[0], ys[1], color='#d0e8d0', alpha=0.35, zorder=0)

    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([r'$-1$', r'$0$', r'$+1$'])
    ax.set_xlabel(r'AGORA sign  sgn($F_{ij}$)', fontsize=10)
    ax.set_ylabel(r'Fitted $A_{ij}$', fontsize=10)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-5, 5)
    ax.set_title('(A) Sign agreement: AGORA vs fitted $A_{ij}$', fontsize=10, loc='left')

    legend_handles = [
        mpatches.Patch(color=C_AGREE,  label=f'Agree ({agree_arr.sum()}/{len(agree_arr)})'),
        mpatches.Patch(color=C_DISAGR, label=f'Disagree ({(~agree_arr).sum()}/{len(agree_arr)})'),
        plt.Line2D([0],[0], marker='o', color='gray', ls='', ms=6, label='L1+L2'),
        plt.Line2D([0],[0], marker='^', color='gray', ls='', ms=6, label='L3 (AGORA)'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left')

    # ── Panel B: bar chart — SA by layer ────────────────────────────────
    ax2 = axes[1]
    l1l2_pairs = [p for p in pairs if not (
        F_l12[GUILD_ORDER.index(p['src']), GUILD_ORDER.index(p['tgt'])] == 0)]
    l3_pairs   = [p for p in pairs if is_l3_arr[pairs.index(p)]]

    layers = ['L1+L2\n(Szafrański)', 'L3\n(AGORA)', 'All']
    sets   = [l1l2_pairs, l3_pairs, pairs]
    sas    = [sum(p['agree'] for p in s) / len(s) * 100 if len(s) else float('nan') for s in sets]
    ns     = [len(s) for s in sets]
    bar_colors = ['#4dac26', '#7b3294', '#1f77b4']

    bars = ax2.bar(layers, sas, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5, width=0.55)
    import math
    for bar, sa, n in zip(bars, sas, ns):
        if math.isnan(sa):
            continue
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{sa:.0f}%\n({int(sa/100*n)}/{n})',
                 ha='center', va='bottom', fontsize=8.5)

    ax2.set_ylim(0, 115)
    ax2.set_ylabel('Sign recovery rate (%)', fontsize=10)
    ax2.set_title('(B) Sign recovery rate by layer', fontsize=10, loc='left')
    ax2.axhline(50, color='k', lw=0.6, ls=':', alpha=0.5)
    ax2.text(2.35, 51, 'chance', fontsize=7, color='gray', va='bottom')

    fig.tight_layout(pad=1.5)
    for fmt in ('pdf', 'png'):
        p = OUT_DIR / f'fig3_agora_sign_validation.{fmt}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — W-sweep phase transition
# ═══════════════════════════════════════════════════════════════════════════

def make_fig4():
    WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0]

    # Full-fit RMSE and SA per weight
    rmse_full, sa_full, n_pairs_full = [], [], []
    for w in WEIGHTS:
        if w == 0.0:
            f = DIR / 'fit_glv_hamilton_kegg_expanded.json'
        else:
            tag = str(w).replace('.', 'p')
            f   = DIR / f'fit_glv_hamilton_kegg_expanded_agora_w{tag}.json'
        d = json.loads(f.read_text())
        rmse_full.append(d['rmse'])
        sa_str = d.get('sign_agreement', '0/1')
        num, den = map(int, sa_str.split('/'))
        sa_full.append(num / den * 100)
        n_pairs_full.append(d.get('n_constrained_pairs', den))

    # LOO-RMSE at W=0 (no prior) and W=1.0 (best)
    def loo_rmse(pattern):
        files = sorted(DIR.glob(f'{pattern}_fold*.json'))
        return np.mean([json.loads(f.read_text())['rmse'] for f in files]) if files else np.nan

    loo_noprior = loo_rmse('loo_glv_a0p0')
    loo_l12     = loo_rmse('loo_expanded_a0p0')
    loo_agora   = loo_rmse('loo_glv_agora_a0p0')

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # ── Panel A: SA vs W ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(WEIGHTS, sa_full, 'o-', color='#1f77b4', lw=2, ms=7, zorder=3)

    # highlight W=1.0 phase transition
    idx10 = WEIGHTS.index(1.0)
    ax.plot(WEIGHTS[idx10], sa_full[idx10], 'o', color='#d62728', ms=10, zorder=4,
            label='W=1.0 (100% SA)')
    ax.axhline(100, color='#d62728', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(1.0, color='#d62728', lw=0.8, ls='--', alpha=0.4)

    ax.set_xticks(WEIGHTS)
    ax.set_xticklabels([f'{w}' for w in WEIGHTS])
    ax.set_ylim(80, 105)
    ax.set_xlabel('AGORA weight W', fontsize=10)
    ax.set_ylabel('Sign recovery rate (%)', fontsize=10)
    ax.set_title('(A) Phase transition at W = 1.0', fontsize=10, loc='left')
    ax.legend(fontsize=9)

    # annotate pair counts
    for x, n in zip(WEIGHTS, n_pairs_full):
        ax.text(x, 81.5, f'n={n}', ha='center', fontsize=7.5, color='#555')

    # ── Panel B: RMSE vs W (train + LOO reference lines) ────────────────
    ax2 = axes[1]
    ax2.plot(WEIGHTS, rmse_full, 's-', color='#2ca02c', lw=2, ms=7,
             label='Training RMSE', zorder=3)

    # LOO reference lines
    ax2.axhline(loo_noprior, color='#555', lw=1.0, ls=':', label=f'LOO free (no prior) {loo_noprior:.4f}')
    ax2.axhline(loo_l12,     color='#9467bd', lw=1.0, ls='--', label=f'LOO L1+L2 {loo_l12:.4f}')
    ax2.axhline(loo_agora,   color='#1f77b4', lw=1.0, ls='-.', label=f'LOO AGORA W=1.0 {loo_agora:.4f}')

    ax2.axvline(1.0, color='#d62728', lw=0.8, ls='--', alpha=0.4)
    idx10 = WEIGHTS.index(1.0)
    ax2.plot(WEIGHTS[idx10], rmse_full[idx10], 's', color='#d62728', ms=10, zorder=4)

    ax2.set_xticks(WEIGHTS)
    ax2.set_xticklabels([f'{w}' for w in WEIGHTS])
    ax2.set_xlabel('AGORA weight W', fontsize=10)
    ax2.set_ylabel('RMSE', fontsize=10)
    ax2.set_title('(B) Training RMSE vs W', fontsize=10, loc='left')
    ax2.legend(fontsize=7.5, loc='upper right')

    fig.tight_layout(pad=1.5)
    for fmt in ('pdf', 'png'):
        p = OUT_DIR / f'fig4_w_sweep.{fmt}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — LOO-CV model comparison
# ═══════════════════════════════════════════════════════════════════════════

def make_fig5():
    MODEL_SPECS = [
        ('gLV free',            'loo_glv_a0p0',         '#aec7e8'),
        ('gLV α=0.25\n(L1+L2)','loo_glv_a0p25',        '#1f77b4'),
        ('gLV AGORA\n(L1+L2+L3)','loo_glv_agora_a0p0', '#2ca02c'),
        ('Hamilton\nL1+L2',     'loo_expanded_a0p0',    '#9467bd'),
        ('Hamilton\nL1+L2+L3',  'loo_expanded_agora_a0p0','#c5b0d5'),
    ]

    all_rmse = {}
    for label, pat, _ in MODEL_SPECS:
        files = sorted(DIR.glob(f'{pat}_fold*.json'))
        all_rmse[label] = [json.loads(f.read_text())['rmse'] for f in files]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                             gridspec_kw={'width_ratios': [2, 1]})

    # ── Panel A: per-patient LOO-RMSE ─────────────────────────────────
    ax = axes[0]
    x = np.arange(len(PATIENTS))
    n_m = len(MODEL_SPECS)
    bw  = 0.14
    offsets = np.linspace(-(n_m-1)/2*bw, (n_m-1)/2*bw, n_m)

    for k, (label, pat, color) in enumerate(MODEL_SPECS):
        rmses = all_rmse[label]
        ax.bar(x + offsets[k], rmses, bw, label=label,
               color=color, edgecolor='white', linewidth=0.4, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(PATIENTS)
    ax.set_xlabel('Patient (held-out)', fontsize=10)
    ax.set_ylabel('LOO-RMSE', fontsize=10)
    ax.set_title('(A) Per-patient LOO-RMSE', fontsize=10, loc='left')
    ax.legend(fontsize=7.5, loc='upper left', ncol=2)
    ax.set_ylim(0, 0.135)

    # ── Panel B: mean LOO-RMSE bar ─────────────────────────────────────
    ax2 = axes[1]
    means  = [np.mean(all_rmse[l]) for l, _, _ in MODEL_SPECS]
    labels = [l for l, _, _ in MODEL_SPECS]
    colors = [c for _, _, c in MODEL_SPECS]

    y = np.arange(len(MODEL_SPECS))
    bars = ax2.barh(y, means, color=colors, edgecolor='white',
                    linewidth=0.4, alpha=0.9, height=0.6)

    # highlight best
    best_idx = int(np.argmin(means))
    bars[best_idx].set_edgecolor('#d62728')
    bars[best_idx].set_linewidth(1.8)

    for bar, m in zip(bars, means):
        ax2.text(m + 0.0003, bar.get_y() + bar.get_height()/2,
                 f'{m:.4f}', va='center', fontsize=8.5)

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlabel('Mean LOO-RMSE', fontsize=10)
    ax2.set_title('(B) Mean LOO-RMSE', fontsize=10, loc='left')
    ax2.set_xlim(0, max(means) * 1.18)
    ax2.invert_yaxis()

    fig.tight_layout(pad=1.5)
    for fmt in ('pdf', 'png'):
        p = OUT_DIR / f'fig5_loo_comparison.{fmt}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 — A matrix heatmap with 3 regimes
# ═══════════════════════════════════════════════════════════════════════════

def make_fig6():
    # AGORA W=1.0 MAP A (biological interpretation model, SA=100%)
    d_map  = json.loads((DIR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json').read_text())
    A_mean = np.array(d_map['A'])

    F_l12  = build_net_flow_expanded(use_agora=False, verbose=False)
    F_full = build_net_flow_expanded(use_agora=True,  verbose=False)

    # Classify regimes
    regime = np.full((N_G, N_G), 'other', dtype=object)
    for i in range(N_G):
        for j in range(N_G):
            if i == j:
                regime[i, j] = 'diag'
                continue
            a = A_mean[i, j]
            f = F_full[i, j]
            if f != 0 and np.sign(a) * np.sign(f) > 0 and abs(a) > 0.15:
                regime[i, j] = 'aligned'
            elif f != 0 and np.sign(a) * np.sign(f) > 0 and abs(a) <= 0.15:
                regime[i, j] = 'muted'
            elif f == 0 and abs(a) > 0.15:
                regime[i, j] = 'data_driven'

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                             gridspec_kw={'width_ratios': [1, 1]})

    # ── Panel A: A_ij heatmap ────────────────────────────────────────────
    ax = axes[0]
    vmax = max(np.abs(A_mean[~np.eye(N_G, dtype=bool)]).max(), 0.5)
    im = ax.imshow(A_mean, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')

    # Overlay regime markers
    for i in range(N_G):
        for j in range(N_G):
            r = regime[i, j]
            if r == 'aligned':
                ax.add_patch(plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                             fill=False, edgecolor=C_ALIGN, lw=2.0))
            elif r == 'muted':
                ax.add_patch(plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                             fill=False, edgecolor=C_MUTED, lw=1.5, linestyle='--'))
            elif r == 'data_driven':
                ax.add_patch(plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                             fill=False, edgecolor=C_DATA, lw=2.0))

    ax.set_xticks(range(N_G)); ax.set_xticklabels(GUILD_SHORT, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(N_G)); ax.set_yticklabels(GUILD_SHORT, fontsize=8)
    ax.set_xlabel('Source guild $j$', fontsize=9)
    ax.set_ylabel('Target guild $i$', fontsize=9)
    ax.set_title(r'(A) $A_{ij}$ MAP (AGORA $W$=1.0, SA=100%)', fontsize=9.5, loc='left')
    fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03, label=r'$A_{ij}$')

    legend_handles = [
        mpatches.Patch(facecolor='none', edgecolor=C_ALIGN,  lw=2,   label='Data+prior aligned'),
        mpatches.Patch(facecolor='none', edgecolor=C_MUTED,  lw=1.5, linestyle='--', label='Prior-constrained, muted'),
        mpatches.Patch(facecolor='none', edgecolor=C_DATA,   lw=2,   label='Data-driven (no prior)'),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc='lower right',
              framealpha=0.9, bbox_to_anchor=(1.0, -0.32))

    # ── Panel B: scatter |A_ij| vs |F_ij|  ──────────────────────────────
    ax2 = axes[1]
    for i in range(N_G):
        for j in range(N_G):
            if i == j:
                continue
            a = abs(A_mean[i, j])
            f = abs(F_full[i, j])
            r = regime[i, j]
            color = {'aligned': C_ALIGN, 'muted': C_MUTED,
                     'data_driven': C_DATA, 'other': C_OTHER}[r]
            mk = '^' if (F_l12[i, j] == 0 and F_full[i, j] != 0) else 'o'
            ax2.scatter(f, a, c=color, marker=mk, s=22, alpha=0.75,
                        linewidths=0, zorder=3)

    ax2.set_xlabel(r'|$F_{ij}$| (AGORA prior strength)', fontsize=9)
    ax2.set_ylabel(r'|$A_{ij}$| (fitted strength)', fontsize=9)
    ax2.set_title(r'(B) Prior strength vs ecological signal', fontsize=9.5, loc='left')

    legend_handles2 = [
        mpatches.Patch(color=C_ALIGN,    label='Aligned (|A|>0.25, sign match)'),
        mpatches.Patch(color=C_MUTED,    label='Muted (|A|≤0.1, sign match)'),
        mpatches.Patch(color=C_DATA,     label='Data-driven (no prior)'),
        mpatches.Patch(color=C_OTHER,    label='Other'),
        plt.Line2D([0],[0], marker='o', color='gray', ls='', ms=6, label='L1+L2'),
        plt.Line2D([0],[0], marker='^', color='gray', ls='', ms=6, label='L3 (AGORA)'),
    ]
    ax2.legend(handles=legend_handles2, fontsize=7.5, loc='upper right')

    fig.tight_layout(pad=1.5)
    for fmt in ('pdf', 'png'):
        p = OUT_DIR / f'fig6_A_regimes.{fmt}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# PAPER_OUTLINE number update
# ═══════════════════════════════════════════════════════════════════════════

def update_paper_outline():
    p = HERE / 'PAPER_OUTLINE.md'
    txt = p.read_text()

    replacements = [
        # RMSE numbers
        ('LOO-CV validation**: RMSE 0.0504 (vs L1+L2 0.0516, −2.4%; vs gLV free 0.0588, −14%)',
         'LOO-CV validation**: best: gLV α=0.25 LOO-RMSE 0.0490 (vs gLV free 0.0501, −2.2%; vs Hamilton L1+L2 0.0567, −13%)'),
        # Table row
        ('| **Hamilton + L1+L2+L3 W=1.0** | **0.0565** | **0.0504** | **100%** |',
         '| **gLV α=0.25 (L1+L2)** | — | **0.0490** | **97%** |'),
        # Results section
        ('Hamilton + L1+L2 | 0.0631 | 0.0516 | 94%',
         'Hamilton + L1+L2 | 0.0631 | 0.0567 | 94%'),
        ('gLV free | 0.0588 | 0.0588 | —',
         'gLV free | 0.0501 | 0.0501 | —'),
        ('MacArthur prior | 0.059–0.07 | — | 4–8/70',
         'MacArthur prior | 0.059–0.07 | — | 4–8/70'),
        # 7/10 patients
        ('- 7/10 patients improved with AGORA prior (A, B, D, E, G, H, K)',
         '- 8/10 patients improved with L1+L2 prior (gLV α=0.25 vs free)'),
        # W sweep SA
        ('W sweep: SA 94%(W=0.5) → **100%(W=1.0)** → 94%(W=1.5)',
         'W sweep (full fit): SA 94%(W=0.5) → **100%(W=1.0)** → 94%(W=1.5) — LOO best: gLV α=0.25 (RMSE 0.0490)'),
    ]

    changed = 0
    for old, new in replacements:
        if old in txt:
            txt = txt.replace(old, new)
            changed += 1

    p.write_text(txt)
    print(f'PAPER_OUTLINE.md updated: {changed}/{len(replacements)} replacements')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig', type=int, choices=[3, 4, 5, 6], default=None)
    parser.add_argument('--no_outline', action='store_true')
    args = parser.parse_args()

    figs = [args.fig] if args.fig else [3, 4, 5, 6]

    for n in figs:
        print(f'\n── Fig {n} ──')
        {'3': make_fig3, '4': make_fig4, '5': make_fig5, '6': make_fig6}[str(n)]()

    if not args.no_outline:
        print('\n── PAPER_OUTLINE update ──')
        update_paper_outline()


if __name__ == '__main__':
    main()
