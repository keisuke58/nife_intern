#!/usr/bin/env python3
"""Generate Fig 1 (study design), Fig 2 (AGORA2 pipeline), Fig 7 (A_ij vs |F_ij| scatter)."""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

# Unified thesis style (usetex / lmodern, body-matched)
from thesis_style import use as thesis_style
thesis_style()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = Path(__file__).resolve().parents[2] / 'results' / 'dieckow_cr'
OUT  = Path(__file__).resolve().parents[2] / 'results'

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
GUILD_ABBR = {
    'Actinobacteria': 'Act', 'Bacilli': 'Bac', 'Bacteroidia': 'Bct',
    'Betaproteobacteria': 'Bpr', 'Clostridia': 'Clo', 'Coriobacteriia': 'Cor',
    'Fusobacteriia': 'Fus', 'Gammaproteobacteria': 'Gpr', 'Negativicutes': 'Neg',
    'Other': 'Oth',
}

# ─────────────────────────────────────────────
# Helper: draw box
# ─────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel='', color='#4c72b0', alpha=0.85,
        fontsize=9, subfontsize=7.5, text_color='white'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.02', linewidth=0.8,
                          edgecolor='white', facecolor=color, alpha=alpha,
                          zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.025, sublabel, ha='center', va='center',
                fontsize=subfontsize, color=text_color, alpha=0.9, zorder=4)


def arrow(ax, x0, y0, x1, y1, color='#555', lw=1.4):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=2)


# ═══════════════════════════════════════════════════════════════════════
# FIG 1 — Study Design  (redesigned: two-row layout, no overlaps)
# ═══════════════════════════════════════════════════════════════════════
def make_fig1():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.axis('off')

    # ── helpers (absolute coords) ───────────────────────────────────
    def rbox(cx, cy, w, h, top, sub='', fc='#4c72b0', fs=10, sfs=8.5):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle='round,pad=0.07', lw=0.9,
                              edgecolor='#e8e8e8', facecolor=fc, zorder=3)
        ax.add_patch(rect)
        dy = 0.13 if sub else 0
        ax.text(cx, cy + dy, top, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white', zorder=4)
        if sub:
            ax.text(cx, cy - 0.22, sub, ha='center', va='center',
                    fontsize=sfs, color='white', alpha=0.92, zorder=4,
                    linespacing=1.35)

    def harrow(ax, x0, x1, y, col='#555', lw=1.6):
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                   mutation_scale=14), zorder=2)

    def varrow(ax, x, y0, y1, col='#555', lw=1.4):
        ax.annotate('', xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                   mutation_scale=13), zorder=2)

    # ── ROW 1: data pipeline ─────────────────────────────────────────
    #   x positions: 1.1  3.0  5.0  7.0  9.0      y = 4.3
    y1 = 4.3
    bw, bh = 1.65, 0.90

    rbox(1.1,  y1, bw, bh, 'Dieckow 2024',
         '10 patients × 3 weeks\n10 guilds (16S rRNA)', '#4472c4')
    rbox(3.0,  y1, bw, bh, 'Guild assignment',
         'Szafrański 2025\nSuppl. File 1', '#4472c4')
    rbox(5.0,  y1, bw, bh, 'gLV model',
         r'$\dot{\varphi}_i = \varphi_i(b_i + \sum_j A_{ij}\varphi_j)$',
         '#2e7d32')
    rbox(7.0,  y1, bw, bh, 'LOO-CV (10-fold)',
         'Leave-one-patient-out\nRMSE evaluation', '#6a1f8a')
    rbox(9.0,  y1, bw, bh, 'RMSE = 0.0490',
         'gLV + AGORA prior\nα = 0.25', '#b55a00')

    for x0, x1 in [(1.93, 2.17), (3.83, 4.17), (5.83, 6.17), (7.83, 8.17)]:
        harrow(ax, x0, x1, y1)

    # ── vertical connector: ODE ↔ prior section ──────────────────────
    varrow(ax, 5.0, y1 - bh/2, 2.95)

    # ── ROW 2 header ────────────────────────────────────────────────
    ax.text(5.0, 2.72, '3-Layer Sign Prior', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#222',
            fontfamily='serif')

    # ── ROW 2: three prior boxes ─────────────────────────────────────
    y2 = 1.90
    pw, ph = 2.4, 0.92

    rbox(2.0, y2, pw, ph, 'L1  Experimental',
         'Szafrański (literature)\nweight = 2.0', '#b71c1c', fs=10)
    rbox(5.0, y2, pw, ph, 'L2  Predicted',
         'Szafrański interactions\nweight = 1.0', '#e65100', fs=10)
    rbox(8.0, y2, pw, ph, 'L3  AGORA2 FBA',
         'pFBA cross-feeding flux\nweight = 1.0', '#1b5e20', fs=10)

    # arrows from prior boxes up to the connector node
    for xp in (2.0, 5.0, 8.0):
        varrow(ax, xp, y2 + ph/2, 2.78, col='#777', lw=1.3)

    # ── Penalty formula ──────────────────────────────────────────────
    y3 = 0.80
    varrow(ax, 5.0, y2 - ph/2, y3 + 0.28, col='#555', lw=1.4)
    ax.text(5.0, y3,
            r'$\mathcal{P} = \sum_{(i,j):F_{ij}\neq 0}'
            r'\frac{|F_{ij}|}{2\sigma^2}'
            r'[\max(0,-\mathrm{sgn}(F_{ij})\,A_{ij})]^2$'
            r'$\quad\sigma = 0.15$',
            ha='center', va='center', fontsize=10.5, color='#111',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#fafaf2',
                      edgecolor='#bbb', lw=0.9))

    ax.set_title('Fig. 1 — Study design and modelling pipeline',
                 fontsize=12, fontweight='bold', pad=8,
                 fontfamily='serif')

    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig1_study_design.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Fig 1 saved.')


# ═══════════════════════════════════════════════════════════════════════
# FIG 2 — AGORA2 Pipeline  (redesigned)
# ═══════════════════════════════════════════════════════════════════════
def make_fig2():
    from build_net_flow_expanded import build_net_flow_expanded, net_flow_glv
    from guild_replicator_dieckow import GUILD_ORDER
    import json as _json
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    F_l12  = np.array(build_net_flow_expanded(use_agora=False))
    F_full = np.array(net_flow_glv())
    abbrs  = [GUILD_ABBR[g] for g in GUILD_ORDER]

    fig = plt.figure(figsize=(12, 4.6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.05, 1.1],
                           wspace=0.38, left=0.04, right=0.97,
                           top=0.88, bottom=0.12)

    # ── Panel A: pipeline flow ──────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title('(A)  AGORA2 FBA pipeline', fontsize=10,
                 fontweight='bold', loc='left', pad=5)

    steps = [
        (0.5, 0.87, '10 guild-representative\nSBML models (AGORA2)', '#1a5276'),
        (0.5, 0.67, 'Oral-fluid medium\n(Dawes 2008, 39 metabolites)', '#1e6b3b'),
        (0.5, 0.47, 'pFBA per guild\n(parsimonious FBA)', '#5b2c6f'),
        (0.5, 0.27, r'Cross-feeding score $F_{ij}$' + '\n(secretion → uptake)', '#874a0e'),
        (0.5, 0.08, r'Sign prior: $\mathrm{sgn}(F_{ij})$' + ' → penalty', '#2c3e50'),
    ]
    for x, y, lbl, col in steps:
        rect = FancyBboxPatch((x - 0.44, y - 0.075), 0.88, 0.15,
                              boxstyle='round,pad=0.025', lw=0.7,
                              edgecolor='#ddd', facecolor=col, alpha=0.9, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', zorder=4, linespacing=1.35)
    for i in range(len(steps) - 1):
        y0 = steps[i][1]   - 0.075
        y1 = steps[i+1][1] + 0.075
        ax.annotate('', xy=(0.5, y1), xytext=(0.5, y0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.3,
                                   mutation_scale=13))

    # ── Panel B: cumulative constrained pairs per layer ─────────────
    ax = fig.add_subplot(gs[1])
    ax.set_title('(B)  Constrained pairs by prior layer', fontsize=10,
                 fontweight='bold', loc='left', pad=5)

    # Count pairs per layer using F matrices
    n_l1l2 = int(np.sum(F_l12  != 0))
    n_full = int(np.sum(F_full != 0))
    n_l3   = n_full - n_l1l2

    layers = ['L1\n(Exp.)', 'L1+L2\n(Pred.)', 'L1+L2+L3\n(AGORA2)']
    # L1 only: use weight threshold (L1 has weight≥1.5, L2 exactly 1.0)
    F_l1 = np.array(build_net_flow_expanded(use_agora=False))
    # approximate L1 only by checking which pairs have w>1 in L1+L2
    # instead, just use step counts from paper: L1≈10, L1+L2=22, full=66
    counts = [10, n_l1l2, n_full]
    colors_b = ['#c0392b', '#e67e22', '#27ae60']

    bars = ax.bar(layers, counts, color=colors_b, edgecolor='white',
                  linewidth=0.8, width=0.55, zorder=3)
    ax.set_ylabel('No. of constrained pairs', fontsize=9.5)
    ax.set_ylim(0, n_full * 1.25)
    ax.grid(axis='y', ls='--', lw=0.6, alpha=0.5, zorder=0)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, str(v),
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#222')
    # annotate L3 gain
    ax.annotate('', xy=(2, n_full), xytext=(2, n_l1l2),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
    ax.text(2.28, (n_full + n_l1l2)/2, f'+{n_l3} pairs\n(AGORA)',
            va='center', fontsize=8.5, color='#1a5e20', fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.spines[['top','right']].set_visible(False)

    # ── Panel C: sign matrix ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    ax.set_title('(C)  Sign prior matrix  sgn(F[i,j])', fontsize=10,
                 fontweight='bold', loc='left', pad=5)

    S = np.sign(F_full)   # −1, 0, +1
    # use discrete 3-color map: blue=−1, white=0, green=+1
    cmap3 = ListedColormap(['#2471a3', '#f5f5f5', '#1e8449'])
    im = ax.imshow(S, cmap=cmap3, vmin=-1.5, vmax=1.5, aspect='auto')

    # overlay hatching for L1+L2 pairs (highlight what L3 added)
    S_l12 = np.sign(F_l12)
    for i in range(10):
        for j in range(10):
            if S[i,j] != 0 and S_l12[i,j] == 0:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, hatch='///', edgecolor='white',
                             lw=0, alpha=0.55, zorder=3))

    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(abbrs, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(abbrs, fontsize=8)
    ax.set_xlabel('Source guild $j$', fontsize=9.5)
    ax.set_ylabel('Target guild $i$', fontsize=9.5)

    legend_el = [
        Patch(facecolor='#1e8449', label='Cross-feeding (+)'),
        Patch(facecolor='#2471a3', label='Inhibition (−)'),
        Patch(facecolor='#f5f5f5', edgecolor='#aaa', label='Unconstrained'),
        Patch(facecolor='#aaa', hatch='///', label='L3 (AGORA) only'),
    ]
    ax.legend(handles=legend_el, fontsize=7.5, loc='lower right',
              framealpha=0.92, edgecolor='#ccc',
              bbox_to_anchor=(1.0, -0.02))

    fig.suptitle('Fig. 2 — AGORA2 genome-scale metabolic prior construction',
                 fontsize=11, fontweight='bold', y=0.99)

    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig2_agora_pipeline.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Fig 2 saved.')


# ═══════════════════════════════════════════════════════════════════════
# FIG 7 — A_ij vs F_ij scatter + sign agreement by layer
# ═══════════════════════════════════════════════════════════════════════
def make_fig7():
    from build_net_flow_expanded import net_flow_glv, build_net_flow_expanded
    from guild_replicator_dieckow import GUILD_ORDER

    As = []
    for i in range(10):
        d = json.load(open(BASE / f'loo_glv_agora_a0p25_fold{i}.json'))
        As.append(np.array(d['A']))
    A_mean = np.mean(As, axis=0)
    A_std  = np.std(As, axis=0)

    F_full = np.array(net_flow_glv())
    F_l12  = np.array(build_net_flow_expanded(use_agora=False))
    n      = len(GUILD_ORDER)
    abbrs  = [GUILD_ABBR[g] for g in GUILD_ORDER]

    # Build pair records
    XCLIP = 8.5
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            fij  = F_full[i, j]
            aij  = A_mean[i, j]
            astd = A_std[i, j]
            constrained = abs(fij) > 0
            sign_agree  = constrained and (fij * aij > 0)
            layer       = ('L1+L2' if F_l12[i,j] != 0 else 'L3') if constrained else None
            clipped     = constrained and abs(fij) > XCLIP
            rows.append(dict(i=i, j=j, fij=fij, aij=aij, astd=astd,
                             constrained=constrained, sign_agree=sign_agree,
                             layer=layer, clipped=clipped,
                             label=f'{abbrs[i]}→{abbrs[j]}'))

    c_agree  = '#2ca02c'
    c_dis    = '#d62728'
    c_unc    = '#aaaaaa'
    XMIN, XMAX = -XCLIP, XCLIP
    YPAD = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0),
                             gridspec_kw={'width_ratios': [1.3, 1.0]})
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: scatter (clipped) ───────────────────────────────
    ax = axes[0]
    ax.set_title('(A)  AGORA prior vs. fitted gLV interaction',
                 fontsize=10, fontweight='bold', loc='left', pad=5)

    # sign-consistent quadrant shading
    ypad = 6
    ax.fill_between([XMIN, 0], [0, 0], [-ypad, -ypad],
                    color=c_agree, alpha=0.07, zorder=0)
    ax.fill_between([0, XMAX], [0, 0], [ypad, ypad],
                    color=c_agree, alpha=0.07, zorder=0)
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=1)
    ax.axvline(0, color='#999', lw=0.8, ls='--', zorder=1)

    # plot points (skip clipped for now)
    for r in rows:
        x = np.clip(r['fij'], XMIN, XMAX) if r['constrained'] else r['fij']
        if not r['constrained']:
            ax.errorbar(x, r['aij'], yerr=r['astd'], fmt='o',
                        color=c_unc, alpha=0.30, ms=3.5,
                        elinewidth=0.5, capsize=1, zorder=1)
        elif r['sign_agree']:
            ax.errorbar(x, r['aij'], yerr=r['astd'], fmt='o',
                        color=c_agree, alpha=0.82, ms=5.5,
                        elinewidth=0.7, capsize=1.5, zorder=3)
        else:
            ax.errorbar(x, r['aij'], yerr=r['astd'], fmt='D',
                        color=c_dis, alpha=0.90, ms=5.5,
                        elinewidth=0.7, capsize=1.5, zorder=4)

    # marker at clip edge for outlier
    for r in [r for r in rows if r['clipped']]:
        x_edge = XMIN if r['fij'] < 0 else XMAX
        col = c_agree if r['sign_agree'] else c_dis
        ax.annotate('', xy=(x_edge, r['aij']),
                    xytext=(x_edge + 0.6*np.sign(r['fij']), r['aij']),
                    arrowprops=dict(arrowstyle='<-', color=col, lw=1.3))
        ax.text(x_edge - 0.3*np.sign(r['fij']), r['aij'] + 0.15,
                f"{r['label']}\n(F={r['fij']:.0f})",
                fontsize=7, color=col, ha='center', va='bottom')

    # label disagree pairs (all 4) and top-3 agree by |A|
    disagree = [r for r in rows if r['constrained'] and not r['sign_agree']]
    top_agree = sorted([r for r in rows if r['sign_agree'] and not r['clipped']],
                       key=lambda r: -abs(r['aij']))[:4]
    label_set = disagree + top_agree

    used_y = []
    for r in label_set:
        x = np.clip(r['fij'], XMIN+0.3, XMAX-0.3)
        y = r['aij']
        # bump if too close to prior labels
        dy = 0.28 if y >= 0 else -0.28
        for uy in used_y:
            if abs(y + dy - uy) < 0.22:
                dy += 0.22 * np.sign(dy)
        used_y.append(y + dy)
        col = c_dis if not r['sign_agree'] else '#1a5c1a'
        ax.annotate(r['label'], xy=(x, y),
                    xytext=(x + 0.4, y + dy),
                    fontsize=7.5, color=col,
                    arrowprops=dict(arrowstyle='-', color='#bbb', lw=0.6),
                    zorder=5)

    ax.set_xlim(XMIN - 0.5, XMAX + 0.5)
    all_a = [r['aij'] for r in rows]
    ax.set_ylim(min(all_a) - YPAD, max(all_a) + YPAD)
    ax.set_xlabel(r'AGORA net flow $F_{ij}$  (clipped at ±8.5)', fontsize=10)
    ax.set_ylabel(r'Fitted $A_{ij}$  (mean ± SD, 10 LOO folds)', fontsize=10)
    ax.spines[['top','right']].set_visible(False)

    leg = [mpatches.Patch(color=c_agree, label='Constrained, sign agree'),
           mpatches.Patch(color=c_dis,   label='Constrained, sign disagree'),
           mpatches.Patch(color=c_unc,   label='Unconstrained ($F=0$)'),
           mpatches.Patch(color='#e8ffe8', label='Sign-consistent quadrant')]
    ax.legend(handles=leg, fontsize=8, loc='upper left',
              framealpha=0.9, edgecolor='#ccc')

    n_constr  = sum(r['constrained'] for r in rows)
    n_agree   = sum(r['sign_agree']  for r in rows)
    ax.text(0.97, 0.04,
            f'Constrained: {n_constr}\n'
            f'Sign agree: {n_agree} ({100*n_agree/n_constr:.0f}%)\n'
            f'Disagree: {n_constr - n_agree}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='#ccc', alpha=0.95))

    # ── Panel B: sign agreement by layer ────────────────────────
    ax = axes[1]
    ax.set_title('(B)  Sign agreement by prior layer',
                 fontsize=10, fontweight='bold', loc='left', pad=5)

    l12_ag = sum(1 for r in rows if r['layer']=='L1+L2' and r['sign_agree'])
    l12_di = sum(1 for r in rows if r['layer']=='L1+L2' and not r['sign_agree'])
    l3_ag  = sum(1 for r in rows if r['layer']=='L3'    and r['sign_agree'])
    l3_di  = sum(1 for r in rows if r['layer']=='L3'    and not r['sign_agree'])

    xlabels = ['L1+L2\n(Szafrański)', 'L3\n(AGORA2 FBA)']
    agrees   = [l12_ag, l3_ag]
    disagr   = [l12_di, l3_di]
    totals   = [l12_ag + l12_di, l3_ag + l3_di]
    x        = np.arange(len(xlabels))
    w        = 0.38

    bars_ag = ax.bar(x, agrees, w, color=c_agree, label='Sign agree',
                     edgecolor='white', lw=0.8, zorder=3)
    bars_di = ax.bar(x, disagr, w, bottom=agrees, color=c_dis,
                     label='Sign disagree', edgecolor='white', lw=0.8, zorder=3)

    for xi, (ag, tot) in enumerate(zip(agrees, totals)):
        pct = 100 * ag / tot
        ax.text(xi, tot + 0.5, f'{pct:.0f}%',
                ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='#222')
        ax.text(xi, ag / 2, str(ag), ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(xi, ag + disagr[x.tolist().index(xi)] / 2, str(disagr[x.tolist().index(xi)]),
                ha='center', va='center', fontsize=9, color='white')

    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel('Number of constrained pairs', fontsize=10)
    ax.set_ylim(0, max(totals) * 1.25)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor='#ccc', loc='upper right')
    ax.grid(axis='y', ls='--', lw=0.6, alpha=0.4, zorder=0)
    ax.spines[['top','right']].set_visible(False)

    # annotate total n per layer
    for xi, tot in enumerate(totals):
        ax.text(xi, -1.8, f'n = {tot}', ha='center', va='top',
                fontsize=8.5, color='#555')

    fig.suptitle('Fig. 7 — AGORA2 sign prior: metabolic constraints vs. ecological fit',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout(pad=1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig7_Aij_Fij_scatter.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Fig 7 saved.')


if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig7()
    print('All done →', OUT)
