#!/usr/bin/env python3
"""fig_network_thesis.py — thesis_style versions of the network figures.

Reads results/dieckow_cr/network_analysis.json and network_rewiring.json
(produced by analyze_interaction_network.py / analyze_network_rewiring.py)
and produces thesis-quality PDFs:

  results/fig_network_backbone_thesis.{pdf,png}
  results/fig_network_rewiring_thesis.{pdf,png}

Font sizes: 9 pt body (thesis_style), node labels 7 pt, edge labels 6 pt.

Run from repo root:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_network_thesis.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root

import json
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import thesis_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

_ROOT = Path(__file__).resolve().parents[2]
_RES  = _ROOT / 'results' / 'dieckow_cr'
_OUT  = _ROOT / 'results'

NET = json.loads((_RES / 'network_analysis.json').read_text())
REW = json.loads((_RES / 'network_rewiring.json').read_text())

SHORT = {g: GUILD_SHORT.get(g, g) for g in GUILD_ORDER}
FAC  = '#2166ac'   # facilitation blue
COMP = '#b2182b'   # competition red


# ── Fig backbone ───────────────────────────────────────────────────────────────
def fig_backbone():
    figsize = thesis_style.use(width_frac=1.0, aspect=0.45)
    backbone = NET['E_concordant_backbone']
    levels   = {g: NET['B_trophic']['meta_levels'][i] for i, g in enumerate(GUILD_ORDER)}
    nodes    = sorted({e['src'] for e in backbone} | {e['dst'] for e in backbone})

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in backbone:
        G.add_edge(e['src'], e['dst'])
    indeg = {n: G.in_degree(n) for n in nodes}

    order   = sorted(nodes, key=lambda n: (levels[n], n))
    stagger = [0.0, 1.0, -1.0, 0.55, -0.55, 1.0, -1.0, 0.5, -0.5]
    pos     = {n: (i, stagger[i % len(stagger)]) for i, n in enumerate(order)}
    nsize   = {n: 700 + 500 * indeg[n] for n in nodes}

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color='#4a8f5b', width=1.4, alpha=0.85,
        arrows=True, arrowstyle='-|>', arrowsize=12,
        node_size=[nsize[n] for n in G.nodes()],
        connectionstyle='arc3,rad=0.14', min_target_margin=12)
    for n in nodes:
        ring = 2.2 if n == 'Negativicutes' else 1.0
        ax.scatter(*pos[n], s=nsize[n], color=GUILD_COLORS[n],
                   edgecolors=(COMP if n == 'Negativicutes' else 'k'),
                   linewidths=ring, zorder=3)
        ax.annotate(SHORT[n], pos[n], ha='center', va='center', fontsize=7,
                    fontweight='bold', zorder=4,
                    color='white' if n in ('Bacteroidia', 'Negativicutes', 'Other') else 'black')
    if 'Negativicutes' in pos:
        x, y = pos['Negativicutes']
        ax.annotate(r'Veillonella (lactate acceptor, 4 incoming)',
                    (x, y), xytext=(x + 0.15, y - 1.65),
                    ha='center', va='top', fontsize=6, style='italic', color=COMP,
                    arrowprops=dict(arrowstyle='-', color=COMP, lw=0.7))

    ax.set_xlabel('Metabolic trophic level (producer $\\to$ consumer)')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Concordant ecological--metabolic cross-feeding backbone')
    ax.text(0.5, 1.02,
            f'{len(backbone)} edges: Hamilton fitted $\\times$ AGORA metabolic signs agree '
            r'(permutation $p=0.0004$)',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7, color='0.4')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-2.0, 2.4)
    ax.margins(x=0.08)
    fig.tight_layout()
    out = _OUT / 'fig_network_backbone_thesis'
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print('wrote %s (%d edges, nodes=%s)' % (out.with_suffix('.pdf'), len(backbone),
                                              [SHORT[n] for n in nodes]))


# ── helpers for rewiring panel ──────────────────────────────────────────────────
def _state_net(ax, P, sp, title, key_edges):
    """5-species circular layout, key edges coloured by effective sign."""
    circ = nx.circular_layout(nx.path_graph(len(sp)))
    pos  = {i: circ[i] for i in range(len(sp))}
    for (j, i) in key_edges:
        col  = FAC if P[i][j] > 0.5 else COMP
        conf = abs(P[i][j] - 0.5) * 2
        ax.annotate('', xy=pos[i], xytext=pos[j],
                    arrowprops=dict(arrowstyle='-|>', color=col,
                                    lw=0.9 + 1.8 * conf, alpha=0.85,
                                    connectionstyle='arc3,rad=0.15',
                                    shrinkA=12, shrinkB=12))
    for i, s in enumerate(sp):
        ax.scatter(*pos[i], s=520,
                   color='#ea2323' if s == 'Pg' else '#cccccc',
                   edgecolors='k', linewidths=0.8, zorder=3)
        ax.annotate(s, pos[i], ha='center', va='center',
                    fontsize=7, fontweight='bold', zorder=4)
    ax.set_title(title, fontsize=9)
    ax.axis('off'); ax.margins(0.18)


# ── Fig rewiring ───────────────────────────────────────────────────────────────
def fig_rewiring():
    figsize = thesis_style.use(width_frac=1.0, aspect=0.42)
    sp  = REW['species']
    Pc  = REW['P_facilitation_commensal']
    Pd  = REW['P_facilitation_dysbiotic']
    idx = {s: i for i, s in enumerate(sp)}
    rewired   = REW['rewired_edges']
    key_edges = sorted({(idx[e['from']], idx[e['to']]) for e in rewired})

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    _state_net(ax1, Pc, sp, 'Commensal (CS, CH)', key_edges)
    _state_net(ax2, Pd, sp, 'Dysbiotic (DS, DH)',  key_edges)

    handles = [Line2D([0], [0], color=FAC,  lw=2.5, label='facilitation'),
               Line2D([0], [0], color=COMP, lw=2.5, label='competition')]
    ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.16),
               ncol=1, frameon=False, fontsize=7)
    fig.text(0.34, 0.94,
             r'\textit{Streptococcus}--\textit{Veillonella} mutualism $\to$ competition in dysbiosis',
             ha='center', fontsize=8, color='0.35')

    # Pg centrality bar chart
    states = list(REW['states'].keys())
    cen    = [REW['pg_eig_centrality'][s] for s in states]
    cols   = ['#7fbf7f', '#7fbf7f', '#d68a8a', '#d68a8a']
    bars   = ax3.bar(range(len(states)), cen, color=cols, edgecolor='k', linewidth=0.7)
    ax3.set_xticks(range(len(states))); ax3.set_xticklabels(states, fontsize=8)
    ax3.set_ylabel('Pg eigenvector centrality')
    ax3.set_title('Pg network role rises into dysbiosis', fontsize=9)
    for b, v in zip(bars, cen):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.006,
                 f'{v:.2f}', ha='center', fontsize=7)
    ax3.axvspan(-0.5, 1.5, color='#7fbf7f', alpha=0.08)
    ax3.axvspan( 1.5, 3.5, color='#d68a8a', alpha=0.08)
    ax3.text(0.5, max(cen) * 1.02, 'commensal', ha='center', fontsize=7, color='#3a7a3a')
    ax3.text(2.5, max(cen) * 1.02, 'dysbiotic',  ha='center', fontsize=7, color='#a33')
    ax3.set_ylim(0, max(cen) * 1.18)
    for s in ('top', 'right'):
        ax3.spines[s].set_visible(False)

    fig.suptitle(r'Commensal$\to$dysbiotic network rewiring (Heine 5-species, gauge-invariant $A_{\mathrm{eff}}$)',
                 y=1.0, fontsize=9, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = _OUT / 'fig_network_rewiring_thesis'
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print('wrote %s (rewired=%d, Pg CS->DH: %.2f->%.2f)' % (
        out.with_suffix('.pdf'), len(rewired), cen[0], cen[-1]))


if __name__ == '__main__':
    fig_backbone()
    fig_rewiring()
    print('done ->', _OUT)
