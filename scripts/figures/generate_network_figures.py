#!/usr/bin/env python3
"""
generate_network_figures.py — 論文品質のネットワーク図（vector PDF）。
analyze_interaction_network.py / analyze_network_rewiring.py の JSON 出力を読み、
pub_style で清書する（再計算しない）。

  Fig 1 (E): concordant ecological–metabolic cross-feeding backbone
             — 代謝の trophic level を x 軸にした syntrophy 順レイアウト。
               Veillonella(Negativicutes) が収束的アクセプター（乳酸クロスフィーディング）。
  Fig 2 (D): commensal→dysbiotic rewiring + Pg のネットワーク役割
             (a) 主要エッジの実効符号反転（commensal vs dysbiotic）
             (b) Pg 固有ベクトル中心性が dysbiosis で上昇（DH 最大）

出力: figures/fig_network_backbone.{pdf,png}, figures/fig_network_rewiring.{pdf,png}
repo root から: python generate_network_figures.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pub_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

pub_style.apply()
_here = Path(__file__).resolve().parents[2]
RES = _here / 'results' / 'dieckow_cr'
FIG = _here / 'figures'
FIG.mkdir(exist_ok=True)

NET = json.loads((RES / 'network_analysis.json').read_text())
REW = json.loads((RES / 'network_rewiring.json').read_text())

SHORT = {g: GUILD_SHORT.get(g, g) for g in GUILD_ORDER}
FAC = '#2166ac'   # facilitation (blue)
COMP = '#b2182b'  # competition (red)


# ────────────────────────────── Fig 1: concordant backbone ──────────────────────────────
def fig_backbone():
    backbone = NET['E_concordant_backbone']
    levels = {g: NET['B_trophic']['meta_levels'][i] for i, g in enumerate(GUILD_ORDER)}
    nodes = sorted({e['src'] for e in backbone} | {e['dst'] for e in backbone})

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in backbone:
        G.add_edge(e['src'], e['dst'])
    indeg = {n: G.in_degree(n) for n in nodes}

    # x = trophic 順位の等間隔（producer 左→consumer 右）、y = ジグザグで重なり回避
    order = sorted(nodes, key=lambda n: (levels[n], n))
    stagger = [0.0, 1.0, -1.0, 0.55, -0.55, 1.0, -1.0, 0.5, -0.5]
    pos = {n: (i, stagger[i % len(stagger)]) for i, n in enumerate(order)}
    nsize = {n: 900 + 650 * indeg[n] for n in nodes}

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color='#4a8f5b', width=1.7, alpha=0.85,
        arrows=True, arrowstyle='-|>', arrowsize=14,
        node_size=[nsize[n] for n in G.nodes()],
        connectionstyle='arc3,rad=0.14', min_target_margin=13)
    for n in nodes:
        ring = 2.4 if n == 'Negativicutes' else 1.2
        ax.scatter(*pos[n], s=nsize[n], color=GUILD_COLORS[n],
                   edgecolors=(COMP if n == 'Negativicutes' else 'k'),
                   linewidths=ring, zorder=3)
        ax.annotate(SHORT[n], pos[n], ha='center', va='center', fontsize=8,
                    fontweight='bold', zorder=4,
                    color='white' if n in ('Bacteroidia', 'Negativicutes', 'Other') else 'black')
    # Veillonella(Negativicutes) を強調（収束的アクセプター）
    if 'Negativicutes' in pos:
        x, y = pos['Negativicutes']
        ax.annotate('Veillonella\n(lactate acceptor,\n4 incoming)', (x, y),
                    xytext=(x + 0.15, y - 1.7), ha='center', va='top',
                    fontsize=7.5, style='italic', color=COMP,
                    arrowprops=dict(arrowstyle='-', color=COMP, lw=0.8))

    ax.set_xlabel('metabolic trophic level  (producer → consumer)')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Concordant ecological–metabolic cross-feeding backbone', pad=24)
    ax.text(0.5, 1.02,
            f'{len(backbone)} edges where data-fitted (Hamilton) and AGORA metabolic signs agree '
            f'(permutation $p=0.0004$)',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='0.3')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-2.0, 2.4)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_network_backbone.pdf')
    fig.savefig(FIG / 'fig_network_backbone.png', dpi=200)
    plt.close(fig)
    print('wrote fig_network_backbone (%d edges, nodes=%s)' % (len(backbone), [SHORT[n] for n in nodes]))


# ────────────────────────────── Fig 2: rewiring + Pg ──────────────────────────────
def _state_net(ax, P, sp, title, key_edges):
    """5 種の主要エッジを実効符号で描く（facilitation 青 / competition 赤）。"""
    pos = nx.circular_layout(nx.path_graph(len(sp)))
    pos = {i: pos[i] for i in range(len(sp))}
    for (j, i) in key_edges:
        col = FAC if P[i][j] > 0.5 else COMP
        conf = abs(P[i][j] - 0.5) * 2
        ax.annotate('', xy=pos[i], xytext=pos[j],
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.0 + 2.2 * conf,
                                    alpha=0.85, connectionstyle='arc3,rad=0.15',
                                    shrinkA=14, shrinkB=14))
    for i, s in enumerate(sp):
        ax.scatter(*pos[i], s=620, color='#ea2323' if s == 'Pg' else '#cccccc',
                   edgecolors='k', linewidths=1.0, zorder=3)
        ax.annotate(s, pos[i], ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)
    ax.set_title(title, fontsize=11)
    ax.axis('off'); ax.margins(0.18)


def fig_rewiring():
    sp = REW['species']
    Pc = REW['P_facilitation_commensal']
    Pd = REW['P_facilitation_dysbiotic']
    idx = {s: i for i, s in enumerate(sp)}
    # 主要エッジ = rewired + Strep-Veillonella ペア
    rewired = REW['rewired_edges']
    key_edges = sorted({(idx[e['from']], idx[e['to']]) for e in rewired})

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); ax3 = fig.add_subplot(gs[2])

    _state_net(ax1, Pc, sp, 'Commensal (CS, CH)', key_edges)
    _state_net(ax2, Pd, sp, 'Dysbiotic (DS, DH)', key_edges)

    # legend
    handles = [Line2D([0], [0], color=FAC, lw=3, label='effective facilitation'),
               Line2D([0], [0], color=COMP, lw=3, label='effective competition')]
    ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.16),
               ncol=1, frameon=False, fontsize=8)
    fig.text(0.34, 0.93,
             'Streptococcus–Veillonella mutualism → competition in dysbiosis',
             ha='center', fontsize=9, color='0.3')

    # (c) Pg centrality across states
    states = list(REW['states'].keys())
    cen = [REW['pg_eig_centrality'][s] for s in states]
    cols = ['#7fbf7f', '#7fbf7f', '#d68a8a', '#d68a8a']  # commensal green / dysbiotic red-ish
    bars = ax3.bar(range(len(states)), cen, color=cols, edgecolor='k', linewidth=0.8)
    ax3.set_xticks(range(len(states))); ax3.set_xticklabels(states)
    ax3.set_ylabel('Pg eigenvector centrality')
    ax3.set_title('Pg network role rises into dysbiosis', fontsize=11)
    for b, v in zip(bars, cen):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.008, '%.2f' % v, ha='center', fontsize=8)
    ax3.axvspan(-0.5, 1.5, color='#7fbf7f', alpha=0.08)
    ax3.axvspan(1.5, 3.5, color='#d68a8a', alpha=0.08)
    ax3.text(0.5, max(cen) * 1.02, 'commensal', ha='center', fontsize=8, color='#3a7a3a')
    ax3.text(2.5, max(cen) * 1.02, 'dysbiotic', ha='center', fontsize=8, color='#a33')
    ax3.set_ylim(0, max(cen) * 1.18)
    for s in ('top', 'right'):
        ax3.spines[s].set_visible(False)

    fig.suptitle('Commensal→dysbiotic network rewiring (Heine 5-species, gauge-invariant $A_{\\mathrm{eff}}$)',
                 y=1.0, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / 'fig_network_rewiring.pdf')
    fig.savefig(FIG / 'fig_network_rewiring.png', dpi=200)
    plt.close(fig)
    print('wrote fig_network_rewiring (rewired edges=%d, Pg centrality CS→DH: %.2f→%.2f)'
          % (len(rewired), cen[0], cen[-1]))


if __name__ == '__main__':
    fig_backbone()
    fig_rewiring()
    print('done →', FIG)
