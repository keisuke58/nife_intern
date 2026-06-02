#!/usr/bin/env python3
"""
analyze_network_rewiring.py  （ネットワーク解析 D）
  Heine 5 種 in-vitro モデルの 4 アトラクター（CS/CH/DS/DH）の相互作用ネットワークが
  commensal → dysbiotic でどう「リワイヤリング」するかを posterior ベースで解析する。

ポイント:
  - 推定 A は全エントリ正だが、replicator dynamics は **列方向の定数シフトに不変**（ゲージ自由度）。
    意味を持つのは列中心化 A_eff = A - mean_i(A_:,j)（= 種 j の平均効果からの偏差 = 実効的な促進/競争）。
    A_eff の符号が状態間で反転 = リワイヤリング。
  - DH の MAP フィットは多峰性のため、単一 MAP でなく posterior サンプルで
    エッジごとの P(A_eff>0) を出す。
  - データは **論文用 10000-particle TMCMC**（results/ultimate_10000p/）。DH = dh_baseline。

入力: results/ultimate_10000p/{commensal_static,commensal_hobic,dysbiotic_static,dh_baseline}/samples.npy
出力: results/dieckow_cr/network_rewiring.json, figures/network_rewiring.pdf/png

repo root から: python analyze_network_rewiring.py
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

from attractor_analysis import theta_to_matrices, SPECIES_ORDER
from paper_data import PAPER_5SP_STATES, PAPER_5SP_SAMPLES, paper_5sp_samples

_here = Path(__file__).resolve().parents[2]
RES = _here / 'results' / 'dieckow_cr'
FIGDIR = _here / 'figures'
FIGDIR.mkdir(exist_ok=True)
NS = len(SPECIES_ORDER)            # 5
PG = SPECIES_ORDER.index('Pg')     # 4

# 論文用データは paper_data.py に一元化（results/ultimate_10000p, DH=dh_baseline）
STATES = PAPER_5SP_STATES
COMMENSAL = ['CS', 'CH']
DYSBIOTIC = ['DS', 'DH']
OFF = ~np.eye(NS, dtype=bool)


def state_effective(samples):
    """samples (M,20) → A_eff スタック (M,5,5)（列中心化）と posterior の P(A_eff>0)。"""
    Aeff = np.empty((len(samples), NS, NS))
    for k, th in enumerate(samples):
        A, _ = theta_to_matrices(np.asarray(th))
        A = A - A.mean(axis=0, keepdims=True)   # 列中心化（ゲージ不変）
        np.fill_diagonal(A, 0.0)
        Aeff[k] = A
    p_pos = (Aeff > 0).mean(axis=0)             # P(effective facilitation)
    return Aeff, p_pos


def eig_centrality(Wabs):
    G = nx.from_numpy_array(np.maximum(Wabs, Wabs.T), create_using=nx.Graph)
    c = nx.eigenvector_centrality_numpy(G, weight='weight')
    return np.array([c[i] for i in range(NS)])


def main():
    data = {}
    for tag, d in STATES.items():
        s = paper_5sp_samples(tag)
        Aeff, p_pos = state_effective(s)
        data[tag] = {'Aeff': Aeff, 'p_pos': p_pos, 'mean': Aeff.mean(0)}

    # commensal / dysbiotic にプール
    comm = np.concatenate([data[t]['Aeff'] for t in COMMENSAL])
    dys = np.concatenate([data[t]['Aeff'] for t in DYSBIOTIC])
    P_comm = (comm > 0).mean(0)
    P_dys = (dys > 0).mean(0)

    # ── リワイヤリング: 実効符号が反転かつ両側 confident ──
    rewired = []
    for i in range(NS):
        for j in range(NS):
            if i == j:
                continue
            pc, pd = P_comm[i, j], P_dys[i, j]
            if (pc - 0.5) * (pd - 0.5) < 0 and (abs(pc - 0.5) > 0.25 and abs(pd - 0.5) > 0.25):
                rewired.append({
                    'from': SPECIES_ORDER[j], 'to': SPECIES_ORDER[i],   # 列 j が行 i に及ぼす効果
                    'commensal': 'facilit.' if pc > 0.5 else 'compet.',
                    'dysbiotic': 'facilit.' if pd > 0.5 else 'compet.',
                    'P_comm_fac': float(pc), 'P_dys_fac': float(pd),
                })

    # ── Pg のネットワーク役割の変化（keystone-in-dysbiosis 検定）──
    # Pg の他種への実効効果 = A_eff[:, PG]（列 Pg）の符号分布
    pg_out = {}
    for tag in STATES:
        col = data[tag]['Aeff'][:, :, PG]            # (M, 5) Pg→others
        pg_out[tag] = {'mean_effect': float(col[:, OFF[:, PG]].mean()),
                       'P_facilit': float((col[OFF[:, PG][None, :].repeat(len(col), 0)] > 0).mean())}
    # Pg 中心性（状態別, posterior mean A_eff）
    pg_cen = {tag: float(eig_centrality(np.abs(data[tag]['mean']))[PG]) for tag in STATES}
    cen_all = {tag: eig_centrality(np.abs(data[tag]['mean'])).tolist() for tag in STATES}

    out = {
        'species': SPECIES_ORDER, 'states': STATES,
        'P_facilitation_commensal': P_comm.tolist(),
        'P_facilitation_dysbiotic': P_dys.tolist(),
        'rewired_edges': rewired,
        'pg_outgoing_effect': pg_out,
        'pg_eig_centrality': pg_cen,
        'eig_centrality_by_state': cen_all,
    }
    (RES / 'network_rewiring.json').write_text(json.dumps(out, indent=2))

    # ── print ──
    print('=== D) Heine 5-species network rewiring (commensal vs dysbiotic) ===')
    print('species:', SPECIES_ORDER, ' (ultimate_10000p posterior, 10000 samples/state)')
    print('\nRewired edges (effective sign flip, both |P-0.5|>0.25):')
    if not rewired:
        print('  (none meeting threshold)')
    for e in rewired:
        print('  %s→%s : commensal=%s (P_fac=%.2f) → dysbiotic=%s (P_fac=%.2f)'
              % (e['from'], e['to'], e['commensal'], e['P_comm_fac'], e['dysbiotic'], e['P_dys_fac']))

    print('\nPg outgoing effective effect (Pg→others) by state:')
    for tag in STATES:
        print('  %s: mean_effect=%+.3f  P(facilitation)=%.2f' %
              (tag, pg_out[tag]['mean_effect'], pg_out[tag]['P_facilit']))
    print('\nPg eigenvector centrality by state:')
    for tag in STATES:
        print('  %s: %.3f' % (tag, pg_cen[tag]))
    cm = np.mean([pg_out[t]['mean_effect'] for t in COMMENSAL])
    dm = np.mean([pg_out[t]['mean_effect'] for t in DYSBIOTIC])
    print('  → Pg mean outgoing effect: commensal=%+.3f  dysbiotic=%+.3f  (Δ=%+.3f)' % (cm, dm, dm - cm))

    make_figure(out, P_comm, P_dys, data, pg_out, pg_cen)
    print(f"\nwrote {RES/'network_rewiring.json'} and {FIGDIR/'network_rewiring.pdf'}")


def make_figure(out, P_comm, P_dys, data, pg_out, pg_cen):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    labs = SPECIES_ORDER

    # (a) P(facilitation) commensal
    for ax, P, ttl in [(axes[0, 0], P_comm, '(a) Commensal: P(effective facilitation)'),
                       (axes[0, 1], P_dys, '(b) Dysbiotic: P(effective facilitation)')]:
        Pm = P.copy(); np.fill_diagonal(Pm, np.nan)
        im = ax.imshow(Pm, cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_xticks(range(NS)); ax.set_xticklabels(labs)
        ax.set_yticks(range(NS)); ax.set_yticklabels(labs)
        ax.set_xlabel('source j (effect of)'); ax.set_ylabel('target i (on)')
        ax.set_title(ttl)
        for i in range(NS):
            for j in range(NS):
                if i != j:
                    ax.text(j, i, '%.2f' % Pm[i, j], ha='center', va='center', fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    # (c) rewiring differential: P_dys - P_comm
    ax = axes[1, 0]
    D = (P_dys - P_comm); np.fill_diagonal(D, np.nan)
    im = ax.imshow(D, cmap='PuOr', vmin=-1, vmax=1)
    ax.set_xticks(range(NS)); ax.set_xticklabels(labs)
    ax.set_yticks(range(NS)); ax.set_yticklabels(labs)
    ax.set_xlabel('source j'); ax.set_ylabel('target i')
    ax.set_title('(c) Rewiring ΔP = dysbiotic − commensal')
    for i in range(NS):
        for j in range(NS):
            if i != j:
                ax.text(j, i, '%+.2f' % D[i, j], ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (d) Pg role across states
    ax = axes[1, 1]
    tags = list(STATES.keys())
    eff = [pg_out[t]['mean_effect'] for t in tags]
    cen = [pg_cen[t] for t in tags]
    x = np.arange(len(tags))
    ax.bar(x - 0.2, eff, 0.4, label='Pg mean outgoing effect', color='#ea2323')
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, cen, 0.4, label='Pg eig-centrality', color='#703a98')
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.axhline(0, color='k', lw=0.7)
    ax.set_ylabel('Pg mean outgoing effective effect', color='#ea2323')
    ax2.set_ylabel('Pg eigenvector centrality', color='#703a98')
    ax.set_title('(d) Pg network role: commensal(CS,CH) vs dysbiotic(DS,DH)')

    fig.tight_layout()
    fig.savefig(FIGDIR / 'network_rewiring.pdf')
    fig.savefig(FIGDIR / 'network_rewiring.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
