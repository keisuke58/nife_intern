#!/usr/bin/env python3
"""
analyze_interaction_network.py
  fit した相互作用行列 A を「10 ギルド上の符号付き有向ネットワーク」として解析する。
  ANALYSIS_NOTES §5 の所見（cross-feeding 方向のみ独立検証 p=0.0004）を足場に、
  ネットワーク中心の 3 解析を行う。

  A) 中心性 → keystone-pathogen / bridge 仮説の定量検定
       - 影響中心性（|A| の固有ベクトル中心性・媒介中心性・PageRank・out/in 強度）
       - 中心性 vs 平均存在量 → 「低存在量・高中心性」= keystone（Bacteroidia/Pg 期待）
       - 媒介中心性 → bridge（Fusobacteriia 期待）
       - no-prior LOO 10 fold で中心性順位の安定性（順位の平均±SD）
  B) trophic coherence / syntrophy 階層（有向 AGORA cross-feeding 網）
       - 栄養段位 s_i（Levine/Johnson）と非整合度 q（小さいほど安定）
       - ランダム化網との比較
  E) 多層 concordant backbone（生態 Hamilton 層 × 代謝 AGORA 層）
       - 両層で符号一致するエッジ（= p=0.0004 を作ったセル）の部分網
       - 代謝 net_flow から流れの向きを注釈

  入力: results/dieckow_cr/loo_{glv,expanded}_noprior_a0p0_fold*.json （no-prior 純データ網）
        results/dieckow_otu/phi_guild.npy （存在量）
        build_net_flow_expanded.net_flow_{glv,hamilton} （AGORA 代謝層）
  出力: results/dieckow_cr/network_analysis.json
        figures/network_analysis.pdf / .png

repo root から: python analyze_interaction_network.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import glob
import json
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from build_net_flow_expanded import net_flow_glv, net_flow_hamilton
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

_here = Path(__file__).resolve().parents[2]
RES = _here / 'results' / 'dieckow_cr'
FIGDIR = _here / 'figures'
FIGDIR.mkdir(exist_ok=True)
SHORT = [GUILD_SHORT.get(g, g) for g in GUILD_ORDER]
COL = [GUILD_COLORS[g] for g in GUILD_ORDER]
N = len(GUILD_ORDER)


# ───────────────────────── data ─────────────────────────
def load_folds(glob_pat):
    files = sorted(glob.glob(str(RES / glob_pat)))
    As = np.stack([np.array(json.load(open(f))['A']) for f in files])
    return As                       # (n_fold, N, N)


def mean_abundance():
    phi = np.load(_here / 'results' / 'dieckow_otu' / 'phi_guild.npy')  # (10,3,10)
    return phi.mean(axis=(0, 1))    # (N,)


# ───────────────────────── A: centrality ─────────────────────────
def centralities(A_signed):
    """|A| を相互作用強度とみた中心性群。A_signed: (N,N) 有向。"""
    W = np.abs(A_signed.copy())
    np.fill_diagonal(W, 0.0)
    # directed graph for PageRank / strengths
    Gd = nx.from_numpy_array(W, create_using=nx.DiGraph)
    # undirected (symmetrized) for eigenvector / betweenness
    Wu = np.maximum(W, W.T)
    Gu = nx.from_numpy_array(Wu, create_using=nx.Graph)

    eig = nx.eigenvector_centrality_numpy(Gu, weight='weight')
    # betweenness: distance = 1/strength (強いほど近い)
    Gb = nx.Graph()
    Gb.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if Wu[i, j] > 0:
                Gb.add_edge(i, j, dist=1.0 / Wu[i, j])
    btw = nx.betweenness_centrality(Gb, weight='dist', normalized=True)
    pr = nx.pagerank(Gd, weight='weight') if W.sum() > 0 else {i: 0 for i in range(N)}

    out_str = W.sum(axis=1)         # 他へ及ぼす影響
    in_str = W.sum(axis=0)          # 他から受ける影響
    return {
        'eigenvector': np.array([eig[i] for i in range(N)]),
        'betweenness': np.array([btw[i] for i in range(N)]),
        'pagerank':    np.array([pr[i] for i in range(N)]),
        'out_strength': out_str,
        'in_strength': in_str,
    }


def centrality_rank_stability(As):
    """fold ごとに eigenvector 中心性順位 → 平均順位±SD。"""
    ranks = []
    for A in As:
        c = centralities(A)['eigenvector']
        # 高い中心性 = 順位1
        order = (-c).argsort()
        rank = np.empty(N, int)
        rank[order] = np.arange(1, N + 1)
        ranks.append(rank)
    ranks = np.array(ranks)
    return ranks.mean(axis=0), ranks.std(axis=0)


# ───────────────────────── B: trophic coherence ─────────────────────────
def trophic_levels(W):
    """MacKay/Johnson (2020) trophic level h と非整合度 F0（閉路に頑健）。
    W[i,j]>0 means i→j (i feeds j)。h は Laplacian L h = (in-out) を最小ノルム解で。
    F0 = Σ W_ij (h_j-h_i-1)^2 / Σ W_ij ∈ [0,1]（0=完全コヒーレント/DAG様, 1=最大非整合）。"""
    A = W.astype(float).copy()
    np.fill_diagonal(A, 0.0)
    w_in = A.sum(axis=0)
    w_out = A.sum(axis=1)
    v = w_in - w_out                 # imbalance
    deg = w_in + w_out
    L = np.diag(deg) - (A + A.T)      # weighted graph Laplacian
    h = np.linalg.lstsq(L, v, rcond=None)[0]   # min-norm（L 特異でも可）
    h = h - h.min()                  # basal を 0 に揃える
    den = A.sum()
    if den <= 0:
        return h, float('nan')
    num = 0.0
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0:
                num += A[i, j] * (h[j] - h[i] - 1.0) ** 2
    return h, float(num / den)


def trophic_random_baseline(W, nperm, rng):
    """エッジ本数を保ったランダム有向網の F0 分布。"""
    B = (W > 0).astype(int)
    np.fill_diagonal(B, 0)
    m = int(B.sum())
    off = [(i, j) for i in range(N) for j in range(N) if i != j]
    fs = []
    for _ in range(nperm):
        sel = rng.choice(len(off), size=m, replace=False)
        Br = np.zeros((N, N))
        for k in sel:
            i, j = off[k]
            Br[i, j] = 1.0
        _, f0 = trophic_levels(Br)
        if np.isfinite(f0):
            fs.append(f0)
    return np.array(fs)


# ───────────────────────── E: concordant backbone ─────────────────────────
def concordant_backbone(A_eco_sym, M_meta_sym, M_meta_dir):
    """生態(対称) と 代謝(対称) の符号が一致する off-diag エッジ。向きは有向代謝網から。"""
    eco = np.sign(A_eco_sym)
    met = np.sign(M_meta_sym)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if met[i, j] != 0 and eco[i, j] == met[i, j]:
                # 流れの向き: 有向代謝網で強い方
                if M_meta_dir[i, j] >= M_meta_dir[j, i]:
                    src, dst = i, j
                else:
                    src, dst = j, i
                edges.append({'src': GUILD_ORDER[src], 'dst': GUILD_ORDER[dst],
                              'sign': int(met[i, j])})
    return edges


# ───────────────────────── run ─────────────────────────
def main():
    rng = np.random.default_rng(0)
    abund = mean_abundance()

    As_glv = load_folds('loo_glv_noprior_a0p0_fold*.json')          # directed
    As_ham = load_folds('loo_expanded_noprior_a0p0_fold*.json')     # symmetric
    A_glv = As_glv.mean(0)
    A_ham = As_ham.mean(0)

    # --- A: centrality ---
    cen = centralities(A_glv)              # 有向データ網（gLV consensus）
    cen_ham = centralities(A_ham)          # 対称データ網（betweenness/bridge 用）
    rk_mean, rk_sd = centrality_rank_stability(As_ham)

    # keystone gap: 中心性順位が存在量順位より高い(=低存在量で高中心性)ほど keystone 的。
    # "Other"(catch-all) は希少すぎて除算/順位を歪めるため除外。
    eig = cen_ham['eigenvector']
    valid = np.array([g != 'Other' for g in GUILD_ORDER])
    def _rank(x, mask):                # 1 = 最大
        r = np.full(N, np.nan)
        idx = np.where(mask)[0]
        order = idx[np.argsort(-x[idx])]
        r[order] = np.arange(1, len(order) + 1)
        return r
    rank_cen = _rank(eig, valid)
    rank_ab = _rank(abund, valid)
    keystone_gap = rank_ab - rank_cen  # >0: 存在量の割に中心的 = keystone 的

    # --- B: trophic coherence ---
    M_dir = np.asarray(net_flow_glv(use_agora=True, competition_weight=0.0))  # 有向 cross-feeding
    s_meta, F0_meta = trophic_levels(np.where(M_dir > 0, M_dir, 0.0))
    f0_null = trophic_random_baseline(np.where(M_dir > 0, 1.0, 0.0), 2000, rng)
    f0_p = float((np.sum(f0_null <= F0_meta) + 1) / (f0_null.size + 1)) if f0_null.size else float('nan')
    # data-driven 有向 cross-feeding（gLV consensus の正エッジ）
    Wglv_pos = np.where(A_glv > 0, A_glv, 0.0); np.fill_diagonal(Wglv_pos, 0)
    s_glv, F0_glv = trophic_levels(Wglv_pos)

    # --- E: concordant backbone ---
    M_meta_sym = np.asarray(net_flow_hamilton(use_agora=True, competition_weight=0.0))
    backbone = concordant_backbone(A_ham, M_meta_sym, M_dir)

    # ── JSON ──
    out = {
        'guilds': GUILD_ORDER,
        'mean_abundance': abund.tolist(),
        'A_centrality': {k: cen[k].tolist() for k in cen},
        'A_centrality_symmetric': {k: cen_ham[k].tolist() for k in cen_ham},
        'eigenvector_rank_mean': rk_mean.tolist(),
        'eigenvector_rank_sd': rk_sd.tolist(),
        'keystone_gap': keystone_gap.tolist(),
        'centrality_rank': rank_cen.tolist(),
        'abundance_rank': rank_ab.tolist(),
        'B_trophic': {
            'meta_levels': s_meta.tolist(), 'meta_F0': F0_meta,
            'F0_null_mean': float(f0_null.mean()) if f0_null.size else None,
            'F0_null_std': float(f0_null.std()) if f0_null.size else None,
            'F0_p_value': f0_p,
            'glv_levels': s_glv.tolist(), 'glv_F0': F0_glv,
        },
        'E_concordant_backbone': backbone,
    }
    (RES / 'network_analysis.json').write_text(json.dumps(out, indent=2))

    # ── print summary ──
    print('\n=== A) Centrality vs abundance (keystone/bridge) ===')
    print('%-8s  abund   eig   btw   PR   out   in   rankCen rankAb keyGap  rank(eig)±sd' % 'guild')
    for i in range(N):
        kg = '%+.0f' % keystone_gap[i] if not np.isnan(keystone_gap[i]) else '  -'
        rc = '%.0f' % rank_cen[i] if not np.isnan(rank_cen[i]) else '-'
        ra = '%.0f' % rank_ab[i] if not np.isnan(rank_ab[i]) else '-'
        print('%-8s  %.3f  %.3f %.3f %.3f %.2f %.2f    %2s     %2s    %4s    %.1f±%.1f' %
              (SHORT[i], abund[i], cen_ham['eigenvector'][i], cen_ham['betweenness'][i],
               cen['pagerank'][i], cen['out_strength'][i], cen['in_strength'][i],
               rc, ra, kg, rk_mean[i], rk_sd[i]))
    gap = np.where(np.isnan(keystone_gap), -np.inf, keystone_gap)
    ks = int(np.argmax(gap)); br = int(np.argmax(cen_ham['betweenness']))
    print(f'  → keystone-like (max rank gap): {SHORT[ks]} ({GUILD_ORDER[ks]})')
    print(f'  → bridge (max betweenness): {SHORT[br]} ({GUILD_ORDER[br]})')
    bi = GUILD_ORDER.index('Bacteroidia')
    print(f'  → Pg-keystone test: Bacteroidia eig-rank={rank_cen[bi]:.0f} abund-rank={rank_ab[bi]:.0f} gap={keystone_gap[bi]:+.0f}')

    print('\n=== B) Trophic coherence (directed cross-feeding) ===')
    print('  AGORA meta net: F0=%.3f  null=%.3f±%.3f  p=%.4f (small F0 = coherent/stable)' %
          (F0_meta, out['B_trophic']['F0_null_mean'], out['B_trophic']['F0_null_std'], f0_p))
    print('  data-driven gLV cross-feeding: F0=%.3f' % F0_glv)
    print('  trophic levels (AGORA, low=producer):')
    for i in np.argsort(s_meta):
        print('    %-8s  h=%.2f' % (SHORT[i], s_meta[i]))

    print('\n=== E) Concordant backbone (eco∩meta, %d edges) ===' % len(backbone))
    for e in backbone:
        d = GUILD_SHORT.get(e['src'], e['src']); t = GUILD_SHORT.get(e['dst'], e['dst'])
        print('    %-8s --(%s)--> %-8s' % (d, '+' if e['sign'] > 0 else '-', t))

    make_figure(out, cen, cen_ham, abund, keystone_gap, s_meta, M_dir, backbone, F0_meta, f0_null)
    print(f"\nwrote {RES/'network_analysis.json'} and {FIGDIR/'network_analysis.pdf'}")


def make_figure(out, cen, cen_ham, abund, keystone_gap, s_meta, M_dir, backbone, F0_meta, f0_null):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) centrality vs abundance — keystone plot
    ax = axes[0, 0]
    eig = cen_ham['eigenvector']
    ax.scatter(abund, eig, c=COL, s=140, edgecolors='k', zorder=3)
    for i in range(N):
        ax.annotate(SHORT[i], (abund[i], eig[i]), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('mean relative abundance'); ax.set_ylabel('eigenvector centrality (|A|)')
    ax.set_title('(a) Keystone test: centrality vs abundance')
    ax.axhline(np.median(eig), ls=':', c='grey'); ax.axvline(np.median(abund), ls=':', c='grey')

    # (b) betweenness — bridge
    ax = axes[0, 1]
    order = np.argsort(cen_ham['betweenness'])[::-1]
    ax.bar(range(N), cen_ham['betweenness'][order], color=[COL[i] for i in order],
           edgecolor='k')
    ax.set_xticks(range(N)); ax.set_xticklabels([SHORT[i] for i in order], rotation=60, ha='right')
    ax.set_ylabel('betweenness centrality'); ax.set_title('(b) Bridge test (betweenness)')

    # (c) trophic hierarchy of directed cross-feeding net
    ax = axes[1, 0]
    G = nx.DiGraph()
    P = (M_dir > 0)
    for i in range(N):
        for j in range(N):
            if i != j and P[i, j]:
                G.add_edge(i, j)
    # x = small jitter by index, y = trophic level
    xpos = {}
    bylevel = {}
    for i in range(N):
        lv = round(s_meta[i], 1)
        bylevel.setdefault(lv, []).append(i)
    pos = {}
    for lv, nodes in bylevel.items():
        for k, i in enumerate(sorted(nodes)):
            pos[i] = (k - (len(nodes) - 1) / 2.0, s_meta[i])
    for i in range(N):
        if i not in pos:
            pos[i] = (0, s_meta[i])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='grey', alpha=0.6,
                           arrows=True, arrowstyle='-|>', arrowsize=12,
                           connectionstyle='arc3,rad=0.08')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(range(N)),
                           node_color=COL, edgecolors='k', node_size=420)
    nx.draw_networkx_labels(G, pos, labels={i: SHORT[i] for i in range(N)},
                            ax=ax, font_size=7)
    ax.set_title('(c) Syntrophy hierarchy (AGORA cross-feeding)\nF0=%.3f (null %.3f)'
                 % (F0_meta, f0_null.mean() if f0_null.size else float('nan')))
    ax.set_ylabel('trophic level'); ax.set_xticks([])

    # (d) concordant backbone
    ax = axes[1, 1]
    Gb = nx.DiGraph()
    Gb.add_nodes_from(range(N))
    idx = {g: i for i, g in enumerate(GUILD_ORDER)}
    for e in backbone:
        Gb.add_edge(idx[e['src']], idx[e['dst']], sign=e['sign'])
    posb = nx.circular_layout(Gb)
    nx.draw_networkx_edges(Gb, posb, ax=ax, edge_color='#1a9850', width=2.0,
                           arrows=True, arrowstyle='-|>', arrowsize=14,
                           connectionstyle='arc3,rad=0.1')
    deg = np.array([Gb.degree(i) for i in range(N)])
    nx.draw_networkx_nodes(Gb, posb, ax=ax, nodelist=list(range(N)),
                           node_color=COL, edgecolors='k',
                           node_size=120 + 90 * deg)
    nx.draw_networkx_labels(Gb, posb, labels={i: SHORT[i] for i in range(N)},
                            ax=ax, font_size=7)
    ax.set_title('(d) Concordant backbone (eco∩meta, %d edges)' % len(backbone))
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(FIGDIR / 'network_analysis.pdf')
    fig.savefig(FIGDIR / 'network_analysis.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
