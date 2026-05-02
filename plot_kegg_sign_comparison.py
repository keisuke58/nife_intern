#!/usr/bin/env python3
"""
Post-process sign-consistency figure for paper S1.

Uses:
  - KEGG 1000p runs (with logL)  →  MAP estimate
  - 10000p Phase-2 TMCMC runs    →  posterior mean (larger, more stable)

Generates:
  docs/figures/dieckow/fig_heine_kegg_sign_comparison.pdf/.png
  results/kegg_sign_summary.json

Sign-agreement: for each (i,j) with net_flow[i,j] != 0,
  check sign(A[i,j]) == sign(net_flow[i,j]).
  Since A is symmetric, A[i,j] = A[j,i].
"""
import json, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
RUNS = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs')
SUPPFILE = (HERE / 'Szafranski_Published_Work' / 'Szafranski_Published_Work'
            / 'public_data' / 'Dieckow'
            / 'Supplementary_File_1_microbe_metabolite_enzyme_interactions.tsv')
FIG_DIR = HERE.parent / 'docs' / 'figures' / 'dieckow'
OUT_JSON = HERE / 'results' / 'kegg_sign_summary.json'

SPECIES = ['So', 'An', 'Vd', 'Fn', 'Pg']
N_SP = 5

KEGG_RUNS = {
    'CS': RUNS / 'CS_kegg_s015_1000p_20260430_183337',
    'CH': RUNS / 'CH_kegg_s015_1000p_20260430_183337',
    'DS': RUNS / 'DS_kegg_s008_1000p_20260430_183337',
    'DH': RUNS / 'DH_kegg_s008_1000p_20260430_183337',
}
P10K_RUNS = {
    'CS': RUNS / 'cs_10000p',
    'CH': RUNS / 'ch_10000p',
    'DS': RUNS / 'ds_10000p',
    'DH': RUNS / 'dh_10000p',
}
# ultimate_10000p has theta_MAP.json (same samples, but MAP is saved)
ULTIMATE_RUNS = {
    'CS': HERE / 'results' / 'ultimate_10000p' / 'commensal_static',
    'CH': HERE / 'results' / 'ultimate_10000p' / 'commensal_hobic',
    'DS': HERE / 'results' / 'ultimate_10000p' / 'dysbiotic_static',
    'DH': HERE / 'results' / 'ultimate_10000p' / 'dh_baseline',
}
CONDITIONS = ['CS', 'CH', 'DS', 'DH']
COND_LABELS = {'CS': 'CS (Commensal Static)', 'CH': 'CH (Commensal HOBIC)',
               'DS': 'DS (Dysbiotic Static)', 'DH': 'DH (Dysbiotic HOBIC)'}


# ── theta → A matrix ──────────────────────────────────────────────────────────

def theta_to_A(theta):
    """Convert theta(20) → symmetric A(5×5). Matches generate_extra_figures_generic.py."""
    A = np.zeros((5, 5))
    A[0,0]=theta[0]; A[0,1]=theta[1]; A[1,0]=theta[1]; A[1,1]=theta[2]
    A[2,2]=theta[5]; A[2,3]=theta[6]; A[3,2]=theta[6]; A[3,3]=theta[7]
    A[0,2]=theta[10]; A[2,0]=theta[10]; A[0,3]=theta[11]; A[3,0]=theta[11]
    A[1,2]=theta[12]; A[2,1]=theta[12]; A[1,3]=theta[13]; A[3,1]=theta[13]
    A[4,4]=theta[14]
    A[0,4]=theta[16]; A[4,0]=theta[16]; A[1,4]=theta[17]; A[4,1]=theta[17]
    A[2,4]=theta[18]; A[4,2]=theta[18]; A[3,4]=theta[19]; A[4,3]=theta[19]
    return A


# ── KEGG net-flow matrix ───────────────────────────────────────────────────────

def build_net_flow():
    GENUS_SP = {
        'Streptococcus': 0, 'Schaalia': 0,
        'Actinomyces': 1,
        'Veillonella': 2, 'Lancefieldella': 2, 'Selenomonas': 2,
        'Fusobacterium': 3, 'Leptotrichia': 3,
        'Porphyromonas': 4, 'Prevotella': 4, 'Tannerella': 4,
    }
    df = pd.read_csv(SUPPFILE, sep='\t')
    pos = np.zeros((N_SP, N_SP))
    neg = np.zeros((N_SP, N_SP))
    for met in df['OBJECT'].unique():
        mdf = df[df['OBJECT'] == met]
        w = float(mdf.apply(lambda r: 2.0 if str(r.get('KEGG','')) not in ('n/a','','nan','NaN')
                            else (2.0 if 'HMDB' in str(r.get('HMDB_ID','')) else 1.0), axis=1).max())
        prod, cons, inhib = set(), set(), set()
        for _, row in mdf.iterrows():
            idx = GENUS_SP.get(str(row['TAXON']).split()[0])
            if idx is None: continue
            if row['RELATIONSHIP'] == 'PRODUCES': prod.add(idx)
            elif row['RELATIONSHIP'] == 'USES': cons.add(idx)
            elif row['RELATIONSHIP'] == 'IS_INHIBITED_BY': inhib.add(idx)
        for src in prod:
            for tgt in cons:
                if src != tgt: pos[tgt, src] += w
            for tgt in inhib:
                if src != tgt: neg[tgt, src] += w
    # eHOMD supplement
    for i, j in [(0,1),(1,3),(2,3),(3,4)]:
        pos[j,i] += 1.0; pos[i,j] += 1.0
    return pos - neg


# ── Sign agreement ─────────────────────────────────────────────────────────────

def sign_agreement(A, net_flow, tol=0.02):
    agree, total = 0, 0
    pairs = []
    for i in range(N_SP):
        for j in range(N_SP):
            if i == j: continue
            f = net_flow[i, j]
            if f == 0: continue
            a = A[i, j]
            if abs(a) < tol: continue
            total += 1
            ok = (np.sign(f) == np.sign(a))
            agree += int(ok)
            pairs.append((SPECIES[i], SPECIES[j], float(f), float(a), bool(ok)))
    return agree, total, pairs


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(net_flow, maps, means):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    vmax = 1.2
    cmap = plt.cm.RdBu_r

    for col, cond in enumerate(CONDITIONS):
        for row_idx, (A, row_label) in enumerate([
                (means[cond], f'{cond}\nposterior mean (10kp)'),
                (maps[cond],  f'{cond}\nMAP-proxy (10kp)')]):
            ax = axes[row_idx, col]
            agree, total, pairs = sign_agreement(A, net_flow)
            pct = 100 * agree / total if total > 0 else 0

            im = ax.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_xticks(range(N_SP)); ax.set_xticklabels(SPECIES, fontsize=8)
            ax.set_yticks(range(N_SP)); ax.set_yticklabels(SPECIES, fontsize=8)
            ax.set_title(f'{row_label}\nSA={agree}/{total} ({pct:.0f}%)', fontsize=7.5)

            for i in range(N_SP):
                for j in range(N_SP):
                    val = A[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=5.5, color='k' if abs(val) < vmax * 0.7 else 'w')

            # Border highlighting: lime=agree, red=disagree
            for i in range(N_SP):
                for j in range(N_SP):
                    if i == j or net_flow[i, j] == 0 or abs(A[i, j]) < 0.02: continue
                    ok = (np.sign(net_flow[i, j]) == np.sign(A[i, j]))
                    color = '#7fff00' if ok else '#ff2222'
                    for spine_pos, (x0,y0,dx,dy) in enumerate([
                            (j-0.5, i-0.5, 1, 0),
                            (j-0.5, i+0.5, 1, 0),
                            (j-0.5, i-0.5, 0, 1),
                            (j+0.5, i-0.5, 0, 1)]):
                        ax.plot([x0, x0+dx], [y0, y0+dy], color=color, lw=1.5, clip_on=False)

    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.02,
                 label='A[i,j]')
    fig.suptitle('KEGG/eHOMD sign-consistency: posterior mean (top, 10 000p) vs MAP (bottom, KEGG 1000p)',
                 fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.97, 1])
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Building KEGG net-flow matrix ...')
    net_flow = build_net_flow()
    nz = int((net_flow != 0).sum())  # includes diagonal (all zero) → all off-diag non-zero
    print(f'  {nz} non-zero off-diagonal entries')
    print(pd.DataFrame(net_flow, index=SPECIES, columns=SPECIES).to_string(float_format='{:.1f}'.format))

    maps, means = {}, {}
    summary = {}

    for cond in CONDITIONS:
        # 10000p Phase-2: posterior mean
        s_10k = np.load(P10K_RUNS[cond] / 'samples.npy')
        means[cond] = theta_to_A(s_10k.mean(axis=0))

        # 10000p MAP from theta_MAP.json (saved by estimate_reduced_nishioka_jax.py)
        with open(ULTIMATE_RUNS[cond] / 'theta_MAP.json') as f:
            d = json.load(f)
        theta_map = np.array([d[str(i)] for i in range(20)])
        maps[cond] = theta_to_A(theta_map)

        # Sign agreement
        ag_mean, tot_mean, _ = sign_agreement(means[cond], net_flow)
        ag_map,  tot_map,  _ = sign_agreement(maps[cond],  net_flow)

        # KEGG 1000p MAP for reference
        s_kegg = np.load(KEGG_RUNS[cond] / 'samples.npy')
        l_kegg = np.load(KEGG_RUNS[cond] / 'logL.npy')
        ag_k, tot_k, _ = sign_agreement(theta_to_A(s_kegg[np.argmax(l_kegg)]), net_flow)

        print(f'{cond}:'
              f'  SA Ā(10kp)={ag_mean}/{tot_mean} ({100*ag_mean/tot_mean:.0f}%)'
              f'  |  SA MAP(10kp)={ag_map}/{tot_map} ({100*ag_map/tot_map:.0f}%)'
              f'  |  SA MAP(KEGG 1k ref)={ag_k}/{tot_k} ({100*ag_k/tot_k:.0f}%)')
        summary[cond] = {
            'SA_mean_agree': ag_mean, 'SA_mean_total': tot_mean,
            'SA_mean_pct':   round(100*ag_mean/tot_mean, 1) if tot_mean else 0,
            'SA_map_agree':  ag_map,  'SA_map_total': tot_map,
            'SA_map_pct':    round(100*ag_map/tot_map, 1) if tot_map else 0,
            'SA_kegg_map_agree': ag_k, 'SA_kegg_map_total': tot_k,
            'SA_kegg_map_pct':   round(100*ag_k/tot_k, 1) if tot_k else 0,
            'logL_MAP_kegg': float(l_kegg.max()),
            'n_kegg_samples': len(s_kegg),
            'n_10k_samples':  len(s_10k),
        }

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump({'net_flow': net_flow.tolist(), 'species': SPECIES,
                   'conditions': summary,
                   'kegg_runs': {k: str(v) for k,v in KEGG_RUNS.items()},
                   'p10k_runs': {k: str(v) for k,v in P10K_RUNS.items()}}, f, indent=2)
    print(f'\nSaved {OUT_JSON}')

    # Figure
    fig = make_figure(net_flow, maps, means)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'fig_heine_kegg_sign_comparison.{ext}', dpi=200, bbox_inches='tight')
    print(f'Saved figures → {FIG_DIR}/fig_heine_kegg_sign_comparison.*')
    plt.close(fig)


if __name__ == '__main__':
    main()
