#!/usr/bin/env python3
"""
Post-process sign-consistency figure for paper S1.

Outputs:
  docs/figures/dieckow/fig_heine_kegg_sign_comparison.pdf/.png
    — 2×4 grid: posterior mean (top) and 10000p MAP (bottom) A-matrices

  docs/figures/dieckow/fig_kegg_sign_probability.pdf/.png
    — P(correct sign) per KEGG pair from 10000 posterior samples (Option A)

  results/kegg_sign_summary.json
"""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
ULTIMATE_RUNS = {
    'CS': HERE / 'results' / 'ultimate_10000p' / 'commensal_static',
    'CH': HERE / 'results' / 'ultimate_10000p' / 'commensal_hobic',
    'DS': HERE / 'results' / 'ultimate_10000p' / 'dysbiotic_static',
    'DH': HERE / 'results' / 'ultimate_10000p' / 'dh_baseline',
}
CONDITIONS = ['CS', 'CH', 'DS', 'DH']


# ── theta → A matrix ──────────────────────────────────────────────────────────

def theta_to_A(theta):
    """Convert theta(20) → symmetric A(5×5)."""
    A = np.zeros((5, 5))
    A[0,0]=theta[0]; A[0,1]=theta[1]; A[1,0]=theta[1]; A[1,1]=theta[2]
    A[2,2]=theta[5]; A[2,3]=theta[6]; A[3,2]=theta[6]; A[3,3]=theta[7]
    A[0,2]=theta[10]; A[2,0]=theta[10]; A[0,3]=theta[11]; A[3,0]=theta[11]
    A[1,2]=theta[12]; A[2,1]=theta[12]; A[1,3]=theta[13]; A[3,1]=theta[13]
    A[4,4]=theta[14]
    A[0,4]=theta[16]; A[4,0]=theta[16]; A[1,4]=theta[17]; A[4,1]=theta[17]
    A[2,4]=theta[18]; A[4,2]=theta[18]; A[3,4]=theta[19]; A[4,3]=theta[19]
    return A


def theta_to_A_batch(samples):
    """Vectorized: (N,20) → (N,5,5)."""
    N = len(samples)
    A = np.zeros((N, 5, 5))
    t = samples
    A[:,0,0]=t[:,0]; A[:,0,1]=t[:,1]; A[:,1,0]=t[:,1]; A[:,1,1]=t[:,2]
    A[:,2,2]=t[:,5]; A[:,2,3]=t[:,6]; A[:,3,2]=t[:,6]; A[:,3,3]=t[:,7]
    A[:,0,2]=t[:,10]; A[:,2,0]=t[:,10]; A[:,0,3]=t[:,11]; A[:,3,0]=t[:,11]
    A[:,1,2]=t[:,12]; A[:,2,1]=t[:,12]; A[:,1,3]=t[:,13]; A[:,3,1]=t[:,13]
    A[:,4,4]=t[:,14]
    A[:,0,4]=t[:,16]; A[:,4,0]=t[:,16]; A[:,1,4]=t[:,17]; A[:,4,1]=t[:,17]
    A[:,2,4]=t[:,18]; A[:,4,2]=t[:,18]; A[:,3,4]=t[:,19]; A[:,4,3]=t[:,19]
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
    for i, j in [(0,1),(1,3),(2,3),(3,4)]:
        pos[j,i] += 1.0; pos[i,j] += 1.0
    return pos - neg


# ── Sign agreement (point estimate) ───────────────────────────────────────────

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


# ── Sign probability from posterior samples (Option A) ────────────────────────

def sign_probability(samples, net_flow):
    """
    Returns prob[i,j] = P(sign(A[i,j]) == sign(F[i,j])) over all samples.
    Only defined for (i,j) where F[i,j] != 0 and i != j; NaN elsewhere.
    """
    A_all = theta_to_A_batch(samples)   # (N, 5, 5)
    prob = np.full((N_SP, N_SP), np.nan)
    for i in range(N_SP):
        for j in range(N_SP):
            if i == j: continue
            f = net_flow[i, j]
            if f == 0: continue
            a_ij = A_all[:, i, j]
            if np.sign(f) > 0:
                prob[i, j] = float((a_ij > 0).mean())
            else:
                prob[i, j] = float((a_ij < 0).mean())
    return prob


# ── Figure 1: A-matrix comparison ─────────────────────────────────────────────

def make_figure_A(net_flow, maps, means):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    vmax = 1.2
    cmap = plt.cm.RdBu_r

    for col, cond in enumerate(CONDITIONS):
        for row_idx, (A, row_label) in enumerate([
                (means[cond], f'{cond}\nposterior mean (10kp)'),
                (maps[cond],  f'{cond}\nMAP (10kp)')]):
            ax = axes[row_idx, col]
            agree, total, _ = sign_agreement(A, net_flow)
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

            for i in range(N_SP):
                for j in range(N_SP):
                    if i == j or net_flow[i, j] == 0 or abs(A[i, j]) < 0.02: continue
                    ok = (np.sign(net_flow[i, j]) == np.sign(A[i, j]))
                    color = '#7fff00' if ok else '#ff2222'
                    for x0, y0, dx, dy in [
                            (j-0.5, i-0.5, 1, 0), (j-0.5, i+0.5, 1, 0),
                            (j-0.5, i-0.5, 0, 1), (j+0.5, i-0.5, 0, 1)]:
                        ax.plot([x0, x0+dx], [y0, y0+dy], color=color, lw=1.5, clip_on=False)

    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.02,
                 label='A[i,j]')
    fig.suptitle('KEGG/eHOMD sign-consistency: posterior mean (top) vs MAP (bottom), 10 000p free posterior',
                 fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.97, 1])
    return fig


# ── Figure 2: Sign probability heatmap ────────────────────────────────────────

def make_figure_signprob(net_flow, sign_probs):
    """
    sign_probs[cond] = (N_SP, N_SP) matrix, NaN for non-KEGG pairs.
    Colormap: 0=surely wrong, 0.5=random, 1=surely correct.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    cmap = plt.cm.RdYlGn   # red<0.5 (wrong), yellow=0.5 (random), green>0.5 (correct)

    for col, cond in enumerate(CONDITIONS):
        ax = axes[col]
        prob = sign_probs[cond]

        # background for non-KEGG cells
        bg = np.full((N_SP, N_SP), np.nan)
        im = ax.imshow(bg, cmap='gray', vmin=0, vmax=1, aspect='auto', alpha=0.1)

        # KEGG cells as scatter-style colored squares
        im2 = ax.imshow(np.where(~np.isnan(prob), prob, np.nan),
                        cmap=cmap, vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(N_SP)); ax.set_xticklabels(SPECIES, fontsize=8)
        ax.set_yticks(range(N_SP)); ax.set_yticklabels(SPECIES, fontsize=8)

        # annotate: P value for KEGG pairs, '—' for non-KEGG
        for i in range(N_SP):
            for j in range(N_SP):
                if i == j:
                    ax.text(j, i, '·', ha='center', va='center', fontsize=9, color='#aaa')
                    continue
                f = net_flow[i, j]
                if f == 0:
                    ax.text(j, i, '—', ha='center', va='center', fontsize=6, color='#bbb')
                    continue
                p = prob[i, j]
                fstr = f'+{abs(f):.0f}' if f > 0 else f'−{abs(f):.0f}'
                txt_color = 'k' if 0.2 < p < 0.8 else 'w'
                ax.text(j, i, f'{p:.2f}', ha='center', va='center',
                        fontsize=6.5, color=txt_color, fontweight='bold')
                ax.text(j, i + 0.34, fstr, ha='center', va='center',
                        fontsize=4.5, color='#444')

        # mean P over KEGG pairs
        valid = prob[~np.isnan(prob)]
        mean_p = valid.mean() if len(valid) > 0 else 0
        ax.set_title(f'{cond}\nP̄={mean_p:.2f}  ({len(valid)} KEGG pairs)', fontsize=8)
        ax.set_xlabel('source j', fontsize=7)
        ax.set_ylabel('target i', fontsize=7)

    plt.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02,
                 label='P(sign correct)')
    fig.suptitle(
        'Posterior sign probability P(sign(A[i,j]) == sign(F[i,j])) — 10 000 free-posterior samples\n'
        'Cell text: probability (bold) | flow weight F[i,j] (small); green>0.5=correct, red<0.5=wrong',
        fontsize=8.5)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    return fig


# ── Figure 3: Sign probability summary bar chart ──────────────────────────────

def make_figure_signprob_bar(net_flow, sign_probs, kegg_sign_probs):
    """
    For each KEGG pair (i,j), show P(correct) across all 4 conditions.
    Also show KEGG 1000p MAP reference.
    """
    # collect all KEGG pairs
    pairs_ij = [(i, j) for i in range(N_SP) for j in range(N_SP)
                if i != j and net_flow[i, j] != 0]
    pair_labels = [f'{SPECIES[i]}←{SPECIES[j]}' for i, j in pairs_ij]

    n_pairs = len(pairs_ij)
    x = np.arange(n_pairs)
    width = 0.18
    cond_colors = {'CS': '#2196F3', 'CH': '#4CAF50', 'DS': '#FF9800', 'DH': '#E91E63'}

    fig, ax = plt.subplots(figsize=(max(12, n_pairs * 0.9), 4.5))

    for ci, cond in enumerate(CONDITIONS):
        probs = [sign_probs[cond][i, j] for i, j in pairs_ij]
        ax.bar(x + (ci - 1.5) * width, probs, width, label=cond,
               color=cond_colors[cond], alpha=0.85, edgecolor='k', linewidth=0.4)

    # reference line for random (0.5)
    ax.axhline(0.5, color='k', linestyle='--', linewidth=0.8, label='random (0.5)')

    # KEGG 1000p MAP sign (binary: 0 or 1)
    kegg_marks = {}
    for cond in CONDITIONS:
        kegg_marks[cond] = [kegg_sign_probs[cond][i, j] for i, j in pairs_ij]

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('P(correct sign)', fontsize=9)
    ax.set_title('Per-pair KEGG sign probability from 10 000 free-posterior samples\n'
                 '(dashed = chance level 0.5)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')

    # add KEGG 1000p reference dots
    for ci, cond in enumerate(CONDITIONS):
        for xi, (i, j) in enumerate(pairs_ij):
            v = kegg_marks[cond][xi]
            ax.plot(xi + (ci - 1.5) * width, v, marker='*', color='k', markersize=5, zorder=5)

    ax.text(0.01, 0.96, '★ = KEGG 1000p MAP sign agreement', transform=ax.transAxes,
            fontsize=6.5, va='top')

    plt.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Building KEGG net-flow matrix ...')
    net_flow = build_net_flow()
    nz = int((net_flow != 0).sum())
    print(f'  {nz} non-zero off-diagonal entries')
    print(pd.DataFrame(net_flow, index=SPECIES, columns=SPECIES).to_string(float_format='{:.1f}'.format))

    maps, means, sign_probs = {}, {}, {}
    kegg_sign_probs = {}   # binary 0/1 from KEGG 1000p MAP
    summary = {}

    for cond in CONDITIONS:
        print(f'\nLoading {cond} ...')
        s_10k = np.load(P10K_RUNS[cond] / 'samples.npy')
        print(f'  10000p samples: {s_10k.shape}')

        means[cond] = theta_to_A(s_10k.mean(axis=0))

        with open(ULTIMATE_RUNS[cond] / 'theta_MAP.json') as f:
            d = json.load(f)
        theta_map = np.array([d[str(i)] for i in range(20)])
        maps[cond] = theta_to_A(theta_map)

        # Sign probability (Option A)
        sign_probs[cond] = sign_probability(s_10k, net_flow)

        # KEGG 1000p MAP binary sign
        s_kegg = np.load(KEGG_RUNS[cond] / 'samples.npy')
        l_kegg = np.load(KEGG_RUNS[cond] / 'logL.npy')
        A_kegg_map = theta_to_A(s_kegg[np.argmax(l_kegg)])
        kegg_prob = np.full((N_SP, N_SP), np.nan)
        for i in range(N_SP):
            for j in range(N_SP):
                if i == j or net_flow[i, j] == 0: continue
                kegg_prob[i, j] = 1.0 if np.sign(A_kegg_map[i,j]) == np.sign(net_flow[i,j]) else 0.0
        kegg_sign_probs[cond] = kegg_prob

        # Sign agreement (point estimates)
        ag_mean, tot_mean, _ = sign_agreement(means[cond], net_flow)
        ag_map,  tot_map,  _ = sign_agreement(maps[cond],  net_flow)
        ag_k, tot_k, _ = sign_agreement(A_kegg_map, net_flow)

        valid_prob = sign_probs[cond][~np.isnan(sign_probs[cond])]
        mean_signprob = float(valid_prob.mean()) if len(valid_prob) > 0 else 0.0
        n_above_50 = int((valid_prob > 0.5).sum())

        print(f'  SA Ā(10kp)={ag_mean}/{tot_mean} ({100*ag_mean/tot_mean:.0f}%)'
              f'  |  SA MAP(10kp)={ag_map}/{tot_map} ({100*ag_map/tot_map:.0f}%)'
              f'  |  SA MAP(KEGG 1k)={ag_k}/{tot_k} ({100*ag_k/tot_k:.0f}%)'
              f'  |  P̄(sign correct)={mean_signprob:.3f}  ({n_above_50}/{len(valid_prob)} pairs>0.5)')

        # Per-pair sign probs for printing
        pairs_ij = [(i, j) for i in range(N_SP) for j in range(N_SP)
                    if i != j and net_flow[i, j] != 0]
        pair_detail = {}
        for i, j in pairs_ij:
            key = f'{SPECIES[i]}<-{SPECIES[j]}'
            pair_detail[key] = {
                'flow': float(net_flow[i, j]),
                'P_correct': round(float(sign_probs[cond][i, j]), 4),
                'kegg_map_sign_ok': bool(kegg_sign_probs[cond][i, j] == 1.0),
            }

        summary[cond] = {
            'SA_mean_agree': ag_mean, 'SA_mean_total': tot_mean,
            'SA_mean_pct':   round(100*ag_mean/tot_mean, 1) if tot_mean else 0,
            'SA_map_agree':  ag_map,  'SA_map_total': tot_map,
            'SA_map_pct':    round(100*ag_map/tot_map, 1) if tot_map else 0,
            'SA_kegg_map_agree': ag_k, 'SA_kegg_map_total': tot_k,
            'SA_kegg_map_pct':   round(100*ag_k/tot_k, 1) if tot_k else 0,
            'mean_sign_prob': round(mean_signprob, 4),
            'n_pairs_above_50pct': n_above_50,
            'n_kegg_pairs': len(valid_prob),
            'logL_MAP_kegg': float(l_kegg.max()),
            'n_kegg_samples': len(s_kegg),
            'n_10k_samples':  len(s_10k),
            'per_pair': pair_detail,
        }

    # Print sign probability table
    print('\n── Sign probability table (P(correct sign) per pair) ──')
    pairs_ij = [(i, j) for i in range(N_SP) for j in range(N_SP)
                if i != j and net_flow[i, j] != 0]
    header = f"{'Pair':12s}  {'F':6s}  " + '  '.join(f'{c:6s}' for c in CONDITIONS)
    print(header)
    print('-' * len(header))
    for i, j in pairs_ij:
        label = f'{SPECIES[i]}<-{SPECIES[j]}'
        f_str = f'{net_flow[i,j]:+.1f}'
        probs_str = '  '.join(f'{sign_probs[c][i,j]:.3f}' for c in CONDITIONS)
        print(f'{label:12s}  {f_str:6s}  {probs_str}')

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump({'net_flow': net_flow.tolist(), 'species': SPECIES,
                   'conditions': summary,
                   'kegg_runs': {k: str(v) for k,v in KEGG_RUNS.items()},
                   'p10k_runs': {k: str(v) for k,v in P10K_RUNS.items()}}, f, indent=2)
    print(f'\nSaved {OUT_JSON}')

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: A-matrix comparison
    fig1 = make_figure_A(net_flow, maps, means)
    for ext in ('pdf', 'png'):
        fig1.savefig(FIG_DIR / f'fig_heine_kegg_sign_comparison.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f'Fig1 saved → {FIG_DIR}/fig_heine_kegg_sign_comparison.*')

    # Figure 2: Sign probability heatmap
    fig2 = make_figure_signprob(net_flow, sign_probs)
    for ext in ('pdf', 'png'):
        fig2.savefig(FIG_DIR / f'fig_kegg_sign_probability.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f'Fig2 saved → {FIG_DIR}/fig_kegg_sign_probability.*')

    # Figure 3: Per-pair bar chart
    fig3 = make_figure_signprob_bar(net_flow, sign_probs, kegg_sign_probs)
    for ext in ('pdf', 'png'):
        fig3.savefig(FIG_DIR / f'fig_kegg_sign_probability_bar.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print(f'Fig3 saved → {FIG_DIR}/fig_kegg_sign_probability_bar.*')


if __name__ == '__main__':
    main()
