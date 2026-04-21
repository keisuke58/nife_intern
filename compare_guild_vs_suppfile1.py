#!/usr/bin/env python3
"""
Compare inferred guild A matrix with Dieckow Supp File 1 predicted interactions.

Supp File 1: species-level PRODUCES/USES/IS_INHIBITED_BY
→ aggregate to guild level → guild-guild metabolite flow matrix
→ compare sign with inferred A matrix
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SUPPFILE = Path('/home/nishioka/IKM_Hiwi/nife/Szafranski_Published_Work/'
               'Szafranski_Published_Work/public_data/Dieckow/'
               'Supplementary_File_1_microbe_metabolite_enzyme_interactions.tsv')
FIT_JSON = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/fit_guild.json')
OUT_DIR  = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_otu')

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
N_G = len(GUILD_ORDER)

# Genus → Guild (same as aggregate_dieckow_guilds.py)
GENUS_GUILD = {
    'Actinomyces':'Actinobacteria','Bifidobacterium':'Actinobacteria',
    'Rothia':'Actinobacteria','Schaalia':'Actinobacteria',
    'Streptococcus':'Bacilli','Gemella':'Bacilli','Granulicatella':'Bacilli',
    'Abiotrophia':'Bacilli','Lactiplantibacillus':'Bacilli',
    'Prevotella':'Bacteroidia','Porphyromonas':'Bacteroidia',
    'Tannerella':'Bacteroidia','Alloprevotella':'Bacteroidia',
    'Capnocytophaga':'Bacteroidia',
    'Neisseria':'Betaproteobacteria','Eikenella':'Betaproteobacteria',
    'Aggregatibacter':'Betaproteobacteria',
    'Fusobacterium':'Fusobacteriia','Leptotrichia':'Fusobacteriia',
    'Haemophilus':'Gammaproteobacteria',
    'Veillonella':'Negativicutes','Selenomonas':'Negativicutes',
    'Megasphaera':'Negativicutes','Dialister':'Negativicutes',
    'Parvimonas':'Clostridia','Mogibacterium':'Clostridia',
    'Peptostreptococcus':'Clostridia','Catonella':'Clostridia',
    'Atopobium':'Coriobacteriia','Olsenella':'Coriobacteriia',
}

def genus_from_taxon(taxon):
    return taxon.split()[0] if taxon else None


def build_metabolite_flow(df):
    """
    For each metabolite: find producers and consumers/inhibited.
    A producer guild → consumer guild = positive predicted interaction.
    A producer guild → inhibited guild = negative predicted interaction.
    Returns (N_G × N_G) count matrices: pos_flow, neg_flow.
    """
    gi = {g: i for i, g in enumerate(GUILD_ORDER)}

    pos_flow = np.zeros((N_G, N_G), dtype=int)   # [target, source] positive
    neg_flow = np.zeros((N_G, N_G), dtype=int)   # [target, source] negative

    mets = df['OBJECT'].unique()
    for met in mets:
        met_df = df[df['OBJECT'] == met]
        producers = set()
        consumers = set()
        inhibited  = set()
        for _, row in met_df.iterrows():
            g = GENUS_GUILD.get(genus_from_taxon(str(row['TAXON'])))
            if g is None:
                continue
            if row['RELATIONSHIP'] == 'PRODUCES':
                producers.add(g)
            elif row['RELATIONSHIP'] == 'USES':
                consumers.add(g)
            elif row['RELATIONSHIP'] == 'IS_INHIBITED_BY':
                inhibited.add(g)
        # Positive: producer promotes consumer
        for src in producers:
            for tgt in consumers:
                if src != tgt:
                    pos_flow[gi[tgt], gi[src]] += 1
        # Negative: producer inhibits target
        for src in producers:
            for tgt in inhibited:
                if src != tgt:
                    neg_flow[gi[tgt], gi[src]] += 1
    return pos_flow, neg_flow


def main():
    df = pd.read_csv(SUPPFILE, sep='\t')
    print(f'Supp File 1: {len(df)} rows, {df["RELATIONSHIP"].unique()} relationships')

    pos_flow, neg_flow = build_metabolite_flow(df)
    net_flow = pos_flow - neg_flow   # positive = net predicted positive effect

    with open(FIT_JSON) as f:
        fit = json.load(f)
    A = np.array(fit['A'])

    # ── Figure: 3-panel comparison ────────────────────────────────────────────
    short = [g[:6] for g in GUILD_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmax_A = max(abs(A[~np.eye(N_G, dtype=bool)]).max(), 0.01)

    # Panel 1: inferred A
    ax = axes[0]
    im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax_A, vmax=vmax_A, aspect='auto')
    ax.set_xticks(range(N_G)); ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N_G)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title('Inferred A matrix\n(JAX Adam, RMSE=0.055)', fontsize=9)
    ax.set_xlabel('Source'); ax.set_ylabel('Target')
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Panel 2: Supp File 1 net predicted flow
    vmax_f = max(abs(net_flow).max(), 1)
    ax = axes[1]
    im2 = ax.imshow(net_flow, cmap='RdBu_r', vmin=-vmax_f, vmax=vmax_f, aspect='auto')
    ax.set_xticks(range(N_G)); ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N_G)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title('Supp File 1 predicted\nnet metabolite flow (pos–neg)', fontsize=9)
    ax.set_xlabel('Source (producer)')
    plt.colorbar(im2, ax=ax, shrink=0.7)
    for i in range(N_G):
        for j in range(N_G):
            if net_flow[i,j] != 0:
                ax.text(j, i, str(net_flow[i,j]), ha='center', va='center',
                        fontsize=6, color='white' if abs(net_flow[i,j])>vmax_f*0.5 else 'black')

    # Panel 3: sign agreement
    ax = axes[2]
    A_sign  = np.sign(A)
    SF_sign = np.sign(net_flow.astype(float))
    agreement = np.zeros((N_G, N_G))
    mask = (SF_sign != 0) & (~np.eye(N_G, dtype=bool))
    agreement[mask & (A_sign == SF_sign)] =  1   # agree
    agreement[mask & (A_sign != SF_sign)] = -1   # disagree
    # agreement == 0 → no prediction from Supp File 1

    cmap3 = matplotlib.colors.ListedColormap(['#d62728', '#cccccc', '#2ca02c'])
    im3 = ax.imshow(agreement, cmap=cmap3, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(N_G)); ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N_G)); ax.set_yticklabels(short, fontsize=7)
    ax.set_title('Sign agreement\n(green=agree, red=disagree, grey=no SF1 pred)', fontsize=9)
    ax.set_xlabel('Source')

    n_agree = int((agreement == 1).sum())
    n_dis   = int((agreement == -1).sum())
    ax.text(0.5, -0.18, f'Agree: {n_agree}  Disagree: {n_dis}  (of {mask.sum()} SF1 predictions)',
            transform=ax.transAxes, ha='center', fontsize=8)

    fig.suptitle('Guild A matrix vs Dieckow Supp File 1 predicted interactions', fontsize=11)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'guild_vs_suppfile1.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f'Saved: {out}')
    plt.close(fig)

    # Text summary
    print('\nSign agreement summary:')
    for i in range(N_G):
        for j in range(N_G):
            if agreement[i,j] != 0:
                status = 'AGREE' if agreement[i,j] == 1 else 'DISAGREE'
                print(f'  {GUILD_ORDER[j][:15]:15s} → {GUILD_ORDER[i][:15]:15s}  '
                      f'A={A[i,j]:+.3f}  SF1_net={net_flow[i,j]:+d}  {status}')


if __name__ == '__main__':
    main()
