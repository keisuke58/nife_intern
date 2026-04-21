#!/usr/bin/env python3
"""
Aggregate Dieckow 16S TSVs into 10 bacterial-class guilds (Dieckow Fig 4a).

Guild → Class mapping:
  1  Actinobacteria   : Actinomyces, Bifidobacterium, Corynebacterium, Rothia, Slackia
  2  Bacilli          : Abiotrophia, Aerococcus, Gemella, Granulicatella,
                        Lacticaseibacillus, Lactiplantibacillus, Limosilactobacillus,
                        Streptococcus
  3  Bacteroidia      : Alloprevotella, Capnocytophaga, Porphyromonas, Prevotella,
                        Prevotella_7, Tannerella
  4  Betaproteobacteria: Aggregatibacter, Cardiobacterium, Eikenella, Kingella, Neisseria
  5  Clostridia       : Anaerococcus, Catonella, Finegoldia, Johnsonella,
                        Lachnoanaerobaculum, Mogibacterium, Oribacterium, Parvimonas,
                        Peptoniphilus, Peptostreptococcus, Solobacterium, Stomatobaculum
  6  Coriobacteriia   : Atopobium, Cryptobacterium, Olsenella
  7  Fusobacteriia    : Fusobacterium, Leptotrichia
  8  Gammaproteobacteria: Haemophilus, Pseudomonas
  9  Negativicutes    : Centipeda, Dialister, Megasphaera, Selenomonas, Veillonella
  10 Spirochaetes/Other: Campylobacter, Treponema, Shuttleworthia, Acanthostaurus,
                         P5D1-392

Outputs:
  results/dieckow_otu/guild_matrix.csv      (30 samples × 10 guilds, %)
  results/dieckow_otu/phi_guild.npy         (10 patients × 3 weeks × 10 guilds, normalised)
  results/dieckow_otu/guild_summary.json
  results/dieckow_otu/guilds_timeseries.pdf + .png
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

TAX_DIR = Path(__file__).parent / 'results' / 'dieckow_taxonomy'
OUT_DIR = Path(__file__).parent / 'results' / 'dieckow_otu'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = list('ABCDEFGHKL')
WEEKS    = [1, 2, 3]

# Dieckow Fig 4a class-level guilds
GUILD_MAP = {
    'Actinomyces':        'Actinobacteria',
    'Bifidobacterium':    'Actinobacteria',
    'Corynebacterium':    'Actinobacteria',
    'Rothia':             'Actinobacteria',
    'Slackia':            'Actinobacteria',
    'Abiotrophia':        'Bacilli',
    'Aerococcus':         'Bacilli',
    'Gemella':            'Bacilli',
    'Granulicatella':     'Bacilli',
    'Lacticaseibacillus': 'Bacilli',
    'Lactiplantibacillus':'Bacilli',
    'Limosilactobacillus':'Bacilli',
    'Streptococcus':      'Bacilli',
    'Alloprevotella':     'Bacteroidia',
    'Capnocytophaga':     'Bacteroidia',
    'Porphyromonas':      'Bacteroidia',
    'Prevotella':         'Bacteroidia',
    'Prevotella_7':       'Bacteroidia',
    'Tannerella':         'Bacteroidia',
    'Aggregatibacter':    'Betaproteobacteria',
    'Cardiobacterium':    'Betaproteobacteria',
    'Eikenella':          'Betaproteobacteria',
    'Kingella':           'Betaproteobacteria',
    'Neisseria':          'Betaproteobacteria',
    'Anaerococcus':       'Clostridia',
    'Catonella':          'Clostridia',
    'Finegoldia':         'Clostridia',
    'Johnsonella':        'Clostridia',
    'Lachnoanaerobaculum':'Clostridia',
    'Mogibacterium':      'Clostridia',
    'Oribacterium':       'Clostridia',
    'Parvimonas':         'Clostridia',
    'Peptoniphilus':      'Clostridia',
    'Peptostreptococcus': 'Clostridia',
    'Solobacterium':      'Clostridia',
    'Stomatobaculum':     'Clostridia',
    'Atopobium':          'Coriobacteriia',
    'Cryptobacterium':    'Coriobacteriia',
    'Olsenella':          'Coriobacteriia',
    'Fusobacterium':      'Fusobacteriia',
    'Leptotrichia':       'Fusobacteriia',
    'Haemophilus':        'Gammaproteobacteria',
    'Pseudomonas':        'Gammaproteobacteria',
    'Centipeda':          'Negativicutes',
    'Dialister':          'Negativicutes',
    'Megasphaera':        'Negativicutes',
    'Selenomonas':        'Negativicutes',
    'Veillonella':        'Negativicutes',
    'Campylobacter':      'Other',
    'Treponema':          'Other',
    'Shuttleworthia':     'Other',
    'Acanthostaurus':     'Other',
    'P5D1-392':           'Other',
}

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]

GUILD_COLORS = [
    '#8B4513', '#2ca02c', '#ff7f0e', '#1f77b4',
    '#17becf', '#bcbd22', '#ffbb78', '#aec7e8',
    '#98df8a', '#999999',
]


def load_guild_matrix():
    rows = {}
    for p in PATIENTS:
        for w in WEEKS:
            sample = f'{p}_{w}'
            tsv = TAX_DIR / f'{sample}_taxonomy.tsv'
            if not tsv.exists():
                continue
            df = pd.read_csv(tsv, sep='\t')
            guild_pct = {g: 0.0 for g in GUILD_ORDER}
            for _, row in df.iterrows():
                guild = GUILD_MAP.get(row['genus'], 'Other')
                guild_pct[guild] += row['percent']
            rows[sample] = guild_pct
    mat = pd.DataFrame(rows).T.reindex(
        [f'{p}_{w}' for p in PATIENTS for w in WEEKS]
    ).fillna(0.0)
    return mat[GUILD_ORDER]


def build_phi_guild(mat):
    phi = np.zeros((len(PATIENTS), len(WEEKS), len(GUILD_ORDER)))
    for i, p in enumerate(PATIENTS):
        for j, w in enumerate(WEEKS):
            sample = f'{p}_{w}'
            if sample not in mat.index:
                continue
            row = mat.loc[sample].values
            total = row.sum()
            phi[i, j] = row / total if total > 0 else row
    return phi


def plot_timeseries(phi):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
    axes = axes.ravel()
    for i, (p, ax) in enumerate(zip(PATIENTS, axes)):
        bottom = np.zeros(3)
        for k, (guild, col) in enumerate(zip(GUILD_ORDER, GUILD_COLORS)):
            vals = phi[i, :, k]
            ax.bar(WEEKS, vals, bottom=bottom, color=col, width=0.6,
                   label=guild if i == 0 else None)
            bottom += vals
        ax.set_title(f'Patient {p}', fontsize=10, fontweight='bold')
        ax.set_xticks(WEEKS)
        ax.set_xticklabels(['W1', 'W2', 'W3'], fontsize=8)
        ax.set_ylim(0, 1.05)
        if i % 5 == 0:
            ax.set_ylabel('Relative abundance', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.legend(loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle('Dieckow 2024 — 10-guild composition (Dieckow Fig 4a classes)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'guilds_timeseries.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f'Saved: {out}')
    plt.close(fig)


def main():
    print('Building guild matrix...')
    mat = load_guild_matrix()
    print(f'  Shape: {mat.shape}')
    print('\nMean guild % across all samples:')
    for g in GUILD_ORDER:
        print(f'  {g:22s}: {mat[g].mean():.1f}%')

    coverage = mat[GUILD_ORDER].sum(axis=1)
    print(f'\nClassified coverage: {coverage.mean():.1f}% mean '
          f'(range {coverage.min():.1f}–{coverage.max():.1f}%)')

    mat.to_csv(OUT_DIR / 'guild_matrix.csv')
    print(f'\nSaved: {OUT_DIR}/guild_matrix.csv')

    phi = build_phi_guild(mat)
    np.save(OUT_DIR / 'phi_guild.npy', phi)
    print(f'Saved: {OUT_DIR}/phi_guild.npy  shape={phi.shape}')

    records = []
    for i, p in enumerate(PATIENTS):
        for j, w in enumerate(WEEKS):
            rec = {'patient': p, 'week': w}
            for k, g in enumerate(GUILD_ORDER):
                rec[g] = float(phi[i, j, k])
            records.append(rec)
    with open(OUT_DIR / 'guild_summary.json', 'w') as f:
        json.dump({'guilds': GUILD_ORDER, 'guild_map': GUILD_MAP,
                   'samples': records}, f, indent=2)
    print(f'Saved: {OUT_DIR}/guild_summary.json')

    print('\nPlotting timeseries...')
    plot_timeseries(phi)


if __name__ == '__main__':
    main()
