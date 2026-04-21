#!/usr/bin/env python3
"""
Aggregate 30 Dieckow taxonomy TSVs into:
  1. Full OTU matrix (samples × genera) → results/dieckow_otu/otu_matrix.csv
  2. 5-species phi matrix (10 patients × 3 weeks × 5 species) → results/dieckow_otu/phi_obs_raw.npy
  3. Summary JSON → results/dieckow_otu/summary.json

5-species mapping (Hamilton ODE labels):
  So = Streptococcus
  An = Actinomyces
  Vd = Veillonella
  Fn = Fusobacterium
  Pg = Porphyromonas
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

TAX_DIR  = Path(__file__).parent / 'results' / 'dieckow_taxonomy'
OUT_DIR  = Path(__file__).parent / 'results' / 'dieckow_otu'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = list('ABCDEFGHKL')   # 10 patients
WEEKS    = [1, 2, 3]
LABELS   = ['So', 'An', 'Vd', 'Fn', 'Pg']
GENUS_MAP = {
    'Streptococcus': 'So',
    'Actinomyces':   'An',
    'Veillonella':   'Vd',
    'Fusobacterium': 'Fn',
    'Porphyromonas': 'Pg',
}


def load_all_tsv():
    frames = []
    for p in PATIENTS:
        for w in WEEKS:
            sample = f'{p}_{w}'
            tsv = TAX_DIR / f'{sample}_taxonomy.tsv'
            if not tsv.exists():
                print(f'MISSING: {tsv}')
                continue
            df = pd.read_csv(tsv, sep='\t')
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_otu_matrix(df):
    pivot = df.pivot_table(index='sample', columns='genus', values='percent', fill_value=0.0)
    # Reindex so rows follow patient/week order
    row_order = [f'{p}_{w}' for p in PATIENTS for w in WEEKS]
    pivot = pivot.reindex(row_order, fill_value=0.0)
    return pivot


def build_phi_obs(pivot):
    """Return (10, 3, 5) array: patients × weeks × species (relative abundance, summing to ≤1)."""
    phi = np.zeros((len(PATIENTS), len(WEEKS), len(LABELS)))
    for i, p in enumerate(PATIENTS):
        for j, w in enumerate(WEEKS):
            sample = f'{p}_{w}'
            if sample not in pivot.index:
                continue
            row = pivot.loc[sample]
            total = 0.0
            for genus, sp in GENUS_MAP.items():
                val = row.get(genus, 0.0) / 100.0   # percent → fraction
                phi[i, j, LABELS.index(sp)] = val
                total += val
            # Normalize so the 5 focal species sum to 1
            if total > 0:
                phi[i, j] /= total
    return phi


def main():
    print('Loading TSVs...')
    df = load_all_tsv()
    print(f'  {df.shape[0]} rows, {df["sample"].nunique()} samples, {df["genus"].nunique()} genera')

    pivot = build_otu_matrix(df)
    print(f'  OTU matrix: {pivot.shape}')

    phi = build_phi_obs(pivot)
    print(f'  phi_obs_raw shape: {phi.shape}  (patients × weeks × species)')

    # Save
    otu_csv = OUT_DIR / 'otu_matrix.csv'
    pivot.to_csv(otu_csv)
    print(f'Saved: {otu_csv}')

    phi_npy = OUT_DIR / 'phi_obs_raw.npy'
    np.save(phi_npy, phi)
    print(f'Saved: {phi_npy}')

    # Summary: per-sample focal species fractions
    records = []
    for i, p in enumerate(PATIENTS):
        for j, w in enumerate(WEEKS):
            rec = {'patient': p, 'week': w, 'sample': f'{p}_{w}'}
            for k, sp in enumerate(LABELS):
                rec[sp] = float(phi[i, j, k])
            records.append(rec)
    summary = {
        'patients': PATIENTS,
        'weeks': WEEKS,
        'labels': LABELS,
        'genus_map': GENUS_MAP,
        'samples': records,
    }
    summary_json = OUT_DIR / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {summary_json}')

    # Print quick overview
    print('\nPer-patient mean φ (averaged over weeks):')
    phi_mean = phi.mean(axis=1)   # (10, 5)
    print(f"  {'Patient':8s} " + '  '.join(f'{sp:>6s}' for sp in LABELS))
    for i, p in enumerate(PATIENTS):
        vals = '  '.join(f'{phi_mean[i,k]:.3f}' for k in range(len(LABELS)))
        print(f'  {p:8s} {vals}')


if __name__ == '__main__':
    main()
