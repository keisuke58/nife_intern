#!/usr/bin/env python3
"""
assign_ct_labels.py — Approximate CT label assignment for Szafranski 127 samples
using 16S composition rules derived from paper's CT definitions.

CT I  : PIH-linked, Bacilli (Streptococcus/So) dominant
CT II : PIM-linked, Betaproteobacteria (Neisseria) dominant — not in 5-sp, assigned by exclusion
CT III: PIM/PI, Bacteroidia (Prevotella) + Fusobacteriia (Fn)
CT IV : PI, high Bacteroidia (Tannerella/Pg) + Fusobacteriia (Fn)

NOTE: CT assignment is approximate because CT is defined on RNA activity profiles
(metatranscriptomics), not DNA composition. Neisseria (CT II marker) is absent from
our 5-species set, so CT II is assigned by exclusion.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

GMM_CSV = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_gmm/gmm_reassigned.csv')
OUT_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/ct_label_analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def assign_ct(row):
    """Rule-based CT from 5-species composition + diagnosis."""
    So, Fn, Pg = row['phi0_So'], row['phi0_Fn'], row['phi0_Pg']

    # CT IV: Pg dominant + Fn elevated (Bacteroidia/Tannerella/Pg + Fusobacteriia)
    if Pg >= 0.15 and Fn >= 0.10:
        return 'CT_IV'

    # CT I: Streptococcus (Bacilli) dominant
    if So >= 0.45:
        return 'CT_I'

    # CT III: Fusobacteriia dominant, Pg moderate (Bacteroidia Prevotella + Fuso)
    if Fn >= 0.20:
        return 'CT_III'

    # CT II: Neisseria proxy — moderate composition, can't distinguish with 5 species
    return 'CT_II'


def main():
    df = pd.read_csv(GMM_CSV)
    print(f'Loaded {len(df)} samples')

    df['ct_assigned'] = df.apply(assign_ct, axis=1)

    # Print summary
    print('\n── CT assignment summary ──')
    ct_counts = df['ct_assigned'].value_counts().sort_index()
    for ct, n in ct_counts.items():
        sub = df[df.ct_assigned == ct]
        diag_dist = sub['diagnosis'].value_counts().to_dict()
        print(f'  {ct}: n={n}  diagnosis={diag_dist}')

    print('\n── CT vs diagnosis cross-tab ──')
    ct_diag = pd.crosstab(df['ct_assigned'], df['diagnosis'])
    print(ct_diag)

    print('\n── CT vs invivo_attr cross-tab ──')
    ct_attr = pd.crosstab(df['ct_assigned'], df['invivo_attr'])
    print(ct_attr)

    # Save updated CSV
    df.to_csv(GMM_CSV.parent / 'gmm_reassigned_ct.csv', index=False)
    print(f'\nSaved: {GMM_CSV.parent}/gmm_reassigned_ct.csv')

    # Plot 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def plot_cm(ax, ct_tab, title, cmap='Blues'):
        im = ax.imshow(ct_tab.values, cmap=cmap, aspect='auto')
        ax.set_xticks(range(ct_tab.shape[1]))
        ax.set_xticklabels(ct_tab.columns, rotation=30, ha='right')
        ax.set_yticks(range(ct_tab.shape[0]))
        ax.set_yticklabels(ct_tab.index)
        for i in range(ct_tab.shape[0]):
            for j in range(ct_tab.shape[1]):
                v = ct_tab.values[i, j]
                ax.text(j, i, str(v), ha='center', va='center',
                        fontsize=11, color='white' if v > ct_tab.values.max() * 0.5 else 'black')
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_cm(axes[0], ct_diag, 'CT (composition-based) vs Diagnosis\n(CT I=So-dom, IV=Pg+Fn-dom)')
    plot_cm(axes[1], ct_attr, 'CT (composition-based) vs in-vivo attractor\n(GMM re-assignment)')

    plt.suptitle('Approximate CT assignment from 16S composition\n'
                 '(CT II = Neisseria-proxy by exclusion; RNA-based CT not available)',
                 fontsize=10, color='gray')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'ct_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/ct_confusion_matrix.png')

    # Composition profile per CT
    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))
    sp_cols = ['phi0_So', 'phi0_An', 'phi0_Vd', 'phi0_Fn', 'phi0_Pg']
    sp_labels = ['So', 'An', 'Vd', 'Fn', 'Pg']
    colors_sp = ['#1565C0', '#558B2F', '#EF6C00', '#6A1B9A', '#B71C1C']

    for ax, ct in zip(axes2, ['CT_I', 'CT_II', 'CT_III', 'CT_IV']):
        sub = df[df.ct_assigned == ct]
        means = sub[sp_cols].mean().values
        stds = sub[sp_cols].std().values
        bars = ax.bar(sp_labels, means, color=colors_sp, alpha=0.8, yerr=stds, capsize=3)
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{ct}  n={len(sub)}\n({", ".join(f"{k}:{v}" for k, v in sub.diagnosis.value_counts().items())})',
                     fontsize=9)
        ax.set_ylabel('Relative abundance')

    plt.suptitle('Mean ± SD composition per CT (composition-based assignment)', fontsize=11)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'ct_composition_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/ct_composition_profiles.png')

    # Compute DI per CT (Shannon-based)
    phi_cols = ['phi0_So', 'phi0_An', 'phi0_Vd', 'phi0_Fn', 'phi0_Pg']
    phi = df[phi_cols].values
    phi_safe = np.clip(phi, 1e-9, None)
    phi_safe = phi_safe / phi_safe.sum(axis=1, keepdims=True)
    H = -(phi_safe * np.log(phi_safe)).sum(axis=1)
    H_max = np.log(5)
    df['DI'] = 1 - H / H_max

    print('\n── DI per CT ──')
    for ct in ['CT_I', 'CT_II', 'CT_III', 'CT_IV']:
        sub = df[df.ct_assigned == ct]
        print(f'  {ct}: DI mean={sub.DI.mean():.3f} ± {sub.DI.std():.3f}')

    return df


if __name__ == '__main__':
    main()
