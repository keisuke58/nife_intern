#!/usr/bin/env python3
"""
Aggregate Dieckow 16S TSVs into class-level guilds (Dieckow Fig 4a).

Guild → Class mapping:
  1  Actinobacteria   : Actinomyces, Bifidobacterium, Corynebacterium, Rothia, Slackia
  2  Bacilli          : Abiotrophia, Aerococcus, Gemella, Granulicatella,
                        Lacticaseibacillus, Lactiplantibacillus, Limosilactobacillus,
                        Streptococcus
  3  Bacteroidia      : Alloprevotella, Porphyromonas, Prevotella,
                        Prevotella_7, Tannerella
  4  Betaproteobacteria: Aggregatibacter, Cardiobacterium, Eikenella, Kingella, Neisseria
  5  Clostridia       : Anaerococcus, Catonella, Finegoldia, Johnsonella,
                        Lachnoanaerobaculum, Mogibacterium, Oribacterium, Parvimonas,
                        Peptoniphilus, Peptostreptococcus, Solobacterium, Stomatobaculum
  6  Coriobacteriia   : Atopobium, Cryptobacterium, Olsenella
  7  Fusobacteriia    : Fusobacterium, Leptotrichia
  8  Gammaproteobacteria: Haemophilus, Pseudomonas
  9  Negativicutes    : Centipeda, Dialister, Megasphaera, Selenomonas, Veillonella
  10 Flavobacteriia   : Capnocytophaga, Bergeyella/Riemerella
  11 Other            : Campylobacter, Treponema, Shuttleworthia, Acanthostaurus,
                        P5D1-392

Outputs:
  results/dieckow_otu/guild_matrix.csv      (30 samples × N guilds, %)
  results/dieckow_otu/phi_guild.npy         (10 patients × 3 weeks × N guilds, normalised)
  results/dieckow_otu/guild_summary.json
  results/dieckow_otu/guilds_timeseries.pdf + .png
  
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS_LIST

TAX_DIR = Path(__file__).parent / 'results' / 'dieckow_taxonomy'
OUT_DIR = Path(__file__).parent / 'results' / 'dieckow_otu'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = list('ABCDEFGHKL')
WEEKS    = [1, 2, 3]
SAMPLES = [f'{p}_{w}' for p in PATIENTS for w in WEEKS]

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
    'Capnocytophaga':     'Flavobacteriia',
    'Bergeyella':         'Flavobacteriia',
    'Riemerella':         'Flavobacteriia',
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

GUILD_COLORS = GUILD_COLORS_LIST
GUILD_INDEX = {g: i for i, g in enumerate(GUILD_ORDER)}


def load_guild_matrix():
    mat = np.zeros((len(SAMPLES), len(GUILD_ORDER)), dtype=float)
    present = np.zeros(len(SAMPLES), dtype=bool)
    for idx, sample in enumerate(SAMPLES):
        tsv = TAX_DIR / f'{sample}_taxonomy.tsv'
        if not tsv.exists():
            continue
        present[idx] = True
        with open(tsv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                guild = GUILD_MAP.get(row['genus'], 'Other')
                pct = float(row['percent'])
                mat[idx, GUILD_INDEX[guild]] += pct
    return mat, present


def load_guild_matrix_from_excel(excel_path, sheet='ClassAndGenus (Ave)', tax_level='class'):
    import pandas as pd

    df = pd.ExcelFile(excel_path).parse(sheet)
    sample_cols = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r'[A-Z][123]', c)]

    mat = np.zeros((len(SAMPLES), len(GUILD_ORDER)), dtype=float)
    present = np.zeros(len(SAMPLES), dtype=bool)
    if str(tax_level).lower() == 'genus':
        rows = df[df['Unnamed: 1'].astype(str).isin(set(GUILD_MAP.keys()))].copy()
        rows = rows.set_index('Unnamed: 1')
        rows = rows.loc[:, sample_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        for col in sample_cols:
            sample = f'{col[0]}_{int(col[1])}'
            if sample not in SAMPLES:
                continue
            idx = SAMPLES.index(sample)
            present[idx] = True
            total_non_other = 0.0
            for genus, val in rows[col].items():
                guild = GUILD_MAP.get(str(genus), 'Other')
                if guild == 'Other':
                    continue
                mat[idx, GUILD_INDEX[guild]] += float(val)
                total_non_other += float(val)
            mat[idx, GUILD_INDEX['Other']] = max(0.0, 100.0 - total_non_other)
    else:
        rows = df[
            df['Unnamed: 0'].apply(lambda x: isinstance(x, (int, float)) and x == x)
            & df['Unnamed: 1'].notna()
        ].copy()
        rows = rows.set_index('Unnamed: 1')
        rows = rows.loc[:, sample_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        for col in sample_cols:
            sample = f'{col[0]}_{int(col[1])}'
            if sample not in SAMPLES:
                continue
            idx = SAMPLES.index(sample)
            present[idx] = True

            total_non_other = 0.0
            for guild in GUILD_ORDER:
                if guild == 'Other':
                    continue
                if guild in rows.index:
                    val = float(rows.loc[guild, col])
                else:
                    val = 0.0
                mat[idx, GUILD_INDEX[guild]] = val
                total_non_other += val

            other = max(0.0, 100.0 - total_non_other)
            mat[idx, GUILD_INDEX['Other']] = other

    return mat, present


def load_guild_matrix_from_structure_excel(excel_path, sheet='Sheet1 (3)'):
    import pandas as pd

    df = pd.ExcelFile(excel_path).parse(sheet, header=None)

    header_row = None
    for r in range(min(80, df.shape[0] - 1)):
        row_p = df.iloc[r].astype(str).str.strip()
        row_w = df.iloc[r + 1].astype(str).str.strip()
        n_p = int(row_p.str.fullmatch(r'[A-Za-z]').sum())
        w_num = pd.to_numeric(row_w, errors='coerce')
        n_w = int((w_num.isin([1, 2, 3])).sum())
        if n_p >= 8 and n_w >= 8:
            header_row = r
            break

    if header_row is None:
        return np.zeros((len(SAMPLES), len(GUILD_ORDER)), dtype=float), np.zeros(len(SAMPLES), dtype=bool)

    header_patients = df.iloc[header_row].astype(str).tolist()
    header_weeks = df.iloc[header_row + 1].astype(str).tolist()

    sample_by_col = {}
    for j, (p, w) in enumerate(zip(header_patients, header_weeks)):
        p = p.strip()
        w = w.strip()
        w_num = pd.to_numeric(w, errors='coerce')
        if not (len(p) == 1 and p.isalpha() and pd.notna(w_num)):
            continue
        w_i = int(w_num)
        if w_i not in (1, 2, 3):
            continue
        sample_by_col[j] = f'{p.upper()}_{w_i}'

    row_labels = df.iloc[:, 0].astype(str).str.strip()
    class_block = df.loc[row_labels.isin([g for g in GUILD_ORDER if g != 'Other'])].copy()
    class_block = class_block.set_index(0)
    class_block = class_block[~class_block.index.duplicated(keep='first')]

    mat = np.zeros((len(SAMPLES), len(GUILD_ORDER)), dtype=float)
    present = np.zeros(len(SAMPLES), dtype=bool)

    for j, sample in sample_by_col.items():
        if sample not in SAMPLES:
            continue
        idx = SAMPLES.index(sample)
        present[idx] = True
        total_non_other = 0.0
        for guild in GUILD_ORDER:
            if guild == 'Other':
                continue
            if guild in class_block.index:
                v = pd.to_numeric(class_block.at[guild, j], errors='coerce')
                val = 0.0 if pd.isna(v) else float(v)
            else:
                val = 0.0
            mat[idx, GUILD_INDEX[guild]] = val
            total_non_other += val
        mat[idx, GUILD_INDEX['Other']] = max(0.0, 100.0 - total_non_other)

    return mat, present


def build_phi_guild(mat):
    phi = np.zeros((len(PATIENTS), len(WEEKS), len(GUILD_ORDER)))
    for i, p in enumerate(PATIENTS):
        for j, w in enumerate(WEEKS):
            sample = f'{p}_{w}'
            idx = SAMPLES.index(sample)
            row = mat[idx]
            total = row.sum()
            phi[i, j] = row / total if total > 0 else row
    return phi


def plot_timeseries(phi, out_stem='guilds_timeseries'):
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
    fig.suptitle('Dieckow 2024 — class-level composition (Dieckow Fig 4a classes)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'{out_stem}.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f'Saved: {out}')
    plt.close(fig)


def _load_biofilm_params_from_structure_summary(excel_path, sheet='Sheet1'):
    import pandas as pd

    df = pd.ExcelFile(excel_path).parse(sheet, header=None)
    s = df.astype(str).apply(lambda col: col.str.strip())

    token = re.compile(r'^[A-L]_[123]$')
    header_row = None
    for r in range(min(20, df.shape[0])):
        n = int(s.iloc[r].apply(lambda x: bool(token.match(x))).sum())
        if n >= 20:
            header_row = r
            break

    if header_row is None:
        raise ValueError(f'Could not find sample header row in {excel_path} sheet={sheet}')

    sample_cols = {}
    for j, v in enumerate(s.iloc[header_row].tolist()):
        if token.match(v):
            sample_cols[j] = v

    def find_row(label):
        for r in range(df.shape[0]):
            if str(s.iat[r, 0]) == label:
                return r
        return None

    r_vol = find_row('VolumeTotal')
    r_live = find_row('PerLive')
    if r_vol is None or r_live is None:
        raise ValueError(f'Missing VolumeTotal/PerLive rows in {excel_path} sheet={sheet}')

    rows = []
    for j, sample in sample_cols.items():
        p, w = sample.split('_', 1)
        w = int(w)
        v_vol = pd.to_numeric(df.iat[r_vol, j], errors='coerce')
        v_live = pd.to_numeric(df.iat[r_live, j], errors='coerce')
        rows.append({
            'patient': p,
            'week': w,
            'volume_total_raw': None if pd.isna(v_vol) else float(v_vol),
            'per_live': None if pd.isna(v_live) else float(v_live),
        })

    out = pd.DataFrame(rows)
    return out


def _plot_supp_fig1_cd(params_df, out_dir):
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = list('ABCDEFGHIJKL')
    weeks = [1, 2, 3]
    week_colors = {1: '#3b4cc0', 2: '#d7191c', 3: '#4dac26'}

    def grouped_points(y_col, y_label, out_stem, y_lim=None, scale=1.0):
        fig, ax = plt.subplots(figsize=(9.2, 3.2))
        width = 0.22
        offsets = {1: -width, 2: 0.0, 3: width}

        xs = []
        ys = []
        cols = []
        for i, p in enumerate(patients):
            for w in weeks:
                sub = params_df[(params_df['patient'] == p) & (params_df['week'] == w)]
                if len(sub) != 1:
                    val = np.nan
                else:
                    v = sub.iloc[0][y_col]
                    val = np.nan if v is None else float(v) * scale
                if np.isnan(val):
                    continue
                xs.append(i + offsets[w])
                ys.append(val)
                cols.append(week_colors[w])

        ax.scatter(xs, ys, s=36, c=cols, marker='s', edgecolors='black', linewidths=0.8, zorder=3)

        ax.set_xticks(range(len(patients)))
        ax.set_xticklabels(patients)
        ax.set_xlabel('Patient')
        ax.set_ylabel(y_label)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        from matplotlib.patches import Patch
        legend = [Patch(facecolor=week_colors[w], edgecolor='black', label=str(w)) for w in weeks]
        ax.legend(handles=legend, title='Weeks', frameon=True, fancybox=False, edgecolor='black',
                  loc='upper right', fontsize=8, title_fontsize=8)

        for ext in ('pdf', 'png'):
            out = out_dir / f'{out_stem}.{ext}'
            fig.savefig(out, bbox_inches='tight', dpi=300)
        plt.close(fig)

    grouped_points(
        y_col='volume_total_raw',
        y_label=r'Biofilm volume (µm$^3$ × 10$^6$)',
        out_stem='supp_fig1_c_biofilm_volume',
        y_lim=(0, 10),
        scale=0.1,
    )
    grouped_points(
        y_col='per_live',
        y_label='Percentage of volume representing\nnon-permeable cells',
        out_stem='supp_fig1_d_percent_live',
        y_lim=(0, 100),
        scale=1.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--excel', default=None, help='Optional Excel source workbook.')
    ap.add_argument('--sheet', default='ClassAndGenus (Ave)', help='Sheet name for class composition (Excel).')
    ap.add_argument('--excel-tax-level', default='class', choices=['class', 'genus'], help='When reading ClassAndGenus Excel sheet: aggregate by class rows or genus rows.')
    ap.add_argument('--supp-fig1-cd', action='store_true', help='Reproduce Supplementary Fig.1 c,d (biofilm parameters).')
    args = ap.parse_args()

    tag = None
    if args.excel:
        if str(args.sheet).strip().lower().startswith('sheet1'):
            tag = 'excel_structure'
        else:
            tag = f'excel_{args.excel_tax_level}'

    if args.supp_fig1_cd:
        if not args.excel:
            raise SystemExit('--supp-fig1-cd requires --excel pointing to Abutment_Structure vs composition.xlsx')
        params = _load_biofilm_params_from_structure_summary(args.excel, sheet=args.sheet)
        _plot_supp_fig1_cd(params, OUT_DIR / 'dieckow_supplementary')
        return

    print('Building guild matrix...')
    if args.excel:
        if str(args.sheet).strip().lower().startswith('sheet1'):
            mat, present = load_guild_matrix_from_structure_excel(args.excel, sheet=args.sheet)
        else:
            mat, present = load_guild_matrix_from_excel(args.excel, sheet=args.sheet, tax_level=args.excel_tax_level)
    else:
        mat, present = load_guild_matrix()
    n_present = int(present.sum())
    print(f'  Shape: ({n_present}, {len(GUILD_ORDER)})')
    if args.excel and (~present).any():
        try:
            import pandas as pd

            summary = pd.ExcelFile(args.excel).parse('Summary')
            summary['Patient'] = summary['Patient'].astype(str)
            summary['Week_num'] = pd.to_numeric(summary['Week'], errors='coerce')
            missing = [SAMPLES[i] for i in range(len(SAMPLES)) if not present[i]]
            print('\nMissing sample IDs in the selected composition table:')
            for s in missing:
                p, w = s.split('_', 1)
                w = int(w)
                m = (summary['Patient'] == p) & (summary['Week_num'] == w)
                if m.any():
                    r = summary.loc[m].iloc[0]
                    fs = r.get('Final set')
                    cont = r.get('Cont%')
                    good = r.get('Good')
                    tot = r.get('Tot')
                    print(f'  {s}  Final set={fs}  Cont%={cont}  Good={good}  Tot={tot}')
                else:
                    print(f'  {s}  (no row in Summary sheet)')
        except Exception:
            pass
    print('\nMean guild % across all samples:')
    means = mat[present].mean(axis=0) if n_present > 0 else np.zeros(len(GUILD_ORDER))
    for g, m in zip(GUILD_ORDER, means):
        print(f'  {g:22s}: {m:.1f}%')

    coverage = mat.sum(axis=1)
    print(f'\nClassified coverage: {coverage.mean():.1f}% mean '
          f'(range {coverage.min():.1f}–{coverage.max():.1f}%)')

    out_guild_csv = OUT_DIR / (f'guild_matrix_{tag}.csv' if tag else 'guild_matrix.csv')
    with open(out_guild_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample'] + GUILD_ORDER)
        for sample, row in zip(SAMPLES, mat):
            writer.writerow([sample] + [f'{x:.6g}' for x in row])
    print(f'\nSaved: {out_guild_csv}')

    denom = mat.sum(axis=1, keepdims=True)
    phi_samples = np.divide(mat, denom, out=np.zeros_like(mat), where=denom > 0)
    conet_abundance_path = OUT_DIR / (f'conet_guild_abundance_{tag}.tsv' if tag else 'conet_guild_abundance.tsv')
    with open(conet_abundance_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['guild'] + SAMPLES)
        for g in GUILD_ORDER:
            k = GUILD_INDEX[g]
            writer.writerow([g] + [f'{x:.10g}' for x in phi_samples[:, k]])
    print(f'Saved: {conet_abundance_path}')

    conet_meta_path = OUT_DIR / (f'conet_sample_metadata_{tag}.tsv' if tag else 'conet_sample_metadata.tsv')
    with open(conet_meta_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['sample', 'patient', 'week'])
        for p in PATIENTS:
            for w in WEEKS:
                writer.writerow([f'{p}_{w}', p, w])
    print(f'Saved: {conet_meta_path}')

    phi = build_phi_guild(mat)
    phi_path = OUT_DIR / (f'phi_guild_{tag}.npy' if tag else 'phi_guild.npy')
    np.save(phi_path, phi)
    print(f'Saved: {phi_path}  shape={phi.shape}')

    records = []
    for i, p in enumerate(PATIENTS):
        for j, w in enumerate(WEEKS):
            rec = {'patient': p, 'week': w}
            for k, g in enumerate(GUILD_ORDER):
                rec[g] = float(phi[i, j, k])
            records.append(rec)
    summary_path = OUT_DIR / (f'guild_summary_{tag}.json' if tag else 'guild_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'guilds': GUILD_ORDER, 'guild_map': GUILD_MAP,
                   'samples': records}, f, indent=2)
    print(f'Saved: {summary_path}')

    print('\nPlotting timeseries...')
    plot_timeseries(phi, out_stem=(f'guilds_timeseries_{tag}' if tag else 'guilds_timeseries'))


if __name__ == '__main__':
    main()
