#!/usr/bin/env python3
"""
joshi_gdi_expanded_check.py
  - Load minimap2 genus profiles from results/joshi_16s_profiles/
  - Match samples to 5-genera Excel for diagnosis labels (H/M/PI)
  - Compute 5-species R/G ratio (validation) and expanded R/G ratio (Treponema/Tannerella/Prevotella added)
  - Report Spearman ρ for both; save figure and CSV
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl

_here  = Path(__file__).resolve().parents[2]
PROF_DIR = _here / 'results' / 'joshi_16s_profiles'
EXCEL    = _here / 'Datasets' / '20260416_mSystems_16S_5genera_all_profiles.xlsx'
FIG_DIR  = _here / 'results' / 'dieckow_paper'
FIG_DIR.mkdir(exist_ok=True)

# 5 guilds mapping from minimap2 keyword → same as our Hamilton model
GUILD_5 = {
    'actinomyces':   'com5',
    'schaalia':      'com5',
    'streptococcus': 'com5',
    'veillonella':   'com5',
    'porphyromonas': 'dys5',
    'fusobacterium': 'dys5',
}
# Expanded: add red complex + Rothia/Dialister
GUILD_EXP = {
    **GUILD_5,
    'tannerella':   'dys_exp',
    'prevotella':   'dys_exp',
    'treponema':    'dys_exp',
    'campylobacter':'dys_exp',
    'rothia':       'com_exp',
    'dialister':    'com_exp',
}

# ---- Load Excel → per-sample 5-genus proportions + diagnosis ----
wb    = openpyxl.load_workbook(EXCEL, read_only=True)
ws    = wb.active
rows  = list(ws.iter_rows(values_only=True))
diags = list(rows[0][1:])          # diagnosis per column
genus_rows = {}
for row in rows[1:]:
    g = str(row[0]).strip().lower() if row[0] else None
    if g in ('actinomyces', 'streptococcus', 'porphyromonas', 'fusobacterium', 'veillonella'):
        vals = np.array([float(v or 0) for v in row[1:]])
        genus_rows[g] = genus_rows.get(g, np.zeros(len(vals))) + vals

n_samples = len(diags)
excel_mat  = np.zeros((n_samples, 5))
g5_order   = ['actinomyces','streptococcus','porphyromonas','fusobacterium','veillonella']
for i, g in enumerate(g5_order):
    if g in genus_rows:
        excel_mat[:, i] = genus_rows[g]
row_sums = excel_mat.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
excel_prop = excel_mat / row_sums   # (n_samples, 5) relative abundances

# Map diagnosis → numeric score
diag_score = {'Health': 0, 'Mucositis': 1, 'Peri-implantitis': 2}
y_excel    = np.array([diag_score.get(str(d), -1) for d in diags])
valid_mask = y_excel >= 0

# ---- Load minimap2 profiles ----
profiles = {}
for f in sorted(PROF_DIR.glob('*.json')):
    d = json.loads(f.read_text())
    samp    = d['sample']
    genera  = d['genera']   # {"keyword|class|flag": count}
    kw_counts = {}
    for key, cnt in genera.items():
        kw = key.split('|')[0]
        kw_counts[kw] = kw_counts.get(kw, 0) + cnt
    profiles[samp] = kw_counts

if not profiles:
    print("No profiles found in", PROF_DIR)
    sys.exit(0)

print(f"Loaded {len(profiles)} minimap2 profiles.")

# ---- Match each minimap2 sample → Excel column via 5-genus cosine similarity ----
def get_5prop(kw_counts):
    v = np.array([float(kw_counts.get(g, 0)) for g in g5_order])
    s = v.sum()
    return v / s if s > 0 else v

matched_samples = []  # (samp, diag, gdi_5, gdi_exp)
for samp, kw_counts in sorted(profiles.items()):
    prop5 = get_5prop(kw_counts)
    if prop5.sum() < 1e-8:
        continue
    # Find best matching Excel column
    sims = np.array([1 - cosine(prop5, excel_prop[i]) if excel_prop[i].sum() > 1e-8 else -1
                     for i in range(n_samples)])
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    if best_sim < 0.80 or not valid_mask[best_idx]:
        continue
    diag = diags[best_idx]

    # 5-species R/G ratio: log(dys5 / com5)
    dys5 = sum(kw_counts.get(g, 0) for g in ['porphyromonas','fusobacterium'])
    com5 = sum(kw_counts.get(g, 0) for g in ['actinomyces','schaalia','streptococcus','veillonella'])
    gdi_5 = np.log((dys5 + 1e-3) / (com5 + 1e-3))

    # Expanded R/G ratio: also Treponema, Tannerella, Prevotella, Campylobacter
    dys_exp = dys5 + sum(kw_counts.get(g, 0) for g in ['treponema','tannerella','prevotella','campylobacter'])
    com_exp = com5 + sum(kw_counts.get(g, 0) for g in ['rothia','dialister'])
    gdi_exp = np.log((dys_exp + 1e-3) / (com_exp + 1e-3))

    matched_samples.append({
        'sample': samp,
        'diag':   str(diag),
        'sim':    float(best_sim),
        'gdi_5':  float(gdi_5),
        'gdi_exp':float(gdi_exp),
    })

df = pd.DataFrame(matched_samples)
n_matched = len(df)
print(f"Matched {n_matched} samples (cosine sim > 0.80)")
if n_matched < 5:
    print("Too few matches — cannot compute ρ.")
    sys.exit(0)

y = np.array([diag_score[d] for d in df['diag']])
rho5,  p5  = spearmanr(df['gdi_5'],  y)
rho_exp, p_exp = spearmanr(df['gdi_exp'], y)

print(f"\n=== R/G ratio Comparison (N={n_matched}) ===")
print(f"  5-species R/G ratio  ρ = {rho5:.3f}  (p={p5:.3e})")
print(f"  Expanded R/G ratio   ρ = {rho_exp:.3f}  (p={p_exp:.3e})")
print(f"  Δρ = {rho_exp - rho5:+.3f}")

for diag, grp in df.groupby('diag'):
    print(f"  {diag:20s}: n={len(grp):3d}  R/G ratio_5={grp['gdi_5'].mean():+.2f}  R/G ratio_exp={grp['gdi_exp'].mean():+.2f}")

# ---- Figure ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = {'Health': '#2196F3', 'Mucositis': '#FF9800', 'Peri-implantitis': '#F44336'}
for ax, col, rho, title in zip(axes,
                                ['gdi_5', 'gdi_exp'],
                                [rho5, rho_exp],
                                ['5-species R/G ratio', 'Expanded R/G ratio (+Treponema/Tannerella/Prevotella)']):
    for diag, grp in df.groupby('diag'):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(grp))
        yscore = np.array([diag_score[d] for d in grp['diag']]) + jitter
        ax.scatter(grp[col], yscore, c=colors.get(str(diag), 'gray'), alpha=0.6, s=20)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['Health', 'Mucositis', 'PI'])
    ax.set_xlabel('R/G ratio'); ax.set_title(f"{title}\nρ = {rho:.3f} (N={n_matched})")
    ax.axvline(0, color='k', lw=0.5, ls='--')
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_gdi_expanded_check.png', dpi=150, bbox_inches='tight')
print(f"Saved figure: {FIG_DIR}/fig_gdi_expanded_check.png")

# Save CSV
df.to_csv(FIG_DIR / 'joshi_gdi_expanded_check.csv', index=False)
print(f"Saved CSV: {FIG_DIR}/joshi_gdi_expanded_check.csv")
