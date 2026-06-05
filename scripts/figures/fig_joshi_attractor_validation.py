# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

"""
Joshi 2025 cross-cohort attractor validation.

Compares Joshi (2025) peri-implantitis dataset's class-level mean compositions
(health vs peri-implantitis; Suppl. Files E and G) against the Dieckow (2024)
community types (CT1 = commensal, CT2 = dysbiotic) fitted by the nife gLV model.

Validation logic:
  - Map Joshi class means → nife 10-guild taxonomy (direct class name match + 'Other')
  - Map Dieckow patient observations → CT1/CT2 mean compositions
  - Compare Joshi health ↔ CT1 and Joshi peri-implantitis ↔ CT2
  - Metrics: Bray-Curtis dissimilarity, Pearson r, radar/bar chart overlay
"""

import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import braycurtis
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parents[2]))
import pub_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

RESULTS_DIR = Path(__file__).parents[2] / 'results' / 'dieckow_cr'
JOSHI_DIR = Path(__file__).parents[2] / 'Datasets' / 'joshi2025_SI'
OUT_DIR = RESULTS_DIR / 'figs'
OUT_DIR.mkdir(exist_ok=True)

GUILDS = GUILD_ORDER  # 10 guilds including 'Other'
GUILDS_9 = [g for g in GUILDS if g != 'Other']  # 9 named guilds

# ── 1. Load Joshi class-level means ──────────────────────────────────────────

def load_joshi_class(csv_path):
    """Return dict: class_name → (mean_health, mean_peri)  (values in %)."""
    rows = list(csv.reader(open(csv_path)))
    header = rows[0]
    h_idx = header.index('mean_health')
    p_idx = header.index('mean_periimplantitis')
    return {r[0]: (float(r[h_idx]), float(r[p_idx])) for r in rows[1:] if r[0]}


def joshi_to_guild_vec(joshi_dict, guilds=GUILDS_9):
    """Map Joshi class means → nife guild vector, renormalize to sum=1 over the 9 guilds.

    BC distance is sensitive to absolute scale, so we renormalize both Joshi and
    Dieckow vectors to the same 9-guild basis (excluding 'Other') for a fair comparison.
    """
    health, peri = [], []
    for g in guilds:
        if g in joshi_dict:
            h, p = joshi_dict[g]
        else:
            h, p = 0.0, 0.0
        health.append(h)
        peri.append(p)
    health = np.array(health)
    peri = np.array(peri)
    # Renormalize to sum=1 over the matched guilds
    health = health / health.sum()
    peri = peri / peri.sum()
    return health, peri


joshi_16s = load_joshi_class(JOSHI_DIR / 'joshi2025_suppE_class.csv')
joshi_rna = load_joshi_class(JOSHI_DIR / 'joshi2025_suppG_class_rnaseq.csv')

joshi_16s_h, joshi_16s_p = joshi_to_guild_vec(joshi_16s)
joshi_rna_h, joshi_rna_p = joshi_to_guild_vec(joshi_rna)

# ── 2. Load Dieckow CT1/CT2 mean compositions ────────────────────────────────

df_guild = pd.read_csv(RESULTS_DIR / 'dieckow_guild_abundance.csv')
# CT1 patients (commensal-dominant) from cross_ct_prediction.json
ct_pred = json.load(open(RESULTS_DIR / 'cross_ct_prediction.json'))
ct1_patients = ct_pred['ct1_patients']  # e.g. ['D','E','G','K','L']

df_guild['patient'] = df_guild['sample'].str[0]
ct1_mask = df_guild['patient'].isin(ct1_patients)

# Renormalize Dieckow to 9-guild basis too (exclude Other) for comparable BC
ct1_raw = df_guild.loc[ct1_mask, GUILDS_9].mean().values
ct2_raw = df_guild.loc[~ct1_mask, GUILDS_9].mean().values
ct1_mean = ct1_raw / ct1_raw.sum()
ct2_mean = ct2_raw / ct2_raw.sum()

# ── 3. Compute similarity metrics ────────────────────────────────────────────

def bc(a, b):
    return braycurtis(a, b)

def pr(a, b):
    return pearsonr(a, b)

print("=== 16S (Joshi) vs Dieckow CT means ===")
print(f"Health  vs CT1 (commensal): BC={bc(joshi_16s_h, ct1_mean):.3f}, r={pr(joshi_16s_h, ct1_mean)[0]:.3f}")
print(f"Health  vs CT2 (dysbiotic): BC={bc(joshi_16s_h, ct2_mean):.3f}, r={pr(joshi_16s_h, ct2_mean)[0]:.3f}")
print(f"Peri    vs CT1 (commensal): BC={bc(joshi_16s_p, ct1_mean):.3f}, r={pr(joshi_16s_p, ct1_mean)[0]:.3f}")
print(f"Peri    vs CT2 (dysbiotic): BC={bc(joshi_16s_p, ct2_mean):.3f}, r={pr(joshi_16s_p, ct2_mean)[0]:.3f}")

print("\n=== RNAseq (Joshi) vs Dieckow CT means ===")
print(f"Health  vs CT1 (commensal): BC={bc(joshi_rna_h, ct1_mean):.3f}, r={pr(joshi_rna_h, ct1_mean)[0]:.3f}")
print(f"Health  vs CT2 (dysbiotic): BC={bc(joshi_rna_h, ct2_mean):.3f}, r={pr(joshi_rna_h, ct2_mean)[0]:.3f}")
print(f"Peri    vs CT1 (commensal): BC={bc(joshi_rna_p, ct1_mean):.3f}, r={pr(joshi_rna_p, ct1_mean)[0]:.3f}")
print(f"Peri    vs CT2 (dysbiotic): BC={bc(joshi_rna_p, ct2_mean):.3f}, r={pr(joshi_rna_p, ct2_mean)[0]:.3f}")

# ── 4. Figure ─────────────────────────────────────────────────────────────────

pub_style.apply()
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38)

x = np.arange(len(GUILDS_9))
w = 0.22
short = [GUILD_SHORT.get(g, g[:4]) for g in GUILDS_9] if hasattr(__import__('guild_replicator_dieckow'), 'GUILD_SHORT') else [g[:5] for g in GUILDS_9]

# Try to get GUILD_SHORT
try:
    from guild_replicator_dieckow import GUILD_SHORT
    short = [GUILD_SHORT[g] for g in GUILDS_9]
except Exception:
    short = [g[:5] for g in GUILDS_9]

colors = {
    'CT1': '#2166ac',
    'CT2': '#d6604d',
    'Joshi_H': '#92c5de',
    'Joshi_P': '#f4a582',
    'RNA_H': '#74add1',
    'RNA_P': '#fdae61',
}

def bar_panel(ax, ax_title, ct1, ct2, j_h, j_p, label_h, label_p):
    ax.bar(x - 1.5*w, ct1*100,  w, color=colors['CT1'],     label='CT1 (commensal)', alpha=0.9)
    ax.bar(x - 0.5*w, j_h*100,  w, color=colors['Joshi_H'], label=label_h,           alpha=0.9)
    ax.bar(x + 0.5*w, ct2*100,  w, color=colors['CT2'],     label='CT2 (dysbiotic)', alpha=0.9)
    ax.bar(x + 1.5*w, j_p*100,  w, color=colors['Joshi_P'], label=label_p,           alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Relative abundance (%)')
    ax.set_title(ax_title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, None)

ax1 = fig.add_subplot(gs[0, :])
bar_panel(ax1,
          'Guild-level composition: Dieckow CT1/CT2 vs Joshi 2025 health/peri-implantitis (full-16S)',
          ct1_mean, ct2_mean, joshi_16s_h, joshi_16s_p,
          'Joshi health', 'Joshi peri-implantitis')

ax2 = fig.add_subplot(gs[1, :])
bar_panel(ax2,
          'Guild-level composition: Dieckow CT1/CT2 vs Joshi 2025 health/peri-implantitis (RNAseq)',
          ct1_mean, ct2_mean, joshi_rna_h, joshi_rna_p,
          'Joshi health (RNA)', 'Joshi peri-implantitis (RNA)')

# Add BC annotation boxes
def bc_box(ax, bc_h_ct1, bc_h_ct2, bc_p_ct1, bc_p_ct2):
    txt = (f"Health vs CT1: BC={bc_h_ct1:.3f}\n"
           f"Health vs CT2: BC={bc_h_ct2:.3f}\n"
           f"Peri vs CT1:   BC={bc_p_ct1:.3f}\n"
           f"Peri vs CT2:   BC={bc_p_ct2:.3f}")
    ax.text(0.01, 0.98, txt, transform=ax.transAxes, fontsize=7.5,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

bc_box(ax1,
       bc(joshi_16s_h, ct1_mean), bc(joshi_16s_h, ct2_mean),
       bc(joshi_16s_p, ct1_mean), bc(joshi_16s_p, ct2_mean))
bc_box(ax2,
       bc(joshi_rna_h, ct1_mean), bc(joshi_rna_h, ct2_mean),
       bc(joshi_rna_p, ct1_mean), bc(joshi_rna_p, ct2_mean))

fig.suptitle('External validation: Joshi 2025 (MHH, cross-sectional) vs Dieckow 2024 gLV community types',
             fontsize=10, y=1.01)

for ext in ('pdf', 'png'):
    fig.savefig(OUT_DIR / f'fig_joshi_Amatrix_validation.{ext}',
                bbox_inches='tight', dpi=150)
    fig.savefig(RESULTS_DIR / f'figs/fig_joshi_Amatrix_validation.{ext}',
                bbox_inches='tight', dpi=150)

# also save to top-level results
for ext in ('pdf', 'png'):
    fig.savefig(Path(__file__).parents[2] / 'results' / 'dieckow_cr' / f'figs/fig_joshi_Amatrix_validation.{ext}',
                bbox_inches='tight', dpi=150)

plt.close()
print("\nFigure saved.")
