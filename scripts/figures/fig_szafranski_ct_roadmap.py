# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

"""
Szafranski 2025 EcoPreprint: 4-CT disease trajectory roadmap.

Panel A: schematic disease progression axis (CT_I → CT_IV) with
         key guild compositions from Joshi 2025 class data, and
         Dieckow CT1/CT2 positions marked.
Panel B: guild-level bar chart for each of the 4 CTs
         (qualitative, based on Joshi class means for CT-matching diagnoses).
Panel C: 10 doubly-validated health-marker ECs — LDA score (biomarker paper)
         vs regression coefficient (severity paper).
"""

import csv, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.stats import pearsonr

import pub_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

ROOT = Path(__file__).parents[2]
JOSHI_DIR = ROOT / 'Datasets' / 'joshi2025_SI'
SZF_DIR   = ROOT / 'Datasets' / 'szafranski2025_ecopreprint'
RESULTS   = ROOT / 'results' / 'dieckow_cr'
OUT       = RESULTS / 'figs'
OUT.mkdir(exist_ok=True)

GUILDS_9 = [g for g in GUILD_ORDER if g != 'Other']

# ── helpers ──────────────────────────────────────────────────────────────────

def load_joshi_class(path):
    rows = list(csv.reader(open(path)))
    hd = rows[0]
    hi, pi = hd.index('mean_health'), hd.index('mean_periimplantitis')
    return {r[0]: (float(r[hi]), float(r[pi])) for r in rows[1:] if r[0]}

def renorm(v):
    s = v.sum()
    return v / s if s > 0 else v

# ── load Joshi class means ────────────────────────────────────────────────────

j16s = load_joshi_class(JOSHI_DIR / 'joshi2025_suppE_class.csv')
jrna = load_joshi_class(JOSHI_DIR / 'joshi2025_suppG_class_rnaseq.csv')

j16s_h = renorm(np.array([j16s.get(g, (0, 0))[0] for g in GUILDS_9]))
j16s_p = renorm(np.array([j16s.get(g, (0, 0))[1] for g in GUILDS_9]))

# ── load Dieckow CT1/CT2 ─────────────────────────────────────────────────────

df = pd.read_csv(RESULTS / 'dieckow_guild_abundance.csv')
ct_pred = json.load(open(RESULTS / 'cross_ct_prediction.json'))
ct1_pat = ct_pred['ct1_patients']
df['pat'] = df['sample'].str[0]
ct1 = renorm(df.loc[df['pat'].isin(ct1_pat),  GUILDS_9].mean().values)
ct2 = renorm(df.loc[~df['pat'].isin(ct1_pat), GUILDS_9].mean().values)

# ── load cross-validated ECs ──────────────────────────────────────────────────

cross_ec = list(csv.reader(open(JOSHI_DIR / 'joshi2025_cross_EC_validation.csv')))
doubly = [(r[0], float(r[2]), r[3]) for r in cross_ec[1:]
          if 'BOTH' in r[1] and r[2]]

sev_ec  = list(csv.reader(open(JOSHI_DIR / 'joshi2025_severity_EC_PD_regression_annotated.csv')))
sev_hd  = sev_ec[0]
rc_idx  = sev_hd.index('Regression co-efficient')
pw_idx  = sev_hd.index('pathway_group')
ec_idx  = sev_hd.index('EC no.')
sev_map = {r[ec_idx].rstrip('§£').strip(): (float(r[rc_idx]), r[pw_idx])
           for r in sev_ec[1:] if r[rc_idx]}

# ── figure ────────────────────────────────────────────────────────────────────

pub_style.apply()
fig = plt.figure(figsize=(15, 9))

gs_top = fig.add_gridspec(1, 1, top=0.97, bottom=0.62, left=0.06, right=0.97)
gs_bot = fig.add_gridspec(1, 2, top=0.55, bottom=0.07, left=0.06, right=0.97, wspace=0.30)

ax_road = fig.add_subplot(gs_top[0])
ax_bar  = fig.add_subplot(gs_bot[0])
ax_ec   = fig.add_subplot(gs_bot[1])

colors_g = [GUILD_COLORS[g] for g in GUILDS_9]
short    = [GUILD_SHORT[g]   for g in GUILDS_9]

# ── Panel A: roadmap ──────────────────────────────────────────────────────────

# 4 CT positions along x-axis (0=PIH … 3=PI)
CT_LABELS = ['CT I\n(PIH)', 'CT II\n(PIM\nNeisseria)', 'CT III\n(PIM\nBacteroidia/Fuso)', 'CT IV\n(PI)']
CT_X      = [0, 1, 2, 3]
CT_COLS   = ['#2166ac', '#74add1', '#fdae61', '#d6604d']
CT_DIAG   = ['PIH', 'PIM', 'PIM', 'PI']

# background gradient
ax_road.set_xlim(-0.5, 3.5)
ax_road.set_ylim(0, 1)
ax_road.set_aspect('auto')
ax_road.axis('off')

grad = np.linspace(0, 1, 200).reshape(1, -1)
ax_road.imshow(grad, aspect='auto', extent=[-0.5, 3.5, 0, 1],
               cmap='RdBu_r', alpha=0.12, zorder=0)

# main arrow
ax_road.annotate('', xy=(3.3, 0.5), xytext=(-0.3, 0.5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='#555'))
ax_road.text(1.5, 0.44, 'Disease progression  →', ha='center', va='top',
             fontsize=9, color='#555')

# CT boxes
for xi, (lbl, col, diag) in enumerate(zip(CT_LABELS, CT_COLS, CT_DIAG)):
    ax_road.add_patch(mpatches.FancyBboxPatch((xi-0.4, 0.55), 0.80, 0.38,
                      boxstyle='round,pad=0.02', fc=col, ec='k', lw=1.0, alpha=0.85, zorder=2))
    ax_road.text(xi, 0.74, lbl, ha='center', va='center', fontsize=8.5,
                 color='white', fontweight='bold', zorder=3)

# guild stacked mini bars at each CT
# CT composition (qualitative — use Joshi health for CT I, peri for CT IV, interpolate)
ct_comps = {
    'CT_I':   j16s_h,
    'CT_IV':  j16s_p,
    'CT_II':  renorm(0.7 * j16s_h + 0.3 * j16s_p),
    'CT_III': renorm(0.3 * j16s_h + 0.7 * j16s_p),
}
ct_keys = ['CT_I', 'CT_II', 'CT_III', 'CT_IV']
bar_h, bar_base = 0.38, 0.06
for xi, ck in enumerate(ct_keys):
    comp = ct_comps[ck]
    bottom = bar_base
    for gi, frac in enumerate(comp):
        ax_road.bar(xi, frac * bar_h, bottom=bottom, width=0.35,
                    color=colors_g[gi], edgecolor='none', zorder=2)
        bottom += frac * bar_h

# Dieckow CT1/CT2 markers
for xi, (label, comp, mk) in enumerate([
    ('Dieckow\nCT1', ct1, '★'), ('Dieckow\nCT2', ct2, '★')
]):
    xpos = 0.05 if xi == 0 else 2.95
    ax_road.text(xpos, 0.07, label, ha='center', va='bottom', fontsize=7.5,
                 color='#222', style='italic',
                 bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='#888', lw=0.8))
    ax_road.annotate('', xy=(0.0 if xi == 0 else 3.0, 0.17),
                     xytext=(xpos, 0.22),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='#444'))

# guild legend
handles = [mpatches.Patch(color=GUILD_COLORS[g], label=GUILD_SHORT[g]) for g in GUILDS_9]
ax_road.legend(handles=handles, loc='upper left', fontsize=6.5,
               ncol=9, framealpha=0.9,
               bbox_to_anchor=(0.0, 1.01), borderaxespad=0)

ax_road.set_title('A   Szafranski 2025 EcoPreprint: 4 community types — disease progression axis\n'
                  '(mini-bars = Joshi 2025 guild composition proxy; Dieckow CT1/CT2 anchors)',
                  fontsize=9, fontweight='bold', loc='left')

# ── Panel B: guild bar comparison CT1 vs CT2 vs Joshi H vs Joshi P ───────────

x = np.arange(len(GUILDS_9))
w = 0.22
ax_bar.bar(x - 1.5*w, ct1*100,   w, color='#2166ac', label='Dieckow CT1', alpha=0.9)
ax_bar.bar(x - 0.5*w, j16s_h*100, w, color='#92c5de', label='Joshi health', alpha=0.9)
ax_bar.bar(x + 0.5*w, j16s_p*100, w, color='#f4a582', label='Joshi peri-implantitis', alpha=0.9)
ax_bar.bar(x + 1.5*w, ct2*100,   w, color='#d6604d', label='Dieckow CT2', alpha=0.9)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax_bar.set_ylabel('Relative abundance (%)')
ax_bar.set_title('B   Guild-level composition: Dieckow CT1/CT2 vs Joshi 2025', fontsize=9, fontweight='bold')
ax_bar.legend(fontsize=7, loc='upper right')

# correlation annotations
from scipy.stats import pearsonr
r16s_h_ct1, _ = pearsonr(j16s_h, ct1)
r16s_p_ct2, _ = pearsonr(j16s_p, ct2)
ax_bar.text(0.01, 0.98,
            f'Joshi health ↔ CT1: r={r16s_h_ct1:.2f}\nJoshi peri ↔ CT2: r={r16s_p_ct2:.2f}',
            transform=ax_bar.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

# ── Panel C: doubly-validated EC scatter ──────────────────────────────────────

ec_labels, lda_vals, rc_vals, pw_vals = [], [], [], []
for ec, lda, _ in doubly:
    ec_clean = ec.rstrip('§£').strip()
    if ec_clean in sev_map:
        rc, pw = sev_map[ec_clean]
        ec_labels.append(ec_clean)
        lda_vals.append(lda)
        rc_vals.append(rc)
        pw_vals.append(pw)

pw_set = sorted(set(pw_vals))
pw_colors = dict(zip(pw_set, plt.cm.tab10(np.linspace(0, 0.8, len(pw_set)))))

ax_ec.axhline(0, color='k', lw=0.5, ls='--')
ax_ec.axvline(0, color='k', lw=0.5, ls='--')
for lda, rc, pw, ec in zip(lda_vals, rc_vals, pw_vals, ec_labels):
    ax_ec.scatter(lda, rc, color=pw_colors[pw], s=55, zorder=3,
                  edgecolors='k', lw=0.4)
    ax_ec.annotate(ec, (lda, rc), fontsize=6.5, ha='left', va='bottom',
                   xytext=(3, 2), textcoords='offset points')

pw_handles = [mpatches.Patch(color=pw_colors[pw], label=pw) for pw in pw_set]
ax_ec.legend(handles=pw_handles, fontsize=6.5, loc='lower right')
ax_ec.set_xlabel('LDA-CAP score (Joshi Biomarker — health association)', fontsize=8)
ax_ec.set_ylabel('Regression coeff. vs PD (Joshi Severity — negative = more healthy)', fontsize=8)
ax_ec.set_title('C   Doubly-validated health-marker ECs\n(Biomarker paper LDA > 0.4  AND  Severity paper PD regression)',
                fontsize=9, fontweight='bold')

n = len(ec_labels)
r_val, p_val = pearsonr(lda_vals, rc_vals) if n > 2 else (float('nan'), float('nan'))
ax_ec.text(0.02, 0.02, f'n={n} ECs\nBiomarker × Severity r={r_val:.2f}',
           transform=ax_ec.transAxes, fontsize=7.5,
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

fig.suptitle('Cross-dataset validation roadmap: Szafranski 2025 CT spectrum × Dieckow 2024 gLV community types × Joshi 2025 biomarkers',
             fontsize=9.5, y=0.995)

for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'fig_szafranski_ct_roadmap.{ext}', bbox_inches='tight', dpi=150)
    fig.savefig(RESULTS / f'fig_szafranski_ct_roadmap.{ext}', bbox_inches='tight', dpi=150)

plt.close()
print("Done →", OUT)
