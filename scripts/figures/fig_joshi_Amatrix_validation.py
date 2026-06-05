# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

"""
Joshi 2025 external validation: dysbiosis direction comparison.

Core question: does the guild-level compositional SHIFT from health to disease
in Joshi 2025 (cross-sectional, peri-implantitis) point in the same direction
as the shift from CT1 (commensal) to CT2 (dysbiotic) in Dieckow 2024 (longitudinal)?

Validation metric: Pearson r between
  Δ_Joshi  = peri-implantitis mean − health mean    (Joshi,  9 guilds)
  Δ_Dieckow = CT2 mean − CT1 mean                   (Dieckow, 9 guilds)

If the two datasets capture the same ecological transition, Δ vectors should
be positively correlated even though absolute compositions differ.
"""

import csv, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import pub_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

RESULTS_DIR = Path(__file__).parents[2] / 'results' / 'dieckow_cr'
JOSHI_DIR   = Path(__file__).parents[2] / 'Datasets' / 'joshi2025_SI'
OUT         = RESULTS_DIR / 'figs'
OUT.mkdir(exist_ok=True)

GUILDS_9 = [g for g in GUILD_ORDER if g != 'Other']

# ── helpers ──────────────────────────────────────────────────────────────────

def load_joshi(path):
    rows = list(csv.reader(open(path)))
    hd = rows[0]
    hi, pi = hd.index('mean_health'), hd.index('mean_periimplantitis')
    return {r[0]: (float(r[hi]), float(r[pi])) for r in rows[1:] if r[0]}


def guild_vec(d, guilds, normalise=True):
    """Return (health_vec, peri_vec) as fractions summing to 1."""
    h = np.array([d.get(g, (0, 0))[0] for g in guilds])
    p = np.array([d.get(g, (0, 0))[1] for g in guilds])
    if normalise:
        h = h / h.sum(); p = p / p.sum()
    return h, p


# ── load data ─────────────────────────────────────────────────────────────────

j16s = load_joshi(JOSHI_DIR / 'joshi2025_suppE_class.csv')
jrna = load_joshi(JOSHI_DIR / 'joshi2025_suppG_class_rnaseq.csv')

j16s_h, j16s_p = guild_vec(j16s, GUILDS_9)
jrna_h, jrna_p = guild_vec(jrna, GUILDS_9)

df = pd.read_csv(RESULTS_DIR / 'dieckow_guild_abundance.csv')
ct_pred = json.load(open(RESULTS_DIR / 'cross_ct_prediction.json'))
ct1_patients = ct_pred['ct1_patients']
df['pat'] = df['sample'].str[0]

ct1_raw = df.loc[df['pat'].isin(ct1_patients),  GUILDS_9].mean().values
ct2_raw = df.loc[~df['pat'].isin(ct1_patients), GUILDS_9].mean().values
ct1 = ct1_raw / ct1_raw.sum()
ct2 = ct2_raw / ct2_raw.sum()

# ── dysbiosis direction vectors ───────────────────────────────────────────────

delta_16s = j16s_p - j16s_h      # Joshi: health → peri-implantitis
delta_rna = jrna_p - jrna_h      # Joshi RNAseq
delta_dck  = ct2 - ct1           # Dieckow: CT1 → CT2

r16s, p16s = pearsonr(delta_16s, delta_dck)
rrna, prna = pearsonr(delta_rna, delta_dck)
rho16s, pp16s = spearmanr(delta_16s, delta_dck)

print("=== Dysbiosis direction correlation ===")
print(f"Joshi 16S  vs Dieckow CT shift: r={r16s:.3f}, p={p16s:.3f}, ρ={rho16s:.3f}")
print(f"Joshi RNAseq vs Dieckow CT shift: r={rrna:.3f}, p={prna:.3f}")

print("\nGuild-level breakdown:")
print(f"{'Guild':<22} Δ_Joshi16S  Δ_JoshiRNA  Δ_Dieckow")
for i, g in enumerate(GUILDS_9):
    print(f"  {g:<20} {delta_16s[i]:+.3f}      {delta_rna[i]:+.3f}      {delta_dck[i]:+.3f}")

# ── figure ────────────────────────────────────────────────────────────────────

pub_style.apply()
colors_guild = [GUILD_COLORS[g] for g in GUILDS_9]
short = [GUILD_SHORT[g] for g in GUILDS_9]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.subplots_adjust(wspace=0.38)

# ── panel A: bar chart of Δ ──────────────────────────────────────────────────
ax = axes[0]
x = np.arange(len(GUILDS_9))
w = 0.28
bars1 = ax.bar(x - w, delta_16s * 100, w, color=colors_guild, edgecolor='k', lw=0.4,
               label='Joshi 16S', alpha=0.9)
bars2 = ax.bar(x,     delta_rna  * 100, w, color=colors_guild, edgecolor='k', lw=0.4,
               alpha=0.5, hatch='///', label='Joshi RNAseq')
bars3 = ax.bar(x + w, delta_dck  * 100, w, color='none', edgecolor=colors_guild, lw=2.0,
               label='Dieckow CT2−CT1')

ax.axhline(0, color='k', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Δ relative abundance (pp)')
ax.set_title('A  Dysbiosis direction per guild', fontweight='bold', fontsize=9)
ax.legend(fontsize=7, loc='lower left')

# ── panel B: scatter Joshi 16S vs Dieckow ────────────────────────────────────
ax = axes[1]
lim = max(abs(np.concatenate([delta_16s, delta_dck]))) * 110
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.plot([-lim, lim], [-lim, lim], '--', color='gray', lw=0.8)
for i, g in enumerate(GUILDS_9):
    ax.scatter(delta_dck[i]*100, delta_16s[i]*100,
               color=GUILD_COLORS[g], s=60, zorder=3, edgecolors='k', lw=0.4)
    ax.annotate(short[i], (delta_dck[i]*100, delta_16s[i]*100),
                fontsize=7, ha='left', va='bottom',
                xytext=(2, 2), textcoords='offset points')

ax.set_xlabel('Δ Dieckow  CT2−CT1 (pp)', fontsize=9)
ax.set_ylabel('Δ Joshi 16S  peri−health (pp)', fontsize=9)
ax.set_title(f'B  16S shift correlation\n'
             f'r = {r16s:.2f},  p = {p16s:.3f},  ρ = {rho16s:.2f}',
             fontsize=9, fontweight='bold')

# ── panel C: scatter Joshi RNAseq vs Dieckow ─────────────────────────────────
ax = axes[2]
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.plot([-lim, lim], [-lim, lim], '--', color='gray', lw=0.8)
for i, g in enumerate(GUILDS_9):
    ax.scatter(delta_dck[i]*100, delta_rna[i]*100,
               color=GUILD_COLORS[g], s=60, zorder=3, edgecolors='k', lw=0.4)
    ax.annotate(short[i], (delta_dck[i]*100, delta_rna[i]*100),
                fontsize=7, ha='left', va='bottom',
                xytext=(2, 2), textcoords='offset points')

ax.set_xlabel('Δ Dieckow  CT2−CT1 (pp)', fontsize=9)
ax.set_ylabel('Δ Joshi RNAseq  peri−health (pp)', fontsize=9)
ax.set_title(f'C  RNAseq shift correlation\n'
             f'r = {rrna:.2f},  p = {prna:.3f}',
             fontsize=9, fontweight='bold')

fig.suptitle('Joshi 2025 × Dieckow 2024: guild-level dysbiosis direction comparison\n'
             '(Joshi: cross-sectional health/peri-implantitis;  Dieckow: CT1→CT2 longitudinal shift)',
             fontsize=9)

for ext in ('pdf', 'png'):
    p = OUT / f'fig_joshi_Amatrix_validation.{ext}'
    fig.savefig(p, bbox_inches='tight', dpi=150)
    # also top-level results
    fig.savefig(RESULTS_DIR / f'fig_joshi_Amatrix_validation.{ext}', bbox_inches='tight', dpi=150)

plt.close()
print("\nFigure saved to", OUT)
