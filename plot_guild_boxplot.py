#!/usr/bin/env python3
"""Guild abundance box plots: Week 1/2/3 side by side, patient variation."""

import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guild_replicator_dieckow import GUILD_COLORS, GUILD_SHORT as SHORT, GUILD_ORDER

PHI_NPY   = Path(__file__).parent / 'results/dieckow_otu/phi_guild_excel_class.npy'
SUMM_JSON = Path(__file__).parent / 'results/dieckow_otu/guild_summary_excel_class.json'
OUT_DIR   = Path(__file__).parent / 'results/dieckow_otu'

phi = np.load(PHI_NPY)          # (10, 3, 11)
with open(SUMM_JSON) as f:
    s = json.load(f)

GUILDS = s.get('guilds', s.get('guild_order'))
N_G    = len(GUILDS)

WEEK_EDGE_COLORS = ['#2266cc', '#cc5500', '#117733']
WEEK_FILL_ALPHA  = 0.55
WEEK_LABELS = ['Week 1', 'Week 2', 'Week 3']
WEEK_HATCHES = ['', '///', '...']

# ── per-guild box plots ───────────────────────────────────────────────────────
ncols = 4
nrows = (N_G + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
axes = axes.flatten()

for g_idx, guild in enumerate(GUILDS):
    ax = axes[g_idx]
    data_by_week = []
    for w in range(3):
        mask = phi[:, w, :].sum(axis=1) > 1e-12
        vals = phi[mask, w, g_idx] * 100          # → %
        data_by_week.append(vals)

    bp = ax.boxplot(
        data_by_week,
        positions=[1, 2, 3],
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.6),
    )
    guild_color = GUILD_COLORS.get(guild, '#aaaaaa')
    for w_i, (patch, hatch) in enumerate(zip(bp['boxes'], WEEK_HATCHES)):
        patch.set_facecolor(guild_color)
        patch.set_alpha(WEEK_FILL_ALPHA + w_i * 0.15)
        patch.set_hatch(hatch)
        patch.set_edgecolor(WEEK_EDGE_COLORS[w_i])
        patch.set_linewidth(1.5)

    # overlay individual points with jitter
    rng = np.random.default_rng(42)
    for w_i, vals in enumerate(data_by_week):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), w_i + 1) + jitter,
            vals,
            color=guild_color, edgecolors=WEEK_EDGE_COLORS[w_i],
            s=24, linewidths=0.8, zorder=3, alpha=0.9,
        )

    ax.set_title(SHORT.get(guild, guild), fontsize=11, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['W1', 'W2', 'W3'], fontsize=9)
    ax.set_ylabel('Abundance (%)', fontsize=8)
    ax.set_xlim(0.4, 3.6)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3, linewidth=0.6)

# hide unused axes
for i in range(N_G, len(axes)):
    axes[i].set_visible(False)

# legend — week distinction via hatch + edge color
patches = [mpatches.Patch(facecolor='#999999', edgecolor=ec, hatch=h,
                           alpha=0.7, label=l, linewidth=1.5)
           for ec, h, l in zip(WEEK_EDGE_COLORS, WEEK_HATCHES, WEEK_LABELS)]
fig.legend(handles=patches, loc='lower right',
           bbox_to_anchor=(0.98, 0.01), fontsize=10, frameon=True)

fig.suptitle('Guild-level relative abundance — patient variation (n≤8 per week)',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()

out_pdf = OUT_DIR / 'guild_boxplot_weeks.pdf'
out_png = OUT_DIR / 'guild_boxplot_weeks.png'
fig.savefig(out_pdf, bbox_inches='tight', dpi=150)
fig.savefig(out_png, bbox_inches='tight', dpi=150)
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')

# ── single combined figure: all guilds × weeks ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(16, 5))

n_w   = 3
width = 0.22
offsets = [-0.24, 0.0, 0.24]
x = np.arange(N_G)

for w in range(n_w):
    mask   = phi[:, w, :].sum(axis=1) > 1e-12
    medians = np.median(phi[mask, w, :] * 100, axis=0)
    q25     = np.percentile(phi[mask, w, :] * 100, 25, axis=0)
    q75     = np.percentile(phi[mask, w, :] * 100, 75, axis=0)
    mins    = np.min(phi[mask, w, :] * 100, axis=0)
    maxs    = np.max(phi[mask, w, :] * 100, axis=0)

    guild_cols = [GUILD_COLORS.get(g, '#aaaaaa') for g in GUILDS]
    pos = x + offsets[w]
    ax2.bar(pos, medians, width, color=guild_cols, alpha=WEEK_FILL_ALPHA + w * 0.15,
            label=WEEK_LABELS[w], edgecolor=WEEK_EDGE_COLORS[w],
            linewidth=1.2, hatch=WEEK_HATCHES[w])
    ax2.vlines(pos, mins, maxs, color='#333333', linewidth=1.0)
    ax2.vlines(pos, q25, q75, color=WEEK_EDGE_COLORS[w], linewidth=3.5, alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels([SHORT.get(g, g) for g in GUILDS], fontsize=10)
ax2.set_ylabel('Relative abundance (%)', fontsize=11)
ax2.set_title('Guild abundance by week — median ± IQR / range  (n≤8)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()

out2_pdf = OUT_DIR / 'guild_boxplot_combined.pdf'
out2_png = OUT_DIR / 'guild_boxplot_combined.png'
fig2.savefig(out2_pdf, bbox_inches='tight', dpi=150)
fig2.savefig(out2_png, bbox_inches='tight', dpi=150)
print(f'Saved: {out2_pdf}')
print(f'Saved: {out2_png}')
