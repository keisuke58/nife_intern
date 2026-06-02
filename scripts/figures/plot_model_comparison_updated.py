#!/usr/bin/env python3
"""Updated model comparison including Hamilton SS v2 LOO-CV (10 patients)."""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / 'results' / 'dieckow_cr'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

# ── Load models ───────────────────────────────────────────────────────────────

def load_std(fname, label, color, patient_order):
    """Load standard LOO JSON (per_patient list format)."""
    fpath = RES / fname
    if not fpath.exists():
        return None
    dd = json.load(open(fpath))
    rmse_by_pat = {p['patient']: p['rmse'] for p in dd['per_patient']}
    vals = [rmse_by_pat.get(p, np.nan) for p in patient_order]
    return {'label': label, 'color': color,
            'mean': dd['loo_rmse_mean'], 'per_pat': vals, 'n_pat': len(dd['per_patient'])}

def load_ss_v2(fname, label, color, patient_order):
    """Load SS v2 aggregate JSON."""
    fpath = RES / fname
    if not fpath.exists():
        return None
    dd = json.load(open(fpath))
    vals = [dd['per_patient'].get(p, np.nan) for p in patient_order]
    return {'label': label, 'color': color,
            'mean': dd['mean'], 'per_pat': vals, 'n_pat': len(dd['per_patient'])}

# 8-patient models (ABCEFGHK — original labeling in these files)
PAT8 = list('ABCEFGHK')
# 10-patient SS v2
PAT10 = list('ABCDEFGHKL')

models_8 = [
    load_std('loo_cv_glv.json',               'gLV (free)',       '#4C8DC4', PAT8),
    load_std('loo_cv_glv_kegg_prior.json',    'gLV + KEGG',       '#5BB85B', PAT8),
    load_std('loo_cv_hamilton.json',          'Hamilton (free)',   '#E0873A', PAT8),
    load_std('loo_cv_hamilton_kegg_prior.json','Hamilton + KEGG',  '#9B59B6', PAT8),
]
models_8 = [m for m in models_8 if m is not None]

ss_v2 = load_ss_v2('loo_ss_v2_summary.json', 'Hamilton SS v2\n(10 pat.)', '#D62728', PAT10)

all_models_mean = models_8 + ([ss_v2] if ss_v2 else [])

fig, (ax_mean, ax_pat) = plt.subplots(1, 2, figsize=(13, 5),
                                       gridspec_kw={'wspace': 0.38, 'width_ratios': [1, 1.6]})

# ── Left: mean LOO-RMSE bar ───────────────────────────────────────────────────
labels = [m['label'] for m in all_models_mean]
means  = [m['mean']  for m in all_models_mean]
colors = [m['color'] for m in all_models_mean]

bars = ax_mean.bar(range(len(labels)), means, color=colors, alpha=0.88,
                   edgecolor='k', linewidth=0.6, width=0.6)
best_val = min(means)
for bar, v in zip(bars, means):
    star = ' ★' if v == best_val else ''
    ax_mean.text(bar.get_x() + bar.get_width() / 2, v + 0.0010,
                 f'{v:.4f}{star}',
                 ha='center', va='bottom', fontsize=8.5,
                 fontweight='bold' if star else 'normal')

# vertical separator before SS v2
if ss_v2:
    sep_x = len(models_8) - 0.5
    ax_mean.axvline(sep_x, color='#aaa', lw=0.8, ls='--')
    ax_mean.text(sep_x + 0.02, max(means) * 1.15, '10-patient', fontsize=7,
                 color='#666', va='top')

ax_mean.set_ylabel('LOO-CV RMSE')
ax_mean.set_title('Mean LOO-CV RMSE\nby model', fontweight='bold')
ax_mean.set_ylim(0, max(means) * 1.28)
ax_mean.set_xticks(range(len(labels)))
ax_mean.set_xticklabels(labels, rotation=20, ha='right', fontsize=8.5)

# ── Right: per-patient (8-patient models + SS v2 on 10 patients) ─────────────
# Show all 10 patients; 8-patient models have NaN for D and L
ALL_PATS = list('ABCDEFGHKL')

x = np.arange(len(ALL_PATS))
n_m = len(all_models_mean)
w_b = 0.8 / n_m
offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w_b

for ci, m in enumerate(all_models_mean):
    rmse_by = {}
    if m['label'].startswith('Hamilton SS v2'):
        rmse_by = {p: v for p, v in
                   zip(PAT10, m['per_pat']) if not np.isnan(v)}
    else:
        rmse_by = {p: v for p, v in
                   zip(PAT8, m['per_pat']) if not np.isnan(v)}
    vals = [rmse_by.get(p, np.nan) for p in ALL_PATS]
    # plot non-nan
    xs_plot = [x[i] + offsets[ci] for i in range(len(ALL_PATS)) if not np.isnan(vals[i])]
    vs_plot = [vals[i] for i in range(len(ALL_PATS)) if not np.isnan(vals[i])]
    label_clean = m['label'].replace('\n', ' ')
    ax_pat.bar(xs_plot, vs_plot, w_b, label=f"{label_clean} ({m['mean']:.4f})",
               color=m['color'], alpha=0.85, edgecolor='k', linewidth=0.3)

ax_pat.set_xticks(x)
ax_pat.set_xticklabels(ALL_PATS, fontsize=9)
ax_pat.set_xlabel('Patient')
ax_pat.set_ylabel('LOO-CV RMSE')
ax_pat.set_title('Per-patient LOO-CV RMSE\n(8-patient models: D,L missing)', fontweight='bold')
ax_pat.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc='upper right',
              title='Model (mean)', title_fontsize=7.5)

fig.suptitle('Guild model LOO-CV comparison — gLV / Hamilton × free / KEGG / SS v2',
             fontsize=11, fontweight='bold', y=1.01)

for ext in ('pdf', 'png'):
    fig.savefig(RES / f'fig_guild_model_comparison.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved fig_guild_model_comparison.*')
