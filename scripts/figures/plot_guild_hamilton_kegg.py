#!/usr/bin/env python3
"""
Figure: Hamilton KEGG-prior guild model results (publication quality).

Outputs:
  results/dieckow_cr/fig_guild_hamilton_kegg_Amatrix.pdf/.png
  results/dieckow_cr/fig_guild_hamilton_kegg_trajectory.pdf/.png
  results/dieckow_cr/fig_guild_model_comparison.pdf/.png
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
RES  = HERE / 'results' / 'dieckow_cr'

from guild_replicator_dieckow import (
    GUILD_ORDER, GUILD_SHORT_LIST, GUILD_COLORS_LIST,
    N_G, predict_trajectory, DT
)
from loo_cv_kegg_prior import build_net_flow, PHI_NPY, PATIENTS

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

GUILD_SHORT = GUILD_SHORT_LIST   # e.g. ['Actin.', 'Bacil.', ...]

# ── Load data ─────────────────────────────────────────────────────────────────

d_ham_kegg = json.load(open(RES / 'fit_glv_hamilton_kegg_prior.json'))
d_ham_free = json.load(open(RES / 'fit_guild_hamilton_masked.json'))
d_glv_kegg = json.load(open(RES / 'fit_glv_8pat_kegg_prior.json'))

A_ham_kegg = np.array(d_ham_kegg['A'])
A_ham_free = np.array(d_ham_free['A'])
A_glv_kegg = np.array(d_glv_kegg['A'])

net_flow = build_net_flow()
net_sym  = (net_flow + net_flow.T) / 2.0
sp_sym   = np.sign(net_sym)

phi_all = np.load(PHI_NPY)
phi_sub = np.stack([phi_all[k] for k in range(phi_all.shape[0])
                    if phi_all[k].sum() > 1e-9])[:len(PATIENTS)]


# ── Fig 1: A matrix heatmap (3 panels) ───────────────────────────────────────

def draw_A_heatmap(ax, A, title, subtitle, sp_sym, vmax=2.0):
    cmap = plt.cm.RdBu_r
    im = ax.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')

    ax.set_xticks(range(N_G))
    ax.set_xticklabels(GUILD_SHORT, fontsize=7.5, rotation=40, ha='right')
    ax.set_yticks(range(N_G))
    ax.set_yticklabels(GUILD_SHORT, fontsize=7.5)
    full_title = f'{title}\n{subtitle}' if subtitle else title
    ax.set_title(full_title, fontsize=10, fontweight='bold', pad=6,
                 linespacing=1.5)

    # cell value annotations
    for i in range(N_G):
        for j in range(N_G):
            v  = A[i, j]
            tc = 'white' if abs(v) > vmax * 0.65 else '#222222'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=5.5, color=tc)

    # KEGG sign borders
    for i in range(N_G):
        for j in range(N_G):
            if i == j or sp_sym[i, j] == 0:
                continue
            ok    = np.sign(A[i, j]) == sp_sym[i, j]
            color = '#00cc00' if ok else '#dd0000'
            for x0, y0, dx, dy in [
                    (j-.48, i-.48, .96, 0), (j-.48, i+.48, .96, 0),
                    (j-.48, i-.48, 0, .96), (j+.48, i-.48, 0, .96)]:
                ax.plot([x0, x0+dx], [y0, y0+dy],
                        color=color, lw=1.5, clip_on=True, zorder=3)

    ax.set_xlim(-0.5, N_G - 0.5)
    ax.set_ylim(N_G - 0.5, -0.5)
    return im


fig1 = plt.figure(figsize=(15, 5.8))
gs   = GridSpec(1, 4, figure=fig1, width_ratios=[1, 1, 1, 0.055],
                left=0.06, right=0.96, top=0.83, bottom=0.17, wspace=0.35)

axes_f1 = [fig1.add_subplot(gs[0, k]) for k in range(3)]
cax     = fig1.add_subplot(gs[0, 3])

panels = [
    (A_ham_kegg,
     'Hamilton + KEGG prior',
     f'RMSE = {d_ham_kegg["rmse"]:.4f}   r = {d_ham_kegg["r"]:.3f}   SA = {d_ham_kegg["sign_agreement"]}'),
    (A_ham_free,
     'Hamilton — free prior',
     '(warm start for KEGG fit)'),
    (A_glv_kegg,
     'gLV + KEGG prior',
     '(reference)'),
]

im_last = None
for ax, (A, title, subtitle) in zip(axes_f1, panels):
    im_last = draw_A_heatmap(ax, A, title, subtitle, sp_sym)

cb = fig1.colorbar(im_last, cax=cax, orientation='vertical')
cb.set_label('A[i, j]', fontsize=9)
cb.ax.tick_params(labelsize=8)

patch_ok  = mpatches.Patch(facecolor='none', edgecolor='#00cc00', lw=1.5,
                            label='KEGG-supported sign ✓')
patch_bad = mpatches.Patch(facecolor='none', edgecolor='#dd0000', lw=1.5,
                            label='Discordant sign ✗')
fig1.legend(handles=[patch_ok, patch_bad], loc='lower center', ncol=2,
            fontsize=9, frameon=False, bbox_to_anchor=(0.47, 0.01))

fig1.suptitle(
    'Guild-level interaction matrices — Hamilton KEGG-prior vs baselines',
    fontsize=12, fontweight='bold', y=0.97)

for ext in ('pdf', 'png'):
    fig1.savefig(RES / f'fig_guild_hamilton_kegg_Amatrix.{ext}', dpi=200, bbox_inches='tight')
plt.close(fig1)
print('Saved fig_guild_hamilton_kegg_Amatrix.*')


# ── Fig 2: Predicted trajectories ────────────────────────────────────────────

b_all = np.array(d_ham_kegg['b_all'])
n_pat = len(PATIENTS)
x_pos = np.arange(N_G)
w     = 0.38

fig2, axes2 = plt.subplots(
    n_pat, 2, figsize=(11, 2.3 * n_pat),
    gridspec_kw={'hspace': 0.58, 'wspace': 0.3})

guild_colors = GUILD_COLORS_LIST

for pi, pat in enumerate(PATIENTS):
    phi_obs = phi_sub[pi]
    b_p     = b_all[pi]
    phi2_pred, phi3_pred = predict_trajectory(phi_obs[0], b_p, A_ham_kegg)

    for wi, (phi_pred, phi_true, wk) in enumerate([
            (phi2_pred, phi_obs[1], 'Week 2'),
            (phi3_pred, phi_obs[2], 'Week 3')]):
        ax = axes2[pi, wi]
        rmse_w = float(np.sqrt(np.mean((phi_pred - phi_true)**2)))
        r_w    = float(np.corrcoef(phi_true, phi_pred)[0, 1])

        ax.bar(x_pos - w/2, phi_true, w, color=guild_colors,
               alpha=0.9,  edgecolor='k', linewidth=0.4)
        ax.bar(x_pos + w/2, phi_pred, w, color=guild_colors,
               alpha=0.55, edgecolor='k', linewidth=0.4, hatch='//')

        ax.set_xlim(-0.6, N_G - 0.4)
        ymax = max(phi_true.max(), phi_pred.max()) * 1.35 + 0.04
        ax.set_ylim(0, min(1.0, ymax))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(GUILD_SHORT, fontsize=6.5, rotation=38, ha='right')
        ax.tick_params(axis='y', labelsize=7)
        ax.set_ylabel('Fraction', fontsize=7.5)
        ax.set_title(f'Patient {pat}  —  {wk}    RMSE = {rmse_w:.4f}',
                     fontsize=8.5, pad=4)
        ax.text(0.985, 0.96, f'r = {r_w:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5, color='#555555')

# single legend
from matplotlib.patches import Patch
leg_handles = [
    Patch(facecolor='#888888', edgecolor='k', lw=0.4, alpha=0.9,  label='Observed'),
    Patch(facecolor='#888888', edgecolor='k', lw=0.4, alpha=0.55,
          hatch='//', label='Predicted'),
]
fig2.legend(handles=leg_handles, loc='upper right', bbox_to_anchor=(0.995, 0.999),
            fontsize=9, frameon=True, framealpha=0.92)

fig2.suptitle(
    f'Hamilton + KEGG prior — predicted vs observed guild fractions\n'
    f'Overall: RMSE = {d_ham_kegg["rmse"]:.4f},  r = {d_ham_kegg["r"]:.3f},  '
    f'Sign agreement = {d_ham_kegg["sign_agreement"]} (100%)',
    fontsize=11, fontweight='bold', y=1.005)

for ext in ('pdf', 'png'):
    fig2.savefig(RES / f'fig_guild_hamilton_kegg_trajectory.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig2)
print('Saved fig_guild_hamilton_kegg_trajectory.*')


# ── Fig 3: Model comparison ───────────────────────────────────────────────────

loo_files = [
    ('loo_cv_glv.json',               'gLV (free)',      '#4C8DC4'),
    ('loo_cv_glv_kegg_prior.json',    'gLV + KEGG',      '#5BB85B'),
    ('loo_cv_hamilton.json',          'Hamilton (free)',  '#E0873A'),
    ('loo_cv_hamilton_kegg_prior.json','Hamilton + KEGG', '#9B59B6'),
]

loo_results = {}
for fname, label, color in loo_files:
    fpath = RES / fname
    if fpath.exists():
        dd = json.load(open(fpath))
        loo_results[label] = {
            'mean':    dd['loo_rmse_mean'],
            'per_pat': [p['rmse'] for p in dd['per_patient']],
            'color':   color,
        }

models = list(loo_results.keys())
means  = [loo_results[m]['mean'] for m in models]
colors = [loo_results[m]['color'] for m in models]

fig3, (ax_mean, ax_pat) = plt.subplots(
    1, 2, figsize=(11, 4.5), gridspec_kw={'wspace': 0.38})

# left: mean bar
bars = ax_mean.bar(models, means, color=colors, alpha=0.88,
                   edgecolor='k', linewidth=0.6, width=0.55)
best_val = min(means)
for bar, m, v in zip(bars, models, means):
    star = '  ★ best' if v == best_val else ''
    ax_mean.text(bar.get_x() + bar.get_width() / 2, v + 0.0008,
                 f'{v:.4f}{star}',
                 ha='center', va='bottom', fontsize=8.5,
                 fontweight='bold' if star else 'normal')

ax_mean.set_ylabel('LOO-CV RMSE', fontsize=9)
ax_mean.set_title('Mean LOO-CV RMSE', fontsize=10, fontweight='bold')
ax_mean.set_ylim(0, max(means) * 1.22)
ax_mean.set_xticks(range(len(models)))
ax_mean.set_xticklabels(models, rotation=15, ha='right', fontsize=8.5)

if 'Hamilton + KEGG' not in loo_results:
    ax_mean.text(0.5, 0.95, '(Hamilton + KEGG LOO-CV in progress…)',
                 transform=ax_mean.transAxes, ha='center', va='top',
                 fontsize=8, color='#888888', style='italic')

# right: per-patient
n_m     = len(models)
x       = np.arange(len(PATIENTS))
w_b     = 0.16
offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w_b

for ci, (m, off) in enumerate(zip(models, offsets)):
    vals = loo_results[m]['per_pat']
    ax_pat.bar(x + off, vals, w_b, label=m,
               color=loo_results[m]['color'], alpha=0.85,
               edgecolor='k', linewidth=0.3)

ax_pat.set_xticks(x)
ax_pat.set_xticklabels(PATIENTS, fontsize=9)
ax_pat.set_xlabel('Patient', fontsize=9)
ax_pat.set_ylabel('LOO-CV RMSE', fontsize=9)
ax_pat.set_title('Per-patient LOO-CV RMSE', fontsize=10, fontweight='bold')
ax_pat.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc='upper right')

fig3.suptitle('Guild model comparison: gLV vs Hamilton × free prior vs KEGG prior',
              fontsize=11, fontweight='bold', y=1.02)

for ext in ('pdf', 'png'):
    fig3.savefig(RES / f'fig_guild_model_comparison.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig3)
print('Saved fig_guild_model_comparison.*')

print('Done.')
