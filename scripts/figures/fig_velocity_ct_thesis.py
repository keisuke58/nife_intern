# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

"""
Velocity direction test: CT1 vs CT2 — thesis figure version.
Run with TeX Live on PATH:
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python scripts/figures/fig_velocity_ct_thesis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import ttest_ind, mannwhitneyu

from thesis_style import use, PALETTE, clean_ax
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

ROOT   = Path(__file__).parents[2]
RESDIR = ROOT / 'results' / 'dieckow_cr'
OUT    = RESDIR / 'figs'
OUT.mkdir(exist_ok=True)

GUILDS   = GUILD_ORDER
GUILDS_9 = [g for g in GUILDS if g != 'Other']

COL_CT1 = PALETTE['health']      # teal
COL_CT2 = PALETTE['dysbiosis']   # orange

# ── load fits ─────────────────────────────────────────────────────────────

def prep_fit(path):
    fit  = json.load(open(path))
    glds = fit['guilds']
    idx  = [glds.index(g) for g in GUILDS]
    A9i  = [GUILDS.index(g) for g in GUILDS_9]
    A    = np.array(fit['A'])[np.ix_(idx, idx)][np.ix_(A9i, A9i)]
    b9   = np.array(fit['b_all'])[:, idx][:, A9i]
    return A, {p: b9[i] for i, p in enumerate(fit['patients'])}, fit['patients']

A_glv, b_glv, pats = prep_fit(RESDIR / 'fit_glv_hamilton_kegg_expanded.json')
A_ham, b_ham, _    = prep_fit(RESDIR / 'fit_guild_hamilton.json')

# ── load data ─────────────────────────────────────────────────────────────

df = pd.read_csv(RESDIR / 'dieckow_guild_abundance.csv')
ct_pred  = json.load(open(RESDIR / 'cross_ct_prediction.json'))
ct1_pats = ct_pred['ct1_patients']
df['pat'] = df['sample'].str[0]
df['tp']  = df['sample'].str[2].astype(int)
phi_raw   = df[GUILDS_9].values.clip(0)
df[GUILDS_9] = phi_raw / (phi_raw.sum(axis=1, keepdims=True) + 1e-12)

def rhs(phi, b, A):
    phi = np.clip(phi, 0, None); phi /= phi.sum() + 1e-12
    r   = b + A @ phi
    return phi * (r - phi @ r)

def cos_sim(obs, pred):
    no, np_ = np.linalg.norm(obs), np.linalg.norm(pred)
    return (obs @ pred) / (no * np_) if no > 1e-8 and np_ > 1e-8 else np.nan

# ── collect records ───────────────────────────────────────────────────────

records = []
for pat in pats:
    rows = df[df['pat'] == pat].sort_values('tp')
    phis = rows[GUILDS_9].values
    ct   = 'CT1' if pat in ct1_pats else 'CT2'
    for model_name, A, b_p_dict in [('gLV', A_glv, b_glv),
                                      ('Hamilton', A_ham, b_ham)]:
        b_p = b_p_dict[pat]
        for ti in range(len(phis) - 1):
            obs  = phis[ti + 1] - phis[ti]
            pred = rhs(phis[ti], b_p, A)
            c    = cos_sim(obs, pred)
            if not np.isnan(c):
                records.append({'pat': pat, 'ct': ct, 'model': model_name,
                                'interval': f'wk{ti+1}'+r'$\to$'+f'wk{ti+2}',
                                'ti': ti, 'cos': c})

rec = pd.DataFrame(records)
ham = rec[rec['model'] == 'Hamilton']
ct1_cos = ham[ham['ct'] == 'CT1']['cos'].values
ct2_cos = ham[ham['ct'] == 'CT2']['cos'].values
_, p_mw = mannwhitneyu(ct1_cos, ct2_cos, alternative='greater')
_, p_t  = ttest_ind(ct1_cos, ct2_cos)

# ── figure: 1×3 layout ────────────────────────────────────────────────────

figsize = use(width_frac=1.0, aspect=0.42)
fig, axes = plt.subplots(1, 3, figsize=figsize,
                          gridspec_kw=dict(wspace=0.48, left=0.09, right=0.97,
                                           top=0.84, bottom=0.20))

ax_bar, ax_box, ax_iv = axes

# ─── Panel A: per-patient mean cosine (Hamilton) ──────────────────────────

pat_mean = (ham.groupby(['pat', 'ct'])['cos'].mean()
              .reset_index()
              .sort_values(['ct', 'cos'], ascending=[True, False]))

x = np.arange(len(pat_mean))
cols = [COL_CT1 if r['ct'] == 'CT1' else COL_CT2 for _, r in pat_mean.iterrows()]
ax_bar.bar(x, pat_mean['cos'], color=cols, edgecolor='k', lw=0.4, alpha=0.85, width=0.65)
ax_bar.axhline(0, color='k', lw=0.6)

for ct, col in [('CT1', COL_CT1), ('CT2', COL_CT2)]:
    m = pat_mean[pat_mean['ct'] == ct]['cos'].mean()
    ax_bar.axhline(m, color=col, lw=1.2, ls='--', alpha=0.8)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(
    [r'\textbf{' + r['pat'] + r'}' for _, r in pat_mean.iterrows()],
    fontsize=7)
ax_bar.set_ylabel(r'Mean cosine similarity $\langle\cos\theta\rangle$')
ax_bar.set_title(r'\textbf{A}\enspace Per-patient velocity accuracy (Hamilton)')
clean_ax(ax_bar)

ax_bar.legend(handles=[Patch(color=COL_CT1, label='CT1 (commensal)'),
                        Patch(color=COL_CT2, label='CT2 (dysbiotic)')],
              fontsize=7, loc='lower right', frameon=False)

ct1_mean = pat_mean[pat_mean['ct']=='CT1']['cos'].mean()
ct2_mean = pat_mean[pat_mean['ct']=='CT2']['cos'].mean()
ax_bar.text(0.02, 0.97,
            f'CT1 mean $= {ct1_mean:+.2f}$\nCT2 mean $= {ct2_mean:+.2f}$',
            transform=ax_bar.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbb', lw=0.6))

# ─── Panel B: box CT1 vs CT2 ─────────────────────────────────────────────

bp = ax_box.boxplot([ct1_cos, ct2_cos], patch_artist=True,
                    medianprops=dict(color='k', lw=1.2),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    flierprops=dict(marker='o', markersize=3, lw=0.5))
for patch, col in zip(bp['boxes'], [COL_CT1, COL_CT2]):
    patch.set_facecolor(col); patch.set_alpha(0.65)

rng = np.random.default_rng(42)
for i, (cos_arr, col) in enumerate([(ct1_cos, COL_CT1), (ct2_cos, COL_CT2)]):
    jit = rng.uniform(-0.12, 0.12, len(cos_arr))
    ax_box.scatter(np.full(len(cos_arr), i + 1) + jit, cos_arr,
                   color=col, s=18, alpha=0.9, edgecolors='k', lw=0.3, zorder=3)

ax_box.axhline(0, color='k', lw=0.6, ls='--')
ax_box.set_xticks([1, 2])
ax_box.set_xticklabels(['CT1\n(commensal)', 'CT2\n(dysbiotic)'])
ax_box.set_ylabel(r'Cosine similarity')
ax_box.set_title(r'\textbf{B}\enspace CT1 vs.\ CT2 (all intervals)')
clean_ax(ax_box)

# significance bracket
y_br = max(ct1_cos.max(), ct2_cos.max()) + 0.06
ax_box.plot([1, 1, 2, 2], [y_br, y_br+0.04, y_br+0.04, y_br], lw=0.8, color='k')
sig_str = r'$\dagger$' if p_t < 0.1 else 'n.s.'
ax_box.text(1.5, y_br + 0.06, f'{sig_str} $p={p_t:.3f}$', ha='center', fontsize=7)

pos_ct1 = np.mean(ct1_cos > 0)
pos_ct2 = np.mean(ct2_cos > 0)
ax_box.text(0.05, 0.05,
            f'CT1: ${pos_ct1*100:.0f}\\%$ correct\n'
            f'CT2: ${pos_ct2*100:.0f}\\%$ correct',
            transform=ax_box.transAxes, fontsize=7, va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbb', lw=0.6))

# ─── Panel C: per-interval × CT (Hamilton only) ───────────────────────────

intervals_raw = ['wk1→wk2', 'wk2→wk3']
intervals_tex = [r'wk1$\to$wk2', r'wk2$\to$wk3']
x4 = np.arange(len(intervals_raw))
w  = 0.28

for ci, (ct, col) in enumerate([('CT1', COL_CT1), ('CT2', COL_CT2)]):
    means, errs = [], []
    for iv in intervals_raw:
        sub = rec[(rec['model']=='Hamilton') & (rec['ct']==ct) &
                  (rec['interval'].str.contains(iv.split('→')[0]))]['cos']
        means.append(sub.mean() if len(sub) else np.nan)
        errs.append(sub.std()/np.sqrt(len(sub)) if len(sub) > 1 else 0)
    offset = (ci - 0.5) * w
    ax_iv.bar(x4 + offset, means, w, yerr=errs, color=col, alpha=0.85,
              edgecolor='k', lw=0.4, capsize=3,
              error_kw=dict(lw=0.8), label=f'CT{ci+1}')

    # p-value annotation (CT1 vs CT2 per interval)
for mi, iv in enumerate(intervals_raw):
    s1 = rec[(rec['model']=='Hamilton')&(rec['ct']=='CT1')&
             (rec['interval'].str.contains(iv.split('→')[0]))]['cos']
    s2 = rec[(rec['model']=='Hamilton')&(rec['ct']=='CT2')&
             (rec['interval'].str.contains(iv.split('→')[0]))]['cos']
    if len(s1) > 1 and len(s2) > 1:
        _, pv = ttest_ind(s1, s2)
        lbl = r'$\dagger$' if pv < 0.1 else 'n.s.'
        ax_iv.text(mi, 0.97, f'{lbl} $p={pv:.2f}$', ha='center', fontsize=6.5,
                   transform=ax_iv.get_xaxis_transform(), va='top')

ax_iv.axhline(0, color='k', lw=0.6)
ax_iv.set_xticks(x4)
ax_iv.set_xticklabels(intervals_tex)
ax_iv.set_ylabel(r'Mean cosine similarity')
ax_iv.set_title(r'\textbf{C}\enspace Per-interval breakdown (Hamilton)')
ax_iv.legend(fontsize=7, frameon=False, loc='lower right')
clean_ax(ax_iv)

fig.suptitle(
    r'Velocity direction test: gLV model predicts commensal dynamics, '
    r'not dysbiotic dynamics',
    fontsize=9)

for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'fig_velocity_ct_thesis.{ext}', dpi=300)

plt.close()
print("Done →", OUT / 'fig_velocity_ct_thesis.pdf')
