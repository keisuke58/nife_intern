#!/usr/bin/env python3
"""
make_pca_lda_figs.py — publication-quality PCA and LDA figures for Dieckow paper.

Uses AGORA W=1.0 LOO fold data (A, b_ho) to simulate LOO predictions with
scipy RK45, then runs PCA/LDA on predicted vs observed compositions.

Output: figures/fig_pca_pred_obs.png, figures/fig_lda_pred_obs.png  (300 DPI)
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

HERE   = Path(__file__).resolve().parent
CR     = HERE.parent / 'results' / 'dieckow_cr'
OTU    = HERE.parent / 'results' / 'dieckow_otu'
OUTDIR = HERE / 'figures'

PATIENTS = list('ABCDEFGHKL')

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
G_SHORT = ['Actin.', 'Bacil.', 'Bact.', 'β-Prot.', 'Clost.',
           'Corio.', 'Fusob.', 'γ-Prot.', 'Negat.', 'Other']
N_G = 10

CT_MAP = {'A': 2, 'B': 2, 'C': 2, 'D': 1, 'E': 1,
          'F': 2, 'G': 1, 'H': 2, 'K': 1, 'L': 1}

NAVY  = '#1a3a5c'
BLUE  = '#2166ac'
RED   = '#d6604d'
GREEN = '#4dac26'
CT1_COLOR = BLUE
CT2_COLOR = RED

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.titleweight':   'bold',
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    8.5,
    'savefig.dpi':        300,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         100,
})


# ── Load observed data ─────────────────────────────────────────────────────────
phi_all = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10)


def replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    fmean = phi @ f
    return phi * (f - fmean)


def integrate_step(phi0, b, A, dt=1.0):
    sol = solve_ivp(replicator_rhs, [0, dt], phi0, args=(b, A),
                    method='RK45', rtol=1e-7, atol=1e-9)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


# ── Collect LOO predictions (AGORA W=1.0) ─────────────────────────────────────
obs_wk2, obs_wk3   = [], []
pred_wk2, pred_wk3 = [], []
pat_labels = []

for fold in range(10):
    path = CR / f'loo_expanded_agora_w1p0_fold{fold}.json'
    if not path.exists():
        print(f'  Missing fold {fold}, skipping')
        continue
    d = json.load(open(path))
    pat   = d['patient']
    p_idx = PATIENTS.index(pat)
    A     = np.array(d['A'])
    b     = np.array(d['b_ho'])

    phi0 = phi_all[p_idx, 0]
    p2   = integrate_step(phi0, b, A)
    p3   = integrate_step(p2,   b, A)

    obs_wk2.append(phi_all[p_idx, 1])
    obs_wk3.append(phi_all[p_idx, 2])
    pred_wk2.append(p2)
    pred_wk3.append(p3)
    pat_labels.append(pat)

obs_wk2  = np.array(obs_wk2)
obs_wk3  = np.array(obs_wk3)
pred_wk2 = np.array(pred_wk2)
pred_wk3 = np.array(pred_wk3)
pat_idx  = [PATIENTS.index(p) for p in pat_labels]
n_pat    = len(pat_labels)
print(f'Loaded {n_pat} folds: {pat_labels}')


def ct_color(pat):
    return CT2_COLOR if CT_MAP.get(pat, 1) == 2 else CT1_COLOR


def ct_marker(pat):
    return 's' if CT_MAP.get(pat, 1) == 2 else 'o'


# ════════════════════════════════════════════════════════════════════════════
# PCA figure
# ════════════════════════════════════════════════════════════════════════════
all_vecs = np.vstack([obs_wk2, obs_wk3, pred_wk2, pred_wk3])
scaler   = StandardScaler()
all_s    = scaler.fit_transform(all_vecs)

pca = PCA(n_components=2)
pca.fit(all_s)
ev = pca.explained_variance_ratio_ * 100

o2s = pca.transform(scaler.transform(obs_wk2))
o3s = pca.transform(scaler.transform(obs_wk3))
p2s = pca.transform(scaler.transform(pred_wk2))
p3s = pca.transform(scaler.transform(pred_wk3))

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
for ax, os, ps, week in zip(axes, [o2s, o3s], [p2s, p3s], [2, 3]):
    for i, (pi, pat) in enumerate(zip(pat_idx, pat_labels)):
        c  = ct_color(pat)
        mk = ct_marker(pat)
        ax.scatter(*os[i], marker=mk, s=80, color=c, zorder=4, alpha=0.9)
        ax.scatter(*ps[i], marker='D', s=55, color=c, zorder=4,
                   facecolors='none', linewidths=1.5, alpha=0.9)
        ax.annotate('', xy=ps[i], xytext=os[i],
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.0, alpha=0.5))
        ax.text(os[i, 0] + 0.06, os[i, 1] + 0.04, pat, fontsize=8, color=c, fontweight='bold')

    ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)')
    ax.set_title(f'({"AB"[week-2]})  PCA — Week {week}', fontweight='bold')
    ax.axhline(0, color='#cccccc', lw=0.6, zorder=0)
    ax.axvline(0, color='#cccccc', lw=0.6, zorder=0)

    # annotation
    ax.text(0.02, 0.97, '● / ■ = observed\n◇ = predicted (LOO)', transform=ax.transAxes,
            fontsize=7.5, va='top', color='#555555')

# shared legend: CT1 vs CT2
ct1_p = mpatches.Patch(color=CT1_COLOR, label='CT1 (commensal)')
ct2_p = mpatches.Patch(color=CT2_COLOR, label='CT2 (dysbiotic)')
axes[0].legend(handles=[ct1_p, ct2_p], loc='lower left', framealpha=0.9)

fig.tight_layout(pad=1.2)
out = OUTDIR / 'fig_pca_pred_obs.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out.name}  ({out.stat().st_size//1024} KB)')


# ════════════════════════════════════════════════════════════════════════════
# LDA figure
# ════════════════════════════════════════════════════════════════════════════
X_lda = np.vstack([obs_wk2, obs_wk3])
y_lda = pat_idx + pat_idx

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_lda, y_lda)

o2l = lda.transform(obs_wk2)
o3l = lda.transform(obs_wk3)
p2l = lda.transform(pred_wk2)
p3l = lda.transform(pred_wk3)

ev_lda = lda.explained_variance_ratio_ * 100 \
         if hasattr(lda, 'explained_variance_ratio_') else [np.nan, np.nan]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
for ax, ol, pl, week in zip(axes, [o2l, o3l], [p2l, p3l], [2, 3]):
    for i, (pi, pat) in enumerate(zip(pat_idx, pat_labels)):
        c  = ct_color(pat)
        mk = ct_marker(pat)
        ax.scatter(*ol[i], marker=mk, s=80, color=c, zorder=4, alpha=0.9)
        ax.scatter(*pl[i], marker='D', s=55, color=c, zorder=4,
                   facecolors='none', linewidths=1.5, alpha=0.9)
        ax.annotate('', xy=pl[i], xytext=ol[i],
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.0, alpha=0.5))
        ax.text(ol[i, 0] + 0.06, ol[i, 1] + 0.04, pat, fontsize=8, color=c, fontweight='bold')

    xlab = f'LD1 ({ev_lda[0]:.1f}%)' if not np.isnan(ev_lda[0]) else 'LD1'
    ylab = f'LD2 ({ev_lda[1]:.1f}%)' if not np.isnan(ev_lda[1]) else 'LD2'
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f'({"AB"[week-2]})  LDA — Week {week}', fontweight='bold')
    ax.axhline(0, color='#cccccc', lw=0.6, zorder=0)
    ax.axvline(0, color='#cccccc', lw=0.6, zorder=0)
    ax.text(0.02, 0.97, '● / ■ = observed\n◇ = predicted (LOO)', transform=ax.transAxes,
            fontsize=7.5, va='top', color='#555555')

axes[0].legend(handles=[ct1_p, ct2_p], loc='lower left', framealpha=0.9)

fig.tight_layout(pad=1.2)
out = OUTDIR / 'fig_lda_pred_obs.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out.name}  ({out.stat().st_size//1024} KB)')
