#!/usr/bin/env python3
"""
plot_pca_lda_pred_obs.py — visualise LOO predictions vs observations via:
  Fig 1: PCA — pred (◇) and obs (●) overlaid, same colour per patient,
               arrows obs→pred, separate panels for wk2 and wk3
  Fig 2: LDA — discriminant axes trained on obs compositions (patient labels),
               pred projected onto the same axes

Usage:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python plot_pca_lda_pred_obs.py
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family':'serif',
                            'font.serif':['Times New Roman','Times','DejaVu Serif'],
                            'mathtext.fontset':'stix', 'font.size':10})
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
from hamilton_ode_jax_nsp import simulate_0d_nsp
from guild_replicator_dieckow import GUILD_ORDER, N_G

CR       = HERE / 'results' / 'dieckow_cr'
OTU      = HERE / 'results' / 'dieckow_otu'
PATIENTS = list('ABCDEFGHKL')

phi_all = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10)

COLORS = plt.cm.tab10.colors[:10]

def pack_theta_A(A_mat):
    n = len(A_mat)
    return jnp.array([A_mat[i][j] for j in range(n) for i in range(j + 1)])


# ── Collect LOO predictions ───────────────────────────────────────────────────
obs_wk2, obs_wk3 = [], []
pred_wk2, pred_wk3 = [], []
pat_labels = []

for fold in range(10):
    path = CR / f'loo_expanded_a0p0_fold{fold}.json'
    if not path.exists():
        continue
    d = json.load(open(path))
    p_idx   = PATIENTS.index(d['patient'])
    theta_A = pack_theta_A(np.array(d['A']))
    b_ho    = jnp.array(d['b_ho'])
    theta   = jnp.concatenate([theta_A, b_ho])
    phi0    = jnp.array(phi_all[p_idx, 0])

    @jax.jit
    def step(phi):
        eq = simulate_0d_nsp(theta, n_sp=N_G, n_steps=100, dt=1e-4,
                             phi_init=phi, c_const=25.0, alpha_const=100.0)[-1]
        s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)

    p2 = np.array(step(phi0))
    p3 = np.array(step(jnp.array(p2)))

    obs_wk2.append(phi_all[p_idx, 1])
    obs_wk3.append(phi_all[p_idx, 2])
    pred_wk2.append(p2)
    pred_wk3.append(p3)
    pat_labels.append(d['patient'])

obs_wk2  = np.array(obs_wk2)
obs_wk3  = np.array(obs_wk3)
pred_wk2 = np.array(pred_wk2)
pred_wk3 = np.array(pred_wk3)
pat_idx  = [PATIENTS.index(p) for p in pat_labels]
n_pat    = len(pat_labels)

print(f'Loaded {n_pat} folds: {pat_labels}')


# ════════════════════════════════════════════════════════════════════════════
# Fig 1: PCA
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

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, os, ps, week in zip(axes, [o2s, o3s], [p2s, p3s], [2, 3]):
    for i, (pi, pat) in enumerate(zip(pat_idx, pat_labels)):
        c = COLORS[pi]
        # observed
        ax.scatter(*os[i], marker='o', s=90, color=c, zorder=4,
                   label=f'{pat}' if week == 2 else None)
        # predicted
        ax.scatter(*ps[i], marker='D', s=70, color=c, zorder=4,
                   facecolors='none', linewidths=1.8)
        # arrow obs → pred
        ax.annotate('', xy=ps[i], xytext=os[i],
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.2, alpha=0.6))
        # patient label at obs
        ax.text(os[i, 0] + 0.04, os[i, 1] + 0.04, pat, fontsize=7, color=c)

    ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)', fontsize=11)
    ax.set_title(f'PCA — Week {week}  (●=obs  ◇=pred)', fontsize=12)
    ax.grid(True, alpha=0.25)

# single legend
handles = [mpatches.Patch(color=COLORS[PATIENTS.index(p)], label=p)
           for p in pat_labels]
axes[0].legend(handles=handles, fontsize=8, loc='best', ncol=2,
               title='Patient', title_fontsize=8)

# loadings inset (wk2 panel)
ax = axes[0]
scale = 0.8
for g, guild in enumerate(GUILD_ORDER):
    dx, dy = pca.components_[0, g] * scale, pca.components_[1, g] * scale
    ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='grey', lw=0.9))
    ax.text(dx * 1.12, dy * 1.12, guild[:4], fontsize=6, color='grey', ha='center')

plt.tight_layout()
fig.savefig(CR / 'fig_pca_pred_obs.png', dpi=300)
fig.savefig(CR / 'fig_pca_pred_obs.pdf')
print('Saved fig_pca_pred_obs.png')


# ════════════════════════════════════════════════════════════════════════════
# Fig 2: LDA
# ════════════════════════════════════════════════════════════════════════════
# Train LDA on obs (wk2 + wk3 combined) with patient label
X_lda = np.vstack([obs_wk2, obs_wk3])
y_lda = pat_idx + pat_idx   # patient labels for each row

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_lda, y_lda)

o2l = lda.transform(obs_wk2)
o3l = lda.transform(obs_wk3)
p2l = lda.transform(pred_wk2)
p3l = lda.transform(pred_wk3)

# explained variance proxy via eigenvalues
ev_lda = lda.explained_variance_ratio_ * 100 if hasattr(lda, 'explained_variance_ratio_') \
         else [np.nan, np.nan]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, ol, pl, week in zip(axes, [o2l, o3l], [p2l, p3l], [2, 3]):
    for i, (pi, pat) in enumerate(zip(pat_idx, pat_labels)):
        c = COLORS[pi]
        ax.scatter(*ol[i], marker='o', s=90, color=c, zorder=4)
        ax.scatter(*pl[i], marker='D', s=70, color=c, zorder=4,
                   facecolors='none', linewidths=1.8)
        ax.annotate('', xy=pl[i], xytext=ol[i],
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.2, alpha=0.6))
        ax.text(ol[i, 0] + 0.06, ol[i, 1] + 0.06, pat, fontsize=7, color=c)

    ax.set_xlabel(f'LD1 ({ev_lda[0]:.1f}%)', fontsize=11)
    ax.set_ylabel(f'LD2 ({ev_lda[1]:.1f}%)', fontsize=11)
    ax.set_title(f'LDA — Week {week}  (●=obs  ◇=pred)', fontsize=12)
    ax.grid(True, alpha=0.25)

axes[0].legend(handles=handles, fontsize=8, loc='best', ncol=2,
               title='Patient', title_fontsize=8)

# LDA loadings (scalings)
ax = axes[0]
sc = np.abs(lda.scalings_).max()
scale_lda = 1.5 / sc
for g, guild in enumerate(GUILD_ORDER):
    dx = lda.scalings_[g, 0] * scale_lda
    dy = lda.scalings_[g, 1] * scale_lda
    ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='grey', lw=0.9))
    ax.text(dx * 1.15, dy * 1.15, guild[:4], fontsize=6, color='grey', ha='center')

plt.tight_layout()
fig.savefig(CR / 'fig_lda_pred_obs.png', dpi=300)
fig.savefig(CR / 'fig_lda_pred_obs.pdf')
print('Saved fig_lda_pred_obs.png')

# ── Summary: distance obs→pred in PCA space ──────────────────────────────────
print('\nEuclidean distance obs→pred in PCA space (smaller = better):')
for week, os, ps in [('wk2', o2s, p2s), ('wk3', o3s, p3s)]:
    dists = np.linalg.norm(os - ps, axis=1)
    for i, (pat, d) in enumerate(zip(pat_labels, dists)):
        print(f'  {pat} {week}: {d:.3f}')
    print(f'  mean {week}: {dists.mean():.3f}')
