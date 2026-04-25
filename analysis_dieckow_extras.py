#!/usr/bin/env python3
"""
analysis_dieckow_extras.py — four extra analyses on the Dieckow guild data.

1. Structural × Composition Spearman correlation heatmap
2. A-matrix stability: eigenvalues + attractor landscape (100 random ICs)
3. Patient PCA / clustering (structural + composition features)
4. Alpha-diversity (Shannon) time series W1→W2→W3

Output: results/dieckow_analysis/  (fig_corr, fig_stability, fig_cluster, fig_diversity)
"""

import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS_LIST, integrate_step
from load_structure_dieckow import load_structural_data, build_occupancy

OUT = _here / 'results' / 'dieckow_analysis'
OUT.mkdir(parents=True, exist_ok=True)

PHI_NPY   = _here / 'results' / 'dieckow_otu' / 'phi_guild_excel_class.npy'
FIT_JSON  = _here / 'results' / 'dieckow_cr'  / 'fit_guild_excel_class.json'
STRUCT_XL = _here / 'Datasets' / 'Abutment_Structure vs composition.xlsx'

PATIENTS_ALL = list('ABCDEFGHKL')   # 10 patients after removing I,J

# ── load data ─────────────────────────────────────────────────────────────────

phi_all = np.load(PHI_NPY)          # (10, 3, 11)
n_p, n_w, n_g = phi_all.shape
guilds = GUILD_ORDER[:n_g]
colors = GUILD_COLORS_LIST[:n_g]

present = phi_all.sum(axis=2) > 1e-12
keep    = present[:, 0]
phi_all  = phi_all[keep]
patients = [p for k, p in zip(keep.tolist(), PATIENTS_ALL) if k]

d       = json.load(open(FIT_JSON))
A_glv   = np.array(d['A'])
b_all   = np.array(d['b_all'])      # (n_patients, n_g_fit)
n_g_fit = A_glv.shape[0]

struct  = load_structural_data(STRUCT_XL)
occ_raw, max_occ = build_occupancy(struct, normalize=True)
pl_raw  = struct.get('PerLive', {})

n_keep  = len(patients)
occ_arr = np.ones((n_keep, n_w))
pl_arr  = np.ones((n_keep, n_w))
for p_idx, pat in enumerate(patients):
    for w in range(n_w):
        key = (pat, w + 1)
        occ_arr[p_idx, w] = occ_raw.get(key, 1.0)
        pl_arr[p_idx, w]  = pl_raw.get(key, 100.0) / 100.0

print(f'Loaded: {n_keep} patients, {n_g} guilds, occ∈[{occ_arr.min():.3f},{occ_arr.max():.3f}]')

# ══════════════════════════════════════════════════════════════════════════════
# 1. Structural × Composition Spearman correlation
# ══════════════════════════════════════════════════════════════════════════════

print('Fig 1: structural × composition correlations ...', flush=True)

# flatten to (n_keep * n_w,) vectors
phi_flat = phi_all.reshape(-1, n_g)     # (n_p*3, n_g)
occ_flat = occ_arr.reshape(-1)
pl_flat  = pl_arr.reshape(-1)

corr_occ = np.array([stats.spearmanr(phi_flat[:, g], occ_flat).statistic for g in range(n_g)])
corr_pl  = np.array([stats.spearmanr(phi_flat[:, g], pl_flat ).statistic for g in range(n_g)])
pval_occ = np.array([stats.spearmanr(phi_flat[:, g], occ_flat).pvalue    for g in range(n_g)])
pval_pl  = np.array([stats.spearmanr(phi_flat[:, g], pl_flat ).pvalue    for g in range(n_g)])

corr_mat = np.column_stack([corr_occ, corr_pl])   # (n_g, 2)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 3]})

# heatmap
ax = axes[0]
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = ax.imshow(corr_mat, aspect='auto', cmap='RdBu_r', norm=norm)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Occupancy\n(ω̃)', 'PerLive\n(q)'], fontsize=9)
ax.set_yticks(range(n_g)); ax.set_yticklabels(guilds, fontsize=8)
ax.set_title('Spearman ρ\nStructure × Composition', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.05)
# mark significant (p<0.05) with *
for g in range(n_g):
    for c, pv in enumerate([pval_occ[g], pval_pl[g]]):
        if pv < 0.05:
            ax.text(c, g, '*', ha='center', va='center', fontsize=12, color='black')

# scatter per guild, color by sign
ax2 = axes[1]
x = np.arange(n_g)
w_bar = 0.38
bars1 = ax2.bar(x - w_bar/2, corr_occ, w_bar, label='Occupancy ω̃', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + w_bar/2, corr_pl,  w_bar, label='PerLive q',   color='tomato',    alpha=0.8)
ax2.axhline(0, color='k', lw=0.7)
ax2.set_xticks(x); ax2.set_xticklabels(guilds, rotation=40, ha='right', fontsize=8)
ax2.set_ylabel('Spearman ρ', fontsize=9)
ax2.set_title('Per-guild correlation with CLSM structural parameters', fontsize=9)
ax2.legend(fontsize=8)
ax2.set_ylim(-1, 1)
# p-value markers
for g in range(n_g):
    if pval_occ[g] < 0.05:
        ax2.text(g - w_bar/2, corr_occ[g] + 0.03 * np.sign(corr_occ[g]), '*', ha='center', fontsize=9)
    if pval_pl[g] < 0.05:
        ax2.text(g + w_bar/2, corr_pl[g]  + 0.03 * np.sign(corr_pl[g]),  '*', ha='center', fontsize=9)

plt.tight_layout()
fig.savefig(OUT / 'fig_struct_composition_corr.pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT / 'fig_struct_composition_corr.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  saved fig_struct_composition_corr', flush=True)

# print top correlations
print('  Top |ρ| with Occupancy:')
for g in np.argsort(np.abs(corr_occ))[::-1][:4]:
    print(f'    {guilds[g]:22s} ρ={corr_occ[g]:+.3f}  p={pval_occ[g]:.3f}')
print('  Top |ρ| with PerLive:')
for g in np.argsort(np.abs(corr_pl))[::-1][:4]:
    print(f'    {guilds[g]:22s} ρ={corr_pl[g]:+.3f}  p={pval_pl[g]:.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. A-matrix stability: eigenvalues + attractor landscape
# ══════════════════════════════════════════════════════════════════════════════

print('Fig 2: stability analysis ...', flush=True)

eigvals = np.linalg.eigvals(A_glv)
stable_fixed = np.all(eigvals.real < 0)

# attractor landscape: run ODE from 200 random ICs, find equilibria
def replicator_rhs(t, phi, b, A):
    f = b + A @ phi; return phi * (f - phi @ f)

N_IC   = 200
T_END  = 500.0
b_mean = b_all.mean(axis=0)[:n_g_fit]   # use mean patient b
rng    = np.random.default_rng(42)
attractor_phi = []

for _ in range(N_IC):
    x0 = rng.dirichlet(np.ones(n_g_fit))
    sol = solve_ivp(replicator_rhs, [0, T_END], x0, args=(b_mean, A_glv),
                    method='RK45', rtol=1e-7, atol=1e-9)
    eq = sol.y[:, -1]
    eq = np.clip(eq, 0, None); eq /= eq.sum()
    attractor_phi.append(eq)

attractor_phi = np.array(attractor_phi)

# cluster attractors by distance (merge if ||a - b||₁ < 0.05)
from scipy.cluster.hierarchy import linkage, fcluster
if N_IC > 1:
    Z = linkage(attractor_phi, method='ward')
    labels = fcluster(Z, t=0.05, criterion='distance')
else:
    labels = np.ones(N_IC, dtype=int)

n_ct = labels.max()
print(f'  Community types found: {n_ct}', flush=True)

# median composition per community type
ct_means = []
for ct in range(1, n_ct + 1):
    ct_means.append(attractor_phi[labels == ct].mean(axis=0))
ct_counts = [(labels == ct).sum() for ct in range(1, n_ct + 1)]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# eigenvalue spectrum
ax = axes[0]
ax.scatter(eigvals.real, eigvals.imag, c='steelblue', s=60, zorder=3)
ax.axvline(0, color='r', lw=1, ls='--')
ax.set_xlabel('Re(λ)', fontsize=9); ax.set_ylabel('Im(λ)', fontsize=9)
ax.set_title(f'gLV A eigenvalues\n({"all stable" if stable_fixed else "unstable present"})', fontsize=9)
ax.grid(True, alpha=0.3)

# community type bar chart
ax = axes[1]
bottom = np.zeros(n_ct)
for g in range(n_g_fit):
    vals = [ct_means[ct][g] for ct in range(n_ct)]
    ax.bar(range(n_ct), vals, bottom=bottom,
           color=GUILD_COLORS_LIST[g] if g < len(GUILD_COLORS_LIST) else '#aaa',
           label=guilds[g] if g < n_g else f'g{g}')
    bottom += np.array(vals)
ax.set_xticks(range(n_ct))
ax.set_xticklabels([f'CT{i+1}\n(n={ct_counts[i]})' for i in range(n_ct)], fontsize=8)
ax.set_ylabel('Guild fraction', fontsize=9)
ax.set_title(f'Attractor community types (N={N_IC} ICs)', fontsize=9)
ax.legend(loc='upper right', fontsize=6, ncol=2)

# attractor scatter (PCA of attractor_phi)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Z2  = pca.fit_transform(attractor_phi)
ax  = axes[2]
sc  = ax.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)', fontsize=9)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)', fontsize=9)
ax.set_title(f'Attractor PCA ({n_ct} clusters)', fontsize=9)
plt.colorbar(sc, ax=ax, label='Community type')

plt.tight_layout()
fig.savefig(OUT / 'fig_stability.pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT / 'fig_stability.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  saved fig_stability', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Patient PCA / clustering
# ══════════════════════════════════════════════════════════════════════════════

print('Fig 3: patient clustering ...', flush=True)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# feature matrix: structural (6) + W1 composition (n_g)
feat_struct = np.column_stack([occ_arr, pl_arr])             # (n_keep, 6)
feat_comp   = phi_all[:, 0, :]                               # W1 composition
feat_all    = np.column_stack([feat_struct, feat_comp])      # (n_keep, 6+n_g)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(feat_all)

pca2 = PCA(n_components=min(4, feat_all.shape[1]))
X_pca = pca2.fit_transform(X_scaled)

km2 = KMeans(n_clusters=2, random_state=0, n_init=20).fit(X_scaled)
km3 = KMeans(n_clusters=3, random_state=0, n_init=20).fit(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# PC1-PC2 by patient label
ax = axes[0]
for i, pat in enumerate(patients):
    ax.scatter(X_pca[i, 0], X_pca[i, 1], s=120, zorder=3)
    ax.text(X_pca[i, 0] + 0.05, X_pca[i, 1], pat, fontsize=9)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.0f}%)', fontsize=9)
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.0f}%)', fontsize=9)
ax.set_title('Patient PCA\n(struct + W1 composition)', fontsize=9)
ax.grid(True, alpha=0.3)

# k=2 clustering
ax = axes[1]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km2.labels_, cmap='Set1', s=120, zorder=3)
for i, pat in enumerate(patients):
    ax.text(X_pca[i, 0] + 0.05, X_pca[i, 1], pat, fontsize=9)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.0f}%)', fontsize=9)
ax.set_title('KMeans k=2', fontsize=9); ax.grid(True, alpha=0.3)

# k=3 clustering
ax = axes[2]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=km3.labels_, cmap='Set2', s=120, zorder=3)
for i, pat in enumerate(patients):
    ax.text(X_pca[i, 0] + 0.05, X_pca[i, 1], pat, fontsize=9)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.0f}%)', fontsize=9)
ax.set_title('KMeans k=3', fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / 'fig_patient_clustering.pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT / 'fig_patient_clustering.png', dpi=150, bbox_inches='tight')
plt.close()

# PCA loadings: top features for PC1/PC2
feat_names = ([f'occ_W{w+1}' for w in range(3)] + [f'pl_W{w+1}' for w in range(3)]
              + guilds)
for pc in range(2):
    top = np.argsort(np.abs(pca2.components_[pc]))[::-1][:4]
    print(f'  PC{pc+1} top loadings: ' + ', '.join(f'{feat_names[j]}={pca2.components_[pc,j]:+.3f}' for j in top))
print(f'  k=2 labels: {dict(zip(patients, km2.labels_.tolist()))}')
print(f'  k=3 labels: {dict(zip(patients, km3.labels_.tolist()))}')
print(f'  saved fig_patient_clustering', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Alpha-diversity (Shannon entropy) time series
# ══════════════════════════════════════════════════════════════════════════════

print('Fig 4: alpha diversity ...', flush=True)

def shannon(phi):
    p = phi[phi > 1e-12]
    return -np.sum(p * np.log(p))

def simpson(phi):
    p = phi[phi > 1e-12]
    return 1 - np.sum(p**2)

H_obs = np.array([[shannon(phi_all[p, w]) for w in range(n_w)] for p in range(n_keep)])
D_obs = np.array([[simpson(phi_all[p, w]) for w in range(n_w)] for p in range(n_keep)])

# also compute predicted diversity from gLV
H_pred = np.zeros((n_keep, n_w))
H_pred[:, 0] = H_obs[:, 0]
for p in range(n_keep):
    b_p = b_all[p, :n_g_fit]
    phi2 = integrate_step(phi_all[p, 0, :n_g_fit], b_p, A_glv)
    phi3 = integrate_step(phi2,                     b_p, A_glv)
    H_pred[p, 1] = shannon(phi2)
    H_pred[p, 2] = shannon(phi3)

weeks = [1, 2, 3]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Shannon per patient
ax = axes[0]
for p, pat in enumerate(patients):
    ax.plot(weeks, H_obs[p], 'o-', lw=1.5, ms=6, label=pat)
ax.set_xticks(weeks); ax.set_xlabel('Week', fontsize=9)
ax.set_ylabel("Shannon H", fontsize=9)
ax.set_title('Shannon diversity per patient (observed)', fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

# Mean ± SD observed vs predicted
ax = axes[1]
H_mean = H_obs.mean(axis=0); H_std = H_obs.std(axis=0)
P_mean = H_pred.mean(axis=0); P_std = H_pred.std(axis=0)
ax.errorbar(weeks, H_mean, yerr=H_std, fmt='o-', color='steelblue', lw=2, capsize=4, label='Observed')
ax.errorbar(weeks, P_mean, yerr=P_std, fmt='s--', color='tomato',    lw=2, capsize=4, label='gLV predicted')
ax.set_xticks(weeks); ax.set_xlabel('Week', fontsize=9)
ax.set_ylabel("Shannon H", fontsize=9)
ax.set_title('Shannon diversity: observed vs gLV predicted\n(mean ± SD)', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Simpson D1 per patient
ax = axes[2]
for p, pat in enumerate(patients):
    ax.plot(weeks, D_obs[p], 'o-', lw=1.5, ms=6, label=pat)
ax.set_xticks(weeks); ax.set_xlabel('Week', fontsize=9)
ax.set_ylabel('Simpson 1−D', fontsize=9)
ax.set_title('Simpson diversity per patient (observed)', fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / 'fig_diversity.pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT / 'fig_diversity.png', dpi=150, bbox_inches='tight')
plt.close()

# stats
for w, wk in enumerate(weeks):
    print(f'  W{wk}: H_obs={H_mean[w]:.3f}±{H_std[w]:.3f}  H_pred={P_mean[w]:.3f}±{P_std[w]:.3f}')
print(f'  saved fig_diversity', flush=True)

print(f'\nAll figures saved to {OUT}', flush=True)
