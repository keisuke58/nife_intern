# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

"""
3-panel bridge between the gLV mathematical model and experimental data.

① Vector field + CT trajectory overlay (gLV phase portrait)
② MTX/16S ratio → empirical μ_i vs gLV-predicted μ_i
③ MDI as gLV observable along patient trajectories
"""

import csv, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr, spearmanr

import pub_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

ROOT     = Path(__file__).parents[2]
JDIR     = ROOT / 'Datasets' / 'joshi2025_SI'
RESDIR   = ROOT / 'results' / 'dieckow_cr'
OUT      = RESDIR / 'figs'
OUT.mkdir(exist_ok=True)

GUILDS   = GUILD_ORDER                           # 10 guilds
FIT_JSON = RESDIR / 'fit_glv_hamilton_kegg_expanded.json'

# ── load fit ─────────────────────────────────────────────────────────────────

fit      = json.load(open(FIT_JSON))
fit_glds = fit['guilds']                         # ['Actinobacteria','Bacilli',...]
pats_ord = fit['patients']                        # ['A','B','C',...]
A_fit    = np.array(fit['A'])                    # (10,10) gLV interaction matrix
b_all    = np.array(fit['b_all'])                # (n_pat, 10)

# reorder fit guilds → GUILD_ORDER
idx      = [fit_glds.index(g) for g in GUILDS]
A        = A_fit[np.ix_(idx, idx)]
b_mean   = b_all[:, idx].mean(axis=0)            # mean b across patients

# ── load Dieckow patient observations ────────────────────────────────────────

df = pd.read_csv(RESDIR / 'dieckow_guild_abundance.csv')
ct_pred  = json.load(open(RESDIR / 'cross_ct_prediction.json'))
ct1_pats = ct_pred['ct1_patients']

df['pat'] = df['sample'].str[0]
GUILDS_9  = [g for g in GUILDS if g != 'Other']

ct1_raw  = df.loc[df['pat'].isin(ct1_pats),  GUILDS_9].mean().values
ct2_raw  = df.loc[~df['pat'].isin(ct1_pats), GUILDS_9].mean().values
ct1      = ct1_raw / ct1_raw.sum()
ct2      = ct2_raw / ct2_raw.sum()

# ── load Joshi class means ────────────────────────────────────────────────────

def read_class_csv(path):
    rows = list(csv.reader(open(path)))
    hd   = rows[0]
    hi   = hd.index('mean_health')
    pi   = hd.index('mean_periimplantitis')
    return {r[0]: (float(r[hi]), float(r[pi])) for r in rows[1:] if r[0]}

j16s = read_class_csv(JDIR / 'joshi2025_suppE_class.csv')
jrna = read_class_csv(JDIR / 'joshi2025_suppG_class_rnaseq.csv')

def guild_vecs(d):
    h = np.array([d.get(g, (0, 0))[0] for g in GUILDS_9])
    p = np.array([d.get(g, (0, 0))[1] for g in GUILDS_9])
    return h / h.sum(), p / p.sum()

j16s_h, j16s_p = guild_vecs(j16s)
jrna_h, jrna_p = guild_vecs(jrna)

# ── gLV replicator ODE ────────────────────────────────────────────────────────
# Uses 9-guild subspace (drop 'Other') with b_mean and A

A9_idx = [GUILDS.index(g) for g in GUILDS_9]
A9     = A[np.ix_(A9_idx, A9_idx)]
b9     = b_mean[A9_idx]

def glv_replicator(t, phi):
    """Replicator gLV: dφ_i/dt = φ_i(b_i + Σ A_ij φ_j) - φ_i Σ_k φ_k(b_k + Σ A_kj φ_j)"""
    phi   = np.clip(phi, 0, None)
    phi  /= phi.sum() + 1e-12
    r     = b9 + A9 @ phi
    dphi  = phi * (r - phi @ r)
    return dphi

def integrate(phi0, T=30, n=300):
    phi0 = np.clip(phi0, 1e-4, None)
    phi0 = phi0 / phi0.sum()
    sol  = solve_ivp(glv_replicator, [0, T], phi0,
                     t_eval=np.linspace(0, T, n), method='RK45',
                     rtol=1e-6, atol=1e-9)
    return sol.t, sol.y.T   # (n, 9)

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────

pub_style.apply()
fig = plt.figure(figsize=(16, 13))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                        top=0.93, bottom=0.07, left=0.06, right=0.97)

ax_vf   = fig.add_subplot(gs[0, :2])   # wide: vector field + trajectories
ax_ct   = fig.add_subplot(gs[0, 2])    # CT diagnosis proportions
ax_mu   = fig.add_subplot(gs[1, :2])   # μ comparison
ax_mdi  = fig.add_subplot(gs[1, 2])    # MDI trajectories

colors_g = [GUILD_COLORS[g] for g in GUILDS_9]
short    = [GUILD_SHORT[g]   for g in GUILDS_9]

# ═══════════════════════════════════════════════════════════════════════════
# Panel A  Vector field in Bacilli–Bacteroidia space
# ═══════════════════════════════════════════════════════════════════════════

BAC_I = GUILDS_9.index('Bacilli')
BCT_I = GUILDS_9.index('Bacteroidia')

# Grid: hold other guilds at mean of ct1+ct2, vary Bacilli & Bacteroidia
mean_bg  = (ct1 + ct2) / 2
N_GRID   = 14
bac_grid = np.linspace(0.02, 0.85, N_GRID)
bct_grid = np.linspace(0.02, 0.55, N_GRID)
U        = np.zeros((N_GRID, N_GRID))
V        = np.zeros((N_GRID, N_GRID))

for i, bac in enumerate(bac_grid):
    for j, bct in enumerate(bct_grid):
        phi = mean_bg.copy()
        phi[BAC_I] = bac
        phi[BCT_I] = bct
        phi = np.clip(phi, 0, None)
        phi /= phi.sum()
        d    = glv_replicator(0, phi)
        U[j, i] = d[BAC_I]
        V[j, i] = d[BCT_I]

BG, BCG = np.meshgrid(bac_grid, bct_grid)
speed   = np.sqrt(U**2 + V**2)
ax_vf.streamplot(BG, BCG, U, V, color=speed, cmap='Blues', linewidth=0.7,
                 density=1.2, arrowsize=0.8)

# Integrate from CT I and CT IV initial conditions
CT_INITS = {
    'CT_I (PIH ≈ Joshi health)':   j16s_h,
    'CT_II (PIM-Neisseria proxy)':  0.7*j16s_h + 0.3*j16s_p,
    'CT_III (PIM-Bacteroidia proxy)': 0.3*j16s_h + 0.7*j16s_p,
    'CT_IV (PI ≈ Joshi peri)':     j16s_p,
}
traj_cols = ['#2166ac', '#74add1', '#fdae61', '#d6604d']

for (lbl, phi0), col in zip(CT_INITS.items(), traj_cols):
    t, traj = integrate(phi0, T=40)
    ax_vf.plot(traj[:, BAC_I], traj[:, BCT_I], '-', color=col, lw=1.8,
               label=lbl, zorder=4)
    ax_vf.scatter(traj[0, BAC_I], traj[0, BCT_I], color=col,
                  s=55, zorder=5, marker='o', edgecolors='k', lw=0.6)
    ax_vf.scatter(traj[-1, BAC_I], traj[-1, BCT_I], color=col,
                  s=70, zorder=5, marker='*', edgecolors='k', lw=0.6)

# Mark Dieckow CT1/CT2
for (lbl, vec, mk) in [('Dieckow\nCT1', ct1, 's'), ('Dieckow\nCT2', ct2, 'D')]:
    ax_vf.scatter(vec[BAC_I], vec[BCT_I], color='k', s=80, marker=mk, zorder=6)
    ax_vf.annotate(lbl, (vec[BAC_I], vec[BCT_I]), fontsize=7.5,
                   xytext=(5, 4), textcoords='offset points')

# Joshi health/peri
for lbl, vec, col, mk in [('Joshi health', j16s_h, '#92c5de', '^'),
                            ('Joshi peri',  j16s_p, '#f4a582', 'v')]:
    ax_vf.scatter(vec[BAC_I], vec[BCT_I], color=col, s=75, marker=mk,
                  edgecolors='k', lw=0.6, zorder=6)
    ax_vf.annotate(lbl, (vec[BAC_I], vec[BCT_I]), fontsize=7, xytext=(4, 3),
                   textcoords='offset points')

ax_vf.set_xlabel('Bacilli (relative abundance)', fontsize=9)
ax_vf.set_ylabel('Bacteroidia (relative abundance)', fontsize=9)
ax_vf.set_title('A  gLV vector field: Bacilli–Bacteroidia plane\n'
                '(trajectories from Szafranski CT I/II/III/IV initial compositions;\n'
                ' ○=start, ★=end t=40d; □=Dieckow CT1, ◇=CT2; Joshi ▲=health ▽=peri)',
                fontsize=8.5, fontweight='bold')
ax_vf.legend(fontsize=6.5, loc='upper right', ncol=2)

# ── Panel B: CT diagnosis proportions (small bar) ────────────────────────────

ct_diag = {
    'CT I':  {'PIH': 0.70, 'PIM': 0.20, 'PI': 0.10},
    'CT II': {'PIH': 0.05, 'PIM': 0.90, 'PI': 0.05},
    'CT III':{'PIH': 0.10, 'PIM': 0.65, 'PI': 0.25},
    'CT IV': {'PIH': 0.05, 'PIM': 0.20, 'PI': 0.75},
}
ct_names  = list(ct_diag.keys())
diag_cols = {'PIH': '#2166ac', 'PIM': '#74add1', 'PI': '#d6604d'}
x         = np.arange(len(ct_names))
bottom    = np.zeros(len(ct_names))
for diag, col in diag_cols.items():
    vals = [ct_diag[ct][diag] for ct in ct_names]
    ax_ct.bar(x, vals, 0.65, bottom=bottom, color=col, label=diag, edgecolor='k', lw=0.4)
    bottom += np.array(vals)

ax_ct.set_xticks(x)
ax_ct.set_xticklabels(ct_names, fontsize=8)
ax_ct.set_ylabel('Proportion of samples')
ax_ct.set_ylim(0, 1.05)
ax_ct.set_title('B  CT diagnosis\ncomposition\n(Szafranski Fig. 3b)', fontsize=8.5, fontweight='bold')
ax_ct.legend(fontsize=7, loc='upper right')
for xi, ct in enumerate(ct_names):
    if xi == 0:
        ax_ct.annotate('≈ Dieckow\nCT1', (xi, 0.5), ha='center', va='center',
                        fontsize=6.5, color='white', fontweight='bold')
    if xi == 3:
        ax_ct.annotate('≈ Dieckow\nCT2', (xi, 0.5), ha='center', va='center',
                        fontsize=6.5, color='white', fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════
# Panel C  μ_i empirical (MTX/16S) vs μ_i gLV predicted
# ═══════════════════════════════════════════════════════════════════════════
# μ_i^obs ∝ RNAseq_i / 16S_i   (activity normalized by biomass)
# μ_i^pred = b_i + Σ_j A_ij φ_j   at CT1 (health) and CT2 (peri)

# Empirical μ at health & peri
mu_obs_h = jrna_h / (j16s_h + 1e-6)
mu_obs_p = jrna_p / (j16s_p + 1e-6)
# z-score within group for comparability
mu_obs_h = (mu_obs_h - mu_obs_h.mean()) / (mu_obs_h.std() + 1e-9)
mu_obs_p = (mu_obs_p - mu_obs_p.mean()) / (mu_obs_p.std() + 1e-9)

# gLV predicted μ
A9_h = [GUILDS.index(g) for g in GUILDS_9]   # same as A9_idx
mu_pred_h = b9 + A9 @ ct1
mu_pred_p = b9 + A9 @ ct2
mu_pred_h = (mu_pred_h - mu_pred_h.mean()) / (mu_pred_h.std() + 1e-9)
mu_pred_p = (mu_pred_p - mu_pred_p.mean()) / (mu_pred_p.std() + 1e-9)

x4 = np.arange(len(GUILDS_9))
w  = 0.22
ax_mu.bar(x4 - 1.5*w, mu_obs_h,  w, color='#92c5de', label='MTX/16S — health',         alpha=0.9, edgecolor='k', lw=0.3)
ax_mu.bar(x4 - 0.5*w, mu_pred_h, w, color='#2166ac', label='gLV μ pred — CT1',          alpha=0.9, edgecolor='k', lw=0.3, hatch='///')
ax_mu.bar(x4 + 0.5*w, mu_obs_p,  w, color='#f4a582', label='MTX/16S — peri-implantitis',alpha=0.9, edgecolor='k', lw=0.3)
ax_mu.bar(x4 + 1.5*w, mu_pred_p, w, color='#d6604d', label='gLV μ pred — CT2',          alpha=0.9, edgecolor='k', lw=0.3, hatch='///')
ax_mu.axhline(0, color='k', lw=0.6)
ax_mu.set_xticks(x4)
ax_mu.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax_mu.set_ylabel('Effective growth rate μ_i (z-scored)', fontsize=8)
ax_mu.set_title('C  Empirical effective growth rate (MTX/16S) vs gLV prediction  μ_i = b_i + Σ A_ij φ_j\n'
                '(solid = Joshi experimental; hatch = gLV model at Dieckow CT1/CT2 composition)',
                fontsize=8.5, fontweight='bold')
ax_mu.legend(fontsize=7, loc='lower right', ncol=2)

# Correlation annotations
rh, ph = pearsonr(mu_obs_h, mu_pred_h)
rp, pp = pearsonr(mu_obs_p, mu_pred_p)
ax_mu.text(0.01, 0.97,
           f'Health:  r(Joshi MTX/16S × gLV CT1) = {rh:.2f}  p={ph:.3f}\n'
           f'Disease: r(Joshi MTX/16S × gLV CT2) = {rp:.2f}  p={pp:.3f}',
           transform=ax_mu.transAxes, fontsize=7.5, va='top',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

# ═══════════════════════════════════════════════════════════════════════════
# Panel D  MDI along gLV trajectories
# MDI ≈ log(φ_Clostridia / (φ_Actinobacteria + φ_Bacilli))
# Pseudoramibacter is Clostridia; Capnocytophaga=Bacteroidia, Gemella=Bacilli
# simplified guild-level proxy: Clostridia / (Bacilli + Actinobacteria)
# ═══════════════════════════════════════════════════════════════════════════

CLO_I = GUILDS_9.index('Clostridia')
ACT_I = GUILDS_9.index('Actinobacteria')

def mdi_from_traj(traj):
    numer = traj[:, CLO_I]
    denom = traj[:, BAC_I] + traj[:, ACT_I]
    return np.log(numer / (denom + 1e-9) + 1e-9)

for (lbl, phi0), col in zip(CT_INITS.items(), traj_cols):
    t, traj = integrate(phi0, T=40)
    mdi     = mdi_from_traj(traj)
    ax_mdi.plot(t, mdi, color=col, lw=1.8, label=lbl)

# Mark Joshi health/peri MDI proxy
def mdi_scalar(vec):
    return np.log(vec[CLO_I] / (vec[BAC_I] + vec[ACT_I] + 1e-9) + 1e-9)

for lbl, vec, col, mk in [('Joshi health', j16s_h, '#92c5de', '^'),
                            ('Dieckow CT1', ct1,    '#2166ac', 's'),
                            ('Joshi peri',  j16s_p, '#f4a582', 'v'),
                            ('Dieckow CT2', ct2,    '#d6604d', 'D')]:
    ax_mdi.axhline(mdi_scalar(vec), color=col, lw=0.9, ls='--', alpha=0.7)
    ax_mdi.annotate(f'{lbl}\n(MDI={mdi_scalar(vec):.2f})',
                    xy=(38, mdi_scalar(vec)), fontsize=6.5, color=col, va='center')

ax_mdi.set_xlabel('Time (days)', fontsize=9)
ax_mdi.set_ylabel('MDI proxy\nlog(Clostridia / (Bacilli + Actinobacteria))', fontsize=8)
ax_mdi.set_title('D  MDI proxy along gLV trajectories\nfrom Szafranski CT initial conditions',
                  fontsize=8.5, fontweight='bold')
ax_mdi.legend(fontsize=6.5, loc='lower right')

fig.suptitle('Connecting gLV mathematical model with experimental data:\n'
             'vector field (①) · empirical vs predicted growth rate (②) · MDI as ODE observable (③)',
             fontsize=10, fontweight='bold')

for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'fig_model_data_bridge.{ext}', bbox_inches='tight', dpi=150)

plt.close()
print("Done →", OUT / 'fig_model_data_bridge.pdf')
