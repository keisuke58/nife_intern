#!/usr/bin/env python3
"""
compute_bc_metrics.py
  1. Training predictions (MAP A+b̂): RMSE vs Bray-Curtis per patient per week
  2. Joshi attractor: BC(initial, equilibrium) by diagnosis — does ODE shift community?
  3. LOO-BC stub: runs when PBS A-matrix fold JSONs are ready
"""
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
from guild_replicator_dieckow import GUILD_ORDER, N_G

CR  = _here / 'results' / 'dieckow_cr'
OTU = _here / 'results' / 'dieckow_otu'
FIG = CR / 'figs'

PATIENTS = list('ABCDEFGHKL')

# ── Bray-Curtis dissimilarity ────────────────────────────────────────────────
def bray_curtis(x, y):
    """BC dissimilarity between two compositional vectors."""
    x, y = np.asarray(x), np.asarray(y)
    return 1.0 - 2.0 * np.minimum(x, y).sum() / (x.sum() + y.sum())

def bc_matrix(pred, obs):
    """BC per row between pred (N,K) and obs (N,K)."""
    return np.array([bray_curtis(pred[i], obs[i]) for i in range(len(pred))])

# ════════════════════════════════════════════════════════════════════════════
# Part 1: Training predictions — RMSE vs BC
# ════════════════════════════════════════════════════════════════════════════
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

phi_all = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10)
d_agora = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
d_l12   = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded.json'))

def pack_theta_A(A_mat):
    """Full symmetric A → upper-triangular flat theta_A (n*(n+1)//2 entries)."""
    n = A_mat.shape[0]
    return jnp.array([A_mat[i,j] for j in range(n) for i in range(j+1)])

def predict_all(fit_dict, phi_obs):
    A_mat = np.array(fit_dict['A'])
    b     = np.array(fit_dict['b_all'])   # (10,10)
    theta_A = pack_theta_A(A_mat)
    n_p   = phi_obs.shape[0]

    def run(theta, phi0):
        eq = simulate_0d_nsp(theta, n_sp=N_G, n_steps=100, dt=1e-4,
                             phi_init=phi0, c_const=25.0, alpha_const=100.0)[-1]
        s = eq.sum(); return jnp.where(s>1e-10, eq/s, jnp.ones(N_G)/N_G)

    run_jit = jax.jit(run)
    ph2, ph3 = [], []
    for p in range(n_p):
        theta = jnp.concatenate([theta_A, jnp.array(b[p])])
        p2 = np.array(run_jit(theta, jnp.array(phi_obs[p,0])))
        p3 = np.array(run_jit(theta, jnp.array(p2)))
        ph2.append(p2); ph3.append(p3)
    return np.array(ph2), np.array(ph3)

print('Computing training predictions (MAP)...', flush=True)
ph2_ag, ph3_ag = predict_all(d_agora, phi_all)
ph2_l2, ph3_l2 = predict_all(d_l12,   phi_all)

rows = []
for p_idx, pat in enumerate(PATIENTS):
    for model, ph2, ph3 in [('AGORA W=1.0', ph2_ag, ph3_ag),
                             ('L1+L2',       ph2_l2, ph3_l2)]:
        for wk, pred, obs in [(2, ph2[p_idx], phi_all[p_idx,1]),
                               (3, ph3[p_idx], phi_all[p_idx,2])]:
            rmse = float(np.sqrt(np.mean((pred - obs)**2)))
            bc   = float(bray_curtis(pred, obs))
            rows.append({'patient': pat, 'week': wk, 'model': model,
                         'rmse': rmse, 'bc': bc})

df_tr = pd.DataFrame(rows)
df_tr.to_csv(CR / 'train_rmse_bc.csv', index=False)

# Summary table
print('\n=== Training: mean RMSE and BC per model ===')
for model in ['L1+L2', 'AGORA W=1.0']:
    sub = df_tr[df_tr.model == model]
    print(f'  {model:15s}: RMSE={sub.rmse.mean():.4f}  BC={sub.bc.mean():.4f}')

# Correlation RMSE vs BC
r_bc, p_bc = stats.pearsonr(df_tr.rmse, df_tr.bc)
print(f'\n  Pearson r(RMSE, BC) over all predictions = {r_bc:.3f}  p={p_bc:.2e}')

# ════════════════════════════════════════════════════════════════════════════
# Part 2: LOO-BC — using saved LOO fold JSONs (needs PBS A-matrix results)
# ════════════════════════════════════════════════════════════════════════════
loo_bc_rows = []
for fold in range(10):
    path_ag = CR / f'loo_expanded_agora_w1p0_fold{fold}.json'
    path_l2 = CR / f'loo_expanded_fold{fold}.json'
    if not path_ag.exists(): continue
    dA = json.load(open(path_ag))
    dL = json.load(open(path_l2))
    pat = dA['patient']
    p_idx = PATIENTS.index(pat)
    phi_ho = phi_all[p_idx]     # (3, 10)

    for model, fd in [('AGORA W=1.0', dA), ('L1+L2', dL)]:
        if 'A' not in fd:
            continue   # old format — skip, wait for PBS
        theta_A_loo = pack_theta_A(np.array(fd['A']))
        b_ho        = jnp.array(fd['b_ho'])
        theta       = jnp.concatenate([theta_A_loo, b_ho])

        def run_loo(theta, phi0):
            eq = simulate_0d_nsp(theta, n_sp=N_G, n_steps=100, dt=1e-4,
                                 phi_init=phi0, c_const=25.0, alpha_const=100.0)[-1]
            s = eq.sum(); return jnp.where(s>1e-10, eq/s, jnp.ones(N_G)/N_G)

        run_loo_jit = jax.jit(run_loo)
        p2 = np.array(run_loo_jit(theta, jnp.array(phi_ho[0])))
        p3 = np.array(run_loo_jit(theta, jnp.array(p2)))
        for wk, pred, obs in [(2, p2, phi_ho[1]), (3, p3, phi_ho[2])]:
            rmse = float(np.sqrt(np.mean((pred-obs)**2)))
            bc   = float(bray_curtis(pred, obs))
            loo_bc_rows.append({'patient':pat,'week':wk,'model':model,'rmse':rmse,'bc':bc})

if loo_bc_rows:
    df_loo = pd.DataFrame(loo_bc_rows)
    df_loo.to_csv(CR / 'loo_rmse_bc.csv', index=False)
    print('\n=== LOO: mean RMSE and BC per model ===')
    for model in ['L1+L2', 'AGORA W=1.0']:
        sub = df_loo[df_loo.model==model]
        if len(sub):
            print(f'  {model:15s}: RMSE={sub.rmse.mean():.4f}  BC={sub.bc.mean():.4f}')
else:
    print('\nLOO A-matrix JSONs not yet available (PBS still running) — skip LOO-BC')

# ════════════════════════════════════════════════════════════════════════════
# Part 3: Joshi — BC(initial, equilibrium) by diagnosis
# ════════════════════════════════════════════════════════════════════════════
from collections import defaultdict
import openpyxl

EXCEL = _here / 'Datasets' / '20260416_mSystems_16S_5genera_all_profiles.xlsx'
GENUS_TO_GUILD = {'Actinomyces':'Actinobacteria','Streptococcus':'Bacilli',
                  'Porphyromonas':'Bacteroidia','Fusobacterium':'Fusobacteriia',
                  'Veillonella':'Negativicutes'}
GUILD_IDX = {g:i for i,g in enumerate(GUILD_ORDER)}
DYS_IDX = [GUILD_IDX[g] for g in ['Fusobacteriia','Bacteroidia','Clostridia']]
COM_IDX = [GUILD_IDX[g] for g in ['Bacilli','Actinobacteria','Negativicutes']]

wb = openpyxl.load_workbook(EXCEL, read_only=True)
ws = wb['Sheet1']
rows_xl = list(ws.iter_rows(values_only=True))
diagnoses = list(rows_xl[0][1:])
n_samp = len(diagnoses)

genus_agg = defaultdict(lambda: np.zeros(n_samp))
for r in rows_xl[1:]:
    g = r[0]
    if g in (None,'Total'): continue
    for s,v in enumerate(r[1:]): genus_agg[g][s] += float(v or 0)
GENUS_NAMES = sorted(genus_agg.keys())
genus_mat = np.array([genus_agg[g] for g in GENUS_NAMES])
col_tot = np.where(genus_mat.sum(0)==0, 1, genus_mat.sum(0))
rel_5 = genus_mat / col_tot
phi_mean_dk = np.load(OTU/'phi_guild.npy').mean(axis=(0,1))

def build_phi0(rel5):
    phi = np.full(N_G, 1e-4)
    obs_total = 0.0
    for g,frac in zip(GENUS_NAMES, rel5):
        if g in GENUS_TO_GUILD:
            phi[GUILD_IDX[GENUS_TO_GUILD[g]]] = max(frac,1e-4)
            obs_total += frac
    rem = max(1.0-obs_total, 0.0)
    unobs = [i for i in range(N_G) if GUILD_ORDER[i] not in GENUS_TO_GUILD.values()]
    dk = phi_mean_dk[unobs]; dk_sum = dk.sum()
    for k,gi in enumerate(unobs):
        phi[gi] = rem*dk[k]/dk_sum if dk_sum>1e-8 else rem/len(unobs)
    phi = np.maximum(phi,1e-6); phi /= phi.sum()
    return phi

phi0_all = np.array([build_phi0(rel_5[:,s]) for s in range(n_samp)])

print('\nComputing Joshi equilibria for BC analysis...', flush=True)
theta_A_agora = pack_theta_A(np.array(d_agora['A']))
b_neutral     = jnp.array(np.full(N_G, 0.1))

def run_eq(phi0):
    theta = jnp.concatenate([theta_A_agora, b_neutral])
    eq = simulate_0d_nsp(theta, n_sp=N_G, n_steps=500, dt=1e-4,
                         phi_init=jnp.array(phi0), c_const=25.0, alpha_const=100.0)[-1]
    s = eq.sum(); return jnp.where(s>1e-10, eq/s, jnp.ones(N_G)/N_G)

run_eq_jit = jax.jit(run_eq)
eq_all = np.array([run_eq_jit(phi0_all[s]) for s in range(n_samp)])

bc_shift = bc_matrix(eq_all, phi0_all)   # how far ODE moves each sample
diag_arr = np.array(diagnoses)
diag_num = np.array([{'Health':0,'Mucositis':1,'Peri-implantitis':2}[d] for d in diag_arr])

def guild_di(phi_mat, eps=1e-4):
    dys = phi_mat[:,DYS_IDX].sum(1)+eps
    com = phi_mat[:,COM_IDX].sum(1)+eps
    return np.log(dys/com)

gdi_init = guild_di(phi0_all)
gdi_eq   = guild_di(eq_all)

print('\n=== Joshi: BC(initial→equilibrium) by diagnosis ===')
DIAG_ORDER = ['Health','Mucositis','Peri-implantitis']
for d in DIAG_ORDER:
    v = bc_shift[diag_arr==d]
    print(f'  {d:22s}: mean BC shift={v.mean():.3f}  std={v.std():.3f}')
h,p = stats.kruskal(*[bc_shift[diag_arr==d] for d in DIAG_ORDER])
print(f'  Kruskal-Wallis p={p:.4f}')
rho,p2 = stats.spearmanr(diag_num, bc_shift)
print(f'  Spearman ρ(severity, BC_shift)={rho:.3f}  p={p2:.4f}')

# ── Figures ──────────────────────────────────────────────────────────────────
DIAG_COLORS = {'Health':'#2ecc71','Mucositis':'#f39c12','Peri-implantitis':'#e74c3c'}
DIAG_SHORT  = {'Health':'PIH','Mucositis':'PIM','Peri-implantitis':'PI'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams.update({'font.size':9, 'font.family':'serif',
                     'font.serif':['Times New Roman','Times','DejaVu Serif'],
                     'mathtext.fontset':'stix'})

# Panel 1: Training RMSE vs BC scatter
ax = axes[0]
colors_map = {'AGORA W=1.0':'#e74c3c','L1+L2':'#2980b9'}
for model, sub in df_tr.groupby('model'):
    ax.scatter(sub.rmse, sub.bc, label=model, color=colors_map[model],
               alpha=0.7, s=50, zorder=3)
ax.set_xlabel('RMSE', fontsize=9); ax.set_ylabel('Bray-Curtis', fontsize=9)
ax.set_title(f'Training: RMSE vs BC\n(r={r_bc:.3f}, p={p_bc:.1e})', fontsize=9)
ax.legend(fontsize=8)

# Panel 2: Joshi — BC shift violin
ax2 = axes[1]
parts = ax2.violinplot([bc_shift[diag_arr==d] for d in DIAG_ORDER],
                        positions=range(3), showmedians=True, widths=0.6)
for pc,d in zip(parts['bodies'], DIAG_ORDER):
    pc.set_facecolor(DIAG_COLORS[d]); pc.set_alpha(0.7)
parts['cmedians'].set_color('k'); parts['cmedians'].set_linewidth(1.5)
for xi,d in enumerate(DIAG_ORDER):
    v = bc_shift[diag_arr==d]
    jitter = np.random.default_rng(42).uniform(-0.15,0.15,len(v))
    ax2.scatter(xi+jitter, v, s=8, color=DIAG_COLORS[d], alpha=0.5, zorder=3)
ax2.set_xticks(range(3))
ax2.set_xticklabels([DIAG_SHORT[d] for d in DIAG_ORDER])
ax2.set_ylabel('BC(initial → equilibrium)', fontsize=9)
ax2.set_title(f'Joshi: how far ODE moves each sample\nKW p={p:.3f}  ρ={rho:.2f}', fontsize=9)

# Panel 3: Joshi — GDI_init vs BC_shift scatter coloured by diagnosis
ax3 = axes[2]
for d in DIAG_ORDER:
    mask = diag_arr==d
    ax3.scatter(gdi_init[mask], bc_shift[mask], label=DIAG_SHORT[d],
                color=DIAG_COLORS[d], alpha=0.6, s=25, zorder=3)
ax3.set_xlabel('Guild DI (initial composition)', fontsize=9)
ax3.set_ylabel('BC(initial → equilibrium)', fontsize=9)
ax3.set_title('Guild DI vs ODE displacement\n(coloured by diagnosis)', fontsize=9)
ax3.legend(fontsize=8)
r3,p3 = stats.pearsonr(gdi_init, bc_shift)
ax3.text(0.05,0.97,f'r={r3:.2f}  p={p3:.2e}',transform=ax3.transAxes,
         va='top',fontsize=7,bbox=dict(boxstyle='round',fc='white',alpha=0.8))

fig.suptitle('RMSE vs Bray-Curtis: training predictions and Joshi attractor', fontsize=10)
fig.tight_layout()
out = FIG / 'fig_rmse_vs_bc.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(FIG / 'fig_rmse_vs_bc.pdf', bbox_inches='tight')
print(f'\nSaved {out} + PDF')
