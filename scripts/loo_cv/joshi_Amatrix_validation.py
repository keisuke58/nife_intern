#!/usr/bin/env python3
"""
joshi_Amatrix_validation.py — A matrix validation against Joshi 2025 data.

Three analyses that directly use the Dieckow A matrix:

  Analysis 1: Velocity-based R/G rate
    d(R/G)/dt = Σ_dys f_i − Σ_com f_j  where f_i = b_i + Σ_j A_ij φ_j
    This is the instantaneous rate of change of R/G ratio given A and φ.
    Positive = A pushes toward dysbiosis at this composition.

  Analysis 2: A matrix permutation test
    Shuffle A rows/cols 200 times → ODE equilibrium R/G ratio
    → check if true A correlation ρ=0.37 beats random A

  Analysis 3: Community type (CT) attractor classification
    Run ODE per Joshi sample → R/G ratio at equilibrium < 0 = CT1 (commensal)
    → test if CT2-predicted samples are enriched in PI

Usage:
    python joshi_Amatrix_validation.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
})

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER, N_G

CR  = _here / 'results' / 'dieckow_cr'
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

EXCEL = _here / 'Datasets' / '20260416_mSystems_16S_5genera_all_profiles.xlsx'
FIT   = CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'

# ── Guild indices ────────────────────────────────────────────────────────────
GUILD_IDX   = {g: i for i, g in enumerate(GUILD_ORDER)}
DYS_GUILDS  = ['Fusobacteriia', 'Bacteroidia', 'Clostridia']
COM_GUILDS  = ['Bacilli', 'Actinobacteria', 'Negativicutes']
DYS_IDX     = [GUILD_IDX[g] for g in DYS_GUILDS]
COM_IDX     = [GUILD_IDX[g] for g in COM_GUILDS]

GENUS_TO_GUILD = {
    'Actinomyces':    'Actinobacteria',
    'Streptococcus':  'Bacilli',
    'Porphyromonas':  'Bacteroidia',
    'Fusobacterium':  'Fusobacteriia',
    'Veillonella':    'Negativicutes',
}

DIAG_ORDER  = ['Health', 'Mucositis', 'Peri-implantitis']
DIAG_SHORT  = {'Health': 'PIH', 'Mucositis': 'PIM', 'Peri-implantitis': 'PI'}
DIAG_COLORS = {'Health': '#2ecc71', 'Mucositis': '#f39c12', 'Peri-implantitis': '#e74c3c'}
DIAG_MAP    = {'Health': 0, 'Mucositis': 1, 'Peri-implantitis': 2}

# ── Load Joshi Excel ─────────────────────────────────────────────────────────
wb   = openpyxl.load_workbook(EXCEL, read_only=True)
ws   = wb['Sheet1']
rows = list(ws.iter_rows(values_only=True))
header     = rows[0]
diag_cols  = {h: i for i, h in enumerate(header) if h in ('PI', 'Health', 'Mucositis')}
genus_rows = {r[0]: r for r in rows[1:] if r[0] in GENUS_TO_GUILD}
GENUS_NAMES = list(GENUS_TO_GUILD.keys())

# Map columns to (diagnosis, sample_index)
sample_meta = []
for diag, col_idx in sorted(diag_cols.items(), key=lambda x: x[1]):
    n_cols = sum(1 for c in header if c == diag)
    for _ in range(n_cols):
        sample_meta.append(diag)

# Build relative abundance matrix: (n_genus, n_samples)
n_samp = len(sample_meta)
rel_5  = np.zeros((len(GENUS_NAMES), n_samp))
for gi, gname in enumerate(GENUS_NAMES):
    row = genus_rows.get(gname)
    if row is None:
        continue
    vals = [v for v in row[1:] if v is not None][:n_samp]
    rel_5[gi, :len(vals)] = vals

diagnoses = sample_meta
diag_arr  = np.array(diagnoses)
diag_num  = np.array([DIAG_MAP[d] for d in diagnoses])

# ── Build 10-guild compositions (zero-impute) ────────────────────────────────
def build_phi0(rel5_col):
    phi = np.full(N_G, 1e-5)
    for g_name, frac in zip(GENUS_NAMES, rel5_col):
        phi[GUILD_IDX[GENUS_TO_GUILD[g_name]]] = max(frac, 1e-5)
    phi /= phi.sum()
    return phi

phi0_all = np.array([build_phi0(rel_5[:, s]) for s in range(n_samp)])  # (127, 10)

# ── Load Dieckow A matrix ────────────────────────────────────────────────────
d_fit = json.load(open(FIT))
A     = np.array(d_fit['A'])           # (10, 10)
b_all = np.array(d_fit['b_all'])       # (10, 10)
b_mean = b_all.mean(axis=0)

# Load LOO folds for better b estimates
_glv_folds = sorted(CR.glob('loo_glv_agora_a0p25_fold*.json'))
if _glv_folds:
    A_glv = np.mean([np.array(json.load(open(f))['A']) for f in _glv_folds], axis=0)
    b_glv = np.mean([np.array(json.load(open(f))['b_ho']) for f in _glv_folds], axis=0)
else:
    A_glv, b_glv = A, b_mean

print(f'Loaded: {n_samp} Joshi samples, A {A.shape}, {len(_glv_folds)} LOO folds')

def rg_ratio(phi_mat, eps=1e-4):
    dys = phi_mat[:, DYS_IDX].sum(axis=1) + eps
    com = phi_mat[:, COM_IDX].sum(axis=1) + eps
    return np.log(dys / com)

rg_raw = rg_ratio(phi0_all)

def print_stats(vals, label):
    rho, p = stats.spearmanr(diag_num, vals)
    h, ph  = stats.kruskal(*[vals[diag_arr == d] for d in DIAG_ORDER])
    mask   = (diag_arr == 'Health') | (diag_arr == 'Peri-implantitis')
    rho_b, p_b = stats.spearmanr(diag_num[mask], vals[mask])
    print(f'  {label}')
    print(f'    ρ(all 3) = {rho:+.3f}  p={p:.2e}  |  '
          f'ρ(H vs PI) = {rho_b:+.3f}  p={p_b:.2e}  |  KW p={ph:.4f}')

print('\n=== Baseline: raw R/G ratio ===')
print_stats(rg_raw, 'Raw log(dys/com)')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Velocity-based R/G rate  d(R/G)/dt using A matrix
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 1: A-matrix velocity R/G rate ===')

def rg_velocity(phi_mat, A, b):
    """
    d(R/G)/dt ≈ Σ_{i∈dys} f_i − Σ_{j∈com} f_j
    where f_i = b_i + Σ_k A_ik φ_k  (guild-specific fitness)
    Positive = A pushes this composition toward dysbiosis.
    """
    f = b + (A @ phi_mat.T).T          # (n_samp, N_G)
    return f[:, DYS_IDX].sum(axis=1) - f[:, COM_IDX].sum(axis=1)

vel_agora = rg_velocity(phi0_all, A,     b_mean)
vel_glv   = rg_velocity(phi0_all, A_glv, b_glv)

print_stats(vel_agora, 'A-velocity (AGORA W=1.0 A, mean b)')
print_stats(vel_glv,   'A-velocity (gLV α=0.25 LOO A)')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: A matrix permutation test (ODE equilibrium R/G ratio)
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 2: A matrix permutation test (n=200) ===')

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

A_j = jnp.array(A)

def run_to_eq(phi0, b_vec, nsteps=300, dt=1e-4):
    theta = jnp.concatenate([A_j.ravel(), jnp.array(b_vec)])
    phibar = simulate_0d_nsp(theta, n_sp=N_G, n_steps=nsteps, dt=dt,
                              phi_init=jnp.array(phi0),
                              c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)

run_jit = jax.jit(run_to_eq)

# True A correlation (ODE)
print('  Computing true A ODE equilibrium...', flush=True)
eq_true = np.array([run_jit(phi0_all[s], b_mean) for s in range(n_samp)])
rg_eq_true = rg_ratio(eq_true)
rho_true, p_true = stats.spearmanr(diag_num, rg_eq_true)
print(f'  True A: ρ={rho_true:.3f}  p={p_true:.4f}', flush=True)

# Permuted A
N_PERM = 200
rng_perm = np.random.default_rng(42)
perm_rhos = []
print(f'  Running {N_PERM} permutations...', flush=True)

for k in range(N_PERM):
    # Symmetric permutation: shuffle guild labels
    perm = rng_perm.permutation(N_G)
    A_perm = A[np.ix_(perm, perm)]
    A_p_j  = jnp.array(A_perm)

    def run_perm(phi0, b_vec):
        theta = jnp.concatenate([A_p_j.ravel(), jnp.array(b_vec)])
        phibar = simulate_0d_nsp(theta, n_sp=N_G, n_steps=300, dt=1e-4,
                                  phi_init=jnp.array(phi0),
                                  c_const=25.0, alpha_const=100.0)
        eq = phibar[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)
    run_perm_jit = jax.jit(run_perm)

    eq_p = np.array([run_perm_jit(phi0_all[s], b_mean) for s in range(n_samp)])
    rg_p = rg_ratio(eq_p)
    rho_p, _ = stats.spearmanr(diag_num, rg_p)
    perm_rhos.append(rho_p)

    if (k + 1) % 50 == 0:
        print(f'  {k+1}/{N_PERM}  perm mean ρ={np.mean(perm_rhos):.3f}', flush=True)

perm_rhos = np.array(perm_rhos)
p_perm = float((perm_rhos >= rho_true).mean())
print(f'\n  True A ρ = {rho_true:.3f}')
print(f'  Perm ρ   = {perm_rhos.mean():.3f} ± {perm_rhos.std():.3f}')
print(f'  Permutation p = {p_perm:.4f} ({int((perm_rhos >= rho_true).sum())}/{N_PERM} >= true)')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: CT attractor classification
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 3: CT attractor classification ===')

# Classify each sample by ODE equilibrium R/G ratio
ct_pred = np.where(rg_eq_true < 0, 'CT1 (commensal)', 'CT2 (dysbiotic)')
ct_bin  = (rg_eq_true >= 0).astype(int)   # 0=CT1, 1=CT2

print('  CT predictions:')
for diag in DIAG_ORDER:
    mask = diag_arr == diag
    n_ct2 = ct_bin[mask].sum()
    n_tot = mask.sum()
    print(f'    {diag:22s}: CT2 = {n_ct2}/{n_tot} = {n_ct2/n_tot:.1%}')

# Fisher's exact test: CT2 enriched in PI vs Health?
ct2_pi     = ct_bin[diag_arr == 'Peri-implantitis'].sum()
ct2_health = ct_bin[diag_arr == 'Health'].sum()
n_pi       = (diag_arr == 'Peri-implantitis').sum()
n_health   = (diag_arr == 'Health').sum()
contingency = np.array([[ct2_health,        n_health - ct2_health],
                         [ct2_pi,            n_pi - ct2_pi]])
odds, p_fisher = stats.fisher_exact(contingency, alternative='less')
print(f'\n  Fisher exact (CT2 enriched in PI vs Health): OR={odds:.2f}  p={p_fisher:.4f}')

# Spearman correlation of CT bin with diagnosis
rho_ct, p_ct = stats.spearmanr(diag_num, ct_bin)
print(f'  Spearman ρ(diagnosis, CT2 prediction) = {rho_ct:.3f}  p={p_ct:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Pathobiont effective growth rate (R/G-free)
# f_i = b_i + Σ_j A_ij φ_j  for dysbiotic guilds
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 4: Pathobiont effective growth rate (R/G-free) ===')

def pathobiont_growth(phi_mat, A, b):
    """Per-sample effective growth rate of each guild under A and b."""
    return b + (A @ phi_mat.T).T          # (n_samp, N_G)

f_agora = pathobiont_growth(phi0_all, A,     b_mean)   # (127, 10)
f_glv   = pathobiont_growth(phi0_all, A_glv, b_glv)

for guild_name, gi in [('Bacteroidia (Porphyromonas)',  GUILD_IDX['Bacteroidia']),
                        ('Fusobacteriia (Fusobacterium)', GUILD_IDX['Fusobacteriia']),
                        ('Bacilli (Streptococcus)',        GUILD_IDX['Bacilli'])]:
    rho_a, p_a = stats.spearmanr(diag_num, f_agora[:, gi])
    rho_g, p_g = stats.spearmanr(diag_num, f_glv[:, gi])
    print(f'  {guild_name}')
    print(f'    AGORA A: ρ={rho_a:+.3f}  p={p_a:.3f}  |  gLV A: ρ={rho_g:+.3f}  p={p_g:.3f}')

# Dysbiotic − commensal mean growth (R/G-free summary)
f_dysbiotic = f_agora[:, DYS_IDX].mean(axis=1)
f_commensal = f_agora[:, COM_IDX].mean(axis=1)
f_diff      = f_dysbiotic - f_commensal
rho_fd, p_fd = stats.spearmanr(diag_num, f_diff)
print(f'  Δf(dys−com) AGORA A: ρ={rho_fd:+.3f}  p={p_fd:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Bray-Curtis from healthy equilibrium (R/G-free)
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 5: Bray-Curtis from healthy equilibrium (R/G-free) ===')

from scipy.spatial.distance import braycurtis

# Healthy equilibrium centroid (from ODE)
health_eq = eq_true[diag_arr == 'Health']
health_centroid = health_eq.mean(axis=0)

bc_from_health = np.array([braycurtis(eq_true[s], health_centroid)
                            for s in range(n_samp)])
rho_bc, p_bc = stats.spearmanr(diag_num, bc_from_health)
print(f'  BC from healthy eq centroid: ρ={rho_bc:.3f}  p={p_bc:.4f}')
for d in DIAG_ORDER:
    v = bc_from_health[diag_arr == d]
    print(f'    {d:22s}: mean BC={v.mean():.3f} ± {v.std():.3f}')

h_bc, ph_bc = stats.kruskal(*[bc_from_health[diag_arr==d] for d in DIAG_ORDER])
print(f'  Kruskal-Wallis H={h_bc:.2f}  p={ph_bc:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Permutation test using Bray-Curtis (R/G-free, true A validation)
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 6: Permutation test — BC outcome (R/G-free, n=200) ===')

perm_rhos_bc = []
for k in range(N_PERM):
    perm = np.random.default_rng(k + 1000).permutation(N_G)
    A_perm = A[np.ix_(perm, perm)]
    A_p_j2  = jnp.array(A_perm)

    def run_perm2(phi0, b_vec):
        theta = jnp.concatenate([A_p_j2.ravel(), jnp.array(b_vec)])
        phibar = simulate_0d_nsp(theta, n_sp=N_G, n_steps=300, dt=1e-4,
                                  phi_init=jnp.array(phi0),
                                  c_const=25.0, alpha_const=100.0)
        eq = phibar[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)
    run_perm2_jit = jax.jit(run_perm2)

    eq_p2  = np.array([run_perm2_jit(phi0_all[s], b_mean) for s in range(n_samp)])
    hc_p   = eq_p2[diag_arr == 'Health'].mean(axis=0)
    bc_p   = np.array([braycurtis(eq_p2[s], hc_p) for s in range(n_samp)])
    rho_p2, _ = stats.spearmanr(diag_num, bc_p)
    perm_rhos_bc.append(rho_p2)

    if (k + 1) % 50 == 0:
        print(f'  {k+1}/{N_PERM}  perm mean ρ={np.mean(perm_rhos_bc):.3f}', flush=True)

perm_rhos_bc = np.array(perm_rhos_bc)
p_perm_bc    = float((perm_rhos_bc >= rho_bc).mean())
print(f'\n  True A BC ρ   = {rho_bc:.3f}')
print(f'  Perm BC ρ     = {perm_rhos_bc.mean():.3f} ± {perm_rhos_bc.std():.3f}')
print(f'  Permutation p = {p_perm_bc:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
results = {
    'n_samples': int(n_samp),
    'analysis1_velocity_rg': {
        'agora': {'rho': float(stats.spearmanr(diag_num, vel_agora)[0]),
                  'p':   float(stats.spearmanr(diag_num, vel_agora)[1])},
        'glv':   {'rho': float(stats.spearmanr(diag_num, vel_glv)[0]),
                  'p':   float(stats.spearmanr(diag_num, vel_glv)[1])},
    },
    'analysis2_permutation_rg': {
        'true_rho':        float(rho_true),
        'true_p':          float(p_true),
        'perm_rho_mean':   float(perm_rhos.mean()),
        'perm_rho_std':    float(perm_rhos.std()),
        'perm_p':          float(p_perm),
        'n_perm':          N_PERM,
    },
    'analysis3_ct': {
        'rho_ct_diag':     float(rho_ct),
        'p_ct_diag':       float(p_ct),
        'fisher_p':        float(p_fisher),
        'fisher_or':       float(odds),
        'ct2_rate_by_diag': {d: float(ct_bin[diag_arr==d].mean()) for d in DIAG_ORDER},
    },
    'analysis4_pathobiont_growth': {
        'Bacteroidia': {'rho': float(stats.spearmanr(diag_num, f_agora[:, GUILD_IDX['Bacteroidia']])[0]),
                        'p':   float(stats.spearmanr(diag_num, f_agora[:, GUILD_IDX['Bacteroidia']])[1])},
        'Fusobacteriia': {'rho': float(stats.spearmanr(diag_num, f_agora[:, GUILD_IDX['Fusobacteriia']])[0]),
                          'p':   float(stats.spearmanr(diag_num, f_agora[:, GUILD_IDX['Fusobacteriia']])[1])},
        'delta_f_dys_com': {'rho': float(rho_fd), 'p': float(p_fd)},
    },
    'analysis5_bray_curtis': {
        'rho_bc_diag': float(rho_bc),
        'p_bc_diag':   float(p_bc),
        'kw_p':        float(ph_bc),
        'mean_bc_by_diag': {d: float(bc_from_health[diag_arr==d].mean()) for d in DIAG_ORDER},
    },
    'analysis6_permutation_bc': {
        'true_rho':        float(rho_bc),
        'perm_rho_mean':   float(perm_rhos_bc.mean()),
        'perm_rho_std':    float(perm_rhos_bc.std()),
        'perm_p':          float(p_perm_bc),
        'n_perm':          N_PERM,
    },
}
out_json = CR / 'joshi_Amatrix_validation.json'
json.dump(results, open(out_json, 'w'), indent=2)
print(f'\nSaved → {out_json}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure: 3-panel summary
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

# Panel A: velocity violin
ax = axes[0]
data_v = [vel_agora[diag_arr == d] for d in DIAG_ORDER]
parts  = ax.violinplot(data_v, positions=range(3),
                        showmedians=True, showextrema=False, widths=0.6)
for pc, d in zip(parts['bodies'], DIAG_ORDER):
    pc.set_facecolor(DIAG_COLORS[d]); pc.set_alpha(0.7)
parts['cmedians'].set_color('k'); parts['cmedians'].set_linewidth(1.5)
for xi, d in enumerate(DIAG_ORDER):
    vals = vel_agora[diag_arr == d]
    jx   = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
    ax.scatter(xi + jx, vals, s=8, color=DIAG_COLORS[d], alpha=0.5, zorder=3)
ax.axhline(0, color='k', lw=0.6, ls='--')
ax.set_xticks(range(3))
ax.set_xticklabels([DIAG_SHORT[d] for d in DIAG_ORDER])
ax.set_ylabel(r'$d(\mathrm{R/G})/dt$ under $A$ matrix')
rho_v, p_v = stats.spearmanr(diag_num, vel_agora)
ax.set_title(f'(A) A-matrix velocity R/G rate\n'
             r'$\rho$' + f'={rho_v:.2f}  p={p_v:.3f}', fontsize=10)

# Panel B: permutation distribution
ax2 = axes[1]
ax2.hist(perm_rhos, bins=30, color='#aec7e8', edgecolor='white', alpha=0.9,
         label=f'Random A ({N_PERM} perms)')
ax2.axvline(rho_true, color='#c0392b', lw=2, label=f'True A (ρ={rho_true:.2f})')
ax2.axvline(perm_rhos.mean(), color='k', lw=1, ls='--',
            label=f'Perm mean={perm_rhos.mean():.2f}')
ax2.set_xlabel(r'Spearman $\rho$ (ODE eq. R/G vs diagnosis)')
ax2.set_ylabel('Count')
ax2.set_title(f'(B) A matrix permutation test\n'
              f'p={p_perm:.3f} (true A vs random)', fontsize=10)
ax2.legend(fontsize=8)

# Panel C: CT classification
ax3 = axes[2]
diags = DIAG_ORDER
ct2_rates = [ct_bin[diag_arr == d].mean() for d in diags]
ct1_rates = [1 - r for r in ct2_rates]
x = np.arange(len(diags))
bars2 = ax3.bar(x, ct2_rates, color='#e74c3c', alpha=0.8, label='CT2 (dysbiotic)')
bars1 = ax3.bar(x, ct1_rates, bottom=ct2_rates, color='#2ecc71', alpha=0.8, label='CT1 (commensal)')
ax3.set_xticks(x)
ax3.set_xticklabels([DIAG_SHORT[d] for d in diags])
ax3.set_ylabel('Fraction of samples')
ax3.set_ylim(0, 1)
ax3.set_title(f'(C) CT attractor prediction\n'
              f'Fisher p={p_fisher:.3f}  ρ={rho_ct:.2f}', fontsize=10)
ax3.legend(fontsize=8, loc='lower right')
for i, r in enumerate(ct2_rates):
    ax3.text(i, r/2, f'{r:.0%}', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')

fig.suptitle('Dieckow $A$ matrix validation against Joshi 2025 ($n=127$)',
             fontsize=11, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    out = FIG / f'fig_joshi_Amatrix_validation.{ext}'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
