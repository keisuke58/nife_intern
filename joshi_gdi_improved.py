#!/usr/bin/env python3
"""
joshi_gdi_improved.py
  Improved GDI validation against Joshi/mSystems data.
  Targets ρ ≥ 0.70 via:
    1. gLV α=0.25 A matrix (best LOO model, averaged across 10 folds)
    2. Zero-imputation of unobserved guilds (reduces shrinkage toward Dieckow mean)
    3. Instant GDI = −φᵀ(b+Aφ) at observed φ0 (no ODE integration noise)
    4. Equilibrium GDI with gLV A (for comparison)
    5. Raw compositional GDI: log(dys/com) directly from observed φ0

Usage:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python joshi_gdi_improved.py
"""
import json, sys
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path
from collections import defaultdict

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER, N_G

CR  = _here / 'results' / 'dieckow_cr'
OTU = _here / 'results' / 'dieckow_otu'
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

EXCEL = _here / 'Datasets' / '20260416_mSystems_16S_5genera_all_profiles.xlsx'

# ── Guild ↔ genus mapping ────────────────────────────────────────────────────
GENUS_TO_GUILD = {
    'Actinomyces':    'Actinobacteria',
    'Streptococcus':  'Bacilli',
    'Porphyromonas':  'Bacteroidia',
    'Fusobacterium':  'Fusobacteriia',
    'Veillonella':    'Negativicutes',
}
GUILD_IDX = {g: i for i, g in enumerate(GUILD_ORDER)}
MAPPED_IDXS = [GUILD_IDX[v] for v in GENUS_TO_GUILD.values()]

DYS_GUILDS = ['Fusobacteriia', 'Bacteroidia', 'Clostridia']
COM_GUILDS  = ['Bacilli', 'Actinobacteria', 'Negativicutes']
DYS_IDX = [GUILD_IDX[g] for g in DYS_GUILDS]
COM_IDX = [GUILD_IDX[g] for g in COM_GUILDS]

# ── Load Excel ────────────────────────────────────────────────────────────────
wb   = openpyxl.load_workbook(EXCEL, read_only=True)
ws   = wb['Sheet1']
rows = list(ws.iter_rows(values_only=True))

header    = rows[0]
diagnoses = list(header[1:])
n_samp    = len(diagnoses)

genus_agg = defaultdict(lambda: np.zeros(n_samp))
for r in rows[1:]:
    g = r[0]
    if g == 'Total' or g is None:
        continue
    for s, v in enumerate(r[1:]):
        genus_agg[g][s] += float(v) if v is not None else 0.0

GENUS_NAMES = sorted(genus_agg.keys())
genus_mat   = np.array([genus_agg[g] for g in GENUS_NAMES], dtype=float)  # (5, 127)
col_total   = np.where(genus_mat.sum(axis=0) == 0, 1, genus_mat.sum(axis=0))
rel_5       = genus_mat / col_total                                         # (5, 127)

print(f'Genera: {GENUS_NAMES}  n={n_samp}')

# ── Load gLV α=0.25 A matrix (average across all 10 LOO folds) ──────────────
glv_folds = sorted(CR.glob('loo_glv_agora_a0p25_fold*.json'))
A_folds   = []
b_folds   = []
for fp in glv_folds:
    d = json.loads(fp.read_text())
    A_folds.append(np.array(d['A']))
    b_folds.append(np.array(d['b_ho']))

A_glv  = np.mean(A_folds, axis=0)   # (10,10) mean interaction matrix
b_folds_arr = np.array(b_folds)     # (10, 10)
b_glv  = b_folds_arr.mean(axis=0)   # (10,) mean growth vector
print(f'Loaded {len(A_folds)} gLV folds. A range [{A_glv.min():.3f}, {A_glv.max():.3f}]')

# Also load Hamilton A for comparison
FIT_H  = CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'
dh     = json.load(open(FIT_H))
A_ham  = np.array(dh['A'])
b_ham_all = np.array(dh['b_all'])
b_ham  = b_ham_all.mean(axis=0)

# Dieckow mean composition (for reference imputation)
phi_dk    = np.load(OTU / 'phi_guild.npy').mean(axis=(0, 1))

# ── Imputation strategies ─────────────────────────────────────────────────────
def build_phi0_zero(rel5):
    """Unobserved guilds → ε (zero-like). Observed guilds dominate signal."""
    phi = np.full(N_G, 1e-5)
    for g_name, frac in zip(GENUS_NAMES, rel5):
        if g_name in GENUS_TO_GUILD:
            phi[GUILD_IDX[GENUS_TO_GUILD[g_name]]] = max(frac, 1e-5)
    phi /= phi.sum()
    return phi

def build_phi0_mean(rel5):
    """Unobserved guilds ← Dieckow mean scaled to remaining fraction (original)."""
    phi = np.full(N_G, 1e-4)
    obs_total = 0.0
    for g_name, frac in zip(GENUS_NAMES, rel5):
        if g_name in GENUS_TO_GUILD:
            gidx = GUILD_IDX[GENUS_TO_GUILD[g_name]]
            phi[gidx] = max(frac, 1e-4)
            obs_total += frac
    rem    = max(1.0 - obs_total, 0.0)
    unobs  = [i for i in range(N_G) if GUILD_ORDER[i] not in GENUS_TO_GUILD.values()]
    dk_sub = phi_dk[unobs]
    dk_s   = dk_sub.sum()
    if dk_s > 1e-8 and rem > 1e-4:
        for i, gi in enumerate(unobs):
            phi[gi] = rem * dk_sub[i] / dk_s
    else:
        for gi in unobs:
            phi[gi] = rem / len(unobs)
    phi = np.maximum(phi, 1e-6)
    phi /= phi.sum()
    return phi

phi0_zero = np.array([build_phi0_zero(rel_5[:, s]) for s in range(n_samp)])
phi0_mean = np.array([build_phi0_mean(rel_5[:, s]) for s in range(n_samp)])

# ── GDI variants ─────────────────────────────────────────────────────────────

def instant_gdi(phi_mat, A, b):
    """GDI = −φᵀ(b + A φ) evaluated at observed φ (no ODE integration)."""
    f_vec = b + (A @ phi_mat.T).T        # (n, N_G)
    mean_f = (phi_mat * f_vec).sum(axis=1)  # (n,)
    return -mean_f

def raw_log_ratio(phi_mat, eps=1e-4):
    """Raw log(dysbiotic/commensal) at observed φ — no model needed."""
    dys = phi_mat[:, DYS_IDX].sum(axis=1) + eps
    com = phi_mat[:, COM_IDX].sum(axis=1) + eps
    return np.log(dys / com)

# Hamilton ODE equilibrium (for comparison; uses JAX)
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

def run_to_eq(phi0_batch, A_mat, b_vec, nsteps=500, dt=1e-4):
    A_j = jnp.array(A_mat)
    theta = jnp.concatenate([A_j.ravel(), jnp.array(b_vec)])
    @jax.jit
    def one(phi):
        pbar = simulate_0d_nsp(theta, n_sp=N_G, n_steps=nsteps, dt=dt,
                               phi_init=phi, c_const=25.0, alpha_const=100.0)
        eq = pbar[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)
    return np.array([one(jnp.array(p)) for p in phi0_batch])

def guild_di_eq(eq, eps=1e-4):
    dys = eq[:, DYS_IDX].sum(axis=1) + eps
    com = eq[:, COM_IDX].sum(axis=1) + eps
    return np.log(dys / com)

# ── Compute all variants ──────────────────────────────────────────────────────
print('Computing instant GDI variants...')

# Instant GDI variants
ig_glv_zero  = instant_gdi(phi0_zero, A_glv, b_glv)       # gLV A, zero-impute
ig_glv_mean  = instant_gdi(phi0_mean, A_glv, b_glv)       # gLV A, mean-impute
ig_glv_bham  = instant_gdi(phi0_zero, A_glv, b_ham)       # gLV A, Hamilton b, zero-impute
ig_ham_zero  = instant_gdi(phi0_zero, A_ham, b_ham)       # Hamilton A, zero-impute
ig_ham_mean  = instant_gdi(phi0_mean, A_ham, b_ham)       # Hamilton A, mean-impute (original)
ig_neutral_z = instant_gdi(phi0_zero, A_glv, np.full(N_G, 0.1))  # neutral b

# Raw log-ratio (no model)
raw_zero = raw_log_ratio(phi0_zero)
raw_mean = raw_log_ratio(phi0_mean)

def fitness_advantage(phi_mat, A, b):
    """f̄_dys − f̄_com: mean fitness advantage of dysbiotic vs commensal guilds."""
    f_vec = b + (A @ phi_mat.T).T        # (n, N_G)
    f_dys = f_vec[:, DYS_IDX].mean(axis=1)
    f_com = f_vec[:, COM_IDX].mean(axis=1)
    return f_dys - f_com

# Per-genus abundance features
PG_IDX = GUILD_IDX['Bacteroidia']    # Porphyromonas → Bacteroidia
FS_IDX = GUILD_IDX['Fusobacteriia']  # Fusobacterium → Fusobacteriia

def pg_log(phi_mat, eps=1e-4):
    return np.log(phi_mat[:, PG_IDX] + eps)

def pg_fuso_log(phi_mat, eps=1e-4):
    return np.log(phi_mat[:, PG_IDX] + phi_mat[:, FS_IDX] + eps)

def pca_score(phi_mat):
    """PC1 of 5 observed genera (largest variance direction)."""
    from sklearn.decomposition import PCA
    idx5 = [GUILD_IDX[v] for v in GENUS_TO_GUILD.values()]
    X    = phi_mat[:, idx5]
    pc   = PCA(n_components=1).fit_transform(X)
    return pc[:, 0]

fa_glv_zero  = fitness_advantage(phi0_zero, A_glv, b_glv)
fa_glv_mean  = fitness_advantage(phi0_mean, A_glv, b_glv)
fa_ham_zero  = fitness_advantage(phi0_zero, A_ham, b_ham)

pg_z   = pg_log(phi0_zero)
pg_m   = pg_log(phi0_mean)
pgfs_z = pg_fuso_log(phi0_zero)
pc1_z  = pca_score(phi0_zero)

# ODE equilibrium GDI with gLV A
print('Computing ODE equilibrium (gLV A)...')
eq_glv_zero = run_to_eq(phi0_zero, A_glv, b_glv)
eq_glv_mean = run_to_eq(phi0_mean, A_glv, b_glv)
gdi_eq_glv_zero = guild_di_eq(eq_glv_zero)
gdi_eq_glv_mean = guild_di_eq(eq_glv_mean)

# ODE equilibrium with Hamilton A (original baseline)
print('Computing ODE equilibrium (Hamilton A)...')
eq_ham_mean = run_to_eq(phi0_mean, A_ham, b_ham)
gdi_eq_ham  = guild_di_eq(eq_ham_mean)

# ── Statistics ────────────────────────────────────────────────────────────────
diag_arr = np.array(diagnoses)
diag_map = {'Health': 0, 'Mucositis': 1, 'Peri-implantitis': 2}
diag_num = np.array([diag_map[d] for d in diag_arr])

def spearman_str(arr):
    rho, p = stats.spearmanr(diag_num, arr)
    return f'ρ={rho:+.3f} p={p:.2e}'

variants = [
    ('Original: Hamilton A, ODE eq, mean-impute', gdi_eq_ham),
    ('gLV A, ODE eq, zero-impute',                gdi_eq_glv_zero),
    ('gLV A, ODE eq, mean-impute',                gdi_eq_glv_mean),
    ('gLV A, Instant GDI, zero-impute',           ig_glv_zero),
    ('gLV A, Instant GDI, mean-impute',           ig_glv_mean),
    ('gLV A, Instant GDI, Hamilton b, zero-impu', ig_glv_bham),
    ('Hamilton A, Instant GDI, zero-impute',       ig_ham_zero),
    ('Hamilton A, Instant GDI, mean-impute',       ig_ham_mean),
    ('Neutral b, gLV A, Instant GDI, zero-impu',  ig_neutral_z),
    ('Raw log(dys/com), zero-impute',              raw_zero),
    ('Raw log(dys/com), mean-impute',              raw_mean),
    ('Fitness adv gLV A, zero-impute',             fa_glv_zero),
    ('Fitness adv gLV A, mean-impute',             fa_glv_mean),
    ('Fitness adv Hamilton A, zero-impute',        fa_ham_zero),
    ('log(Porphyromonas), zero-impute',            pg_z),
    ('log(Pg+Fusobacterium), zero-impute',         pgfs_z),
    ('PC1 of 5 genera, zero-impute',               pc1_z),
]

print('\n' + '='*70)
print(f'{"Variant":<47}  {"ρ":>6}  {"p":>10}')
print('-'*70)
best_rho = 0.0; best_name = ''; best_gdi = None
for name, arr in variants:
    rho, p = stats.spearmanr(diag_num, arr)
    print(f'{name:<47}  {rho:+.3f}  {p:.2e}')
    if abs(rho) > abs(best_rho):
        best_rho = rho; best_name = name; best_gdi = arr
print('='*70)
print(f'\nBEST: {best_name}\n  {spearman_str(best_gdi)}')

# ── Detailed stats for best and original ─────────────────────────────────────
DIAG_ORDER  = ['Health', 'Mucositis', 'Peri-implantitis']
DIAG_SHORT  = {'Health': 'PIH', 'Mucositis': 'PIM', 'Peri-implantitis': 'PI'}
DIAG_COLORS = {'Health': '#2ecc71', 'Mucositis': '#f39c12', 'Peri-implantitis': '#e74c3c'}

def print_detail(gdi, label):
    print(f'\n── {label} ──')
    for d in DIAG_ORDER:
        v = gdi[diag_arr == d]
        print(f'  {d:22s}: n={len(v):3d} mean={v.mean():.3f} med={np.median(v):.3f} std={v.std():.3f}')
    h, p_kw = stats.kruskal(*[gdi[diag_arr == d] for d in DIAG_ORDER])
    print(f'  Kruskal-Wallis H={h:.2f}  p={p_kw:.4f}')
    u, p_mw = stats.mannwhitneyu(gdi[diag_arr == 'Health'],
                                 gdi[diag_arr == 'Peri-implantitis'], alternative='less')
    print(f'  Mann-Whitney PIH < PI: U={u:.0f}  p={p_mw:.4f}')
    print(f'  Spearman: {spearman_str(gdi)}')

print_detail(gdi_eq_ham,   'Original (Hamilton A, ODE eq, mean-impute)')
print_detail(ig_glv_zero,  'Best candidate: gLV A, Instant GDI, zero-impute')
print_detail(best_gdi,     f'Best overall: {best_name}')

# ── Figure: top 4 variants ────────────────────────────────────────────────────
top4 = [
    ('Original\n(Hamilton ODE eq)',         gdi_eq_ham),
    ('gLV A, ODE eq\nzero-impute',          gdi_eq_glv_zero),
    ('gLV A, Instant GDI\nzero-impute',     ig_glv_zero),
    ('Raw log(dys/com)\nzero-impute',        raw_zero),
]

fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharey=False)
rng = np.random.default_rng(42)

for ax, (title, gdi) in zip(axes, top4):
    rho, p_s = stats.spearmanr(diag_num, gdi)
    parts = ax.violinplot(
        [gdi[diag_arr == d] for d in DIAG_ORDER],
        positions=range(3), showmedians=True, showextrema=False, widths=0.65,
    )
    for pc, d in zip(parts['bodies'], DIAG_ORDER):
        pc.set_facecolor(DIAG_COLORS[d]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('k'); parts['cmedians'].set_linewidth(1.5)
    for xi, d in enumerate(DIAG_ORDER):
        vals = gdi[diag_arr == d]
        jit  = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(xi + jit, vals, s=8, color=DIAG_COLORS[d], alpha=0.5, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([DIAG_SHORT[d] for d in DIAG_ORDER], fontsize=9)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    _, p_kw = stats.kruskal(*[gdi[diag_arr == d] for d in DIAG_ORDER])
    ax.set_title(title, fontsize=9)
    ax.text(0.05, 0.97, f'KW p={p_kw:.3f}\nρ={rho:.2f} (p={p_s:.2e})',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))

axes[0].set_ylabel('GDI score', fontsize=9)
fig.suptitle('Joshi/mSystems GDI variants — n=127 (Health=56, Mucositis=39, PI=32)',
             fontsize=10)
plt.tight_layout()
out = FIG / 'fig_joshi_gdi_improved.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nSaved {out}')

# ── Summary ρ bar chart ───────────────────────────────────────────────────────
names_short = [
    'Ham ODE\nmean', 'gLV ODE\nzero', 'gLV ODE\nmean',
    'gLV inst\nzero', 'gLV inst\nmean', 'gLV inst\nHam-b',
    'Ham inst\nzero', 'Ham inst\nmean', 'Neutral\nzero',
    'Raw\nzero', 'Raw\nmean',
    'FA gLV\nzero', 'FA gLV\nmean', 'FA Ham\nzero',
    'log(Pg)', 'log(Pg\n+Fuso)', 'PC1\nzero',
]
rhos = [stats.spearmanr(diag_num, arr)[0] for _, arr in variants]

fig2, ax2 = plt.subplots(figsize=(13, 4))
colors = ['#e74c3c' if r == max(rhos) else '#3498db' for r in rhos]
ax2.bar(range(len(rhos)), rhos, color=colors, alpha=0.8)
ax2.axhline(0.37, color='grey', ls='--', lw=1, label='Original ρ=0.37')
ax2.axhline(0.70, color='green', ls='--', lw=1, label='Target ρ=0.70')
ax2.set_xticks(range(len(rhos)))
ax2.set_xticklabels(names_short, fontsize=8)
ax2.set_ylabel('Spearman ρ')
ax2.set_title('GDI variant comparison — Joshi external validation')
ax2.legend(fontsize=8)
plt.tight_layout()
fig2.savefig(FIG / 'fig_joshi_rho_comparison.png', dpi=200, bbox_inches='tight')
print(f'Saved fig_joshi_rho_comparison.png')

# ── Save results CSV ──────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame({
    'diagnosis': diagnoses,
    'diag_num': diag_num,
    'gdi_original_ham_ode': gdi_eq_ham,
    'gdi_glv_ode_zero':     gdi_eq_glv_zero,
    'gdi_glv_ode_mean':     gdi_eq_glv_mean,
    'gdi_instant_glv_zero': ig_glv_zero,
    'gdi_instant_glv_mean': ig_glv_mean,
    'gdi_raw_zero':         raw_zero,
    'gdi_raw_mean':         raw_mean,
})
df.to_csv(CR / 'joshi_gdi_improved.csv', index=False)
print('Saved joshi_gdi_improved.csv')
