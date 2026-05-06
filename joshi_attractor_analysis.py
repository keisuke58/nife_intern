#!/usr/bin/env python3
"""
joshi_attractor_analysis.py
  Cross-sectional Joshi/mSystems 5-genus data → Hamilton ODE attractor analysis.

  Steps:
    1. Load mSystems 16S 5-genus × 127 sample Excel (Health / Mucositis / PI)
    2. Map genera to 10 guild indices; fill unobserved guilds with Dieckow mean
    3. Normalise to φ ∈ Δ^9 (simplex)
    4. Load Dieckow AGORA W=1.0 A matrix and mean / CT-specific b̂
    5. Run Hamilton ODE to equilibrium per sample
    6. Compute "Guild DI" = log(dysbiotic guilds / commensal guilds) at eq.
    7. Statistical test + figures
"""
import json, sys
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER, N_G

CR  = _here / 'results' / 'dieckow_cr'
OTU = _here / 'results' / 'dieckow_otu'
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

EXCEL = _here / 'Datasets' / '20260416_mSystems_16S_5genera_all_profiles.xlsx'
FIT   = CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'

# ── Guild ↔ genus mapping ────────────────────────────────────────────────────
# mSystems Excel genera → Szafrański guild
GENUS_TO_GUILD = {
    'Actinomyces':    'Actinobacteria',
    'Streptococcus':  'Bacilli',
    'Porphyromonas':  'Bacteroidia',
    'Fusobacterium':  'Fusobacteriia',
    'Veillonella':    'Negativicutes',
}
GUILD_IDX = {g: i for i, g in enumerate(GUILD_ORDER)}

# Ecological DI guilds
DYS_GUILDS = ['Fusobacteriia', 'Bacteroidia', 'Clostridia']     # ↑ with disease
COM_GUILDS = ['Bacilli', 'Actinobacteria', 'Negativicutes']      # ↓ with disease
DYS_IDX = [GUILD_IDX[g] for g in DYS_GUILDS]
COM_IDX = [GUILD_IDX[g] for g in COM_GUILDS]

# ── Load Excel data ──────────────────────────────────────────────────────────
wb = openpyxl.load_workbook(EXCEL, read_only=True)
ws = wb['Sheet1']
rows = list(ws.iter_rows(values_only=True))

header     = rows[0]          # ('Genus', 'PI', 'Health', ...)
diagnoses  = list(header[1:]) # per-sample label
n_samp     = len(diagnoses)

# Aggregate species-level rows to genus-level (sum all rows per genus)
from collections import defaultdict
genus_agg = defaultdict(lambda: np.zeros(n_samp))
for r in rows[1:]:
    g = r[0]
    if g == 'Total' or g is None:
        continue
    for s, v in enumerate(r[1:]):
        genus_agg[g][s] += float(v) if v is not None else 0.0

GENUS_NAMES = sorted(genus_agg.keys())   # ['Actinomyces', 'Fusobacterium', ...]
genus_mat   = np.array([genus_agg[g] for g in GENUS_NAMES], dtype=float)  # (5, 127)
print(f'Genera: {GENUS_NAMES}  samples: {n_samp}')
for g in GENUS_NAMES:
    print(f'  {g}: total reads = {genus_mat[GENUS_NAMES.index(g)].sum():.0f}')

# Normalise raw counts per sample → relative abundance (5 genera / all reads)
col_total = genus_mat.sum(axis=0)
col_total = np.where(col_total == 0, 1, col_total)
rel_5 = genus_mat / col_total                  # (5, 127)

# ── Load Dieckow fit ─────────────────────────────────────────────────────────
d    = json.load(open(FIT))
A    = np.array(d['A'])                        # (10, 10)
b_all = np.array(d['b_all'])                   # (10, 10) patients × guilds
b_mean = b_all.mean(axis=0)                    # (10,) mean across patients

# CT-specific b̂
ct_map = json.load(open(OTU / 'community_types.json'))['patient_ct']
patients = d['patients']
ct_arr   = np.array([ct_map[p] for p in patients])
b_ct1 = b_all[ct_arr == 1].mean(axis=0)
b_ct2 = b_all[ct_arr == 2].mean(axis=0)

# Dieckow mean guild composition (for filling unobserved guilds)
phi_dieckow = np.load(OTU / 'phi_guild.npy')          # (10, 3, 10)
phi_mean_dk = phi_dieckow.mean(axis=(0, 1))            # mean over patients & weeks

# ── Build initial conditions for each Joshi sample ──────────────────────────
# Observed 5 guilds from Joshi, rest from Dieckow mean (scaled to fill remaining fraction)
def build_phi0(rel5_sample):
    """
    rel5_sample : (5,) relative abundance of 5 genera (sums to ≤1, genus-aggregated)
    Strategy: place observed guilds, fill rest with Dieckow mean, renormalise.
    """
    phi = np.full(N_G, 1e-4)   # small baseline for unobserved

    # Assign observed genera (genus-aggregated, no double-counting)
    obs_total = 0.0
    for g_name, frac in zip(GENUS_NAMES, rel5_sample):
        if g_name in GENUS_TO_GUILD:
            gidx = GUILD_IDX[GENUS_TO_GUILD[g_name]]
            phi[gidx] = max(frac, 1e-4)
            obs_total += frac

    # Unobserved guilds: use Dieckow mean scaled to remaining fraction
    rem = max(1.0 - obs_total, 0.0)
    unobs = [i for i in range(N_G) if GUILD_ORDER[i] not in GENUS_TO_GUILD.values()]
    dk_unobs = phi_mean_dk[unobs]
    dk_unobs_sum = dk_unobs.sum()
    if dk_unobs_sum > 1e-8 and rem > 1e-4:
        for i, gi in enumerate(unobs):
            phi[gi] = rem * dk_unobs[i] / dk_unobs_sum
    else:
        for gi in unobs:
            phi[gi] = rem / len(unobs)

    phi = np.maximum(phi, 1e-6)
    phi /= phi.sum()
    return phi

phi0_all = np.array([build_phi0(rel_5[:, s]) for s in range(n_samp)])  # (127, 10)

# ── Hamilton ODE simulation ──────────────────────────────────────────────────
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

A_j = jnp.array(A)

def run_to_eq(phi0, b_vec, nsteps=500, dt=1e-4):
    theta = jnp.concatenate([A_j.ravel(), jnp.array(b_vec)])
    phibar = simulate_0d_nsp(theta, n_sp=N_G, n_steps=nsteps, dt=dt,
                              phi_init=jnp.array(phi0),
                              c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]
    s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)

run_jit = jax.jit(run_to_eq)

print(f'Running ODE for {n_samp} samples (neutral b̂)...', flush=True)
b_neutral = np.full(N_G, 0.1)
eq_neutral = np.array([run_jit(phi0_all[s], b_neutral) for s in range(n_samp)])

print(f'Running ODE for {n_samp} samples (mean b̂)...', flush=True)
eq_mean = np.array([run_jit(phi0_all[s], b_mean)   for s in range(n_samp)])
eq_ct1  = np.array([run_jit(phi0_all[s], b_ct1)    for s in range(n_samp)])
eq_ct2  = np.array([run_jit(phi0_all[s], b_ct2)    for s in range(n_samp)])

def guild_di(eq, eps=1e-4):
    """log(dysbiotic / commensal) at equilibrium."""
    dys = eq[:, DYS_IDX].sum(axis=1) + eps
    com = eq[:, COM_IDX].sum(axis=1) + eps
    return np.log(dys / com)

gdi_neutral = guild_di(eq_neutral)
gdi_mean    = guild_di(eq_mean)
gdi_ct1     = guild_di(eq_ct1)
gdi_ct2     = guild_di(eq_ct2)

# ── Statistics ───────────────────────────────────────────────────────────────
diag_arr = np.array(diagnoses)
diag_map  = {'Health': 0, 'Mucositis': 1, 'Peri-implantitis': 2}
diag_num  = np.array([diag_map[d] for d in diag_arr])

DIAG_ORDER  = ['Health', 'Mucositis', 'Peri-implantitis']
DIAG_COLORS = {'Health': '#2ecc71', 'Mucositis': '#f39c12', 'Peri-implantitis': '#e74c3c'}
DIAG_SHORT  = {'Health': 'PIH', 'Mucositis': 'PIM', 'Peri-implantitis': 'PI'}

def print_stats(gdi, label):
    print(f'\n── {label} ──')
    groups = {d: gdi[diag_arr == d] for d in DIAG_ORDER}
    for d, v in groups.items():
        print(f'  {d:20s}: n={len(v):3d}  mean={v.mean():.3f}  median={np.median(v):.3f}  std={v.std():.3f}')
    h_stat, p_kw = stats.kruskal(*[groups[d] for d in DIAG_ORDER])
    print(f'  Kruskal-Wallis H={h_stat:.2f}  p={p_kw:.4f}')
    t, p_mw = stats.mannwhitneyu(groups['Health'], groups['Peri-implantitis'], alternative='less')
    print(f'  Mann-Whitney Health < PI: U={t:.0f}  p={p_mw:.4f}')
    # Spearman correlation with diagnosis ordinal
    rho, p_sp = stats.spearmanr(diag_num, gdi)
    print(f'  Spearman ρ(diag, GDI)={rho:.3f}  p={p_sp:.4f}')

print_stats(gdi_neutral, 'Neutral b̂')
print_stats(gdi_mean,    'Mean Dieckow b̂')

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
plt.rcParams.update({'font.size': 9})

configs = [
    (gdi_neutral, 'Neutral b̂\n(b=0.1 all)'),
    (gdi_mean,    'Mean Dieckow b̂'),
    (gdi_ct1,     'CT1 b̂'),
    (gdi_ct2,     'CT2 b̂'),
]

for ax, (gdi, title) in zip(axes, configs):
    parts = ax.violinplot(
        [gdi[diag_arr == d] for d in DIAG_ORDER],
        positions=range(3),
        showmedians=True, showextrema=False, widths=0.6,
    )
    for pc, d in zip(parts['bodies'], DIAG_ORDER):
        pc.set_facecolor(DIAG_COLORS[d]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('k'); parts['cmedians'].set_linewidth(1.5)

    # Overlay jitter
    for xi, d in enumerate(DIAG_ORDER):
        vals = gdi[diag_arr == d]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(xi + jitter, vals, s=8, color=DIAG_COLORS[d], alpha=0.5, zorder=3)

    ax.set_xticks(range(3))
    ax.set_xticklabels([DIAG_SHORT[d] for d in DIAG_ORDER], fontsize=9)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_title(title, fontsize=9)

    # Kruskal p-value
    _, p = stats.kruskal(*[gdi[diag_arr == d] for d in DIAG_ORDER])
    rho, p_sp = stats.spearmanr(diag_num, gdi)
    ax.text(0.05, 0.97, f'KW p={p:.3f}\nρ={rho:.2f} p={p_sp:.3f}',
            transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

axes[0].set_ylabel('Guild DI = log(dysbiotic/commensal)', fontsize=9)
fig.suptitle('Joshi/mSystems attractor analysis: Dieckow AGORA W=1.0 A matrix\n'
             'n=127 (Health=56, Mucositis=39, PI=32)',
             fontsize=10)

out = FIG / 'fig_joshi_attractor_analysis.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved {out}')

# ── Equilibrium guild composition by diagnosis ────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
GS = [g[:6] for g in GUILD_ORDER]
x = np.arange(N_G)
for ax, diag in zip(axes2, DIAG_ORDER):
    mask = diag_arr == diag
    eq_d = eq_mean[mask]
    m = eq_d.mean(axis=0)
    s = eq_d.std(axis=0)
    colors = ['#e74c3c' if i in DYS_IDX else '#2ecc71' if i in COM_IDX else '#95a5a6'
              for i in range(N_G)]
    ax.bar(x, m, yerr=s, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(GS, fontsize=7, rotation=45, ha='right')
    ax.set_title(f'{DIAG_SHORT[diag]} (n={mask.sum()})', fontsize=9)
    ax.set_ylim(0, 0.6)
axes2[0].set_ylabel('Equilibrium φ (mean Dieckow b̂)', fontsize=9)
fig2.suptitle('Equilibrium guild composition by diagnosis (Joshi attractor)', fontsize=10)
out2 = FIG / 'fig_joshi_eq_guild_composition.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved {out2}')

# ── Save results JSON ─────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame({
    'sample_idx': range(n_samp),
    'diagnosis': diagnoses,
    'gdi_neutral': gdi_neutral,
    'gdi_mean_b': gdi_mean,
    'gdi_ct1_b': gdi_ct1,
    'gdi_ct2_b': gdi_ct2,
})
df.to_csv(CR / 'joshi_attractor_results.csv', index=False)
print('Saved joshi_attractor_results.csv')
