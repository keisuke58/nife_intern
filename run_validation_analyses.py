#!/usr/bin/env python3
"""
run_validation_analyses.py — AGORA sign constraint validation strengthening.

Analysis 1: Medium sensitivity
  Run pFBA with 3 oral media (v1/saliva, v2/realistic saliva, GCF)
  → compare sign agreement with fitted A for each medium

Analysis 2: Random baseline
  Randomly permute sign constraint ±1, refit gLV (full cohort, not LOO)
  100 iterations → distribution of training RMSE and sign agreement
  → show AGORA sign constraint is significantly better than random

Usage:
    python run_validation_analyses.py [--gpu 1] [--n-random 100]
    python run_validation_analyses.py --skip-fba [--gpu 1]  # random baseline only
    XLA_FLAGS='--xla_gpu_enable_command_buffer=' python run_validation_analyses.py ...  # OOM fix
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',       type=int,   default=2)
parser.add_argument('--n-random',  type=int,   default=100)
parser.add_argument('--skip-fba',  action='store_true', help='skip FBA (use cached results)')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import optax

from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import build_net_flow_expanded
from guild_agora_signs import (
    ORAL_MEDIUM, ORAL_MEDIUM_V2, ORAL_MEDIUM_GCF,
    GUILD_REPS, find_model_path, load_model, apply_medium,
    run_pfba, get_secretions, get_substrates,
)

CR     = _here / 'results' / 'dieckow_cr'
AGORA_DIR = _here / 'data' / 'homd_db' / 'agora_gems'
OUT_JSON  = CR / 'validation_sensitivity.json'

PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
PATIENTS = list('ABCDEFGHKL')
N_P = 10
SIGMA = 0.15
LAM   = 1e-4

# ── Load best A (AGORA W=1.0 MAP) and phi_obs ─────────────────────────────
d_best = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
A_best = np.array(d_best['A'])
phi_obs_np = np.load(PHI_NPY)
phi_obs    = jnp.array(phi_obs_np)

# ── Reference sign matrix (L1+L2 base) ────────────────────────────────────
net_l12 = build_net_flow_expanded(use_agora=False, verbose=False)
net_sym_l12 = (net_l12 + net_l12.T) / 2.0


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Medium sensitivity FBA
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('ANALYSIS 1: Medium sensitivity')
print('='*60)

def run_pfba_medium(medium_dict, label):
    """Run pFBA for all guilds with given medium, return sign matrix."""
    print(f'\n  Medium: {label}')
    sign_mat = np.zeros((N_G, N_G))
    secretions = {}
    substrates = {}
    for guild in GUILD_ORDER:
        path = find_model_path(AGORA_DIR, GUILD_REPS.get(guild, [guild]))
        if path is None:
            print(f'    [MISSING] {guild}')
            continue
        try:
            m = load_model(path)
            apply_medium(m, guild, medium_dict=medium_dict)
            sol = run_pfba(m)
            if sol is None:
                print(f'    [INFEASIBLE] {guild}')
                continue
            secretions[guild] = get_secretions(sol, m)
            substrates[guild] = get_substrates(m)
            mu = sol.objective_value
            print(f'    {guild:22s}: μ={mu:.3f}  sec={len(secretions[guild])}')
        except Exception as e:
            print(f'    [ERROR] {guild}: {e}')

    # Pairwise sign from secretion/uptake
    for src in secretions:
        j = GUILD_ORDER.index(src)
        for tgt in substrates:
            if src == tgt: continue
            i = GUILD_ORDER.index(tgt)
            cross = sum(
                flux for met, flux in secretions[src].items()
                if met in substrates[tgt]
            )
            if cross > 0.01:
                sign_mat[i, j] = +1

    # Symmetrize
    sign_sym = (sign_mat + sign_mat.T)
    sign_sym = np.sign(sign_sym)

    # Compute agreement with fitted A_best
    pairs_with_sign = [(i, j) for i in range(N_G) for j in range(i+1, N_G)
                       if sign_sym[i, j] != 0]
    agree = sum(1 for i, j in pairs_with_sign
                if np.sign(A_best[i, j]) == sign_sym[i, j])
    n_pairs = len(pairs_with_sign)

    # Also compute agreement excluding muted (|A| < 0.10)
    strong_pairs = [(i, j) for i, j in pairs_with_sign if abs(A_best[i, j]) > 0.10]
    agree_strong = sum(1 for i, j in strong_pairs
                       if np.sign(A_best[i, j]) == sign_sym[i, j])

    pct  = agree / n_pairs * 100 if n_pairs else 0
    pct_s = agree_strong / len(strong_pairs) * 100 if strong_pairs else 0

    print(f'  → {label}: {agree}/{n_pairs} = {pct:.1f}%  '
          f'(|A|>0.10: {agree_strong}/{len(strong_pairs)} = {pct_s:.1f}%)')
    return {
        'medium': label, 'agree': agree, 'total': n_pairs,
        'pct': round(pct, 1),
        'agree_strong': agree_strong, 'total_strong': len(strong_pairs),
        'pct_strong': round(pct_s, 1),
        'sign_matrix': sign_sym.tolist(),
    }

media_results = {}
if not args.skip_fba:
    for label, med in [('v1_saliva', ORAL_MEDIUM),
                        ('v2_saliva', ORAL_MEDIUM_V2),
                        ('gcf',       ORAL_MEDIUM_GCF)]:
        media_results[label] = run_pfba_medium(med, label)
    # Save intermediate FBA results immediately
    _interim = {'medium_sensitivity': media_results, 'random_baseline': None}
    with open(OUT_JSON, 'w') as _f:
        json.dump(_interim, _f, indent=2)
    print(f'\n  [Intermediate] FBA results saved → {OUT_JSON}')
else:
    # Load previously saved FBA results if available
    if OUT_JSON.exists():
        _prev = json.load(open(OUT_JSON))
        media_results = _prev.get('medium_sensitivity', {})
        print(f'  Loaded cached FBA results from {OUT_JSON}')
    else:
        print('  (skipped — no cached results found)')


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Random baseline (shuffle sign constraint)
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('ANALYSIS 2: Random baseline (sign permutation)')
print('='*60)

# gLV fitter (simplified — full fit, not LOO)
from hamilton_ode_jax_nsp import simulate_0d_nsp

n_sp = N_G
n_A  = n_sp * (n_sp + 1) // 2
triu_r, triu_c = np.tril_indices(n_sp)

def make_A(theta_A):
    A = jnp.zeros((n_sp, n_sp))
    A = A.at[triu_r, triu_c].set(theta_A)
    return A + A.T - jnp.diag(jnp.diag(A))

def pred_week(theta_A, b_p, phi0):
    A = make_A(theta_A)
    theta = jnp.concatenate([A[triu_r, triu_c], b_p])
    phi2 = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=100, dt=1e-2,
                            phi_init=phi0, c_const=25.0, alpha_const=100.0)[-1]
    phi3 = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=100, dt=1e-2,
                            phi_init=phi2, c_const=25.0, alpha_const=100.0)[-1]
    return phi2, phi3

_pred_all = jax.jit(jax.vmap(pred_week, in_axes=(None, 0, 0)))

def fit_glv_with_signs(sign_mat_np, n_steps=300, seed=0):
    """Fit gLV with given sign matrix as constraint. Return (rmse, sign_agreement)."""
    sign_mat_j = jnp.array(sign_mat_np)
    mask = (sign_mat_np != 0) & (~np.eye(n_sp, dtype=bool))

    rng = np.random.default_rng(seed)
    theta_A = jnp.array(rng.normal(0, 0.1, n_A))
    b_all   = jnp.array(np.full((N_P, n_sp), 0.1))

    @jax.jit
    def loss(theta_A, b_all):
        A = make_A(theta_A)
        phi2_all, phi3_all = _pred_all(theta_A, b_all, phi_obs[:, 0])
        sq = (jnp.sum((phi2_all - phi_obs[:, 1])**2) +
              jnp.sum((phi3_all - phi_obs[:, 2])**2))
        # sign penalty
        sgn_viol = jnp.where(
            mask,
            jnp.maximum(0.0, -sign_mat_j * A)**2 / (2 * SIGMA**2),
            0.0
        )
        return sq + jnp.sum(sgn_viol) + LAM * jnp.sum(theta_A**2)

    opt = optax.adam(1e-3)
    state = opt.init((theta_A, b_all))

    for _ in range(n_steps):
        grads = jax.grad(loss, argnums=(0, 1))(theta_A, b_all)
        updates, state = opt.update(grads, state)
        theta_A = theta_A + updates[0]
        b_all   = b_all   + updates[1]

    A_fit = np.array(make_A(theta_A))
    phi2, phi3 = _pred_all(theta_A, b_all, phi_obs[:, 0])
    rmse = float(jnp.sqrt(jnp.mean((phi2 - phi_obs[:, 1])**2 +
                                    (phi3 - phi_obs[:, 2])**2)))

    # sign agreement
    pairs = [(i, j) for i in range(n_sp) for j in range(i+1, n_sp)
             if sign_mat_np[i, j] != 0]
    sa = np.mean([np.sign(A_fit[i, j]) == sign_mat_np[i, j]
                  for i, j in pairs]) if pairs else np.nan
    return rmse, sa, A_fit

# Load AGORA W=1.0 sign matrix as reference
net_agora   = build_net_flow_expanded(use_agora=True, verbose=False, agora_weight=1.0)
net_sym_agora = (net_agora + net_agora.T) / 2.0
sign_agora  = np.sign(net_sym_agora)

# Constrained pairs
pairs_constrained = [(i, j) for i in range(n_sp) for j in range(i+1, n_sp)
                     if sign_agora[i, j] != 0]
print(f'  Constrained pairs: {len(pairs_constrained)}')

# Fit with AGORA signs (reference)
print('  Fitting reference (AGORA W=1.0 signs)...')
rmse_agora, sa_agora, _ = fit_glv_with_signs(sign_agora, n_steps=500)
print(f'  AGORA reference: RMSE={rmse_agora:.4f}  SA={sa_agora:.1%}')

# Random permutations
print(f'\n  Running {args.n_random} random permutations...')
random_rmses, random_sas = [], []
rng = np.random.default_rng(42)
t0 = time.time()

for k in range(args.n_random):
    # Random ±1 for each constrained pair
    rand_sign = np.zeros((n_sp, n_sp))
    for i, j in pairs_constrained:
        s = rng.choice([-1, 1])
        rand_sign[i, j] = s
        rand_sign[j, i] = s

    rmse_r, sa_r, _ = fit_glv_with_signs(rand_sign, n_steps=300, seed=k)
    random_rmses.append(rmse_r)
    random_sas.append(sa_r)

    if (k+1) % 10 == 0:
        elapsed = time.time() - t0
        print(f'  {k+1:3d}/{args.n_random}  mean RMSE={np.mean(random_rmses):.4f}  '
              f't={elapsed:.0f}s')

random_rmses = np.array(random_rmses)
random_sas   = np.array(random_sas)

pct_rank = float((random_rmses > rmse_agora).mean()) * 100
print(f'\n  === Random baseline summary ===')
print(f'  AGORA RMSE:  {rmse_agora:.4f}')
print(f'  Random RMSE: {random_rmses.mean():.4f} ± {random_rmses.std():.4f}')
print(f'  AGORA beats {pct_rank:.0f}% of random permutations')
print(f'  AGORA SA:    {sa_agora:.1%}')
print(f'  Random SA:   {random_sas.mean():.1%} ± {random_sas.std():.1%}  (expected ~50%)')

# ── Save results ─────────────────────────────────────────────────────────
result = {
    'medium_sensitivity': media_results,
    'random_baseline': {
        'n_random': args.n_random,
        'agora_rmse': float(rmse_agora),
        'agora_sa': float(sa_agora),
        'random_rmse_mean': float(random_rmses.mean()),
        'random_rmse_std':  float(random_rmses.std()),
        'random_sa_mean':   float(random_sas.mean()),
        'random_sa_std':    float(random_sas.std()),
        'pct_agora_better': float(pct_rank),
        'random_rmses': random_rmses.tolist(),
        'random_sas':   random_sas.tolist(),
    }
}
with open(OUT_JSON, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nSaved → {OUT_JSON}')
