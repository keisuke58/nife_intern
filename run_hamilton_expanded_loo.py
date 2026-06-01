#!/usr/bin/env python3
"""
run_hamilton_expanded_loo.py — LOO-CV for Hamilton expanded-flow model.

For each held-out patient:
  1. Train A on remaining 9 patients (ODE + sign prior from net_flow_hamilton)
  2. Fit b for held-out patient (A fixed)
  3. Predict weeks 2,3 → LOO-CV RMSE

Usage:
    python run_hamilton_expanded_loo.py --hold 0 --gpu 0 --alpha 0.5
    python run_hamilton_expanded_loo.py --hold 0 --gpu 0 --alpha 0.0  # no competition
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hold',      type=int, required=True)
parser.add_argument('--gpu',       type=int, default=0)
parser.add_argument('--fit-file',  type=str, default='fit_glv_hamilton_kegg_expanded.json',
                    help='JSON fit file used for warm start')
parser.add_argument('--tag',       type=str, default='',
                    help='output filename tag, e.g. "agora_w1p0"')
parser.add_argument('--alpha',        type=float, default=0.0,
                    help='competition_weight for net_flow_hamilton (0=cross-feeding only)')
parser.add_argument('--use-agora',    action='store_true',
                    help='include AGORA2 FBA layer (L3) in sign prior')
parser.add_argument('--agora-medium', type=str, default='v1',
                    choices=['v1', 'v2', 'micom', 'gcf'],
                    help='AGORA mode: v1=blood-plasma pFBA, v2=saliva pFBA, micom=community FBA, gcf=GCF peri-implantitis')
parser.add_argument('--micom-fraction', type=float, default=0.5,
                    help='MICOM cooperative tradeoff fraction tau (default 0.5, Diener 2020)')
parser.add_argument('--no-prior',     action='store_true',
                    help='disable sign prior entirely (pure data fit, baseline)')
parser.add_argument('--prior-mode',  type=str, default='sign',
                    choices=['sign', 'combined'],
                    help='sign: sign penalty only; combined: sign + OLS-calibrated magnitude penalty')
parser.add_argument('--gamma-mag',   type=float, default=0.5,
                    help='[combined] weight for magnitude penalty')
parser.add_argument('--sigma-mag',   type=float, default=0.3,
                    help='[combined] std for magnitude Gaussian prior')
parser.add_argument('--c-agora',     type=float, default=None,
                    help='[combined] scaling constant c; None = OLS-calibrate from fit-file')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER, N_G

PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
PATIENTS = list('ABCDEFGHKL')
SIGMA    = 0.15
LAM      = 1e-4
NSTEPS   = 100
OUT_DIR  = _here / 'results' / 'dieckow_cr'

phi_all = np.load(PHI_NPY)   # (10,3,10)
n_sp = N_G
n_A  = n_sp * (n_sp + 1) // 2

# Warm-start fit file (A, b only — net_flow rebuilt below)
_fit_path = OUT_DIR / args.fit_file
_fit_d    = json.load(open(_fit_path))

# Sign prior: always rebuilt from net_flow_hamilton with the requested alpha.
# This is the only thing that changes across the alpha scan.
from build_net_flow_expanded import net_flow_hamilton
net_sym  = net_flow_hamilton(use_agora=args.use_agora, competition_weight=args.alpha,
                             agora_medium=args.agora_medium,
                             micom_fraction=args.micom_fraction)
print(f'net_flow_hamilton  alpha={args.alpha}  agora={args.use_agora}  '
      f'medium={args.agora_medium}  fraction={args.micom_fraction}  '
      f'pos={int((net_sym>0).sum())}  neg={int((net_sym<0).sum())}', flush=True)
sp_mat   = np.sign(net_sym)
mask_np  = (sp_mat != 0) & (~np.eye(n_sp, dtype=bool))

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from scipy.optimize import minimize as scipy_minimize
from hamilton_ode_jax_nsp import simulate_0d_nsp

hold   = args.hold
tr_idx = [i for i in range(10) if i != hold]
phi_tr = phi_all[tr_idx]   # (9,3,10)
phi_ho = phi_all[hold]     # (3,10)
n_tr   = len(tr_idx)

print(f'LOO fold {hold}: held={PATIENTS[hold]}  train={[PATIENTS[i] for i in tr_idx]}', flush=True)

# Warm start from fit file
d    = _fit_d  # already loaded above
A_w  = np.array(d['A'])
b_w  = np.array(d['b_all'])   # shape (10,10)

diag_idx_np = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])

def _unpack_A(v):
    A = jnp.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A = A.at[i, j].set(v[idx]); A = A.at[j, i].set(v[idx]); idx += 1
    return A

theta_A_warm = np.array([A_w[i, j] for j in range(n_sp) for i in range(j + 1)])
b_tr_warm    = b_w[tr_idx]    # (9,10)

phi_tr_j  = jnp.array(phi_tr)
net_sym_j = jnp.array(net_sym)

# ── Combined prior: load AGORA Phi + OLS-calibrate c ─────────────────────────
PRIOR_MODE = args.prior_mode
GAMMA_MAG  = args.gamma_mag
SIGMA_MAG  = args.sigma_mag
phi_mac_j  = None
mac_mask_j = None
C_AGORA    = args.c_agora

if PRIOR_MODE == 'combined' and args.use_agora:
    from guild_agora_signs import get_agora_phi_matrix
    agora_dir = _here / 'data' / 'homd_db' / 'agora_gems'
    print('Computing AGORA Phi matrix for combined prior...', flush=True)
    phi_mac_np, mac_mask_np = get_agora_phi_matrix(agora_dir, verbose=False)
    phi_mac_j  = jnp.array(phi_mac_np)
    mac_mask_j = jnp.array(mac_mask_np)
    if C_AGORA is None:
        A_cal   = np.array(d['A'])
        _mask   = mac_mask_np & (~np.eye(n_sp, dtype=bool))
        F_cal   = phi_mac_np[_mask]
        A_cal_v = A_cal[_mask]
        C_AGORA = float((F_cal @ A_cal_v) / (F_cal @ F_cal))
        print(f'OLS c = {C_AGORA:.4f}  (r={np.corrcoef(F_cal, A_cal_v)[0,1]:.3f})', flush=True)
elif PRIOR_MODE == 'combined':
    print('WARNING: combined mode requires --use-agora; falling back to sign', flush=True)
    PRIOR_MODE = 'sign'

def _run_sim(theta, phi_init):
    phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=NSTEPS, dt=1e-4,
                              phi_init=phi_init, c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

def _pred_two(theta_A, b_p, phi0):
    theta = jnp.concatenate([theta_A, b_p])
    phi2  = _run_sim(theta, phi0)
    phi3  = _run_sim(theta, phi2)
    return phi2, phi3

_pred_all = jax.vmap(_pred_two, in_axes=(None, 0, 0))

_no_prior = args.no_prior

@jax.jit
def loss_fn(theta_A, b_tr):
    phi2_all, phi3_all = _pred_all(theta_A, b_tr, phi_tr_j[:, 0])
    sq   = jnp.sum((phi2_all - phi_tr_j[:, 1])**2) + \
           jnp.sum((phi3_all - phi_tr_j[:, 2])**2)
    rmse = jnp.sqrt(sq / (n_tr * 2 * n_sp))
    if _no_prior:
        return rmse + LAM * jnp.sum(theta_A**2)
    A      = _unpack_A(theta_A)
    sp_j   = jnp.sign(net_sym_j)
    s_mask = (sp_j != 0) & (~jnp.eye(n_sp, dtype=bool))
    pen_sign = jnp.where(s_mask,
                         jnp.abs(net_sym_j) * jnp.maximum(0., -sp_j * A)**2 / (2*SIGMA**2),
                         0.).sum()
    if PRIOR_MODE == 'combined' and phi_mac_j is not None:
        pen_mag = jnp.where(mac_mask_j & s_mask,
                            (A - C_AGORA * phi_mac_j)**2 / (2.0 * SIGMA_MAG**2),
                            0.).sum()
        pen = pen_sign + GAMMA_MAG * pen_mag
    else:
        pen = pen_sign
    return rmse + pen + LAM * jnp.sum(theta_A**2)

n_flat = n_A + n_tr * n_sp

@jax.jit
def _lag(x_flat):
    ta = x_flat[:n_A]; ba = x_flat[n_A:].reshape(n_tr, n_sp)
    loss, (gA, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(ta, ba)
    return loss, jnp.concatenate([gA, gb.ravel()])

print('Compiling...', flush=True)
x0 = np.concatenate([theta_A_warm, b_tr_warm.ravel()])
init_loss = float(loss_fn(jnp.array(theta_A_warm), jnp.array(b_tr_warm)))
print(f'Initial loss={init_loss:.5f}', flush=True)

call_count = [0]; t0 = time.time()

def fg(x):
    call_count[0] += 1
    loss, grad = _lag(jnp.array(x))
    if call_count[0] % 100 == 1:
        print(f'  iter {call_count[0]:4d}  loss={float(loss):.5f}  ({time.time()-t0:.1f}s)',
              flush=True)
    return float(loss), np.array(grad, dtype=np.float64)

bounds = [(None, None)] * n_flat
for k in diag_idx_np:
    bounds[k] = (None, 0.)

res = scipy_minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7})
print(f'  {res.message}  iters={call_count[0]}  loss={res.fun:.5f}', flush=True)

ta_opt = jnp.array(res.x[:n_A])
A_opt  = _unpack_A(ta_opt)

# Fit b for held-out patient
b_ho_warm = jnp.array(b_w[hold])

def _run_sim_ho(theta_A, b_p, phi0):
    theta = jnp.concatenate([theta_A, b_p])
    phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=NSTEPS, dt=1e-4,
                              phi_init=phi0, c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

@jax.jit
def loss_b(b_p):
    phi_ho_j = jnp.array(phi_ho)
    phi2 = _run_sim_ho(ta_opt, b_p, phi_ho_j[0])
    phi3 = _run_sim_ho(ta_opt, b_p, phi2)
    sq   = jnp.sum((phi2 - phi_ho_j[1])**2) + jnp.sum((phi3 - phi_ho_j[2])**2)
    return jnp.sqrt(sq / (2 * n_sp))

def fg_b(b):
    bj = jnp.array(b)
    loss, grad = jax.value_and_grad(loss_b)(bj)
    return float(loss), np.array(grad, dtype=np.float64)

res_b = scipy_minimize(fg_b, np.array(b_ho_warm), jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
b_ho_opt = jnp.array(res_b.x)

# Evaluate
phi_ho_j = jnp.array(phi_ho)
phi2 = np.array(_run_sim_ho(ta_opt, b_ho_opt, phi_ho_j[0]))
phi3 = np.array(_run_sim_ho(ta_opt, b_ho_opt, jnp.array(phi2)))
rmse = float(np.sqrt((np.mean((phi2 - phi_ho[1])**2) + np.mean((phi3 - phi_ho[2])**2)) / 2))

A_np    = np.array(A_opt)
n_agree = int(((np.sign(A_np) == sp_mat) & mask_np).sum())
n_tot   = int(mask_np.sum())

print(f'\nFold {hold} ({PATIENTS[hold]}): LOO-RMSE={rmse:.4f}  SA={n_agree}/{n_tot}', flush=True)

alpha_tag    = f'_a{str(args.alpha).replace(".","p")}'
agora_tag    = '_agora' if args.use_agora else ''
medium_tag   = f'_med{args.agora_medium}' if args.use_agora and args.agora_medium != 'v1' else ''
frac_tag     = f'_f{str(args.micom_fraction).replace(".","p")}' \
               if args.use_agora and args.agora_medium == 'micom' and args.micom_fraction != 0.5 else ''
noprior_tag  = '_noprior' if args.no_prior else ''
combined_tag = f'_comb_g{args.gamma_mag:.2f}'.replace('.', 'p') if PRIOR_MODE == 'combined' else ''
tag          = f'_{args.tag}' if args.tag else ''
fname = f'loo_expanded{noprior_tag}{agora_tag}{medium_tag}{frac_tag}{alpha_tag}{combined_tag}{tag}_fold{hold}.json'
out = {'patient': PATIENTS[hold], 'hold_idx': hold, 'rmse': rmse,
       'sign_agreement': f'{n_agree}/{n_tot}', 'fit_file': args.fit_file,
       'alpha': args.alpha, 'use_agora': args.use_agora,
       'agora_medium': args.agora_medium, 'micom_fraction': args.micom_fraction,
       'no_prior': args.no_prior,
       'A': A_np.tolist(), 'b_ho': np.array(b_ho_opt).tolist(),
       'model': f'Hamilton expanded LOO (no_prior={args.no_prior}, alpha={args.alpha}, agora={args.use_agora}, medium={args.agora_medium}, fraction={args.micom_fraction}, {n_tot//2} constrained pairs, nsteps={NSTEPS})'}
json.dump(out, open(OUT_DIR / fname, 'w'), indent=2)
print(f'Saved {fname}', flush=True)
