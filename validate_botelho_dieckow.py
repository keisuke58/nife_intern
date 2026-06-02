#!/usr/bin/env python3
"""
validate_botelho_dieckow.py — External validation of Dieckow A matrix on Botelho 2021.

Strategy:
  - A matrix: fixed from Dieckow AGORA W=1.0 fit (10 patients, 3 weeks)
  - For each of 15 Botelho patients:
      * Fit patient-specific b̂ using T00→T02 only (A fixed)
      * Predict T04..T12 (5 steps) from T02
  - Compare vs: (a) random A permutation, (b) no-A baseline (b only, A=0)
  - Metrics: RMSE, Bray-Curtis per timepoint

Usage:
    python validate_botelho_dieckow.py [--gpu 2]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.spatial.distance import braycurtis

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',    type=int, default=2)
parser.add_argument('--n-perm', type=int, default=200)
args = parser.parse_args()

import os; os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

DATA = _here / 'data' / 'prjna725874'
CR   = _here / 'results' / 'dieckow_cr'
OUT  = _here / 'results' / 'botelho_validation'
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
phi_all = np.load(DATA / 'phi_guild.npy')           # (15, 7, 10)
meta    = json.load(open(DATA / 'metadata.json'))
PATIENTS   = meta['patients']       # ['1',...,'15']
TIMEPOINTS = meta['timepoints']     # ['00','02','04','06','08','10','12']  months
N_P, N_T   = phi_all.shape[:2]     # 15, 7

# Botelho months → time in years
T_MONTHS = np.array([int(t) for t in TIMEPOINTS], dtype=float)
T_YEARS  = T_MONTHS / 12.0

print(f'Botelho 2021: {N_P} patients × {N_T} timepoints × {N_G} guilds')
print(f'Timepoints: {TIMEPOINTS} months')

# ── Load Dieckow A matrix ──────────────────────────────────────────────────────
d_fit = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
A_dieckow = np.array(d_fit['A'])    # (10, 10)
print(f'\nDieckow A: diagonal mean={np.diag(A_dieckow).mean():.3f}  '
      f'off-diag abs mean={np.abs(A_dieckow[~np.eye(N_G,dtype=bool)]).mean():.3f}')

# ── gLV ODE integrator ─────────────────────────────────────────────────────────
def glv_ode(t, phi, A, b):
    f    = b + A @ phi
    fbar = phi @ f
    dphi = phi * (f - fbar)
    return dphi

def integrate_glv(phi0, A, b, t_eval):
    """Integrate gLV from phi0 over t_eval (years). Returns (len(t_eval), N_G)."""
    sol = solve_ivp(glv_ode, [t_eval[0], t_eval[-1]], phi0, args=(A, b),
                    t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8,
                    dense_output=False)
    if not sol.success:
        return np.tile(phi0, (len(t_eval), 1))
    phi_traj = sol.y.T  # (n_t, N_G)
    # renormalise
    s = phi_traj.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, phi_traj / s, phi_traj)

def fit_b(phi_obs, A, t_train_idx, reg=1e-3):
    """
    Fit patient-specific b using observed timepoints t_train_idx.
    phi_obs: (N_T, N_G), t_train_idx: indices into T_YEARS to use for fitting.
    A: fixed.
    """
    phi0 = phi_obs[t_train_idx[0]]

    def loss(b):
        t_eval  = T_YEARS[t_train_idx]
        phi_hat = integrate_glv(phi0, A, b, t_eval)
        residual = phi_hat - phi_obs[t_train_idx]
        return float(np.sum(residual**2)) + reg * float(np.sum(b**2))

    b0  = np.zeros(N_G)
    res = minimize(loss, b0, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-12})
    return res.x

def predict_trajectory(phi0, A, b, t_pred_idx):
    """Predict phi at t_pred_idx timepoints."""
    t_eval = T_YEARS[t_pred_idx]
    return integrate_glv(phi0, A, b, t_eval)

# ── Per-patient fit & predict ──────────────────────────────────────────────────
# Train b on T00 and T02 (first two timepoints = 0, 2 months)
# Predict T04..T12 (timepoints 2..6, i.e., months 4,6,8,10,12)
TRAIN_IDX = [0, 1]       # T00, T02
PRED_IDX  = list(range(2, N_T))  # T04..T12

def eval_model(A, label='model'):
    rmses, bcs = [], []
    for p in range(N_P):
        phi_obs = phi_all[p]                                  # (7, 10)
        b_hat   = fit_b(phi_obs, A, TRAIN_IDX)
        phi_hat = predict_trajectory(phi_obs[PRED_IDX[0]], A, b_hat, PRED_IDX)

        # RMSE across prediction timepoints
        rmse_p = np.sqrt(np.mean((phi_hat - phi_obs[PRED_IDX])**2))
        # BC per timepoint, then average
        bc_p   = np.mean([braycurtis(phi_hat[i], phi_obs[PRED_IDX[i]])
                          for i in range(len(PRED_IDX))])
        rmses.append(rmse_p)
        bcs.append(bc_p)
    return np.array(rmses), np.array(bcs)

print('\n── Dieckow A (external validation) ──────────────────────────────────')
t0 = time.time()
rmse_dk, bc_dk = eval_model(A_dieckow, 'Dieckow A')
print(f'  RMSE: {rmse_dk.mean():.4f} ± {rmse_dk.std():.4f}  '
      f'BC: {bc_dk.mean():.4f} ± {bc_dk.std():.4f}  [{time.time()-t0:.0f}s]', flush=True)

print('\n── A=0 baseline (b only, no interactions) ───────────────────────────')
rmse_b0, bc_b0 = eval_model(np.zeros((N_G, N_G)), 'A=0')
print(f'  RMSE: {rmse_b0.mean():.4f} ± {rmse_b0.std():.4f}  '
      f'BC: {bc_b0.mean():.4f} ± {bc_b0.std():.4f}', flush=True)

print('\n── Persistence baseline (predict T00 for all future t) ──────────────')
rmse_pers = np.array([
    np.sqrt(np.mean((phi_all[p, PRED_IDX] - phi_all[p, 0:1])**2))
    for p in range(N_P)])
bc_pers = np.array([
    np.mean([braycurtis(phi_all[p, 0], phi_all[p, ti]) for ti in PRED_IDX])
    for p in range(N_P)])
print(f'  RMSE: {rmse_pers.mean():.4f} ± {rmse_pers.std():.4f}  '
      f'BC: {bc_pers.mean():.4f} ± {bc_pers.std():.4f}', flush=True)

print(f'\n── Summary ──────────────────────────────────────────────────────────')
print(f'  Dieckow A vs A=0:        ΔRMSE={rmse_dk.mean()-rmse_b0.mean():+.4f}  '
      f'ΔBC={bc_dk.mean()-bc_b0.mean():+.4f}')
print(f'  Dieckow A vs persistence: ΔRMSE={rmse_dk.mean()-rmse_pers.mean():+.4f}  '
      f'ΔBC={bc_dk.mean()-bc_pers.mean():+.4f}')

# ── A matrix permutation test ─────────────────────────────────────────────────
print(f'\n── Permutation test (n={args.n_perm}) ────────────────────────────────')
rng = np.random.default_rng(42)
perm_rmses, perm_bcs = [], []
t0 = time.time()
for k in range(args.n_perm):
    perm = rng.permutation(N_G)
    A_perm = A_dieckow[np.ix_(perm, perm)]
    r, b  = eval_model(A_perm)
    perm_rmses.append(r.mean())
    perm_bcs.append(b.mean())
    if (k+1) % 50 == 0:
        print(f'  {k+1}/{args.n_perm}  perm RMSE={np.mean(perm_rmses):.4f}  '
              f't={time.time()-t0:.0f}s', flush=True)

perm_rmses = np.array(perm_rmses)
perm_bcs   = np.array(perm_bcs)
p_rmse = float((perm_rmses <= rmse_dk.mean()).mean())
p_bc   = float((perm_bcs   <= bc_dk.mean()  ).mean())
print(f'\n  Dieckow RMSE={rmse_dk.mean():.4f}  perm mean={perm_rmses.mean():.4f} ± {perm_rmses.std():.4f}  p={p_rmse:.4f}')
print(f'  Dieckow BC  ={bc_dk.mean():.4f}  perm mean={perm_bcs.mean():.4f} ± {perm_bcs.std():.4f}  p={p_bc:.4f}')

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    'dataset': 'Botelho 2021 PRJNA725874',
    'n_patients': N_P, 'n_timepoints': N_T, 'n_guilds': N_G,
    'train_timepoints': [TIMEPOINTS[i] for i in TRAIN_IDX],
    'pred_timepoints':  [TIMEPOINTS[i] for i in PRED_IDX],
    'dieckow_A': {
        'rmse_mean': float(rmse_dk.mean()), 'rmse_std': float(rmse_dk.std()),
        'bc_mean':   float(bc_dk.mean()),   'bc_std':   float(bc_dk.std()),
        'rmse_per_patient': rmse_dk.tolist(),
    },
    'A_zero_baseline': {
        'rmse_mean': float(rmse_b0.mean()), 'bc_mean': float(bc_b0.mean()),
    },
    'persistence_baseline': {
        'rmse_mean': float(rmse_pers.mean()), 'bc_mean': float(bc_pers.mean()),
    },
    'permutation_test': {
        'n_perm': args.n_perm,
        'perm_rmse_mean': float(perm_rmses.mean()), 'perm_rmse_std': float(perm_rmses.std()),
        'perm_bc_mean':   float(perm_bcs.mean()),   'perm_bc_std':  float(perm_bcs.std()),
        'p_rmse': p_rmse, 'p_bc': p_bc,
    },
}
out_json = OUT / 'validation_results.json'
json.dump(results, open(out_json, 'w'), indent=2)
print(f'\nSaved → {out_json}')

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

# Panel A: per-patient RMSE
ax = axes[0]
x = np.arange(N_P)
ax.bar(x, rmse_dk, color='#2980b9', alpha=0.8, label=f'Dieckow A (mean={rmse_dk.mean():.3f})')
ax.axhline(rmse_b0.mean(),   color='#e67e22', lw=1.5, ls='--', label=f'A=0 ({rmse_b0.mean():.3f})')
ax.axhline(rmse_pers.mean(), color='#95a5a6', lw=1.5, ls=':',  label=f'Persistence ({rmse_pers.mean():.3f})')
ax.set_xticks(x); ax.set_xticklabels(PATIENTS, fontsize=7)
ax.set_xlabel('Patient'); ax.set_ylabel('RMSE (T04–T12)')
ax.set_title('(A) Per-patient prediction RMSE\n(train on T00–T02, predict T04–T12)', fontsize=10)
ax.legend(fontsize=8)

# Panel B: permutation test RMSE
ax2 = axes[1]
ax2.hist(perm_rmses, bins=25, color='#aec7e8', edgecolor='white', alpha=0.9,
         label=f'Random A (n={args.n_perm})')
ax2.axvline(rmse_dk.mean(), color='#c0392b', lw=2, label=f'Dieckow A ({rmse_dk.mean():.3f})')
ax2.axvline(perm_rmses.mean(), color='k', lw=1, ls='--',
            label=f'Perm mean ({perm_rmses.mean():.3f})')
ax2.set_xlabel('Mean RMSE across 15 patients')
ax2.set_ylabel('Count')
ax2.set_title(f'(B) A matrix permutation test\n(RMSE; p={p_rmse:.3f})', fontsize=10)
ax2.legend(fontsize=8)

# Panel C: per-timepoint BC
ax3 = axes[2]
bc_per_t = np.array([
    np.mean([braycurtis(
        integrate_glv(phi_all[p, PRED_IDX[0]], A_dieckow,
                      fit_b(phi_all[p], A_dieckow, TRAIN_IDX), T_YEARS[PRED_IDX])[ti],
        phi_all[p, PRED_IDX[ti]])
             for p in range(N_P)])
    for ti in range(len(PRED_IDX))
])
t_labels = [TIMEPOINTS[i] for i in PRED_IDX]
ax3.plot(range(len(PRED_IDX)), bc_per_t, 'o-', color='#2980b9', lw=2,
         label=f'Dieckow A')
ax3.set_xticks(range(len(PRED_IDX)))
ax3.set_xticklabels([f'T{t}m' for t in t_labels])
ax3.set_xlabel('Prediction timepoint (months)')
ax3.set_ylabel('Mean Bray-Curtis')
ax3.set_title('(C) Prediction accuracy over time\n(BC from observed)', fontsize=10)
ax3.legend(fontsize=8)

fig.suptitle('External validation: Dieckow $A$ matrix predicts Botelho 2021 (n=15 patients)',
             fontsize=11, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'fig_botelho_validation.{ext}', dpi=150, bbox_inches='tight')
    print(f'Saved {OUT}/fig_botelho_validation.{ext}')
