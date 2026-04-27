#!/usr/bin/env python3
"""
compare_guild_models.py — 10-guild Dieckow: 4 alternative model comparison

Models vs reference gLV (RMSE=0.047, r=0.963):
  1. DT-gLV    discrete-time gLV (linear regression, no ODE)
  2. CR        consumer-resource rank-3 (A=C@P, replicator ODE)
  3. HOI-gLV   gLV + self-interaction c_i*phi_i^2 (replicator ODE)
  4. SDE-gLV   gLV + multiplicative noise (MC prediction intervals)

LOO-CV: approximate (fast) — fix shared params from full fit,
  re-fit patient-specific b from W1→W2, predict W3.
For CR/HOI: true LOO uses fixed C,P and refits b only.
DT-gLV: true LOO (re-fit A on 9 patients, b analytically).

Outputs:
  results/dieckow_cr/model_comparison.json
  results/dieckow_cr/fig_model_comparison.{pdf,png}
  docs/figures/dieckow/fig_model_comparison.{pdf,png}
"""

import json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import pub_style; pub_style.apply()

CR_DIR  = _here / 'results' / 'dieckow_cr'
OTU_DIR = _here / 'results' / 'dieckow_otu'
DOCS    = Path(__file__).resolve().parents[1] / 'docs' / 'figures' / 'dieckow'

# ── Data ──────────────────────────────────────────────────────────────────────
ref      = json.load(open(CR_DIR / 'fit_guild.json'))
N_G      = len(ref['guilds'])
PATIENTS = ref['patients']
N_P      = len(PATIENTS)
PAT10    = list('ABCDEFGHKL')

phi_10   = np.load(OTU_DIR / 'phi_guild.npy')
pat_idx  = [PAT10.index(p) for p in PATIENTS]
phi_all  = phi_10[pat_idx][:, :, :N_G].astype(float)   # (N_P, 3, N_G)

A_ref    = np.array(ref['A'],     dtype=float)
b_ref    = np.array(ref['b_all'], dtype=float)           # (N_P, N_G)

EPS = 1e-8
K_CR = 3

# ── shared ODE (solve_ivp for accuracy) ──────────────────────────────────────
def rep_ivp(phi0, b, A):
    """Replicator ODE via solve_ivp (accurate, NumPy)."""
    def rhs(t, phi):
        f = b + A @ phi; fbar = phi @ f; return phi * (f - fbar)
    sol = solve_ivp(rhs, [0, 1.0], phi0, method='RK45', rtol=1e-6, atol=1e-8)
    p = np.clip(sol.y[:, -1], 0, None); s = p.sum()
    return p / s if s > 1e-12 else np.ones(N_G) / N_G

# ── JAX Euler replicator (lax.fori_loop — no unrolling) ──────────────────────
def sim_jax(phi0, b, A, n_steps=200, dt=0.005):
    """Replicator Euler (JAX, lax.fori_loop — avoids 200-step graph unrolling)."""
    phi = jnp.array(phi0, jnp.float64)
    def step(_, phi):
        f = b + A @ phi; fbar = phi @ f
        phi = jnp.clip(phi + phi * (f - fbar) * dt, 1e-10, None)
        return phi / phi.sum()
    return jax.lax.fori_loop(0, n_steps, step, phi)

def metrics(obs_list, pred_list):
    o = np.concatenate([np.array(x).flatten() for x in obs_list])
    p = np.concatenate([np.array(x).flatten() for x in pred_list])
    return float(np.sqrt(np.mean((o-p)**2))), float(np.corrcoef(o, p)[0,1])


# ══════════════════════════════════════════════════════════════════════════════
# Model 0: reference gLV  (solve_ivp)
# ══════════════════════════════════════════════════════════════════════════════
def eval_glv(A, b_all, phi_all):
    obs, pred = [], []
    for i in range(N_P):
        p2 = rep_ivp(phi_all[i,0], b_all[i], A)
        p3 = rep_ivp(phi_all[i,1], b_all[i], A)
        obs  += [phi_all[i,1], phi_all[i,2]]
        pred += [p2, p3]
    return metrics(obs, pred)

def loo_glv_w3(A, b_all, phi_all):
    """LOO-CV (W3 only): fix A, fit b from W1→W2 with scipy, predict W3."""
    held_rmses = []
    for held in range(N_P):
        def loss_b(b_flat):
            p2 = rep_ivp(phi_all[held,0], b_flat, A)
            return np.sum((p2 - phi_all[held,1])**2)
        res = minimize(loss_b, b_all[held], method='L-BFGS-B',
                       options={'maxiter': 200})
        b_h = res.x
        p3  = rep_ivp(phi_all[held,1], b_h, A)
        held_rmses.append(float(np.sqrt(np.mean((p3 - phi_all[held,2])**2))))
    return float(np.mean(held_rmses)), held_rmses


# ══════════════════════════════════════════════════════════════════════════════
# Model 1: Discrete-Time gLV  (linear regression)
# log(phi_{t+1,i} / phi_{t,i}) = b_{p,i} + sum_j A_{ij} phi_{t,j}
# ══════════════════════════════════════════════════════════════════════════════
def fit_dtglv(phi_tr):
    n_tr = len(phi_tr)
    A = np.zeros((N_G, N_G)); b_mat = np.zeros((n_tr, N_G))
    for g in range(N_G):
        y, X = [], []
        for pi in range(n_tr):
            for t in range(2):
                y.append(np.log(phi_tr[pi,t+1,g]+EPS) - np.log(phi_tr[pi,t,g]+EPS))
                d = np.zeros(n_tr); d[pi] = 1.0
                X.append(np.concatenate([phi_tr[pi,t], d]))
        c, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
        A[g] = c[:N_G]; b_mat[:,g] = c[N_G:]
    return A, b_mat

def pred_dtglv(A, b, phi0):
    lr = np.clip(b + A @ phi0, -20, 20)   # clip to avoid overflow
    phi1 = phi0 * np.exp(lr)
    phi1 = np.clip(phi1, 0, None); s = phi1.sum()
    return phi1 / s if s > 1e-12 else np.ones(N_G) / N_G

def loo_dtglv(phi_all):
    held_rmses = []
    for held in range(N_P):
        tr_idx = [i for i in range(N_P) if i != held]
        A, _   = fit_dtglv(phi_all[tr_idx])
        # fit b_held: b = Δlog(phi_W1→W2) - A@phi_W1
        phi_W1  = phi_all[held, 0]
        phi_W2  = phi_all[held, 1]
        b_h = (np.log(phi_W2+EPS) - np.log(phi_W1+EPS)) - A @ phi_W1
        b_h = np.clip(b_h, -20, 20)
        p3  = pred_dtglv(A, b_h, phi_W2)
        held_rmses.append(float(np.sqrt(np.mean((p3 - phi_all[held,2])**2))))
    return float(np.mean(held_rmses)), held_rmses


# ══════════════════════════════════════════════════════════════════════════════
# Model 2: Consumer-Resource  (A_eff = C @ P, rank K_CR=3)
# Pre-compile JAX grad once for n_tr=N_P; LOO re-fits b only.
# ══════════════════════════════════════════════════════════════════════════════
def _unpack_cr(theta):
    nCP = N_G * K_CR
    return (theta[:nCP].reshape(N_G,K_CR),
            theta[nCP:2*nCP].reshape(K_CR,N_G),
            theta[2*nCP:].reshape(N_P, N_G))

def _loss_cr(theta, phi_j, lam=0.05):
    nCP = N_G * K_CR
    C   = theta[:nCP].reshape(N_G, K_CR)
    P   = theta[nCP:2*nCP].reshape(K_CR, N_G)
    b_m = theta[2*nCP:].reshape(N_P, N_G)
    A   = C @ P; total = 0.0
    for i in range(N_P):
        p2 = sim_jax(phi_j[i,0], b_m[i], A)
        p3 = sim_jax(phi_j[i,1], b_m[i], A)
        total += jnp.sum((p2 - phi_j[i,1])**2) + jnp.sum((p3 - phi_j[i,2])**2)
    return total / (2*N_P) + lam * jnp.sum(A**2)

def fit_cr_full(phi_all):
    phi_j = jnp.array(phi_all)
    U, s, Vt = np.linalg.svd(A_ref)
    C0 = U[:, :K_CR] * np.sqrt(np.maximum(s[:K_CR], 0))
    P0 = (Vt[:K_CR].T * np.sqrt(np.maximum(s[:K_CR], 0))).T
    theta0 = jnp.array(np.concatenate([C0.flatten(), P0.flatten(), b_ref.flatten()]))

    loss_fn = lambda th: _loss_cr(th, phi_j)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    print('  CR: compiling JAX...', flush=True)
    _ = vg(theta0)
    print('  CR: compiled.', flush=True)

    def fn(x): v, g = vg(jnp.array(x)); return float(v), np.array(g)
    res = minimize(fn, np.array(theta0), method='L-BFGS-B', jac=True,
                   options={'maxiter': 500, 'ftol':1e-10, 'gtol':1e-6})
    return _unpack_cr(jnp.array(res.x))

def fit_b_held_cr(C, P, phi01, lam=0.01):
    A_eff = jnp.array(C @ P)
    def loss(b): return jnp.sum((sim_jax(phi01[0], b, A_eff)-phi01[1])**2)+lam*jnp.sum(b**2)
    vg = jax.jit(jax.value_and_grad(loss))
    b0 = jnp.zeros(N_G, jnp.float64)
    _ = vg(b0)
    def fn(x): v,g = vg(jnp.array(x)); return float(v), np.array(g)
    res = minimize(fn, np.zeros(N_G), method='L-BFGS-B', jac=True,
                   options={'maxiter':200})
    return np.array(res.x)

def loo_cr_approx(C, P, phi_all):
    """Approximate LOO: C,P fixed from full fit; re-fit b from W1→W2."""
    A_eff = np.array(C @ P)
    held_rmses = []
    for held in range(N_P):
        b_h = fit_b_held_cr(C, P, phi_all[held,:2])
        p3  = rep_ivp(phi_all[held,1], b_h, A_eff)
        held_rmses.append(float(np.sqrt(np.mean((p3 - phi_all[held,2])**2))))
    return float(np.mean(held_rmses)), held_rmses


# ══════════════════════════════════════════════════════════════════════════════
# Model 3: HOI-gLV  (f_i = b_i + A@phi + c_i * phi_i^2)
# ══════════════════════════════════════════════════════════════════════════════
def _loss_hoi(theta, phi_j, lam_A=0.05, lam_c=0.5):
    nA = N_G*N_G
    A   = theta[:nA].reshape(N_G,N_G)
    c   = theta[nA:nA+N_G]
    b_m = theta[nA+N_G:].reshape(N_P, N_G)
    total = 0.0
    for i in range(N_P):
        def sim_hoi(phi0, bi=b_m[i]):
            phi = jnp.array(phi0, jnp.float64)
            def step(_, phi):
                f = bi + A @ phi + c * phi**2; fbar = phi @ f
                phi = jnp.clip(phi + phi*(f-fbar)*0.005, 1e-10, None)
                return phi / phi.sum()
            return jax.lax.fori_loop(0, 200, step, phi)
        p2 = sim_hoi(phi_j[i,0]); p3 = sim_hoi(phi_j[i,1])
        total += jnp.sum((p2-phi_j[i,1])**2) + jnp.sum((p3-phi_j[i,2])**2)
    return total/(2*N_P) + lam_A*jnp.sum(A**2) + lam_c*jnp.sum(c**2)

def fit_hoi_full(phi_all):
    phi_j = jnp.array(phi_all)
    theta0 = jnp.array(np.concatenate([A_ref.flatten(), np.zeros(N_G), b_ref.flatten()]))

    loss_fn = lambda th: _loss_hoi(th, phi_j)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    print('  HOI: compiling JAX...', flush=True)
    _ = vg(theta0)
    print('  HOI: compiled.', flush=True)

    def fn(x): v,g = vg(jnp.array(x)); return float(v), np.array(g)
    res = minimize(fn, np.array(theta0), method='L-BFGS-B', jac=True,
                   options={'maxiter':500, 'ftol':1e-10, 'gtol':1e-6})
    nA = N_G*N_G
    x  = np.array(res.x)
    return x[:nA].reshape(N_G,N_G), x[nA:nA+N_G], x[nA+N_G:].reshape(N_P,N_G)

def sim_hoi_np(phi0, b, A, c, n_steps=200, dt=0.005):
    phi = np.clip(phi0, 1e-10, None).astype(float); phi /= phi.sum()
    for _ in range(n_steps):
        f = b + A @ phi + c*phi**2; fbar = phi @ f
        phi = np.clip(phi + phi*(f-fbar)*dt, 1e-10, None); phi /= phi.sum()
    return phi

def fit_b_held_hoi(A, c, phi01, lam=0.01):
    A_j = jnp.array(A); c_j = jnp.array(c)
    def loss(b):
        phi = jnp.array(phi01[0], jnp.float64)
        def step(_, phi):
            f = b + A_j@phi + c_j*phi**2; fbar = phi@f
            phi = jnp.clip(phi+phi*(f-fbar)*0.005,1e-10,None); return phi/phi.sum()
        phi = jax.lax.fori_loop(0, 200, step, phi)
        return jnp.sum((phi-phi01[1])**2) + lam*jnp.sum(b**2)
    vg = jax.jit(jax.value_and_grad(loss))
    b0 = jnp.zeros(N_G, jnp.float64); _ = vg(b0)
    def fn(x): v,g=vg(jnp.array(x)); return float(v),np.array(g)
    res = minimize(fn, np.zeros(N_G), method='L-BFGS-B', jac=True,
                   options={'maxiter':200})
    return np.array(res.x)

def loo_hoi_approx(A_hoi, c_hoi, phi_all):
    held_rmses = []
    for held in range(N_P):
        b_h = fit_b_held_hoi(A_hoi, c_hoi, phi_all[held,:2])
        p3  = sim_hoi_np(phi_all[held,1], b_h, A_hoi, c_hoi)
        held_rmses.append(float(np.sqrt(np.mean((p3 - phi_all[held,2])**2))))
    return float(np.mean(held_rmses)), held_rmses


# ══════════════════════════════════════════════════════════════════════════════
# Model 4: SDE-gLV  (multiplicative noise, Euler-Maruyama MC)
# ══════════════════════════════════════════════════════════════════════════════
def estimate_sigma(A, b_all, phi_all):
    resid = []
    for i in range(N_P):
        for t in range(2):
            resid.append(phi_all[i,t+1] - rep_ivp(phi_all[i,t], b_all[i], A))
    return np.std(np.array(resid), axis=0)

def pred_sde_mc(A, b, phi0, sigma, n_mc=300, n_steps=200, dt=0.005):
    rng = np.random.default_rng(42)
    trajs = np.zeros((n_mc, N_G))
    for mc in range(n_mc):
        phi = np.clip(phi0, 1e-10, None); phi /= phi.sum()
        for _ in range(n_steps):
            f = b + A @ phi; fbar = phi @ f
            phi = phi + phi*(f-fbar)*dt + sigma*phi*np.sqrt(dt)*rng.standard_normal(N_G)
            phi = np.clip(phi, 1e-10, None); phi /= phi.sum()
        trajs[mc] = phi
    mean  = trajs.mean(0)
    ci_lo = np.percentile(trajs, 2.5,  0)
    ci_hi = np.percentile(trajs, 97.5, 0)
    return mean, ci_lo, ci_hi


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
results = {}

# ── 0: reference gLV ─────────────────────────────────────────────────────────
print('=== Model 0: gLV (reference) ===', flush=True)
rmse0, r0 = eval_glv(A_ref, b_ref, phi_all)
loo0_d    = json.load(open(CR_DIR / 'loo_cv_glv.json'))
loo0      = float(loo0_d['loo_rmse_mean'])
# Also compute our W3-only LOO for consistent comparison
print('  computing W3 LOO...', flush=True)
loo0_w3, _ = loo_glv_w3(A_ref, b_ref, phi_all)
results['glv'] = {'rmse':rmse0,'r':r0,'loo_rmse':loo0,'loo_w3':loo0_w3,'label':'gLV (ref)'}
print(f'  RMSE={rmse0:.4f}  r={r0:.3f}  LOO(orig)={loo0:.4f}  LOO(W3)={loo0_w3:.4f}')

# ── 1: DT-gLV ────────────────────────────────────────────────────────────────
print('=== Model 1: DT-gLV ===', flush=True)
A_dt, b_dt = fit_dtglv(phi_all)
obs, pred = [], []
for i in range(N_P):
    obs  += [phi_all[i,1], phi_all[i,2]]
    pred += [pred_dtglv(A_dt,b_dt[i],phi_all[i,0]),
             pred_dtglv(A_dt,b_dt[i],phi_all[i,1])]
rmse1, r1    = metrics(obs, pred)
loo1, _      = loo_dtglv(phi_all)
results['dtglv'] = {'rmse':rmse1,'r':r1,'loo_rmse':loo1,'label':'DT-gLV'}
print(f'  RMSE={rmse1:.4f}  r={r1:.3f}  LOO(W3)={loo1:.4f}')

# ── 2: CR K=3 ────────────────────────────────────────────────────────────────
print('=== Model 2: Consumer-Resource K=3 ===', flush=True)
C_cr, P_cr, b_cr = fit_cr_full(phi_all)
A_eff_cr = np.array(C_cr @ P_cr)
obs, pred = [], []
for i in range(N_P):
    obs  += [phi_all[i,1], phi_all[i,2]]
    pred += [rep_ivp(phi_all[i,0], np.array(b_cr[i]), A_eff_cr),
             rep_ivp(phi_all[i,1], np.array(b_cr[i]), A_eff_cr)]
rmse2, r2    = metrics(obs, pred)
loo2, _      = loo_cr_approx(C_cr, P_cr, phi_all)
results['cr'] = {'rmse':rmse2,'r':r2,'loo_rmse':loo2,
                 'C':np.array(C_cr).tolist(),'P':np.array(P_cr).tolist(),
                 'label':f'CR (K={K_CR})'}
print(f'  RMSE={rmse2:.4f}  r={r2:.3f}  LOO(W3)={loo2:.4f}')

# ── 3: HOI-gLV ───────────────────────────────────────────────────────────────
print('=== Model 3: HOI-gLV ===', flush=True)
A_hoi, c_hoi, b_hoi = fit_hoi_full(phi_all)
obs, pred = [], []
for i in range(N_P):
    obs  += [phi_all[i,1], phi_all[i,2]]
    pred += [sim_hoi_np(phi_all[i,0], b_hoi[i], A_hoi, c_hoi),
             sim_hoi_np(phi_all[i,1], b_hoi[i], A_hoi, c_hoi)]
rmse3, r3    = metrics(obs, pred)
loo3, _      = loo_hoi_approx(A_hoi, c_hoi, phi_all)
results['hoi'] = {'rmse':rmse3,'r':r3,'loo_rmse':loo3,
                  'c_vals':c_hoi.tolist(),'label':'HOI-gLV'}
print(f'  RMSE={rmse3:.4f}  r={r3:.3f}  LOO(W3)={loo3:.4f}')
print(f'  c (self-HOI): {np.round(c_hoi,3)}')

# ── 4: SDE-gLV ───────────────────────────────────────────────────────────────
print('=== Model 4: SDE-gLV ===', flush=True)
sigma = estimate_sigma(A_ref, b_ref, phi_all)
obs_l, pred_l = [], []; cov_n = cov_d = 0
for i in range(N_P):
    for t in range(2):
        mn, lo, hi = pred_sde_mc(A_ref, b_ref[i], phi_all[i,t], sigma)
        obs_v = phi_all[i,t+1]
        obs_l.append(obs_v); pred_l.append(mn)
        cov_n += int(np.sum((obs_v>=lo)&(obs_v<=hi))); cov_d += N_G
rmse4, r4 = metrics(obs_l, pred_l); cov = cov_n/cov_d
results['sde'] = {'rmse':rmse4,'r':r4,'loo_rmse':loo0_w3,
                  'coverage_95':float(cov),'sigma':sigma.tolist(),'label':'SDE-gLV'}
print(f'  RMSE={rmse4:.4f}  r={r4:.3f}  LOO=same as gLV  CI coverage={cov:.1%}')

# ── Save ─────────────────────────────────────────────────────────────────────
json.dump(results, open(CR_DIR/'model_comparison.json','w'), indent=2)
print(f'\nSaved: {CR_DIR}/model_comparison.json')

# ── Print summary table ───────────────────────────────────────────────────────
print('\n' + '='*60)
print(f"{'Model':15} {'Train RMSE':>11} {'r':>7} {'LOO(W3) RMSE':>14}")
print('-'*60)
for k in ('glv','dtglv','cr','hoi','sde'):
    d = results[k]
    loo_v = d.get('loo_w3', d['loo_rmse'])
    print(f"  {d['label']:13} {d['rmse']:>11.4f} {d['r']:>7.3f} {loo_v:>14.4f}")
print('='*60)

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════
print('Generating figure...', flush=True)
keys   = ['glv','dtglv','cr','hoi','sde']
labels = [results[k]['label'] for k in keys]
trRMSE = [results[k]['rmse']     for k in keys]
loRMSE = [results[k].get('loo_w3',results[k]['loo_rmse']) for k in keys]
rvals  = [results[k]['r']        for k in keys]
COLORS = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974']

x = np.arange(len(keys)); w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
b1 = ax.bar(x-w/2, trRMSE, w, label='Train RMSE', alpha=0.88,
            color=COLORS)
b2 = ax.bar(x+w/2, loRMSE, w, label='LOO-CV RMSE (W3)', alpha=0.55,
            color=COLORS, hatch='////')
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Train vs LOO-CV (W3 prediction)', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0, max(loRMSE)*1.35)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax = axes[1]
bars = ax.bar(x, rvals, color=COLORS, alpha=0.88)
ax.set_ylabel('Pearson r', fontsize=12)
ax.set_title('Prediction correlation (r)', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0.5, 1.05)
ax.axhline(rvals[0], color=COLORS[0], lw=1.0, ls='--', alpha=0.45)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.suptitle('10-guild model comparison  (Dieckow 10-patient cohort)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

for d in (CR_DIR, DOCS):
    fig.savefig(d/'fig_model_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(d/'fig_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig_model_comparison')
print('\nAll done.')
