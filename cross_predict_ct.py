#!/usr/bin/env python3
"""
cross_predict_ct.py — CT1↔CT2 cross-community-type prediction.

Strategy:
  1. Fit A on CT1 patients (E,G,K; n=3), fix A, fit b for CT2 patients from W1→W2
  2. Predict W2,W3 for CT2 using CT1-A + CT2-b
  3. Repeat: fit A on CT2 (A,B,C,F,H; n=5), predict CT1 (D,E,G,K,L; n=5)
  4. Compare: within-CT LOO RMSE vs cross-CT prediction RMSE

Outputs:
  results/dieckow_cr/cross_ct_prediction.json
  docs/figures/dieckow/fig_cross_ct.{pdf,png}
"""

import json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import pub_style; pub_style.apply()
from guild_replicator_dieckow import GUILD_ORDER, GUILD_SHORT

CR_DIR  = _here / 'results' / 'dieckow_cr'
OTU_DIR = _here / 'results' / 'dieckow_otu'
DOCS    = Path(__file__).resolve().parents[1] / 'docs' / 'figures' / 'dieckow'

# ── Data ──────────────────────────────────────────────────────────────────────
ref     = json.load(open(CR_DIR / 'fit_guild.json'))
N_G     = len(ref['guilds'])
PATIENTS = ref['patients']      # 10 patients used in fit
N_P = len(PATIENTS)

PAT10   = list('ABCDEFGHKL')
phi_10  = np.load(OTU_DIR / 'phi_guild.npy')
pat_idx = [PAT10.index(p) for p in PATIENTS]
phi_all = phi_10[pat_idx][:, :, :N_G].astype(float)   # (N_P, 3, N_G)

A_full = np.array(ref['A'])
b_full = np.array(ref['b_all'])

# CT labels from community_types.json
ct_json = json.load(open(OTU_DIR / 'community_types.json'))
pat_ct  = ct_json['patient_ct']   # {'A':2, 'B':2, ...}

CT1_pats = [p for p in PATIENTS if pat_ct.get(p, 0) == 1]   # D,E,G,K,L
CT2_pats = [p for p in PATIENTS if pat_ct.get(p, 0) == 2]   # A,B,C,F,H
CT1_idx  = [PATIENTS.index(p) for p in CT1_pats]
CT2_idx  = [PATIENTS.index(p) for p in CT2_pats]
print(f'CT1 ({len(CT1_pats)}): {CT1_pats}')
print(f'CT2 ({len(CT2_pats)}): {CT2_pats}')

EPS = 1e-8
LAM_A = 1e-4
LAM_B = 1e-2


# ── ODE helpers ───────────────────────────────────────────────────────────────
def rep_ivp(phi0, b, A):
    def rhs(t, phi):
        f = b + A @ phi; fbar = phi @ f; return phi * (f - fbar)
    sol = solve_ivp(rhs, [0, 1.0], phi0, method='RK45', rtol=1e-6, atol=1e-8)
    p = np.clip(sol.y[:, -1], 0, None); s = p.sum()
    return p / s if s > 1e-12 else np.ones(N_G) / N_G


# ── Fit A + b on a patient subset ─────────────────────────────────────────────
def fit_A_on_subset(phi_sub, A_init, b_init, n_starts=3):
    n_tr = len(phi_sub)

    def pack(A, b_mat): return np.concatenate([A.flatten(), b_mat.flatten()])
    def unpack(x):
        A   = x[:N_G*N_G].reshape(N_G, N_G)
        b_m = x[N_G*N_G:].reshape(n_tr, N_G)
        return A, b_m

    def loss(x):
        A, b_m = unpack(x)
        sq = 0.0
        for i in range(n_tr):
            p2 = rep_ivp(phi_sub[i, 0], b_m[i], A)
            p3 = rep_ivp(phi_sub[i, 1], b_m[i], A)
            sq += np.sum((p2 - phi_sub[i, 1])**2) + np.sum((p3 - phi_sub[i, 2])**2)
        return sq / (2*n_tr) + LAM_A * np.sum(A**2) + LAM_B * np.sum(b_m**2)

    best_val, best_x = np.inf, None
    for s in range(n_starts):
        noise = 0.0 if s == 0 else 0.1
        x0 = pack(A_init + noise*np.random.randn(*A_init.shape),
                  b_init + noise*np.random.randn(*b_init.shape))
        res = minimize(loss, x0, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-6})
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    A_fit, b_fit = unpack(best_x)
    return A_fit, b_fit


# ── Fit b for target patients given fixed A ────────────────────────────────────
def fit_b_fixed_A(A_fixed, phi_target):
    """Fit patient-specific b from W1→W2 only, then predict W3."""
    n_tgt = len(phi_target)
    b_fit = np.zeros((n_tgt, N_G))
    for i in range(n_tgt):
        def loss(b):
            p2 = rep_ivp(phi_target[i, 0], b, A_fixed)
            return np.sum((p2 - phi_target[i, 1])**2) + LAM_B * np.sum(b**2)
        res = minimize(loss, b_full[i] if i < N_P else np.zeros(N_G),
                       method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
        b_fit[i] = res.x
    return b_fit


def eval_rmse(A, b_mat, phi_obs, label=''):
    obs_all, pred_all = [], []
    for i in range(len(phi_obs)):
        p2 = rep_ivp(phi_obs[i, 0], b_mat[i], A)
        p3 = rep_ivp(phi_obs[i, 1], b_mat[i], A)
        obs_all  += [phi_obs[i, 1], phi_obs[i, 2]]
        pred_all += [p2, p3]
    o = np.concatenate([x.flatten() for x in obs_all])
    p = np.concatenate([x.flatten() for x in pred_all])
    rmse = float(np.sqrt(np.mean((o - p)**2)))
    r    = float(np.corrcoef(o, p)[0, 1])
    if label:
        print(f'  {label}: RMSE={rmse:.4f}  r={r:.3f}')
    return rmse, r


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Full-cohort reference
# ══════════════════════════════════════════════════════════════════════════════
print('=== Full cohort (reference) ===')
rmse_full, r_full = eval_rmse(A_full, b_full, phi_all, 'full cohort')

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CT1-train → CT2-predict
# ══════════════════════════════════════════════════════════════════════════════
print('=== CT1-train A → predict CT2 ===')
phi_ct1 = phi_all[CT1_idx]
phi_ct2 = phi_all[CT2_idx]
b_ct1_init = b_full[CT1_idx]
b_ct2_init = b_full[CT2_idx]

print('  fitting A on CT1 patients...', flush=True)
A_ct1, b_ct1_fit = fit_A_on_subset(phi_ct1, A_full.copy(), b_ct1_init)
rmse_ct1_train, r_ct1_train = eval_rmse(A_ct1, b_ct1_fit, phi_ct1, 'CT1 train')

print('  fitting b for CT2 with A_CT1 fixed...', flush=True)
b_ct2_from_ct1 = fit_b_fixed_A(A_ct1, phi_ct2)
rmse_ct1_to_ct2, r_ct1_to_ct2 = eval_rmse(A_ct1, b_ct2_from_ct1, phi_ct2, 'CT1→CT2 predict')

# Within-CT2 self-fit (for comparison)
print('  fitting A on CT2 patients...', flush=True)
A_ct2, b_ct2_fit = fit_A_on_subset(phi_ct2, A_full.copy(), b_ct2_init)
rmse_ct2_train, r_ct2_train = eval_rmse(A_ct2, b_ct2_fit, phi_ct2, 'CT2 train')

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CT2-train → CT1-predict
# ══════════════════════════════════════════════════════════════════════════════
print('=== CT2-train A → predict CT1 ===')
print('  fitting b for CT1 with A_CT2 fixed...', flush=True)
b_ct1_from_ct2 = fit_b_fixed_A(A_ct2, phi_ct1)
rmse_ct2_to_ct1, r_ct2_to_ct1 = eval_rmse(A_ct2, b_ct1_from_ct2, phi_ct1, 'CT2→CT1 predict')

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Within-CT LOO (train on 2 CT1 patients, predict 3rd)
# ══════════════════════════════════════════════════════════════════════════════
print('=== Within-CT LOO ===')
loo_ct1, loo_ct2 = [], []

# CT1 LOO (3 patients — leave one out)
for hi in range(len(CT1_idx)):
    tr_idx = [j for j in range(len(CT1_idx)) if j != hi]
    if len(tr_idx) < 2:
        continue
    phi_tr = phi_ct1[tr_idx]
    b_tr   = b_ct1_init[tr_idx]
    A_lo, _ = fit_A_on_subset(phi_tr, A_full.copy(), b_tr, n_starts=2)
    b_hi = fit_b_fixed_A(A_lo, phi_ct1[[hi]])
    p2 = rep_ivp(phi_ct1[hi, 0], b_hi[0], A_lo)
    p3 = rep_ivp(phi_ct1[hi, 1], b_hi[0], A_lo)
    r = float(np.sqrt((np.sum((p2-phi_ct1[hi,1])**2)+np.sum((p3-phi_ct1[hi,2])**2))/(2*N_G)))
    loo_ct1.append(r)
    print(f'  CT1 LOO {CT1_pats[hi]}: RMSE={r:.4f}')

# CT2 LOO (5 patients)
for hi in range(len(CT2_idx)):
    tr_idx = [j for j in range(len(CT2_idx)) if j != hi]
    phi_tr = phi_ct2[tr_idx]
    b_tr   = b_ct2_init[tr_idx]
    A_lo, _ = fit_A_on_subset(phi_tr, A_full.copy(), b_tr, n_starts=2)
    b_hi = fit_b_fixed_A(A_lo, phi_ct2[[hi]])
    p2 = rep_ivp(phi_ct2[hi, 0], b_hi[0], A_lo)
    p3 = rep_ivp(phi_ct2[hi, 1], b_hi[0], A_lo)
    r = float(np.sqrt((np.sum((p2-phi_ct2[hi,1])**2)+np.sum((p3-phi_ct2[hi,2])**2))/(2*N_G)))
    loo_ct2.append(r)
    print(f'  CT2 LOO {CT2_pats[hi]}: RMSE={r:.4f}')

loo_ct1_mean = float(np.mean(loo_ct1)) if loo_ct1 else float('nan')
loo_ct2_mean = float(np.mean(loo_ct2)) if loo_ct2 else float('nan')
print(f'  CT1 within-LOO mean: {loo_ct1_mean:.4f}')
print(f'  CT2 within-LOO mean: {loo_ct2_mean:.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
results = {
    'full_cohort':    {'rmse': rmse_full,     'r': r_full,     'label': 'Full cohort'},
    'ct1_train':      {'rmse': rmse_ct1_train,'r': r_ct1_train,'label': 'CT1 in-sample'},
    'ct2_train':      {'rmse': rmse_ct2_train,'r': r_ct2_train,'label': 'CT2 in-sample'},
    'ct1_to_ct2':     {'rmse': rmse_ct1_to_ct2,'r': r_ct1_to_ct2,'label': 'CT1→CT2 predict'},
    'ct2_to_ct1':     {'rmse': rmse_ct2_to_ct1,'r': r_ct2_to_ct1,'label': 'CT2→CT1 predict'},
    'loo_ct1_mean':   loo_ct1_mean,
    'loo_ct2_mean':   loo_ct2_mean,
    'loo_ct1_per':    {p: v for p, v in zip(CT1_pats, loo_ct1)},
    'loo_ct2_per':    {p: v for p, v in zip(CT2_pats, loo_ct2)},
    'ct1_patients':   CT1_pats,
    'ct2_patients':   CT2_pats,
}
json.dump(results, open(CR_DIR / 'cross_ct_prediction.json', 'w'), indent=2)
print(f'\nSaved: {CR_DIR}/cross_ct_prediction.json')

print('\n' + '='*62)
print(f"{'Scenario':28} {'RMSE':>8} {'r':>7}")
print('-'*62)
for k in ('full_cohort','ct1_train','ct2_train','ct1_to_ct2','ct2_to_ct1'):
    d = results[k]
    print(f"  {d['label']:26} {d['rmse']:>8.4f} {d['r']:>7.3f}")
print(f"  {'CT1 within-LOO':26} {loo_ct1_mean:>8.4f}")
print(f"  {'CT2 within-LOO':26} {loo_ct2_mean:>8.4f}")
print('='*62)

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════
print('Generating figure...', flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: RMSE bar comparison
ax = axes[0]
scenarios = [
    ('Full cohort\n(reference)',  rmse_full,     '#4C72B0', 0.88),
    ('CT1 in-sample\n(n=5)',      rmse_ct1_train,'#55A868', 0.88),
    ('CT2 in-sample\n(n=5)',      rmse_ct2_train,'#C44E52', 0.88),
    ('CT1 within-LOO',            loo_ct1_mean,  '#55A868', 0.55),
    ('CT2 within-LOO',            loo_ct2_mean,  '#C44E52', 0.55),
    ('CT1→CT2\ncross-predict',    rmse_ct1_to_ct2,'#8172B2',0.88),
    ('CT2→CT1\ncross-predict',    rmse_ct2_to_ct1,'#CCB974',0.88),
]
xlbls = [s[0] for s in scenarios]
yvals = [s[1] for s in scenarios]
cols  = [s[2] for s in scenarios]
alph  = [s[3] for s in scenarios]
bars = ax.bar(range(len(scenarios)), yvals,
              color=cols, width=0.65)
for bar, a in zip(bars, alph):
    bar.set_alpha(a)
ax.axhline(rmse_full, color='#4C72B0', lw=1.2, ls='--', alpha=0.5)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(xlbls, fontsize=8.5)
ax.set_ylabel('RMSE', fontsize=11)
ax.set_title('CT cross-prediction vs in-sample', fontsize=11, fontweight='bold')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Right: A-matrix difference heatmap CT2−CT1
ax = axes[1]
dA = A_ct2 - A_ct1
import matplotlib.colors as mcolors
vmax = np.abs(dA).max()
SHORT = [GUILD_SHORT.get(g, g[:5]) for g in ref['guilds']]
im = ax.imshow(dA, cmap='PiYG', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(N_G)); ax.set_yticks(range(N_G))
ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=7.5)
ax.set_yticklabels(SHORT, fontsize=7.5)
ax.set_title(r'$\Delta A = A_{\rm CT2} - A_{\rm CT1}$', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ΔA')
for i in range(N_G):
    for j in range(N_G):
        v = dA[i, j]
        if abs(v) > 0.3 * vmax:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6, color='black')

plt.suptitle('CT1/CT2 cross-community-type prediction  (Dieckow 10-patient cohort)',
             fontsize=12, fontweight='bold')
plt.tight_layout()

for d in (CR_DIR, DOCS):
    fig.savefig(d / 'fig_cross_ct.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(d / 'fig_cross_ct.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig_cross_ct')
print('\nAll done.')
