#!/usr/bin/env python3
"""
dieckow_cross_prediction.py

4つの Heine MAP の A 行列を Dieckow in-vivo データに適用し、
week2/3 予測 RMSE を Dieckow 自己 fit と比較する。

A: Heine条件MAP から抽出 (15 params)
b: Dieckow joint-TMCMC の患者別推定値を使用
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''   # CPU で十分
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
from hamilton_ode_jax_nsp import simulate_0d_nsp

RUNS_DIR = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs')
FITS_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
OBS_JSON = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_obs_matrix_5sp.json')
OUT_DIR  = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cross_prediction')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP = 5; N_A = 15; DT = 1e-4; C = 25.0; ALPHA = 100.0
N_STEPS = 2500

HEINE_CONDITIONS = {
    'CS': 'commensal_static',
    'CH': 'commensal_hobic',
    'DS': 'dysbiotic_static',
    'DH': 'dh_baseline',
}


def load_theta_map(run_name):
    d = json.load(open(RUNS_DIR / run_name / 'theta_MAP.json'))
    if isinstance(d, list):
        return np.array(d)
    for k in ['theta_MAP', 'theta_map', 'MAP']:
        if k in d:
            return np.array(d[k])
    return np.array(list(d.values())[0])


def load_dieckow_obs():
    raw = json.load(open(OBS_JSON))
    phi_list, valid = [], []
    for p in PATIENTS:
        obs_p = np.array(raw['obs'][p])   # (5, 3)
        phi_w = obs_p.T                    # (3, 5)
        if np.any(np.isnan(phi_w)) or np.any(phi_w.sum(axis=1) < 0.01):
            continue
        phi_w = np.clip(phi_w, 1e-6, 1.0)
        phi_w /= phi_w.sum(axis=1, keepdims=True)
        phi_list.append(phi_w)
        valid.append(p)
    return np.array(phi_list), valid   # (N, 3, 5)


def load_dieckow_b():
    """Dieckow TMCMC joint fit の患者別 b (N_p=1000, flat prior)."""
    d = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    theta = np.array(d['theta_map'])
    b = theta[N_A:].reshape(len(PATIENTS), N_SP)
    return b   # (10, 5)


def predict_weeks(theta_20, phi_obs_i):
    """theta (20-dim: 15 A + 5 b), phi_obs_i (3,5) → pred (2,5) for weeks 2,3."""
    theta_jax = jnp.array(theta_20)
    phi_ic = jnp.array(phi_obs_i[0])

    def run_week(phi_start):
        traj = simulate_0d_nsp(theta_jax, n_sp=N_SP, n_steps=N_STEPS,
                               dt=DT, phi_init=phi_start,
                               c_const=C, alpha_const=ALPHA)
        raw = traj[-1]
        return raw / jnp.maximum(raw.sum(), 1e-12)

    phi_w2 = jax.jit(run_week)(phi_ic)
    phi_w3 = jax.jit(run_week)(phi_w2)
    return np.array(phi_w2), np.array(phi_w3)


def compute_rmse(phi_obs, phi_preds):
    """phi_obs (N,3,5), phi_preds (N,2,5) → scalar RMSE."""
    obs_w23 = phi_obs[:, 1:, :]
    return float(np.sqrt(np.mean((phi_preds - obs_w23) ** 2)))


def main():
    print('Loading data...')
    phi_obs, valid_patients = load_dieckow_obs()
    N = len(valid_patients)
    print(f'  {N} patients: {valid_patients}')

    b_dieckow = load_dieckow_b()   # (10, 5) — indexed by PATIENTS order
    patient_idx = {p: PATIENTS.index(p) for p in valid_patients}

    # ── Dieckow self-prediction baseline (TMCMC joint, N_p=1000) ──────────────
    d_self = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    rmse_self = d_self['rmse']

    results = {'Dieckow self (TMCMC Np=1000)': rmse_self}
    A_matrices = {}

    # ── Cross-prediction: each Heine condition A + Dieckow b ──────────────────
    for label, run in HEINE_CONDITIONS.items():
        theta_heine = load_theta_map(run)
        A_heine = theta_heine[:N_A]   # 15 A params (shared)
        A_matrices[label] = A_heine

        preds = []
        for pi, p in enumerate(valid_patients):
            pidx = patient_idx[p]
            b_p = b_dieckow[pidx]                        # patient-specific b
            theta_combined = np.concatenate([A_heine, b_p])
            w2, w3 = predict_weeks(theta_combined, phi_obs[pi])
            preds.append([w2, w3])

        phi_preds = np.array(preds)   # (N, 2, 5)
        rmse = compute_rmse(phi_obs, phi_preds)
        results[f'Heine {label} A + Dieckow b'] = rmse
        print(f'  {label}: RMSE = {rmse:.4f}')

    # ── Also: Heine A + Heine b (fixed, same for all patients) ──────────────
    for label, run in HEINE_CONDITIONS.items():
        theta_heine = load_theta_map(run)
        preds = []
        for pi in range(N):
            w2, w3 = predict_weeks(theta_heine, phi_obs[pi])
            preds.append([w2, w3])
        phi_preds = np.array(preds)
        rmse = compute_rmse(phi_obs, phi_preds)
        results[f'Heine {label} A+b (fixed)'] = rmse
        print(f'  {label} fixed: RMSE = {rmse:.4f}')

    print(f'\nDieckow self (TMCMC): {rmse_self:.4f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: all RMSEs
    ax = axes[0]
    keys   = list(results.keys())
    values = list(results.values())
    colors = ['steelblue'] + ['coral']*4 + ['salmon']*4
    bars = ax.barh(keys, values, color=colors[:len(keys)], alpha=0.85)
    ax.axvline(rmse_self, color='steelblue', lw=1.5, ls='--', alpha=0.7)
    ax.set_xlabel('RMSE (weeks 2+3)')
    ax.set_title('Cross-prediction: Heine A on Dieckow in-vivo')
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # A matrix correlation heatmap: Heine vs Dieckow
    d_self_full = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    A_dieckow = np.array(d_self_full['theta_map'])[:N_A]
    conds = list(HEINE_CONDITIONS.keys())
    corr_mat = np.zeros((len(conds)+1, len(conds)+1))
    labels_all = conds + ['Dieckow']
    all_A = [A_matrices[c] for c in conds] + [A_dieckow]
    for i in range(len(all_A)):
        for j in range(len(all_A)):
            corr_mat[i, j] = float(np.corrcoef(all_A[i], all_A[j])[0, 1])

    ax2 = axes[1]
    im = ax2.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax2.set_xticks(range(len(labels_all))); ax2.set_xticklabels(labels_all)
    ax2.set_yticks(range(len(labels_all))); ax2.set_yticklabels(labels_all)
    ax2.set_title('A matrix correlation (Heine × Dieckow MAP)')
    for i in range(len(labels_all)):
        for j in range(len(labels_all)):
            ax2.text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                     fontsize=9, color='black' if abs(corr_mat[i,j]) < 0.7 else 'white')
    plt.colorbar(im, ax=ax2)

    plt.suptitle('Dieckow in-vivo cross-prediction analysis', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'cross_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {OUT_DIR}/cross_prediction.png')

    # Save summary JSON
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump({'results': results,
                   'A_corr_labels': labels_all,
                   'A_corr_matrix': corr_mat.tolist()}, f, indent=2)
    print(f'Saved: {OUT_DIR}/summary.json')


if __name__ == '__main__':
    main()
