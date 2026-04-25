#!/usr/bin/env python3
"""
loo_cv_dieckow.py — Leave-one-patient-out cross-validation for gLV and Hamilton+struct.

For each held-out patient p:
  1. Fit shared A on remaining 7 patients (L-BFGS-B, 3 starts)
  2. Fit patient-specific b for p with A fixed (L-BFGS-B)
  3. Predict W2 and W3 for p; compute RMSE

Outputs:
  results/dieckow_cr/loo_cv_glv.json
  results/dieckow_cr/loo_cv_hamilton.json

Usage:
  python loo_cv_dieckow.py [--model glv|hamilton|both]
"""

import argparse, json, os, sys, time
from pathlib import Path

# Force CPU to avoid GPU OOM during JAX compilation of LOO-CV loops
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G, default_A, pack, unpack
from load_structure_dieckow import load_structural_data, build_occupancy

PHI_NPY   = _here / 'results' / 'dieckow_otu' / 'phi_guild_excel_class.npy'
STRUCT_XL = _here / 'Datasets' / 'Abutment_Structure vs composition.xlsx'
OUT_GLV   = _here / 'results' / 'dieckow_cr' / 'loo_cv_glv.json'
OUT_HAM   = _here / 'results' / 'dieckow_cr' / 'loo_cv_hamilton.json'

PATIENTS_ALL = list('ABCDEFGHKL')
LAM = 1e-4
N_STARTS_OUTER = 3   # starts for A fit on train set
N_STARTS_INNER = 1   # starts for b fit on held-out (b only)


# ── gLV helpers ──────────────────────────────────────────────────────────────

def replicator_rhs(t, phi, b_eff, A):
    f = b_eff + A @ phi
    return phi * (f - phi @ f)


def integrate_step(phi0, b_eff, A):
    sol = solve_ivp(replicator_rhs, [0, 1.0], phi0, args=(b_eff, A),
                    method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


def integrate_step_struct(phi0, b_p, A, q_pw):
    b_eff = b_p * (0.5 + 0.5 * q_pw)
    return integrate_step(phi0, b_eff, A)


def rmse_glv(A, b_all, phi_obs, pl_arr):
    n_p = phi_obs.shape[0]
    sq, cnt = 0.0, 0
    for p in range(n_p):
        phi2 = integrate_step_struct(phi_obs[p, 0], b_all[p], A, pl_arr[p, 0])
        phi3 = integrate_step_struct(phi2, b_all[p], A, pl_arr[p, 1])
        sq  += np.sum((phi2 - phi_obs[p, 1])**2) + np.sum((phi3 - phi_obs[p, 2])**2)
        cnt += 2 * phi_obs.shape[2]
    return np.sqrt(sq / cnt)


def fit_glv_A_fixed_b(phi_train, pl_train, n_g, n_p_train, A_init, b_init, n_starts):
    """Fit shared A + patient b on training set."""
    bounds_A = [(None, 0.0) if i == j else (None, None)
                for i in range(n_g) for j in range(n_g)]
    bounds_b = [(None, None)] * (n_p_train * n_g)
    bounds = bounds_A + bounds_b
    rng = np.random.default_rng(42)
    best_rmse, best_theta = np.inf, None
    for s in range(n_starts):
        if s == 0:
            theta0 = np.concatenate([A_init.ravel(), b_init.ravel()])
        else:
            theta0 = np.concatenate([
                (A_init + rng.normal(0, 0.05, A_init.shape)).ravel(),
                (b_init + rng.normal(0, 0.05, b_init.shape)).ravel()
            ])
        def obj(th):
            A = th[:n_g*n_g].reshape(n_g, n_g)
            b = th[n_g*n_g:].reshape(n_p_train, n_g)
            return rmse_glv(A, b, phi_train, pl_train) + LAM * np.sum(A**2)
        res = minimize(obj, theta0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        A_opt = res.x[:n_g*n_g].reshape(n_g, n_g)
        b_opt = res.x[n_g*n_g:].reshape(n_p_train, n_g)
        r = rmse_glv(A_opt, b_opt, phi_train, pl_train)
        if r < best_rmse:
            best_rmse = r
            best_theta = res.x
    return best_theta[:n_g*n_g].reshape(n_g, n_g), best_theta[n_g*n_g:].reshape(n_p_train, n_g), best_rmse


def fit_b_held_out(phi_p, pl_p, A, n_g, b_init):
    """Fit b for single held-out patient with A fixed."""
    def obj(b):
        phi2 = integrate_step_struct(phi_p[0], b, A, pl_p[0])
        phi3 = integrate_step_struct(phi2, b, A, pl_p[1])
        sq = np.sum((phi2 - phi_p[1])**2) + np.sum((phi3 - phi_p[2])**2)
        return np.sqrt(sq / (2 * n_g))
    res = minimize(obj, b_init, method='L-BFGS-B',
                   options={'maxiter': 1000, 'ftol': 1e-12})
    return res.x


def predict_rmse_held_out(phi_p, pl_p, A, b_p):
    phi2 = integrate_step_struct(phi_p[0], b_p, A, pl_p[0])
    phi3 = integrate_step_struct(phi2, b_p, A, pl_p[1])
    n_g = len(b_p)
    sq = np.sum((phi2 - phi_p[1])**2) + np.sum((phi3 - phi_p[2])**2)
    return np.sqrt(sq / (2 * n_g)), phi2, phi3


# ── Hamilton helpers ──────────────────────────────────────────────────────────

def _hamilton_available():
    try:
        import jax  # noqa
        return True
    except ImportError:
        return False


def run_loo_glv(phi_all, pl_arr, patients, n_g):
    print('=== LOO-CV gLV+struct ===', flush=True)
    n_p = len(patients)
    A_global = default_A()[:n_g, :n_g]
    b_global = np.full((n_p, n_g), 0.1)
    # warm-start: full-data fit
    warm = _here / 'results' / 'dieckow_cr' / 'fit_guild_glv_struct.json'
    if warm.exists():
        d = json.load(open(warm))
        A_global = np.array(d['A'])[:n_g, :n_g]
        b_arr = np.array(d['b_all'])
        b_global = b_arr[:n_p, :n_g]
        print('  warm-start from fit_guild_glv_struct.json', flush=True)

    loo_rmses, results = [], []
    for p_held in range(n_p):
        t0 = time.time()
        train_idx = [i for i in range(n_p) if i != p_held]
        phi_train = phi_all[train_idx]
        pl_train = pl_arr[train_idx]
        b_train0 = b_global[train_idx]
        A_fit, _, train_rmse = fit_glv_A_fixed_b(
            phi_train, pl_train, n_g, len(train_idx),
            A_global.copy(), b_train0.copy(), N_STARTS_OUTER)
        b_p0 = b_global[p_held]
        b_p_fit = fit_b_held_out(phi_all[p_held], pl_arr[p_held], A_fit, n_g, b_p0)
        rmse_p, phi2_p, phi3_p = predict_rmse_held_out(
            phi_all[p_held], pl_arr[p_held], A_fit, b_p_fit)
        loo_rmses.append(float(rmse_p))
        results.append({'patient': patients[p_held], 'rmse': float(rmse_p),
                        'train_rmse': float(train_rmse)})
        print(f'  {patients[p_held]} held-out RMSE={rmse_p:.5f}  train={train_rmse:.5f}  ({time.time()-t0:.1f}s)', flush=True)

    loo_mean = float(np.mean(loo_rmses))
    print(f'\nLOO mean RMSE (gLV+struct): {loo_mean:.5f}', flush=True)
    json.dump({'loo_rmse_mean': loo_mean, 'per_patient': results,
               'model': 'gLV+struct LOO-CV'}, open(OUT_GLV, 'w'), indent=2)
    print(f'Saved: {OUT_GLV}', flush=True)
    return loo_mean


def run_loo_hamilton(phi_all, pl_arr, patients, n_g, occ_norm):
    print('=== LOO-CV Hamilton+struct ===', flush=True)
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        import optax
    except ImportError:
        print('JAX/optax not available — skipping Hamilton LOO-CV', flush=True)
        return None

    sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
    from hamilton_ode_jax_nsp import simulate_0d_nsp, count_params, theta_to_matrices

    n_sp = n_g
    n_A = n_sp * (n_sp + 1) // 2
    n_params_total = n_A + n_sp

    warm = _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json'
    if not warm.exists():
        print('  fit_guild_hamilton_masked.json not found — skipping', flush=True)
        return None
    d = json.load(open(warm))
    A_warm = np.array(d['A'])
    b_warm = np.array(d['b_all'])
    n_p = len(patients)

    def pack_upper(A):
        rows = []
        for j in range(n_sp):
            for i in range(j + 1):
                rows.append(A[i, j])
        return np.array(rows)

    def unpack_upper(v):
        A = np.zeros((n_sp, n_sp))
        idx = 0
        for j in range(n_sp):
            for i in range(j + 1):
                A[i, j] = A[j, i] = v[idx]; idx += 1
        return A

    N_STEPS = 2500
    LR = 3e-3
    EPOCHS = 800
    LAM_H = 1e-4

    def simulate_patient(A_upper_jax, b_p_jax, phi_init, psi_val, alpha_val):
        theta = jnp.concatenate([A_upper_jax, b_p_jax])
        phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=N_STEPS, dt=1e-4,
                                 phi_init=phi_init, psi_init=psi_val,
                                 c_const=25.0, alpha_const=alpha_val)
        eq = phibar[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    simulate_jit = jax.jit(simulate_patient)

    def loss_train(A_upper, b_mat, phi_train, occ_train, pl_train):
        total = 0.0
        n_tr = phi_train.shape[0]
        for p in range(n_tr):
            phi_w1 = jnp.array(phi_train[p, 0]) * occ_train[p, 0]
            psi_w1 = jnp.clip(pl_train[p, 0], 1e-4, 0.9999)
            alpha_w1 = 100.0 * (0.5 + 0.5 * pl_train[p, 0])
            pred_w2 = simulate_jit(A_upper, b_mat[p], phi_w1, psi_w1, alpha_w1)
            phi_w2_abs = pred_w2 * occ_train[p, 1]
            psi_w2 = jnp.clip(pl_train[p, 1], 1e-4, 0.9999)
            alpha_w2 = 100.0 * (0.5 + 0.5 * pl_train[p, 1])
            pred_w3 = simulate_jit(A_upper, b_mat[p], phi_w2_abs, psi_w2, alpha_w2)
            obs_w2 = jnp.array(phi_train[p, 1])
            obs_w3 = jnp.array(phi_train[p, 2])
            total = total + jnp.mean((pred_w2 - obs_w2)**2) + jnp.mean((pred_w3 - obs_w3)**2)
        reg = LAM_H * jnp.sum(A_upper**2)
        return total / n_tr + reg

    def fit_train(phi_tr, occ_tr, pl_tr, A_init_up, b_init_tr):
        A_up = jnp.array(A_init_up)
        b_mat = jnp.array(b_init_tr)
        opt = optax.adam(LR)
        params = (A_up, b_mat)
        opt_state = opt.init(params)
        grad_fn = jax.jit(jax.value_and_grad(
            lambda p: loss_train(p[0], p[1], phi_tr, occ_tr, pl_tr), argnums=0))
        for _ in range(EPOCHS):
            (val, grads) = grad_fn(params)
            updates, opt_state2 = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            opt_state = opt_state2
        return np.array(params[0]), np.array(params[1])

    loo_rmses, results = [], []
    for p_held in range(n_p):
        t0 = time.time()
        train_idx = [i for i in range(n_p) if i != p_held]
        phi_tr = phi_all[train_idx]
        occ_tr = occ_norm[train_idx]
        pl_tr = pl_arr[train_idx]
        A_up0 = pack_upper(A_warm[:n_g, :n_g])
        b_tr0 = b_warm[train_idx, :n_g] if b_warm.shape[0] >= n_p else np.full((len(train_idx), n_g), 0.1)

        A_up_fit, b_tr_fit = fit_train(phi_tr, occ_tr, pl_tr, A_up0, b_tr0)

        # fit b for held-out
        b_held0 = (b_warm[p_held, :n_g] if b_warm.shape[0] > p_held
                   else np.full(n_g, 0.1))
        b_held = jnp.array(b_held0)
        A_up_f = jnp.array(A_up_fit)
        opt2 = optax.adam(LR)
        params2 = b_held
        opt_state2 = opt2.init(params2)
        phi_ph = phi_all[p_held]
        occ_ph = occ_norm[p_held]
        pl_ph = pl_arr[p_held]
        def loss_b(b_p):
            phi_w1 = jnp.array(phi_ph[0]) * occ_ph[0]
            psi_w1 = jnp.clip(pl_ph[0], 1e-4, 0.9999)
            alpha_w1 = 100.0 * (0.5 + 0.5 * pl_ph[0])
            pred_w2 = simulate_jit(A_up_f, b_p, phi_w1, psi_w1, alpha_w1)
            phi_w2_abs = pred_w2 * occ_ph[1]
            psi_w2 = jnp.clip(pl_ph[1], 1e-4, 0.9999)
            alpha_w2 = 100.0 * (0.5 + 0.5 * pl_ph[1])
            pred_w3 = simulate_jit(A_up_f, b_p, phi_w2_abs, psi_w2, alpha_w2)
            return (jnp.mean((pred_w2 - jnp.array(phi_ph[1]))**2) +
                    jnp.mean((pred_w3 - jnp.array(phi_ph[2]))**2))
        grad_b = jax.jit(jax.value_and_grad(loss_b))
        for _ in range(400):
            val, g = grad_b(params2)
            upd, opt_state2 = opt2.update(g, opt_state2)
            params2 = optax.apply_updates(params2, upd)

        b_p_final = np.array(params2)
        A_final = unpack_upper(A_up_fit)
        phi_w1 = phi_ph[0] * occ_ph[0]
        pred_w2 = np.array(simulate_jit(
            jnp.array(A_up_fit), jnp.array(b_p_final),
            jnp.array(phi_w1), float(np.clip(pl_ph[0], 1e-4, 0.9999)),
            100.0 * (0.5 + 0.5 * float(pl_ph[0]))))
        phi_w2_abs = pred_w2 * occ_ph[1]
        pred_w3 = np.array(simulate_jit(
            jnp.array(A_up_fit), jnp.array(b_p_final),
            jnp.array(phi_w2_abs), float(np.clip(pl_ph[1], 1e-4, 0.9999)),
            100.0 * (0.5 + 0.5 * float(pl_ph[1]))))
        n_g_local = phi_ph.shape[1]
        rmse_p = float(np.sqrt(
            (np.sum((pred_w2 - phi_ph[1])**2) + np.sum((pred_w3 - phi_ph[2])**2))
            / (2 * n_g_local)))
        loo_rmses.append(rmse_p)
        results.append({'patient': patients[p_held], 'rmse': rmse_p})
        print(f'  {patients[p_held]} held-out RMSE={rmse_p:.5f}  ({time.time()-t0:.1f}s)', flush=True)

    loo_mean = float(np.mean(loo_rmses))
    print(f'\nLOO mean RMSE (Hamilton+struct): {loo_mean:.5f}', flush=True)
    json.dump({'loo_rmse_mean': loo_mean, 'per_patient': results,
               'model': 'Hamilton+struct LOO-CV'}, open(OUT_HAM, 'w'), indent=2)
    print(f'Saved: {OUT_HAM}', flush=True)
    return loo_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='both', choices=['glv', 'hamilton', 'both'])
    args = parser.parse_args()

    t0 = time.time()
    phi_all = np.load(PHI_NPY)
    n_p_raw, n_w, n_g = phi_all.shape
    guilds = GUILD_ORDER[:n_g]
    present = phi_all.sum(axis=2) > 1e-12
    keep = present[:, 0]
    phi_all = phi_all[keep]
    patients = [p for k, p in zip(keep.tolist(), PATIENTS_ALL) if k]
    n_p = len(patients)
    print(f'Loaded: {n_p} patients, {n_g} guilds', flush=True)

    struct = load_structural_data(STRUCT_XL)
    occ_raw, _ = build_occupancy(struct, normalize=True)
    pl_raw = struct.get('PerLive', {})
    pl_arr = np.ones((n_p, n_w))
    occ_norm = np.ones((n_p, n_w))
    for p_idx, pat in enumerate(patients):
        for w in range(n_w):
            pl_arr[p_idx, w] = pl_raw.get((pat, w + 1), 100.0) / 100.0
            occ_norm[p_idx, w] = occ_raw.get((pat, w + 1), 1.0)

    print(f'PerLive range: [{pl_arr.min():.3f}, {pl_arr.max():.3f}]', flush=True)

    if args.model in ('glv', 'both'):
        run_loo_glv(phi_all, pl_arr, patients, n_g)
    if args.model in ('hamilton', 'both'):
        run_loo_hamilton(phi_all, pl_arr, patients, n_g, occ_norm)

    print(f'\nTotal time: {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
