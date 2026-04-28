#!/usr/bin/env python3
"""
loo_cv_kegg_prior.py — LOO-CV for gLV with KEGG/HMDB-weighted sign prior.

Also runs full-cohort Hamilton ODE fit with the same sign prior.

Outputs:
  results/dieckow_cr/loo_cv_glv_kegg_prior.json
  results/dieckow_cr/fit_glv_hamilton_kegg_prior.json (Hamilton full fit)

Usage:
  python loo_cv_kegg_prior.py [--model glv|hamilton|both]
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

PHI_NPY   = _here / 'results' / 'dieckow_otu' / 'phi_guild_excel_class.npy'
SUPPFILE  = _here / 'Szafranski_Published_Work' / 'Szafranski_Published_Work' / \
            'public_data' / 'Dieckow' / \
            'Supplementary_File_1_microbe_metabolite_enzyme_interactions.tsv'
KEGG_FIT  = _here / 'results' / 'dieckow_cr' / 'fit_glv_8pat_kegg_prior.json'
OUT_LOO   = _here / 'results' / 'dieckow_cr' / 'loo_cv_glv_kegg_prior.json'
OUT_HAM   = _here / 'results' / 'dieckow_cr' / 'fit_glv_hamilton_kegg_prior.json'

PATIENTS  = list('ABCEFGHK')
LAM       = 1e-4
SIGMA     = 0.15
N_STARTS  = 3

GENUS_GUILD = {
    'Actinomyces': 'Actinobacteria', 'Bifidobacterium': 'Actinobacteria',
    'Rothia': 'Actinobacteria', 'Schaalia': 'Actinobacteria',
    'Streptococcus': 'Bacilli', 'Gemella': 'Bacilli', 'Granulicatella': 'Bacilli',
    'Abiotrophia': 'Bacilli', 'Lactiplantibacillus': 'Bacilli',
    'Prevotella': 'Bacteroidia', 'Porphyromonas': 'Bacteroidia',
    'Tannerella': 'Bacteroidia', 'Alloprevotella': 'Bacteroidia',
    'Capnocytophaga': 'Flavobacteriia',
    'Neisseria': 'Betaproteobacteria', 'Eikenella': 'Betaproteobacteria',
    'Aggregatibacter': 'Betaproteobacteria',
    'Fusobacterium': 'Fusobacteriia', 'Leptotrichia': 'Fusobacteriia',
    'Haemophilus': 'Gammaproteobacteria',
    'Veillonella': 'Negativicutes', 'Selenomonas': 'Negativicutes',
    'Megasphaera': 'Negativicutes', 'Dialister': 'Negativicutes',
    'Parvimonas': 'Clostridia', 'Mogibacterium': 'Clostridia',
    'Peptostreptococcus': 'Clostridia', 'Catonella': 'Clostridia',
    'Atopobium': 'Coriobacteriia', 'Olsenella': 'Coriobacteriia',
}


# ── Build KEGG/HMDB-weighted metabolite-flow matrix ──────────────────────────

def build_net_flow():
    gi = {g: i for i, g in enumerate(GUILD_ORDER)}
    df = pd.read_csv(SUPPFILE, sep='\t')

    def met_weight(row):
        kegg = str(row.get('KEGG', ''))
        hmdb = str(row.get('HMDB_ID', ''))
        if kegg not in ('n/a', '', 'nan', 'NaN'):
            return 2.0
        if 'HMDB' in hmdb:
            return 2.0
        return 1.0

    pos = np.zeros((N_G, N_G))
    neg = np.zeros((N_G, N_G))
    for met in df['OBJECT'].unique():
        mdf = df[df['OBJECT'] == met]
        w = float(mdf.apply(met_weight, axis=1).max())
        prod, cons, inhib = set(), set(), set()
        for _, row in mdf.iterrows():
            g = GENUS_GUILD.get(str(row['TAXON']).split()[0])
            if g is None:
                continue
            if row['RELATIONSHIP'] == 'PRODUCES':
                prod.add(g)
            elif row['RELATIONSHIP'] == 'USES':
                cons.add(g)
            elif row['RELATIONSHIP'] == 'IS_INHIBITED_BY':
                inhib.add(g)
        for src in prod:
            for tgt in cons:
                if src != tgt:
                    pos[gi[tgt], gi[src]] += w
            for tgt in inhib:
                if src != tgt:
                    neg[gi[tgt], gi[src]] += w
    return pos - neg


# ── gLV helpers ───────────────────────────────────────────────────────────────

def replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)


def integrate_step(phi0, b, A):
    sol = solve_ivp(replicator_rhs, [0, 1.0], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


def rmse_glv(A, b_all, phi_obs):
    n_p = phi_obs.shape[0]
    sq, cnt = 0.0, 0
    for p in range(n_p):
        phi2 = integrate_step(phi_obs[p, 0], b_all[p], A)
        phi3 = integrate_step(phi2, b_all[p], A)
        sq  += np.sum((phi2 - phi_obs[p, 1])**2) + np.sum((phi3 - phi_obs[p, 2])**2)
        cnt += 2 * N_G
    return np.sqrt(sq / cnt)


def sign_penalty(A, net_flow):
    sp = np.sign(net_flow)
    mask = (sp != 0) & (~np.eye(N_G, dtype=bool))
    pen = 0.0
    for i in range(N_G):
        for j in range(N_G):
            if mask[i, j]:
                w = abs(net_flow[i, j])
                v = max(0.0, -sp[i, j] * A[i, j])
                pen += w * v * v / (2 * SIGMA * SIGMA)
    return pen


def make_obj(phi_obs, net_flow):
    n_p = phi_obs.shape[0]

    def obj(theta):
        A = theta[:N_G * N_G].reshape(N_G, N_G)
        b = theta[N_G * N_G:].reshape(n_p, N_G)
        return rmse_glv(A, b, phi_obs) + sign_penalty(A, net_flow) + LAM * np.sum(A ** 2)

    return obj


def bounds_diag_neg(n_p):
    ba = [(None, 0.0) if i == j else (None, None)
          for i in range(N_G) for j in range(N_G)]
    bb = [(None, None)] * (n_p * N_G)
    return ba + bb


def fit_full(phi_obs, net_flow, A_init, b_init, n_starts=N_STARTS):
    n_p = phi_obs.shape[0]
    obj = make_obj(phi_obs, net_flow)
    bnds = bounds_diag_neg(n_p)
    x0 = np.concatenate([A_init.ravel(), b_init.ravel()])
    best_x, best_f = x0, obj(x0)
    rng = np.random.default_rng(0)
    for s in range(n_starts):
        noise = rng.normal(0, 0.05, x0.shape) if s > 0 else np.zeros_like(x0)
        res = minimize(obj, x0 + noise, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
    A = best_x[:N_G * N_G].reshape(N_G, N_G)
    b = best_x[N_G * N_G:].reshape(n_p, N_G)
    return A, b


def fit_b_only(phi_p, A, b0):
    def obj(b):
        phi2 = integrate_step(phi_p[0], b, A)
        phi3 = integrate_step(phi2, b, A)
        sq = np.sum((phi2 - phi_p[1]) ** 2) + np.sum((phi3 - phi_p[2]) ** 2)
        return np.sqrt(sq / (2 * N_G))
    res = minimize(obj, b0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    return res.x


def held_out_rmse(phi_p, A, b):
    phi2 = integrate_step(phi_p[0], b, A)
    phi3 = integrate_step(phi2, b, A)
    sq = np.sum((phi2 - phi_p[1]) ** 2) + np.sum((phi3 - phi_p[2]) ** 2)
    return float(np.sqrt(sq / (2 * N_G)))


# ── LOO-CV ────────────────────────────────────────────────────────────────────

def run_loo(phi_all, net_flow, A_warm, b_warm):
    n_p = phi_all.shape[0]
    loo_rmses, results = [], []

    for hold in range(n_p):
        t0 = time.time()
        tr = [i for i in range(n_p) if i != hold]
        phi_tr = phi_all[tr]
        b_tr0 = b_warm[tr]
        A_fit, b_fit = fit_full(phi_tr, net_flow, A_warm.copy(), b_tr0.copy())
        tr_rmse = rmse_glv(A_fit, b_fit, phi_tr)

        b_p = fit_b_only(phi_all[hold], A_fit, b_warm[hold])
        rmse_p = held_out_rmse(phi_all[hold], A_fit, b_p)
        loo_rmses.append(rmse_p)
        results.append({'patient': PATIENTS[hold], 'rmse': float(rmse_p),
                        'train_rmse': float(tr_rmse)})
        print(f'  {PATIENTS[hold]}: LOO={rmse_p:.5f}  train={tr_rmse:.5f}'
              f'  ({time.time()-t0:.1f}s)', flush=True)

    mean = float(np.mean(loo_rmses))
    print(f'\nKEGG-prior gLV LOO mean (8 pat, {N_G} guild): {mean:.5f}', flush=True)
    out = {'loo_rmse_mean': mean, 'per_patient': results,
           'model': f'gLV KEGG/HMDB-prior LOO-CV (8 patients, {N_G} guilds, sigma={SIGMA})'}
    json.dump(out, open(OUT_LOO, 'w'), indent=2)
    print(f'Saved: {OUT_LOO}', flush=True)
    return mean


# ── Hamilton full fit with sign prior ─────────────────────────────────────────

def run_hamilton_kegg(phi_all, net_flow):
    """Full-cohort Hamilton ODE fit with KEGG/HMDB sign prior (symmetric A)."""
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
    except ImportError:
        print('JAX not available — skipping Hamilton fit', flush=True)
        return

    sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
    from hamilton_ode_jax_nsp import simulate_0d_nsp

    n_p = phi_all.shape[0]
    N_STEPS = 2500
    LAM_H = 1e-4

    warm = _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json'
    if not warm.exists():
        print('  fit_guild_hamilton_masked.json not found — skipping Hamilton', flush=True)
        return
    d = json.load(open(warm))
    b_warm_raw = np.array(d['b_all'])
    if b_warm_raw.ndim != 2:
        b_warm_raw = np.asarray(b_warm_raw).reshape(n_p, -1)
    if b_warm_raw.shape[0] < n_p:
        b_warm_raw = np.vstack([b_warm_raw, np.full((n_p - b_warm_raw.shape[0], b_warm_raw.shape[1]), 0.1)])
    # n_sp from actual Hamilton b dimension (may differ from N_G if 'Other' guild excluded)
    n_sp = b_warm_raw.shape[1]
    n_A = n_sp * (n_sp + 1) // 2
    A_warm = np.array(d['A'])[:n_sp, :n_sp]
    b_warm = b_warm_raw[:n_p, :n_sp]

    def pack_upper(A):
        return np.array([A[i, j] for j in range(n_sp) for i in range(j + 1)])

    def unpack_upper(v):
        A = np.zeros((n_sp, n_sp))
        idx = 0
        for j in range(n_sp):
            for i in range(j + 1):
                A[i, j] = A[j, i] = v[idx]; idx += 1
        return A

    @jax.jit
    def _sim(theta, phi0, psi, alpha):
        pb = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=N_STEPS, dt=1e-4,
                             phi_init=phi0, psi_init=psi, c_const=25.0, alpha_const=alpha)
        eq = pb[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    print('  Pre-compiling Hamilton forward pass...', flush=True)
    _dummy_theta = jnp.ones(n_A + n_sp) * 0.1
    _ = np.array(_sim(_dummy_theta, jnp.ones(n_sp) / n_sp,
                      jnp.array(0.5), jnp.array(50.0)))
    print('  Done.', flush=True)

    def sim_np(A_up, b_p, phi0, psi, alpha):
        import jax.numpy as jnp
        theta = np.concatenate([A_up, b_p])
        return np.array(_sim(jnp.array(theta), jnp.array(phi0),
                             jnp.array(float(psi)), jnp.array(float(alpha))))

    # Slice net_flow to n_sp (Hamilton may use fewer guilds than gLV)
    net_flow_sp = net_flow[:n_sp, :n_sp]
    # Build symmetric sign prior from net_flow (average both directions)
    net_sym = (net_flow_sp + net_flow_sp.T) / 2.0

    def sign_pen_sym(A_up):
        A = unpack_upper(A_up)
        sp_mat = np.sign(net_sym)
        mask = (sp_mat != 0) & (~np.eye(n_sp, dtype=bool))
        pen = 0.0
        for i in range(n_sp):
            for j in range(i + 1, n_sp):
                if mask[i, j] or mask[j, i]:
                    s_sign = sp_mat[i, j] if abs(net_sym[i, j]) >= abs(net_sym[j, i]) else sp_mat[j, i]
                    w = max(abs(net_sym[i, j]), abs(net_sym[j, i]))
                    v = max(0.0, -s_sign * A[i, j])
                    pen += w * v * v / (2 * SIGMA * SIGMA)
        return pen

    # Slice phi_all to n_sp (renormalise in case the dropped guild had mass)
    phi_sp = phi_all[:, :, :n_sp]
    s = phi_sp.sum(axis=2, keepdims=True)
    phi_sp = np.where(s > 0, phi_sp / s, np.ones_like(phi_sp) / n_sp)

    def obj_hamilton(theta_flat):
        A_up = theta_flat[:n_A]
        b_mat = theta_flat[n_A:].reshape(n_p, n_sp)
        total = 0.0
        for p in range(n_p):
            phi2 = sim_np(A_up, b_mat[p], phi_sp[p, 0], 0.5, 100.0)
            phi3 = sim_np(A_up, b_mat[p], phi2, 0.5, 100.0)
            total += np.mean((phi2 - phi_sp[p, 1]) ** 2)
            total += np.mean((phi3 - phi_sp[p, 2]) ** 2)
        rmse = np.sqrt(total / n_p)
        return rmse + sign_pen_sym(theta_flat[:n_A]) + LAM_H * np.sum(theta_flat[:n_A] ** 2)

    x0 = np.concatenate([pack_upper(A_warm), b_warm.ravel()])
    print(f'  Initial Hamilton obj: {obj_hamilton(x0):.5f}', flush=True)
    t0 = time.time()
    res = minimize(obj_hamilton, x0, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
    A_opt = unpack_upper(res.x[:n_A])
    b_opt = res.x[n_A:].reshape(n_p, n_sp)

    # Compute pure RMSE
    sq, cnt = 0.0, 0
    all_obs, all_pred = [], []
    for p in range(n_p):
        phi2 = sim_np(res.x[:n_A], b_opt[p], phi_sp[p, 0], 0.5, 100.0)
        phi3 = sim_np(res.x[:n_A], b_opt[p], phi2, 0.5, 100.0)
        sq += np.sum((phi2 - phi_sp[p, 1]) ** 2) + np.sum((phi3 - phi_sp[p, 2]) ** 2)
        cnt += 2 * n_sp
        all_obs.extend(phi_sp[p, 1].tolist() + phi_sp[p, 2].tolist())
        all_pred.extend(phi2.tolist() + phi3.tolist())
    rmse = float(np.sqrt(sq / cnt))
    r = float(np.corrcoef(all_obs, all_pred)[0, 1])

    # Sign agreement
    sp_mat = np.sign(net_sym)
    mask = (sp_mat != 0) & (~np.eye(n_sp, dtype=bool))
    n_agree = int(((np.sign(A_opt) == sp_mat) & mask).sum())
    n_tot = int(mask.sum())

    print(f'  Hamilton KEGG-prior: RMSE={rmse:.4f}, r={r:.4f}  ({time.time()-t0:.1f}s)', flush=True)
    print(f'  Sign agreement: {n_agree}/{n_tot} ({100*n_agree/n_tot:.0f}%)', flush=True)

    out = {
        'rmse': rmse, 'r': r,
        'sign_agreement': f'{n_agree}/{n_tot}',
        'patients': PATIENTS,
        'guilds': GUILD_ORDER[:n_sp],
        'A': A_opt.tolist(),
        'b_all': b_opt.tolist(),
        'model': f'Hamilton gLV KEGG/HMDB-prior (8 pat, {N_G} guild, sigma={SIGMA})',
    }
    json.dump(out, open(OUT_HAM, 'w'), indent=2)
    print(f'Saved: {OUT_HAM}', flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='both', choices=['glv', 'hamilton', 'both'])
    args = parser.parse_args()

    t0 = time.time()
    print('Building KEGG/HMDB-weighted metabolite flow...', flush=True)
    net_flow = build_net_flow()
    print(f'  Non-zero pairs: {(net_flow != 0).sum() - N_G}, '
          f'max |flow|={np.abs(net_flow).max():.1f}', flush=True)

    phi_all = np.load(PHI_NPY)
    keep = [PATIENTS.index(p) for p in PATIENTS
            if p in list('ABCEFGHK')]
    phi_sub = np.stack([phi_all[k] for k in range(phi_all.shape[0])
                        if phi_all[k].sum() > 1e-9])[:len(PATIENTS)]

    # Use warm-start from full KEGG-prior fit
    if KEGG_FIT.exists():
        d = json.load(open(KEGG_FIT))
        A_warm = np.array(d['A'])
        b_warm = np.array(d['b_all'])
        print(f'Warm-start: {KEGG_FIT.name}', flush=True)
    else:
        A_warm = np.zeros((N_G, N_G))
        np.fill_diagonal(A_warm, -0.1)
        b_warm = np.full((len(PATIENTS), N_G), 0.1)

    if args.model in ('glv', 'both'):
        print('\n=== LOO-CV gLV KEGG-prior ===', flush=True)
        run_loo(phi_sub, net_flow, A_warm, b_warm)

    if args.model in ('hamilton', 'both'):
        print('\n=== Hamilton ODE KEGG-prior (full cohort) ===', flush=True)
        run_hamilton_kegg(phi_sub, net_flow)

    print(f'\nTotal: {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
