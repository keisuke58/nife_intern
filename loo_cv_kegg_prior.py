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

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

PHI_NPY   = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
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
            if g is None or g not in gi:
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



# ── Hamilton full fit with sign prior (JAX autodiff + Adam) ──────────────

def run_hamilton_kegg(phi_all, net_flow, nsteps=500, n_epochs=5000, lr=3e-3):
    """Full-cohort Hamilton ODE fit with KEGG/HMDB sign prior.
    Uses jax.grad through simulate_0d_nsp (lax.scan-based) + Adam.
    nsteps=500: reduced from 2500 to avoid XLA compilation blowup on CPU.
    """
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
    except ImportError:
        print('JAX not available — skipping Hamilton fit', flush=True)
        return

    sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
    from hamilton_ode_jax_nsp import simulate_0d_nsp

    n_p  = phi_all.shape[0]
    n_sp = N_G
    n_A  = n_sp * (n_sp + 1) // 2
    LAM_H = 1e-4

    # warm start
    warm = _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json'
    if warm.exists():
        d        = json.load(open(warm))
        A_full   = np.array(d['A'])[:n_sp, :n_sp]
        theta_A0 = np.array([A_full[i, j] for j in range(n_sp) for i in range(j + 1)])
        b_raw    = np.array(d['b_all'])
        if b_raw.ndim == 2 and b_raw.shape[0] >= n_p and b_raw.shape[1] >= n_sp:
            b_all0 = b_raw[:n_p, :n_sp].copy()
        else:
            b_all0 = np.full((n_p, n_sp), 0.1)
        print(f'  Warm-start from {warm.name}', flush=True)
    else:
        diag_idx0        = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])
        theta_A0         = np.zeros(n_A)
        theta_A0[diag_idx0] = -0.1
        b_all0           = np.full((n_p, n_sp), 0.1)

    theta_A      = jnp.array(theta_A0)
    b_all        = jnp.array(b_all0)
    phi_obs      = jnp.array(phi_all)
    net_sym      = jnp.array((net_flow + net_flow.T) / 2.0)
    diag_idx_jnp = jnp.array([j * (j + 1) // 2 + j for j in range(n_sp)])

    def _run_sim(theta, phi_init):
        phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=nsteps, dt=1e-4,
                                  phi_init=phi_init, c_const=25.0, alpha_const=100.0)
        eq = phibar[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    def _pred_two_weeks(theta_A, b_p, phi0):
        """Predict weeks 2 and 3 for one patient."""
        theta = jnp.concatenate([theta_A, b_p])
        phi2  = _run_sim(theta, phi0)
        phi3  = _run_sim(theta, phi2)
        return phi2, phi3

    # vmap over patients (axis 0): theta_A is shared (None), b and phi0 are per-patient (0)
    _pred_all = jax.vmap(_pred_two_weeks, in_axes=(None, 0, 0))

    @jax.jit
    def pred_week(theta_A, b, phi0):
        """Single-patient single-week prediction (for evaluation)."""
        return _run_sim(jnp.concatenate([theta_A, b]), phi0)

    def _unpack_A(v):
        A = jnp.zeros((n_sp, n_sp))
        idx = 0
        for j in range(n_sp):
            for i in range(j + 1):
                A = A.at[i, j].set(v[idx]); A = A.at[j, i].set(v[idx]); idx += 1
        return A

    @jax.jit
    def loss_fn(theta_A, b_all):
        # vmap replaces the Python for-loop — single scan graph instead of n_p copies
        phi2_all, phi3_all = _pred_all(theta_A, b_all, phi_obs[:, 0])
        sq = jnp.sum((phi2_all - phi_obs[:, 1]) ** 2) + jnp.sum((phi3_all - phi_obs[:, 2]) ** 2)
        rmse     = jnp.sqrt(sq / (n_p * 2 * n_sp))
        A        = _unpack_A(theta_A)
        sp_mat   = jnp.sign(net_sym)
        mask     = (sp_mat != 0) & (~jnp.eye(n_sp, dtype=bool))
        mismatch = jnp.maximum(0.0, -sp_mat * A)
        pen      = jnp.where(mask, jnp.abs(net_sym) * mismatch ** 2 / (2 * SIGMA ** 2), 0.0).sum()
        return rmse + pen + LAM_H * jnp.sum(theta_A ** 2)

    # Use scipy L-BFGS-B with JAX analytical gradients — far fewer iterations than Adam
    from scipy.optimize import minimize as scipy_minimize

    n_flat = n_A + n_p * n_sp

    diag_idx_np = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])

    @jax.jit
    def _loss_and_grad_flat(x_flat):
        ta = x_flat[:n_A]
        ba = x_flat[n_A:].reshape(n_p, n_sp)
        loss, (gA, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(ta, ba)
        return loss, jnp.concatenate([gA, gb.ravel()])

    print(f'  Compiling Hamilton JAX (n_sp={n_sp}, nsteps={nsteps})...', flush=True)
    x0 = np.concatenate([np.array(theta_A), np.array(b_all).ravel()])
    init_loss = float(loss_fn(theta_A, b_all))
    print(f'  Initial loss={init_loss:.5f}', flush=True)

    call_count = [0]
    t0 = time.time()

    def fg(x):
        call_count[0] += 1
        xj = jnp.array(x)
        loss, grad = _loss_and_grad_flat(xj)
        if call_count[0] % 50 == 1:
            print(f'  iter {call_count[0]:4d}  loss={float(loss):.5f}  ({time.time()-t0:.1f}s)',
                  flush=True)
        return float(loss), np.array(grad, dtype=np.float64)

    # Bounds: diagonal of A (theta_A[diag_idx]) ≤ 0; all others free
    bounds = [(None, None)] * n_flat
    for k in diag_idx_np:
        bounds[k] = (None, 0.0)

    res = scipy_minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': n_epochs, 'ftol': 1e-12, 'gtol': 1e-7})
    print(f'  L-BFGS-B: {res.message}  iters={call_count[0]}  loss={res.fun:.5f}', flush=True)

    x_opt    = res.x
    theta_A  = jnp.array(x_opt[:n_A])
    b_all    = jnp.array(x_opt[n_A:].reshape(n_p, n_sp))
    A_opt = np.array(_unpack_A(theta_A))
    b_opt = np.array(b_all)

    sq, cnt, all_obs, all_pred = 0.0, 0, [], []
    for p in range(n_p):
        phi2 = np.array(pred_week(theta_A, b_all[p], phi_obs[p, 0]))
        phi3 = np.array(pred_week(theta_A, b_all[p], jnp.array(phi2)))
        sq  += float(np.sum((phi2 - phi_all[p, 1]) ** 2) + np.sum((phi3 - phi_all[p, 2]) ** 2))
        cnt += 2 * n_sp
        all_obs  += phi_all[p, 1].tolist() + phi_all[p, 2].tolist()
        all_pred += phi2.tolist() + phi3.tolist()
    rmse = float(np.sqrt(sq / cnt))
    r    = float(np.corrcoef(all_obs, all_pred)[0, 1])

    sp_mat  = np.sign((net_flow + net_flow.T) / 2.0)
    mask    = (sp_mat != 0) & (~np.eye(n_sp, dtype=bool))
    n_agree = int(((np.sign(A_opt) == sp_mat) & mask).sum())
    n_tot   = int(mask.sum())
    print(f'  Hamilton KEGG-prior: RMSE={rmse:.4f}, r={r:.4f}  ({time.time()-t0:.1f}s)', flush=True)
    print(f'  Sign agreement: {n_agree}/{n_tot} ({100*n_agree/n_tot:.0f}%)', flush=True)

    out = {
        'rmse': rmse, 'r': r,
        'sign_agreement': f'{n_agree}/{n_tot}',
        'patients': PATIENTS, 'guilds': GUILD_ORDER,
        'A': A_opt.tolist(), 'b_all': b_opt.tolist(),
        'model': (f'Hamilton JAX/Adam KEGG/HMDB-prior ({n_p} pat, {N_G} guild, '
                  f'sigma={SIGMA}, nsteps={nsteps}, epochs={n_epochs}, lr={lr})'),
    }
    json.dump(out, open(OUT_HAM, 'w'), indent=2)
    print(f'Saved: {OUT_HAM}', flush=True)


# ── Main ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='both', choices=['glv', 'hamilton', 'both'])
    parser.add_argument('--nsteps', type=int,   default=int(os.environ.get('NSTEPS',  2500)))
    parser.add_argument('--epochs', type=int,   default=int(os.environ.get('MAXITER', 5000)))
    parser.add_argument('--lr',     type=float, default=3e-3)
    args = parser.parse_args()

    t0 = time.time()
    print('Building KEGG/HMDB-weighted metabolite flow...', flush=True)
    net_flow = build_net_flow()
    print(f'  Non-zero pairs: {(net_flow != 0).sum() - N_G}, '
          f'max |flow|={np.abs(net_flow).max():.1f}', flush=True)

    phi_all = np.load(PHI_NPY)
    phi_sub = np.stack([phi_all[k] for k in range(phi_all.shape[0])
                        if phi_all[k].sum() > 1e-9])[:len(PATIENTS)]

    if KEGG_FIT.exists():
        d      = json.load(open(KEGG_FIT))
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
        print(f'\n=== Hamilton ODE KEGG-prior (JAX/Adam) '
              f'nsteps={args.nsteps} epochs={args.epochs} lr={args.lr} ===', flush=True)
        run_hamilton_kegg(phi_sub, net_flow,
                          nsteps=args.nsteps, n_epochs=args.epochs, lr=args.lr)

    print(f'\nTotal: {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
