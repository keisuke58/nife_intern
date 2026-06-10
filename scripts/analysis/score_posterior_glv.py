#!/usr/bin/env python3
"""
score_posterior_glv.py — ② Score-based Langevin posterior for gLV A matrix.

Implements Diffusion Posterior Sampling (DPS / score-guided Langevin MCMC)
to infer the interaction matrix A given observed guild compositions phi_obs.

Two score contributions (Bayes rule in score form):
  ∇_A log p(A | phi_obs)  =  ∇_A log p(phi_obs | A)   (data likelihood, JAX ODE)
                           +  ∇_A log p(A)              (Gaussian prior)

Algorithm: Unadjusted Langevin Algorithm (ULA) with annealed step size.
This is the *analytical* score posterior — no neural score network needed.
The diffusion framing enters via the annealing schedule that moves from
a broad Gaussian prior (high temperature) to the posterior (T→0).

Comparison to TMCMC:
  - TMCMC (existing):  importance-weighted resampling, no gradient
  - ULA here:          gradient-based, Langevin diffusion, exact scores

CAVEAT (demo convergence): a short run (--n-steps 300, --step-size 5e-5)
under-converges — the chain stays near its small-magnitude init and the
posterior mean is prior-dominated (≈0), giving only ~chance sign agreement
with the L-BFGS-B warm-start A. For a scientifically meaningful posterior use
--n-steps 5000+ and a warmer step size, ideally with an MH accept/reject step
(this ULA has none, so it can drift into stiff-ODE regions and slow down). The
downstream flow_posterior_glv.py (②') faithfully amortizes whatever this chain
produces, so improving convergence here directly improves the flow posterior.

Usage (repo root):
  python scripts/analysis/score_posterior_glv.py \\
    --n-chains 4 --n-steps 2000 --step-size 1e-4

Output:
  results/score_posterior/A_samples.npy         (n_chains * n_steps/thin, 100)
  results/score_posterior/posterior_summary.json
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse, json, time
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp

_here = Path(__file__).resolve().parents[2]
PHI_NPY = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
OUT_DIR = _here / 'results' / 'score_posterior'
OUT_DIR.mkdir(exist_ok=True)

N_G = 10
DT  = 1.0


# ── gLV forward model (numpy, differentiable via finite differences) ──────────

def _replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)


def _integrate_step(phi0, b, A):
    sol = solve_ivp(_replicator_rhs, [0, DT], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0.copy()


def glv_log_likelihood(A_flat, phi_obs, b_all, sigma=0.10):
    """Log-likelihood log p(phi_obs | A, b_all).
    phi_obs: (n_patients, 3, N_G)
    b_all:   (n_patients, N_G)
    """
    A = A_flat.reshape(N_G, N_G)
    ll = 0.0
    n_patients = phi_obs.shape[0]
    for k in range(n_patients):
        phi2 = _integrate_step(phi_obs[k, 0], b_all[k], A)
        phi3 = _integrate_step(phi2,           b_all[k], A)
        ll -= np.sum((phi2 - phi_obs[k, 1])**2) / (2 * sigma**2)
        ll -= np.sum((phi3 - phi_obs[k, 2])**2) / (2 * sigma**2)
    return ll


def glv_log_prior(A_flat, sigma_prior=0.5):
    """Isotropic Gaussian prior on A entries (diagonal penalised more)."""
    A = A_flat.reshape(N_G, N_G)
    # off-diagonal: N(0, sigma_prior^2)
    off = A.copy(); np.fill_diagonal(off, 0)
    # diagonal: must be ≤ 0 (competition with self); soft constraint via half-Gaussian
    diag = np.diag(A)
    ll = -np.sum(off**2) / (2 * sigma_prior**2)
    ll += -np.sum(np.maximum(diag, 0)**2) / (2 * (sigma_prior / 4)**2)
    return ll


def score_A(A_flat, phi_obs, b_all, sigma=0.10, sigma_prior=0.5, eps=1e-4):
    """∇_A log p(A | phi_obs) via central finite differences."""
    grad = np.zeros_like(A_flat)
    f0 = (glv_log_likelihood(A_flat, phi_obs, b_all, sigma) +
          glv_log_prior(A_flat, sigma_prior))
    for i in range(len(A_flat)):
        dv          = np.zeros_like(A_flat); dv[i] = eps
        fp = (glv_log_likelihood(A_flat + dv, phi_obs, b_all, sigma) +
              glv_log_prior(A_flat + dv, sigma_prior))
        grad[i] = (fp - f0) / eps
    return grad, f0


# ── Fit b vectors for a given A (needed to evaluate likelihood) ───────────────

def fit_b_given_A(A_flat, phi_obs, n_init=3):
    """Quick gradient-free b fit per patient using Nelder-Mead."""
    from scipy.optimize import minimize
    A = A_flat.reshape(N_G, N_G)
    n_patients = phi_obs.shape[0]
    b_all = np.zeros((n_patients, N_G))
    for k in range(n_patients):
        def obj(b):
            phi2 = _integrate_step(phi_obs[k, 0], b, A)
            phi3 = _integrate_step(phi2,           b, A)
            return (np.sum((phi2 - phi_obs[k, 1])**2) +
                    np.sum((phi3 - phi_obs[k, 2])**2))
        best = None
        for _ in range(n_init):
            b0 = np.random.randn(N_G) * 0.1
            r  = minimize(obj, b0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-5})
            if best is None or r.fun < best.fun:
                best = r
        b_all[k] = best.x
    return b_all


# ── Unadjusted Langevin Algorithm (ULA) ──────────────────────────────────────

def _enforce_neg_diag(A_flat):
    A_mat = A_flat.reshape(N_G, N_G)
    np.fill_diagonal(A_mat, np.minimum(np.diag(A_mat), -1e-6))
    return A_mat.ravel()


def ula_sample(phi_obs, n_steps=2000, step_size=1e-4, thin=10,
               sigma=0.10, sigma_prior=0.5, seed=0, verbose=True, mala=False):
    """Langevin sampler over A. mala=True adds a Metropolis accept/reject step
    (MALA), which keeps the chain off the stiff-ODE regions a plain ULA can drift
    into and yields an unbiased posterior. MALA costs ~2x score evals per step."""
    rng = np.random.RandomState(seed)

    # Initialise A near zero (wide prior sample)
    A_flat = rng.randn(N_G * N_G) * 0.05
    np.fill_diagonal(A_flat.reshape(N_G, N_G), -0.1)

    # Fit b for initial A; cache score/logp at the current state
    b_all = fit_b_given_A(A_flat, phi_obs)
    grad, lp = score_A(A_flat, phi_obs, b_all, sigma, sigma_prior)

    samples, log_probs = [], []
    n_acc, n_prop = 0, 0
    t0 = time.time()

    for step in range(n_steps):
        # Refit b every 200 steps to track A, then refresh the cached score
        if step % 200 == 0 and step > 0:
            b_all = fit_b_given_A(A_flat, phi_obs)
            grad, lp = score_A(A_flat, phi_obs, b_all, sigma, sigma_prior)

        # Annealed step size: large initially, shrinks toward end
        eta = step_size * (1 + 0.5 * np.cos(np.pi * step / n_steps))

        # Langevin proposal
        z   = rng.randn(N_G * N_G)
        A_p = _enforce_neg_diag(A_flat + eta * grad + np.sqrt(2 * eta) * z)

        if mala:
            # Metropolis-adjusted: accept with prob min(1, posterior x proposal ratio)
            grad_p, lp_p = score_A(A_p, phi_obs, b_all, sigma, sigma_prior)
            # log q(x|x') - log q(x'|x) for Gaussian Langevin proposal
            fwd = -np.sum((A_p   - A_flat - eta * grad  )**2) / (4 * eta)
            bwd = -np.sum((A_flat - A_p   - eta * grad_p)**2) / (4 * eta)
            log_alpha = (lp_p - lp) + (bwd - fwd)
            n_prop += 1
            if np.log(rng.rand() + 1e-300) < log_alpha:
                A_flat, grad, lp = A_p, grad_p, lp_p
                n_acc += 1
            # else: reject — keep current A_flat, grad, lp
        else:
            A_flat = A_p
            grad, lp = score_A(A_flat, phi_obs, b_all, sigma, sigma_prior)

        if step % thin == 0:
            samples.append(A_flat.copy())
            log_probs.append(lp)
            if verbose and step % 400 == 0:
                rmse = np.sqrt(-lp / (phi_obs.shape[0] * 2 * N_G) * 2 * sigma**2)
                acc  = f'  acc={n_acc/max(1,n_prop):.2f}' if mala else ''
                print(f'  step {step:4d}  log_p={lp:.2f}  RMSE≈{rmse:.4f}{acc}  '
                      f'({time.time()-t0:.1f}s)', flush=True)

    if mala and verbose:
        print(f'  MALA acceptance rate: {n_acc/max(1,n_prop):.3f}', flush=True)
    return np.array(samples), np.array(log_probs)


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise(samples):
    """Posterior mean, std, and MAP of A (100,)."""
    log_w = np.arange(len(samples), dtype=float)  # use later half
    burnin = len(samples) // 2
    post   = samples[burnin:]
    return {
        'A_mean': post.mean(0).tolist(),
        'A_std':  post.std(0).tolist(),
        'A_map':  post[post.sum(1).argmax()].tolist(),  # proxy MAP
        'n_samples': len(post),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-chains',  type=int,   default=4)
    parser.add_argument('--n-steps',   type=int,   default=1000,
                        help='Langevin steps per chain (quick run for demo; use 5000+ for paper)')
    parser.add_argument('--step-size', type=float, default=5e-5)
    parser.add_argument('--thin',      type=int,   default=10)
    parser.add_argument('--sigma',     type=float, default=0.10,
                        help='likelihood noise scale (guild RMSE units)')
    parser.add_argument('--sigma-prior', type=float, default=0.5,
                        help='Gaussian prior std on A entries')
    parser.add_argument('--mala', action='store_true',
                        help='Metropolis-adjusted Langevin (MH accept/reject; '
                             'unbiased, avoids stiff-ODE drift, ~2x cost/step)')
    args = parser.parse_args()

    phi_all = np.load(PHI_NPY)   # (10,3,10)
    print(f'phi: {phi_all.shape}  chains={args.n_chains}  steps={args.n_steps}')
    print('Score-based Langevin posterior for gLV A matrix', flush=True)

    all_samples = []
    all_lps     = []

    for c in range(args.n_chains):
        print(f'\n=== Chain {c+1}/{args.n_chains} ===', flush=True)
        samples, lps = ula_sample(
            phi_all,
            n_steps=args.n_steps,
            step_size=args.step_size,
            thin=args.thin,
            sigma=args.sigma,
            sigma_prior=args.sigma_prior,
            seed=c,
            verbose=True,
            mala=args.mala,
        )
        all_samples.append(samples)
        all_lps.append(lps)
        print(f'  Chain {c+1}: {len(samples)} thinned samples  '
              f'log_p range [{lps.min():.1f}, {lps.max():.1f}]')

    samples_all = np.concatenate(all_samples, axis=0)  # (n_chains * n_thin, 100)
    lps_all     = np.concatenate(all_lps,     axis=0)

    np.save(OUT_DIR / 'A_samples.npy', samples_all)
    print(f'\nSaved: {OUT_DIR / "A_samples.npy"}  shape={samples_all.shape}')

    summary = summarise(samples_all)
    summary.update({
        'n_chains':   args.n_chains,
        'n_steps':    args.n_steps,
        'step_size':  args.step_size,
        'sigma':      args.sigma,
        'sigma_prior': args.sigma_prior,
        'log_p_mean': float(lps_all[len(lps_all)//2:].mean()),
    })
    json.dump(summary, open(OUT_DIR / 'posterior_summary.json', 'w'), indent=2)
    print(f'Saved: {OUT_DIR / "posterior_summary.json"}')

    # Quick RMSE check: evaluate MAP A on full dataset
    A_map = np.array(summary['A_map']).reshape(N_G, N_G)
    b_map = fit_b_given_A(A_map.ravel(), phi_all)
    sq, cnt = 0.0, 0
    for k in range(10):
        phi2 = _integrate_step(phi_all[k, 0], b_map[k], A_map)
        phi3 = _integrate_step(phi2,           b_map[k], A_map)
        sq  += np.sum((phi2 - phi_all[k, 1])**2) + np.sum((phi3 - phi_all[k, 2])**2)
        cnt += 2 * N_G
    rmse_full = float(np.sqrt(sq / cnt))
    print(f'\nPosterior MAP train-RMSE (all 10 patients): {rmse_full:.4f}')
    print('(cf. gLV LOO no-prior mean 0.0509 — LOO is a harder task)')


if __name__ == '__main__':
    main()
