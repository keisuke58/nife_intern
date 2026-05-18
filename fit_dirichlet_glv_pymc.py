"""
Dirichlet-gLV model with PyMC on Dieckow 2024 guild-level data.

Model:
  φ_pred(t+1) = ODE_integrate(φ(t), A, b_p)    [gLV replicator, 1 week step]
  φ_obs(t+1)  ~ Dirichlet(α * φ_pred(t+1))     [Dirichlet likelihood]

  α (concentration) = shared scalar, controls noise level
  A = symmetric interaction matrix (shared across patients)
  b_p = patient-specific growth rates

Usage:
  python fit_dirichlet_glv_pymc.py [--draws 500] [--tune 500]
Output:
  results/mdsine2/dirichlet_glv_trace.nc
  results/mdsine2/A_posterior_mean.csv
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.integrate import solve_ivp

# ── data ────────────────────────────────────────────────
HERE = Path(__file__).parent
with open(HERE / 'results/dieckow_otu/guild_summary.json') as f:
    gs = json.load(f)

GUILDS   = gs['guilds']
PATIENTS = list('ABCDEFGHKL')
N_G      = len(GUILDS)   # 10
N_P      = len(PATIENTS) # 10

obs = {}
for s in gs['samples']:
    obs[(s['patient'], s['week'])] = np.array([s[g] for g in GUILDS], dtype=float)

# φ_obs: (N_P, 3, N_G) — weeks 1,2,3
phi_obs = np.zeros((N_P, 3, N_G))
for i, pat in enumerate(PATIENTS):
    for w in range(3):
        phi_obs[i, w] = obs[(pat, w + 1)]

phi_obs = np.clip(phi_obs, 1e-6, 1)
phi_obs /= phi_obs.sum(axis=-1, keepdims=True)

# ── ODE integration (numpy, for likelihood) ──────────────
def replicator_step(phi, A, b):
    """One-week RK45 integration of replicator ODE."""
    def rhs(t, x):
        f = b + A @ x
        return x * (f - x @ f)
    sol = solve_ivp(rhs, [0, 1.0], phi, method='RK45', max_step=0.05)
    y = np.clip(sol.y[:, -1], 1e-8, 1)
    return y / y.sum()

# ── PyMC model ───────────────────────────────────────────
def build_model():
    with pm.Model() as model:

        # Shared interaction matrix A (upper triangle → symmetric)
        A_upper = pm.Normal('A_upper',
                            mu=0, sigma=1,
                            shape=N_G * (N_G - 1) // 2)
        # Self-limitation: diagonal strictly negative
        A_diag  = pm.HalfNormal('A_diag_pos', sigma=1, shape=N_G)

        # Assemble A
        idx_u = np.triu_indices(N_G, k=1)
        A = pt.zeros((N_G, N_G))
        A = pt.set_subtensor(A[idx_u],        A_upper)
        A = pt.set_subtensor(A[idx_u[1], idx_u[0]], A_upper)  # symmetrise
        A = pt.set_subtensor(A[np.arange(N_G), np.arange(N_G)], -pt.abs(A_diag))

        # Patient-specific b
        b_all = pm.Normal('b', mu=0, sigma=1, shape=(N_P, N_G))

        # Concentration parameter (higher = less noise)
        alpha = pm.Exponential('alpha', lam=0.1)

        # Likelihood: predict wk2 and wk3 from wk1
        A_np = A.eval() if hasattr(A, 'eval') else None  # fallback for scan

        # Use observed phi as Dirichlet observations
        for i, pat in enumerate(PATIENTS):
            b_p = b_all[i]
            for w_from, w_to in [(0, 1), (1, 2)]:  # wk1→2, wk2→3
                phi_in  = phi_obs[i, w_from]
                phi_out = phi_obs[i, w_to]

                # Deterministic prediction via ODE (pytensor op wrapping numpy)
                phi_pred = pm.Deterministic(
                    f'phi_pred_{pat}_w{w_to+1}',
                    pt.as_tensor_variable(
                        replicator_step(phi_in,
                                        A.eval() if hasattr(A, 'eval') else np.zeros((N_G,N_G)),
                                        b_p.eval() if hasattr(b_p, 'eval') else np.zeros(N_G))
                    )
                )

                # Dirichlet likelihood
                pm.Dirichlet(
                    f'obs_{pat}_w{w_to+1}',
                    a=alpha * phi_pred,
                    observed=phi_out
                )

    return model

# ── main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--tune',  type=int, default=500)
    parser.add_argument('--chains', type=int, default=2)
    args = parser.parse_args()

    OUT = HERE / 'results/mdsine2'
    OUT.mkdir(exist_ok=True)

    # Simpler approach: use MAP + NUTS on A and b jointly
    # (ODE inside PyMC is tricky; use pytensor-compatible integrator)
    print("Building model...")

    # Lightweight version: skip ODE, use linear Dirichlet regression
    # φ(t+1) ~ Dirichlet(α * softmax(b_p + A @ φ(t)))
    with pm.Model() as model:
        A_upper = pm.Normal('A_upper', mu=0, sigma=0.5,
                            shape=N_G * (N_G - 1) // 2)
        A_diag_pos = pm.HalfNormal('A_diag_pos', sigma=0.5, shape=N_G)
        b_mat = pm.Normal('b', mu=0, sigma=0.5, shape=(N_P, N_G))
        alpha = pm.Exponential('alpha', lam=0.1)

        idx_u = np.triu_indices(N_G, k=1)
        A = pt.zeros((N_G, N_G))
        A = pt.set_subtensor(A[idx_u[0], idx_u[1]], A_upper)
        A = pt.set_subtensor(A[idx_u[1], idx_u[0]], A_upper)
        A = pt.set_subtensor(A[np.arange(N_G), np.arange(N_G)], -pt.abs(A_diag_pos))

        for i, pat in enumerate(PATIENTS):
            b_p = b_mat[i]
            for w_from, w_to in [(0, 1), (1, 2)]:
                phi_in  = pt.as_tensor_variable(phi_obs[i, w_from])
                phi_out = phi_obs[i, w_to]

                # Linear prediction (one Euler step ≈ replicator)
                f     = b_p + pt.dot(A, phi_in)
                fbar  = pt.dot(phi_in, f)
                dphi  = phi_in * (f - fbar)
                phi_pred_raw = phi_in + dphi
                phi_pred = pt.clip(phi_pred_raw, 1e-6, 1)
                phi_pred = phi_pred / phi_pred.sum()

                pm.Dirichlet(
                    f'obs_{pat}_w{w_to+1}',
                    a=alpha * phi_pred,
                    observed=phi_out
                )

        print(f"Sampling: {args.draws} draws, {args.tune} tune, {args.chains} chains")
        trace = pm.sample(draws=args.draws, tune=args.tune,
                          chains=args.chains, target_accept=0.9,
                          progressbar=True)

    # Save
    import arviz as az
    trace.to_netcdf(str(OUT / 'dirichlet_glv_trace.nc'))
    print(f"Trace saved: {OUT}/dirichlet_glv_trace.nc")

    # Extract A matrix
    A_upper_samples = trace.posterior['A_upper'].values.reshape(-1, N_G*(N_G-1)//2)
    A_diag_samples  = trace.posterior['A_diag_pos'].values.reshape(-1, N_G)

    A_mean_full = np.zeros((N_G, N_G))
    A_upper_mean = A_upper_samples.mean(axis=0)
    A_mean_full[idx_u[0], idx_u[1]] = A_upper_mean
    A_mean_full[idx_u[1], idx_u[0]] = A_upper_mean
    np.fill_diagonal(A_mean_full, -np.abs(A_diag_samples.mean(axis=0)))

    df = pd.DataFrame(A_mean_full, index=GUILDS, columns=GUILDS).round(4)
    df.to_csv(OUT / 'A_posterior_mean.csv')
    print(df)

    print(f"\nalpha posterior mean: {trace.posterior['alpha'].values.mean():.3f}")
    print("Done.")

if __name__ == '__main__':
    main()
