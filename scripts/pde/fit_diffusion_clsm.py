#!/usr/bin/env python3
"""
fit_diffusion_clsm.py — Fit diffusion coefficients D_i and advection u
from CLSM/FISH z-profile data (Heine 2025 flow chamber).

Awaiting data from Szafrański lab.  This script is ready to run once
z-profile CSVs or NPY arrays are available.

Expected data format
--------------------
CSV with columns: day, z_um, species, intensity_norm
  - z_um       : depth from substratum in µm
  - intensity_norm : relative FISH intensity (sum across species = 1 at each z)
  - species    : one of S.oralis, A.naeslundii, Vd/Vp, F.nucleatum, P.gingivalis

Fitting strategy
----------------
  - A, b already estimated from bulk time-series (TMCMC / gLV LOO)
  - Free params: log D_i (i=1..N_SP) and log u  (N_SP+1 params)
  - Objective: MSE between PDE z-profiles and FISH z-profiles at observed days
  - Optimiser: scipy L-BFGS-B (gradient-free; JAX vmap makes forward pass fast)

Usage
-----
    # With real data (CSV):
    python fit_diffusion_clsm.py --cond DH --data z_profiles_DH.csv

    # With synthetic test data (no CSV needed):
    python fit_diffusion_clsm.py --cond DH --synthetic
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from nsp_pde_1d_heine import (
    simulate_nsp_pde, load_A_b, load_obs,
    D_DEFAULT, U_DEFAULT, N_SP, DAYS,
    COLORS, SHORT, CONDITIONS, OUT_DIR as OUT_DIR_1D,
)

GLV_FIT = HERE / 'results' / 'heine2025' / 'fit_glv_heine.json'
OUT_DIR = HERE / 'results' / 'diffusion_fit'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES_LABELS = ['S.oralis', 'A.naeslundii', 'Vd/Vp', 'F.nucleatum', 'P.gingivalis']


# ── Data loader ────────────────────────────────────────────────────────────

def load_zprofile_csv(csv_path: Path, cond_tag: str,
                      days_keep=None, Nz_interp=40):
    """
    Load observed z-profiles from CLSM CSV.

    Returns:
        obs : (T, Nz_interp, N_SP)  interpolated onto uniform grid
        z_grid : (Nz_interp,)  normalised [0,1]
        t_days : list of day values
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df['condition'] == cond_tag] if 'condition' in df.columns else df

    if days_keep is None:
        days_keep = sorted(df['day'].unique())

    z_max = df['z_um'].max()
    z_grid = np.linspace(0, 1, Nz_interp)

    obs = np.zeros((len(days_keep), Nz_interp, N_SP))
    for t, day in enumerate(days_keep):
        sub = df[df['day'] == day]
        z_raw = sub['z_um'].values / z_max   # normalise
        for s, sp in enumerate(SPECIES_LABELS):
            col = [c for c in sub.columns if sp.lower() in c.lower()]
            if not col:
                continue
            y_raw = sub[col[0]].values
            y_raw = np.maximum(y_raw, 0)
            obs[t, :, s] = np.interp(z_grid, z_raw, y_raw)

    # renormalise per z-node
    row_sum = obs.sum(axis=-1, keepdims=True)
    obs = np.where(row_sum > 1e-12, obs / row_sum, obs)
    return obs, z_grid, list(days_keep)


def make_synthetic_data(A, b, phi_obs, Nz=40, noise_std=0.02,
                        D_true=None, u_true=0.005):
    """Generate synthetic z-profiles for testing (no real data needed)."""
    if D_true is None:
        D_true = np.array(D_DEFAULT)
    phi_bulk = jnp.array(phi_obs[0])
    phi_ic   = np.tile(phi_obs[0], (Nz, 1))
    t_out    = np.array(DAYS, dtype=float)
    phi_out  = simulate_nsp_pde(
        A, b, phi_bulk, phi_ic, t_out,
        Nz=Nz, D=jnp.array(D_true), u=u_true,
    )
    phi_out  = phi_out + noise_std * np.random.randn(*phi_out.shape)
    phi_out  = np.clip(phi_out, 0, 1)
    z_grid   = np.linspace(0, 1, Nz)
    return phi_out, z_grid, DAYS


# ── Loss and optimisation ──────────────────────────────────────────────────

def build_loss(A, b, phi_bulk, phi_ic, obs, t_days, Nz, dt=0.05):
    """
    Returns a scalar loss function of log_theta = [log D_1..D_N, log u].
    Forward pass uses simulate_nsp_pde (numpy/jax, no grad through PDE).
    """
    t_out = np.array(t_days, dtype=float)

    def loss(log_theta):
        D = jnp.exp(jnp.array(log_theta[:N_SP], dtype=jnp.float64))
        u = float(np.exp(log_theta[N_SP]))
        phi_pred = simulate_nsp_pde(
            A, b, phi_bulk, phi_ic, t_out,
            Nz=Nz, D=D, u=u, dt=dt,
        )   # (T, Nz, N_SP)
        # MSE over non-nan entries
        diff  = phi_pred - obs
        valid = np.isfinite(diff)
        mse   = float(np.sum(diff[valid]**2) / (valid.sum() + 1e-12))
        return mse

    return loss


def fit_diffusion(A, b, phi_obs, obs_zprofile, z_grid, t_days,
                  Nz=40, dt=0.05, n_restarts=3):
    """
    Fit log(D_i) and log(u) by L-BFGS-B minimisation of MSE on z-profiles.

    Returns:
        D_fit : (N_SP,) fitted diffusivities
        u_fit : scalar advection
        result : scipy OptimizeResult
    """
    phi_bulk = jnp.array(phi_obs[0])
    phi_ic   = np.tile(phi_obs[0], (Nz, 1))

    loss_fn = build_loss(A, b, phi_bulk, phi_ic, obs_zprofile, t_days, Nz, dt)

    # bounds: D ∈ [1e-5, 0.5], u ∈ [0, 0.1] (log scale)
    lb = np.full(N_SP + 1, np.log(1e-5))
    ub = np.array([np.log(0.5)] * N_SP + [np.log(0.1)])
    bounds = list(zip(lb, ub))

    best_res = None
    rng = np.random.default_rng(42)

    # mean start = [log D_1..D_N, log u]  (N_SP + 1 entries)
    x0_mean = np.append(np.log(np.array(D_DEFAULT)), np.log(U_DEFAULT))

    for trial in range(n_restarts):
        x0 = x0_mean + 0.3 * rng.standard_normal(N_SP + 1)
        x0 = np.clip(x0, lb, ub)
        res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12})
        print(f'  Trial {trial+1}: loss={res.fun:.6f}  success={res.success}')
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    D_fit = np.exp(best_res.x[:N_SP])
    u_fit = float(np.exp(best_res.x[N_SP]))
    return D_fit, u_fit, best_res


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_fit(phi_pred, obs, z_grid, t_days, D_fit, u_fit, cond_tag, out_path=None):
    """Overlay fitted PDE vs observed z-profiles per species per day."""
    import matplotlib.pyplot as plt
    T = len(t_days)
    fig, axes = plt.subplots(N_SP, T, figsize=(2.2 * T, 2.2 * N_SP), sharex=True)
    for s in range(N_SP):
        for k, day in enumerate(t_days):
            ax = axes[s, k]
            ax.plot(phi_pred[k, :, s], z_grid, color=COLORS[s], lw=2, label='Fit')
            ax.plot(obs[k, :, s], z_grid, 'o', color=COLORS[s], ms=3,
                    alpha=0.6, label='FISH')
            ax.set_xlim(0, 1)
            if k == 0:
                ax.set_ylabel(f'{SHORT[s]}\nz', fontsize=7)
            if s == 0:
                ax.set_title(f'Day {day:.0f}', fontsize=8)
            if s == N_SP - 1:
                ax.set_xlabel('φᵢ', fontsize=7)
    axes[0, -1].legend(fontsize=7)
    d_str = '  '.join(f'D_{SHORT[i]}={D_fit[i]:.4f}' for i in range(N_SP))
    fig.suptitle(f'{cond_tag} — D fit\n{d_str}  u={u_fit:.4f}', fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fit D_i from CLSM z-profiles')
    parser.add_argument('--cond',      choices=['CS','CH','DS','DH'], default='DH')
    parser.add_argument('--data',      type=str, default=None,
                        help='CSV with CLSM z-profiles (columns: day,z_um,species,...)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (no real data needed)')
    parser.add_argument('--Nz',        type=int,   default=40)
    parser.add_argument('--dt',        type=float, default=0.05)
    parser.add_argument('--restarts',  type=int,   default=3)
    parser.add_argument('--crossdiff', action='store_true',
                        help='Use rigorous volume-filling cross-diffusion model '
                             '(nsp_pde_1d_heine_xdiff) instead of the diagonal baseline')
    args = parser.parse_args()

    if args.crossdiff:
        # Swap the forward model: rebind the module-global simulate_nsp_pde that
        # build_loss / make_synthetic_data / the final plot all look up at call time.
        global simulate_nsp_pde
        from nsp_pde_1d_heine_xdiff import simulate_nsp_pde as _sim_xdiff
        simulate_nsp_pde = _sim_xdiff
        print('Forward model: volume-filling CROSS-DIFFUSION (rigorous, simplex-preserving)')
    else:
        print('Forward model: diagonal diffusion + φ0-slack (baseline)')

    cond_map = {c[2]: c[:2] for c in CONDITIONS}
    condition, cultivation = cond_map[args.cond]

    print(f'Condition: {args.cond} ({condition} {cultivation})')
    A, b    = load_A_b(args.cond)
    phi_obs = load_obs(condition, cultivation)

    if args.synthetic:
        print('Using synthetic z-profile data ...')
        obs, z_grid, t_days = make_synthetic_data(A, b, phi_obs, Nz=args.Nz)
        D_true = np.array(D_DEFAULT)
        print(f'  True D = {D_true}  u = {U_DEFAULT}')
    elif args.data:
        csv_path = Path(args.data)
        obs, z_grid, t_days = load_zprofile_csv(csv_path, args.cond, Nz_interp=args.Nz)
        D_true = None
        print(f'  Loaded {csv_path}  shape={obs.shape}')
    else:
        print('No data provided. Use --synthetic or --data <csv>.')
        print('Waiting for CLSM z-profile data from Szafrański lab.')
        return

    print(f'Fitting D_i and u (Nz={args.Nz}, dt={args.dt}, restarts={args.restarts})...')
    D_fit, u_fit, res = fit_diffusion(
        A, b, phi_obs, obs, z_grid, t_days,
        Nz=args.Nz, dt=args.dt, n_restarts=args.restarts,
    )
    print(f'\nFitted D = {D_fit}')
    print(f'Fitted u = {u_fit:.5f}')
    if D_true is not None:
        print(f'True    D = {D_true}')
        rel_err = np.abs(D_fit - D_true) / (D_true + 1e-12)
        print(f'Rel err   = {rel_err.round(3)}')

    # Save fitted params
    out_json = OUT_DIR / f'D_fit_{args.cond}.json'
    result   = {
        'cond': args.cond,
        'D_fit': D_fit.tolist(),
        'u_fit': u_fit,
        'D_species': SHORT,
        'loss': float(res.fun),
        'success': bool(res.success),
    }
    out_json.write_text(json.dumps(result, indent=2))
    print(f'Saved: {out_json}')

    # Plot
    phi_pred = simulate_nsp_pde(
        A, b, jnp.array(phi_obs[0]), np.tile(phi_obs[0], (args.Nz, 1)),
        np.array(t_days, dtype=float),
        Nz=args.Nz, D=jnp.array(D_fit), u=u_fit, dt=args.dt,
    )
    plot_fit(phi_pred, obs, z_grid, t_days, D_fit, u_fit, args.cond,
             out_path=OUT_DIR / f'fit_{args.cond}.png')


if __name__ == '__main__':
    main()
