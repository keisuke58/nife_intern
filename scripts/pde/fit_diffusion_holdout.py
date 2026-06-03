#!/usr/bin/env python3
"""
fit_diffusion_holdout.py — Spatial out-of-sample validation for the diffusion fit.

Leave-one-DAY-out analogue of the bulk LOO-CV, but in the *spatial* dimension:
the reaction-diffusion forward model (D_i, u) is fit to the FISH depth profiles of
all observed days EXCEPT one held-out day, then asked to PREDICT the held-out day's
depth profile. The held-out RMSE quantifies whether the fitted diffusivities
generalise across time, rather than merely interpolating the training days.

This is a thin sibling of `fit_diffusion_clsm.py`: it reuses that module's data
loader, forward model, loss builder and L-BFGS-B fit machinery, only changing
*which days enter the loss* and adding a held-out prediction + RMSE report.

Usage
-----
    # fit CH on days {1,6,10,15}, predict held-out day 21
    python scripts/pde/fit_diffusion_holdout.py --cond CH --holdout-day 21

    # same, but with the cross-diffusion forward model and a finer grid
    python scripts/pde/fit_diffusion_holdout.py --cond DH --holdout-day 15 \
        --crossdiff --Nz 48 --restarts 8

Outputs (results/diffusion_fit/)
--------------------------------
    D_fit_holdout_<COND>_d<DAY>.json   held-out RMSE + fitted D_i, u + train days
    fit_holdout_<COND>_d<DAY>.png      predicted vs observed depth profile, held-out day
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
# scripts/pde on sys.path so the sibling `fit_diffusion_clsm` imports as a bare module
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))

import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Reuse the sibling's machinery. fit_diffusion_clsm lives in scripts/pde, which we
# added to sys.path above, so the bare import resolves.
import fit_diffusion_clsm as fdc
from fit_diffusion_clsm import (
    load_zprofile_csv, build_loss,
    OUT_DIR, SPECIES_LABELS,
)
from nsp_pde_1d_heine import (
    load_A_b, load_obs, N_SP, SHORT, COLORS, CONDITIONS, D_DEFAULT, U_DEFAULT,
)
from scipy.optimize import minimize

# 5-species canonical order (matches GUILD/SHORT and the CSV columns).
SPECIES_ORDER = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']


def plot_holdout(phi_pred, obs, z_grid, day, D_fit, u_fit, cond_tag,
                 held_rmse, out_path=None):
    """Predicted vs observed depth profile for the single held-out day, all 5 species.

    phi_pred, obs : (Nz, N_SP) arrays for the held-out day.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, N_SP, figsize=(2.4 * N_SP, 3.4), sharey=True)
    for s, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(phi_pred[:, s], z_grid, color=COLORS[s], lw=2, label='Predicted')
        ax.plot(obs[:, s], z_grid, 'o', color=COLORS[s], ms=3, alpha=0.6,
                label='FISH (held-out)')
        ax.set_xlim(0, 1)
        ax.set_title(f'{SHORT[s]}\nD={D_fit[s]:.4f}', fontsize=8, color=COLORS[s])
        ax.set_xlabel('φᵢ', fontsize=8)
        if s == 0:
            ax.set_ylabel('z (depth)', fontsize=8)
    axes[-1].legend(fontsize=7)
    fig.suptitle(f'{cond_tag} — held-out day {day} prediction  '
                 f'(RMSE={held_rmse:.4f}, u={u_fit:.4f})', fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def rmse(pred, obs):
    """Root-mean-square error over finite entries of (pred - obs)."""
    diff = np.asarray(pred) - np.asarray(obs)
    valid = np.isfinite(diff)
    if valid.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.sum(diff[valid] ** 2) / valid.sum()))


def fit_diffusion_train(A, b, phi_obs, obs_train, t_train,
                        Nz=40, dt=0.05, n_restarts=3,
                        maxiter=300, ftol=1e-9, gtol=1e-6, fd_eps=1e-3):
    """
    Fit log(D_i), log(u) on the TRAINING days only.

    Mirrors `fit_diffusion_clsm.fit_diffusion` exactly (same bounds, restarts,
    L-BFGS-B options, seed) but the loss is built over `obs_train` / `t_train`,
    so the held-out day never enters the objective.
    """
    phi_bulk = jnp.array(phi_obs[0])
    phi_ic = np.tile(phi_obs[0], (Nz, 1))

    loss_fn = build_loss(A, b, phi_bulk, phi_ic, obs_train, t_train, Nz, dt)

    lb = np.full(N_SP + 1, np.log(1e-5))
    ub = np.array([np.log(0.5)] * N_SP + [np.log(0.1)])
    bounds = list(zip(lb, ub))

    best_res = None
    rng = np.random.default_rng(42)
    x0_mean = np.append(np.log(np.array(D_DEFAULT)), np.log(U_DEFAULT))

    for trial in range(n_restarts):
        x0 = x0_mean + 0.3 * rng.standard_normal(N_SP + 1)
        x0 = np.clip(x0, lb, ub)
        res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'eps': fd_eps})
        msg = res.message if isinstance(res.message, str) else \
            res.message.decode(errors='ignore')
        print(f'  Trial {trial + 1}: loss={res.fun:.6f}  success={res.success}  ({msg})')
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    D_fit = np.exp(best_res.x[:N_SP])
    u_fit = float(np.exp(best_res.x[N_SP]))
    return D_fit, u_fit, best_res


def main():
    parser = argparse.ArgumentParser(
        description='Leave-one-day-out spatial validation of the diffusion fit')
    parser.add_argument('--cond', choices=['CS', 'CH', 'DS', 'DH'], default='DH')
    parser.add_argument('--holdout-day', type=int, default=21,
                        help='Day held out from the fit and then predicted')
    parser.add_argument('--data', type=str,
                        default='results/diffusion_fit/zprofiles_all_ti.csv',
                        help='CSV with CLSM z-profiles (columns: condition,day,z_um,<species>)')
    parser.add_argument('--Nz', type=int, default=40)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=300,
                        help='L-BFGS-B max iterations per restart')
    parser.add_argument('--ftol', type=float, default=1e-9,
                        help='L-BFGS-B ftol (1e-12 was unreachable -> success=False)')
    parser.add_argument('--gtol', type=float, default=1e-6, help='L-BFGS-B gtol')
    parser.add_argument('--fd-eps', type=float, default=1e-3,
                        help='finite-difference step for the numerical gradient')
    parser.add_argument('--crossdiff', action='store_true',
                        help='Use the volume-filling cross-diffusion forward model '
                             '(nsp_pde_1d_heine_xdiff) instead of the diagonal baseline')
    args = parser.parse_args()

    # Swap the forward model exactly as fit_diffusion_clsm does. build_loss and the
    # held-out prediction both look up simulate_nsp_pde on the fdc module at call
    # time, so rebinding the module global is what actually takes effect.
    if args.crossdiff:
        from nsp_pde_1d_heine_xdiff import simulate_nsp_pde as _sim_xdiff
        fdc.simulate_nsp_pde = _sim_xdiff
        print('Forward model: volume-filling CROSS-DIFFUSION (rigorous, simplex-preserving)')
    else:
        print('Forward model: diagonal diffusion + phi0-slack (baseline)')

    cond_map = {c[2]: c[:2] for c in CONDITIONS}
    condition, cultivation = cond_map[args.cond]
    print(f'Condition: {args.cond} ({condition} {cultivation})')

    A, b = load_A_b(args.cond)
    phi_obs = load_obs(condition, cultivation)

    # ── Load ALL observed days, then split into train / held-out ──
    csv_path = Path(args.data)
    obs_all, z_grid, all_days = load_zprofile_csv(csv_path, args.cond, Nz_interp=args.Nz)
    all_days = [int(d) for d in all_days]
    print(f'  Loaded {csv_path}  shape={obs_all.shape}  days={all_days}')

    hd = int(args.holdout_day)
    if hd not in all_days:
        raise SystemExit(
            f'holdout-day {hd} not in observed days {all_days} for cond {args.cond}')
    if len(all_days) < 2:
        raise SystemExit('need >= 2 observed days to hold one out')

    hd_idx = all_days.index(hd)
    train_idx = [i for i in range(len(all_days)) if i != hd_idx]
    train_days = [all_days[i] for i in train_idx]
    obs_train = obs_all[train_idx]        # (T-1, Nz, N_SP)
    obs_holdout = obs_all[hd_idx]         # (Nz, N_SP)
    print(f'  Train days: {train_days}   Held-out day: {hd}')

    # ── Fit D_i, u on the TRAINING days only ──
    print(f'Fitting D_i and u on train days (Nz={args.Nz}, dt={args.dt}, '
          f'restarts={args.restarts})...')
    D_fit, u_fit, res = fit_diffusion_train(
        A, b, phi_obs, obs_train, train_days,
        Nz=args.Nz, dt=args.dt, n_restarts=args.restarts,
        maxiter=args.maxiter, ftol=args.ftol, gtol=args.gtol, fd_eps=args.fd_eps,
    )
    print(f'\nFitted D = {D_fit}')
    print(f'Fitted u = {u_fit:.5f}')

    # ── PREDICT the held-out day with the train-fitted params ──
    # Integrate over [train_days ∪ holdout]; the held-out day was NOT in the loss.
    pred_days = sorted(set(train_days) | {hd})
    pred_idx = pred_days.index(hd)
    phi_bulk = jnp.array(phi_obs[0])
    phi_ic = np.tile(phi_obs[0], (args.Nz, 1))
    phi_pred_all = fdc.simulate_nsp_pde(
        A, b, phi_bulk, phi_ic, np.array(pred_days, dtype=float),
        Nz=args.Nz, D=jnp.array(D_fit), u=u_fit, dt=args.dt,
    )                                      # (len(pred_days), Nz, N_SP)
    phi_pred_holdout = phi_pred_all[pred_idx]   # (Nz, N_SP)

    held_rmse = rmse(phi_pred_holdout, obs_holdout)
    train_rmse_each = {
        int(d): rmse(phi_pred_all[pred_days.index(d)], obs_all[all_days.index(d)])
        for d in train_days
    }
    print(f'\nHeld-out day {hd} RMSE = {held_rmse:.5f}')
    print(f'Train-day RMSEs       = '
          f'{ {k: round(v, 5) for k, v in train_rmse_each.items()} }')

    # ── Save JSON (keyed to the 5-species order) ──
    out_json = OUT_DIR / f'D_fit_holdout_{args.cond}_d{hd}.json'
    result = {
        'cond': args.cond,
        'holdout_day': hd,
        'train_days': train_days,
        'held_out_rmse': held_rmse,
        'train_rmse_per_day': train_rmse_each,
        'D_fit': D_fit.tolist(),
        'u_fit': u_fit,
        'D_species': SHORT,
        'species_order': SPECIES_ORDER,
        'train_loss': float(res.fun),
        'success': bool(res.success),
        'crossdiff': bool(args.crossdiff),
        'hparams': {'Nz': args.Nz, 'dt': args.dt, 'restarts': args.restarts,
                    'maxiter': args.maxiter, 'ftol': args.ftol, 'gtol': args.gtol,
                    'fd_eps': args.fd_eps},
    }
    out_json.write_text(json.dumps(result, indent=2))
    print(f'Saved: {out_json}')

    # ── Figure: predicted vs observed depth profile for the held-out day, all 5 species ──
    plot_holdout(
        phi_pred_holdout, obs_holdout, z_grid, hd, D_fit, u_fit,
        args.cond, held_rmse,
        out_path=OUT_DIR / f'fit_holdout_{args.cond}_d{hd}.png',
    )


if __name__ == '__main__':
    main()
