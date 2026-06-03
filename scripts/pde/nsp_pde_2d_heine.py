#!/usr/bin/env python3
"""
nsp_pde_2d_heine.py — 2D spatial NSP (Hamilton) PDE for Heine 2025 5-species data.

A faithful 2D generalisation of ``nsp_pde_1d_heine.py`` (same numerics family:
Lie operator-splitting, per-node implicit-Euler Newton reaction via ``jax.vmap``,
explicit finite-difference transport).

Domain: (x, z)
    x : lateral direction          [0, Lx]   (within the biofilm patch)
    z : depth into biofilm / toward bulk  [0, Lz]

PDE (operator-split, strong form, Method of Lines):
    ∂φᵢ/∂t = Dᵢ (∂²φᵢ/∂x² + ∂²φᵢ/∂z²) − u ∂φᵢ/∂z  +  NSP_reaction(φᵢ, ψᵢ, φ₀, γ)
    ∂ψᵢ/∂t =                                          NSP_reaction(...)
    ∂φ₀/∂t =                                          NSP_reaction(...)
    γ enforces  Σᵢ φᵢ + φ₀ = 1   (local constraint at each (x,z) node)

    Advection acts only in z (toward the bulk); there is no lateral flow, so
    transport in x is pure diffusion.

    NSP state per node: g = [φ₁..φₙ, φ₀, ψ₁..ψₙ, γ]  (dim 2N+2 = 12)

Operator splitting (Lie):
    1. Reaction step : NSP implicit-Euler Newton at each (x,z) node, the *same*
       single-node reaction used by the 1D model, vmapped over both axes
       (flattened to a leading (Nx·Nz,) batch).
    2. Transport step: Dᵢ(∂ₓₓ + ∂_zz)φᵢ − u ∂_z φᵢ  (explicit FD, φ only).

Boundary conditions:
    z = 0  (substratum) : no-flux   ∂φᵢ/∂z = 0
    z = Lz (bulk)       : Dirichlet  φᵢ = φ_bulk,ᵢ  (day-1 median)
    x = 0, x = Lx       : **no-flux** ∂φᵢ/∂x = 0  (chosen over periodic: a finite
                          biofilm patch has insulated lateral walls — mass that
                          reaches a side does not wrap around to the other side).

Species order (canonical, shared with the 1D model):
    0: S. oralis   1: A. naeslundii   2: V. dispar/parvula
    3: F. nucleatum  4: P. gingivalis

Usage:
    python scripts/pde/nsp_pde_2d_heine.py                 # demo on all 4 conditions
    python scripts/pde/nsp_pde_2d_heine.py --cond DH       # Dysbiotic HOBIC only
    python scripts/pde/nsp_pde_2d_heine.py --Nx 24 --Nz 24 --dt 0.02 --steps 200

Note: D_i and u are placeholder values pending CLSM z-profile data
(Heine et al. 2025 HOBIC flow chamber); reused verbatim from the 1D model.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
})
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Reuse the 1D model: constants, data loaders, NSP per-node reaction, and the
# initial-state builder. The 1D module lives at the repo root (bare sibling).
# Importing it also inserts the Tmcmc202601 NSP core path onto sys.path and pulls
# in _make_initial_state / _newton_step, which we reuse here.
import nsp_pde_1d_heine as nsp1d
from nsp_pde_1d_heine import (
    N_SP, N_STATE, DAYS, SHORT, COLORS,
    CONDITIONS, SPECIES_MAP,
    D_DEFAULT, U_DEFAULT,
    NSP_PARAMS_BASE,
    load_obs, load_A_b,
    make_nsp_step,                 # vmappable single-node Newton step builder
)
from nsp_pde_1d_heine import _make_initial_state  # re-exported from NSP core

# Indices of the two species we snapshot in the demo figure.
IDX_PG = SHORT.index('P.g')   # P. gingivalis (4)
IDX_SO = SHORT.index('S.o')   # S. oralis     (0)

HERE    = Path(__file__).resolve().parents[2]   # repo root
OUT_DIR = HERE / 'results' / 'nsp_pde_2d'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 2D transport operator (explicit FD on φᵢ only) ────────────────────────────

def transport_step_2d(phi, D, u, dx, dz, dt, phi_bulk):
    """One explicit FD transport step on φ.

    Args:
        phi      : (Nx, Nz, N_SP)
        D        : (N_SP,) diffusivities
        u        : scalar advection velocity (z, toward bulk)
        dx, dz   : grid spacings
        dt       : transport time step
        phi_bulk : (N_SP,) Dirichlet value at z = Lz

    BCs:  x = 0/Lx no-flux ;  z = 0 no-flux ;  z = Lz Dirichlet (φ_bulk).
    Central diff for diffusion, first-order upwind for z-advection (u ≥ 0).
    """
    dp = jnp.zeros_like(phi)

    # ── x-diffusion (no-flux walls via mirror ghost nodes) ──
    # interior: (φ[i+1] - 2φ[i] + φ[i-1]) / dx²
    d2x_int = (phi[2:, :, :] - 2.0 * phi[1:-1, :, :] + phi[:-2, :, :]) / dx ** 2
    dp = dp.at[1:-1, :, :].add(D[None, None, :] * d2x_int)
    # x = 0 wall: ghost = φ[1]  →  2(φ[1] - φ[0]) / dx²
    d2x_lo = 2.0 * (phi[1, :, :] - phi[0, :, :]) / dx ** 2
    dp = dp.at[0, :, :].add(D[None, :] * d2x_lo)
    # x = Lx wall: ghost = φ[-2]  →  2(φ[-2] - φ[-1]) / dx²
    d2x_hi = 2.0 * (phi[-2, :, :] - phi[-1, :, :]) / dx ** 2
    dp = dp.at[-1, :, :].add(D[None, :] * d2x_hi)

    # ── z-diffusion ──
    # interior
    d2z_int = (phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]) / dz ** 2
    dp = dp.at[:, 1:-1, :].add(D[None, None, :] * d2z_int)
    # z = 0 no-flux ghost node: ghost = φ[1] → 2(φ[1] - φ[0]) / dz²
    d2z_0 = 2.0 * (phi[:, 1, :] - phi[:, 0, :]) / dz ** 2
    dp = dp.at[:, 0, :].add(D[None, :] * d2z_0)
    # (z = Lz is Dirichlet — overwritten below, so its update is irrelevant)

    # ── z-advection upwind (u ≥ 0 → backward difference) ──
    adv = u * (phi[:, 1:-1, :] - phi[:, :-2, :]) / dz
    dp = dp.at[:, 1:-1, :].add(-adv)

    phi_new = jnp.clip(phi + dt * dp, 0.0, 1.0)

    # Dirichlet BC at z = Lz
    phi_new = phi_new.at[:, -1, :].set(phi_bulk[None, :])
    return phi_new


transport_step_2d_jit = jax.jit(transport_step_2d, static_argnames=['dx', 'dz', 'dt'])


# ── State helpers (2D) ────────────────────────────────────────────────────────

def state_from_phi_2d(phi_NxNz):
    """phi_NxNz: (Nx, Nz, N_SP) → NSP state (Nx, Nz, N_STATE)."""
    Nx, Nz, _ = phi_NxNz.shape
    active_mask = jnp.ones(N_SP, dtype=jnp.int64)
    flat = phi_NxNz.reshape(Nx * Nz, N_SP)
    init_fn = jax.vmap(lambda p: _make_initial_state(p, N_SP, active_mask, psi_init=0.999))
    return init_fn(flat).reshape(Nx, Nz, N_STATE)


def extract_phi_2d(state):
    """state: (Nx, Nz, N_STATE) → φᵢ (Nx, Nz, N_SP)."""
    return state[:, :, :N_SP]


def inject_phi_2d(state, phi_new):
    """Replace φᵢ in state; renormalise φ₀ = 1 − Σφᵢ at each node."""
    phi_sum = jnp.sum(phi_new, axis=-1, keepdims=True)
    phi0 = jnp.clip(1.0 - phi_sum, 1e-10, 1.0 - 1e-10)
    s = state.at[:, :, :N_SP].set(phi_new)
    s = s.at[:, :, N_SP].set(phi0[:, :, 0])
    return s


def make_nsp_step_2d(A, b):
    """Wrap the 1D vmapped single-node Newton step to act on a 2D grid.

    Reuses ``nsp_pde_1d_heine.make_nsp_step`` (which vmaps a single-node
    ``_newton_step`` over a leading batch axis) by flattening (Nx, Nz) → Nx·Nz.
    """
    step_flat = make_nsp_step(A, b)   # vmapped over leading (batch,) axis

    @jax.jit
    def react(state_2d):
        Nx, Nz, _ = state_2d.shape
        flat = state_2d.reshape(Nx * Nz, N_STATE)
        flat = step_flat(flat)
        return flat.reshape(Nx, Nz, N_STATE)

    return react


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_nsp_pde_2d(A, b, phi_bulk, phi_ic, t_out_days,
                        Nx=24, Nz=24, D=D_DEFAULT, u=U_DEFAULT, dt=0.02,
                        n_nsp_per_transport=5, max_steps=None):
    """Run the 2D NSP PDE.

    Args:
        A, b        : interaction matrix and growth rates (5×5, 5)
        phi_bulk    : Dirichlet BC at z=Lz                (5,)
        phi_ic      : initial φᵢ field, (Nx, Nz, 5) or (Nz, 5) broadcast over x
        t_out_days  : times to snapshot (in days)
        Nx, Nz      : grid nodes (x, z)
        D           : diffusivities (5,)
        u           : advection velocity (scalar, z toward bulk)
        dt          : transport time step in days
        n_nsp_per_transport : NSP Newton steps per transport step
        max_steps   : if set, stop after this many transport steps (demo cap)

    Returns:
        phi_out : (len(t_out_days), Nx, Nz, N_SP)
    """
    phi_ic = np.asarray(phi_ic, dtype=np.float64)
    if phi_ic.ndim == 2:                      # (Nz, N_SP) → broadcast over x
        phi_ic = np.tile(phi_ic[None, :, :], (Nx, 1, 1))

    dx = 1.0 / (Nx - 1) if Nx > 1 else 1.0
    dz = 1.0 / (Nz - 1)

    # Explicit-diffusion CFL guard (2D): dt ≤ 0.5 / (D_max (1/dx² + 1/dz²))
    cfl_limit = 0.5 / (float(jnp.max(D)) * (1.0 / dx ** 2 + 1.0 / dz ** 2))
    if dt > cfl_limit:
        print(f'  [warn] dt={dt:.5f} > 2D CFL limit {cfl_limit:.5f}; clipping to 0.9·CFL')
        dt = 0.9 * cfl_limit

    state = state_from_phi_2d(jnp.array(phi_ic, dtype=jnp.float64))
    react = make_nsp_step_2d(A, b)

    t_curr = 0.0
    t_end = float(t_out_days[-1])
    snapshots = {}

    def _snap(t, st):
        for tgt in t_out_days:
            if abs(t - tgt) < dt * 0.6 and tgt not in snapshots:
                snapshots[tgt] = np.array(extract_phi_2d(st))

    _snap(t_curr, state)

    n_done = 0
    while t_curr < t_end - 1e-9:
        # reaction
        for _ in range(n_nsp_per_transport):
            state = react(state)
        # transport
        phi = extract_phi_2d(state)
        phi = transport_step_2d_jit(phi, D, u, dx, dz, dt, phi_bulk)
        state = inject_phi_2d(state, phi)

        t_curr += dt
        n_done += 1
        _snap(t_curr, state)
        if max_steps is not None and n_done >= max_steps:
            break

    # ensure a final snapshot exists even if no t_out target landed at t_end
    if not snapshots or max(snapshots) < t_curr - dt:
        snapshots[t_curr] = np.array(extract_phi_2d(state))

    keys = [t for t in t_out_days if t in snapshots] or [max(snapshots)]
    phi_out = np.stack([snapshots[t] for t in keys], axis=0)
    return phi_out, keys


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_snapshot_2d(phi_final, cond_tag, day, out_path=None,
                     species_idx=(IDX_PG, IDX_SO)):
    """Depth–x heatmaps of a couple of species at the final time.

    phi_final : (Nx, Nz, N_SP) — the final snapshot.
    """
    n_panel = len(species_idx)
    fig, axes = plt.subplots(1, n_panel, figsize=(3.2 * n_panel, 3.4))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, species_idx):
        field = phi_final[:, :, s].T          # (Nz, Nx): rows = depth, cols = x
        vmax = float(field.max()) or 1e-3
        im = ax.imshow(field, origin='lower', aspect='auto',
                       cmap='YlOrRd', vmin=0.0, vmax=vmax,
                       extent=[0, 1, 0, 1])
        ax.set_title(SHORT[s], fontsize=10, color=COLORS[s])
        ax.set_xlabel('x (lateral)', fontsize=8)
        ax.set_ylabel('z (depth; 0=substratum, 1=bulk)', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'NSP 2D PDE — {cond_tag}  (Day {day:.0f}, x/z no-flux walls, z=L Dirichlet)',
                 fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='2D NSP spatial PDE for Heine 2025')
    parser.add_argument('--cond', choices=['CS', 'CH', 'DS', 'DH', 'all'], default='all')
    parser.add_argument('--Nx', type=int, default=24)
    parser.add_argument('--Nz', type=int, default=24)
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Transport time step in days (default 0.02)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Cap on number of transport steps (demo); '
                             'default runs to the final observation day')
    parser.add_argument('--n_nsp', type=int, default=5,
                        help='NSP Newton steps per transport step')
    parser.add_argument('--D_scale', type=float, default=1.0)
    parser.add_argument('--u', type=float, default=U_DEFAULT)
    args = parser.parse_args()

    conds = [c for c in CONDITIONS if args.cond == 'all' or c[2] == args.cond]
    D = D_DEFAULT * args.D_scale

    for condition, cultivation, tag in conds:
        print(f'\n=== {tag} ({condition} {cultivation}) ===')
        A, b = load_A_b(tag)
        phi_obs = load_obs(condition, cultivation)
        phi_bulk = jnp.array(phi_obs[0], dtype=jnp.float64)   # Day-1 median as bulk BC

        # IC: uniform = Day-1 bulk composition at every (x,z) node
        phi_ic = np.tile(phi_obs[0], (args.Nx, args.Nz, 1))

        # Time horizon: if --steps given, the demo stops there; otherwise run to
        # the last observation day. We always snapshot the final state.
        if args.steps is not None:
            t_out = np.array([args.steps * args.dt], dtype=float)
        else:
            t_out = np.array(DAYS, dtype=float)

        print(f'  Nx={args.Nx}, Nz={args.Nz}, dt={args.dt}d, '
              f'n_nsp={args.n_nsp}, D_scale={args.D_scale}, '
              f'steps={"to day %g" % t_out[-1] if args.steps is None else args.steps}')

        phi_out, keys = simulate_nsp_pde_2d(
            A, b, phi_bulk, phi_ic, t_out,
            Nx=args.Nx, Nz=args.Nz, D=D, u=args.u, dt=args.dt,
            n_nsp_per_transport=args.n_nsp, max_steps=args.steps,
        )
        final_day = keys[-1]
        phi_final = phi_out[-1]
        print(f'  phi_out.shape={phi_out.shape}  (final snapshot day {final_day:.3g})')

        # Required demo outputs (task spec): per-condition npy + 2-panel figure.
        npy_path = OUT_DIR / f'phi_2d_{tag}.npy'
        png_path = OUT_DIR / f'nsp_pde_2d_{tag}.png'
        np.save(npy_path, phi_out)
        print(f'Saved: {npy_path}')
        plot_snapshot_2d(phi_final, tag, final_day, out_path=png_path)

    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
