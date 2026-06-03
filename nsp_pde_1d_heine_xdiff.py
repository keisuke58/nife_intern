#!/usr/bin/env python3
"""
nsp_pde_1d_heine_xdiff.py — Rigorous (simplex-preserving) variant of the 1D NSP
spatial PDE for Heine 2025, using VOLUME-FILLING CROSS-DIFFUSION instead of the
diagonal-diffusion + φ0-slack scheme in nsp_pde_1d_heine.py.

Why this variant
----------------
The baseline (nsp_pde_1d_heine.py) diffuses each φ_i independently with its own
D_i and then patches the simplex constraint by recomputing φ0 = 1 - Σφ_i after
every transport step. That is NOT conservative / simplex-preserving.

This variant uses the volume-filling (size-exclusion) cross-diffusion flux
derived from a biased random walk with crowding (cf. Painter & Hillen 2002;
Burger et al.; and the cross-diffusion gLV framework of Freingruber et al. 2025):

    J_i = -D_i [ (1 - ρ) ∂φ_i/∂z  +  φ_i ∂ρ/∂z ],     ρ = Σ_j φ_j
    ∂φ_i/∂t = -∂J_i/∂z  -  u ∂φ_i/∂z  +  NSP_reaction
    φ0 = 1 - ρ   (void fraction; a genuine consequence, not a patch)

Key properties:
  * Degenerate flux: J_i → 0 as ρ → 1 (no space) or φ_i → 0  ⇒  0 ≤ φ_i, ρ ≤ 1
    are preserved by construction (rigorous; the clip below is numerical safety
    only and should essentially never activate).
  * Cross-coupling enters through the SHARED crowding gradient ∂ρ/∂z, so species
    interact through diffusion — yet only N_SP self-mobilities D_i are free
    (same parameter count as the diagonal model ⇒ still identifiable from
    z-profiles; no N_SP^2 = 25 parameter blow-up).
  * Conservative finite-volume discretisation (fluxes defined on cell faces).

Boundary conditions:
    z = 0 (substratum): no-flux  ⇒  J = 0 at the left face
    z = L (bulk):       Dirichlet φ_i = φ_bulk,i

Drop-in: exposes simulate_nsp_pde(...) with the SAME signature as
nsp_pde_1d_heine.simulate_nsp_pde, so fit_diffusion_clsm.py can use it directly
(see its --crossdiff flag).

Usage:
    python nsp_pde_1d_heine_xdiff.py --cond DH --Nz 40 --dt 0.05

NOTE: not runnable in this sandbox (no JAX/numpy installed). Validate on first
cluster run — sanity checks: φ0 = 1-Σφ stays in [0,1], profiles stay smooth,
mass behaves. The diagonal baseline (nsp_pde_1d_heine.py) is left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Reuse all data loaders, constants, NSP reaction and plotting from the baseline.
# Importing it also wires NSP_PATH onto sys.path (so hamilton_ode_jax_nsp resolves).
from nsp_pde_1d_heine import (
    N_SP, N_STATE, DAYS, SHORT, COLORS, CONDITIONS,
    D_DEFAULT, U_DEFAULT,
    load_obs, load_A_b, make_nsp_step, extract_phi,
    plot_depth_profiles, plot_heatmaps, plot_obs_vs_bulk,
    _make_initial_state,
)

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / 'results' / 'nsp_pde_xdiff'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Transport: volume-filling cross-diffusion (conservative FV) ─────────────

def transport_step_xdiff(phi, D, u, dz, dt, phi_bulk):
    """
    One explicit FV step of volume-filling cross-diffusion + upwind advection.

    phi      : (Nz, N_SP)
    D        : (N_SP,)  self-mobilities
    phi_bulk : (N_SP,)  Dirichlet value at z=L
    BC: no-flux at z=0 (left face flux = 0); Dirichlet at z=L.
    """
    rho = jnp.sum(phi, axis=-1)                       # (Nz,)  occupied fraction

    # Face-centred quantities between node k and k+1  (length Nz-1)
    phi_face = 0.5 * (phi[1:, :] + phi[:-1, :])       # (Nz-1, N_SP)
    rho_face = 0.5 * (rho[1:] + rho[:-1])             # (Nz-1,)
    dphi_dz  = (phi[1:, :] - phi[:-1, :]) / dz        # (Nz-1, N_SP)
    drho_dz  = (rho[1:] - rho[:-1]) / dz              # (Nz-1,)

    # Volume-filling flux on faces:  J = -D [ (1-ρ) ∂zφ + φ ∂zρ ]
    J = -D[None, :] * ((1.0 - rho_face)[:, None] * dphi_dz
                       + phi_face * drho_dz[:, None])  # (Nz-1, N_SP)

    dp = jnp.zeros_like(phi)
    # interior nodes 1..Nz-2:  -(J[k] - J[k-1]) / dz
    dp = dp.at[1:-1, :].add(-(J[1:, :] - J[:-1, :]) / dz)
    # node 0: no-flux left face  ⇒  -(J[0] - 0) / dz
    dp = dp.at[0, :].add(-J[0, :] / dz)

    # advection (upwind, u ≥ 0 → backward diff) on interior φ
    adv = u * (phi[1:-1, :] - phi[:-2, :]) / dz
    dp  = dp.at[1:-1, :].add(-adv)

    phi_new = jnp.clip(phi + dt * dp, 0.0, 1.0)        # numerical safety only
    phi_new = phi_new.at[-1, :].set(phi_bulk)          # Dirichlet at bulk
    return phi_new


transport_xdiff_jit = jax.jit(transport_step_xdiff, static_argnames=['dz', 'dt'])


def inject_phi_xdiff(state_Nz, phi_new):
    """Write φ_i back; φ0 = 1 - Σφ_i is now a genuine void fraction (not a patch)."""
    phi0 = jnp.clip(1.0 - jnp.sum(phi_new, axis=-1, keepdims=True), 1e-10, 1.0)
    s = state_Nz.at[:, :N_SP].set(phi_new)
    s = s.at[:, N_SP].set(phi0[:, 0])
    return s


# ── Simulation (same signature as nsp_pde_1d_heine.simulate_nsp_pde) ────────

def simulate_nsp_pde_xdiff(A, b, phi_bulk, phi_ic, t_out_days,
                           Nz=40, D=D_DEFAULT, u=U_DEFAULT, dt=0.05,
                           n_nsp_per_transport=5):
    """Lie operator splitting: NSP reaction Newton steps, then cross-diffusion transport."""
    dz = 1.0 / (Nz - 1)

    cfl = 0.5 * dz ** 2 / (float(jnp.max(D)) + 1e-12)   # explicit-diffusion limit
    if dt > cfl:
        print(f'  [warn] dt={dt:.4f} > diffusion CFL {cfl:.5f}; clipping')
        dt = 0.9 * cfl

    active_mask = jnp.ones(N_SP, dtype=jnp.int64)
    state = jax.vmap(
        lambda p: _make_initial_state(p, N_SP, active_mask, psi_init=0.999)
    )(jnp.array(phi_ic, dtype=jnp.float64))             # (Nz, N_STATE)

    nsp_vmap = make_nsp_step(A, b)

    t_curr, t_end, snapshots = 0.0, t_out_days[-1], {}

    def _snap(t, st):
        for tgt in t_out_days:
            if abs(t - tgt) < dt * 0.6 and tgt not in snapshots:
                snapshots[tgt] = np.array(extract_phi(st))

    _snap(t_curr, state)
    while t_curr < t_end - 1e-9:
        for _ in range(n_nsp_per_transport):
            state = nsp_vmap(state)
        phi   = extract_phi(state)
        phi   = transport_xdiff_jit(phi, D, u, dz, dt, phi_bulk)
        state = inject_phi_xdiff(state, phi)
        t_curr += dt
        _snap(t_curr, state)

    return np.stack([snapshots.get(t, np.full((Nz, N_SP), np.nan))
                     for t in t_out_days], axis=0)


# Drop-in alias: import this name to swap the diagonal model for cross-diffusion.
simulate_nsp_pde = simulate_nsp_pde_xdiff


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='NSP 1D volume-filling cross-diffusion PDE — Heine 2025')
    ap.add_argument('--cond',    choices=['CS', 'CH', 'DS', 'DH', 'all'], default='all')
    ap.add_argument('--Nz',      type=int,   default=40)
    ap.add_argument('--dt',      type=float, default=0.05)
    ap.add_argument('--n_nsp',   type=int,   default=5)
    ap.add_argument('--D_scale', type=float, default=1.0)
    ap.add_argument('--u',       type=float, default=U_DEFAULT)
    args = ap.parse_args()

    conds = [c for c in CONDITIONS if args.cond == 'all' or c[2] == args.cond]
    for condition, cultivation, tag in conds:
        print(f'\n=== {tag} ({condition} {cultivation}) [cross-diffusion] ===')
        A, b     = load_A_b(tag)
        phi_obs  = load_obs(condition, cultivation)
        phi_bulk = jnp.array(phi_obs[0], dtype=jnp.float64)
        phi_ic   = np.tile(phi_obs[0], (args.Nz, 1))
        z_grid   = np.linspace(0, 1, args.Nz)
        D        = D_DEFAULT * args.D_scale
        t_out    = np.array(DAYS, dtype=float)

        print(f'  Nz={args.Nz}, dt={args.dt}d, n_nsp={args.n_nsp}, D_scale={args.D_scale}')
        phi_out = simulate_nsp_pde_xdiff(
            A, b, phi_bulk, phi_ic, t_out,
            Nz=args.Nz, D=D, u=args.u, dt=args.dt,
            n_nsp_per_transport=args.n_nsp,
        )
        print(f'  phi_out.shape={phi_out.shape}')

        stem = f'{tag}_Nz{args.Nz}_dt{args.dt}_Dscale{args.D_scale}_xdiff'
        plot_depth_profiles(phi_out, t_out, z_grid, tag, out_path=OUT_DIR / f'depth_{stem}.png')
        plot_heatmaps(phi_out, t_out, z_grid, tag, out_path=OUT_DIR / f'heatmap_{stem}.png')
        plot_obs_vs_bulk(phi_out, phi_obs, t_out, tag, out_path=OUT_DIR / f'bulk_vs_obs_{stem}.png')
        np.save(OUT_DIR / f'phi_pde_{stem}.npy', phi_out)

    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
