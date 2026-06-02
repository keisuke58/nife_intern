#!/usr/bin/env python3
"""
nsp_pde_2d_heine.py — 2D NSP (Hamilton) PDE for Heine 2025 5-species data.

Domain: (x, z)
    x : lateral / flow direction  [0, Lx]
    z : depth into biofilm         [0, Lz]

PDE (operator-split, Method of Lines):
    ∂φᵢ/∂t = Dᵢ(∂²φᵢ/∂x² + ∂²φᵢ/∂z²) − u ∂φᵢ/∂z  +  NSP_reaction(...)
    ∂ψᵢ/∂t = NSP_reaction(...)
    γ enforces Σᵢ φᵢ + φ₀ = 1  (local constraint at each node)

Operator splitting (Lie):
    1. Reaction : NSP implicit-Euler Newton at each (x,z) node (vmap over flat grid)
    2. Transport : 2D central-diff diffusion + z-upwind advection

Boundary conditions:
    x         : periodic  (lateral wrap-around)
    z = 0     : no-flux   ∂φᵢ/∂z = 0  (substratum)
    z = Lz    : Dirichlet φᵢ = φ_bulk  (bulk / gingival fluid)

Usage:
    python nsp_pde_2d_heine.py                        # all 4 conditions
    python nsp_pde_2d_heine.py --cond DH              # Dysbiotic HOBIC only
    python nsp_pde_2d_heine.py --Nx 30 --Nz 50       # finer grid
    python nsp_pde_2d_heine.py --D_scale 2.0          # faster diffusion

Note: D_i and u are placeholder values pending CLSM z-profile data.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import json
import sys
from pathlib import Path

import matplotlib
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

NSP_PATH = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
sys.path.insert(0, str(NSP_PATH))
from hamilton_ode_jax_nsp import _residual, _newton_step, _make_initial_state

import pandas as pd

HERE     = Path(__file__).resolve().parents[2]
GLV_FIT  = HERE / 'results' / 'heine2025' / 'fit_glv_heine.json'
DATA_CSV = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data'
                '/fig3_species_distribution_replicates.csv')
OUT_DIR  = HERE / 'results' / 'nsp_pde_2d'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SP    = 5
N_STATE = 2 * N_SP + 2   # 12
DAYS    = [1, 3, 6, 10, 15, 21]
SHORT   = ['S.o', 'A.n', 'Vd/Vp', 'F.n', 'P.g']
COLORS  = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#d62728']

CONDITIONS = [
    ('Commensal', 'Static', 'CS'),
    ('Commensal', 'HOBIC',  'CH'),
    ('Dysbiotic', 'Static', 'DS'),
    ('Dysbiotic', 'HOBIC',  'DH'),
]
SPECIES_MAP = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',   'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula',  'F. nucleatum', 'P. gingivalis_W83'],
}

NSP_PARAMS_BASE = {
    "n_sp":              N_SP,
    "dt_h":              1e-3,
    "Kp1":               1e-4,
    "Eta":               jnp.ones(N_SP,  dtype=jnp.float64),
    "EtaPhi":            jnp.ones(N_SP,  dtype=jnp.float64),
    "c":                 25.0,
    "alpha":             100.0,
    "K_hill":            jnp.array(0.0,  dtype=jnp.float64),
    "n_hill":            jnp.array(4.0,  dtype=jnp.float64),
    "hill_gate_species": jnp.array([-1, 0], dtype=jnp.int32),
    "active_mask":       jnp.ones(N_SP, dtype=jnp.int64),
}

# Placeholder diffusivities (normalised: D_eff/L² × t_day)
D_DEFAULT = jnp.array([0.015, 0.008, 0.012, 0.012, 0.005], dtype=jnp.float64)
U_DEFAULT = 0.005


# ── Data helpers ───────────────────────────────────────────────────────────

def load_obs(condition, cultivation):
    df  = pd.read_csv(DATA_CSV)
    sp  = SPECIES_MAP[condition]
    sub = df[(df['condition'] == condition) & (df['cultivation'] == cultivation)]
    obs = np.zeros((len(DAYS), N_SP))
    for t, day in enumerate(DAYS):
        row = sub[sub['day'] == day]
        for s, name in enumerate(sp):
            vals = row[row['species'] == name]['distribution_pct'].values
            obs[t, s] = np.median(vals) if len(vals) else 0.0
    sums = obs.sum(axis=1, keepdims=True)
    return np.where(sums > 1e-12, obs / sums, obs)


def load_A_b(cond_tag):
    d = json.loads(GLV_FIT.read_text())
    A = jnp.array(d[cond_tag]['A'], dtype=jnp.float64)
    b = jnp.array(d[cond_tag]['b'], dtype=jnp.float64)
    return A, b


# ── Reaction step (vmap over flat grid) ───────────────────────────────────

def make_nsp_reaction(A, b):
    params = {**NSP_PARAMS_BASE, "A": A, "b_diag": b}
    step_fn = jax.jit(jax.vmap(lambda g: _newton_step(g, params)))

    def react(state_NxNz):
        """state_NxNz: (Nx, Nz, N_STATE)"""
        Nx, Nz, _ = state_NxNz.shape
        flat  = state_NxNz.reshape(Nx * Nz, N_STATE)
        flat  = step_fn(flat)
        return flat.reshape(Nx, Nz, N_STATE)

    return react


# ── Transport step (2D FD) ─────────────────────────────────────────────────

def transport_step_2d(phi, D, u, dx, dz, dt, phi_bulk):
    """
    phi      : (Nx, Nz, N_SP)
    D        : (N_SP,)
    phi_bulk : (N_SP,)  — Dirichlet at z=L
    x: periodic,  z=0: no-flux,  z=L: Dirichlet
    """
    dp = jnp.zeros_like(phi)

    # x-direction diffusion (periodic)
    phi_xp = jnp.roll(phi, -1, axis=0)
    phi_xm = jnp.roll(phi,  1, axis=0)
    d2x    = (phi_xp - 2.0 * phi + phi_xm) / dx ** 2
    dp     = dp + D[None, None, :] * d2x

    # z-direction diffusion — interior
    d2z_int = (phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]) / dz ** 2
    dp      = dp.at[:, 1:-1, :].add(D[None, None, :] * d2z_int)

    # z=0 no-flux ghost node
    d2z_0 = 2.0 * (phi[:, 1, :] - phi[:, 0, :]) / dz ** 2
    dp    = dp.at[:, 0, :].add(D[None, :] * d2z_0)

    # z-advection upwind (u ≥ 0 → toward bulk)
    adv = u * (phi[:, 1:-1, :] - phi[:, :-2, :]) / dz
    dp  = dp.at[:, 1:-1, :].add(-adv)

    phi_new = jnp.clip(phi + dt * dp, 0.0, 1.0)
    phi_new = phi_new.at[:, -1, :].set(phi_bulk[None, :])
    return phi_new


transport_2d_jit = jax.jit(transport_step_2d, static_argnames=['dx', 'dz', 'dt'])


# ── State helpers ──────────────────────────────────────────────────────────

def state_from_phi_2d(phi_NxNz):
    """phi_NxNz: (Nx, Nz, N_SP) → state (Nx, Nz, N_STATE)"""
    Nx, Nz, _ = phi_NxNz.shape
    active_mask = jnp.ones(N_SP, dtype=jnp.int64)
    flat_phi = phi_NxNz.reshape(Nx * Nz, N_SP)
    init_fn  = jax.vmap(lambda p: _make_initial_state(p, N_SP, active_mask, psi_init=0.999))
    return init_fn(flat_phi).reshape(Nx, Nz, N_STATE)


def extract_phi_2d(state):
    return state[:, :, :N_SP]


def inject_phi_2d(state, phi_new):
    phi_sum = jnp.sum(phi_new, axis=-1, keepdims=True)
    phi0    = jnp.clip(1.0 - phi_sum, 1e-10, 1.0)
    s = state.at[:, :, :N_SP].set(phi_new)
    s = s.at[:, :, N_SP].set(phi0[:, :, 0])
    return s


# ── Simulation ─────────────────────────────────────────────────────────────

def simulate_nsp_pde_2d(A, b, phi_bulk, phi_ic,
                         t_out_days,
                         Nx=20, Nz=40,
                         D=D_DEFAULT, u=U_DEFAULT,
                         dt=0.02, n_nsp_per_transport=5):
    """
    Args:
        phi_ic : (Nx, Nz, N_SP)  or  (Nz, N_SP)  (broadcast over x)
        phi_bulk : (N_SP,)
    Returns:
        phi_out : (T, Nx, Nz, N_SP)
    """
    if phi_ic.ndim == 2:   # (Nz, N_SP) → broadcast
        phi_ic = np.tile(phi_ic[None, :, :], (Nx, 1, 1))

    dx = 1.0 / (Nx - 1) if Nx > 1 else 1.0
    dz = 1.0 / (Nz - 1)

    # CFL check
    cfl_limit = 0.5 / (float(D.max()) * (1.0/dx**2 + 1.0/dz**2))
    if dt > cfl_limit:
        print(f'  [warn] dt={dt:.4f} > CFL limit {cfl_limit:.4f}; clipping')
        dt = 0.9 * cfl_limit

    state = state_from_phi_2d(jnp.array(phi_ic, dtype=jnp.float64))
    react = make_nsp_reaction(A, b)

    t_curr    = 0.0
    t_end     = t_out_days[-1]
    snapshots = {}

    def _snap(t, st):
        for tgt in t_out_days:
            if abs(t - tgt) < dt * 0.6 and tgt not in snapshots:
                snapshots[tgt] = np.array(extract_phi_2d(st))

    _snap(t_curr, state)

    while t_curr < t_end - 1e-9:
        for _ in range(n_nsp_per_transport):
            state = react(state)
        phi = extract_phi_2d(state)
        phi = transport_2d_jit(phi, D, u, dx, dz, dt, phi_bulk)
        state = inject_phi_2d(state, phi)
        t_curr += dt
        _snap(t_curr, state)

    phi_out = np.stack([snapshots.get(t, np.full((Nx, Nz, N_SP), np.nan))
                        for t in t_out_days], axis=0)
    return phi_out   # (T, Nx, Nz, N_SP)


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_species_snapshots(phi_out, t_days, cond_tag, out_path=None):
    """φᵢ(x,z) colourmap at last timepoint, one panel per species."""
    T = len(t_days)
    t_idx = T - 1
    fig, axes = plt.subplots(1, N_SP, figsize=(2.6 * N_SP, 3.0))
    for s, ax in enumerate(axes):
        im = ax.imshow(
            phi_out[t_idx, :, :, s].T,
            origin='lower', aspect='auto',
            cmap='YlOrRd', vmin=0, vmax=phi_out[t_idx, :, :, s].max() or 1e-3,
        )
        ax.set_title(SHORT[s], fontsize=9, color=COLORS[s])
        ax.set_xlabel('x (lateral)', fontsize=8)
        ax.set_ylabel('z (depth)',   fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'NSP 2D — {cond_tag}  Day {t_days[t_idx]:.0f}', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_depth_mean(phi_out, t_days, cond_tag, out_path=None):
    """x-averaged φᵢ(z,t) per species — comparable to 1D output."""
    phi_mean = phi_out.mean(axis=1)   # (T, Nz, N_SP)
    Nz = phi_mean.shape[1]
    z  = np.linspace(0, 1, Nz)
    fig, axes = plt.subplots(1, N_SP, figsize=(2.2 * N_SP, 3.2), sharey=True)
    for s, ax in enumerate(axes):
        for k, day in enumerate(t_days):
            alpha = 0.3 + 0.7 * k / max(len(t_days) - 1, 1)
            ax.plot(phi_mean[k, :, s], z, color=COLORS[s], alpha=alpha,
                    label=f'Day {day:.0f}')
        ax.set_title(SHORT[s], fontsize=9, color=COLORS[s])
        ax.set_xlabel('⟨φᵢ⟩_x', fontsize=8)
        if s == 0:
            ax.set_ylabel('z (depth)', fontsize=8)
    axes[-1].legend(fontsize=6, loc='lower right')
    fig.suptitle(f'NSP 2D x-mean depth profiles — {cond_tag}', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_bulk_vs_obs(phi_out, phi_obs, t_days, cond_tag, out_path=None):
    """Bulk (z=L, x-mean) vs observed composition."""
    phi_bulk_pred = phi_out[:, :, -1, :].mean(axis=1)   # (T, N_SP)
    fig, axes = plt.subplots(1, N_SP, figsize=(2.2 * N_SP, 3.2))
    for s, ax in enumerate(axes):
        ax.plot(t_days, phi_bulk_pred[:, s], color=COLORS[s], lw=2, label='2D bulk')
        ax.plot(DAYS[:len(phi_obs)], phi_obs[:, s], 'o--', color=COLORS[s],
                alpha=0.6, label='Observed')
        ax.set_title(SHORT[s], fontsize=9)
        ax.set_xlabel('Day')
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('Relative abundance')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'NSP 2D bulk vs observed — {cond_tag}', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NSP 2D PDE — Heine 2025')
    parser.add_argument('--cond',    choices=['CS','CH','DS','DH','all'], default='all')
    parser.add_argument('--Nx',      type=int,   default=20)
    parser.add_argument('--Nz',      type=int,   default=40)
    parser.add_argument('--dt',      type=float, default=0.02)
    parser.add_argument('--n_nsp',   type=int,   default=5)
    parser.add_argument('--D_scale', type=float, default=1.0)
    parser.add_argument('--u',       type=float, default=U_DEFAULT)
    args = parser.parse_args()

    conds = [c for c in CONDITIONS if args.cond == 'all' or c[2] == args.cond]
    D     = D_DEFAULT * args.D_scale

    for condition, cultivation, tag in conds:
        print(f'\n=== {tag} ({condition} {cultivation}) ===')
        A, b     = load_A_b(tag)
        phi_obs  = load_obs(condition, cultivation)
        phi_bulk = jnp.array(phi_obs[0], dtype=jnp.float64)

        # IC: uniform = Day-1 bulk at every (x,z) node
        phi_ic = np.tile(phi_obs[0], (args.Nx, args.Nz, 1))

        t_out = np.array(DAYS, dtype=float)
        print(f'  Nx={args.Nx}, Nz={args.Nz}, dt={args.dt}d, n_nsp={args.n_nsp}')

        phi_out = simulate_nsp_pde_2d(
            A, b, phi_bulk, phi_ic, t_out,
            Nx=args.Nx, Nz=args.Nz,
            D=D, u=args.u, dt=args.dt,
            n_nsp_per_transport=args.n_nsp,
        )
        print(f'  phi_out.shape={phi_out.shape}')

        stem = f'{tag}_Nx{args.Nx}_Nz{args.Nz}_dt{args.dt}'
        plot_species_snapshots(phi_out, t_out, tag,
                               out_path=OUT_DIR / f'snapshot_{stem}.png')
        plot_depth_mean(phi_out, t_out, tag,
                        out_path=OUT_DIR / f'depth_mean_{stem}.png')
        plot_bulk_vs_obs(phi_out, phi_obs, t_out, tag,
                         out_path=OUT_DIR / f'bulk_vs_obs_{stem}.png')
        np.save(OUT_DIR / f'phi_pde_{stem}.npy', phi_out)

    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
