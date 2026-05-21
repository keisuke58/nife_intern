#!/usr/bin/env python3
"""
nsp_pde_1d_heine.py — Spatial NSP (Hamilton) PDE for Heine 2025 5-species HOBIC data.

PDE (operator-split, strong form, Method of Lines):
    ∂φᵢ/∂t = Dᵢ ∂²φᵢ/∂z² − u ∂φᵢ/∂z  +  NSP_reaction(φᵢ, ψᵢ, φ₀, γ)
    ∂ψᵢ/∂t =                                NSP_reaction(φᵢ, ψᵢ, φ₀, γ)
    ∂φ₀/∂t =                                NSP_reaction(φ₀, ...)
    γ enforces Σᵢ φᵢ + φ₀ = 1  (local constraint at each z)

    NSP state per node: g(z) = [φ₁..φₙ, φ₀, ψ₁..ψₙ, γ]  (dim 2N+2 = 12)

Operator splitting (Lie):
    1. Reaction step : NSP implicit-Euler Newton at each z-node (jax.vmap)
    2. Transport step: Dᵢ∂²φᵢ/∂z² − u∂φᵢ/∂z  (explicit FD, φ only)

Boundary conditions:
    z = 0 (substratum): no-flux  ∂φᵢ/∂z = 0
    z = L (bulk):       Dirichlet φᵢ = φ_bulk,ᵢ  (day-1 median)

Species order:
    0: S. oralis   1: A. naeslundii   2: V. dispar/parvula
    3: F. nucleatum  4: P. gingivalis

Usage:
    python nsp_pde_1d_heine.py                      # all 4 conditions
    python nsp_pde_1d_heine.py --cond DH            # Dysbiotic HOBIC only
    python nsp_pde_1d_heine.py --Nz 40 --dt 0.05   # finer grid
    python nsp_pde_1d_heine.py --gradient_ic        # dysbiotic gradient IC

Note: D_i and u are placeholder values pending CLSM z-profile data
(Heine et al. 2025 HOBIC flow chamber, data request to Szafrański lab).
"""

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
import pandas as pd

# ── JAX imports ────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# NSP core from Tmcmc202601
NSP_PATH = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
sys.path.insert(0, str(NSP_PATH))
from hamilton_ode_jax_nsp import _residual, _newton_step, _make_initial_state

# ── Paths ──────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
GLV_FIT  = HERE / 'results' / 'heine2025' / 'fit_glv_heine.json'
DATA_CSV = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data'
                '/fig3_species_distribution_replicates.csv')
OUT_DIR  = HERE / 'results' / 'nsp_pde'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
N_SP   = 5
N_STATE = 2 * N_SP + 2   # 12
DAYS   = [1, 3, 6, 10, 15, 21]

SHORT  = ['S.o', 'A.n', 'Vd/Vp', 'F.n', 'P.g']
COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#d62728']

CONDITIONS = [
    ('Commensal', 'Static',  'CS'),
    ('Commensal', 'HOBIC',   'CH'),
    ('Dysbiotic', 'Static',  'DS'),
    ('Dysbiotic', 'HOBIC',   'DH'),
]
SPECIES_MAP = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar', 'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula', 'F. nucleatum', 'P. gingivalis_W83'],
}

# ── NSP model constants (same as 0D LOO fits) ─────────────────────────────
NSP_PARAMS_BASE = {
    "n_sp":             N_SP,
    "dt_h":             1e-3,            # NSP time step (internal)
    "Kp1":              1e-4,
    "Eta":              jnp.ones(N_SP,  dtype=jnp.float64),
    "EtaPhi":           jnp.ones(N_SP,  dtype=jnp.float64),
    "c":                25.0,
    "alpha":            100.0,
    "K_hill":           jnp.array(0.0,  dtype=jnp.float64),
    "n_hill":           jnp.array(4.0,  dtype=jnp.float64),
    "hill_gate_species":jnp.array([-1, 0], dtype=jnp.int32),
    "active_mask":      jnp.ones(N_SP, dtype=jnp.int64),
}

# ── Placeholder diffusivities (normalised: D_eff / L²  ×  t_day) ──────────
# L ~ 100 µm, t_day = 1 day
# D_eff ~ 0.01–0.3 × D_water(bacteria) ~ 0.01–0.1 µm²/s
# → D_norm ~ 0.001 – 0.05
D_DEFAULT = jnp.array([0.015, 0.008, 0.012, 0.012, 0.005], dtype=jnp.float64)
#                        So     An     Vd     Fn     Pg
U_DEFAULT = 0.005   # advection toward bulk (normalised)


# ── Data loading ───────────────────────────────────────────────────────────

def load_obs(condition: str, cultivation: str) -> np.ndarray:
    """Load median φᵢ(t) from CSV. Returns (6, 5) array (days × species)."""
    df  = pd.read_csv(DATA_CSV)
    sp  = SPECIES_MAP[condition]
    sub = df[(df['condition'] == condition) & (df['cultivation'] == cultivation)]
    obs = np.zeros((len(DAYS), N_SP))
    for t, day in enumerate(DAYS):
        row = sub[sub['day'] == day]
        for s, name in enumerate(sp):
            vals = row[row['species'] == name]['distribution_pct'].values
            obs[t, s] = np.median(vals) if len(vals) else 0.0
    # Normalise rows to sum = 1
    sums = obs.sum(axis=1, keepdims=True)
    return np.where(sums > 1e-12, obs / sums, obs)


def load_A_b(cond_tag: str):
    """Load fitted A (5×5) and b (5) from gLV Heine fit for given condition tag."""
    d = json.loads(GLV_FIT.read_text())
    A = jnp.array(d[cond_tag]['A'], dtype=jnp.float64)
    b = jnp.array(d[cond_tag]['b'], dtype=jnp.float64)
    return A, b


# ── NSP per-node reaction (vmapped over z-axis) ───────────────────────────

def make_nsp_step(A, b):
    """Return a vmappable single-node NSP Newton step closure."""
    params = {
        **NSP_PARAMS_BASE,
        "A":      A,
        "b_diag": b,
    }

    def step_one(g_prev):
        return _newton_step(g_prev, params)

    return jax.jit(jax.vmap(step_one))   # maps over leading (Nz,) axis


# ── Transport operator (explicit FD on φᵢ only) ───────────────────────────

def transport_step(phi_Nz_N, D, u, dz, dt, phi_bulk):
    """
    Apply one explicit FD transport step to phi_Nz_N: (Nz, N_SP).
    Central diff for diffusion, upwind for advection.
    BC: no-flux at z=0, Dirichlet at z=L.
    """
    phi = phi_Nz_N
    Nz  = phi.shape[0]
    dp  = jnp.zeros_like(phi)

    # diffusion interior: d²φ/dz² ≈ (φ[i+1]-2φ[i]+φ[i-1]) / dz²
    d2  = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / dz**2
    dp  = dp.at[1:-1, :].add(D[None, :] * d2)

    # no-flux at z=0 (ghost node = φ[1])
    d2_0 = 2 * (phi[1, :] - phi[0, :]) / dz**2
    dp   = dp.at[0, :].add(D * d2_0)

    # advection upwind (u ≥ 0 → left side)
    adv  = u * (phi[1:-1, :] - phi[:-2, :]) / dz
    dp   = dp.at[1:-1, :].add(-adv)

    phi_new = phi + dt * dp
    phi_new = jnp.clip(phi_new, 0.0, 1.0)

    # Dirichlet BC at z=L
    phi_new = phi_new.at[-1, :].set(phi_bulk)
    return phi_new


transport_step_jit = jax.jit(transport_step,
                              static_argnames=['dz', 'dt'])


# ── Extract / inject φ from NSP state ─────────────────────────────────────

def extract_phi(state_Nz):
    """state_Nz: (Nz, N_STATE) → φᵢ (Nz, N_SP)"""
    return state_Nz[:, :N_SP]


def inject_phi(state_Nz, phi_new):
    """Replace φᵢ in state_Nz with phi_new; renormalise φ₀ = 1 - Σφᵢ."""
    phi_sum = jnp.sum(phi_new, axis=-1, keepdims=True)
    phi0    = jnp.clip(1.0 - phi_sum, 1e-10, 1.0 - 1e-10)
    new_state = state_Nz.at[:, :N_SP].set(phi_new)
    new_state = new_state.at[:, N_SP].set(phi0[:, 0])
    return new_state


# ── Main simulation ────────────────────────────────────────────────────────

def simulate_nsp_pde(A, b, phi_bulk, phi_ic, t_out_days,
                     Nz=40, D=D_DEFAULT, u=U_DEFAULT, dt=0.05,
                     n_nsp_per_transport=5):
    """
    Run NSP PDE simulation.

    Args:
        A, b       : interaction matrix and growth rates (5×5, 5)
        phi_bulk   : Dirichlet BC at z=L  (5,)
        phi_ic     : initial φᵢ profile   (Nz, 5)
        t_out_days : times to snapshot (must start at 0 if from uniform IC)
        Nz         : spatial nodes
        D          : diffusivities (5,)
        u          : advection velocity (scalar)
        dt         : transport time step in days
        n_nsp_per_transport : NSP reaction steps per transport step

    Returns:
        phi_out : (len(t_out_days), Nz, N_SP)
    """
    L   = 1.0
    dz  = L / (Nz - 1)
    active_mask = jnp.ones(N_SP, dtype=jnp.int64)

    # Build initial NSP state (Nz, 12)
    def init_node(phi_i):
        return _make_initial_state(phi_i, N_SP, active_mask, psi_init=0.999)

    state = jax.vmap(init_node)(jnp.array(phi_ic, dtype=jnp.float64))  # (Nz, 12)

    nsp_vmap = make_nsp_step(A, b)

    t_curr  = 0.0
    t_end   = t_out_days[-1]
    snapshots = {}

    def _snap_if_close(t, state_arr):
        for tgt in t_out_days:
            if abs(t - tgt) < dt * 0.6 and tgt not in snapshots:
                snapshots[tgt] = np.array(extract_phi(state_arr))

    _snap_if_close(t_curr, state)

    while t_curr < t_end - 1e-9:
        # ── reaction: n_nsp_per_transport Newton steps ──
        for _ in range(n_nsp_per_transport):
            state = nsp_vmap(state)

        # ── transport: explicit FD on φ ──
        phi = extract_phi(state)
        phi = transport_step_jit(phi, D, u, dz, dt, phi_bulk)
        state = inject_phi(state, phi)

        t_curr += dt
        _snap_if_close(t_curr, state)

    phi_out = np.stack([snapshots.get(t, np.full((Nz, N_SP), np.nan))
                        for t in t_out_days], axis=0)
    return phi_out


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_depth_profiles(phi_out, t_days, z_grid, cond_tag, out_path=None):
    """Stacked area profiles φᵢ(z) at each time point."""
    T   = len(t_days)
    fig, axes = plt.subplots(1, T, figsize=(2.5 * T, 3.6), sharey=True)
    for k, ax in enumerate(np.array(axes).ravel()):
        phi_t  = phi_out[k]
        bottom = np.zeros(len(z_grid))
        for s in range(N_SP):
            ax.fill_betweenx(z_grid, bottom, bottom + phi_t[:, s],
                             color=COLORS[s], alpha=0.85)
            bottom += phi_t[:, s]
        ax.set_xlim(0, 1)
        ax.set_ylim(z_grid[0], z_grid[-1])
        ax.set_title(f'Day {t_days[k]:.0f}', fontsize=9)
        ax.set_xlabel('Abundance')
        if k == 0:
            ax.set_ylabel('z (depth)')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[s], alpha=0.85) for s in range(N_SP)]
    fig.legend(handles, SHORT, loc='lower center', ncol=N_SP, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'NSP PDE — {cond_tag} depth profiles', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_heatmaps(phi_out, t_days, z_grid, cond_tag, out_path=None):
    """φᵢ(z,t) heatmap per species."""
    fig, axes = plt.subplots(1, N_SP, figsize=(2.4 * N_SP, 3.0))
    for s, ax in enumerate(axes):
        im = ax.imshow(
            phi_out[:, :, s].T, aspect='auto', origin='lower',
            extent=[t_days[0], t_days[-1], z_grid[0], z_grid[-1]],
            cmap='YlOrRd', vmin=0,
        )
        ax.set_title(SHORT[s], fontsize=9, color=COLORS[s])
        ax.set_xlabel('t (days)', fontsize=8)
        ax.set_ylabel('z', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle(f'NSP PDE $\\phi_i(z,t)$ — {cond_tag}', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_obs_vs_bulk(phi_out, phi_obs, t_days, cond_tag, out_path=None):
    """
    Overlay: NSP bulk (z=L) prediction vs observed mean composition.
    """
    fig, axes = plt.subplots(1, N_SP, figsize=(2.2 * N_SP, 3.2), sharey=False)
    for s, ax in enumerate(axes):
        ax.plot(t_days, phi_out[:, -1, s], color=COLORS[s], lw=2,
                label='PDE bulk (z=L)')
        ax.plot(DAYS[:len(phi_obs)], phi_obs[:, s], 'o--', color=COLORS[s],
                alpha=0.6, label='Observed mean')
        ax.set_title(SHORT[s], fontsize=9)
        ax.set_xlabel('Day')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, t_days[-1])
    axes[0].set_ylabel('Relative abundance')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'NSP PDE bulk vs observed — {cond_tag}', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NSP spatial PDE for Heine 2025')
    parser.add_argument('--cond',    choices=['CS','CH','DS','DH','all'], default='all')
    parser.add_argument('--Nz',      type=int,   default=40)
    parser.add_argument('--dt',      type=float, default=0.1,
                        help='Transport time step in days (default 0.1)')
    parser.add_argument('--n_nsp',   type=int,   default=5,
                        help='NSP Newton steps per transport step')
    parser.add_argument('--D_scale', type=float, default=1.0)
    parser.add_argument('--u',       type=float, default=U_DEFAULT)
    parser.add_argument('--gradient_ic', action='store_true',
                        help='Linear gradient IC: z=0 uses Day-21 composition')
    args = parser.parse_args()

    conds = [c for c in CONDITIONS if args.cond == 'all' or c[2] == args.cond]

    for condition, cultivation, tag in conds:
        print(f'\n=== {tag} ({condition} {cultivation}) ===')
        A, b   = load_A_b(tag)
        phi_obs = load_obs(condition, cultivation)
        phi_bulk = jnp.array(phi_obs[0], dtype=jnp.float64)   # Day 1 median as bulk BC

        Nz = args.Nz
        z_grid = np.linspace(0, 1, Nz)
        D = D_DEFAULT * args.D_scale

        # Initial condition
        if args.gradient_ic:
            phi_sub = phi_obs[-1]   # Day-21 (most differentiated)
            frac = np.linspace(0, 1, Nz)[:, None]
            phi_ic = (1 - frac) * phi_sub[None, :] + frac * phi_obs[0][None, :]
        else:
            phi_ic = np.tile(phi_obs[0], (Nz, 1))   # uniform = Day-1 bulk

        t_out = np.array(DAYS, dtype=float)

        print(f'  Nz={Nz}, dt={args.dt}d, n_nsp={args.n_nsp}, D_scale={args.D_scale}')
        print(f'  A_diag={np.array(A).diagonal().round(3)},  b={np.array(b).round(3)}')

        phi_out = simulate_nsp_pde(
            A, b, phi_bulk, phi_ic, t_out,
            Nz=Nz, D=D, u=args.u, dt=args.dt,
            n_nsp_per_transport=args.n_nsp,
        )
        print(f'  Simulation done. phi_out.shape={phi_out.shape}')

        stem = f'{tag}_Nz{Nz}_dt{args.dt}_Dscale{args.D_scale}'
        plot_depth_profiles(phi_out, t_out, z_grid, tag,
                            out_path=OUT_DIR / f'depth_{stem}.png')
        plot_heatmaps(phi_out, t_out, z_grid, tag,
                      out_path=OUT_DIR / f'heatmap_{stem}.png')
        plot_obs_vs_bulk(phi_out, phi_obs, t_out, tag,
                         out_path=OUT_DIR / f'bulk_vs_obs_{stem}.png')

        np.save(OUT_DIR / f'phi_pde_{stem}.npy', phi_out)
        meta = {
            'condition': condition, 'cultivation': cultivation, 'cond_tag': tag,
            'Nz': Nz, 'dt': args.dt, 'n_nsp': args.n_nsp,
            'D_scale': args.D_scale, 'u': args.u,
            'D_values': np.array(D).tolist(),
            'phi_bulk': np.array(phi_bulk).tolist(),
            'note': 'D_i placeholder — fit from CLSM z-profiles (Heine et al. 2025)',
        }
        (OUT_DIR / f'meta_{stem}.json').write_text(json.dumps(meta, indent=2))

    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
