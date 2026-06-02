#!/usr/bin/env python3
"""
glv_pde_1d.py — gLV reaction-diffusion PDE along biofilm depth axis z.

PDE (strong form, Method of Lines):
    ∂φᵢ/∂t = Dᵢ ∂²φᵢ/∂z² − u ∂φᵢ/∂z + φᵢ(bᵢ + Σⱼ Aᵢⱼ φⱼ)

Boundary conditions:
    z = 0 (substratum): no-flux  ∂φᵢ/∂z = 0
    z = L (bulk):       Dirichlet φᵢ = φ_bulk,ᵢ  (bulk composition from Week 1 mean)

Spatial grid: uniform finite differences, Nz nodes, dz = L / (Nz-1)
Diffusion: central differences (2nd order)
Advection: upwind (1st order, u ≥ 0 = advection toward bulk)

Usage:
    python glv_pde_1d.py                          # simulate with default D_i
    python glv_pde_1d.py --fold 0                 # use LOO fold-0 A matrix
    python glv_pde_1d.py --D_scale 0.5            # halve all diffusivities
    python glv_pde_1d.py --show_time              # time-lapse animation frames

The A matrix is loaded from LOO-CV results (best model: gLV AGORA α=0.25).
D_i and u are placeholder values; replace with fitted values once CLSM data
are available (pending data request to Szafrański lab, Heine et al. 2025).
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import json
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
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parents[2]
LOO_DIR  = HERE / 'results' / 'dieckow_cr'
PHI_NPY  = HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy'
OUT_DIR  = HERE / 'results' / 'glv_pde'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Guild definitions (same order as guild_replicator_dieckow.py)
# ---------------------------------------------------------------------------
GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
GUILD_SHORT = [
    'Actin.', 'Bacil.', 'Bact.', 'β-Prot.',
    'Clost.', 'Corio.', 'Fusob.', 'γ-Prot.',
    'Negat.', 'Other',
]
GUILD_COLORS = [
    '#976832', '#20af53', '#ea2323', '#25b0e1',
    '#98c557', '#b69e7c', '#fac20b', '#156fb5',
    '#703a98', '#050608',
]
N_G = len(GUILD_ORDER)

# ---------------------------------------------------------------------------
# Placeholder diffusion parameters
# (to be fitted from CLSM z-profiles once data are available)
# Typical values for bacteria in oral biofilm EPS matrix:
#   D_eff ~ 0.01 – 0.3 × D_water  (Stewart 1998, Biotechnol. Bioeng.)
#   D_water (bacteria-sized object) ~ 0.1–1 µm²/s
#   Here: normalised to biofilm thickness L = 100 µm, t_scale = 1 day
#   → D_norm = D_eff / L² × t_scale
# ---------------------------------------------------------------------------
D_PLACEHOLDER = {
    # More motile / faster-growing guilds get higher D
    'Actinobacteria':    0.008,
    'Bacilli':           0.015,  # Streptococcus — coloniser, higher spread
    'Bacteroidia':       0.005,
    'Betaproteobacteria':0.010,
    'Clostridia':        0.004,
    'Coriobacteriia':    0.004,
    'Fusobacteriia':     0.012,  # Fusobacterium — bridge organism
    'Gammaproteobacteria':0.006,
    'Negativicutes':     0.010,  # Veillonella — metabolic cross-feeder
    'Other':             0.006,
}
D_DEFAULT = np.array([D_PLACEHOLDER[g] for g in GUILD_ORDER])

U_DEFAULT = 0.005   # advection velocity (toward bulk), normalised units


# ---------------------------------------------------------------------------
# Load A matrix from LOO fold
# ---------------------------------------------------------------------------
def load_A(fold: int = 0) -> np.ndarray:
    """Load gLV AGORA α=0.25 A matrix from LOO fold."""
    f = LOO_DIR / f'loo_glv_agora_a0p25_fold{fold}.json'
    if not f.exists():
        raise FileNotFoundError(f'LOO result not found: {f}')
    d = json.loads(f.read_text())
    return np.array(d['A'])


def load_fold_data(fold: int):
    """Return (A, b_train_mean, hold_idx) for a given LOO fold.

    b_train_mean: mean of b_ho from all other folds (= training patients' mean b).
    hold_idx: index of held-out patient in phi_obs.
    """
    bs_train = []
    hold_idx = None
    for k in range(10):
        f = LOO_DIR / f'loo_glv_agora_a0p25_fold{k}.json'
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if k == fold:
            hold_idx = d['hold_idx']
        else:
            bs_train.append(np.array(d['b_ho']))
    A = load_A(fold)
    b_train = np.mean(bs_train, axis=0) if bs_train else np.zeros(N_G)
    return A, b_train, hold_idx


# ---------------------------------------------------------------------------
# PDE right-hand side (Method of Lines)
# ---------------------------------------------------------------------------
def pde_rhs(t, state, A, b, D, u, dz, Nz, phi_bulk):
    """
    state : (Nz × N_G,) flattened  [row = z-node, col = guild]
    Returns d(state)/dt as same shape.

    Reaction term uses replicator form (consistent with fitted A, b values):
        reaction_i = φᵢ (fᵢ - <f>)
        fᵢ = bᵢ + Σⱼ Aᵢⱼ φⱼ
        <f> = Σᵢ φᵢ fᵢ
    This keeps Σᵢ φᵢ conserved at each z (in the absence of diffusion/advection).
    """
    phi = state.reshape(Nz, N_G)   # (Nz, N_G)

    # --- reaction: replicator form ---
    f     = b[None, :] + phi @ A.T          # (Nz, N_G)  fitness
    fmean = (phi * f).sum(axis=-1)           # (Nz,)      mean fitness
    rxn   = phi * (f - fmean[:, None])       # (Nz, N_G)

    dphi = np.zeros_like(phi)

    # --- diffusion: ∂²φ/∂z² via central differences ---
    # interior nodes
    diff_int = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / dz**2
    dphi[1:-1, :] += D[None, :] * diff_int

    # z=0 no-flux: ghost node φ[-1] = φ[1]  → d²φ/dz² ≈ 2(φ[1]-φ[0])/dz²
    dphi[0, :] += D * 2 * (phi[1, :] - phi[0, :]) / dz**2

    # z=L Dirichlet φ = phi_bulk: already fixed by BC — just set dphi=0 there
    dphi[-1, :] = 0.0

    # --- advection: −u ∂φ/∂z via upwind (u≥0 → upwind from left) ---
    # interior + bottom node
    dphi[1:-1, :] -= u * (phi[1:-1, :] - phi[:-2, :]) / dz
    dphi[0, :]    -= u * 0.0   # no-flux already handled

    # reaction
    dphi += rxn
    # enforce BC at bulk
    dphi[-1, :] = 0.0

    return dphi.ravel()


# ---------------------------------------------------------------------------
# Initial condition: uniform profile = bulk composition
# ---------------------------------------------------------------------------
def make_ic(phi_bulk: np.ndarray, Nz: int) -> np.ndarray:
    """Uniform initial profile equal to bulk composition."""
    return np.tile(phi_bulk, (Nz, 1)).ravel()


def make_ic_gradient(phi_substratum: np.ndarray, phi_bulk: np.ndarray,
                     Nz: int) -> np.ndarray:
    """Linear gradient from substratum to bulk."""
    z_frac = np.linspace(0, 1, Nz)[:, None]
    phi = (1 - z_frac) * phi_substratum[None, :] + z_frac * phi_bulk[None, :]
    return phi.ravel()


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
def simulate(A, b, D, u, phi_bulk, phi0_1d, t_days, Nz=60, L=1.0):
    """
    Run the gLV PDE from t=0 to t=t_days[-1].
    Returns phi_out: (len(t_days), Nz, N_G)
    """
    dz    = L / (Nz - 1)
    t_span = (0.0, t_days[-1])

    def rhs(t, y):
        return pde_rhs(t, y, A, b, D, u, dz, Nz, phi_bulk)

    sol = solve_ivp(
        rhs, t_span, phi0_1d,
        method='Radau',   # stiff solver — needed for diffusion + large A values
        t_eval=t_days,
        rtol=1e-4, atol=1e-6,
        dense_output=False,
        max_step=0.1,
    )
    if not sol.success:
        print(f'WARNING: solver message: {sol.message}')

    n_t_out = sol.y.shape[1]
    if n_t_out < len(t_days):
        print(f'WARNING: solver returned {n_t_out}/{len(t_days)} time points')
    phi_out = sol.y.T[:n_t_out].reshape(n_t_out, Nz, N_G)
    # clip negative and re-normalise at each (t, z)
    phi_out = np.clip(phi_out, 0, None)
    s = phi_out.sum(axis=-1, keepdims=True)
    phi_out = np.where(s > 1e-12, phi_out / s, phi_out)
    return phi_out, t_days[:n_t_out]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_depth_profiles(phi_out, t_days, z_grid, title='', out_path=None):
    """
    One panel per time point showing guild fractions vs depth z.
    phi_out: (T, Nz, N_G)
    """
    T = len(t_days)
    ncols = min(T, 6)
    nrows = (T + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.2 * nrows),
                             sharey=True, sharex=True)
    axes = np.array(axes).ravel()

    for k, ax in enumerate(axes):
        if k >= T:
            ax.set_visible(False)
            continue
        phi_t = phi_out[k]   # (Nz, N_G)
        bottom = np.zeros(len(z_grid))
        for g in range(N_G):
            ax.fill_betweenx(z_grid, bottom, bottom + phi_t[:, g],
                             color=GUILD_COLORS[g], alpha=0.85)
            bottom += phi_t[:, g]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, z_grid[-1])
        ax.set_title(f't = {t_days[k]:.0f} d', fontsize=9)
        ax.set_xlabel('Relative abundance')
        if k % ncols == 0:
            ax.set_ylabel('z (normalised)')

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=GUILD_COLORS[g], alpha=0.85)
               for g in range(N_G)]
    fig.legend(handles, GUILD_SHORT, loc='lower center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_guild_heatmaps(phi_out, t_days, z_grid, out_path=None):
    """
    Heatmap of φᵢ(z, t) for each guild.
    phi_out: (T, Nz, N_G)
    """
    ncols = 5
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows))
    axes = axes.ravel()

    for g, ax in enumerate(axes):
        if g >= N_G:
            ax.set_visible(False)
            continue
        # phi_g: (T, Nz)  → imshow needs (Nz, T) → transpose
        phi_g = phi_out[:, :, g].T   # (Nz, T)
        im = ax.imshow(
            phi_g, aspect='auto', origin='lower',
            extent=[t_days[0], t_days[-1], z_grid[0], z_grid[-1]],
            cmap='YlOrRd', vmin=0, vmax=phi_out[:, :, g].max() + 1e-6,
        )
        ax.set_title(GUILD_SHORT[g], fontsize=9, color=GUILD_COLORS[g])
        ax.set_xlabel('t (days)', fontsize=8)
        ax.set_ylabel('z', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(r'$\phi_i(z,\,t)$ — gLV PDE prediction', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_bulk_vs_wall(phi_out, t_days, out_path=None):
    """
    Compare time course at substratum (z=0) vs bulk (z=L).
    phi_out: (T, Nz, N_G)
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, zidx, zlabel in zip(axes, [0, -1], ['Substratum (z=0)', 'Bulk (z=L)']):
        for g in range(N_G):
            ax.plot(t_days, phi_out[:, zidx, g],
                    color=GUILD_COLORS[g], label=GUILD_SHORT[g], lw=1.8)
        ax.set_title(zlabel, fontsize=10)
        ax.set_xlabel('t (days)')
        ax.set_ylabel('Relative abundance')
        ax.set_ylim(0, 1)
        ax.set_xlim(t_days[0], t_days[-1])

    handles = [plt.Line2D([0], [0], color=GUILD_COLORS[g], lw=2)
               for g in range(N_G)]
    fig.legend(handles, GUILD_SHORT, loc='lower center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('gLV PDE — substratum vs bulk time course', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='gLV reaction-diffusion PDE simulator')
    parser.add_argument('--fold',    type=int,   default=0,
                        help='LOO fold index for A matrix (0-9)')
    parser.add_argument('--D_scale', type=float, default=1.0,
                        help='Scale factor applied to all D_i (default 1.0)')
    parser.add_argument('--u',       type=float, default=U_DEFAULT,
                        help='Advection velocity (default %.4f)' % U_DEFAULT)
    parser.add_argument('--Nz',      type=int,   default=60,
                        help='Number of spatial nodes (default 60)')
    parser.add_argument('--t_end',   type=float, default=21.0,
                        help='Simulation end time in days (default 21)')
    parser.add_argument('--n_frames',type=int,   default=7,
                        help='Number of output time frames (default 7)')
    parser.add_argument('--gradient_ic', action='store_true',
                        help='Use linear gradient IC (substratum = Dysbiotic composition)')
    args = parser.parse_args()

    # --- load fold-specific A, b, and held-out index ---
    A, b, hold_idx = load_fold_data(args.fold)
    print(f'Loaded fold {args.fold}: hold_idx={hold_idx}')

    # --- load phi_obs; exclude held-out patient for bulk BC ---
    phi_obs = np.load(PHI_NPY)   # (10, 3, 10)
    train_rows = [i for i in range(10) if i != hold_idx]
    phi_bulk = phi_obs[train_rows, 0, :].mean(axis=0)   # mean Week 1 of training patients
    phi_bulk = phi_bulk / phi_bulk.sum()

    # --- parameters ---
    D = D_DEFAULT * args.D_scale
    u = args.u
    Nz = args.Nz
    L = 1.0   # normalised biofilm thickness
    z_grid = np.linspace(0, L, Nz)

    t_days = np.linspace(0, args.t_end, args.n_frames)

    print(f'  D_scale={args.D_scale}, u={u:.4f}, Nz={Nz}, t_end={args.t_end} d')
    print(f'  Bulk composition (Week 1 mean): {dict(zip(GUILD_SHORT, phi_bulk.round(3)))}')

    # --- initial condition ---
    if args.gradient_ic:
        # substratum starts dysbiotic (patient A, week 3 = most dysbiotic)
        phi_sub = phi_obs[0, 2, :] / phi_obs[0, 2, :].sum()
        phi0 = make_ic_gradient(phi_sub, phi_bulk, Nz)
        ic_label = 'gradient IC (dysbiotic→bulk)'
    else:
        phi0 = make_ic(phi_bulk, Nz)
        ic_label = 'uniform IC (bulk)'

    print(f'  IC: {ic_label}')

    # --- simulate ---
    print('Running PDE simulation...', flush=True)
    phi_out, t_actual = simulate(A, b, D, u, phi_bulk, phi0, t_days, Nz=Nz, L=L)
    print('Done.')

    # --- plots ---
    tag = f'fold{args.fold}_Dscale{args.D_scale}_u{u:.3f}'

    plot_depth_profiles(
        phi_out, t_actual, z_grid,
        title=r'gLV PDE: $\phi_i(z,t)$ depth profiles',
        out_path=OUT_DIR / f'depth_profiles_{tag}.png',
    )
    plot_guild_heatmaps(
        phi_out, t_actual, z_grid,
        out_path=OUT_DIR / f'heatmaps_{tag}.png',
    )
    plot_bulk_vs_wall(
        phi_out, t_actual,
        out_path=OUT_DIR / f'bulk_vs_wall_{tag}.png',
    )

    # --- save snapshot data ---
    np.save(OUT_DIR / f'phi_pde_{tag}.npy', phi_out)
    meta = {
        'fold': args.fold, 'D_scale': args.D_scale, 'u': u,
        'Nz': Nz, 't_end': args.t_end, 'ic': ic_label,
        'D_values': D.tolist(), 'phi_bulk': phi_bulk.tolist(),
        'note': 'D_i placeholder — fit from CLSM z-profiles (Heine et al. 2025)',
    }
    (OUT_DIR / f'meta_{tag}.json').write_text(json.dumps(meta, indent=2))
    print(f'Outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
