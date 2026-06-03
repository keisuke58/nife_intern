#!/usr/bin/env python3
"""
pinn_3d_inverse.py — 3-D Physics-Informed Neural Network (PINN) for the Heine
2025 5-species spatial reaction-diffusion INVERSE problem, extending the 1-D
PINN (scripts/pde/pinn_diffusion_inverse.py) from (z,t) to the full (x,y,z,t)
voxel field.

  Species order (canonical, 5-sp):  So, An, Vd/Vp, Fn, Pg.

Forward PDE (strong form), diagonal-diffusion default:

    ∂φ_i/∂t = D_i (∂_xx + ∂_yy + ∂_zz) φ_i − u ∂_z φ_i + R_i(φ)     i = 0..4

With --crossdiff the diagonal Laplacian transport is replaced by the
volume-filling / size-exclusion cross-diffusion divergence (same flux as
nsp_pde_1d_heine_xdiff.py, generalised to 3 dimensions):

    J_i = −D_i [ (1 − ρ) ∇φ_i + φ_i ∇ρ ],     ρ = Σ_j φ_j
    ∂φ_i/∂t = −∇·J_i − u ∂_z φ_i + R_i(φ)

so species couple through the SHARED crowding gradient ∇ρ, yet only the five
self-mobilities D_i are free (no extra parameters — cross-diffusion reuses the
same D_i).  All spatial derivatives are taken by autodiff (jax.jacrev / grad)
on the network, with the (x,y,z,t) → physical chain-rule scale factors carried
through.

Reaction term R_i(φ)  (smooth gLV / replicator form, autodiff-friendly,
identical to the 1-D PINN):

    g_i = b_i + (A φ)_i ;  R_i = φ_i (g_i − φ·g)

with A, b from nsp_pde_1d_heine.load_A_b(cond) (gLV Heine fit), guarded by a
fit_glv_heine.json fallback via load_A_b_np().

INVERSE problem: the diffusivities D_i (5, as log_D) and the advection u (scalar)
are TRAINABLE leaves, inferred jointly with the network weights θ by minimising

    L = w_data · MSE_data(φ_θ, φ_obs)  +  w_phys · MSE_residual(collocation)

Data (3-D supervision, from scripts/analysis/fish_3d_batch.py):
    results/fish_3d/ic/phi_xyz_{cond}_d{day}.npy   cond∈{CH,DH}, day∈{1,6,10,15,21}
    each (X,Y,Z,5) of RAW decoded FISH intensity (downsampled).  Per voxel we
    normalise to compositional fractions φ_i = raw_i / Σ_j raw_j and keep an
    occupancy = Σ_j raw_j / (99th-percentile of Σ over all voxels), clipped to
    [0,1].  The supervised target is φ_i · occupancy, so Σ_i target_i ≤ 1 like
    the PDE state (the PDE void fraction is 1 − ρ).  Billions of voxels exist in
    total, so a fixed --n-data random voxels are subsampled per (cond,day).
    Coordinates are normalised to [0,1]: x=x/Lx, y=y/Ly, z=z/Lz, t=day/21.

⚠ ILLUSTRATIVE / METHODOLOGY DEMONSTRATION ⚠
The spatial voxel field is abundant but there are only 5 timepoints and few
replicates, so the inferred D_i and u are NOT quantitatively trustworthy — they
are a *proof of concept* that the 3-D inverse PINN executes and is well-posed.
With 5 days the temporal derivative (and hence absolute D/u scaling) is weakly
constrained; treat the numbers as illustrative only.  Do not over-claim.

Optimiser:  optax.adam if available, else the hand-rolled JAX Adam from the 1-D
file (this env has no optax, so the fallback is exercised).

CLI:
    python scripts/pde/pinn_3d_inverse.py --cond DH --epochs 4000
    python scripts/pde/pinn_3d_inverse.py --cond CH --crossdiff
    python scripts/pde/pinn_3d_inverse.py --cond DH --smoke           # quick self-test
    python scripts/pde/pinn_3d_inverse.py --cond DH --crossdiff --smoke

Outputs (results/pinn_3d/):
    pinn3d_D_fit_<COND>[_xdiff].json   inferred D_i, u, final losses, hparams
    pinn3d_fit_<COND>[_xdiff].png      predicted vs observed xy & xz mean-projections
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ── Optional JAX (graceful skip for ast-only environments) ──────────────────
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _HAVE_JAX = True
except Exception as _e:  # pragma: no cover - exercised only without JAX
    _HAVE_JAX = False
    _JAX_ERR = repr(_e)

# ── Optional optax (hand-rolled Adam fallback otherwise) ────────────────────
try:
    import optax  # type: ignore
    _HAVE_OPTAX = True
except Exception:
    _HAVE_OPTAX = False

# Reuse the 1-D PINN's reaction term, A/b loader, MLP init and Adam fallback.
from pinn_diffusion_inverse import (  # noqa: E402
    load_A_b_np, reaction, init_mlp_params, adam_init, adam_update,
)

HERE = Path(__file__).resolve().parents[2]          # repo root
DATA_DIR = HERE / 'results' / 'fish_3d' / 'ic'
OUT_DIR = HERE / 'results' / 'pinn_3d'

N_SP = 5
SHORT = ['S.o', 'A.n', 'Vd/Vp', 'F.n', 'P.g']
COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
DAYS_ALL = [1, 6, 10, 15, 21]
T_MAX = 21.0          # normalisation: t = day / 21


# ────────────────────────────────────────────────────────────────────────────
# Data loading — 3-D RAW FISH voxels → normalised (x,y,z,t,φ) training points.
# ────────────────────────────────────────────────────────────────────────────

def _normalise_voxels(raw):
    """RAW (...,5) intensities → (φ·occupancy) (...,5), Σ ≤ 1, plus occupancy.

    φ_i = raw_i / Σ_j raw_j  (compositional fraction; uniform where empty)
    occ = Σ_j raw_j / percentile99(Σ)  clipped to [0,1]
    target = φ_i · occ   ⇒  Σ_i target_i = occ ≤ 1  (void fraction = 1 − occ).
    """
    raw = np.maximum(raw, 0.0)
    tot = raw.sum(axis=-1)                                    # (...)
    p99 = np.percentile(tot[tot > 0], 99) if np.any(tot > 0) else 1.0
    p99 = max(float(p99), 1e-9)
    occ = np.clip(tot / p99, 0.0, 1.0)                       # (...)
    frac = np.where(tot[..., None] > 1e-9, raw / np.maximum(tot[..., None], 1e-9),
                    np.full_like(raw, 1.0 / N_SP))
    target = frac * occ[..., None]                          # (...,5), Σ ≤ 1
    return target.astype(np.float64), occ.astype(np.float64)


def load_voxels_3d(cond, days=DAYS_ALL, n_data=20000, seed=0, data_dir=DATA_DIR):
    """Load + subsample the 3-D FISH voxels for one condition.

    Returns a dict with stacked, normalised supervision points across all days:
        x,y,z,t : (Npts,) coordinates in [0,1]
        phi     : (Npts, 5) = φ·occupancy  (Σ ≤ 1)
        per_day : {day: {'phi':(X,Y,Z,5), 'occ':(X,Y,Z), 'shape':(X,Y,Z)}} for plots
        days, Lx, Ly, Lz_by_day
    """
    rng = np.random.default_rng(seed)
    xs, ys, zs, ts, phis = [], [], [], [], []
    per_day = {}
    Lx = Ly = None
    Lz_by_day = {}
    used_days = []
    for d in days:
        p = Path(data_dir) / f'phi_xyz_{cond}_d{d}.npy'
        if not p.exists():
            print(f'  [skip] missing {p.name}')
            continue
        raw = np.load(p)                                     # (X,Y,Z,5) raw
        X, Y, Z, _ = raw.shape
        Lx, Ly = X, Y
        Lz_by_day[d] = Z
        used_days.append(d)
        target, occ = _normalise_voxels(raw)                # (X,Y,Z,5), (X,Y,Z)

        # store a downsampled full grid for plotting (mean projections)
        per_day[d] = {'phi': target, 'occ': occ, 'shape': (X, Y, Z)}

        # subsample n_data random voxels for the data loss
        n_vox = X * Y * Z
        n_take = min(n_data, n_vox)
        flat = rng.choice(n_vox, size=n_take, replace=False)
        ix, iy, iz = np.unravel_index(flat, (X, Y, Z))
        # normalised coordinates in [0,1]  (Z normalised per-day by that day's depth)
        xs.append(ix / max(X - 1, 1))
        ys.append(iy / max(Y - 1, 1))
        zs.append(iz / max(Z - 1, 1))
        ts.append(np.full(n_take, d / T_MAX))
        phis.append(target[ix, iy, iz, :])

    if not used_days:
        raise FileNotFoundError(f'No phi_xyz_{cond}_d*.npy found in {data_dir}')

    return {
        'x': np.concatenate(xs), 'y': np.concatenate(ys),
        'z': np.concatenate(zs), 't': np.concatenate(ts),
        'phi': np.concatenate(phis, axis=0),
        'per_day': per_day, 'days': used_days,
        'Lx': Lx, 'Ly': Ly, 'Lz_by_day': Lz_by_day, 'cond': cond,
    }


def make_synthetic_3d(cond, n_per_day=12, days=(1, 6, 10, 15, 21), seed=0):
    """Tiny smooth synthetic 3-D dataset for the --smoke self-test (no .npy needed).

    Builds a small (n,n,n,5) grid per day so the plotting path also exercises.
    """
    rng = np.random.default_rng(seed)
    n = 6
    xs, ys, zs, ts, phis = [], [], [], [], []
    per_day = {}
    for d in days:
        gx = np.linspace(0, 1, n)
        XX, YY, ZZ = np.meshgrid(gx, gx, gx, indexing='ij')
        base = np.stack([
            0.5 - 0.3 * ZZ,
            0.2 + 0.1 * ZZ,
            0.1 + 0.05 * XX,
            0.1 + 0.2 * ZZ * (d / T_MAX),
            0.1 + 0.3 * ZZ * (d / T_MAX),
        ], axis=-1)
        base = np.maximum(base + 0.01 * rng.standard_normal(base.shape), 1e-3)
        occ = 0.6 + 0.3 * ZZ                                 # smooth occupancy < 1
        frac = base / base.sum(axis=-1, keepdims=True)
        target = frac * occ[..., None]
        per_day[d] = {'phi': target, 'occ': occ, 'shape': (n, n, n)}
        # subsample
        idx = rng.choice(n * n * n, size=min(n_per_day, n * n * n), replace=False)
        ix, iy, iz = np.unravel_index(idx, (n, n, n))
        xs.append(ix / (n - 1)); ys.append(iy / (n - 1)); zs.append(iz / (n - 1))
        ts.append(np.full(len(idx), d / T_MAX))
        phis.append(target[ix, iy, iz, :])
    return {
        'x': np.concatenate(xs), 'y': np.concatenate(ys),
        'z': np.concatenate(zs), 't': np.concatenate(ts),
        'phi': np.concatenate(phis, axis=0),
        'per_day': per_day, 'days': list(days),
        'Lx': n, 'Ly': n, 'Lz_by_day': {d: n for d in days}, 'cond': cond,
    }


# ────────────────────────────────────────────────────────────────────────────
# PINN network: φ_θ(x,y,z,t): R⁴ → R⁵ on/under the simplex.
# ────────────────────────────────────────────────────────────────────────────

def mlp_forward(params, xyzt):
    """MLP with tanh hidden activations. xyzt: (4,) -> raw logits (N_SP+1,)."""
    h = xyzt
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def phi_net(params, x, y, z, t):
    """φ_θ(x,y,z,t): scalars -> (5,) UNDER the simplex (Σφ ≤ 1).

    Use softmax over N_SP+1 logits (the extra logit = void fraction φ0) and drop
    φ0, so the 5 returned species fractions are non-negative and sum to ≤ 1 —
    consistent with the PDE state where ρ = Σφ_i ≤ 1 and 1−ρ is the void.
    """
    logits = mlp_forward(params, jnp.array([x, y, z, t]))     # (N_SP+1,)
    full = jax.nn.softmax(logits)                            # (N_SP+1,) sums to 1
    return full[:N_SP]                                       # drop void → Σ ≤ 1


# ── PDE residual at one collocation point ───────────────────────────────────
#   All derivatives are taken w.r.t. the NORMALISED coords (x,y,z,t ∈ [0,1]) by
#   autodiff (jax.jacrev grad-of-grad for the 2nd derivatives), and the physical
#   operators carry the chain-rule scale factors (sx,sy,sz,st):
#       ∂/∂t_phys   = (1/st) ∂/∂t_norm
#       ∂²/∂x_phys² = (1/sx²) ∂²/∂x_norm²   (and y, z)
#   D_i and u are therefore reported in NORMALISED units.

def pde_residual_diag(params, x, y, z, t, log_D, u, A, b, scales):
    """Diagonal-diffusion residual at (x,y,z,t):

        ∂_t φ − D (∂_xx+∂_yy+∂_zz) φ + u ∂_z φ − R(φ) = 0
    """
    sx, sy, sz, st = scales
    phi = phi_net(params, x, y, z, t)
    # per-axis 2nd derivatives via grad-of-grad (each (5,))
    d_xx = jax.jacrev(jax.jacrev(lambda a: phi_net(params, a, y, z, t)))(x)
    d_yy = jax.jacrev(jax.jacrev(lambda bb: phi_net(params, x, bb, z, t)))(y)
    d_zz = jax.jacrev(jax.jacrev(lambda c: phi_net(params, x, y, c, t)))(z)
    lap = d_xx / sx**2 + d_yy / sy**2 + d_zz / sz**2
    d_z = jax.jacrev(lambda c: phi_net(params, x, y, c, t))(z)
    d_t = jax.jacrev(lambda dd: phi_net(params, x, y, z, dd))(t)
    D = jnp.exp(log_D)
    R = reaction(phi, A, b)
    res = (d_t / st) - D * lap + u * (d_z / sz) - R
    return res


def pde_residual_xdiff(params, x, y, z, t, log_D, u, A, b, scales):
    """Volume-filling cross-diffusion residual at (x,y,z,t).

    ∂_t φ_i = −∇·J_i − u ∂_z φ_i + R_i ,
    J_i = −D_i [ (1−ρ) ∇φ_i + φ_i ∇ρ ],  ρ = Σ_j φ_j

    ⇒ −∇·J_i = D_i ∇·[ (1−ρ) ∇φ_i + φ_i ∇ρ ]
             = D_i [ (1−ρ) Δφ_i  − ∇ρ·∇φ_i  +  ∇φ_i·∇ρ  +  φ_i Δρ ]
             = D_i [ (1−ρ) Δφ_i  +  φ_i Δρ ]
    (the two cross terms ∓∇ρ·∇φ_i cancel).  Δ = ∂_xx+∂_yy+∂_zz, ρ = Σφ, and
    Δρ = Σ_j Δφ_j.  Same D_i as the diagonal model (no extra parameters).
    Per-axis scale factors are carried through (sx,sy,sz,st).
    """
    sx, sy, sz, st = scales
    f = lambda a, bb, c: phi_net(params, a, bb, c, t)        # noqa: E731 (fix t)
    phi = f(x, y, z)
    # per-axis 2nd derivatives of φ (each (5,))
    d_xx = jax.jacrev(jax.jacrev(lambda a: f(a, y, z)))(x)
    d_yy = jax.jacrev(jax.jacrev(lambda bb: f(x, bb, z)))(y)
    d_zz = jax.jacrev(jax.jacrev(lambda c: f(x, y, c)))(z)
    lap_phi = d_xx / sx**2 + d_yy / sy**2 + d_zz / sz**2     # Δφ_i  (5,)
    rho = jnp.sum(phi)                                       # scalar
    lap_rho = jnp.sum(lap_phi)                               # Δρ = Σ Δφ_i (scalar)

    # advection / time derivatives (1st order)
    d_z = jax.jacrev(lambda c: f(x, y, c))(z)               # (5,)
    d_t = jax.jacrev(lambda dd: phi_net(params, x, y, z, dd))(t)  # (5,)

    D = jnp.exp(log_D)
    div_negJ = D * ((1.0 - rho) * lap_phi + phi * lap_rho)   # −∇·J_i  (5,)
    R = reaction(phi, A, b)
    res = (d_t / st) - div_negJ + u * (d_z / sz) - R
    return res


# ────────────────────────────────────────────────────────────────────────────
# Losses
# ────────────────────────────────────────────────────────────────────────────

def make_loss(A, b, scales, w_data, w_phys, crossdiff):
    A = jnp.asarray(A); b = jnp.asarray(b)
    resid = pde_residual_xdiff if crossdiff else pde_residual_diag

    batched_phi = jax.vmap(phi_net, in_axes=(None, 0, 0, 0, 0))
    batched_res = jax.vmap(
        resid,
        in_axes=(None, 0, 0, 0, 0, None, None, None, None, None))

    def loss_fn(opt_params, dx, dy, dz, dt_, dphi, cx, cy, cz, ct):
        net = opt_params['net']
        log_D = opt_params['log_D']
        u = opt_params['u']

        phi_pred = batched_phi(net, dx, dy, dz, dt_)         # (Nd, 5)
        data_loss = jnp.mean((phi_pred - dphi) ** 2)

        res = batched_res(net, cx, cy, cz, ct, log_D, u, A, b, scales)
        phys_loss = jnp.mean(res ** 2)

        total = w_data * data_loss + w_phys * phys_loss
        return total, (data_loss, phys_loss)

    return loss_fn


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def train(data, A, b, *, epochs, lr, hidden=64, depth=4, n_collocation=512,
          w_data=1.0, w_phys=1.0, seed=0, crossdiff=False, D_init=None,
          u_init=0.005, batch_data=4096, log_every=None):
    """Train the 3-D PINN; returns (params_dict, history)."""
    key = jax.random.PRNGKey(seed)
    key, kinit = jax.random.split(key)

    # 4 inputs (x,y,z,t) -> N_SP+1 logits (softmax incl. void fraction)
    layer_sizes = [4] + [hidden] * depth + [N_SP + 1]
    net = init_mlp_params(layer_sizes, kinit)

    if D_init is None:
        D_init = np.array([0.015, 0.008, 0.012, 0.012, 0.005])  # nsp D_DEFAULT
    opt_params = {
        'net': net,
        'log_D': jnp.log(jnp.asarray(D_init, dtype=jnp.float64)),
        'u': jnp.asarray(float(u_init), dtype=jnp.float64),
    }

    scales = (1.0, 1.0, 1.0, 1.0)   # x,y,z,t all normalised to [0,1]
    loss_fn = make_loss(A, b, scales, w_data, w_phys, crossdiff)
    val_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    all_x = jnp.asarray(data['x']); all_y = jnp.asarray(data['y'])
    all_z = jnp.asarray(data['z']); all_t = jnp.asarray(data['t'])
    all_phi = jnp.asarray(data['phi'])
    n_data = all_x.shape[0]
    batch_data = min(batch_data, n_data)

    use_optax = _HAVE_OPTAX
    if use_optax:
        opt = optax.adam(lr)
        opt_state = opt.init(opt_params)
    else:
        opt_state = adam_init(opt_params)

    @jax.jit
    def step(opt_params, opt_state, dx, dy, dz, dt_, dphi, cx, cy, cz, ct):
        (total, aux), grads = val_and_grad(
            opt_params, dx, dy, dz, dt_, dphi, cx, cy, cz, ct)
        if use_optax:
            updates, opt_state2 = opt.update(grads, opt_state, opt_params)
            opt_params2 = optax.apply_updates(opt_params, updates)
        else:
            opt_params2, opt_state2 = adam_update(opt_params, grads, opt_state, lr)
        return opt_params2, opt_state2, total, aux

    history = {'total': [], 'data': [], 'phys': []}
    if log_every is None:
        log_every = max(1, epochs // 10)

    days_norm = jnp.asarray([d / T_MAX for d in data['days']])

    for ep in range(epochs):
        key, kc, kd, ks = jax.random.split(key, 4)
        # fresh interior collocation points (x,y,z uniform; t snapped to data days
        # so the residual is informative where the data lives — only 5 days)
        col = jax.random.uniform(kc, (n_collocation, 3), dtype=jnp.float64)
        cx, cy, cz = col[:, 0], col[:, 1], col[:, 2]
        t_idx = jax.random.randint(ks, (n_collocation,), 0, days_norm.shape[0])
        ct = days_norm[t_idx]

        # minibatch of supervision voxels
        if batch_data < n_data:
            bidx = jax.random.randint(kd, (batch_data,), 0, n_data)
            dx, dy, dz, dt_, dphi = (all_x[bidx], all_y[bidx], all_z[bidx],
                                     all_t[bidx], all_phi[bidx])
        else:
            dx, dy, dz, dt_, dphi = all_x, all_y, all_z, all_t, all_phi

        opt_params, opt_state, total, aux = step(
            opt_params, opt_state, dx, dy, dz, dt_, dphi, cx, cy, cz, ct)
        dloss, ploss = aux
        history['total'].append(float(total))
        history['data'].append(float(dloss))
        history['phys'].append(float(ploss))

        if ep % log_every == 0 or ep == epochs - 1:
            print(f'  epoch {ep:5d}  total={float(total):.3e}  '
                  f'data={float(dloss):.3e}  phys={float(ploss):.3e}  '
                  f'D={np.exp(np.asarray(opt_params["log_D"])).round(4)}  '
                  f'u={float(opt_params["u"]):.4f}')

    return opt_params, history


# ────────────────────────────────────────────────────────────────────────────
# Plotting — predicted vs observed xy and xz mean-projections per species.
# ────────────────────────────────────────────────────────────────────────────

def _pred_grid(net, day, nx, ny, nz):
    """Predict φ on an (nx,ny,nz,5) grid at the given (normalised) day."""
    gx = np.linspace(0, 1, nx); gy = np.linspace(0, 1, ny); gz = np.linspace(0, 1, nz)
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing='ij')
    pts_x = jnp.asarray(XX.ravel()); pts_y = jnp.asarray(YY.ravel())
    pts_z = jnp.asarray(ZZ.ravel())
    pts_t = jnp.full(pts_x.shape, day / T_MAX)
    pred = jax.vmap(phi_net, in_axes=(None, 0, 0, 0, 0))(net, pts_x, pts_y, pts_z, pts_t)
    return np.asarray(pred).reshape(nx, ny, nz, N_SP)


def plot_fit(opt_params, data, out_path, crossdiff):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 8,
    })
    import matplotlib.pyplot as plt

    days = data['days']
    net = opt_params['net']
    # use the last available day for the spatial projection comparison
    day = days[-1]
    obs = data['per_day'][day]['phi']                       # (X,Y,Z,5)
    X, Y, Z = data['per_day'][day]['shape']
    # cap the prediction grid resolution for speed
    nx, ny, nz = min(X, 32), min(Y, 32), max(min(Z, 16), 2)
    pred = _pred_grid(net, day, nx, ny, nz)

    obs_xy = obs.mean(axis=2)    # (X,Y,5)
    obs_xz = obs.mean(axis=1)    # (X,Z,5)
    pred_xy = pred.mean(axis=2)  # (nx,ny,5)
    pred_xz = pred.mean(axis=1)  # (nx,nz,5)

    # 4 rows (obs-xy, pred-xy, obs-xz, pred-xz) × N_SP cols
    fig, axes = plt.subplots(4, N_SP, figsize=(2.0 * N_SP, 7.5), squeeze=False)
    row_data = [(obs_xy, 'obs xy'), (pred_xy, 'PINN xy'),
                (obs_xz, 'obs xz'), (pred_xz, 'PINN xz')]
    for r, (img, lab) in enumerate(row_data):
        for s in range(N_SP):
            ax = axes[r][s]
            im = ax.imshow(img[:, :, s].T, origin='lower', aspect='auto',
                           cmap='viridis', vmin=0, vmax=max(1e-6, float(img[:, :, s].max())))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(SHORT[s], color=COLORS[s], fontsize=9)
            if s == 0:
                ax.set_ylabel(lab, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    D = np.exp(np.asarray(opt_params['log_D']))
    u = float(opt_params['u'])
    dstr = '  '.join(f'D[{SHORT[i]}]={D[i]:.4f}' for i in range(N_SP))
    mode = 'cross-diffusion' if crossdiff else 'diagonal'
    fig.suptitle(f"3-D PINN inverse fit — {data['cond']} ({mode}), day {day}\n"
                 f"{dstr}  u={u:.4f}   [ILLUSTRATIVE — 5 timepoints only]",
                 fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def run(args):
    if not _HAVE_JAX:
        raise RuntimeError(f'JAX not importable: {_JAX_ERR}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cond = args.cond

    A, b = load_A_b_np(cond)
    print(f'Condition {cond}: A diag={A.diagonal().round(3)} b={b.round(3)}')

    if args.smoke:
        print('SMOKE: synthetic 3-D data, tiny net, few epochs')
        data = make_synthetic_3d(cond)
        epochs = args.epochs if args.epochs else 8
        hidden, depth, ncol = 12, 2, 48
        batch_data = 64
    else:
        data = load_voxels_3d(cond, n_data=args.n_data, seed=args.seed)
        epochs = args.epochs if args.epochs else 4000
        hidden, depth, ncol = args.hidden, args.depth, args.collocation
        batch_data = args.batch_data

    print(f'Data: {len(data["x"])} voxels, days={data["days"]}, '
          f'Lx={data["Lx"]} Ly={data["Ly"]} Lz(by day)={data["Lz_by_day"]}')
    print(f'  crossdiff={args.crossdiff}  '
          f'phi range=[{data["phi"].min():.3f},{data["phi"].max():.3f}]  '
          f'Σphi max={data["phi"].sum(axis=1).max():.3f}')

    opt_params, history = train(
        data, A, b, epochs=epochs, lr=args.lr,
        hidden=hidden, depth=depth, n_collocation=ncol,
        w_data=args.w_data, w_phys=args.w_phys, seed=args.seed,
        crossdiff=args.crossdiff, batch_data=batch_data,
    )

    D_fit = np.exp(np.asarray(opt_params['log_D'])).tolist()
    u_fit = float(opt_params['u'])

    # sanity: predicted φ in [0,1] and Σφ ≤ 1 on a small probe grid
    probe = _pred_grid(opt_params['net'], data['days'][-1], 6, 6, 4)
    phi_min, phi_max = float(probe.min()), float(probe.max())
    sum_max = float(probe.sum(axis=-1).max())

    suffix = '_xdiff' if args.crossdiff else ''
    if args.smoke:
        suffix += '_smoke'

    result = {
        'cond': cond,
        'crossdiff': bool(args.crossdiff),
        'D_fit': D_fit,
        'u_fit': u_fit,
        'D_species': SHORT,
        'final_loss_total': history['total'][-1],
        'final_loss_data': history['data'][-1],
        'final_loss_phys': history['phys'][-1],
        'init_loss_total': history['total'][0],
        'epochs': epochs,
        'optimizer': 'optax.adam' if _HAVE_OPTAX else 'handrolled_jax_adam',
        'phi_probe_min': phi_min,
        'phi_probe_max': phi_max,
        'phi_probe_sum_max': sum_max,
        'hparams': {
            'lr': args.lr, 'hidden': hidden, 'depth': depth,
            'collocation': ncol, 'w_data': args.w_data, 'w_phys': args.w_phys,
            'n_data': args.n_data, 'batch_data': batch_data, 'seed': args.seed,
        },
        'days': data['days'],
        'units': 'D_i, u in normalised units (x,y,z in [0,1], t = day/21)',
        'note': ('ILLUSTRATIVE / methodology demonstration only. Spatial voxels '
                 'are abundant but there are only 5 timepoints and few '
                 'replicates, so absolute D_i and u are weakly identifiable and '
                 'should NOT be over-interpreted quantitatively.'),
        'smoke': bool(args.smoke),
    }
    out_json = OUT_DIR / f'pinn3d_D_fit_{cond}{suffix}.json'
    out_json.write_text(json.dumps(result, indent=2))
    print(f'Saved: {out_json}')
    print(f'Inferred D = {np.round(D_fit, 5)}  u = {u_fit:.5f}')
    print(f'Probe φ ∈ [{phi_min:.3f},{phi_max:.3f}]  Σφ max={sum_max:.3f}')

    if not args.no_plot:
        plot_fit(opt_params, data, OUT_DIR / f'pinn3d_fit_{cond}{suffix}.png',
                 args.crossdiff)

    return result


def main():
    p = argparse.ArgumentParser(
        description='3-D PINN inverse fit of D_i, u for the Heine 5-species '
                    'spatial reaction-diffusion PDE (ILLUSTRATIVE).')
    p.add_argument('--cond', choices=['CH', 'DH'], default='DH')
    p.add_argument('--epochs', type=int, default=0,
                   help='training epochs (0 -> sensible default per mode)')
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--w-data', dest='w_data', type=float, default=1.0)
    p.add_argument('--w-phys', dest='w_phys', type=float, default=1.0)
    p.add_argument('--crossdiff', action='store_true',
                   help='use volume-filling cross-diffusion transport (same D_i)')
    p.add_argument('--n-data', dest='n_data', type=int, default=20000,
                   help='# random voxels subsampled per (cond,day) for data loss')
    p.add_argument('--batch-data', dest='batch_data', type=int, default=4096,
                   help='# supervision voxels per training step (minibatch)')
    p.add_argument('--collocation', type=int, default=512,
                   help='# PDE-residual collocation points per epoch')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--depth', type=int, default=4, help='# hidden layers')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='quick end-to-end self-test on tiny synthetic 3-D data')
    args = p.parse_args()

    if not _HAVE_JAX:
        print(f'[pinn3d] JAX unavailable ({_JAX_ERR}); cannot run. '
              f'Code is import-clean; run on the GPU host with JAX installed.')
        return
    run(args)


if __name__ == '__main__':
    main()
