#!/usr/bin/env python3
"""
pinn_diffusion_inverse.py — Physics-Informed Neural Network (PINN) for the 1D
reaction-diffusion INVERSE problem on Heine 2025 5-species depth profiles.

Forward PDE (strong form, same operator as nsp_pde_1d_heine.py):

    ∂φ_i/∂t = D_i ∂²φ_i/∂z² − u ∂φ_i/∂z + R_i(φ)      i = 0..4 (So,An,Vd/Vp,Fn,Pg)

Boundary conditions (encoded only weakly, via the data loss at the measured
z-extremes — z=0 substratum, z=L bulk — exactly where the FISH profiles live):
    z = 0  : substratum (no-flux is *softly* implied by the data + smoothness)
    z = L  : bulk Dirichlet (the day-1 bulk composition appears in the data)

Reaction term R_i(φ)  (smooth gLV / replicator form, autodiff-friendly):
    g_i  = b_i + (A φ)_i                       # per-capita gLV growth
    R_i  = φ_i (g_i − φ·g)                      # replicator mean-field subtraction
keeps Σ_i φ_i (+ slack) on the simplex and reuses the SAME fitted A, b that
nsp_pde_1d_heine.load_A_b(cond) returns from the gLV Heine fit.  (The full
Hamilton NSP reaction in hamilton_ode_jax_nsp.py is an implicit Newton solve and
is not a smooth pointwise R(φ) — unusable inside a PDE residual — so we use the
gLV replicator term it was fit alongside.)

INVERSE problem: the diffusivities D_i (5) and the advection u are TRAINABLE
parameters, inferred jointly with the network weights θ by minimising

    L = w_data · MSE_data(φ_θ, φ_obs)  +  w_phys · MSE_residual(collocation)

φ_θ(z,t) is a small MLP R² → R⁵ with a softmax-like normalisation so its outputs
stay on the species simplex.

Autodiff:  ∂_z, ∂_zz, ∂_t of φ_θ via jax.jacrev / jax.grad (forward-over-reverse).

Optimiser:  optax.adam if available, else a hand-rolled JAX Adam (this env has no
optax, so the fallback is exercised).

CLI:
    python scripts/pde/pinn_diffusion_inverse.py --cond DH --epochs 4000
    python scripts/pde/pinn_diffusion_inverse.py --cond CH --data results/diffusion_fit/zprofiles_all_ti.csv
    python scripts/pde/pinn_diffusion_inverse.py --smoke          # quick self-test

Outputs (results/pinn_diffusion/):
    pinn_D_fit_<COND>.json   inferred D_i, u, final losses, hyperparams
    pinn_fit_<COND>.png      PINN φ(z) vs observed FISH profiles per species/day
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
import json
import math
from functools import partial
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

HERE = Path(__file__).resolve().parents[2]   # repo root
DATA_DEFAULT = HERE / 'results' / 'diffusion_fit' / 'zprofiles_all_ti.csv'
OUT_DIR = HERE / 'results' / 'pinn_diffusion'

N_SP = 5
SHORT = ['S.o', 'A.n', 'Vd/Vp', 'F.n', 'P.g']
COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#d62728']
SPECIES_COLS = ['S.oralis', 'A.naeslundii', 'Vd/Vp', 'F.nucleatum', 'P.gingivalis']

# Physical normalisation: z normalised to [0,1] over the measured depth, t in days.
DAYS_ALL = [1, 6, 10, 15, 21]


# ────────────────────────────────────────────────────────────────────────────
# A, b loading — reuse the gLV Heine fit via the PDE module if available.
# ────────────────────────────────────────────────────────────────────────────

def load_A_b_np(cond: str):
    """Return (A 5×5, b 5) as numpy arrays for the condition tag.

    Prefer nsp_pde_1d_heine.load_A_b (the canonical loader). That import pulls
    in the Tmcmc NSP core; if it is unavailable (e.g. on a bare box) fall back
    to reading results/heine2025/fit_glv_heine.json directly.
    """
    try:
        from nsp_pde_1d_heine import load_A_b  # noqa: WPS433
        A, b = load_A_b(cond)
        return np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    except Exception:
        fit = HERE / 'results' / 'heine2025' / 'fit_glv_heine.json'
        d = json.loads(fit.read_text())
        return (np.asarray(d[cond]['A'], dtype=np.float64),
                np.asarray(d[cond]['b'], dtype=np.float64))


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────

def load_zprofiles(csv_path: Path, cond: str):
    """Load (z,t,φ) training points from the CLSM z-profile CSV.

    Columns: condition, day, z_um, S.oralis, A.naeslundii, Vd/Vp,
             F.nucleatum, P.gingivalis.

    Returns dict with:
        z   : (Npts,) normalised depth in [0,1]
        t   : (Npts,) normalised time in [0,1]
        phi : (Npts, 5) simplex-normalised abundances
        days, z_um_max, t_max  (for de-normalising / plotting)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df['condition'] == cond].copy()
    if df.empty:
        raise ValueError(f'No rows for condition {cond!r} in {csv_path}')

    days = sorted(df['day'].unique())
    z_max = float(df['z_um'].max())
    t_max = float(max(days))

    z = df['z_um'].to_numpy(dtype=np.float64) / z_max
    t = df['day'].to_numpy(dtype=np.float64) / t_max
    phi = df[SPECIES_COLS].to_numpy(dtype=np.float64)
    phi = np.maximum(phi, 0.0)
    rs = phi.sum(axis=1, keepdims=True)
    phi = np.where(rs > 1e-12, phi / rs, np.full_like(phi, 1.0 / N_SP))

    return {
        'z': z, 't': t, 'phi': phi,
        'days': days, 'z_max': z_max, 't_max': t_max, 'cond': cond,
    }


def make_synthetic(cond: str, n_z=20, days=(1, 6, 10, 15), seed=0):
    """Tiny smooth synthetic dataset for the --smoke self-test (no CSV needed)."""
    rng = np.random.default_rng(seed)
    z_max, t_max = 60.0, float(max(days))
    zg = np.linspace(0.0, 1.0, n_z)
    rows_z, rows_t, rows_phi = [], [], []
    for d in days:
        for zz in zg:
            # smooth depth/time-varying composition, then simplex-normalise
            base = np.array([
                0.5 - 0.3 * zz,
                0.2 + 0.1 * zz,
                0.1,
                0.1 + 0.2 * zz * (d / t_max),
                0.1 + 0.3 * zz * (d / t_max),
            ])
            base = np.maximum(base + 0.01 * rng.standard_normal(N_SP), 1e-3)
            base /= base.sum()
            rows_z.append(zz)
            rows_t.append(d / t_max)
            rows_phi.append(base)
    return {
        'z': np.array(rows_z), 't': np.array(rows_t), 'phi': np.array(rows_phi),
        'days': list(days), 'z_max': z_max, 't_max': t_max, 'cond': cond,
    }


# ────────────────────────────────────────────────────────────────────────────
# PINN network (small MLP) — all functions below need JAX.
# ────────────────────────────────────────────────────────────────────────────

def init_mlp_params(layer_sizes, key, scale=1.0):
    """Glorot-ish init for an MLP. Returns list of (W, b) tuples."""
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for (nin, nout), k in zip(zip(layer_sizes[:-1], layer_sizes[1:]), keys):
        std = scale * math.sqrt(2.0 / (nin + nout))
        W = std * jax.random.normal(k, (nin, nout), dtype=jnp.float64)
        b = jnp.zeros((nout,), dtype=jnp.float64)
        params.append((W, b))
    return params


def mlp_forward(params, zt):
    """MLP with tanh hidden activations. zt: (2,) -> raw logits (5,)."""
    h = zt
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def phi_net(params, z, t):
    """φ_θ(z,t): (scalar z, scalar t) -> (5,) on the simplex via softmax."""
    logits = mlp_forward(params, jnp.array([z, t]))
    return jax.nn.softmax(logits)


# ── Derivatives via autodiff ────────────────────────────────────────────────

def phi_and_derivs(params, z, t):
    """Return φ, ∂_t φ, ∂_z φ, ∂_zz φ at (z,t), each shape (5,)."""
    f = lambda zz, tt: phi_net(params, zz, tt)            # noqa: E731
    phi = f(z, t)
    dphi_dt = jax.jacrev(f, argnums=1)(z, t)              # (5,)
    dphi_dz = jax.jacrev(f, argnums=0)(z, t)              # (5,)
    d2phi_dz2 = jax.jacrev(jax.jacrev(f, argnums=0), argnums=0)(z, t)  # (5,)
    return phi, dphi_dt, dphi_dz, d2phi_dz2


# ── gLV / replicator reaction term R_i(φ) ───────────────────────────────────

def reaction(phi, A, b):
    """Smooth gLV replicator reaction (mean-field subtraction keeps Σφ on simplex).

    g_i = b_i + (Aφ)_i ;  R_i = φ_i (g_i − φ·g).
    """
    g = b + A @ phi
    mean_g = jnp.dot(phi, g)
    return phi * (g - mean_g)


# ── PDE residual at one collocation point ───────────────────────────────────

def pde_residual(params, z, t, log_D, u, A, b, z_scale, t_scale):
    """Residual of ∂_t φ − D ∂_zz φ + u ∂_z φ − R(φ) = 0 at (z,t).

    z,t are NORMALISED to [0,1]; derivatives are taken w.r.t. the normalised
    coordinates, so the physical operators carry the chain-rule scale factors:
        ∂/∂t_phys   = (1/t_scale) ∂/∂t_norm
        ∂²/∂z_phys² = (1/z_scale²) ∂²/∂z_norm²
    D_i and u absorb whatever units make the residual dimensionless here; they
    are reported in NORMALISED units (consistent with nsp_pde_1d_heine D_DEFAULT).
    """
    phi, dphi_dt, dphi_dz, d2phi_dz2 = phi_and_derivs(params, z, t)
    D = jnp.exp(log_D)                       # (5,) positive
    R = reaction(phi, A, b)
    res = (dphi_dt / t_scale) - D * (d2phi_dz2 / z_scale**2) \
        + u * (dphi_dz / z_scale) - R
    return res                               # (5,)


# ────────────────────────────────────────────────────────────────────────────
# Losses
# ────────────────────────────────────────────────────────────────────────────

def make_loss(A, b, z_scale, t_scale, w_data, w_phys):
    A = jnp.asarray(A); b = jnp.asarray(b)

    batched_phi = jax.vmap(phi_net, in_axes=(None, 0, 0))
    batched_res = jax.vmap(pde_residual,
                           in_axes=(None, 0, 0, None, None, None, None, None, None))

    def loss_fn(opt_params, data_z, data_t, data_phi, col_z, col_t):
        net = opt_params['net']
        log_D = opt_params['log_D']
        u = opt_params['u']

        # data loss
        phi_pred = batched_phi(net, data_z, data_t)          # (Nd, 5)
        data_loss = jnp.mean((phi_pred - data_phi) ** 2)

        # physics loss
        res = batched_res(net, col_z, col_t, log_D, u, A, b, z_scale, t_scale)
        phys_loss = jnp.mean(res ** 2)

        total = w_data * data_loss + w_phys * phys_loss
        return total, (data_loss, phys_loss)

    return loss_fn


# ────────────────────────────────────────────────────────────────────────────
# Hand-rolled Adam (used when optax is absent)
# ────────────────────────────────────────────────────────────────────────────

def adam_init(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {'m': m, 'v': v, 'step': jnp.array(0, dtype=jnp.int64)}


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    step = state['step'] + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1 - b1) * g,
                               state['m'], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1 - b2) * (g * g),
                               state['v'], grads)
    bc1 = 1 - b1 ** step
    bc2 = 1 - b2 ** step
    new_params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps),
        params, m, v)
    return new_params, {'m': m, 'v': v, 'step': step}


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def train(data, A, b, *, epochs, lr, hidden=64, depth=4, n_collocation=512,
          w_data=1.0, w_phys=1.0, seed=0, D_init=None, u_init=0.005,
          log_every=None):
    """Train the PINN; returns (params_dict, history)."""
    key = jax.random.PRNGKey(seed)
    key, kinit = jax.random.split(key)

    layer_sizes = [2] + [hidden] * depth + [N_SP]
    net = init_mlp_params(layer_sizes, kinit)

    if D_init is None:
        D_init = np.array([0.015, 0.008, 0.012, 0.012, 0.005])  # nsp D_DEFAULT
    opt_params = {
        'net': net,
        'log_D': jnp.log(jnp.asarray(D_init, dtype=jnp.float64)),
        'u': jnp.asarray(float(u_init), dtype=jnp.float64),
    }

    z_scale = 1.0   # z normalised to [0,1]
    t_scale = 1.0   # t normalised to [0,1]
    loss_fn = make_loss(A, b, z_scale, t_scale, w_data, w_phys)
    val_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    data_z = jnp.asarray(data['z']); data_t = jnp.asarray(data['t'])
    data_phi = jnp.asarray(data['phi'])

    use_optax = _HAVE_OPTAX
    if use_optax:
        opt = optax.adam(lr)
        opt_state = opt.init(opt_params)
    else:
        opt_state = adam_init(opt_params)

    @jax.jit
    def step(opt_params, opt_state, col_z, col_t):
        (total, aux), grads = val_and_grad(
            opt_params, data_z, data_t, data_phi, col_z, col_t)
        if use_optax:
            updates, opt_state2 = opt.update(grads, opt_state, opt_params)
            opt_params2 = optax.apply_updates(opt_params, updates)
        else:
            opt_params2, opt_state2 = adam_update(opt_params, grads, opt_state, lr)
        return opt_params2, opt_state2, total, aux

    history = {'total': [], 'data': [], 'phys': []}
    if log_every is None:
        log_every = max(1, epochs // 10)

    for ep in range(epochs):
        key, kc = jax.random.split(key)
        # fresh interior collocation points each epoch (z,t in [0,1])
        col = jax.random.uniform(kc, (n_collocation, 2), dtype=jnp.float64)
        col_z, col_t = col[:, 0], col[:, 1]

        opt_params, opt_state, total, aux = step(opt_params, opt_state, col_z, col_t)
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
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def plot_fit(opt_params, data, out_path):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 9,
    })
    import matplotlib.pyplot as plt

    days = data['days']
    t_max = data['t_max']
    z_grid = np.linspace(0.0, 1.0, 60)
    net = opt_params['net']
    pred = jax.vmap(lambda zz, tt: phi_net(net, zz, tt), in_axes=(0, None))

    T = len(days)
    fig, axes = plt.subplots(N_SP, T, figsize=(2.0 * T, 1.9 * N_SP),
                             sharex=True, sharey=True, squeeze=False)
    for s in range(N_SP):
        for k, day in enumerate(days):
            ax = axes[s][k]
            tn = day / t_max
            phi_pred = np.asarray(pred(jnp.asarray(z_grid), jnp.float64(tn)))
            ax.plot(phi_pred[:, s], z_grid, color=COLORS[s], lw=2, label='PINN')
            # observed points at this day
            mask = np.isclose(data['t'] * t_max, day)
            ax.plot(data['phi'][mask, s], data['z'][mask], 'o', color=COLORS[s],
                    ms=2.5, alpha=0.55, label='FISH')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if k == 0:
                ax.set_ylabel(f'{SHORT[s]}\nz/L', fontsize=7)
            if s == 0:
                ax.set_title(f'Day {day:.0f}', fontsize=8)
            if s == N_SP - 1:
                ax.set_xlabel(r'$\phi_i$', fontsize=7)
    axes[0][-1].legend(fontsize=6)
    D = np.exp(np.asarray(opt_params['log_D']))
    u = float(opt_params['u'])
    dstr = '  '.join(f'D[{SHORT[i]}]={D[i]:.4f}' for i in range(N_SP))
    fig.suptitle(f"PINN inverse fit — {data['cond']}\n{dstr}  u={u:.4f}",
                 fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
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
        print('SMOKE: synthetic data, few epochs')
        data = make_synthetic(cond)
        epochs = args.epochs if args.epochs else 8
        hidden, depth, ncol = 16, 2, 64
    else:
        data = load_zprofiles(Path(args.data), cond)
        epochs = args.epochs if args.epochs else 4000
        hidden, depth, ncol = args.hidden, args.depth, args.collocation

    print(f'Data: {len(data["z"])} points, days={data["days"]}, '
          f'z_max={data["z_max"]:.1f}um, t_max={data["t_max"]:.0f}d')

    opt_params, history = train(
        data, A, b, epochs=epochs, lr=args.lr,
        hidden=hidden, depth=depth, n_collocation=ncol,
        w_data=args.w_data, w_phys=args.w_phys, seed=args.seed,
    )

    D_fit = np.exp(np.asarray(opt_params['log_D'])).tolist()
    u_fit = float(opt_params['u'])
    result = {
        'cond': cond,
        'D_fit': D_fit,
        'u_fit': u_fit,
        'D_species': SHORT,
        'final_loss_total': history['total'][-1],
        'final_loss_data': history['data'][-1],
        'final_loss_phys': history['phys'][-1],
        'epochs': epochs,
        'optimizer': 'optax.adam' if _HAVE_OPTAX else 'handrolled_jax_adam',
        'hparams': {
            'lr': args.lr, 'hidden': hidden, 'depth': depth,
            'collocation': ncol, 'w_data': args.w_data, 'w_phys': args.w_phys,
            'seed': args.seed,
        },
        'units': 'D_i, u in normalised units (z/L in [0,1], t in days/t_max)',
        'smoke': bool(args.smoke),
    }
    suffix = '_smoke' if args.smoke else ''
    out_json = OUT_DIR / f'pinn_D_fit_{cond}{suffix}.json'
    out_json.write_text(json.dumps(result, indent=2))
    print(f'Saved: {out_json}')
    print(f'Inferred D = {np.round(D_fit, 5)}  u = {u_fit:.5f}')

    if not args.no_plot:
        plot_fit(opt_params, data, OUT_DIR / f'pinn_fit_{cond}{suffix}.png')

    return result


def main():
    p = argparse.ArgumentParser(description='PINN inverse fit of D_i, u for 1D RD PDE')
    p.add_argument('--cond', choices=['CH', 'DH', 'CS', 'DS'], default='DH')
    p.add_argument('--data', type=str, default=str(DATA_DEFAULT),
                   help='CLSM z-profile CSV (condition,day,z_um,<5 species>)')
    p.add_argument('--epochs', type=int, default=0,
                   help='training epochs (0 -> sensible default per mode)')
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--depth', type=int, default=4, help='# hidden layers')
    p.add_argument('--collocation', type=int, default=512,
                   help='# PDE-residual collocation points per epoch')
    p.add_argument('--w-data', dest='w_data', type=float, default=1.0)
    p.add_argument('--w-phys', dest='w_phys', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='quick end-to-end self-test on tiny synthetic data')
    args = p.parse_args()

    if not _HAVE_JAX:
        print(f'[pinn] JAX unavailable ({_JAX_ERR}); cannot run. '
              f'Code is import-clean; run on the GPU host with JAX installed.')
        return
    run(args)


if __name__ == '__main__':
    main()
