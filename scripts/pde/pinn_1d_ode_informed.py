#!/usr/bin/env python3
"""
pinn_1d_ode_informed.py — 1D (z,t) PINN inverse problem with ODE mean-field anchor.

Extends pinn_diffusion_inverse.py by adding a third loss term:

    L = w_data · MSE(φ_θ(z_k, t_k), φ_FISH(z_k, t_k))      [5 FISH timepoints]
      + w_ode  · MSE(⟨φ_θ⟩_z(t_j), φ_ODE(t_j))              [421 ODE timepoints]
      + w_phys · MSE(PDE residual at collocation pts)

⟨φ_θ⟩_z(t) = (1/N_z) Σ_q φ_θ(z_q, t)   with z_q = linspace(0,1,N_Z_QUAD)

This anchors the depth-averaged prediction to the gLV ODE continuous trajectory
(from heine_ode_continuous.py / results/heine2025/ode_continuous_<COND>.csv),
providing dense temporal supervision (421 pts) that the sparse FISH data alone
cannot supply. The ODE trajectory is itself derived from the MAP gLV fit (A, b),
so this is self-consistent: the PINN learns the spatial structure needed to make
the depth-averaged solution match the known ODE dynamics.

PDE:
    ∂φ_i/∂t = D_i ∂²φ_i/∂z² − u ∂φ_i/∂z + R_i(φ)

R_i = gLV replicator (same A, b from fit_glv_heine.json).

CLI:
    python scripts/pde/pinn_1d_ode_informed.py --cond DH --epochs 6000
    python scripts/pde/pinn_1d_ode_informed.py --cond CH --w-ode 5.0
    python scripts/pde/pinn_1d_ode_informed.py --smoke

Outputs (results/pinn_ode_informed/):
    pinn_ode_D_fit_<COND>.json
    pinn_ode_fit_<COND>.png
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import os as _os
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _HAVE_JAX = True
except Exception as _e:
    _HAVE_JAX = False
    _JAX_ERR = repr(_e)

try:
    import optax
    _HAVE_OPTAX = True
except Exception:
    _HAVE_OPTAX = False

HERE = Path(__file__).resolve().parents[2]
DATA_DEFAULT = HERE / "results" / "diffusion_fit" / "zprofiles_all_ti.csv"
ODE_DIR_DEFAULT = HERE / "results" / "heine2025"
OUT_DIR = HERE / "results" / "pinn_ode_informed"

N_SP = 5
SHORT = ["S.o", "A.n", "Vd/Vp", "F.n", "P.g"]
COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]
SPECIES_COLS = ["S.oralis", "A.naeslundii", "Vd/Vp", "F.nucleatum", "P.gingivalis"]

# ODE CSV species → PINN canonical index mapping
ODE_SPECIES_TO_IDX = {
    "S. oralis":            0,
    "A. naeslundii":        1,
    "V. dispar":            2,
    "V. parvula":           2,
    "F. nucleatum":         3,
    "P. gingivalis_20709":  4,
    "P. gingivalis_W83":    4,
}

DAYS_ALL = [1, 6, 10, 15, 21]


# ── A, b loading ─────────────────────────────────────────────────────────────

def load_A_b_np(cond: str):
    try:
        from nsp_pde_1d_heine import load_A_b
        A, b = load_A_b(cond)
        return np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    except Exception:
        fit = HERE / "results" / "heine2025" / "fit_glv_heine.json"
        d = json.loads(fit.read_text())
        return (np.asarray(d[cond]["A"], dtype=np.float64),
                np.asarray(d[cond]["b"], dtype=np.float64))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_zprofiles(csv_path: Path, cond: str):
    """Load FISH z-profile CSV → (z, t, phi) arrays, normalised to [0,1]."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["condition"] == cond].copy()
    if df.empty:
        raise ValueError(f"No rows for condition {cond!r} in {csv_path}")

    days = sorted(df["day"].unique())
    z_max = float(df["z_um"].max())
    t_max = float(max(days))

    z = df["z_um"].to_numpy(dtype=np.float64) / z_max
    t = df["day"].to_numpy(dtype=np.float64) / t_max
    phi = df[SPECIES_COLS].to_numpy(dtype=np.float64)
    phi = np.maximum(phi, 0.0)
    rs = phi.sum(axis=1, keepdims=True)
    phi = np.where(rs > 1e-12, phi / rs, np.full_like(phi, 1.0 / N_SP))

    return {"z": z, "t": t, "phi": phi,
            "days": days, "z_max": z_max, "t_max": t_max, "cond": cond}


def load_ode_trajectory(cond: str, ode_dir: Path, t_max: float,
                        n_subsample: int | None = None, seed: int = 0):
    """Load ODE continuous trajectory from ode_continuous_<COND>.csv.

    Returns:
        ode_t   : (N,) normalised time in [0,1]  (t_day / t_max)
        ode_phi : (N, 5) mean-field composition from the gLV ODE
    """
    import pandas as pd
    csv_path = ode_dir / f"ode_continuous_{cond}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ODE trajectory CSV not found: {csv_path}\n"
            f"Run: python scripts/analysis/heine_ode_continuous.py --conditions {cond}"
        )
    df = pd.read_csv(csv_path)

    # pivot long → wide: (t_day, species) → (N_t, 5)
    t_days_all = sorted(df["t_day"].unique())
    phi_wide = np.zeros((len(t_days_all), N_SP), dtype=np.float64)
    for j, sp in enumerate(df["species"].unique()):
        idx = ODE_SPECIES_TO_IDX.get(sp)
        if idx is None:
            continue
        sp_df = df[df["species"] == sp].set_index("t_day")["phi"]
        for k, td in enumerate(t_days_all):
            phi_wide[k, idx] = float(sp_df.get(td, 0.0))

    # simplex-normalise rows
    rs = phi_wide.sum(axis=1, keepdims=True)
    phi_wide = phi_wide / np.where(rs > 1e-12, rs, 1.0)

    ode_t = np.array(t_days_all, dtype=np.float64) / t_max
    ode_phi = phi_wide

    if n_subsample is not None and n_subsample < len(ode_t):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ode_t), size=n_subsample, replace=False)
        idx.sort()
        ode_t = ode_t[idx]
        ode_phi = ode_phi[idx]

    return ode_t, ode_phi


def make_synthetic(cond: str, n_z=20, days=(1, 6, 10, 15), seed=0):
    """Synthetic data for --smoke self-test."""
    rng = np.random.default_rng(seed)
    z_max, t_max = 60.0, float(max(days))
    zg = np.linspace(0.0, 1.0, n_z)
    rows_z, rows_t, rows_phi = [], [], []
    for d in days:
        for zz in zg:
            base = np.array([0.5 - 0.3 * zz, 0.2 + 0.1 * zz, 0.1,
                             0.1 + 0.2 * zz * (d / t_max),
                             0.1 + 0.3 * zz * (d / t_max)])
            base = np.maximum(base + 0.01 * rng.standard_normal(N_SP), 1e-3)
            base /= base.sum()
            rows_z.append(zz); rows_t.append(d / t_max); rows_phi.append(base)
    # synthetic ODE trajectory: uniform in z ⟹ mean = ODE value → just the FISH mean
    t_ode = np.linspace(0.0, 1.0, 40)
    phi_ode = np.tile(np.array([0.35, 0.15, 0.1, 0.2, 0.2]), (len(t_ode), 1))
    return (
        {"z": np.array(rows_z), "t": np.array(rows_t), "phi": np.array(rows_phi),
         "days": list(days), "z_max": z_max, "t_max": t_max, "cond": cond},
        t_ode, phi_ode,
    )


# ── Network ───────────────────────────────────────────────────────────────────

def init_mlp_params(layer_sizes, key, scale=1.0):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for (nin, nout), k in zip(zip(layer_sizes[:-1], layer_sizes[1:]), keys):
        std = scale * math.sqrt(2.0 / (nin + nout))
        W = std * jax.random.normal(k, (nin, nout), dtype=jnp.float64)
        b = jnp.zeros((nout,), dtype=jnp.float64)
        params.append((W, b))
    return params


def mlp_forward(params, zt):
    h = zt
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def phi_net(params, z, t):
    """φ_θ(z,t) → (5,) simplex via softmax."""
    return jax.nn.softmax(mlp_forward(params, jnp.array([z, t])))


# ── PDE derivatives & residual ────────────────────────────────────────────────

def phi_and_derivs(params, z, t):
    f = lambda zz, tt: phi_net(params, zz, tt)
    phi     = f(z, t)
    dphi_dt = jax.jacrev(f, argnums=1)(z, t)
    dphi_dz = jax.jacrev(f, argnums=0)(z, t)
    d2phi_dz2 = jax.jacrev(jax.jacrev(f, argnums=0), argnums=0)(z, t)
    return phi, dphi_dt, dphi_dz, d2phi_dz2


def reaction(phi, A, b):
    g = b + A @ phi
    return phi * (g - jnp.dot(phi, g))


def pde_residual(params, z, t, log_D, u, A, b, z_scale, t_scale):
    phi, dphi_dt, dphi_dz, d2phi_dz2 = phi_and_derivs(params, z, t)
    D = jnp.exp(log_D)
    R = reaction(phi, A, b)
    return (dphi_dt / t_scale) - D * (d2phi_dz2 / z_scale**2) \
         + u * (dphi_dz / z_scale) - R


# ── Loss with ODE anchor ──────────────────────────────────────────────────────

def make_loss(A, b, z_scale, t_scale, w_data, w_phys, w_ode, n_z_quad=32):
    A = jnp.asarray(A); b = jnp.asarray(b)
    z_quad = jnp.linspace(0.0, 1.0, n_z_quad, dtype=jnp.float64)

    batched_phi = jax.vmap(phi_net,      in_axes=(None, 0, 0))
    batched_res = jax.vmap(pde_residual, in_axes=(None, 0, 0, None, None, None, None, None, None))

    def ode_loss_at_t(net, t_norm, phi_ode_target):
        """MSE between depth-mean prediction and ODE target at one time point."""
        phi_z = jax.vmap(lambda z: phi_net(net, z, t_norm))(z_quad)  # (N_z_quad, 5)
        phi_mean = jnp.mean(phi_z, axis=0)                            # (5,)
        return jnp.mean((phi_mean - phi_ode_target) ** 2)

    def loss_fn(opt_params, data_z, data_t, data_phi, col_z, col_t, ode_t, ode_phi):
        net   = opt_params["net"]
        log_D = opt_params["log_D"]
        u     = opt_params["u"]

        # 1) FISH spatial loss
        phi_pred = batched_phi(net, data_z, data_t)
        data_loss = jnp.mean((phi_pred - data_phi) ** 2)

        # 2) PDE physics loss
        res = batched_res(net, col_z, col_t, log_D, u, A, b, z_scale, t_scale)
        phys_loss = jnp.mean(res ** 2)

        # 3) ODE mean-field anchor loss
        #    vmap over the N_ode sampled time points
        ode_losses = jax.vmap(lambda t_j, phi_j: ode_loss_at_t(net, t_j, phi_j))(
            ode_t, ode_phi)
        ode_loss = jnp.mean(ode_losses)

        total = w_data * data_loss + w_phys * phys_loss + w_ode * ode_loss
        return total, (data_loss, phys_loss, ode_loss)

    return loss_fn


# ── Adam fallback ─────────────────────────────────────────────────────────────

def adam_init(params):
    return {"m": jax.tree_util.tree_map(jnp.zeros_like, params),
            "v": jax.tree_util.tree_map(jnp.zeros_like, params),
            "step": jnp.array(0, dtype=jnp.int64)}


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    step = state["step"] + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1*mm + (1-b1)*g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2*vv + (1-b2)*(g*g), state["v"], grads)
    bc1 = 1 - b1**step;  bc2 = 1 - b2**step
    new_p = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr*(mm/bc1)/(jnp.sqrt(vv/bc2)+eps),
        params, m, v)
    return new_p, {"m": m, "v": v, "step": step}


# ── Training ──────────────────────────────────────────────────────────────────

def train(data, ode_t_full, ode_phi_full, A, b, *,
          epochs, lr, hidden=64, depth=4, n_collocation=512,
          w_data=1.0, w_phys=1.0, w_ode=5.0, n_z_quad=32,
          n_ode_per_epoch=64, seed=0, D_init=None, u_init=0.005,
          log_every=None):
    key = jax.random.PRNGKey(seed)
    key, kinit = jax.random.split(key)

    layer_sizes = [2] + [hidden] * depth + [N_SP]
    net = init_mlp_params(layer_sizes, kinit)

    if D_init is None:
        D_init = np.array([0.015, 0.008, 0.012, 0.012, 0.005])
    opt_params = {
        "net":   net,
        "log_D": jnp.log(jnp.asarray(D_init, dtype=jnp.float64)),
        "u":     jnp.asarray(float(u_init), dtype=jnp.float64),
    }

    loss_fn      = make_loss(A, b, 1.0, 1.0, w_data, w_phys, w_ode, n_z_quad)
    val_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    data_z   = jnp.asarray(data["z"])
    data_t   = jnp.asarray(data["t"])
    data_phi = jnp.asarray(data["phi"])

    use_optax = _HAVE_OPTAX
    if use_optax:
        opt = optax.adam(lr); opt_state = opt.init(opt_params)
    else:
        opt_state = adam_init(opt_params)

    @jax.jit
    def step(opt_params, opt_state, col_z, col_t, ode_t, ode_phi):
        (total, aux), grads = val_and_grad(
            opt_params, data_z, data_t, data_phi, col_z, col_t, ode_t, ode_phi)
        if use_optax:
            updates, opt_state2 = opt.update(grads, opt_state, opt_params)
            opt_params2 = optax.apply_updates(opt_params, updates)
        else:
            opt_params2, opt_state2 = adam_update(opt_params, grads, opt_state, lr)
        return opt_params2, opt_state2, total, aux

    history = {"total": [], "data": [], "phys": [], "ode": []}
    if log_every is None:
        log_every = max(1, epochs // 10)

    N_ode = len(ode_t_full)
    n_ode = min(n_ode_per_epoch, N_ode)

    for ep in range(epochs):
        key, kc, ko = jax.random.split(key, 3)

        # fresh PDE collocation points
        col = jax.random.uniform(kc, (n_collocation, 2), dtype=jnp.float64)
        col_z, col_t = col[:, 0], col[:, 1]

        # subsample ODE time points
        idx = np.sort(np.random.choice(N_ode, size=n_ode, replace=False))
        ode_t_ep   = jnp.asarray(ode_t_full[idx])
        ode_phi_ep = jnp.asarray(ode_phi_full[idx])

        opt_params, opt_state, total, aux = step(
            opt_params, opt_state, col_z, col_t, ode_t_ep, ode_phi_ep)
        dloss, ploss, oloss = aux
        history["total"].append(float(total)); history["data"].append(float(dloss))
        history["phys"].append(float(ploss));  history["ode"].append(float(oloss))

        if ep % log_every == 0 or ep == epochs - 1:
            D = np.exp(np.asarray(opt_params["log_D"])).round(4)
            print(f"  epoch {ep:5d}  total={float(total):.3e}  "
                  f"data={float(dloss):.3e}  phys={float(ploss):.3e}  "
                  f"ode={float(oloss):.3e}  "
                  f"D={D}  u={float(opt_params['u']):.4f}")

    return opt_params, history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_fit(opt_params, data, ode_t, ode_phi, out_path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9,
    })
    import matplotlib.pyplot as plt

    days    = data["days"]
    t_max   = data["t_max"]
    z_grid  = np.linspace(0.0, 1.0, 60)
    net     = opt_params["net"]
    pred_fn = jax.vmap(lambda zz, tt: phi_net(net, zz, tt), in_axes=(0, None))

    T = len(days)
    # left block: FISH spatial profiles (N_SP rows × T cols)
    # right block: time-course — depth-mean PINN vs ODE (N_SP rows × 1 col)
    fig, axes = plt.subplots(N_SP, T + 1,
                             figsize=(2.0 * (T + 1), 1.9 * N_SP),
                             squeeze=False)

    # de-normalise ODE time for plotting
    t_ode_days = ode_t * t_max

    for s in range(N_SP):
        # ── spatial panels (cols 0..T-1) ────────────────────────────────────
        for k, day in enumerate(days):
            ax = axes[s][k]
            tn = day / t_max
            phi_p = np.asarray(pred_fn(jnp.asarray(z_grid), jnp.float64(tn)))
            ax.plot(phi_p[:, s], z_grid, color=COLORS[s], lw=2, label="PINN")
            mask = np.isclose(data["t"] * t_max, day)
            ax.plot(data["phi"][mask, s], data["z"][mask], "o",
                    color=COLORS[s], ms=2.5, alpha=0.55, label="FISH")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if k == 0:
                ax.set_ylabel(f"{SHORT[s]}\nz/L", fontsize=7)
            if s == 0:
                ax.set_title(f"Day {day:.0f}", fontsize=8)
            if s == N_SP - 1:
                ax.set_xlabel(r"$\phi_i$", fontsize=7)

        # ── temporal mean panel (last col) ──────────────────────────────────
        ax = axes[s][T]
        # PINN depth-mean at each ODE time point
        pinn_means = np.array([
            float(np.mean(np.asarray(
                pred_fn(jnp.asarray(np.linspace(0, 1, 32)),
                        jnp.float64(tj)))[:, s]))
            for tj in ode_t
        ])
        ax.plot(t_ode_days, ode_phi[:, s], "--", color=COLORS[s],
                lw=1.2, alpha=0.7, label="ODE")
        ax.plot(t_ode_days, pinn_means, color=COLORS[s], lw=1.8, label="PINN⟨z⟩")
        ax.set_xlim(0, t_max); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        if s == 0:
            ax.set_title("⟨φ⟩_z(t)", fontsize=8)
            ax.legend(fontsize=6)
        if s == N_SP - 1:
            ax.set_xlabel("day", fontsize=7)

    D   = np.exp(np.asarray(opt_params["log_D"]))
    u   = float(opt_params["u"])
    dstr = "  ".join(f"D[{SHORT[i]}]={D[i]:.4f}" for i in range(N_SP))
    fig.suptitle(f"PINN+ODE anchor — {data['cond']}\n{dstr}  u={u:.4f}",
                 fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args):
    if not _HAVE_JAX:
        raise RuntimeError(f"JAX not importable: {_JAX_ERR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cond = args.cond
    A, b = load_A_b_np(cond)
    print(f"Condition {cond}: A diag={A.diagonal().round(3)}  b={b.round(3)}")

    if args.smoke:
        print("SMOKE: synthetic data, few epochs")
        data, ode_t, ode_phi = make_synthetic(cond)
        epochs  = args.epochs if args.epochs else 12
        hidden, depth, ncol = 16, 2, 32
    else:
        data = load_zprofiles(Path(args.data), cond)
        epochs = args.epochs if args.epochs else 6000
        hidden, depth, ncol = args.hidden, args.depth, args.collocation
        ode_t, ode_phi = load_ode_trajectory(
            cond, Path(args.ode_dir), t_max=data["t_max"])

    print(f"FISH data:  {len(data['z'])} pts, days={data['days']}, "
          f"z_max={data['z_max']:.1f} um, t_max={data['t_max']:.0f} d")
    print(f"ODE anchor: {len(ode_t)} time pts  (w_ode={args.w_ode})")

    opt_params, history = train(
        data, ode_t, ode_phi, A, b,
        epochs=epochs, lr=args.lr,
        hidden=hidden, depth=depth, n_collocation=ncol,
        w_data=args.w_data, w_phys=args.w_phys, w_ode=args.w_ode,
        n_z_quad=args.n_z_quad, n_ode_per_epoch=args.n_ode_per_epoch,
        seed=args.seed,
    )

    D_fit = np.exp(np.asarray(opt_params["log_D"])).tolist()
    u_fit = float(opt_params["u"])
    result = {
        "cond":             cond,
        "D_fit":            D_fit,
        "u_fit":            u_fit,
        "D_species":        SHORT,
        "final_loss_total": history["total"][-1],
        "final_loss_data":  history["data"][-1],
        "final_loss_phys":  history["phys"][-1],
        "final_loss_ode":   history["ode"][-1],
        "epochs":           epochs,
        "optimizer":        "optax.adam" if _HAVE_OPTAX else "handrolled_jax_adam",
        "hparams": {
            "lr": args.lr, "hidden": hidden, "depth": depth,
            "collocation": ncol, "w_data": args.w_data, "w_phys": args.w_phys,
            "w_ode": args.w_ode, "n_z_quad": args.n_z_quad,
            "n_ode_per_epoch": args.n_ode_per_epoch, "seed": args.seed,
        },
        "units": "D_i, u in normalised units (z/L in [0,1], t in days/t_max)",
        "smoke": bool(args.smoke),
    }
    suffix = "_smoke" if args.smoke else ""
    out_json = OUT_DIR / f"pinn_ode_D_fit_{cond}{suffix}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_json}")
    print(f"Inferred D = {np.round(D_fit, 5)}  u = {u_fit:.5f}")

    if not args.no_plot:
        plot_fit(opt_params, data, ode_t, ode_phi,
                 OUT_DIR / f"pinn_ode_fit_{cond}{suffix}.png")

    return result


def main():
    p = argparse.ArgumentParser(description="1D PINN + ODE mean-field anchor")
    p.add_argument("--cond",    choices=["CH", "DH", "CS", "DS"], default="DH")
    p.add_argument("--data",    type=str, default=str(DATA_DEFAULT),
                   help="FISH z-profile CSV")
    p.add_argument("--ode-dir", type=str, default=str(ODE_DIR_DEFAULT),
                   help="Directory containing ode_continuous_<COND>.csv")
    p.add_argument("--epochs",  type=int, default=0)
    p.add_argument("--lr",      type=float, default=2e-3)
    p.add_argument("--hidden",  type=int, default=64)
    p.add_argument("--depth",   type=int, default=4)
    p.add_argument("--collocation",     type=int, default=512)
    p.add_argument("--w-data",  dest="w_data", type=float, default=1.0)
    p.add_argument("--w-phys",  dest="w_phys", type=float, default=1.0)
    p.add_argument("--w-ode",   dest="w_ode",  type=float, default=5.0,
                   help="Weight for ODE mean-field anchor loss (default 5.0)")
    p.add_argument("--n-z-quad",        dest="n_z_quad",        type=int, default=32,
                   help="z-quadrature points for depth mean (default 32)")
    p.add_argument("--n-ode-per-epoch", dest="n_ode_per_epoch", type=int, default=64,
                   help="ODE time points sampled per epoch (default 64)")
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--smoke",   action="store_true")
    args = p.parse_args()

    if not _HAVE_JAX:
        print(f"[pinn_ode_informed] JAX unavailable ({_JAX_ERR}); cannot run.")
        return
    run(args)


if __name__ == "__main__":
    main()
