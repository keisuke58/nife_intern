#!/usr/bin/env python3
"""
pinn_1d_virtual_time.py — 1D (z,t) PINN with ODE-interpolated virtual timepoints.

Instead of adding an ODE loss term (which constrains bulk mean ≠ spatial profile),
we *expand* the FISH training data by generating virtual z-profiles at intermediate
days.  Each virtual profile uses the nearest observed FISH day as spatial template,
scaled by the ODE composition ratio:

    φ̃_i(z, t*) ∝ φ_FISH_i(z, t_k) × [φ_ODE_i(t*) / φ_ODE_i(t_k)]   (then renorm)

where t_k = nearest observed FISH day.  This preserves the spatial shape while
following the ODE temporal dynamics — consistent because it never constrains
bulk vs spatial at the same time point.

Sweep mode (--sweep): runs n_virtual ∈ {5,11,21,42} with --sweep-epochs each
and reports the best n_virtual by final data loss.

CLI:
    python scripts/pde/pinn_1d_virtual_time.py --cond DH --n-virtual 21
    python scripts/pde/pinn_1d_virtual_time.py --cond DH --sweep
    python scripts/pde/pinn_1d_virtual_time.py --cond CH --n-virtual 21 --epochs 6000

Outputs (results/pinn_virtual/):
    pinn_virt_D_fit_<COND>_nv<N>.json
    pinn_virt_fit_<COND>_nv<N>.png
    pinn_virt_sweep_<COND>.json   (sweep mode)
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

HERE     = Path(__file__).resolve().parents[2]
DATA_CSV = HERE / "results" / "diffusion_fit" / "zprofiles_all_ti.csv"
ODE_DIR  = HERE / "results" / "heine2025"
OUT_DIR  = HERE / "results" / "pinn_virtual"

N_SP     = 5
SHORT    = ["S.o", "A.n", "Vd/Vp", "F.n", "P.g"]
COLORS   = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]
SPECIES_COLS = ["S.oralis", "A.naeslundii", "Vd/Vp", "F.nucleatum", "P.gingivalis"]
ODE_SP_IDX = {
    "S. oralis": 0, "A. naeslundii": 1,
    "V. dispar": 2, "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4, "P. gingivalis_W83": 4,
}


# ── A, b ─────────────────────────────────────────────────────────────────────

def load_A_b(cond: str):
    fit = HERE / "results" / "heine2025" / "fit_glv_heine.json"
    d = json.loads(fit.read_text())
    return (np.asarray(d[cond]["A"], dtype=np.float64),
            np.asarray(d[cond]["b"], dtype=np.float64))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_zprofiles_raw(csv_path: Path, cond: str):
    """Return dict: day -> (z_norm array, phi array (N_z, 5))."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["condition"] == cond].copy()
    days    = sorted(df["day"].unique())
    z_max   = float(df["z_um"].max())
    t_max   = float(max(days))
    per_day = {}
    for day in days:
        sub = df[df["day"] == day]
        z   = sub["z_um"].to_numpy(dtype=np.float64) / z_max
        phi = sub[SPECIES_COLS].to_numpy(dtype=np.float64)
        phi = np.maximum(phi, 0.0)
        rs  = phi.sum(axis=1, keepdims=True)
        phi = np.where(rs > 1e-12, phi / rs, 1.0 / N_SP)
        per_day[day] = (z, phi)
    return per_day, days, z_max, t_max


def load_ode(cond: str, t_max: float):
    """Return (t_days_all, phi_ode (N_t, 5))."""
    import pandas as pd
    csv = ODE_DIR / f"ode_continuous_{cond}.csv"
    df  = pd.read_csv(csv)
    t_all = sorted(df["t_day"].unique())
    phi   = np.zeros((len(t_all), N_SP), dtype=np.float64)
    for sp, idx in ODE_SP_IDX.items():
        sub = df[df["species"] == sp].set_index("t_day")["phi"]
        for k, td in enumerate(t_all):
            if td in sub.index:
                phi[k, idx] = float(sub[td])
    rs = phi.sum(axis=1, keepdims=True)
    phi /= np.where(rs > 1e-12, rs, 1.0)
    return np.array(t_all), phi


def make_virtual_profiles(per_day, days, ode_t_days, ode_phi, n_virtual: int, t_max: float):
    """Generate n_virtual time points (including the 5 real FISH days).

    Virtual times are evenly spaced between day 1 and day 21.
    Each virtual profile borrows the spatial shape from the nearest FISH day
    and scales it by the ODE composition ratio at the target day.

    Returns: z (N_pts,), t (N_pts,) [normalised], phi (N_pts, 5)
    """
    # target days: evenly spaced across [day_min, day_max], always include real days
    real_days = np.array(days, dtype=float)
    d_min, d_max = real_days.min(), real_days.max()

    if n_virtual <= len(days):
        target_days = real_days
    else:
        # uniform grid; force real days to be included
        grid = np.linspace(d_min, d_max, n_virtual)
        target_days = np.unique(np.concatenate([grid, real_days]))

    # ODE phi interpolator (linear)
    def ode_at(t_day):
        return np.array([np.interp(t_day, ode_t_days, ode_phi[:, i])
                         for i in range(N_SP)])

    all_z, all_t, all_phi = [], [], []
    for t_day in target_days:
        # nearest real FISH day as spatial template
        nearest = real_days[np.argmin(np.abs(real_days - t_day))]
        z_src, phi_src = per_day[int(nearest)]   # (N_z,), (N_z, 5)

        phi_ode_src = ode_at(nearest)             # (5,)
        phi_ode_tgt = ode_at(t_day)               # (5,)

        # multiplicative ODE scaling, column-wise
        # avoid div-by-zero: if source ODE is near 0 keep source value
        scale = np.where(phi_ode_src > 1e-4,
                         phi_ode_tgt / phi_ode_src,
                         np.ones(N_SP))
        phi_virt = phi_src * scale[np.newaxis, :]   # (N_z, 5)
        phi_virt = np.maximum(phi_virt, 0.0)
        rs = phi_virt.sum(axis=1, keepdims=True)
        phi_virt = phi_virt / np.where(rs > 1e-12, rs, 1.0)

        n_z = len(z_src)
        all_z.append(z_src)
        all_t.append(np.full(n_z, t_day / t_max))
        all_phi.append(phi_virt)

    return (np.concatenate(all_z),
            np.concatenate(all_t),
            np.concatenate(all_phi, axis=0),
            target_days)


# ── Network (same as pinn_diffusion_inverse) ──────────────────────────────────

def init_mlp(layer_sizes, key):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for (ni, no), k in zip(zip(layer_sizes[:-1], layer_sizes[1:]), keys):
        std = math.sqrt(2.0 / (ni + no))
        params.append((std * jax.random.normal(k, (ni, no), dtype=jnp.float64),
                        jnp.zeros((no,), dtype=jnp.float64)))
    return params


def phi_net(params, z, t):
    h = jnp.array([z, t])
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return jax.nn.softmax(h @ W + b)


def phi_and_derivs(params, z, t):
    f = lambda z_, t_: phi_net(params, z_, t_)
    phi      = f(z, t)
    dphi_dt  = jax.jacrev(f, 1)(z, t)
    dphi_dz  = jax.jacrev(f, 0)(z, t)
    d2phi_dz2 = jax.jacrev(jax.jacrev(f, 0), 0)(z, t)
    return phi, dphi_dt, dphi_dz, d2phi_dz2


def reaction(phi, A, b):
    g = b + A @ phi
    return phi * (g - jnp.dot(phi, g))


def pde_residual(params, z, t, log_D, u, A, b):
    phi, dphi_dt, dphi_dz, d2phi_dz2 = phi_and_derivs(params, z, t)
    D   = jnp.exp(log_D)
    R   = reaction(phi, A, b)
    return dphi_dt - D * d2phi_dz2 + u * dphi_dz - R


# ── Loss ──────────────────────────────────────────────────────────────────────

def make_loss(A, b, w_data, w_phys):
    A_ = jnp.asarray(A); b_ = jnp.asarray(b)
    v_phi = jax.vmap(phi_net,      in_axes=(None, 0, 0))
    v_res = jax.vmap(pde_residual, in_axes=(None, 0, 0, None, None, None, None))

    def loss_fn(opt, dz, dt, dphi, cz, ct):
        net = opt["net"]; log_D = opt["log_D"]; u = opt["u"]
        pred      = v_phi(net, dz, dt)
        data_loss = jnp.mean((pred - dphi) ** 2)
        res       = v_res(net, cz, ct, log_D, u, A_, b_)
        phys_loss = jnp.mean(res ** 2)
        return w_data * data_loss + w_phys * phys_loss, (data_loss, phys_loss)

    return loss_fn


# ── Adam fallback ─────────────────────────────────────────────────────────────

def adam_init(p):
    return {"m": jax.tree_util.tree_map(jnp.zeros_like, p),
            "v": jax.tree_util.tree_map(jnp.zeros_like, p),
            "s": jnp.array(0, dtype=jnp.int64)}

def adam_step(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    s = st["s"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1*m+(1-b1)*g, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2*v+(1-b2)*g*g, st["v"], g)
    p2 = jax.tree_util.tree_map(
        lambda p, m, v: p - lr*(m/(1-b1**s))/(jnp.sqrt(v/(1-b2**s))+eps), p, m, v)
    return p2, {"m": m, "v": v, "s": s}


# ── Train ─────────────────────────────────────────────────────────────────────

def train(data_z, data_t, data_phi, A, b, *,
          epochs, lr=2e-3, hidden=64, depth=4, n_col=512,
          w_data=1.0, w_phys=1.0, seed=0, log_every=None):

    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    layer_sizes = [2] + [hidden]*depth + [N_SP]
    net = init_mlp(layer_sizes, ki)
    opt = {"net": net,
           "log_D": jnp.log(jnp.array([0.015,0.008,0.012,0.012,0.005])),
           "u":     jnp.float64(0.005)}

    loss_fn  = make_loss(A, b, w_data, w_phys)
    vag      = jax.value_and_grad(loss_fn, has_aux=True)
    dz = jnp.asarray(data_z); dt = jnp.asarray(data_t); dphi = jnp.asarray(data_phi)

    if _HAVE_OPTAX:
        solver   = optax.adam(lr)
        opt_st   = solver.init(opt)
    else:
        opt_st   = adam_init(opt)

    @jax.jit
    def step(opt, opt_st, cz, ct):
        (tot, aux), grads = vag(opt, dz, dt, dphi, cz, ct)
        if _HAVE_OPTAX:
            upd, opt_st2 = solver.update(grads, opt_st, opt)
            opt2 = optax.apply_updates(opt, upd)
        else:
            opt2, opt_st2 = adam_step(opt, grads, opt_st, lr)
        return opt2, opt_st2, tot, aux

    hist = {"total": [], "data": [], "phys": []}
    log_every = log_every or max(1, epochs // 10)

    for ep in range(epochs):
        key, kc = jax.random.split(key)
        col = jax.random.uniform(kc, (n_col, 2), dtype=jnp.float64)
        opt, opt_st, tot, (dl, pl) = step(opt, opt_st, col[:,0], col[:,1])
        hist["total"].append(float(tot))
        hist["data"].append(float(dl))
        hist["phys"].append(float(pl))
        if ep % log_every == 0 or ep == epochs-1:
            D = np.exp(np.asarray(opt["log_D"])).round(4)
            print(f"  ep {ep:5d}  data={float(dl):.3e}  phys={float(pl):.3e}"
                  f"  D={D}  u={float(opt['u']):.4f}")

    return opt, hist


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_fit(opt, data_z, data_t, data_phi, t_max, days_real, cond, n_virtual, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family":"serif","font.size":9})

    z_grid = np.linspace(0, 1, 60)
    pred_fn = jax.vmap(lambda z, t: phi_net(opt["net"], z, t), in_axes=(0, None))

    fig, axes = plt.subplots(N_SP, len(days_real),
                             figsize=(2.0*len(days_real), 1.9*N_SP),
                             sharex=True, sharey=True, squeeze=False)
    for s in range(N_SP):
        for k, day in enumerate(days_real):
            ax = axes[s][k]
            tn = day / t_max
            pp = np.asarray(pred_fn(jnp.asarray(z_grid), jnp.float64(tn)))
            ax.plot(pp[:, s], z_grid, color=COLORS[s], lw=2, label="PINN")
            mask = np.isclose(data_t * t_max, day)
            ax.plot(data_phi[mask, s], data_z[mask], "o",
                    color=COLORS[s], ms=2.5, alpha=0.55, label="FISH")
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            if k == 0: ax.set_ylabel(f"{SHORT[s]}\nz/L", fontsize=7)
            if s == 0: ax.set_title(f"Day {day:.0f}", fontsize=8)
            if s == N_SP-1: ax.set_xlabel(r"$\phi_i$", fontsize=7)

    D   = np.exp(np.asarray(opt["log_D"]))
    dstr = "  ".join(f"D[{SHORT[i]}]={D[i]:.4f}" for i in range(N_SP))
    fig.suptitle(f"PINN virtual-time ({n_virtual} pts) — {cond}\n{dstr}", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ── Run one condition / n_virtual ─────────────────────────────────────────────

def run_one(cond, n_virtual, epochs, lr, hidden, depth, n_col, w_data, w_phys, seed,
            no_plot=False, suffix=""):
    A, b = load_A_b(cond)
    per_day, days, z_max, t_max = load_zprofiles_raw(DATA_CSV, cond)
    ode_t_days, ode_phi = load_ode(cond, t_max)

    data_z, data_t, data_phi, target_days = make_virtual_profiles(
        per_day, days, ode_t_days, ode_phi, n_virtual, t_max)

    print(f"\n[{cond}] n_virtual={n_virtual}  "
          f"({len(days)} real + {len(target_days)-len(days)} virtual days)  "
          f"→ {len(data_z)} training pts")

    opt, hist = train(data_z, data_t, data_phi, A, b,
                      epochs=epochs, lr=lr, hidden=hidden, depth=depth,
                      n_col=n_col, w_data=w_data, w_phys=w_phys, seed=seed)

    D_fit = np.exp(np.asarray(opt["log_D"])).tolist()
    result = {
        "cond": cond, "n_virtual": n_virtual,
        "n_target_days": int(len(target_days)),
        "n_train_pts": int(len(data_z)),
        "D_fit": D_fit, "u_fit": float(opt["u"]),
        "D_species": SHORT,
        "final_loss_data":  hist["data"][-1],
        "final_loss_phys":  hist["phys"][-1],
        "final_loss_total": hist["total"][-1],
        "epochs": epochs,
        "hparams": {"lr": lr, "hidden": hidden, "depth": depth,
                    "n_col": n_col, "w_data": w_data, "w_phys": w_phys},
    }
    tag = f"{cond}_nv{n_virtual}{suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"pinn_virt_D_fit_{tag}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  saved {json_path.name}  data_loss={hist['data'][-1]:.4e}")

    if not no_plot:
        plot_fit(opt, data_z, data_t, data_phi, t_max, days, cond, n_virtual,
                 OUT_DIR / f"pinn_virt_fit_{tag}.png")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond",       choices=["CH","DH","CS","DS"], default="DH")
    p.add_argument("--n-virtual",  dest="n_virtual", type=int, default=21,
                   help="Total time points incl. real FISH days (default 21 = 1/day)")
    p.add_argument("--sweep",      action="store_true",
                   help="Sweep n_virtual ∈ {5,11,21,42} and report best")
    p.add_argument("--sweep-epochs", dest="sweep_epochs", type=int, default=1000,
                   help="Epochs per sweep run (default 1000)")
    p.add_argument("--epochs",     type=int, default=4000)
    p.add_argument("--lr",         type=float, default=2e-3)
    p.add_argument("--hidden",     type=int, default=64)
    p.add_argument("--depth",      type=int, default=4)
    p.add_argument("--collocation",type=int, default=512)
    p.add_argument("--w-data",     dest="w_data", type=float, default=1.0)
    p.add_argument("--w-phys",     dest="w_phys", type=float, default=1.0)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--no-plot",    action="store_true")
    args = p.parse_args()

    if not _HAVE_JAX:
        print(f"JAX unavailable: {_JAX_ERR}"); return

    if args.sweep:
        candidates = [5, 11, 21, 42]
        print(f"Sweep n_virtual ∈ {candidates}, {args.sweep_epochs} epochs each")
        sweep_results = []
        for nv in candidates:
            r = run_one(args.cond, nv, args.sweep_epochs,
                        args.lr, args.hidden, args.depth,
                        args.collocation, args.w_data, args.w_phys, args.seed,
                        no_plot=True, suffix="_sweep")
            sweep_results.append(r)

        sweep_results.sort(key=lambda r: r["final_loss_data"])
        print("\n── Sweep results (sorted by data_loss) ──")
        for r in sweep_results:
            print(f"  n_virtual={r['n_virtual']:3d}  "
                  f"data_loss={r['final_loss_data']:.4e}  "
                  f"n_train={r['n_train_pts']}")
        best = sweep_results[0]
        print(f"\n→ Best: n_virtual={best['n_virtual']}  "
              f"data_loss={best['final_loss_data']:.4e}")

        sweep_path = OUT_DIR / f"pinn_virt_sweep_{args.cond}.json"
        sweep_path.write_text(json.dumps(sweep_results, indent=2))
        print(f"Saved sweep summary: {sweep_path.name}")

    else:
        run_one(args.cond, args.n_virtual, args.epochs,
                args.lr, args.hidden, args.depth,
                args.collocation, args.w_data, args.w_phys, args.seed,
                no_plot=args.no_plot)


if __name__ == "__main__":
    main()
