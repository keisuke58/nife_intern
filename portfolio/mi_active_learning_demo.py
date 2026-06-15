#!/usr/bin/env python3
"""Materials-Informatics active-learning demo: GP surrogate + Bayesian optimization.

A self-contained illustration of the core MI loop used in materials / process
design: a black-box "experiment" is expensive, so we fit a Gaussian-process
surrogate to a handful of observations, use an *uncertainty-aware* acquisition
function (Expected Improvement) to pick the next most informative experiment,
and repeat. The point is sample efficiency — reaching the optimum in far fewer
expensive evaluations than random / grid search, with calibrated uncertainty at
every step.

This mirrors the surrogate+BO workflow in my SiC laser-dicing project
(separate repo: sic-dicing-gp) and the gradient-boosting mechanics surrogate in
masterarbeit_ansys_fem/extensions/b3_ml_surrogate.py, distilled to a single
runnable file.

Run:  python portfolio/mi_active_learning_demo.py
Out:  portfolio/figs/mi_active_learning.png  (+ .pdf)

Deps: numpy, scipy, scikit-learn, matplotlib (all standard in the cluster env).
No GPU, no project data — fully reproducible from a fixed seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# The "expensive experiment": a 2-D process-yield response surface.
# Think of x = (normalized laser power, normalized feed speed) and y = device
# yield. Unknown to the optimizer; only queried pointwise, with measurement noise.
# --------------------------------------------------------------------------- #
def true_response(X: np.ndarray) -> np.ndarray:
    """Smooth multimodal surface on [0,1]^2 with a single global optimum."""
    x1, x2 = X[:, 0], X[:, 1]
    bumps = (
        1.10 * np.exp(-((x1 - 0.70) ** 2 + (x2 - 0.30) ** 2) / 0.030)  # global max
        + 0.85 * np.exp(-((x1 - 0.25) ** 2 + (x2 - 0.75) ** 2) / 0.050)  # decoy
        + 0.30 * np.exp(-((x1 - 0.50) ** 2 + (x2 - 0.50) ** 2) / 0.200)
    )
    ridge = 0.15 * np.sin(3.0 * np.pi * x1) * np.cos(2.0 * np.pi * x2)
    return bumps + 0.10 * ridge


def observe(X: np.ndarray, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    return true_response(X) + rng.normal(0.0, noise, size=len(X))


def build_gp() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 1.0))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=4, alpha=0.0
    )


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01):
    """EI acquisition — balances exploitation (high mu) and exploration (high sigma)."""
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def run_strategy(strategy: str, grid: np.ndarray, n_init: int, n_iter: int, seed: int):
    """Active-learning (BO) vs random-sampling baseline; returns best-so-far curve."""
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(grid), size=n_init, replace=False)
    X = grid[init_idx]
    y = observe(X, rng)

    best_curve = [float(true_response(X).max())]
    gp = build_gp()
    for _ in range(n_iter):
        if strategy == "bo":
            gp.fit(X, y)
            mu, sigma = gp.predict(grid, return_std=True)
            acq = expected_improvement(mu, sigma, best=y.max())
            next_idx = int(np.argmax(acq))
        else:  # random baseline
            next_idx = int(rng.integers(len(grid)))
        xn = grid[next_idx][None, :]
        X = np.vstack([X, xn])
        y = np.append(y, observe(xn, rng))
        best_curve.append(float(true_response(X).max()))
    return np.array(best_curve), X, y, gp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--n-repeats", type=int, default=20, help="seeds for the curve band")
    ap.add_argument("--grid", type=int, default=60, help="grid per axis")
    args = ap.parse_args()

    g = np.linspace(0, 1, args.grid)
    gx, gy = np.meshgrid(g, g)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    f_true = true_response(grid)
    global_opt = float(f_true.max())

    # Repeat both strategies over many seeds to get mean +/- std convergence bands.
    bo_curves, rnd_curves = [], []
    for s in range(args.n_repeats):
        bo_curves.append(run_strategy("bo", grid, args.n_init, args.n_iter, seed=100 + s)[0])
        rnd_curves.append(run_strategy("rnd", grid, args.n_init, args.n_iter, seed=100 + s)[0])
    bo_curves = np.array(bo_curves)
    rnd_curves = np.array(rnd_curves)

    # One representative BO run for the surrogate-vs-truth panels.
    _, Xbo, ybo, gp = run_strategy("bo", grid, args.n_init, args.n_iter, seed=0)
    mu, sigma = gp.predict(grid, return_std=True)

    regret_bo = global_opt - bo_curves[:, -1]
    regret_rnd = global_opt - rnd_curves[:, -1]
    speedup_msg = (
        f"final simple-regret  BO={regret_bo.mean():.4f}  random={regret_rnd.mean():.4f}  "
        f"({regret_rnd.mean() / max(regret_bo.mean(), 1e-9):.1f}x lower with BO)"
    )
    print(speedup_msg)

    # ----------------------------- plotting ------------------------------- #
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    extent = [0, 1, 0, 1]
    shp = (args.grid, args.grid)

    # (a) ground-truth response + sampled points (colored by acquisition order)
    ax = axes[0]
    im = ax.imshow(f_true.reshape(shp), origin="lower", extent=extent, cmap="viridis", aspect="auto")
    ax.scatter(Xbo[: args.n_init, 0], Xbo[: args.n_init, 1], c="white", s=35, edgecolors="k",
               label="initial design", zorder=3)
    order = np.arange(args.n_init, len(Xbo))
    ax.scatter(Xbo[args.n_init:, 0], Xbo[args.n_init:, 1], c=order, cmap="autumn", s=42,
               edgecolors="k", label="BO-selected", zorder=4)
    j = int(np.argmax(f_true))
    ax.scatter([grid[j, 0]], [grid[j, 1]], marker="*", c="red", s=260, edgecolors="k",
               label="true optimum", zorder=5)
    ax.set_title("(a) True response + BO sampling")
    ax.set_xlabel("process param 1"); ax.set_ylabel("process param 2")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="yield")

    # (b) GP posterior std = where the model is still uncertain (drives exploration)
    ax = axes[1]
    im = ax.imshow(sigma.reshape(shp), origin="lower", extent=extent, cmap="magma", aspect="auto")
    ax.scatter(Xbo[:, 0], Xbo[:, 1], c="cyan", s=18, edgecolors="k", zorder=3)
    ax.set_title("(b) GP posterior uncertainty (UQ)")
    ax.set_xlabel("process param 1"); ax.set_ylabel("process param 2")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="posterior std")

    # (c) convergence: BO vs random, mean +/- std over repeats
    ax = axes[2]
    xx = np.arange(bo_curves.shape[1])
    for curves, color, lab in [(bo_curves, "#1b9e77", "Bayesian opt (EI)"),
                               (rnd_curves, "#d95f02", "random search")]:
        m, sd = curves.mean(0), curves.std(0)
        ax.plot(xx, m, color=color, lw=2, label=lab)
        ax.fill_between(xx, m - sd, m + sd, color=color, alpha=0.20)
    ax.axhline(global_opt, ls="--", c="k", lw=1, label="global optimum")
    ax.set_title("(c) Sample efficiency")
    ax.set_xlabel("# expensive experiments"); ax.set_ylabel("best yield found")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.02, 0.02, f"BO reaches optimum in\n~{int(np.argmax(bo_curves.mean(0) > global_opt - 0.02))} "
            f"evals vs random's many more", transform=ax.transAxes, fontsize=7,
            va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.suptitle("Materials-Informatics active learning: GP surrogate + Bayesian optimization",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    outdir = HERE / "figs"
    outdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"mi_active_learning.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir / 'mi_active_learning.png'}")


if __name__ == "__main__":
    main()
