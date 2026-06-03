#!/usr/bin/env python3
# [nife-pathshim]
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
"""
verify_pde_numerics.py — numerical verification of the 1-D spatial-PDE TRANSPORT
operators used by the thesis spatial model:

  * diagonal scheme        (nsp_pde_1d_heine.py:transport_step)
  * volume-filling x-diff   (nsp_pde_1d_heine_xdiff.py:transport_step_xdiff)

This is the verification protocol referenced in the thesis numerics section
(Soleimani / secondary-examiner defence). It is deliberately REACTION-OFF and
pure-numpy: it isolates the discretisation, needs no JAX / NSP core / cluster,
and runs in well under a second on the login node (light, matplotlib-class).

Axis (i)+(ii)  Diffusion convergence: the central second-difference operator with
               the code's ghost-node no-flux boundary is built explicitly; its
               leading non-trivial eigenvalue is compared to the analytic
               -D(pi/L)^2 of the cos(pi z/L) mode at Nz = 20,40,80,160,320. The
               observed order of accuracy (slope of log error vs log dz) should be
               ~2, confirming the diffusion stencil is second order in space.
               (The overall scheme is first order: upwind advection + Lie split.)

Axis (iii)     Conservation: two species peaks that diffuse and overlap drive the
               local occupancy rho = sum_i phi_i above 1. The diagonal scheme
               diffuses each phi_i independently and clips to [0,1]; the clip then
               destroys mass. The volume-filling cross-diffusion flux is degenerate
               (J_i -> 0 as rho -> 1), so rho stays <= 1 and total mass is conserved
               to machine precision by the conservative finite-volume update.
               Both run with no-flux at BOTH ends so the only mass change is the
               scheme's own (non-)conservation.

Outputs (results/nsp_pde/):
    verify_convergence.png, verify_conservation.png, verify_pde_numerics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "results" / "nsp_pde"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = ["S.o", "A.n", "Vd/Vp", "F.n", "P.g"]
COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]


# ── Axis (i)+(ii): diffusion stencil convergence via operator eigenvalue ──────

def diffusion_operator(Nz, L=1.0, D=1.0):
    """Dense (Nz x Nz) matrix of D d^2/dz^2 with the code's ghost-node no-flux BCs
    at BOTH ends (interior: central; node 0: 2(phi1-phi0)/dz^2; node Nz-1 mirror)."""
    dz = L / (Nz - 1)
    M = np.zeros((Nz, Nz))
    idx = np.arange(1, Nz - 1)
    M[idx, idx - 1] = 1.0
    M[idx, idx] = -2.0
    M[idx, idx + 1] = 1.0
    # ghost-node no-flux (matches transport_step: 2*(phi[1]-phi[0])/dz^2)
    M[0, 0], M[0, 1] = -2.0, 2.0
    M[-1, -1], M[-1, -2] = -2.0, 2.0
    return D * M / dz**2


def convergence_study(L=1.0, D=1.0, grids=(20, 40, 80, 160, 320)):
    lam_exact = -D * (np.pi / L) ** 2            # cos(pi z/L) mode eigenvalue
    dzs, errs = [], []
    for Nz in grids:
        evals = np.linalg.eigvals(diffusion_operator(Nz, L, D)).real
        lam_h = evals[np.argmin(np.abs(evals - lam_exact))]
        dzs.append(L / (Nz - 1))
        errs.append(abs(lam_h - lam_exact))
    dzs, errs = np.array(dzs), np.array(errs)
    # observed order between successive grids and overall least-squares slope
    orders = np.log(errs[:-1] / errs[1:]) / np.log(dzs[:-1] / dzs[1:])
    slope = np.polyfit(np.log(dzs), np.log(errs), 1)[0]
    return dict(grids=list(grids), dz=dzs.tolist(), err=errs.tolist(),
                pairwise_order=orders.tolist(), ls_slope=float(slope),
                lam_exact=float(lam_exact))


# ── Axis (iii): conservation — diagonal+clip vs volume-filling cross-diffusion ─

def step_diagonal(phi, D, dz, dt):
    """One explicit transport step, diagonal diffusion, no-flux both ends, then
    clip to [0,1] (faithful to nsp_pde_1d_heine.transport_step, u=0)."""
    dp = np.zeros_like(phi)
    dp[1:-1] += D[None, :] * (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dz**2
    dp[0] += D * 2 * (phi[1] - phi[0]) / dz**2
    dp[-1] += D * 2 * (phi[-2] - phi[-1]) / dz**2
    return np.clip(phi + dt * dp, 0.0, 1.0)


def step_xdiff(phi, D, dz, dt):
    """One explicit conservative-FV step of volume-filling cross-diffusion,
    no-flux both ends (faithful to nsp_pde_1d_heine_xdiff.transport_step_xdiff, u=0)."""
    rho = phi.sum(axis=-1)
    phi_face = 0.5 * (phi[1:] + phi[:-1])
    rho_face = 0.5 * (rho[1:] + rho[:-1])
    dphi = (phi[1:] - phi[:-1]) / dz
    drho = (rho[1:] - rho[:-1]) / dz
    J = -D[None, :] * ((1.0 - rho_face)[:, None] * dphi + phi_face * drho[:, None])
    dp = np.zeros_like(phi)
    dp[1:-1] += -(J[1:] - J[:-1]) / dz       # interior divergence
    dp[0] += -J[0] / dz                       # left face flux = 0
    dp[-1] += J[-1] / dz                      # right face flux = 0
    return np.clip(phi + dt * dp, 0.0, 1.0)


def conservation_study(Nz=80, L=1.0, n_steps=300):
    z = np.linspace(0, L, Nz)
    D = np.array([0.05, 0.05, 0.02, 0.02, 0.01])
    # two peaks that diffuse toward each other -> rho > 1 in the overlap
    phi0 = np.zeros((Nz, 5))
    phi0[:, 0] = 0.75 * np.exp(-((z - 0.35) / 0.10) ** 2)
    phi0[:, 1] = 0.75 * np.exp(-((z - 0.65) / 0.10) ** 2)
    phi0[:, 2] = 0.05
    phi = {"diagonal": phi0.copy(), "xdiff": phi0.copy()}
    dt = 0.4 * 0.5 * (L / (Nz - 1)) ** 2 / D.max()   # 0.4 of the diffusion CFL
    M0 = phi0.sum() * (L / (Nz - 1))
    rec = {k: {"mass": [], "max_rho": []} for k in phi}
    for _ in range(n_steps):
        phi["diagonal"] = step_diagonal(phi["diagonal"], D, L / (Nz - 1), dt)
        phi["xdiff"] = step_xdiff(phi["xdiff"], D, L / (Nz - 1), dt)
        for k in phi:
            rec[k]["mass"].append(phi[k].sum() * (L / (Nz - 1)))
            rec[k]["max_rho"].append(float(phi[k].sum(-1).max()))
    out = {"Nz": Nz, "n_steps": n_steps, "dt": float(dt), "M0": float(M0)}
    for k in phi:
        m = np.array(rec[k]["mass"])
        out[k] = {
            "mass_rel_drift": float((m[-1] - M0) / M0),
            "max_rho_overall": float(np.max(rec[k]["max_rho"])),
            "_mass": m.tolist(), "_max_rho": rec[k]["max_rho"],
        }
    return out, phi0, z


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_convergence(conv, path):
    dz, err = np.array(conv["dz"]), np.array(conv["err"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.loglog(dz, err, "o-", color="#1f77b4", label="eigenvalue error")
    ref = err[0] * (dz / dz[0]) ** 2
    ax.loglog(dz, ref, "k--", lw=1, label=r"$\mathcal{O}(\Delta z^2)$ reference")
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel(r"$|\lambda_h - \lambda_{\mathrm{exact}}|$")
    ax.set_title(f"Diffusion stencil convergence (slope {conv['ls_slope']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_conservation(cons, phi0, z, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].stackplot(z, *[phi0[:, s] for s in range(5)],
                      colors=COLORS, labels=SHORT, alpha=0.85)
    axes[0].plot(z, phi0.sum(-1), "k-", lw=1.2, label=r"$\rho=\sum\phi_i$")
    axes[0].axhline(1.0, color="r", ls=":", lw=1)
    axes[0].set_title("Initial profile"); axes[0].set_xlabel("z")
    axes[0].set_ylabel("abundance"); axes[0].legend(fontsize=6, ncol=2)
    steps = np.arange(cons["n_steps"]) * cons["dt"]
    for k, c in (("diagonal", "#d62728"), ("xdiff", "#1f77b4")):
        axes[1].plot(steps, np.array(cons[k]["_mass"]) / cons["M0"], color=c,
                     label=f"{k}")
        axes[2].plot(steps, cons[k]["_max_rho"], color=c, label=f"{k}")
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].set_title("Total mass / $M_0$"); axes[1].set_xlabel("t"); axes[1].legend(fontsize=8)
    axes[2].axhline(1.0, color="r", ls=":", lw=1, label=r"$\rho=1$")
    axes[2].set_title(r"$\max_z \rho(z,t)$"); axes[2].set_xlabel("t"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description="Verify 1-D spatial-PDE transport operators")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    conv = convergence_study()
    cons, phi0, z = conservation_study(n_steps=args.steps)

    plot_convergence(conv, OUT / "verify_convergence.png")
    plot_conservation(cons, phi0, z, OUT / "verify_conservation.png")

    summary = {"convergence": conv,
               "conservation": {k: v for k, v in cons.items()
                                if not isinstance(v, dict)} |
               {k: {kk: vv for kk, vv in cons[k].items() if not kk.startswith("_")}
                for k in ("diagonal", "xdiff")}}
    (OUT / "verify_pde_numerics.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Convergence (diffusion stencil) ===")
    print(f"  grids {conv['grids']}")
    print(f"  pairwise observed order: {[round(o,2) for o in conv['pairwise_order']]}")
    print(f"  least-squares slope    : {conv['ls_slope']:.3f}  (expect ~2)")
    print("=== Conservation (no-flux both ends, reaction off) ===")
    for k in ("diagonal", "xdiff"):
        print(f"  {k:9s}: mass rel. drift {cons[k]['mass_rel_drift']:+.2e}, "
              f"max rho {cons[k]['max_rho_overall']:.3f}")
    print(f"\nJSON: {OUT/'verify_pde_numerics.json'}")


if __name__ == "__main__":
    main()
