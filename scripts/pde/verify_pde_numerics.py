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
(Soleimani / secondary-examiner defence). It is deliberately REACTION-OFF,
ADVECTION-OFF and pure-numpy: it isolates the spatial discretisation, needs no
JAX / NSP core / cluster, and runs in well under a second on the login node
(light, matplotlib-class). The conservation tests run in a CLOSED BOX (no-flux at
BOTH ends) instead of the production Dirichlet-at-z=L: a Dirichlet end is an open
boundary that injects/removes mass, which would mask the scheme's own
(non-)conservation. Every stencil below is bit-faithful to the production code
(audited 2026-06-04); the only deliberate differences are the closed-box right BC
and u=0 / reaction-off, which isolate the operator under test.

WHAT THE TESTS ESTABLISH (and, just as importantly, what they do NOT)
─────────────────────────────────────────────────────────────────────
(i)+(ii) DIFFUSION CONVERGENCE. The central second-difference operator with the
    code's ghost-node no-flux boundary is built explicitly; the error of its
    eigenvalue closest to the analytic -D(pi/L)^2 of the cos(pi z/L) mode is
    measured at Nz = 20,40,80,160,320. The slope of log error vs log dz is ~2.
    CAVEAT (honest scoping): cos(pi z/L) is the homogeneous-Neumann eigenmode and
    satisfies u'(0)=0 AND u'''(0)=0, so the boundary node is O(dz^2) for THIS mode
    — a best case. A separate boundary-order study (boundary_order_study) applies
    the operator to a generic Neumann-compatible function (u'(0)=0 but u'''(0)!=0)
    and recovers only O(dz^1) at the wall. Honest claim: the INTERIOR central
    second-difference is 2nd order, and the ghost-node boundary stencil is 2nd
    order only for Neumann-compatible modes. The OVERALL split solver is 1st order
    anyway (upwind advection + Lie splitting).

(iii) CONSERVATION. Two species peaks diffuse in a closed box; we track the total
    mass (uniform-weight sum) and the local occupancy rho = sum_i phi_i.
      - The DIAGONAL scheme is NOT mass-conserving, and NOT because of the [0,1]
        clip. In the sub-critical benchmark the clip fires 0 times (max phi_i<1),
        and removing it leaves the mass drift bit-identical. The drift is the
        ghost-node no-flux stencil 2*(phi[1]-phi[0])/dz^2: it is NOT in
        flux-divergence form, so the discrete operator has ZERO ROW SUMS (it
        preserves constants) but NONZERO COLUMN SUMS at the two wall nodes
        (+-D/dz^2). d(mass)/dt = dz * (colsum)^T phi, so the nonzero boundary
        columns inject/remove mass every step (~+2.18e-3 relative here).
      - The VOLUME-FILLING cross-diffusion flux is a conservative finite-volume
        scheme (fluxes on cell faces, divergence form) so its column sums vanish
        and total mass is conserved to MACHINE PRECISION in the sub-critical
        regime (drift = 0).
    CROWDING REGIME (the whole point of the volume-filling scheme). A separate
    SUPERCRITICAL initial condition (peaks chosen so rho > 1) shows what the
    sub-critical benchmark cannot:
      - The diagonal clip is PER-SPECIES (np.clip on each phi_i, not on rho), so
        even at max rho = 1.56 it fires 0 times (max phi_i ~ 0.89 < 1). The
        diagonal scheme therefore CANNOT bound rho and lets it overshoot 1
        unphysically. THIS — not "the clip destroys mass" — is the correct
        motivation for the volume-filling scheme.
      - The volume-filling J -> 0 degeneracy as rho -> 1 is what should cap rho.
        But note its conservation is regime-dependent in this EXPLICIT discretisation:
        under crowding the explicit update overshoots [0,1] and the "safety" clip
        fires, breaking machine-precision conservation (motivating an implicit /
        positivity-preserving step for production crowded runs).

Outputs (results/nsp_pde/):
    verify_convergence.png, verify_conservation.png, verify_pde_numerics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thesis_style import use as thesis_style   # shared usetex/lmodern thesis style

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


def operator_sums(Nz=10, L=1.0, D=1.0):
    """Conservation diagnostic of the diagonal diffusion operator.
    Mass (uniform-weight sum) is conserved iff the COLUMN sums (1^T M) vanish;
    constants are preserved iff the ROW sums (M 1) vanish. The ghost-node no-flux
    stencil preserves constants but is NOT mass-conserving at the two wall nodes."""
    M = diffusion_operator(Nz, L, D)
    row = M.sum(axis=1)            # M 1  — preserves constants if ~0
    col = M.sum(axis=0)            # 1^T M — conserves mass if ~0
    return dict(Nz=Nz, max_abs_row_sum=float(np.max(np.abs(row))),
                max_abs_col_sum=float(np.max(np.abs(col))),
                col_sums=col.round(6).tolist(),
                nonzero_col_idx=[int(i) for i in np.where(np.abs(col) > 1e-9)[0]])


def boundary_order_study(L=1.0, D=1.0, grids=(20, 40, 80, 160, 320)):
    """Node-0 truncation error of the ghost-node boundary stencil for
      (a) cos(pi z/L)         — Neumann eigenmode, u'(0)=u'''(0)=0  -> O(dz^2)
      (b) g(z)=sin(z)-z       — Neumann-compatible, u'(0)=0 but u'''(0)!=0 -> O(dz^1)
    Demonstrates the boundary stencil is 2nd order ONLY for special modes.
    The interior central difference is uniformly 2nd order (reported separately)."""
    def err_at_wall(fn, d2fn, Nz):
        z = np.linspace(0, L, Nz)
        dz = L / (Nz - 1)
        f = fn(z)
        approx0 = D * 2 * (f[1] - f[0]) / dz**2       # ghost-node no-flux at node 0
        return abs(approx0 - D * d2fn(z[0]))
    out = {}
    for name, fn, d2fn in (
        ("cos_neumann_mode", lambda z: np.cos(np.pi * z / L),
         lambda z: -(np.pi / L) ** 2 * np.cos(np.pi * z / L)),
        ("generic_neumann",  lambda z: np.sin(z) - z,
         lambda z: -np.sin(z)),
    ):
        dzs = np.array([L / (Nz - 1) for Nz in grids])
        errs = np.array([err_at_wall(fn, d2fn, Nz) for Nz in grids])
        slope = float(np.polyfit(np.log(dzs), np.log(errs + 1e-300), 1)[0])
        out[name] = dict(dz=dzs.tolist(), err=errs.tolist(), wall_order=slope)
    return out


# ── Axis (iii): conservation — diagonal+clip vs volume-filling cross-diffusion ─

def step_diagonal(phi, D, dz, dt, clip=True):
    """One explicit transport step, diagonal diffusion, no-flux both ends, then
    (optionally) clip each phi_i to [0,1] (faithful to
    nsp_pde_1d_heine.transport_step, u=0). Returns (phi_new, n_clipped)."""
    dp = np.zeros_like(phi)
    dp[1:-1] += D[None, :] * (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dz**2
    dp[0] += D * 2 * (phi[1] - phi[0]) / dz**2
    dp[-1] += D * 2 * (phi[-2] - phi[-1]) / dz**2
    raw = phi + dt * dp
    if clip:
        new = np.clip(raw, 0.0, 1.0)
        n_clipped = int(np.count_nonzero(raw != new))
        return new, n_clipped
    return raw, 0


def step_xdiff(phi, D, dz, dt, clip=True):
    """One explicit conservative-FV step of volume-filling cross-diffusion,
    no-flux both ends (faithful to nsp_pde_1d_heine_xdiff.transport_step_xdiff,
    u=0). The clip is a numerical-safety guard. Returns (phi_new, n_clipped)."""
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
    raw = phi + dt * dp
    if clip:
        new = np.clip(raw, 0.0, 1.0)
        n_clipped = int(np.count_nonzero(raw != new))
        return new, n_clipped
    return raw, 0


def make_ic(Nz, L=1.0, crowded=False):
    """Two Gaussian species peaks + a constant background.
      sub-critical: peaks 0.30 apart, amp 0.75 -> max rho ~ 0.80 (clip never fires)
      crowded     : peaks 0.10 apart, amp 0.90 -> max rho ~ 1.56 (rho exceeds 1)"""
    z = np.linspace(0, L, Nz)
    phi0 = np.zeros((Nz, 5))
    if crowded:
        amp, c0, c1, sig, bg = 0.90, 0.45, 0.55, 0.12, 0.05
    else:
        amp, c0, c1, sig, bg = 0.75, 0.35, 0.65, 0.10, 0.05
    phi0[:, 0] = amp * np.exp(-((z - c0) / sig) ** 2)
    phi0[:, 1] = amp * np.exp(-((z - c1) / sig) ** 2)
    phi0[:, 2] = bg
    return phi0, z


def conservation_study(Nz=80, L=1.0, n_steps=300, crowded=False):
    phi0, z = make_ic(Nz, L, crowded)
    D = np.array([0.05, 0.05, 0.02, 0.02, 0.01])
    dz = L / (Nz - 1)
    dt = 0.4 * 0.5 * dz ** 2 / D.max()               # 0.4 of the diffusion CFL
    M0 = phi0.sum() * dz

    state = {  # (phi, cumulative clip count) for each (scheme, clip-on/off)
        "diagonal":      [phi0.copy(), 0],
        "diagonal_noclip": [phi0.copy(), 0],
        "xdiff":         [phi0.copy(), 0],
        "xdiff_noclip":  [phi0.copy(), 0],
    }
    rec = {k: {"mass": [], "max_rho": [], "raw_min": [], "raw_max": []} for k in state}
    steppers = {
        "diagonal":        lambda p: step_diagonal(p, D, dz, dt, clip=True),
        "diagonal_noclip": lambda p: step_diagonal(p, D, dz, dt, clip=False),
        "xdiff":           lambda p: step_xdiff(p, D, dz, dt, clip=True),
        "xdiff_noclip":    lambda p: step_xdiff(p, D, dz, dt, clip=False),
    }
    for _ in range(n_steps):
        for k, fn in steppers.items():
            # capture the pre-clip raw extent from the clip=False sibling logic
            raw, _ = (step_diagonal if k.startswith("diagonal") else step_xdiff)(
                state[k][0], D, dz, dt, clip=False)
            new, nclip = fn(state[k][0])
            state[k][0] = new
            state[k][1] += nclip
            rec[k]["mass"].append(new.sum() * dz)
            rec[k]["max_rho"].append(float(new.sum(-1).max()))
            rec[k]["raw_min"].append(float(raw.min()))
            rec[k]["raw_max"].append(float(raw.max()))

    out = {"Nz": Nz, "n_steps": n_steps, "dt": float(dt), "M0": float(M0),
           "crowded": crowded, "ic_max_rho": float(phi0.sum(-1).max())}
    for k in state:
        m = np.array(rec[k]["mass"])
        out[k] = {
            "mass_rel_drift": float((m[-1] - M0) / M0),
            "max_rho_overall": float(np.max(rec[k]["max_rho"])),
            "n_clipped_total": int(state[k][1]),
            "raw_min_overall": float(np.min(rec[k]["raw_min"])),
            "raw_max_overall": float(np.max(rec[k]["raw_max"])),
            "_mass": m.tolist(), "_max_rho": rec[k]["max_rho"],
        }
    return out, phi0, z


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_convergence(conv, bdry, path):
    dz, err = np.array(conv["dz"]), np.array(conv["err"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=thesis_style(1.0, aspect=0.42))
    ax.loglog(dz, err, "o-", color="#1f77b4", label="eigenvalue error")
    ref = err[0] * (dz / dz[0]) ** 2
    ax.loglog(dz, ref, "k--", lw=1, label=r"$\mathcal{O}(\Delta z^2)$ reference")
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel(r"$|\lambda_h - \lambda_{\mathrm{exact}}|$")
    ax.set_title(f"Diffusion stencil (interior+Neumann mode)\nslope {conv['ls_slope']:.3f}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    # boundary-order honesty panel
    _disp = {"cos_neumann_mode": "cos (Neumann mode)", "generic_neumann": "generic Neumann"}
    for name, c, mk in (("cos_neumann_mode", "#1f77b4", "o"),
                        ("generic_neumann", "#d62728", "s")):
        d = bdry[name]
        ax2.loglog(d["dz"], d["err"], mk + "-", color=c,
                   label=f"{_disp[name]} (wall order {d['wall_order']:.2f})")
    dzb = np.array(bdry["generic_neumann"]["dz"])
    ax2.loglog(dzb, bdry["generic_neumann"]["err"][0] * (dzb / dzb[0]), "k:",
               lw=1, label=r"$\mathcal{O}(\Delta z)$")
    ax2.set_xlabel(r"$\Delta z$"); ax2.set_ylabel("node-0 truncation error")
    ax2.set_title("Ghost-node BOUNDARY order\n(2nd only for Neumann modes)")
    ax2.legend(fontsize=7); ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {path} (+ .pdf)")


def plot_conservation(sub, crow, path):
    fig, axes = plt.subplots(2, 2, figsize=thesis_style(1.0, aspect=0.66))
    for row, (cons, tag) in enumerate(((sub, "sub-critical IC"), (crow, "crowded IC"))):
        steps = np.arange(cons["n_steps"]) * cons["dt"]
        axm, axr = axes[row]
        for k, c, lab in (("diagonal", "#d62728", "diagonal"),
                          ("xdiff", "#1f77b4", "xdiff")):
            axm.plot(steps, np.array(cons[k]["_mass"]) / cons["M0"], color=c,
                     label=f"{lab} (drift {cons[k]['mass_rel_drift']:+.1e}, "
                           f"clip {cons[k]['n_clipped_total']})")
            axr.plot(steps, cons[k]["_max_rho"], color=c, label=lab)
        axm.axhline(1.0, color="k", ls=":", lw=1)
        axm.set_title(f"Total mass / $M_0$ — {tag}")
        axm.set_xlabel("t"); axm.legend(fontsize=7)
        axr.axhline(1.0, color="r", ls=":", lw=1, label=r"$\rho=1$")
        axr.set_title(rf"$\max_z\rho$ — {tag} (IC max {cons['ic_max_rho']:.2f})")
        axr.set_xlabel("t"); axr.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {path} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description="Verify 1-D spatial-PDE transport operators")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    conv = convergence_study()
    bdry = boundary_order_study()
    sums = operator_sums()
    sub, _, _ = conservation_study(n_steps=args.steps, crowded=False)
    crow, _, _ = conservation_study(n_steps=args.steps, crowded=True)

    plot_convergence(conv, bdry, OUT / "verify_convergence.png")
    plot_conservation(sub, crow, OUT / "verify_conservation.png")

    def _strip(c):
        return {k: ({kk: vv for kk, vv in c[k].items() if not kk.startswith("_")}
                    if isinstance(c[k], dict) else c[k]) for k in c}

    summary = {"convergence": conv, "boundary_order": bdry, "operator_sums": sums,
               "conservation_subcritical": _strip(sub),
               "conservation_crowded": _strip(crow)}
    (OUT / "verify_pde_numerics.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Convergence (diffusion stencil) ===")
    print(f"  grids {conv['grids']}")
    print(f"  pairwise observed order: {[round(o,2) for o in conv['pairwise_order']]}")
    print(f"  least-squares slope    : {conv['ls_slope']:.4f}  (expect ~2, interior+Neumann mode)")
    print(f"  BOUNDARY wall order: cos-mode {bdry['cos_neumann_mode']['wall_order']:.2f} "
          f"(best case) vs generic-Neumann {bdry['generic_neumann']['wall_order']:.2f} (~1)")
    print("=== Operator conservation diagnostic (diagonal) ===")
    print(f"  max|row sum|={sums['max_abs_row_sum']:.2e} (->0: preserves constants), "
          f"max|col sum|={sums['max_abs_col_sum']:.2e} (!=0 at walls -> NOT mass conserving)")
    print(f"  nonzero column indices: {sums['nonzero_col_idx']} (Nz={sums['Nz']}: only the 4 wall columns)")
    print("=== Conservation: SUB-CRITICAL IC (clip should never fire) ===")
    for k in ("diagonal", "xdiff"):
        s = sub[k]
        print(f"  {k:9s}: drift {s['mass_rel_drift']:+.2e}, max rho {s['max_rho_overall']:.3f}, "
              f"clip fired {s['n_clipped_total']}x, raw[min,max]=[{s['raw_min_overall']:.3f},{s['raw_max_overall']:.3f}]")
    print(f"  diagonal drift with clip REMOVED: {sub['diagonal_noclip']['mass_rel_drift']:+.2e} "
          f"(=> drift is the boundary stencil, not the clip)")
    print(f"=== Conservation: CROWDED IC (max rho {crow['ic_max_rho']:.2f} > 1) ===")
    note = {"diagonal": "per-species clip cannot bound rho -> rho overshoots 1",
            "xdiff": "explicit step overshoots; safety clip breaks conservation"}
    for k in ("diagonal", "xdiff"):
        s = crow[k]
        print(f"  {k:9s}: drift {s['mass_rel_drift']:+.2e}, max rho {s['max_rho_overall']:.3f}, "
              f"clip fired {s['n_clipped_total']}x  [{note[k]}]")
    print(f"\nJSON: {OUT/'verify_pde_numerics.json'}")


if __name__ == "__main__":
    main()
