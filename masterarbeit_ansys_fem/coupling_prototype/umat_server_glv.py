"""gLV UMAT server — coupled bio-mechanics at each Gauss point.

Generalized Lotka-Volterra in REPLICATOR form (Σφᵢ = 1 conserved):
    dφᵢ/dt = φᵢ (fᵢ(φ) - <f>(φ))
    fᵢ(φ)  = bᵢ + Σⱼ Aᵢⱼ φⱼ
    <f>     = Σᵢ φᵢ fᵢ

This is the form used to FIT the parameters in fit_glv_heine.json.
Using this form ensures consistency with the benchmarked RMSE=0.0490 (Dieckow LOO).
φᵢ ∈ [0,1] always, numerically stable.

STATEV layout (DEPVAR=5):
    [0..4]  φ₁..φ₅  relative species fractions (Σ=1)
            (init = CLSM t=1 day values, spatially uniform)

protocol (request — uses umat_socket_ve.f which appends DTIME):
    NSTATV COORDS(3) DFGRD1(9) STATEV(5) E nu DTIME
response:
    STRESS(6) DDSDDE(36) STATEV_new(5)

Species order: [S.oralis, A.naeslundii, V.parvula, F.nucleatum, P.gingivalis]
P. gingivalis = index 4 → drives biofilm growth.

Run:
    python umat_server_glv.py --cond DH --bio-days 21 --port 50007
        --gamma 0.4
"""
from __future__ import annotations
import argparse, json, socketserver
from pathlib import Path
import numpy as np

import fs_kernel as fs

ARGS   = None
ZPROF  = None   # fallback depth profile (Hamilton calibration) for spatial IC variation
A_GLV  = None   # (5,5) interaction matrix
B_GLV  = None   # (5,) intrinsic growth rates
X0_GLV = None   # (5,) initial abundances (t=1 day from CLSM)
PG_IDX = 4      # P. gingivalis column index

# t=1 day CLSM abundances (DH condition, from ode_continuous_DH.csv first time point)
_X0_DH = np.array([0.0305, 0.010, 0.9490, 0.0050, 0.0055])
_X0_CH = np.array([0.0305, 0.010, 0.9490, 0.0050, 0.0055])  # approximate; update if CH t=0 differs


def glv_rhs(phi, A, b):
    """gLV replicator: dφᵢ/dt = φᵢ(fᵢ - <f>), conserves Σφᵢ=1.
    This is the form used to fit parameters in fit_glv_heine.json."""
    f   = b + A @ phi
    f_mean = phi @ f
    return phi * (f - f_mean)


def rk4_step(phi, A, b, dt, n_sub=10):
    """RK4 integration of gLV replicator for time dt, n_sub sub-steps for stability."""
    h = dt / n_sub
    for _ in range(n_sub):
        k1 = glv_rhs(phi, A, b)
        k2 = glv_rhs(np.clip(phi + 0.5 * h * k1, 0.0, 1.0), A, b)
        k3 = glv_rhs(np.clip(phi + 0.5 * h * k2, 0.0, 1.0), A, b)
        k4 = glv_rhs(np.clip(phi + h * k3,        0.0, 1.0), A, b)
        phi = np.clip(phi + h / 6.0 * (k1 + 2*k2 + 2*k3 + k4), 0.0, 1.0)
        s   = phi.sum()
        if s > 1e-12:
            phi /= s    # re-normalize to correct floating-point drift
    return phi


def to_voigt(s):
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[0, 2], s[1, 2]])


def handle_record(tokens):
    it = iter(tokens)
    nstatev = int(float(next(it)))
    coords  = np.array([float(next(it)) for _ in range(3)])
    F       = np.array([float(next(it)) for _ in range(9)]).reshape(3, 3)
    statev  = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else X0_GLV.copy()
    E       = float(next(it))
    nu      = float(next(it))
    dtime   = float(next(it))   # Abaqus DTIME (0..1 scale)

    # --- gLV replicator ODE step ---
    phi_old = statev[:5]
    # first call: Abaqus initialises STATEV=0, set from CLSM t=1 day IC
    if phi_old.sum() < 1e-12:
        phi_old = X0_GLV.copy()
        s = phi_old.sum()
        if s > 1e-12:
            phi_old /= s    # ensure Σφ=1

    dtime_bio = dtime * ARGS.bio_days           # Abaqus t ∈ [0,1] → bio days
    phi_new = rk4_step(phi_old, A_GLV, B_GLV, dtime_bio, n_sub=ARGS.n_sub)
    phi_pg  = float(phi_new[PG_IDX])

    # --- mechanical growth (same framework as NSP server) ---
    z    = np.clip((coords[ARGS.coord_axis] - ARGS.coord_offset) / ARGS.height, 0.0, 1.0)
    # use phi_pg from gLV ODE (spatially uniform in pure gLV; depends on IC for spatial variation)
    drive = phi_pg * ARGS.gamma
    Fg, _ = fs.growth_gradient_iso(1.0 + drive)
    sigma, ddsdde = fs.stress_and_tangent(F, Fg, E, nu)
    stress = to_voigt(sigma)

    sv_new = np.zeros(nstatev)
    sv_new[:5] = phi_new
    return list(stress) + list(ddsdde.reshape(-1)) + list(sv_new)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.decode().strip()
            if not line or line.upper() == "QUIT":
                break
            out = handle_record(line.split())
            self.wfile.write((" ".join(f"{v:.16e}" for v in out) + "\n").encode())


def main():
    global ARGS, ZPROF, A_GLV, B_GLV, X0_GLV
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",       default="127.0.0.1")
    ap.add_argument("--port",       type=int, default=50007)
    ap.add_argument("--cond",       default="DH", choices=["DH","CH","DS","CS"])
    ap.add_argument("--glv-fit",    default=None,
                    help="path to fit_glv_heine.json (auto-detected if None)")
    ap.add_argument("--bio-days",   type=float, default=21.0, dest="bio_days",
                    help="total biological time (days) mapped to Abaqus t=0..1")
    ap.add_argument("--n-sub",      type=int, default=10, dest="n_sub",
                    help="RK4 sub-steps per DTIME for stability")
    ap.add_argument("--gamma",      type=float, default=0.4,
                    help="growth magnitude scale (φ_Pg * gamma -> growth drive)")
    ap.add_argument("--growth_mode", default="iso", choices=["iso"])
    ap.add_argument("--height",     type=float, default=1.0)
    ap.add_argument("--coord-axis", type=int, default=1, dest="coord_axis")
    ap.add_argument("--coord-offset", type=float, default=0.0, dest="coord_offset")
    ap.add_argument("--profile",    default=None,
                    help="(optional) fallback depth profile JSON for spatial IC")
    ARGS = ap.parse_args()

    # load gLV fit
    glv_fit_path = ARGS.glv_fit or str(
        Path(__file__).resolve().parent.parent.parent / "results" / "heine2025" / "fit_glv_heine.json")
    fit = json.loads(Path(glv_fit_path).read_text())
    A_GLV = np.array(fit[ARGS.cond]["A"])
    B_GLV = np.array(fit[ARGS.cond]["b"])
    X0_GLV = _X0_DH.copy() if ARGS.cond in ("DH", "DS") else _X0_CH.copy()

    if ARGS.profile:
        p = json.loads(Path(ARGS.profile).read_text())
        ZPROF = (np.array(p["z_norm"]), np.array(p["phi_Pg"]))

    species = fit[ARGS.cond]["species"]
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((ARGS.host, ARGS.port), Handler) as srv:
        print(f"[umat_server_glv] cond={ARGS.cond}  bio_days={ARGS.bio_days}")
        print(f"  species: {species}")
        print(f"  x0(t=1d): {X0_GLV.tolist()}")
        print(f"  gLV replicator: dφᵢ/dt = φᵢ(fᵢ - <f>)  [fitted form, Σφᵢ=1]")
        srv.serve_forever()


if __name__ == "__main__":
    main()
