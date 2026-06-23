"""Viscoelastic column server — NSP growth + 2-arm Maxwell Prony relaxation.

Extends umat_server_col.py: after the elastic neo-Hookean stress is computed, the deviatoric
part is attenuated by the incremental Prony recurrence (Simo & Hughes 1998, Algorithm 10.1).

Prony parameters from prony_biofilm.json (2 arms, tau = 6.6 s and 903 s).

STATEV layout (DEPVAR=19):
    [0]      growth ramp counter (same as base server)
    [1..6]   h1[6]  — Prony arm 1 deviatoric overstress (Voigt: 11,22,33,12,13,23)
    [7..12]  h2[6]  — Prony arm 2 deviatoric overstress
    [13..18] sigma_e_old[6] — previous elastic Cauchy stress (for increment computation)

protocol (request, VE version):
    NSTATV COORDS(3) DFGRD1(9) STATEV(19) E nu DTIME
response:
    STRESS(6) DDSDDE(36) STATEV_new(19)

Run:
    python umat_server_ve.py --port 50007 --gamma 0.4 --nramp 20 --height 1.0
        --profile phipg_depth_DH.json --prony prony_biofilm.json
"""
from __future__ import annotations
import argparse, json, socketserver
from pathlib import Path
import numpy as np

import fs_kernel as fs

ARGS  = None
ZPROF = None    # (z_norm, phi_Pg)
G_INF = None    # long-time elastic fraction
ARMS  = None    # list of (g_i, tau_i_s)


def to_voigt(s):
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[0, 2], s[1, 2]])


_DEVSEL = np.array([1, 1, 1, 0, 0, 0], dtype=float)   # 1 for normal, 0 for shear dofs


def deviatoric(v6):
    p = (v6[0] + v6[1] + v6[2]) / 3.0
    return v6 - p * _DEVSEL


def prony_update(sigma_e_new, sigma_e_old, h_prev, dtime, g_i, tau_i):
    """Incremental Maxwell arm update (Simo & Hughes Alg. 10.1, deviatoric only)."""
    if dtime <= 0:
        return h_prev
    xi  = dtime / tau_i
    gam = np.exp(-xi)
    beta = g_i * (1.0 - gam) / xi
    ds_dev = deviatoric(sigma_e_new) - deviatoric(sigma_e_old)
    return gam * h_prev + beta * ds_dev


def handle_record(tokens):
    it = iter(tokens)
    nstatev = int(float(next(it)))
    coords  = np.array([float(next(it)) for _ in range(3)])
    F       = np.array([float(next(it)) for _ in range(9)]).reshape(3, 3)
    statev  = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else np.zeros(19)
    E       = float(next(it))
    nu      = float(next(it))
    dtime   = float(next(it))     # VE extension: time increment from Fortran DTIME

    k    = int(round(statev[0]))
    z    = (coords[ARGS.coord_axis] - ARGS.coord_offset) / ARGS.height
    phi_pg = float(np.interp(z, ZPROF[0], ZPROF[1]))
    frac = min(k / max(ARGS.nramp, 1), 1.0)

    # --- elastic growth stress (same as base server) ---
    if ARGS.growth_mode == "depth":
        Fg, _ = fs.growth_gradient_depth(phi_pg, frac * ARGS.gamma)
    else:
        drive = frac * ARGS.gamma * phi_pg
        Fg, _ = fs.growth_gradient_iso(1.0 + drive)
    sigma_mat, ddsdde_e = fs.stress_and_tangent(F, Fg, E, nu)
    sigma_e_new = to_voigt(sigma_mat)

    # --- Prony viscoelastic correction ---
    h1_old      = statev[1:7]
    h2_old      = statev[7:13]
    sigma_e_old = statev[13:19]

    g1, tau1 = ARMS[0]
    g2, tau2 = ARMS[1]
    h1_new = prony_update(sigma_e_new, sigma_e_old, h1_old, dtime, g1, tau1)
    h2_new = prony_update(sigma_e_new, sigma_e_old, h2_old, dtime, g2, tau2)

    # viscoelastic Cauchy stress (Simo & Hughes eq. 10.38)
    stress_ve = sigma_e_new - (h1_new + h2_new)

    # consistent algorithmic tangent (scale deviatoric part by (g_inf + Σ beta_k))
    xi1   = dtime / tau1 if dtime > 0 else 1e-10
    xi2   = dtime / tau2 if dtime > 0 else 1e-10
    beta1 = g1 * (1.0 - np.exp(-xi1)) / xi1
    beta2 = g2 * (1.0 - np.exp(-xi2)) / xi2
    scale_dev = G_INF + beta1 + beta2   # multiplier on deviatoric part of C_alg
    # apply scale_dev to the deviatoric part of ddsdde_e
    ddsdde_e_flat = ddsdde_e.reshape(6, 6)
    # isochoric scaling: C_alg_VE ≈ K_bulk * IxI/3 + scale_dev * (C_alg_e - K_bulk*IxI/3)
    # approximate: scale full tangent by scale_dev (conservative)
    ddsdde_ve = scale_dev * ddsdde_e_flat

    # update STATEV
    sv_new = np.zeros(nstatev)
    sv_new[0]    = float(k + 1)
    sv_new[1:7]  = h1_new
    sv_new[7:13] = h2_new
    sv_new[13:19] = sigma_e_new

    return list(stress_ve) + list(ddsdde_ve.reshape(-1)) + list(sv_new)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.decode().strip()
            if not line or line.upper() == "QUIT":
                break
            out = handle_record(line.split())
            self.wfile.write((" ".join(f"{v:.16e}" for v in out) + "\n").encode())


def main():
    global ARGS, ZPROF, G_INF, ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--port",    type=int, default=50007)
    ap.add_argument("--profile", default="phipg_depth_DH.json")
    ap.add_argument("--prony",   default="prony_biofilm.json")
    ap.add_argument("--gamma",   type=float, default=0.4)
    ap.add_argument("--growth_mode", default="iso", choices=["iso", "depth"])
    ap.add_argument("--nramp",   type=int, default=20)
    ap.add_argument("--height",  type=float, default=1.0)
    ap.add_argument("--coord-axis",   type=int,   default=1, dest="coord_axis")
    ap.add_argument("--coord-offset", type=float, default=0.0, dest="coord_offset")
    ARGS = ap.parse_args()

    p = json.loads(Path(ARGS.profile).read_text())
    ZPROF = (np.array(p["z_norm"]), np.array(p["phi_Pg"]))
    prony_data = json.loads(Path(ARGS.prony).read_text())
    G_INF = prony_data["g_inf"]
    ARMS  = [(arm["g_i"], arm["tau_i_s"]) for arm in prony_data["prony"]]

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((ARGS.host, ARGS.port), Handler) as srv:
        print(f"[umat_server_ve] {ARGS.host}:{ARGS.port} g_inf={G_INF:.3f} "
              f"arms={[(g,round(t)) for g,t in ARMS]}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
