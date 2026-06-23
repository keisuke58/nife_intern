"""Column server — depth-graded isotropic growth → residual stress (the FEM-only result).

The scientific payload single-element runs cannot show: a multi-element biofilm column where each
Gauss point grows by the LOCAL P. gingivalis fraction φ_Pg(z) read from the CLSM depth profile
(`phipg_depth_DH.json`). Under lateral confinement (substrate-bonded film, free top), the
differential growth with depth develops an in-plane **residual-stress profile** σ_xx(z).

numpy-only (surrogate path; no JAX). Isotropic volumetric growth θ(z,t) = 1 + (t/T)·γ·φ_Pg(z),
ramped over the step for convergence; Fg = θ^(1/3) I; neo-Hookean Fe.

  request : NSTATV  COORDS(3)  DFGRD1(9,row-major)  STATEV(1)=step  E  nu
  response: STRESS(6, Abaqus Voigt)  DDSDDE(36)  STATEV_new(1)=step+1

Run:  python umat_server_col.py --port 50007 --gamma 0.4 --nramp 20 --height 1.0
"""
from __future__ import annotations
import argparse
import json
import socketserver
from pathlib import Path
import numpy as np

import fs_kernel as fs

ARGS = None
ZPROF = None   # (z_norm array, phi_Pg array)


def to_voigt(s):
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[0, 2], s[1, 2]])


def growth_for(phi_pg, frac):
    """Build the growth gradient Fg for the local Pg fraction (mode iso | depth)."""
    drive = frac * ARGS.gamma * phi_pg
    if ARGS.growth_mode == "depth":          # anisotropic substratum-normal
        Fg, _ = fs.growth_gradient_depth(phi_pg, frac * ARGS.gamma)
    else:                                    # isotropic volumetric (default)
        Fg, _ = fs.growth_gradient_iso(1.0 + drive)
    return Fg


def handle_record(tokens):
    it = iter(tokens)
    nstatev = int(float(next(it)))
    coords = np.array([float(next(it)) for _ in range(3)])
    F = np.array([float(next(it)) for _ in range(9)]).reshape(3, 3)
    statev = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else np.zeros(1)
    E = float(next(it)); nu = float(next(it))

    k = int(round(statev[0]))
    if ARGS.geometry == "helical":
        # 3-D helical: depth = r - r_surface(theta, z)
        x, y, zc = coords[0], coords[1], coords[2]
        r     = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        r_s   = ARGS.R0 + ARGS.amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * zc / ARGS.pitch - theta))
        z     = np.clip((r - r_s) / ARGS.height, 0.0, 1.0)
    else:
        z = np.clip((coords[ARGS.coord_axis] - ARGS.coord_offset) / ARGS.height, 0.0, 1.0)
    phi_pg = float(np.interp(z, ZPROF[0], ZPROF[1]))  # local Pg fraction
    frac = min(k / max(ARGS.nramp, 1), 1.0)           # growth ramp over the step
    Fg = growth_for(phi_pg, frac)
    sigma, ddsdde = fs.stress_and_tangent(F, Fg, E, nu)   # consistent neo-Hookean Jacobian
    stress = to_voigt(sigma)
    return list(stress) + list(ddsdde.reshape(-1)) + [float(k + 1)]


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.decode().strip()
            if not line or line.upper() == "QUIT":
                break
            out = handle_record(line.split())
            self.wfile.write((" ".join(f"{v:.16e}" for v in out) + "\n").encode())


def main():
    global ARGS, ZPROF
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50007)
    ap.add_argument("--profile", default="phipg_depth_DH.json")
    ap.add_argument("--gamma", type=float, default=0.4, help="growth per unit phi_Pg (CLSM-calibrated "
                    "magnitude ~17 for full thickness match; 0.4 keeps the demo in easy convergence)")
    ap.add_argument("--growth_mode", default="iso", choices=["iso", "depth"],
                    help="iso=volumetric (lateral confinement -> in-plane residual stress); "
                         "depth=substratum-normal anisotropic")
    ap.add_argument("--nramp", type=int, default=20, help="increments to reach full growth")
    ap.add_argument("--height", type=float, default=1.0, help="column physical height (z-norm)")
    ap.add_argument("--coord-axis", type=int, default=2, dest="coord_axis",
                    help="COORDS index for depth (2=z for 3-D C3D8 column; 1=y for 2-D C3D8 implant; "
                         "0=r for CGAX4 axisymmetric)")
    ap.add_argument("--coord-offset", type=float, default=0.0, dest="coord_offset",
                    help="subtract before dividing by height (e.g. R0 for CGAX4 axisymmetric)")
    ap.add_argument("--geometry", default="column", choices=["column", "helical"],
                    help="column=coord[axis]/height (default); helical=3-D helix depth from surface")
    ap.add_argument("--R0",    type=float, default=2.0, help="[helical] implant core radius")
    ap.add_argument("--pitch", type=float, default=1.0, help="[helical] thread pitch")
    ap.add_argument("--amp",   type=float, default=0.18, help="[helical] thread amplitude")
    ARGS = ap.parse_args()
    p = json.loads(Path(ARGS.profile).read_text())
    ZPROF = (np.array(p["z_norm"]), np.array(p["phi_Pg"]))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((ARGS.host, ARGS.port), Handler) as srv:
        print(f"[umat_server_col] {ARGS.host}:{ARGS.port} mode={ARGS.growth_mode} gamma={ARGS.gamma} "
              f"nramp={ARGS.nramp} profile={ARGS.profile} ({len(ZPROF[0])} pts, consistent tangent)")
        srv.serve_forever()


if __name__ == "__main__":
    main()
