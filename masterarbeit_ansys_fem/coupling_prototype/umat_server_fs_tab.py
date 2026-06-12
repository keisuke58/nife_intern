"""Tabulated finite-strain server — NO JAX at solve time (the surrogate promotion path).

The NSP composition trajectory phi_Pg[k] is precomputed offline (precompute step in README /
nsp_traj_DH.json). This server is pure numpy: it reads the trajectory and evaluates the
neo-Hookean F=Fe.Fg stress per Gauss point, so it is as lightweight as the placeholder server.

STATEV here is a single step counter (SDV1): Abaqus passes the start-of-increment value on every
equilibrium iteration and commits counter+1 when the increment converges -> one NSP step per
converged increment, exactly as the JAX server did, but with no solve-time JAX.

  request : NSTATV(=1)  DFGRD1(9,row-major)  STATEV(1)=step  E  nu
  response: STRESS(6, Abaqus Voigt)  DDSDDE(36)  STATEV_new(1)=step+1

Run:  python umat_server_fs_tab.py --port 50007 --traj nsp_traj_DH.json
"""
from __future__ import annotations
import argparse
import json
import socketserver
from pathlib import Path
import numpy as np

import fs_kernel as fs                       # pure numpy, NO JAX
from material_model import elastic_tangent

ARGS = None
TRAJ = None


def to_voigt(s):
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[0, 2], s[1, 2]])


def handle_record(tokens):
    it = iter(tokens)
    nstatev = int(float(next(it)))
    F = np.array([float(next(it)) for _ in range(9)]).reshape(3, 3)
    statev = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else np.zeros(1)
    E = float(next(it)); nu = float(next(it))

    k = int(round(statev[0]))
    phi_pg = TRAJ[min(k, len(TRAJ) - 1)]
    Fg, _ = fs.growth_gradient(phi_pg, ARGS.cond)               # depth growth diag(1,1,lambda_z)
    sigma, ddsdde = fs.stress_and_tangent(F, Fg, E, nu)         # consistent neo-Hookean Jacobian
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
    global ARGS, TRAJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50007)
    ap.add_argument("--cond", default="DH")
    ap.add_argument("--traj", default="nsp_traj_DH.json")
    ARGS = ap.parse_args()
    TRAJ = json.loads(Path(ARGS.traj).read_text())["phi_Pg"]
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((ARGS.host, ARGS.port), Handler) as srv:
        print(f"[umat_server_fs_tab] {ARGS.host}:{ARGS.port} cond={ARGS.cond} "
              f"traj={len(TRAJ)} steps (numpy-only, F=Fe.Fg)")
        srv.serve_forever()


if __name__ == "__main__":
    main()
