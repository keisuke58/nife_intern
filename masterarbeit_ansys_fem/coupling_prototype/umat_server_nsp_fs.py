"""Finite-strain NSP-backed server — returns neo-Hookean Cauchy stress through F = Fe.Fg.

Replaces umat_server_nsp.py for the NLGEOM run. Wire protocol v2 carries the deformation gradient
(not small strain), since the finite-strain law needs F:

  request : NSTATV(=12)  DFGRD1(9, row-major)  STATEV(12)  E  nu
  response: STRESS(6, Abaqus Voigt 11,22,33,12,13,23)  DDSDDE(36)  STATEV_new(12)

Per Gauss point: advance NSP one step -> phi_Pg -> Fg=diag(1,1,lambda_z) -> Fe=F.Fg^-1 ->
neo-Hookean Cauchy stress. DDSDDE returned as the isotropic elastic tangent (approximate material
Jacobian — adequate for the free-swelling 1-element smoke test; a consistent spatial tangent is the
production TODO). cond/beta fixed server-side via CLI.

Run:  python umat_server_nsp_fs.py --port 50007 --cond DH
"""
from __future__ import annotations
import argparse
import socketserver
import numpy as np

import nsp_mechanics_fs as fs
import nsp_material_model as nm
from material_model import elastic_tangent

ARGS = None

# Cauchy (3x3) -> Abaqus Voigt (11,22,33,12,13,23)
def to_voigt(s):
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[0, 2], s[1, 2]])


def handle_record(tokens):
    it = iter(tokens)
    nstatev = int(float(next(it)))
    F = np.array([float(next(it)) for _ in range(9)]).reshape(3, 3)
    statev = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else np.zeros(0)
    E = float(next(it)); nu = float(next(it))

    props = {"E": E, "nu": nu, "cond": ARGS.cond, "phi_init": np.array(ARGS.phi0)}
    # advance composition; init NSP state if STATEV arrives all-zero (Abaqus *DEPVAR)
    if statev.size != nm.N_STATE or not np.any(statev):
        statev = np.asarray(nm.make_initial_state(props["phi_init"]), float)
    g_new = np.asarray(fs._nsp_step_for(props["cond"])(statev), float)
    phi_pg = g_new[fs.PG_INDEX]

    sigma, lam_z = fs.stress_from_F(F, phi_pg, props)
    stress = to_voigt(sigma)
    ddsdde = elastic_tangent(E, nu)            # approximate Jacobian
    out = list(stress) + list(ddsdde.reshape(-1)) + list(g_new)
    return out


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.decode().strip()
            if not line or line.upper() == "QUIT":
                break
            out = handle_record(line.split())
            self.wfile.write((" ".join(f"{v:.16e}" for v in out) + "\n").encode())


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50007)
    ap.add_argument("--cond", default="DH", choices=["CS", "CH", "DS", "DH"])
    ap.add_argument("--beta", type=float, default=17.0)  # accepted for runner compat; FS uses calib
    ap.add_argument("--phi0", type=float, nargs=5, default=[0.30, 0.20, 0.20, 0.15, 0.15])
    ARGS = ap.parse_args()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((ARGS.host, ARGS.port), Handler) as srv:
        print(f"[umat_server_nsp_fs] {ARGS.host}:{ARGS.port} cond={ARGS.cond} "
              f"(finite-strain F=Fe.Fg, STATEV={nm.N_STATE})")
        srv.serve_forever()


if __name__ == "__main__":
    main()
