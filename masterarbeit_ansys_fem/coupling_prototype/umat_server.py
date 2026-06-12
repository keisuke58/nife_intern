"""Constitutive server — the Python side of the UMAT/UserMat bridge.

Prototyping rung 2: keep the material model in Python (where it was developed) and let the FEM
solver reach it over a socket, one request per Gauss-point evaluation. This sidesteps embedding
CPython inside the ANSYS/Abaqus Fortran build during early prototyping; once the model is trusted
it can be promoted to an in-process call or an offline-tabulated surrogate for speed.

ASCII protocol (list-directed-friendly, so Fortran READ/WRITE handle it):
  request  : nstatev  STRAN(6)  DSTRAN(6)  STATEV(nstatev)  E  nu       (whitespace-separated)
  response : STRESS(6)  DDSDDE(36, row-major)  STATEV_new(nstatev)

Run:  python umat_server.py --port 50007
Test (no Abaqus needed):  python umat_client_test.py
"""
from __future__ import annotations
import argparse
import socketserver
import numpy as np

import material_model as mm


def handle_record(tokens):
    """Parse one request record -> produce one response record (list of floats)."""
    it = iter(tokens)
    nstatev = int(float(next(it)))
    stran = np.array([float(next(it)) for _ in range(6)])
    dstran = np.array([float(next(it)) for _ in range(6)])
    statev = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else np.zeros(0)
    E = float(next(it)); nu = float(next(it))

    stress, ddsdde, statev_new = mm.evaluate(stran, dstran, statev, {"E": E, "nu": nu})
    out = list(stress) + list(np.asarray(ddsdde).reshape(-1)) + list(np.atleast_1d(statev_new))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50007)
    a = ap.parse_args()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((a.host, a.port), Handler) as srv:
        print(f"[umat_server] listening on {a.host}:{a.port}  (model: {mm.__name__})")
        srv.serve_forever()


if __name__ == "__main__":
    main()
