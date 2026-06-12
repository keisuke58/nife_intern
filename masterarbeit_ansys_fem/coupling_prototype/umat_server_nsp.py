"""NSP-backed constitutive server — drop-in replacement for umat_server.py.

Serves the mechanically-closed NSP model (nsp_mechanics_model) over the SAME wire protocol, so
abaqus/umat.f needs no change — only set *DEPVAR 12 in the deck and point the C client here.

  request : NSTATV(=12)  STRAN(6)  DSTRAN(6)  STATEV(12)  E  nu
  response: STRESS(6)  DDSDDE(36 row-major)  STATEV_new(12)

The composition coupling (cond, beta) is fixed server-side via CLI, keeping the wire format
identical to the placeholder server. Abaqus *DEPVAR initialises STATEV to 0; the model detects the
all-zero state on the first increment and initialises a valid NSP state from --phi0.

Run:  python umat_server_nsp.py --port 50007 --cond DH --beta 0.05
"""
from __future__ import annotations
import argparse
import socketserver
import numpy as np

import nsp_mechanics_model as mech

ARGS = None  # set in main()


def handle_record(tokens):
    it = iter(tokens)
    nstatev = int(float(next(it)))
    stran = np.array([float(next(it)) for _ in range(6)])
    dstran = np.array([float(next(it)) for _ in range(6)])
    statev = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else np.zeros(0)
    E = float(next(it)); nu = float(next(it))

    props = {"E": E, "nu": nu, "beta": ARGS.beta, "cond": ARGS.cond,
             "phi_init": np.array(ARGS.phi0)}
    stress, ddsdde, statev_new = mech.evaluate(stran, dstran, statev, props)
    return list(stress) + list(np.asarray(ddsdde).reshape(-1)) + list(np.atleast_1d(statev_new))


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
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--phi0", type=float, nargs=5, default=[0.30, 0.20, 0.20, 0.15, 0.15])
    ARGS = ap.parse_args()
    with socketserver.ThreadingTCPServer((ARGS.host, ARGS.port), Handler) as srv:
        print(f"[umat_server_nsp] {ARGS.host}:{ARGS.port}  cond={ARGS.cond} beta={ARGS.beta} "
              f"STATEV={mech.NSTATEV}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
