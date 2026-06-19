"""Two-way constitutive server — drop-in replacement for umat_server_nsp.py with stress feedback.

Extends the umat_server_nsp wire protocol with an extra σ_prev(6) field appended to the request, so
the server can throttle the next growth eigenstrain by the previous-increment stress (mechanotrans-
duction). The headline (one-way) wire format is preserved on the response side; the only protocol
change is on the request side, and a sentinel zero-σ_prev exactly reproduces the headline.

Request format (newline-terminated, whitespace-separated tokens):

    NSTATV  STRAN(6)  DSTRAN(6)  STATEV(NSTATV)  E  nu  SIG_PREV(6)

Response (unchanged from umat_server_nsp):

    STRESS(6)  DDSDDE(36 row-major)  STATEV_new(NSTATV)

Run:

    python umat_server_twoway.py --port 50008 --cond DH --beta 0.05 --sigma-c 0.5

Set `--no-feedback` to recover the one-way headline byte-for-byte; useful for the regression check.
"""
from __future__ import annotations

import argparse
import socketserver

import numpy as np

import nsp_mechanics_twoway as mech

ARGS = None


def handle_record(tokens):
    it = iter(tokens)
    nstatev = int(float(next(it)))
    stran = np.array([float(next(it)) for _ in range(6)])
    dstran = np.array([float(next(it)) for _ in range(6)])
    statev = np.array([float(next(it)) for _ in range(nstatev)]) if nstatev else np.zeros(0)
    E = float(next(it)); nu = float(next(it))
    sigma_prev = np.array([float(next(it)) for _ in range(6)])

    props = {"E": E, "nu": nu, "beta": ARGS.beta, "cond": ARGS.cond,
             "phi_init": np.array(ARGS.phi0),
             "sigma_c": ARGS.sigma_c, "feedback": not ARGS.no_feedback}
    stress, ddsdde, statev_new = mech.evaluate_twoway(stran, dstran, statev, props, sigma_prev)
    return list(stress) + list(np.asarray(ddsdde).reshape(-1)) + list(np.atleast_1d(statev_new))


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if not tokens:
                continue
            try:
                resp = handle_record(tokens)
            except Exception as e:
                self.wfile.write(f"ERROR {e}\n".encode())
                continue
            self.wfile.write((" ".join(f"{v:.16e}" for v in resp) + "\n").encode())


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=50008)
    p.add_argument("--cond", default="DH", choices=["CS", "CH", "DS", "DH"])
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--sigma-c", type=float, default=0.5,
                   help="carrying-capacity stress scale (same units as σ); smaller → stronger feedback.")
    p.add_argument("--phi0", nargs=5, type=float, default=[0.1, 0.1, 0.1, 0.1, 0.6])
    p.add_argument("--no-feedback", action="store_true",
                   help="bypass the mechanotransduction term — reproduces the one-way headline.")
    ARGS = p.parse_args()

    with socketserver.ThreadingTCPServer(("127.0.0.1", ARGS.port), Handler) as srv:
        print(f"[twoway] serving on 127.0.0.1:{ARGS.port}  cond={ARGS.cond}  "
              f"beta={ARGS.beta}  sigma_c={ARGS.sigma_c}  feedback={not ARGS.no_feedback}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
