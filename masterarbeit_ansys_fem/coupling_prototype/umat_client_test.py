"""End-to-end test of the bridge protocol WITHOUT Abaqus.

Mimics exactly what the Fortran UMAT will do: open a socket, send a request record per Gauss-point
evaluation, parse the response, and check it against a direct in-process call to material_model.
Proves the wire protocol is correct before any Fortran is involved.

Run (in two terminals, or it self-spawns the server):
    python umat_client_test.py
"""
from __future__ import annotations
import socket
import subprocess
import sys
import time
import numpy as np

import material_model as mm

HOST, PORT = "127.0.0.1", 50097


def request_line(stran, dstran, statev, E, nu):
    toks = [len(statev), *stran, *dstran, *statev, E, nu]
    return " ".join(repr(float(x)) for x in toks) + "\n"


def main():
    srv = subprocess.Popen([sys.executable, "umat_server.py", "--port", str(PORT)])
    try:
        for _ in range(50):  # wait for server
            try:
                s = socket.create_connection((HOST, PORT), timeout=0.2); break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")

        E, nu = 5.0e3, 0.45
        rng_paths = [
            np.array([1e-3, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 2e-3, 0, 0]),
            np.array([5e-4, -5e-4, 0, 0, 1e-3, 0]),
        ]
        stran = np.zeros(6); statev = np.zeros(mm.NSTATEV)
        max_err = 0.0
        with s, s.makefile("rw") as f:
            for k, dstran in enumerate(rng_paths):
                f.write(request_line(stran, dstran, statev, E, nu)); f.flush()
                resp = np.array([float(x) for x in f.readline().split()])
                stress_wire = resp[:6]
                ddsdde_wire = resp[6:42].reshape(6, 6)
                # reference: direct in-process call
                stress_ref, C_ref, _ = mm.evaluate(stran, dstran, statev, {"E": E, "nu": nu})
                err = max(np.max(np.abs(stress_wire - stress_ref)),
                          np.max(np.abs(ddsdde_wire - C_ref)))
                max_err = max(max_err, err)
                print(f" gp-eval {k}: |stress|={np.linalg.norm(stress_wire):9.4f}  "
                      f"wire-vs-direct err={err:.2e}")
                stran = stran + dstran
            f.write("QUIT\n"); f.flush()
        ok = max_err < 1e-8
        print(f"\n bridge protocol {'PASS' if ok else 'FAIL'} (max err {max_err:.2e})")
        sys.exit(0 if ok else 1)
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
