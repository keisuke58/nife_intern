"""End-to-end test of the NSP-backed bridge WITHOUT Abaqus — STATEV round-trip + growth stress.

Mimics what umat.f does across a load history: send STRAN/DSTRAN/STATEV per increment, feed the
returned STATEV(12) back in (as Abaqus carries SDVs between increments), and check the wire result
matches a direct in-process call. Confirms (a) the 12-dim NSP state persists correctly over the
socket and (b) confined growth stress emerges as Pg evolves.

Run:  python umat_client_nsp_test.py
"""
from __future__ import annotations
import socket
import subprocess
import sys
import time
import numpy as np

import nsp_mechanics_model as mech

HOST, PORT = "127.0.0.1", 50098
COND, BETA = "DH", 0.05
PHI0 = [0.30, 0.20, 0.20, 0.15, 0.15]


def req_line(stran, dstran, statev, E, nu):
    toks = [len(statev), *stran, *dstran, *statev, E, nu]
    return " ".join(repr(float(x)) for x in toks) + "\n"


def main():
    srv = subprocess.Popen([sys.executable, "umat_server_nsp.py", "--port", str(PORT),
                            "--cond", COND, "--beta", str(BETA),
                            "--phi0", *map(str, PHI0)])
    try:
        for _ in range(80):
            try:
                s = socket.create_connection((HOST, PORT), timeout=0.2); break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")
        s.settimeout(60)   # first NSP eval triggers JAX JIT compile; don't time out reads

        E, nu = 5.0e3, 0.45
        props = {"E": E, "nu": nu, "beta": BETA, "cond": COND, "phi_init": np.array(PHI0)}
        stran = np.zeros(6); dstran = np.zeros(6)
        sv_wire = np.zeros(mech.NSTATEV)      # server initialises from PHI0 on first call
        sv_ref = np.zeros(mech.NSTATEV)
        max_err = 0.0
        print(f"{'inc':>3} {'phi_Pg':>8} {'sigma_hyd':>10}  wire-vs-direct")
        with s, s.makefile("rw") as f:
            for k in range(6):
                f.write(req_line(stran, dstran, sv_wire, E, nu)); f.flush()
                resp = np.array([float(x) for x in f.readline().split()])
                stress_w = resp[:6]; sv_wire = resp[42:42 + mech.NSTATEV]
                # in-process reference with its own carried state
                stress_r, _, sv_ref = mech.evaluate(stran, dstran, sv_ref, props)
                err = max(np.max(np.abs(stress_w - stress_r)),
                          np.max(np.abs(sv_wire - sv_ref)))
                max_err = max(max_err, err)
                print(f"{k:>3} {sv_wire[mech.PG_INDEX]:8.4f} {stress_w[:3].mean():10.3f}  {err:.2e}")
            f.write("QUIT\n"); f.flush()
        ok = max_err < 1e-8
        print(f"\n NSP bridge + STATEV round-trip {'PASS' if ok else 'FAIL'} (max err {max_err:.2e})")
        sys.exit(0 if ok else 1)
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
