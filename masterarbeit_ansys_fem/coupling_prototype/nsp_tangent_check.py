"""Rung-1 tangent check for the REAL Heine NSP material model.

Verifies, at a Gauss point, that the AD consistent tangent  J = d g_new / d g_prev  of the NSP
implicit-Euler step matches a finite-difference Jacobian. A mismatch would break the outer FEM
Newton iteration once this model replaces the phenomenological law.

Run:  python nsp_tangent_check.py            # default condition DH
      python nsp_tangent_check.py --cond CS
"""
from __future__ import annotations
import argparse
import numpy as np
import jax.numpy as jnp

import nsp_material_model as nm


def fd_jacobian(step, g, h=1e-6):
    g = np.asarray(g, float)
    n = g.size
    J = np.zeros((n, n))
    for j in range(n):
        gp = g.copy(); gp[j] += h
        gm = g.copy(); gm[j] -= h
        J[:, j] = (np.asarray(step(jnp.array(gp))) - np.asarray(step(jnp.array(gm)))) / (2 * h)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="DH", choices=["CS", "CH", "DS", "DH"])
    a = ap.parse_args()

    params = nm.load_params(a.cond)
    step, tangent = nm.make_evaluator(params)

    A = np.asarray(params["A"])
    print(f"=== Heine NSP material model — condition {a.cond} ===")
    print(f" A diag = {A.diagonal().round(3)}")
    print(f" b      = {np.asarray(params['b_diag']).round(3)}")

    # a few plausible compositions to probe (5 species, sum<=1)
    probes = {
        "even":            np.full(N := 5, 1.0 / 5),
        "So-dominant":     np.array([0.70, 0.10, 0.10, 0.05, 0.05]),
        "Pg-enriched(DH)": np.array([0.15, 0.10, 0.10, 0.25, 0.40]),
    }

    max_err = 0.0
    for name, phi in probes.items():
        g = nm.make_initial_state(phi)
        g_new = np.asarray(step(g))
        J_ad = np.asarray(tangent(g))
        J_fd = fd_jacobian(step, g)
        scale = np.max(np.abs(J_ad)) + 1e-30
        err = np.max(np.abs(J_ad - J_fd)) / scale
        max_err = max(max_err, err)
        phi_new = g_new[:5]
        print(f"\n probe '{name}':")
        print(f"   phi_in  = {phi.round(3)}")
        print(f"   phi_out = {phi_new.round(4)}  (sum={phi_new.sum():.4f})")
        print(f"   tangent J: shape {J_ad.shape}, ‖J‖={scale:.3e}, "
              f"AD-vs-FD rel-err = {err:.2e}")

    ok = max_err < 1e-4
    print(f"\n=== tangent check {'PASS' if ok else 'FAIL'} "
          f"(max AD-vs-FD rel-err {max_err:.2e}) ===")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
