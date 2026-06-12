"""Side-by-side: gLV vs NSP material model at a Gauss point — response + tangent check.

Both load the same Heine fit (A, b). For each probe composition we report the one-step update and
verify each model's AD consistent tangent against finite differences. This is the comparison the
thesis needs: same data, two constitutive laws, both FEM-ready (consistent tangent), differing
state size (gLV: 5, NSP: 12) and stiffness.

Run:  python compare_glv_nsp.py --cond DH
"""
from __future__ import annotations
import argparse
import numpy as np
import jax.numpy as jnp

import glv_material_model as glv
import nsp_material_model as nm


def fd_jac(step, x, h=1e-6):
    x = np.asarray(x, float); n = x.size
    J = np.zeros((n, n))
    for j in range(n):
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        J[:, j] = (np.asarray(step(jnp.array(xp))) - np.asarray(step(jnp.array(xm)))) / (2 * h)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="DH", choices=["CS", "CH", "DS", "DH"])
    a = ap.parse_args()

    probes = {
        "even":            np.full(5, 0.2),
        "So-dominant":     np.array([0.70, 0.10, 0.10, 0.05, 0.05]),
        "Pg-enriched(DH)": np.array([0.15, 0.10, 0.10, 0.25, 0.40]),
    }

    # gLV
    glv_step, glv_tan, (A, b) = glv.make_evaluator(a.cond)
    # NSP
    params = nm.load_params(a.cond)
    nsp_step, nsp_tan = nm.make_evaluator(params)

    print(f"=== condition {a.cond} ===  A_diag={np.asarray(A).diagonal().round(3)}")
    print(f"{'probe':18} {'model':5} {'phi_out (Pg last)':38} {'state':6} {'tan rel-err':>11}")
    worst = {"gLV": 0.0, "NSP": 0.0}
    for name, phi in probes.items():
        # gLV
        pg = np.asarray(glv_step(jnp.array(phi)))
        Jg, Jgfd = np.asarray(glv_tan(jnp.array(phi))), fd_jac(glv_step, phi)
        eg = np.max(np.abs(Jg - Jgfd)) / (np.max(np.abs(Jg)) + 1e-30)
        worst["gLV"] = max(worst["gLV"], eg)
        print(f"{name:18} {'gLV':5} {str(pg.round(4)):38} {'5':6} {eg:11.2e}")
        # NSP
        g = nm.make_initial_state(phi)
        gn = np.asarray(nsp_step(g)); pn = gn[:5]
        Jn, Jnfd = np.asarray(nsp_tan(g)), fd_jac(nsp_step, g)
        en = np.max(np.abs(Jn - Jnfd)) / (np.max(np.abs(Jn)) + 1e-30)
        worst["NSP"] = max(worst["NSP"], en)
        print(f"{'':18} {'NSP':5} {str(pn.round(4)):38} {'12':6} {en:11.2e}")
    print(f"\n tangent check: gLV max rel-err {worst['gLV']:.2e}, "
          f"NSP max rel-err {worst['NSP']:.2e}")
    ok = worst["gLV"] < 1e-4 and worst["NSP"] < 1e-4
    print(f" => {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
