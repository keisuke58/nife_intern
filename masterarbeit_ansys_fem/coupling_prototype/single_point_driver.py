"""Single-Gauss-point constitutive driver — de-risk the material model with NO FEM.

This is prototyping rung 1: drive `material_model.evaluate` along prescribed strain paths and
verify (a) the stress response is sane and (b) the returned consistent tangent DDSDDE matches a
finite-difference of the stress. A material model that fails the tangent check will wreck Newton
convergence inside ANSYS/Abaqus, so catch it here — long before any cross-language coupling.

Run:  python single_point_driver.py
"""
from __future__ import annotations
import numpy as np
import material_model as mm

PROPS = {"E": 5.0e3, "nu": 0.45}  # placeholder: soft, near-incompressible (biofilm-ish)


def fd_tangent(strain, dstrain, state, props, h=1e-7):
    """Numerical DDSDDE = d(stress)/d(dstrain) by central differences."""
    n = 6
    J = np.zeros((n, n))
    for j in range(n):
        dp = dstrain.copy(); dp[j] += h
        dm = dstrain.copy(); dm[j] -= h
        sp, _, _ = mm.evaluate(strain, dp, state, props)
        sm, _, _ = mm.evaluate(strain, dm, state, props)
        J[:, j] = (sp - sm) / (2.0 * h)
    return J


def run_path(name, strain_path):
    print(f"\n=== strain path: {name} ===")
    state = np.zeros(mm.NSTATEV)
    strain = np.zeros(6)
    max_err = 0.0
    for k, dstrain in enumerate(strain_path):
        dstrain = np.asarray(dstrain, float)
        stress, C, state = mm.evaluate(strain, dstrain, state, props=PROPS)
        Cfd = fd_tangent(strain, dstrain, state, PROPS)
        err = np.max(np.abs(C - Cfd)) / (np.max(np.abs(C)) + 1e-30)
        max_err = max(max_err, err)
        strain = strain + dstrain
        print(f" step {k}: |stress|={np.linalg.norm(stress):10.4f}  "
              f"tangent rel-err={err:.2e}")
    ok = max_err < 1e-5
    print(f" --> tangent check {'PASS' if ok else 'FAIL'} (max rel-err {max_err:.2e})")
    return ok


def main():
    de = 1e-3
    paths = {
        "uniaxial-11":  [[de, 0, 0, 0, 0, 0]] * 5,
        "hydrostatic":  [[de, de, de, 0, 0, 0]] * 5,
        "simple-shear-12": [[0, 0, 0, de, 0, 0]] * 5,
    }
    results = {n: run_path(n, p) for n, p in paths.items()}
    print("\n=== summary ===")
    for n, ok in results.items():
        print(f"  {n:18} {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
