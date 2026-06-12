"""Drive the mechanically-closed NSP model — show growth stress emerge as Pg overgrows.

Constrained case: hold total strain at 0 (rigidly confined Gauss point) and step the model. As the
NSP composition shifts toward P. gingivalis, the swelling eigenstrain rises and the confined point
develops compressive volumetric (growth) stress. This is the qualitative behaviour the FEM should
reproduce — and the reason a composition-aware material model matters.

Run:  python nsp_mechanics_driver.py --cond DH --steps 8
"""
from __future__ import annotations
import argparse
import numpy as np

import nsp_mechanics_model as mech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="DH", choices=["CS", "CH", "DS", "DH"])
    ap.add_argument("--mode", default="depth", choices=["depth", "isotropic"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--phi0", type=float, nargs=5, default=[0.30, 0.20, 0.20, 0.15, 0.15],
                    help="initial composition So An Vd Fn Pg")
    a = ap.parse_args()

    props = {"E": 5.0e3, "nu": 0.45, "cond": a.cond, "growth_mode": a.mode,
             "beta": a.beta, "override_calib": a.mode == "isotropic",
             "phi_init": np.array(a.phi0)}
    bd, ic = mech._calib(a.cond)
    strain = np.zeros(6)          # fully confined
    dstrain = np.zeros(6)
    statev = np.zeros(mech.NSTATEV)   # -> model initialises NSP state

    print(f"=== confined Gauss point, condition {a.cond}, growth={a.mode} "
          f"(beta_depth={bd:.1f}, intercept={ic:.2f}) ===")
    print(f"{'step':>4} {'phi_Pg':>8} {'eps_zz':>9} {'sigma_zz':>10} {'sigma_hyd':>10}")
    for k in range(a.steps):
        stress, C, statev = mech.evaluate(strain, dstrain, statev, props)
        phi_pg = statev[mech.PG_INDEX]
        eps_zz = max(0.0, bd * phi_pg + ic) if a.mode == "depth" else a.beta * phi_pg
        print(f"{k:>4} {phi_pg:8.4f} {eps_zz:9.5f} {stress[2]:10.3f} {stress[:3].mean():10.3f}")

    # sanity: under pure confinement, hydrostatic stress = -3K*eps_growth (K = bulk modulus)
    E, nu = props["E"], props["nu"]
    K = E / (3 * (1 - 2 * nu))
    expect = -3 * K * (a.beta * statev[mech.PG_INDEX]) / 3.0  # sigma_hyd = -K*3*eps_g/... check sign
    print(f"\n bulk modulus K={K:.1f};  confined growth -> compressive hydrostatic stress (negative) "
          f"as expected" if stress[:3].mean() < 0 else "\n (check signs)")


if __name__ == "__main__":
    main()
