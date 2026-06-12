"""Finite-strain driver — show bounded growth stress, and small-strain vs finite-strain.

Run:  python nsp_mechanics_fs_driver.py --cond DH --steps 6
"""
from __future__ import annotations
import argparse
import numpy as np

import nsp_mechanics_fs as fs
from nsp_mechanics_model import _calib

I3 = np.eye(3)


def small_strain_confined(phi_pg, props):
    """Old linearised estimate: confined, sigma_zz = -3K? actually C:(−eps_zz e_zz)."""
    from material_model import elastic_tangent
    bd, ic = _calib(props["cond"])
    eps_zz = max(0.0, bd * phi_pg + ic)
    C = elastic_tangent(props["E"], props["nu"])
    eps = -eps_zz * np.array([0, 0, 1, 0, 0, 0.0])
    return (C @ eps)[2]  # sigma_zz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="DH", choices=["CS", "CH", "DS", "DH"])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--phi0", type=float, nargs=5, default=[0.30, 0.20, 0.20, 0.15, 0.15])
    a = ap.parse_args()
    props = {"E": 5.0e3, "nu": 0.45, "cond": a.cond, "phi_init": np.array(a.phi0)}
    mu, _ = fs.lame(props["E"], props["nu"])

    statev = np.zeros(fs.NSTATEV)
    print(f"=== finite-strain F=Fe.Fg, condition {a.cond} (mu={mu:.0f}) ===")
    print(f"{'step':>4} {'phi_Pg':>7} {'lam_z':>6} | "
          f"{'CONFINED sig_zz/mu':>18} {'(small-strain sig_zz/mu)':>24} | "
          f"{'FREE-TOP sig_xx/mu':>18}")
    for k in range(a.steps):
        statev, phi_pg = fs.advance(statev, props)
        # confined finite-strain
        s_conf, lam_z = fs.stress_from_F(I3.copy(), phi_pg, props)
        # small-strain confined (for contrast)
        ss = small_strain_confined(phi_pg, props)
        # free-top substrate-bonded film
        s_free, Fzz, _ = fs.relax_free_top(phi_pg, props)
        print(f"{k:>4} {phi_pg:7.4f} {lam_z:6.3f} | "
              f"{s_conf[2,2]/mu:18.3f} {ss/mu:24.1f} | "
              f"{s_free[0,0]/mu:18.5f}")

    print("\n Reading: confined growth costs ~15-20*mu in BOTH formulations (140% growth can't be "
          "confined\n cheaply) — but only F=Fe.Fg is kinematically valid at this stretch. The "
          "realistic flow-chamber\n BC is free-top: z-only growth is ~stress-free -> biofilms "
          "thicken without mechanical failure.\n Meaningful residual/delamination stress needs "
          "spatial growth gradients -> the FEM job, not 1 point.")


if __name__ == "__main__":
    main()
