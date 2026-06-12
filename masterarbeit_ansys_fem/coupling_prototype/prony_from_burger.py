#!/usr/bin/env python3
# [nife-pathshim]
"""Burgers (creep) -> Prony (relaxation) series for the B3 biofilm *VISCOELASTIC deck.

Biofilm rheology is reported as a 4-element **Burgers creep** fit: a Maxwell element (spring G_M,
dashpot eta_M) in series with a Kelvin-Voigt element (spring G_K, dashpot eta_K). Stoodley/Shaw
rheometer creep gives shear modulus G ~ 0.2-24 Pa and viscosity eta ~ 10-3000 Pa.s (Biofouling 2004;
J. Bacteriol. 2019 review). Abaqus *VISCOELASTIC wants a **Prony relaxation** series g_i(tau_i)
normalized by the instantaneous modulus G_0.

The Burgers relaxation modulus is **exactly a two-term Prony series** (no fitting needed). In Laplace
space, with the standard operator form  sigma + p1 sigma' + p2 sigma'' = q1 eps' + q2 eps'':
    p1 = eta_M/G_M + eta_M/G_K + eta_K/G_K
    p2 = eta_M*eta_K/(G_M*G_K)
    q1 = eta_M
    q2 = eta_M*eta_K/G_K
    G_hat(s) = (q2 s + q1)/(p2 s^2 + p1 s + 1)
Poles at s=-r_i, r_i = roots of p2 r^2 - p1 r + 1 = 0 (both real, positive). Residues give
    G(t) = C1 exp(-t/lambda1) + C2 exp(-t/lambda2),  lambda_i = 1/r_i,
    C_i = (q1 - q2 r_i)/(p2 (r_j - r_i)),  C1+C2 = G_M (instantaneous modulus, checked).
Prony: g_i = C_i/G_M, tau_i = lambda_i, g_inf = 0 (Burgers is a fluid -> fully relaxes).

NOTE on scale: the *normalized* Prony shape (g_i, tau_i) is what transfers; the absolute modulus is
set by *ELASTIC (instantaneous G_0). Long-time rheology gives G_0 ~ O(Pa); short-time AFM gives
E ~ 1.4-2.8 MPa (oral biofilm, PMC8355335). Pick G_0 per the timescale you model; report sigma/mu.

Usage:
    python prony_from_burger.py                                  # representative biofilm set
    python prony_from_burger.py --GM 5 --etaM 3000 --GK 10 --etaK 100
Writes prony_biofilm.json and prints the *VISCOELASTIC, TIME=PRONY block.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def burgers_prony(GM, etaM, GK, etaK):
    """Exact 2-term Prony of the Burgers relaxation modulus. Returns (g_list, tau_list, G0)."""
    p1 = etaM / GM + etaM / GK + etaK / GK
    p2 = etaM * etaK / (GM * GK)
    q1 = etaM
    q2 = etaM * etaK / GK
    disc = p1 * p1 - 4.0 * p2
    if disc < 0:
        raise ValueError("complex relaxation times (unexpected for Burgers)")
    sq = np.sqrt(disc)
    r1 = (p1 + sq) / (2.0 * p2)
    r2 = (p1 - sq) / (2.0 * p2)
    C1 = (q1 - q2 * r1) / (p2 * (r2 - r1))
    C2 = (q1 - q2 * r2) / (p2 * (r1 - r2))
    G0 = C1 + C2                      # == GM
    g = [C1 / G0, C2 / G0]
    tau = [1.0 / r1, 1.0 / r2]
    # sort by ascending tau for readability
    order = np.argsort(tau)
    g = [g[i] for i in order]
    tau = [tau[i] for i in order]
    return g, tau, G0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--GM", type=float, default=5.0, help="Maxwell shear modulus [Pa]")
    ap.add_argument("--etaM", type=float, default=3000.0, help="Maxwell viscosity [Pa.s]")
    ap.add_argument("--GK", type=float, default=10.0, help="Kelvin shear modulus [Pa]")
    ap.add_argument("--etaK", type=float, default=100.0, help="Kelvin viscosity [Pa.s]")
    ap.add_argument("--ginf", type=float, default=0.05,
                    help="long-term elastic shear floor g_inf (Abaqus needs >0; raw Burgers=0). "
                         "g_i are rescaled to sum=1-ginf. Use 0.0 for the exact fluid Burgers Prony.")
    args = ap.parse_args()

    g, tau, G0 = burgers_prony(args.GM, args.etaM, args.GK, args.etaK)
    # Burgers is a fluid (sum g_i = 1, g_inf = 0). Abaqus *VISCOELASTIC requires a long-term
    # elastic floor g_inf > 0 (a solid). Rescale the (deviatoric) weights to sum = 1 - ginf,
    # keeping the relaxation SHAPE (tau_i, relative weights) from the rheology.
    if args.ginf > 0:
        s = sum(g)
        g = [gi / s * (1.0 - args.ginf) for gi in g]
    g_inf = 1.0 - sum(g)

    out = {
        "burgers": {"GM": args.GM, "etaM": args.etaM, "GK": args.GK, "etaK": args.etaK},
        "tau_K_s": args.etaK / args.GK, "tau_M_s": args.etaM / args.GM,
        "G0_instantaneous_Pa": float(G0), "g_inf": float(g_inf),
        "prony": [{"g_i": float(gi), "tau_i_s": float(ti)} for gi, ti in zip(g, tau)],
    }
    (HERE / "prony_biofilm.json").write_text(json.dumps(out, indent=2))

    print(f"# Burgers -> Prony (exact 2-term).  G0_instantaneous = {G0:.4g} Pa (= G_M), "
          f"g_inf = {g_inf:.3g}")
    print(f"# retardation tau_K = {args.etaK/args.GK:.4g} s, Maxwell tau_M = {args.etaM/args.GM:.4g} s")
    print("*VISCOELASTIC, TIME=PRONY")
    for gi, ti in zip(g, tau):
        print(f"{gi:.6f}, 0.0, {ti:.6g}")   # g_shear_i, g_bulk_i(=0, ~incompressible), tau_i
    print(f"# columns: g_shear_i, g_bulk_i(=0: deviatoric-only, biofilm ~incompressible), tau_i[s].")
    print(f"# sum g_shear_i = {sum(g):.6f}, long-term floor g_inf = {g_inf:.4f} (>0 required by Abaqus).")


if __name__ == "__main__":
    main()
