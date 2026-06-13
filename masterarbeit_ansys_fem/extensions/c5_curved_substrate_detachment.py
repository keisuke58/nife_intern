"""C5 — Curved-substrate (implant-thread) delamination driving force + an absolute detachment criterion.

Every mechanics model in the thesis (column, film, b4) sits on a FLAT substratum. A dental implant is
not flat: it has a threaded, micro-rough titanium surface with radii of curvature from ~mm (abutment /
implant body) down to ~10-50 um (thread crests, micro-thread / SLA roughness). This script adds the two
missing pieces:

  (#3 curvature) A growing biofilm bonded to a CURVED substrate stores BENDING strain energy in addition
      to the in-plane membrane (residual-stress) energy. Using thin-film delamination mechanics
      (Hutchinson & Suo 1992), the steady-state energy release rate driving delamination is
          G_total(R) = G_membrane + G_bending
                     = sigma^2 h /(2 Ebar)  +  Ebar h^3 /(24 R^2),
      Ebar = E/(1-nu^2). The curvature amplification is scale-free in the modulus:
          A(R) = G_total/G_membrane = 1 + h^2 /(12 R^2 eps_res^2),   sigma = Ebar*eps_res.
      So when R drops to the film-thickness scale (implant threads, h/R ~ 1), bending energy can rival or
      exceed the membrane term -> implant surface geometry materially changes the detachment driving force.

  (#4 criterion) Delamination occurs when G_total >= Gamma, the biofilm-titanium interface toughness
      (work of adhesion). Literature biofilm/cell adhesion energies are LOW, ~1e-3 to 1e-1 J/m^2. Because
      the biofilm modulus spans Pa (long-time rheology) to ~MPa (short-time AFM), the ABSOLUTE prediction
      is uncertain; we therefore report (a) the scale-free DH/CH driving-force ratio and curvature
      amplification, and (b) a detachment REGIME diagram over (substrate radius R, biofilm modulus G0)
      with the literature Gamma band, rather than a single false-precision number.

Real-data inputs: per-condition film thickness h (CLSM, beta_calibration.json) and a per-condition
residual in-plane mismatch strain eps_res (peak confined elastic strain from the CLSM-calibrated depth
growth, via b4). DH packs Pg near the substratum -> larger eps_res AND larger h -> larger G_total.

Run:  python masterarbeit_ansys_fem/extensions/c5_curved_substrate_detachment.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "masterarbeit_ansys_fem" / "extensions"))
from thesis_style import use  # noqa: E402
from b4_viscoelastic_growth_stress import depth_phi_pg, beta_calib, NU  # noqa: E402

RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
Z0 = 0.25                                   # interface confinement length (consistent with b4 / c3)

# Implant-geometry curvature scales (radius of curvature, um)
GEOM = [("micro-thread /\nSLA roughness", 20.0), ("thread crest", 120.0),
        ("implant body /\nabutment", 2000.0)]
# Biofilm-titanium interface toughness band (work of adhesion), J/m^2 (low, like cell/biofilm adhesion)
GAMMA_LO, GAMMA_HI = 1e-3, 1e-1


def residual_strain_and_h(cond):
    """Per-condition (eps_res, h_um) from CLSM: peak confined elastic strain + endpoint thickness."""
    bd, ic = beta_calib()["DH"]["beta_depth"], beta_calib()["DH"]["intercept"]   # Pg-driven law (see b4)
    zn, pg = depth_phi_pg(cond)
    eps_g = np.clip(bd * pg + ic, 0.0, None)
    confine = np.exp(-zn / Z0)
    eps_res = float(np.max(eps_g * confine))            # peak confined (locked-in) mismatch strain
    h_um = beta_calib()[cond]["h_um_range"][1]          # endpoint biofilm thickness
    return eps_res, h_um


def G_components(eps_res, h_um, R_um, G0):
    """Thin-film delamination energy release rate (J/m^2). h,R in um -> m; G0 (shear mod) in Pa.
    Ebar = E/(1-nu^2), E = 2 G0 (1+nu); sigma = Ebar*eps_res (equibiaxial residual)."""
    h = h_um * 1e-6
    R = R_um * 1e-6
    Ebar = 2 * G0 * (1 + NU) / (1 - NU ** 2)
    sigma = Ebar * eps_res
    G_mem = sigma ** 2 * h / (2 * Ebar)                 # = Ebar*eps_res^2*h/2
    G_bend = Ebar * h ** 3 / (24 * R ** 2)
    return G_mem, G_bend


def main():
    props = {c: residual_strain_and_h(c) for c in ("DH", "CH")}
    print("Per-condition CLSM inputs (residual mismatch strain, endpoint thickness):")
    for c in ("DH", "CH"):
        print("  %s: eps_res=%.3f  h=%.1f um" % (c, props[c][0], props[c][1]))

    # scale-free curvature amplification A(R) = 1 + h^2/(12 R^2 eps_res^2)
    R = np.logspace(np.log10(10), np.log10(3000), 400)   # um
    fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.94, aspect=0.46))

    for c, col in (("DH", RED), ("CH", BLUE)):
        eps_res, h = props[c]
        A = 1.0 + h ** 2 / (12 * R ** 2 * eps_res ** 2)
        a1.loglog(R, A, "-", color=col, lw=1.5, label="%s ($h{=}%.0f\\,\\mu m$, $\\varepsilon{=}%.2f$)"
                  % (c, h, eps_res))
    a1.axhline(2.0, ls=":", color="0.5", lw=0.8)
    a1.text(2200, 2.1, "bending =\nmembrane", fontsize=5.6, color="0.4", ha="right")
    ymax = a1.get_ylim()[1]
    for lab, Rg in GEOM:
        a1.axvline(Rg, ls="--", color="0.7", lw=0.7)
        a1.text(Rg, ymax * 0.6, lab, rotation=90, fontsize=5.2, color="0.45", va="top", ha="right")
    a1.set_xlabel(r"substrate radius of curvature $R$ ($\mu$m)")
    a1.set_ylabel(r"curvature amplification $G_{\mathrm{total}}/G_{\mathrm{membrane}}$")
    a1.set_title(r"Curvature amplifies driving force", fontsize=8)
    a1.legend(fontsize=5.8, loc="upper right")

    # detachment regime: (R, G0) plane; contour G_total = Gamma. Use DH (worst case) + CH curves.
    G0grid = np.logspace(0, 6, 240)          # biofilm shear modulus Pa -> MPa
    print("\nDetachment regime (G_total vs interface toughness Gamma=%.0e..%.0e J/m^2):" % (GAMMA_LO, GAMMA_HI))
    for c, col, ls in (("DH", RED, "-"), ("CH", BLUE, "--")):
        eps_res, h = props[c]
        # critical modulus G0* where G_total(R,G0*) = Gamma, per R (use Gamma_HI = harder interface)
        G0star_lo, G0star_hi = [], []
        for Rg in R:
            Gm, Gb = G_components(eps_res, h, Rg, 1.0)   # per unit G0 (G scales linearly with G0)
            Gtot_per_G0 = Gm + Gb
            G0star_lo.append(GAMMA_LO / Gtot_per_G0)
            G0star_hi.append(GAMMA_HI / Gtot_per_G0)
        a2.fill_between(R, G0star_lo, G0star_hi, color=col, alpha=0.18)
        a2.loglog(R, G0star_hi, ls, color=col, lw=1.4, label="%s detach threshold" % c)
        # report at thread crest
        Gm, Gb = G_components(eps_res, h, 120.0, 1.0)
        print("  %s @ thread crest R=120um: detaches if G0 >~ %.1e..%.1e Pa  (membrane/bending=%.2f)"
              % (c, GAMMA_LO / (Gm + Gb), GAMMA_HI / (Gm + Gb), Gm / Gb))

    a2.axhspan(1.0, 50.0, color="0.8", alpha=0.3)
    a2.text(12, 7, "long-time rheology\n$G_0\\sim$O(Pa)", fontsize=5.2, color="0.35")
    a2.axhspan(1e5, 3e6, color="0.8", alpha=0.3)
    a2.text(12, 4e5, "short-time AFM\n$G_0\\sim$MPa", fontsize=5.2, color="0.35")
    for lab, Rg in GEOM:
        a2.axvline(Rg, ls="--", color="0.7", lw=0.7)
    a2.set_xlabel(r"substrate radius of curvature $R$ ($\mu$m)")
    a2.set_ylabel(r"biofilm modulus $G_0$ (Pa) at detachment")
    a2.set_title(r"Detachment regime (DH detaches $13\times$ easier)", fontsize=8)
    a2.legend(fontsize=5.8, loc="lower left")
    a2.set_ylim(1, 3e6)

    # scale-free DH/CH ratio of total driving force at thread crest
    eD, hD = props["DH"]; eC, hC = props["CH"]
    GmD, GbD = G_components(eD, hD, 120.0, 1.0); GmC, GbC = G_components(eC, hC, 120.0, 1.0)
    print("\nDH/CH total driving-force ratio @ thread crest = %.2f  (scale-free in G0)"
          % ((GmD + GbD) / (GmC + GbC)))

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "F20_curved_substrate_detachment.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", OUT / "F20_curved_substrate_detachment.pdf")


if __name__ == "__main__":
    main()
