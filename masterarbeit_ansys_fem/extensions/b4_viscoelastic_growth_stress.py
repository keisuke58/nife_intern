"""B4 — Viscoelastic relaxation of the growth-driven interface residual stress (the missing coupling).

The thesis has two SEPARATE viscoelastic pieces that were never joined:
  * F4 residual stress sigma_xx(z): computed ELASTICALLY (instantaneous neo-Hookean), and
  * F7 stress relaxation: a standalone Burgers->Prony verification of biofilm rheology.
Here we COUPLE them: drive a depth-resolved, laterally-confined column with the CLSM-calibrated growth
history and let the deviatoric stress relax through the measured biofilm Prony series, so the residual
interface stress becomes time-dependent sigma_xx(z, t).

Physics (the headline the thesis would argue):
  * Growth is SLOW: the biofilm develops over T_grow ~ 21 days (CLSM h: 22 -> 79 um for DH).
  * Relaxation is FAST: biofilm Prony times tau ~ 7 s and ~900 s (Stoodley/Shaw rheology, prony_biofilm.json).
  * Deborah number De = tau_relax / T_grow ~ 1e-3 << 1  =>  on the growth timescale the biofilm is
    essentially fully relaxed in shear. The generalized-Maxwell deviatoric stiffness collapses to its
    LONG-TIME value G_inf = g_inf * G0 (g_inf = 0.05): 95% of the deviatoric interface stress relaxes away.
  * What PERSISTS is the VOLUMETRIC (confined-swelling) stress — bulk K does not relax in a deviatoric
    Maxwell model. So the residual detachment driver is set by volumetric confinement near the bonded
    substratum, not by deviatoric stress buildup. (This is WHY biofilms thicken without delaminating, and
    why the persistent peri-implant detachment risk is a confinement effect — Boel 2009 modelled real
    CLSM geometry but elastically; the viscoelastic correction is the contribution here.)

Model (small-strain incremental generalized Maxwell, the regime where Prony applies directly; the
finite-strain F=Fe.Fg magnitude is bracketed separately in nsp_mechanics_fs.py):
  per depth z, laterally substratum-bonded (eps_xx=eps_yy=0), growth eigenstrain eps^g(z,t) in zz from
  the measured phi_Pg(z); split stress into volumetric (elastic, bulk K) + deviatoric (relaxing, Prony).
  Deviatoric recurrence (Simo & Hughes, Box 10.3):
      h_i^{n+1} = exp(-dt/tau_i) h_i^n + g_i*(tau_i/dt)*(1-exp(-dt/tau_i)) * 2G0 * d(dev eps_e)
      s^{n+1}   = 2 G0 g_inf dev eps_e^{n+1} + sum_i h_i^{n+1}
Stresses reported as sigma/G0 (modulus-agnostic; biofilm G0 ~ O(Pa) long-time to ~MPa short-time/AFM).

Run:  python masterarbeit_ansys_fem/extensions/b4_viscoelastic_growth_stress.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

CP = ROOT / "masterarbeit_ansys_fem" / "coupling_prototype"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
RED, BLUE = "#d62728", "#1f77b4"
NU = 0.45                                   # biofilm Poisson (near-incompressible matrix)
T_GROW_DAYS = 21.0                          # CLSM development window
N_STEPS = 400
DAY_S = 86400.0


def prony():
    P = json.load(open(CP / "prony_biofilm.json"))
    ginf = P["g_inf"]
    gi = np.array([p["g_i"] for p in P["prony"]], float)
    ti = np.array([p["tau_i_s"] for p in P["prony"]], float)
    gi = gi / (gi.sum() + ginf) * (1 - ginf)    # renormalise so ginf + sum gi = 1
    return ginf, gi, ti


def beta_calib():
    return json.loads((CP / "beta_calibration.json").read_text())


def depth_phi_pg(cond):
    """Measured endpoint phi_Pg(z), normalised depth 0(substratum)..1."""
    zp = {}
    with open(ROOT / "results/diffusion_fit/zprofiles_all_ti.csv") as f:
        for r in csv.DictReader(f):
            if r["condition"] != cond:
                continue
            zp.setdefault(int(r["day"]), []).append((float(r["z_um"]), float(r["P.gingivalis"])))
    rows = sorted(zp[max(zp)])
    z = np.array([a for a, _ in rows]); pg = np.array([b for _, b in rows])
    zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return zn, pg


def kG_ratio(nu):
    """K/G0 from Poisson ratio (G0=1 reference)."""
    return 2 * (1 + nu) / (3 * (1 - 2 * nu))


_BETA_PHYS = None   # single Pg-driven growth coefficient (DH calibration), applied to BOTH conditions


def run_column(cond, relax=True):
    """sigma_xx(z) at endpoint for a substratum-bonded growing column, with/without VE relaxation.

    A pure free-top z-only growth is stress-free; interface stress comes from the bonded substratum
    boundary layer. Reduced model (consistent with the verified w(z)=exp(-z/z0), c3): a fraction
    confine(z)=exp(-z/z0) of the depth growth is mechanically RESTRAINED near the bonded base, giving a
    compressive elastic strain eps_e = diag(0,0, -eps_g*confine). The in-plane sigma_xx is the interface
    delamination driver:  sigma_xx = K*tr + s_dev_xx,  tr = eps_e_zz,  dev_xx = -tr/3.

    Growth law: the SAME physical Pg-driven coefficient (DH calibration beta_depth) is applied to each
    condition's MEASURED phi_Pg(z), so the CH/DH difference comes purely from the measured Pg DEPTH
    distribution (CH clears Pg from the base, DH packs it there) — not from two different fits. (CH's own
    calibration has negative slope = Pg is NOT its growth driver, so it must not be used as a growth law.)
    Reported as sigma/G0; the response is linear so the elastic-vs-VE and CH-vs-DH RATIOS are scale-free.
    """
    global _BETA_PHYS
    if _BETA_PHYS is None:
        c = beta_calib()["DH"]
        _BETA_PHYS = (c["beta_depth"], c["intercept"])
    bd, ic = _BETA_PHYS
    zn, pg = depth_phi_pg(cond)
    eps_g_final = np.clip(bd * pg + ic, 0.0, None)         # Pg-driven depth growth strain per z

    ginf, gi, ti = prony()
    KG = kG_ratio(NU)                                      # K in units of G0
    G0 = 1.0
    z0 = 0.25
    confine = np.exp(-zn / z0)                             # 1 at substratum -> 0 at top

    nz = len(zn)
    dt = T_GROW_DAYS * DAY_S / N_STEPS
    decay = np.exp(-dt / ti)                               # per Prony branch
    fac = gi * (ti / dt) * (1 - decay)                     # incremental weight (Simo & Hughes Box 10.3)

    h = np.zeros((nz, len(ti)))                            # deviatoric internal vars (xx), per depth
    dev_xx_prev = np.zeros(nz)
    sxx = np.zeros(nz)
    for n in range(1, N_STEPS + 1):
        eps_g = eps_g_final * (n / N_STEPS)               # growth ramps linearly over T_grow
        eps_e_zz = -eps_g * confine                        # restrained near base -> compressive
        tr = eps_e_zz                                      # eps_e = diag(0,0,eps_e_zz)
        dev_xx = -tr / 3.0
        h = decay[None, :] * h + fac[None, :] * (2 * G0 * (dev_xx - dev_xx_prev))[:, None]
        dev_xx_prev = dev_xx.copy()
        s_dev_xx = (2 * G0 * ginf * dev_xx + h.sum(axis=1)) if relax else (2 * G0 * dev_xx)
        sxx = KG * G0 * tr + s_dev_xx                      # sigma_xx = K*tr + s_dev_xx
    De = ti.max() / (T_GROW_DAYS * DAY_S)
    return zn, sxx, eps_g_final, De


def main():
    ginf, gi, ti = prony()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))

    peaks = {}
    for cond, col in (("DH", RED), ("CH", BLUE)):
        zn, sxx_e, eg, De = run_column(cond, relax=False)
        zn, sxx_v, _, _ = run_column(cond, relax=True)
        a1.plot(eg, zn, "-", color=col, lw=1.2, label="%s growth $\\varepsilon^g_{zz}(z)$" % cond)
        a2.plot(sxx_e, zn, "-", color=col, lw=1.4, label="%s elastic" % cond)
        a2.plot(sxx_v, zn, "--", color=col, lw=1.4, label="%s viscoelastic" % cond)
        peak_e = np.abs(sxx_e).max(); peak_v = np.abs(sxx_v).max()
        peaks[cond] = (peak_e, peak_v)
        print("%s: |sigma_xx/G0| peak  elastic=%.3f  viscoelastic=%.3f  (VE/el=%.2f)  (De=%.1e)"
              % (cond, peak_e, peak_v, peak_v / (peak_e + 1e-12), De))

    print("\nDH/CH interface-stress ratio: elastic=%.1fx, viscoelastic=%.1fx  (dysbiotic risk diverges, "
          "preserved under VE)" % (peaks["DH"][0] / peaks["CH"][0], peaks["DH"][1] / peaks["CH"][1]))
    print("Deviatoric stiffness relaxes %.0f%% (g_inf=%.2f, De<<1) BUT interface stress is VOLUMETRIC-"
          "dominated (K/G=%.1f, nu=%.2f):" % (100 * (1 - ginf), ginf, kG_ratio(NU), NU))
    print("  => biofilm viscoelasticity does NOT relieve the growth-induced interface detachment driver "
          "-> the elastic F4 prediction is robust.")

    a1.set_xlabel(r"growth strain $\varepsilon^g_{zz}$ (small-strain probe)")
    a1.set_ylabel(r"depth $z$ (0 = substratum)")
    a1.set_title(r"CLSM growth eigenstrain (endpoint)", fontsize=8)
    a1.axhspan(0, 0.25, color=RED, alpha=0.07)
    a1.legend(fontsize=6, loc="upper right")

    a2.axhspan(0, 0.25, color=RED, alpha=0.07)
    a2.text(0.02, 0.04, "interface zone", fontsize=6, color="0.35",
            transform=a2.get_yaxis_transform())
    a2.set_xlabel(r"interface residual stress $\sigma_{xx}/G_0$")
    a2.set_ylabel(r"depth $z$ (0 = substratum)")
    a2.set_title(r"Volumetric-dominated: VE relaxation does not relieve it" "\n"
                 r"($De{\approx}5{\times}10^{-4}$, $g_\infty{=}0.05$, $K/G{\approx}10$)", fontsize=7)
    a2.legend(fontsize=5.8, loc="lower right")

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "F19_viscoelastic_growth_stress.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", OUT / "F19_viscoelastic_growth_stress.pdf")


if __name__ == "__main__":
    main()
