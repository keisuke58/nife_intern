"""Huiskes / Frost mechanostat bone remodeling — the "standard language" of peri-implant FEM.

Most peri-implant FEM literature reports stress maps and updates bone density via the Huiskes (1987,
2000) site-specific remodeling law (apparent density ρ driven by strain-energy density per unit
density, lazy zone, formation/resorption). This script implements the canonical equations, runs them
on a 1-D axial profile (substratum → crestal → apical) representative of the implant interface, and
shows three things:

  (1) the standard Huiskes/Frost remodeling converges to a stable density distribution under steady
      peri-implant loading;
  (2) the mechanostat threshold k separates "stable osseointegration" (interface densifies) from
      "stress shielding" (interface resorbs) — the classic implant-FEM clinical translation;
  (3) the phenomenological Huiskes law is the *sharp-threshold limit* of the mechanistic RANKL/OPG
      remodeling model:
          formation rate    ≈  +B (S − k(1+w))       ↔   OB activation above the upper threshold
          resorption rate   ≈  −B (k(1−w) − S)       ↔   RANKL/OPG-driven OC activation below it
      Writing each lineage as a graded (softplus, width τ) activation gives an ODE that is identical
      to Huiskes as τ → 0; for a finite biological τ the smooth model converges to the lazy-zone
      centre and the steady-state density drift vs. Huiskes scales ~linearly with τ (≈3 % at
      τ = 0.005·k under physiological load — the *measured* value is reported in panel C, not assumed
      to be zero). See rankl_opg_density_proxy() and notes/bone_remodeling_huiskes_2026-06-19.md.
      Figure: fig_bone_remodeling_huiskes.pdf.

This gives the thesis access to the "standard language" of dental FEM examiners (Huiskes is
universally cited) while reusing the mechanistic disease model. The phase-field (B.1) handles
*structural* dysbiosis → detachment; the bone-remodeling here handles *density* updates that change
the elastic substrate of the cortical / cancellous bone.

Equations (Huiskes 2000):
    S(x) = U(x) / ρ(x)                              SED per unit density (the "stimulus")
    dρ/dt = +B · (S − k(1+w))    if S > k(1+w)      formation
            −B · (k(1−w) − S)    if S < k(1−w)      resorption
            0                    else (lazy zone)
    E(ρ) = C · ρ^γ                                  Currey power law (γ ~ 2 for trabecular bone)

Units: ρ [g/cm³], U [J/m³], E [MPa]. Lazy zone width w ~ 0.35 (Huiskes), reference SED-per-density
k ~ 0.004 J/g (Huiskes 2000).

Run:  python bone_remodeling_huiskes.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

# ── Model parameters (Huiskes 2000 defaults) ─────────────────────────────────
K_STIM = 4.0e-3        # reference SED-per-density [J/g]
LAZY_W = 0.35          # lazy zone half-width (fractional)
B_RATE = 1.0           # remodeling rate constant [g/(cm³ × time-unit)]
GAMMA = 2.0            # Currey power-law exponent
C_PREFAC = 3790.0      # Currey prefactor so that E(ρ=1.85) ≈ 13700 MPa (cortical)
RHO_MIN = 0.01         # numerical floor (no negative density)
RHO_MAX = 2.10         # cortical-like cap [g/cm³]


def youngs_from_density(rho: np.ndarray) -> np.ndarray:
    return C_PREFAC * np.power(np.clip(rho, RHO_MIN, RHO_MAX), GAMMA)


def stimulus(U: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return U / np.maximum(rho, RHO_MIN)


def update_density(rho: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
    S = stimulus(U, rho)
    K_hi = K_STIM * (1.0 + LAZY_W)
    K_lo = K_STIM * (1.0 - LAZY_W)
    drho = np.zeros_like(rho)
    high = S > K_hi
    low = S < K_lo
    drho[high] = +B_RATE * (S[high] - K_hi)
    drho[low] = -B_RATE * (K_lo - S[low])
    return np.clip(rho + dt * drho, RHO_MIN, RHO_MAX)


def peri_implant_sed_profile(z: np.ndarray, load_factor: float = 1.0) -> np.ndarray:
    """1-D axial SED profile representative of crestal peri-implant loading.

    Higher SED near the crest (z ~ 0) due to load funneling; falls off apically. The load_factor
    knob lets us model stress shielding (low SED → resorption) or hyperload (high SED → formation).
    """
    return load_factor * 6.0e-3 * np.exp(-z / 4.0) + 5.0e-4
    # 6 mJ/cm³ peak near crest, decays with depth z [mm]


def converge_to_steady(rho0: np.ndarray, U: np.ndarray, dt: float = 0.5,
                       n_iter: int = 5000, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    rho = rho0.copy()
    hist = [rho.copy()]
    for k in range(n_iter):
        rho_new = update_density(rho, U, dt)
        if np.linalg.norm(rho_new - rho) / max(np.linalg.norm(rho), 1e-12) < tol:
            rho = rho_new
            break
        rho = rho_new
        if k % 200 == 0:
            hist.append(rho.copy())
    hist.append(rho.copy())
    return rho, np.array(hist)


RANKL_TAU_FRAC = 0.005      # graded-activation width as a fraction of K_STIM (→0 recovers Huiskes)


def rankl_opg_density_proxy(U: np.ndarray, rho0: np.ndarray, dt: float = 0.5,
                            n_iter: int = 400000, tol: float = 1e-8,
                            tau: float | None = None) -> np.ndarray:
    """Reduced RANKL/OPG remodeling, written as a graded-threshold (smooth) mechanostat.

    Mechanism (the disease model's reduced form): osteoblast (OB) formation activates *above* the
    upper mechanostat threshold K_hi = K(1+w) (overload → microdamage → TGF-β release → OB
    recruitment); osteoclast (OC) resorption activates *below* the lower threshold K_lo = K(1−w)
    (disuse → RANKL/OPG up). Each lineage is recruited *progressively* over a width τ as the stimulus
    crosses its threshold (graded, not a hard switch):

        dρ/dt = B · [ a_OB · softplus(S − K_hi; τ)  −  a_OC · softplus(K_lo − S; τ) ],
        softplus(x; τ) = τ·log(1 + e^{x/τ})  →  max(x, 0)  as τ → 0.

    EQUIVALENCE — read carefully (this is the honest, measured statement; see
    notes/bone_remodeling_huiskes_2026-06-19.md):

      • In the hard-threshold limit τ → 0, softplus → ReLU and this law becomes *identical* to the
        Huiskes/Frost update (update_density), dead band and all. So the Huiskes law is exactly the
        sharp-threshold limit of the mechanistic RANKL/OPG model — that is the equivalence.

      • For a finite, biologically graded τ > 0 the softplus tails leak into the lazy zone, so the
        smooth model has a *single* attractor at the lazy-zone centre S = K (where the two softplus
        terms balance) rather than Huiskes' whole dead-band interval. The resulting steady-state
        density drift vs. Huiskes is therefore NOT zero; it scales ~linearly with τ
        (≈3 % at τ = 0.005·K under physiological load, larger under disuse — panel C reports the
        measured value). A unique-fixed-point ODE cannot reproduce Huiskes' history-dependent dead
        band exactly except in the τ → 0 limit; we report the achieved agreement rather than claim
        spurious pointwise identity.

      a_OB = a_OC = 1 (= the Huiskes formation/resorption gains; no separate calibration constant).

    This replaces an earlier draft whose proxy (linear-OB a_OB=1/K + sigmoid-OC) had NO dead band,
    a fixed point off the lazy zone, and an explicit-Euler clip-to-clip limit cycle in the resorption
    tail — it reported a meaningless 63 % "drift". The three defects (dead band / calibration /
    stability) are fixed here; the residual few-percent drift below is real and τ-controlled.

    Integrates to steady state (tol-based early stop); convergence is slow because the lazy-zone
    drive is O(τ), hence the high n_iter ceiling.
    """
    K_hi = K_STIM * (1.0 + LAZY_W)
    K_lo = K_STIM * (1.0 - LAZY_W)
    if tau is None:
        tau = RANKL_TAU_FRAC * K_STIM
    a_OB = a_OC = 1.0

    def softplus(x: np.ndarray) -> np.ndarray:
        return tau * np.logaddexp(0.0, x / tau)   # numerically stable τ·log(1+e^{x/τ})

    rho = rho0.copy()
    for _ in range(n_iter):
        S = stimulus(U, rho)
        drho = B_RATE * (a_OB * softplus(S - K_hi) - a_OC * softplus(K_lo - S))
        rho_new = np.clip(rho + dt * drho, RHO_MIN, RHO_MAX)
        if np.linalg.norm(rho_new - rho) / max(np.linalg.norm(rho), 1e-12) < tol:
            return rho_new
        rho = rho_new
    return rho


def main() -> None:
    z = np.linspace(0.0, 10.0, 101)                     # 0 = crest, 10 mm apical
    rho0 = np.full_like(z, 1.20)                        # initial cancellous density [g/cm³]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # Panel A: Huiskes convergence under normal load
    U_norm = peri_implant_sed_profile(z, load_factor=1.0)
    rho_norm, hist = converge_to_steady(rho0, U_norm)
    cmap = plt.get_cmap("viridis")
    for k, r in enumerate(hist):
        axes[0].plot(z, r, color=cmap(k / len(hist)), lw=0.7, alpha=0.6)
    axes[0].plot(z, rho_norm, color="k", lw=2.0, label="steady state")
    axes[0].set_xlabel("depth z [mm] (0 = crest)")
    axes[0].set_ylabel(r"$\rho$ [g/cm³]")
    axes[0].set_title("(A) Huiskes convergence (normal load)", fontsize=11, loc="left")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, lw=0.4, alpha=0.4)

    # Panel B: stress shielding vs hyper-load
    U_shield = peri_implant_sed_profile(z, load_factor=0.30)
    U_hyper = peri_implant_sed_profile(z, load_factor=2.50)
    rho_shield, _ = converge_to_steady(rho0, U_shield)
    rho_hyper, _ = converge_to_steady(rho0, U_hyper)
    axes[1].plot(z, rho_shield, color="#1f77b4", lw=2.0, label="stress shielding (×0.30)")
    axes[1].plot(z, rho_norm, color="black", lw=2.0, label="physiological (×1.00)")
    axes[1].plot(z, rho_hyper, color="#d62728", lw=2.0, label="hyper-load (×2.50)")
    axes[1].set_xlabel("depth z [mm]")
    axes[1].set_ylabel(r"steady-state $\rho$ [g/cm³]")
    axes[1].set_title("(B) Loading regime → density distribution", fontsize=11, loc="left")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, lw=0.4, alpha=0.4)

    # Panel C: Huiskes IS the sharp-threshold limit of the RANKL/OPG mechanostat.
    # Headline graded width, plus a near-limit τ to show convergence to Huiskes as τ → 0.
    tau_head = RANKL_TAU_FRAC * K_STIM
    tau_limit = 0.001 * K_STIM
    rho_rankl = rankl_opg_density_proxy(U_norm, rho0, tau=tau_head)
    rho_limit = rankl_opg_density_proxy(U_norm, rho0, tau=tau_limit)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(a)) * 100.0
    rel_err = rel(rho_norm, rho_rankl)
    rel_lim = rel(rho_norm, rho_limit)
    axes[2].plot(z, rho_norm, "k-", lw=2.0, label="Huiskes (1987/2000)")
    axes[2].plot(z, rho_rankl, "r--", lw=2.0,
                 label=fr"RANKL/OPG, $\tau$={RANKL_TAU_FRAC:.3f}·k  (|Δ|={rel_err:.1f}%)")
    axes[2].plot(z, rho_limit, color="#1f77b4", ls=":", lw=1.8,
                 label=fr"RANKL/OPG, $\tau$=0.001·k  (|Δ|={rel_lim:.1f}%)")
    axes[2].set_xlabel("depth z [mm]")
    axes[2].set_ylabel(r"steady-state $\rho$ [g/cm³]")
    axes[2].set_title(r"(C) Huiskes = $\tau\!\to\!0$ limit of RANKL/OPG", fontsize=11, loc="left")
    axes[2].annotate(r"graded threshold $\tau\!\to\!0$" "\n" r"$\Rightarrow$ exact Huiskes ($|\Delta|\to 0$)",
                     xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=8,
                     bbox=dict(boxstyle="round", fc="#fffbe6", ec="0.6", lw=0.5))
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, lw=0.4, alpha=0.4)

    fig.suptitle("Bone remodeling — Huiskes/Frost mechanostat for peri-implant bone density",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_bone_remodeling_huiskes.pdf")
    fig.savefig(OUT / "fig_bone_remodeling_huiskes.png", dpi=200)
    print(f"wrote {(OUT / 'fig_bone_remodeling_huiskes.pdf').name}")
    print(f"RANKL/OPG vs Huiskes drift: tau={RANKL_TAU_FRAC:.3f}*k -> {rel_err:.2f}% ; "
          f"tau=0.001*k -> {rel_lim:.2f}%  (-> 0 as tau -> 0)")


if __name__ == "__main__":
    main()
