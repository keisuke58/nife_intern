"""Standard ("定番") dental-implant bone-biomechanics relations — the textbook laws a peri-implant
FEM is expected to use, collected as cited, reusable numpy functions.

Why this module exists: the project already implements Huiskes site-specific remodeling
(bone_remodeling_huiskes.py), the RANKL/OPG disease model (fig_periimplantitis_rankl_opg.py), the
overload mechanostat (fig_periimplantitis_remodel.py) and clinical calibration against a longitudinal
cohort (fig_periimplantitis_clinical_calibration.py). What was MISSING from that set is the canonical
**Frost mechanostat in microstrain units** (the most-cited bone-adaptation language) with the
**dual-threshold overload→resorption** branch, plus the standard **density→modulus** laws and the
**ISO-14801 peri-implant reference band** for sanity-checking. Those are gathered here.

Values are order-of-magnitude / literature-ranged where the literature is (Frost was explicit that the
microstrain setpoints are "circa" values). Citations inline; see
notes/fem_literature_validation_2026-06-16.md for the validation table.

NOT implemented (no data): a patient-specific CT Hounsfield→density calibration (ρ = c·HU + d) needs
scanner/phantom-specific c, d that must be re-derived per machine — the function is provided but the
default constants are Keyak's K2HPO4 phantom and must not be reused blindly.
"""
from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Density → Young's modulus power laws   E[MPa] = a · ρ^b   (ρ in g/cm³)
#    Helgason et al. 2008, Clin Biomech 23:135-146 tabulates these.
# ─────────────────────────────────────────────────────────────────────────────
POISSON_RATIO = 0.30                 # standard for cortical & trabecular bone


def E_carter_hayes(rho_app: np.ndarray) -> np.ndarray:
    """Carter & Hayes 1977 (J Bone Joint Surg 59A:954): E = 3790·ρ_app³ (the famous E∝ρ³).
    Coefficient is unit/strain-rate-convention dependent (≈3790 at physiological strain rate)."""
    return 3790.0 * np.power(np.maximum(rho_app, 0.0), 3.0)


def E_morgan_femoral_neck(rho_app: np.ndarray) -> np.ndarray:
    """Morgan, Bayraktar & Keaveny 2003 (J Biomech 36:897), femoral-neck site law E = 6850·ρ_app^1.49
    — the best-validated site-specific apparent-density relation (Schileo 2008)."""
    return 6850.0 * np.power(np.maximum(rho_app, 0.0), 1.49)


def E_keyak_1998(rho_ash: np.ndarray) -> np.ndarray:
    """Keyak et al. 1998 piecewise femur-FEM relation (ρ_ash, g/cm³). The canonical CT-FEM law."""
    rho = np.asarray(rho_ash, float)
    return np.where(rho <= 0.27, 33900.0 * rho ** 2.20,
                    np.where(rho < 0.60, 5307.0 * rho + 469.0,
                             10200.0 * rho ** 2.01))


ASH_TO_APP = 0.60                    # ρ_ash/ρ_app ≈ 0.6 (Schileo 2008, both bone types)


def hu_to_ash_density(hu: np.ndarray, slope: float = 7.0e-4, intercept: float = -0.009) -> np.ndarray:
    """CT Hounsfield → ash density (Keyak K2HPO4 phantom, 140 kVp). SCANNER-SPECIFIC — re-derive
    slope/intercept per machine/phantom before use; defaults are illustrative only."""
    return slope * np.asarray(hu, float) + intercept


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frost mechanostat microstrain windows   (Frost 1987 Anat Rec 219:1; Frost 2003 Anat Rec 275A:1081)
#    "circa" setpoints — order of magnitude, not sharp thresholds.
# ─────────────────────────────────────────────────────────────────────────────
MES_R = 100.0        # MESr ~50-100 µε : below → disuse / net resorption
MES_M = 1500.0       # MESm ~1500 µε   : above → modeling / net formation
MES_P = 3000.0       # MESp ~3000 µε   : above → pathological overload (microdamage → resorption)
MES_FX = 25000.0     # ~2.5 % strain   : fracture

FROST_WINDOWS = (
    (0.0,   MES_R,  "disuse"),          # net resorption (under-load)
    (MES_R, MES_M,  "adapted"),         # homeostasis / lazy zone
    (MES_M, MES_P,  "formation"),       # net apposition (mild overload)
    (MES_P, MES_FX, "overload"),        # pathological overload → resorption / woven bone
    (MES_FX, np.inf, "fracture"),
)


def frost_class(microstrain: np.ndarray) -> np.ndarray:
    """Classify an effective-strain field (µε) into Frost windows (returns array of label strings)."""
    ue = np.asarray(microstrain, float)
    out = np.full(ue.shape, "adapted", dtype=object)
    for lo, hi, name in FROST_WINDOWS:
        out[(ue >= lo) & (ue < hi)] = name
    return out


def frost_remodeling_sign(microstrain: np.ndarray) -> np.ndarray:
    """Dual-threshold mechanostat sign: −1 resorption (disuse OR pathological overload), 0 maintenance,
    +1 formation. This is the branch the monotonic Huiskes SED rule lacks — the UPPER overload
    threshold is why peak crestal/shoulder strain drives marginal bone LOSS, not apposition."""
    ue = np.asarray(microstrain, float)
    sign = np.zeros(ue.shape, float)
    sign[ue < MES_R] = -1.0                       # disuse → resorption
    sign[(ue >= MES_M) & (ue < MES_P)] = +1.0     # mild overload → formation
    sign[ue >= MES_P] = -1.0                       # pathological overload → resorption
    return sign


# ─────────────────────────────────────────────────────────────────────────────
# 3. ISO-14801 peri-implant reference band (for sanity-checking a crestal stress/strain result).
#    ISO 14801:2016 fixes the 30° worst-case geometry; the ~100 N magnitude is a mastication
#    convention, NOT prescribed by the standard.
# ─────────────────────────────────────────────────────────────────────────────
PERIIMPLANT_FEM_REFERENCE = {
    "load_N": 100.0, "angle_deg": 30.0,
    "crestal_vM_MPa": (10.0, 50.0),       # normal/good bone, ~100 N oblique (Geng 2001; Baggi 2008)
    "crestal_vM_typical": 30.0,           # Pirc 2024: 29.3 MPa @120 N/35°
    "crestal_vM_poorbone_max": 150.0,     # thin cortex / narrow implant / high load
    "strain_ue": (200.0, 2500.0),         # physiological peri-implant strain (Talreja 2023: 330-2090)
    "cortical_yield_tensile_MPa": 100.0,
    "cortical_yield_compressive_MPa": 150.0,
}


def is_in_physiological_range(vm_mpa: float, strain_ue: float | None = None) -> str:
    """Human-readable sanity verdict for a crestal cortical vM stress (and optional strain)."""
    lo, hi = PERIIMPLANT_FEM_REFERENCE["crestal_vM_MPa"]
    yld = PERIIMPLANT_FEM_REFERENCE["cortical_yield_tensile_MPa"]
    sf = yld / vm_mpa if vm_mpa > 0 else float("inf")
    band = ("within" if lo <= vm_mpa <= hi else
            "below" if vm_mpa < lo else "above-normal (poor-bone/overload regime)")
    msg = f"crestal vM = {vm_mpa:.1f} MPa: {band} the 10-50 MPa normal band; SF vs tensile yield = {sf:.1f}×"
    if strain_ue is not None:
        slo, shi = PERIIMPLANT_FEM_REFERENCE["strain_ue"]
        msg += f"; strain = {strain_ue:.0f} µε ({'physiological' if slo <= strain_ue <= shi else 'outside ' + f'{slo:.0f}-{shi:.0f} µε'})"
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# 4. Clinical marginal-bone-loss (MBL) reference magnitudes.
# ─────────────────────────────────────────────────────────────────────────────
MBL_REFERENCE = {
    "albrektsson_1986_after_yr1_mm_per_yr": 0.2,    # success criterion: <0.2 mm/yr after first year
    "adell_1981_steady_mm_per_yr": 0.1,             # 15-yr Brånemark
    "first_year_mm": (1.0, 1.5),                     # healing + first-year remodeling (sources vary)
    "periimplantitis_threshold_mm": 2.0,            # progressive loss with inflammation
    "onset_location": "crestal",                    # Oh, Yoon, Misch & Wang 2002 J Periodontol 73:322
}


if __name__ == "__main__":
    # quick self-check / demo of the standard relations
    print("density→E [MPa] at ρ_app=1.85 (cortical-ish):",
          f"Carter-Hayes={E_carter_hayes(1.85):.0f}, Morgan-fn={E_morgan_femoral_neck(1.85):.0f}")
    print("Keyak piecewise at ρ_ash=[0.2,0.4,1.1]:", E_keyak_1998(np.array([0.2, 0.4, 1.1])).round(0))
    print("Frost sign at [50, 800, 2000, 4000] µε:", frost_remodeling_sign(np.array([50., 800., 2000., 4000.])))
    print(is_in_physiological_range(28.0, 1970.0))
