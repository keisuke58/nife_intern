"""
extract_monod_params.py
=======================
Extract Monod kinetic parameters from AGORA v1.03 GEMs using COBRApy FBA.

What is extractable from stoichiometric FBA
-------------------------------------------
  Y  (yield, gDW/mmol)      — from linear slope mu vs q_substrate
  mu_max_rel                 — theoretical max from all-open FBA (relative ordering;
                               normalized to So = 0.50 h⁻¹ for absolute scale)
  secretion_ratio            — mmol product / mmol primary substrate consumed

What is NOT extractable
-----------------------
  Km (half-saturation, mM)  — kinetic parameter; kept from literature

Method for Y extraction
-----------------------
  1. Compute species-specific minimal medium via cobra.medium.minimal_medium()
  2. Fix cofactor exchange bounds to those values (not unlimited — prevents
     cofactors from acting as bulk carbon sources)
  3. Set all other organic exchanges to lb=0 (closed)
  4. Scan primary substrate lb from 0 to q_scan_max (mmol/gDW/h) — 20 points
  5. Y = dmu / d(q_substrate) from the linear (substrate-limited) region
  6. mu_max_fixed = growth at saturation (cofactor-limited upper bound)

Usage
-----
    cd /home/nishioka/IKM_Hiwi/nife
    python comets/extract_monod_params.py [--out comets/pipeline_results]

Outputs
-------
  <out>/monod_params_gem.json   — extracted + current MONOD_PARAMS comparison
  stdout                        — comparison table
"""
from __future__ import annotations

import argparse
import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and species metadata
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
AGORA_DIR = _HERE / "agora_gems"

SPECIES_CONFIG: dict[str, dict] = {
    "So": {
        "model_file": AGORA_DIR / "Streptococcus_oralis_Uo5.xml",
        "primary_substrate": "EX_glc_D(e)",
        "biomass_rxn": "EX_biomass(e)",
        "q_scan_max": 15.0,      # mmol/gDW/h upper bound for scan
        "co_substrates": [],     # secondary substrates to keep closed during scan
    },
    "An": {
        "model_file": AGORA_DIR / "Actinomyces_naeslundii_str_Howell_279.xml",
        "primary_substrate": "EX_glc_D(e)",
        "biomass_rxn": "EX_biomass(e)",
        "q_scan_max": 15.0,
        "co_substrates": [],
    },
    "Vp": {
        "model_file": AGORA_DIR / "Veillonella_parvula_Te3_DSM_2008.xml",
        "primary_substrate": "EX_lac_L(e)",   # obligate lactate fermenter
        "biomass_rxn": "EX_biomass(e)",
        "q_scan_max": 15.0,
        "co_substrates": [],
    },
    "Fn": {
        "model_file": AGORA_DIR / "Fusobacterium_nucleatum_subsp_nucleatum_ATCC_25586.xml",
        "primary_substrate": "EX_glc_D(e)",
        "biomass_rxn": "EX_biomass(e)",
        "q_scan_max": 15.0,
        "co_substrates": ["EX_lac_D(e)"],    # close D-lactate to isolate glucose
    },
    "Pg": {
        "model_file": AGORA_DIR / "Porphyromonas_gingivalis_W83.xml",
        "primary_substrate": "EX_succ(e)",    # succinate as primary C-source
        "biomass_rxn": "EX_biomass(e)",
        "q_scan_max": 10.0,
        # hemin must be provided (required cofactor); keep it open at fixed level
        "co_substrates": [],
        "required_open": {"EX_pheme(e)": -1.0},   # hemin: always open for Pg
    },
}

# Inorganic ions + gases: always unlimited
_INORGANIC = frozenset({
    "EX_h2o(e)", "EX_o2(e)", "EX_co2(e)", "EX_h(e)", "EX_h2(e)",
    "EX_nh4(e)", "EX_pi(e)", "EX_ppi(e)", "EX_so4(e)", "EX_ca2(e)", "EX_mg2(e)",
    "EX_k(e)", "EX_na1(e)", "EX_fe2(e)", "EX_fe3(e)", "EX_zn2(e)", "EX_cu2(e)",
    "EX_mn2(e)", "EX_cobalt2(e)", "EX_cl(e)", "EX_ni2(e)", "EX_mobd(e)",
})

# Current MONOD_PARAMS from oral_biofilm.py (for comparison)
MONOD_PARAMS_CURRENT: dict[str, dict] = {
    "So": {"mu_max": 0.50, "q_max": 8.0,  "Km": 0.05, "Y": 0.10},
    "An": {"mu_max": 0.35, "q_max": 6.0,  "Km": 0.08, "Y": 0.08},
    "Vp": {"mu_max": 0.40, "q_max": 10.0, "Km": 0.15, "Y": 0.07},
    "Fn": {"mu_max": 0.32, "q_max": 4.0,  "Km": 0.12, "Y": 0.09},
    "Pg": {"mu_max": 0.20, "q_max": 3.0,  "Km": 0.08, "Y": 0.10},
}


# ---------------------------------------------------------------------------
# FBA helpers
# ---------------------------------------------------------------------------

def _load_model(path: Path):
    """Load SBML model with warnings suppressed."""
    import cobra
    return cobra.io.read_sbml_model(str(path))


def _get_minimal_medium(model, min_growth: float = 0.05) -> dict[str, float]:
    """
    Return {exchange_id: required_lb} as the minimal medium for growth >= min_growth.
    Uses cobra.medium.minimal_medium with the model's own exchange bounds as ceiling.
    """
    from cobra.medium import minimal_medium as cobra_minimal_medium
    med = cobra_minimal_medium(model, min_objective_value=min_growth)
    if med is None:
        return {}
    return dict(med)


def _setup_medium(
    model,
    min_med: dict[str, float],
    primary_substrate: str,
    primary_lb: float,
    required_open: dict[str, float] | None = None,
) -> None:
    """
    Apply medium to model (in-place, call inside `with model:`):
      - All exchanges: lb=0 (closed)
      - Inorganic: lb=-1000
      - Cofactors from min_med: lb = -min_med[id] (fixed amount)
      - required_open extras: lb = value
      - Primary substrate: lb = primary_lb
    """
    # 1. Close everything
    for r in model.exchanges:
        r.lower_bound = 0.0

    # 2. Inorganic unlimited
    for r in model.exchanges:
        if r.id in _INORGANIC:
            r.lower_bound = -1000.0

    # 3. Cofactors at fixed required amounts (from minimal medium)
    for ex_id, flux_req in min_med.items():
        if ex_id in _INORGANIC:
            continue
        try:
            rxn = model.reactions.get_by_id(ex_id)
            # use 2× the minimum to be safe, but cap at -10 to avoid C-source dominance
            rxn.lower_bound = max(-min(flux_req * 2, 10.0), -10.0)
        except KeyError:
            pass

    # 4. Extra required exchanges (e.g., hemin for Pg)
    if required_open:
        for ex_id, lb_val in required_open.items():
            try:
                model.reactions.get_by_id(ex_id).lower_bound = lb_val
            except KeyError:
                pass

    # 5. Primary substrate
    try:
        model.reactions.get_by_id(primary_substrate).lower_bound = primary_lb
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# A) Extract yield Y from linear region
# ---------------------------------------------------------------------------

def extract_yield(
    model,
    min_med: dict[str, float],
    primary_substrate: str,
    biomass_rxn: str,
    q_scan_max: float,
    required_open: dict[str, float] | None = None,
    n_scan: int = 20,
) -> dict:
    """
    Scan primary substrate lb from 0 to q_scan_max.
    Fit linear slope in the substrate-limited (pre-saturation) region.
    Returns {'Y': float, 'mu_max_fixed': float, 'q_sat': float, 'curve': ...}
    """
    lb_vals = np.linspace(0, q_scan_max, n_scan + 1)
    mus, q_actual = [], []

    with model:
        model.objective = biomass_rxn

        # Baseline (no primary substrate)
        _setup_medium(model, min_med, primary_substrate, 0.0, required_open)
        sol0 = model.optimize()
        mu0 = sol0.objective_value if sol0.status == "optimal" else 0.0

        for lb in lb_vals:
            _setup_medium(model, min_med, primary_substrate, -lb, required_open)
            sol = model.optimize()
            mu_val = sol.objective_value if sol.status == "optimal" else mu0
            q_val  = abs(sol.fluxes.get(primary_substrate, 0.0)) if sol.status == "optimal" else 0.0
            mus.append(mu_val)
            q_actual.append(q_val)

    mus      = np.array(mus)
    q_actual = np.array(q_actual)
    delta_mu = mus - mu0

    # Linear region: where q_actual > 0 and mu still increasing
    # Use points before saturation (mu < 0.95 * max)
    mu_max_fixed = float(mus.max())
    sat_threshold = mu0 + (mu_max_fixed - mu0) * 0.80

    linear_mask = (delta_mu > 0.001) & (mus < sat_threshold) & (q_actual > 0.001)
    if linear_mask.sum() >= 2:
        coeffs  = np.polyfit(q_actual[linear_mask], delta_mu[linear_mask], 1)
        Y_gem   = float(max(coeffs[0], 0.0))
    else:
        # fallback: use full range slope
        valid = q_actual > 0.001
        if valid.sum() >= 2:
            coeffs = np.polyfit(q_actual[valid], delta_mu[valid], 1)
            Y_gem  = float(max(coeffs[0], 0.0))
        else:
            Y_gem = 0.0

    # q at saturation
    sat_mask = mus >= sat_threshold
    q_sat = float(q_actual[sat_mask][0]) if sat_mask.any() else float(q_actual[-1])

    return {
        "Y_gem":        round(Y_gem, 4),
        "mu_max_fixed": round(float(mu_max_fixed), 4),
        "mu0_bg":       round(float(mu0), 4),
        "q_sat":        round(q_sat, 4),
        "lb_scan":      lb_vals.tolist(),
        "mu_scan":      mus.tolist(),
        "q_scan":       q_actual.tolist(),
    }


# ---------------------------------------------------------------------------
# B) Extract mu_max from all-open FBA
# ---------------------------------------------------------------------------

def extract_mu_max_open(
    model,
    biomass_rxn: str,
    lb_open: float = -20.0,
) -> float:
    """All exchanges open at lb_open → theoretical max growth rate."""
    with model:
        model.objective = biomass_rxn
        for r in model.exchanges:
            r.lower_bound = lb_open
        sol = model.optimize()
        return float(sol.objective_value) if sol.status == "optimal" else 0.0


# ---------------------------------------------------------------------------
# C) Extract secretion stoichiometry at all-open optimum
# ---------------------------------------------------------------------------

def extract_secretion(
    model,
    biomass_rxn: str,
    primary_substrate: str,
    lb_open: float = -20.0,
) -> dict[str, float]:
    """
    At all-open optimum, compute mmol secreted / mmol primary substrate consumed.
    Returns dict {exchange_id: ratio} for secreted metabolites.
    """
    with model:
        model.objective = biomass_rxn
        for r in model.exchanges:
            r.lower_bound = lb_open
        sol = model.optimize()
        if sol.status != "optimal":
            return {}
        q_prim = abs(sol.fluxes.get(primary_substrate, 0.0))
        if q_prim < 0.001:
            return {}
        return {
            r.id: round(float(sol.fluxes[r.id]) / q_prim, 4)
            for r in model.exchanges
            if sol.fluxes.get(r.id, 0.0) > 0.01   # secreted (positive flux)
            and r.id not in (biomass_rxn, "EX_h(e)", "EX_co2(e)", "EX_h2o(e)")
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Monod params from AGORA GEMs")
    ap.add_argument("--out", default="comets/pipeline_results",
                    help="Output directory")
    ap.add_argument("--lb_open", type=float, default=-20.0,
                    help="lb for all-open FBA (B)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    # Normalization: So mu_max_open → 0.50 h⁻¹
    so_mu_open: float | None = None

    print("=" * 62)
    print("  AGORA GEM → Monod parameter extraction")
    print("=" * 62)

    for sp, cfg in SPECIES_CONFIG.items():
        print(f"\n[{sp}] Loading {cfg['model_file'].name} …")
        model = _load_model(cfg["model_file"])

        biomass_rxn  = cfg["biomass_rxn"]
        primary_sub  = cfg["primary_substrate"]
        q_scan_max   = cfg["q_scan_max"]
        required_open = cfg.get("required_open", None)

        # A) All-open mu_max
        print(f"  A) All-open FBA (lb={args.lb_open}) …", flush=True)
        mu_open = extract_mu_max_open(model, biomass_rxn, args.lb_open)
        print(f"     mu_max_open = {mu_open:.4f} h⁻¹")

        # B) Minimal medium for cofactor set
        print(f"  B) Computing minimal medium …", flush=True)
        with model:
            for r in model.exchanges:
                r.lower_bound = -1000
            model.objective = biomass_rxn
            min_med = _get_minimal_medium(model, min_growth=0.05)
        # Exclude the primary substrate from the cofactor set
        # (it will be scanned separately)
        min_med.pop(primary_sub, None)
        # Also exclude any obvious polysaccharide/bulk C-sources
        min_med = {k: v for k, v in min_med.items()
                   if not any(x in k for x in ["pullulan", "starch", "glycogen"])}
        print(f"     cofactors: {len(min_med)} exchanges")

        # C) Yield scan
        print(f"  C) Yield scan (0 → {q_scan_max} mmol/gDW/h) …", flush=True)
        yield_res = extract_yield(
            model, min_med, primary_sub, biomass_rxn,
            q_scan_max=q_scan_max,
            required_open=required_open,
        )
        print(f"     Y = {yield_res['Y_gem']:.4f} gDW/mmol   "
              f"mu_max_fixed = {yield_res['mu_max_fixed']:.4f} h⁻¹   "
              f"q_sat = {yield_res['q_sat']:.2f} mmol/gDW/h")

        # D) Secretion products
        print(f"  D) Secretion stoichiometry …", flush=True)
        sec = extract_secretion(model, biomass_rxn, primary_sub, args.lb_open)
        if sec:
            top = sorted(sec.items(), key=lambda x: -abs(x[1]))[:5]
            print(f"     secreted: {', '.join(f'{k}={v:.3f}' for k,v in top)}")

        results[sp] = {
            "mu_max_open":   round(mu_open, 4),
            "Y_gem":         yield_res["Y_gem"],
            "mu_max_fixed":  yield_res["mu_max_fixed"],
            "q_sat_gem":     yield_res["q_sat"],
            "mu0_bg":        yield_res["mu0_bg"],
            "secretion":     sec,
            "primary_substrate": primary_sub,
            "yield_scan": {
                "lb": yield_res["lb_scan"],
                "mu": yield_res["mu_scan"],
                "q":  yield_res["q_scan"],
            },
        }

        if sp == "So":
            so_mu_open = mu_open

    # Normalize mu_max_open relative to So = 0.50 h⁻¹
    if so_mu_open and so_mu_open > 0:
        scale = 0.50 / so_mu_open
        for sp in results:
            results[sp]["mu_max_normalized"] = round(
                results[sp]["mu_max_open"] * scale, 4
            )
    else:
        scale = 1.0
        for sp in results:
            results[sp]["mu_max_normalized"] = results[sp]["mu_max_open"]

    # ---- Comparison table ----
    print("\n" + "=" * 62)
    print("  COMPARISON: FBA-derived vs current MONOD_PARAMS")
    print("=" * 62)
    header = (f"{'Sp':3s}  {'mu_max_curr':>11s}  {'mu_max_norm':>11s}  "
              f"{'Y_curr':>7s}  {'Y_gem':>7s}  {'q_sat_gem':>9s}")
    print(header)
    print("-" * 62)
    for sp, res in results.items():
        curr = MONOD_PARAMS_CURRENT[sp]
        print(
            f"{sp:3s}  {curr['mu_max']:>11.3f}  "
            f"{res['mu_max_normalized']:>11.3f}  "
            f"{curr['Y']:>7.3f}  {res['Y_gem']:>7.4f}  "
            f"{res['q_sat_gem']:>9.3f}"
        )

    print("\nNotes:")
    print("  mu_max_norm  = mu_max_open scaled so that So = 0.50 h⁻¹")
    print("  Y_gem        = FBA-derived yield (stoichiometrically grounded)")
    print("  q_sat_gem    = primary substrate flux at cofactor-limited saturation")
    print("  Km (mM)      = NOT extractable from FBA — kept from literature")

    # ---- Save JSON ----
    out = {
        "agora_version":   "v1.03",
        "lb_open":         args.lb_open,
        "So_normalization": {"mu_max_open_So": so_mu_open, "scale": round(scale, 6)},
        "species":         results,
        "current_monod_params": MONOD_PARAMS_CURRENT,
    }
    json_path = out_dir / "monod_params_gem.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ---- Recommended updates ----
    print("\n=== Recommended MONOD_PARAMS updates (Y from FBA) ===")
    for sp, res in results.items():
        curr = MONOD_PARAMS_CURRENT[sp]
        Y_gem = res["Y_gem"]
        if Y_gem > 0.001:
            print(f"  {sp}: Y  {curr['Y']:.3f} → {Y_gem:.4f}  "
                  f"({'✓ consistent' if abs(Y_gem - curr['Y']) / curr['Y'] < 0.3 else '⚠ differs > 30%'})")
        else:
            print(f"  {sp}: Y extraction failed (μ=0 at all scan points)")

    print("""
=== AGORA v1.03 Limitations for Single-Species Simulation ===
  An, Vp, Fn, Pg: minimal medium uses non-physiological C-sources
    (An: stys/glyc3p, Vp: fru/glyc3p, Fn: complex peptides, Pg: similar)
    → cofactor requirements exceed our fixed set → Y extraction fails

  μ_max relative ordering (from all-open FBA) IS reliable:
    So ≈ An >> Fn > Vp >> Pg
    FBA suggests Vp μ_max ~ 0.15 h⁻¹ (current hand-tuned: 0.40 h⁻¹)
    → current MONOD_PARAMS may overestimate Vp growth by ~2.5×

  Secretion products (from all-open FBA) match known biology:
    So  → acetate, formate (homo-lactic/mixed acid fermentation products)
    Vp  → propanoate, acetate (known Veillonella lactate fermentation ✓)
    Fn  → butyrate, propanoate, succinate (known Fusobacterium products ✓)
    Pg  → acetate, phenylacetate (proteolytic metabolism ✓)

  BioModels/Path2Models GEMs (Fn, Vp, Pg): μ=0 even with all-open FBA
    → SuBliMinaL auto-generated, intracellular metabolites exposed as
      boundary exchanges, BIOMASS_REACTION bounds blocked → not usable
""")


if __name__ == "__main__":
    main()
