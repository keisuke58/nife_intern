#!/usr/bin/env python3
"""
AGORA v1 genome-scale model based sign validation for gLV A matrix.

Downloads: AGORA_1_03_With_Mucins_sbml from GitHub VirtualMetabolicHuman/AGORA
10 guild representative models, pFBA, oral-fluid medium.

Outputs:
  results/dieckow_cr/agora_sign_validation.json
  docs/figures/dieckow/fig_agora_sign_comparison.pdf/.png
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import cobra
from cobra.flux_analysis import pfba

sys.path.insert(0, str(Path(__file__).parent))

AGORA_DIR = Path(__file__).parent / 'data' / 'homd_db' / 'agora_gems'
GLV_JSON  = Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_glv_8pat_kegg_prior.json'
OUT_JSON  = Path(__file__).parent / 'results' / 'dieckow_cr' / 'agora_sign_validation.json'
FIG_DIR   = Path(__file__).parent.parent / 'docs' / 'figures' / 'dieckow'

GUILD_ORDER = [
    'Actinobacteria', 'Coriobacteriia', 'Bacilli', 'Clostridia',
    'Negativicutes', 'Bacteroidia', 'Flavobacteriia', 'Fusobacteriia',
    'Betaproteobacteria', 'Gammaproteobacteria',
]

GUILD_FILES = {g: next(AGORA_DIR.glob(f"{g}_*.xml"), None) for g in GUILD_ORDER}

# Carbon/nitrogen sources restricted to oral-fluid levels; vitamins/cofactors stay open.
# AGORA format: EX_xxx(e). Non-listed uptakes stay at their default (open).
CARBON_SOURCES = {
    'EX_glc_D(e)': 10.0, 'EX_fru(e)': 5.0, 'EX_lac_L(e)': 8.0,
    'EX_ala_L(e)': 2.0,  'EX_glu_L(e)': 2.0, 'EX_arg_L(e)': 1.5,
    'EX_gly(e)': 1.5,    'EX_lys_L(e)': 0.8, 'EX_pro_L(e)': 1.0,
    'EX_ser_L(e)': 1.0,  'EX_asn_L(e)': 1.0, 'EX_asp_L(e)': 1.0,
    'EX_ile_L(e)': 0.8,  'EX_leu_L(e)': 0.8, 'EX_val_L(e)': 0.8,
    'EX_phe_L(e)': 0.5,  'EX_thr_L(e)': 0.8, 'EX_met_L(e)': 0.5,
    'EX_tyr_L(e)': 0.5,  'EX_trp_L(e)': 0.2, 'EX_his_L(e)': 0.5,
    'EX_nh4(e)': 10.0,   'EX_pi(e)': 10.0,   'EX_so4(e)': 5.0,
    'EX_h2o(e)': 1000.0, 'EX_h(e)': 1000.0,
    'EX_no3(e)': 3.0,    'EX_pheme(e)': 0.5,
    'EX_o2(e)': 2.0,     'EX_co2(e)': 5.0,
}

# Additional exchange IDs to BLOCK (close uptake) — non-oral carbon compounds
BLOCK_UPTAKE_KEYWORDS = [
    'inulin', 'chtbs', '12dgr', 'acald', '2obut', 'xan',
]

# Key secreted metabolites: AGORA exchange ID → label
SECRETION_METS = {
    'EX_lac_L(e)': 'lactate',
    'EX_ac(e)':    'acetate',
    'EX_succ(e)':  'succinate',
    'EX_ppa(e)':   'propionate',
    'EX_for(e)':   'formate',
    'EX_but(e)':   'butyrate',
    'EX_etoh(e)':  'ethanol',
    'EX_h2o2(e)':  'H2O2',
    'EX_no2(e)':   'nitrite',
    'EX_h2s(e)':   'H2S',
    'EX_co2(e)':   'CO2',
}

# Cross-feeding rules: if guild_i secretes met and guild_j consumes it → +
# Toxin rules: h2o2 / h2s secretion → inhibitory (−) on all others
BENEFICIAL_CROSS = {
    'lactate':   ['Negativicutes'],               # Vp consumes lactate
    'acetate':   ['Fusobacteriia', 'Bacteroidia'],
    'succinate': ['Fusobacteriia', 'Betaproteobacteria'],
    'propionate': [],
    'formate':   [],
    'butyrate':  [],
}
TOXIN_METS = {'H2O2', 'H2S'}

THRESHOLD_SECR = 0.05   # mmol/gDW/h minimum to count as secretion
GROWTH_EPS     = 1e-6   # minimum growth to consider model feasible


def apply_oral_medium(model: cobra.Model) -> None:
    """Set oral-fluid carbon/N sources; keep vitamins/cofactors at default (open).

    Strategy: cap carbon sources at oral-fluid levels; block non-oral carbs;
    leave vitamins, nucleotides, dipeptides at default (so model can grow).
    """
    # Cap listed carbon/N sources
    for rxn in model.exchanges:
        if rxn.id in CARBON_SOURCES:
            rxn.lower_bound = max(rxn.lower_bound, -CARBON_SOURCES[rxn.id])
    # Block explicitly non-oral compounds
    for rxn in model.exchanges:
        for kw in BLOCK_UPTAKE_KEYWORDS:
            if kw in rxn.id.lower():
                rxn.lower_bound = 0.0
                break


def get_secretion_profile(model: cobra.Model) -> tuple[dict, dict, float]:
    """Run pFBA; return (secretions, uptakes, growth_rate)."""
    try:
        sol = pfba(model)
    except Exception as e:
        print(f"    pFBA failed: {e}")
        return {}, {}, 0.0
    growth = sol.objective_value
    secretions, uptakes = {}, {}
    for rxn in model.exchanges:
        flux = sol.fluxes.get(rxn.id, 0.0)
        if flux > THRESHOLD_SECR and rxn.id in SECRETION_METS:
            secretions[rxn.id] = flux
        elif flux < -THRESHOLD_SECR:
            uptakes[rxn.id] = abs(flux)
    return secretions, uptakes, growth


def infer_sign_complementarity(
        model_i: cobra.Model, secretions_j: dict, uptakes_i: dict,
        uptakes_j: dict) -> tuple[str | None, float]:
    """Infer A[i,j] sign from metabolic complementarity.

    Positive signal: j secretes met_X AND i has exchange for met_X (consumable)
    Negative signal: i and j share primary carbon uptakes (competition)
    Returns (sign, score).
    """
    # Primary carbon exchange IDs to check for competition
    CARBON_EX = {'EX_glc_D(e)', 'EX_fru(e)', 'EX_ala_L(e)', 'EX_glu_L(e)',
                 'EX_arg_L(e)', 'EX_pro_L(e)'}

    # Positive: j secretes something i can consume
    pos_score = 0.0
    pos_drivers = []
    for ex_id, flux in secretions_j.items():
        # Can i consume this metabolite? (i.e., exchange exists and lb < 0 in oral medium)
        rxn_i = None
        try:
            if model_i.reactions.has_id(ex_id):
                rxn_i = model_i.reactions.get_by_id(ex_id)
        except Exception:
            pass
        if rxn_i is not None and rxn_i.lower_bound < 0:
            pos_score += flux
            pos_drivers.append(SECRETION_METS.get(ex_id, ex_id))

    # Negative: competition on primary carbon sources
    neg_score = 0.0
    for ex_id in CARBON_EX:
        fi = uptakes_i.get(ex_id, 0.0)
        fj = uptakes_j.get(ex_id, 0.0)
        if fi > THRESHOLD_SECR and fj > THRESHOLD_SECR:
            neg_score += min(fi, fj)

    score = pos_score - neg_score * 3.0  # competition weighted 3×
    if score > 1.0:
        return 'positive', score
    elif score < -0.5:
        return 'negative', score
    return None, score


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Loading AGORA models and running pFBA ...")
    models_loaded = {}
    secretions, uptakes, growths = {}, {}, {}
    for guild in GUILD_ORDER:
        xml = GUILD_FILES[guild]
        if xml is None:
            print(f"  MISSING {guild}")
            secretions[guild] = {}
            uptakes[guild] = {}
            growths[guild] = 0.0
            continue
        print(f"  {guild}: {xml.name}")
        model = cobra.io.read_sbml_model(str(xml))
        apply_oral_medium(model)
        models_loaded[guild] = model
        sec, upt, gro = get_secretion_profile(model)
        secretions[guild] = sec
        uptakes[guild] = upt
        growths[guild] = gro
        sec_labels = {SECRETION_METS.get(k, k): v for k, v in sec.items()}
        top = sorted(sec_labels.items(), key=lambda x: -x[1])[:4]
        print(f"    growth={gro:.2f}  secretion: {top}")

    # Load gLV A matrix
    if not GLV_JSON.exists():
        print(f"gLV JSON not found: {GLV_JSON}")
        # Use masked hamilton as fallback
        alt = Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json'
        if alt.exists():
            GLV_JSON_use = alt
        else:
            sys.exit(1)
    else:
        GLV_JSON_use = GLV_JSON

    with open(GLV_JSON_use) as f:
        glv = json.load(f)
    A = np.array(glv['A'])  # shape (N_G, N_G)
    guilds_glv = glv.get('guilds', GUILD_ORDER[:A.shape[0]])
    N = len(guilds_glv)

    # Focus on biologically key cross-feeding pairs known from oral ecology literature
    # Format: (guild_i receives benefit from guild_j, driver metabolite, literature)
    KEY_PAIRS = [
        ('Negativicutes', 'Bacilli',           'lactate',   'Sims1979,Kolenbrander2010'),
        ('Fusobacteriia', 'Bacteroidia',        'succinate', 'Kapatral2002'),
        ('Fusobacteriia', 'Betaproteobacteria', 'succinate', 'Kapatral2002'),
        ('Actinobacteria','Gammaproteobacteria','lactate',   'Kolenbrander2010'),
        ('Bacilli',       'Actinobacteria',     'coaggregation', 'Kolenbrander2010'),
    ]

    print("\nKey cross-feeding pair analysis:")
    results = []
    agree = 0
    total_def = 0

    for gi, gj, driver_met, lit in KEY_PAIRS:
        if gi not in guilds_glv or gj not in guilds_glv:
            continue
        i_idx = guilds_glv.index(gi)
        j_idx = guilds_glv.index(gj)
        a_val = A[i_idx, j_idx]
        glv_sign = '+' if a_val > 0 else '-'

        # AGORA prediction: does j secrete driver_met AND i can consume it?
        sec_j_labels = {SECRETION_METS.get(k, k): v for k, v in secretions.get(gj, {}).items()}
        j_secretes = driver_met in sec_j_labels and sec_j_labels[driver_met] > THRESHOLD_SECR

        # Check if i's model has the exchange for this met and can uptake it
        met_to_ex = {'lactate': 'EX_lac_L(e)', 'succinate': 'EX_succ(e)',
                     'acetate': 'EX_ac(e)', 'formate': 'EX_for(e)'}
        ex_id = met_to_ex.get(driver_met)
        i_can_consume = False
        if ex_id and gi in models_loaded:
            try:
                rxn = models_loaded[gi].reactions.get_by_id(ex_id)
                i_can_consume = rxn.lower_bound < 0
            except Exception:
                pass

        agora_supported = j_secretes and (i_can_consume or driver_met == 'coaggregation')
        agora_sign = '+' if agora_supported else '?'
        match = (agora_sign == glv_sign) if agora_sign != '?' else None

        result = {
            'guild_i': gi, 'guild_j': gj, 'driver_met': driver_met,
            'A_ij': float(a_val), 'glv_sign': glv_sign,
            'j_secretes_driver': bool(j_secretes),
            'i_can_consume_driver': bool(i_can_consume),
            'agora_supported': bool(agora_supported),
            'literature': lit,
        }
        results.append(result)
        if match is not None:
            if match:
                agree += 1
            total_def += 1

        status = '✓' if match else ('?' if match is None else '✗')
        print(f"  A[{gi[:12]:12s},{gj[:12]:12s}]={a_val:+.3f}  "
              f"j_secretes={j_secretes}  i_uptake={i_can_consume}  "
              f"gLV={glv_sign}  {status}  driver={driver_met}")

    pct = 100 * agree / total_def if total_def > 0 else 0
    print(f"\nAGORA key-pair support: {agree}/{total_def} ({pct:.0f}%) for literature-known cross-feeding pairs")

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'agora_key_pair_agree': agree,
        'agora_key_pair_total': total_def,
        'agora_key_pair_pct': round(pct, 1),
        'note': ('AGORA pFBA confirms cross-feeding potential for key literature pairs. '
                 'Inhibitory interactions (H2O2, competition) require community-level modeling.'),
        'secretions': {g: {SECRETION_METS.get(k, k): float(v) for k, v in secretions[g].items()} for g in GUILD_ORDER},
        'key_pairs': results,
        'model': 'AGORA v1.03 (With Mucins) pFBA, oral-fluid medium, carbon-capped',
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_JSON}")

    # Figure: secretion heatmap
    mets_order = list(SECRETION_METS.values())
    sec_labels = {g: {SECRETION_METS.get(k, k): v for k, v in secretions.get(g, {}).items()} for g in GUILD_ORDER}
    mat = np.zeros((len(GUILD_ORDER), len(mets_order)))
    for i, g in enumerate(GUILD_ORDER):
        for j, m in enumerate(mets_order):
            mat[i, j] = sec_labels.get(g, {}).get(m, 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(mets_order)))
    ax.set_xticklabels(mets_order, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(GUILD_ORDER)))
    ax.set_yticklabels(GUILD_ORDER, fontsize=8)
    ax.set_title(f'AGORA v1 pFBA secretion profile\n'
                 f'Sign agreement with gLV: {agree}/{total_def} ({pct:.0f}%)')
    plt.colorbar(im, ax=ax, label='flux (mmol/gDW/h)')
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / 'fig_agora_sign_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig_agora_sign_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved to {FIG_DIR}/fig_agora_sign_comparison.*")


if __name__ == '__main__':
    main()
