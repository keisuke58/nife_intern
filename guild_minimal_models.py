#!/usr/bin/env python3
"""
Manually curated minimal metabolic models for 10 oral guilds.

Reactions are based on:
  - HOMD (Human Oral Microbiome Database, homd.org)
  - Kolenbrander et al. 2010 Nature Reviews Microbiology
  - Mager et al. 2003 (oral fluid metabolomics)
  - Takahashi & Nyvad 2011 (ecological caries model)

Each guild model encodes:
  - Primary carbon metabolism (fermentation pathways)
  - Key secretion products (lactate, H2O2, CO2, SCFA, nitrite, H2S)
  - Known cross-feeding substrates consumed
  - Exchange reactions for the oral-fluid medium

Run:
  python guild_minimal_models.py [--plot]
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import cobra
from cobra import Model, Reaction, Metabolite
from cobra.flux_analysis import pfba

sys.path.insert(0, str(Path(__file__).parent))

GUILD_ORDER = [
    'Actinobacteria', 'Coriobacteriia', 'Bacilli', 'Clostridia',
    'Negativicutes', 'Bacteroidia', 'Flavobacteriia', 'Fusobacteriia',
    'Betaproteobacteria', 'Gammaproteobacteria',
]

ANAEROBIC = {'Clostridia', 'Bacteroidia', 'Fusobacteriia', 'Negativicutes'}

# ── Oral medium (exchange reaction IDs → upper-bound mmol/gDW/h) ────────────
ORAL_MEDIUM = {
    'EX_glc__D_e': 10.0, 'EX_fru_e': 5.0, 'EX_lac__L_e': 5.0,
    'EX_ala__L_e': 2.0,  'EX_glu__L_e': 2.0, 'EX_arg__L_e': 1.5,
    'EX_gly_e': 1.5,     'EX_lys__L_e': 0.8, 'EX_pro__L_e': 1.0,
    'EX_ser__L_e': 1.0,  'EX_asn__L_e': 1.0, 'EX_asp__L_e': 1.0,
    'EX_ile__L_e': 0.8,  'EX_leu__L_e': 0.8, 'EX_val__L_e': 0.8,
    'EX_phe__L_e': 0.5,  'EX_thr__L_e': 0.8, 'EX_met__L_e': 0.5,
    'EX_tyr__L_e': 0.5,  'EX_trp__L_e': 0.2, 'EX_his__L_e': 0.5,
    'EX_nh4_e': 10.0,    'EX_pi_e': 10.0,    'EX_so4_e': 5.0,
    'EX_h2o_e': 1000.0,  'EX_h_e': 1000.0,
    'EX_no3_e': 3.0,     'EX_pheme_e': 0.5,  'EX_mqn8_e': 0.5,
    'EX_o2_e': 2.0,      'EX_co2_e': 5.0,
}

SECRETABLE_EX = {
    'EX_lac__L_e', 'EX_ac_e', 'EX_succ_e', 'EX_ppa_e', 'EX_for_e', 'EX_butyrate_e', 'EX_etoh_e',
    'EX_h2o2_e', 'EX_no2_e', 'EX_nh4_e', 'EX_h2s_e', 'EX_co2_e',
}

CROSSFEED_METS = {
    'lac__L_e', 'ac_e', 'succ_e', 'ppa_e', 'for_e', 'butyrate_e', 'etoh_e', 'h2o2_e', 'no2_e', 'h2s_e',
}

TOXIC_METS = {'h2o2_e', 'h2s_e'}
H2S_SENSITIVE = {'Betaproteobacteria', 'Gammaproteobacteria'}

PRIMARY_SUBSTRATES = {
    'glc__D_e', 'fru_e', 'lac__L_e',
    'glu__L_e', 'asp__L_e', 'ala__L_e', 'gly_e', 'lys__L_e', 'arg__L_e', 'val__L_e',
}


def _met(mid, name='', comp='c'):
    m = Metabolite(f'{mid}_{comp}', name=name, compartment=comp)
    return m


def _ex(model, met_id, lb=-1000, ub=1000):
    """Add exchange reaction if not already present."""
    ex_id = f'EX_{met_id}_e'
    if ex_id in model.reactions:
        return
    m_e = model.metabolites.get_by_id(f'{met_id}_e') if f'{met_id}_e' in [m.id for m in model.metabolites] else _met(met_id, comp='e')
    r = Reaction(ex_id, lower_bound=lb, upper_bound=ub)
    r.add_metabolites({m_e: -1})
    model.add_reactions([r])


def _transport(model, met_id, lb=-1000, ub=1000):
    """Add cytoplasm ↔ extracellular transport."""
    r_id = f'TR_{met_id}'
    if r_id in model.reactions:
        return
    mc = _met(met_id, comp='c')
    me = _met(met_id, comp='e')
    r = Reaction(r_id, lower_bound=lb, upper_bound=ub)
    r.add_metabolites({me: -1, mc: 1})
    model.add_reactions([r])
    _ex(model, met_id)


def _rxn(model, rxn_id, stoich: dict, bounds=(-1000, 1000)):
    r = Reaction(rxn_id, lower_bound=bounds[0], upper_bound=bounds[1])
    mets = {}
    for mid, s in stoich.items():
        if mid not in [m.id for m in model.metabolites]:
            comp = 'e' if mid.endswith('_e') else 'c'
            base = mid[:-2] if mid.endswith('_e') or mid.endswith('_c') else mid
            m = _met(base, comp=comp)
            m.id = mid
        else:
            m = model.metabolites.get_by_id(mid)
        mets[m] = s
    r.add_metabolites(mets)
    model.add_reactions([r])


def build_bacilli():
    """Streptococcus gordonii — homofermentative lactic acid bacteria + H2O2."""
    m = Model('Bacilli')
    # Glycolysis: glucose → 2 lactate (homolactic fermentation)
    _rxn(m, 'GLCpts',  {'glc__D_e': -1, 'glc__D_c': 1})
    _rxn(m, 'GLYCOL',  {'glc__D_c': -1, 'lac__L_c': 2, 'h_c': 2}, bounds=(0, 1000))
    # H2O2 production (NADH oxidase, hallmark of S. gordonii)
    _rxn(m, 'H2O2ox',  {'o2_e': -1, 'h2o2_c': 2}, bounds=(0, 1000))
    # Arginine deiminase system (ADI) → NH3 + ATP (facultative)
    _rxn(m, 'ADI',     {'arg__L_c': -1, 'nh4_c': 1, 'co2_c': 1}, bounds=(0, 1000))
    # Transports + exchanges
    for mid in ['glc__D','lac__L','h2o2','arg__L','nh4','co2','h','o2']:
        _transport(m, mid)
    # Biomass proxy
    _rxn(m, 'BIOMASS',{'glc__D_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_actinobacteria():
    """Actinomyces naeslundii — mixed-acid fermenter + nitrate reduction → nitrite."""
    m = Model('Actinobacteria')
    _rxn(m, 'GLCup',   {'glc__D_e': -1, 'glc__D_c': 1})
    # Mixed-acid: glucose → lactate + acetate
    _rxn(m, 'GLYCOL',  {'glc__D_c': -1, 'lac__L_c': 1, 'ac_c': 0.5, 'co2_c': 0.5}, bounds=(0,1000))
    # Nitrate reductase (key oral nitrate reducer → nitrite)
    _rxn(m, 'NRA',     {'no3_e': -1, 'no2_c': 1}, bounds=(0, 1000))
    for mid in ['glc__D','lac__L','ac','no3','no2','co2','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'glc__D_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_coriobacteriia():
    """Atopobium parvulum — lactate fermentation → acetate + ethanol."""
    m = Model('Coriobacteriia')
    _rxn(m, 'LACup',   {'lac__L_e': -1, 'lac__L_c': 1})
    # Lactate → acetate + ethanol
    _rxn(m, 'LACFERM', {'lac__L_c': -1, 'ac_c': 0.5, 'etoh_c': 0.5, 'co2_c': 0.5}, bounds=(0,1000))
    for mid in ['lac__L','ac','etoh','co2','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'lac__L_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_negativicutes():
    """Veillonella parvula — obligate lactate fermenter → propionate + acetate + CO2."""
    m = Model('Negativicutes')
    # Cannot ferment glucose — consumes lactate only
    _rxn(m, 'LACup',   {'lac__L_e': -1, 'lac__L_c': 1})
    # Lactate → propionate + acetate + CO2 (Wood-Werkman pathway)
    _rxn(m, 'LACFERM', {'lac__L_c': -2, 'ppa_c': 1, 'ac_c': 1, 'co2_c': 1}, bounds=(0,1000))
    for mid in ['lac__L','ppa','ac','co2','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'lac__L_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_bacteroidia():
    """Prevotella melaninogenica — proteo-saccharolytic anaerobe → SCFA + H2S."""
    m = Model('Bacteroidia')
    _rxn(m, 'GLCup',   {'glc__D_e': -1, 'glc__D_c': 1})
    _rxn(m, 'AAup',    {'glu__L_e': -1, 'glu__L_c': 1})
    _rxn(m, 'ASPup',   {'asp__L_e': -1, 'asp__L_c': 1})
    # Saccharolytic: glucose → succinate + acetate
    _rxn(m, 'GLCFERM', {'glc__D_c': -1, 'succ_c': 1, 'ac_c': 1}, bounds=(0,1000))
    # Proteolytic: amino acids → SCFA + H2S
    _rxn(m, 'AAFERM',  {'glu__L_c': -1, 'butyrate_c': 0.5, 'h2s_c': 0.5}, bounds=(0,1000))
    _rxn(m, 'ASPFERM', {'asp__L_c': -1, 'succ_c': 1, 'nh4_c': 1}, bounds=(0,1000))
    for mid in ['glc__D','glu__L','asp__L','succ','ac','butyrate','h2s','nh4','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'glc__D_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_fusobacteriia():
    """Fusobacterium nucleatum — amino acid fermenter → butyrate + H2S."""
    m = Model('Fusobacteriia')
    for aa in ['glu__L','lys__L','gly','ala__L','his__L']:
        _rxn(m, f'{aa}_up', {f'{aa}_e': -1, f'{aa}_c': 1})
    _rxn(m, 'AAFERM',  {'glu__L_c': -1, 'butyrate_c': 1, 'nh4_c': 1}, bounds=(0,1000))
    _rxn(m, 'LYFERM',  {'lys__L_c': -1, 'butyrate_c': 1, 'nh4_c': 1}, bounds=(0,1000))
    _rxn(m, 'GLYFERM', {'gly_c': -1, 'ac_c': 1, 'nh4_c': 1}, bounds=(0,1000))
    _rxn(m, 'H2Sprod', {'glu__L_c': -0.5, 'h2s_c': 1}, bounds=(0,1000))
    for mid in ['glu__L','lys__L','gly','ala__L','butyrate','h2s','ac','nh4','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'glu__L_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_clostridia():
    """Parvimonas micra — amino acid fermenter → butyrate + formate."""
    m = Model('Clostridia')
    for aa in ['glu__L','ala__L','val__L']:
        _rxn(m, f'{aa}_up', {f'{aa}_e': -1, f'{aa}_c': 1})
    _rxn(m, 'AAFERM', {'glu__L_c': -1, 'butyrate_c': 0.5, 'for_c': 0.5, 'nh4_c': 1}, bounds=(0,1000))
    for mid in ['glu__L','ala__L','val__L','butyrate','for','nh4','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'glu__L_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_flavobacteriia():
    """Capnocytophaga ochracea — gliding bacterium, ferments glucose/mannose → succinate."""
    m = Model('Flavobacteriia')
    _rxn(m, 'GLCup',   {'glc__D_e': -1, 'glc__D_c': 1})
    _rxn(m, 'GLCFERM', {'glc__D_c': -1, 'succ_c': 1, 'ac_c': 0.5, 'co2_c': 0.5}, bounds=(0,1000))
    for mid in ['glc__D','succ','ac','co2','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'glc__D_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_betaproteobacteria():
    """Neisseria mucosa / Eikenella corrodens — aerobic, glucose → CO2, H2O2."""
    m = Model('Betaproteobacteria')
    _rxn(m, 'GLCup',   {'glc__D_e': -1, 'glc__D_c': 1})
    # Aerobic TCA → CO2
    _rxn(m, 'TCA',     {'glc__D_c': -1, 'o2_e': -3, 'co2_c': 6}, bounds=(0,1000))
    # H2O2 production (Neisseria NADH oxidase)
    _rxn(m, 'H2O2ox',  {'o2_e': -1, 'h2o2_c': 2}, bounds=(0,500))
    for mid in ['glc__D','co2','h2o2','h','o2']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'glc__D_c': -1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


def build_gammaproteobacteria():
    """Haemophilus parainfluenzae — requires heme+NAD, glucose fermentation."""
    m = Model('Gammaproteobacteria')
    _rxn(m, 'GLCup',   {'glc__D_e': -1, 'glc__D_c': 1})
    _rxn(m, 'HEMEup',  {'pheme_e': -1, 'pheme_c': 1})  # requires exogenous heme (V-factor)
    _rxn(m, 'GLCFERM', {'glc__D_c': -1, 'lac__L_c': 1, 'succ_c': 0.5}, bounds=(0,1000))
    for mid in ['glc__D','lac__L','succ','pheme','h']:
        _transport(m, mid)
    _rxn(m, 'BIOMASS',{'glc__D_c': -0.5, 'pheme_c': -0.1}, bounds=(0, 1000))
    m.objective = 'BIOMASS'
    return m


GUILD_BUILDERS = {
    'Actinobacteria':       build_actinobacteria,
    'Coriobacteriia':       build_coriobacteriia,
    'Bacilli':              build_bacilli,
    'Clostridia':           build_clostridia,
    'Negativicutes':        build_negativicutes,
    'Bacteroidia':          build_bacteroidia,
    'Flavobacteriia':       build_flavobacteriia,
    'Fusobacteriia':        build_fusobacteriia,
    'Betaproteobacteria':   build_betaproteobacteria,
    'Gammaproteobacteria':  build_gammaproteobacteria,
}


def apply_medium(model, guild):
    medium = {}
    for rxn in model.exchanges:
        medium[rxn.id] = ORAL_MEDIUM.get(rxn.id, 0.0)
    if guild in ANAEROBIC:
        if 'EX_o2_e' in medium:
            medium['EX_o2_e'] = 0.0
    model.medium = medium


def solve_growth_with_secretions(model):
    mu = float(model.slim_optimize(error_value=np.nan))
    if not np.isfinite(mu) or mu <= 1e-9:
        return None, float(mu) if np.isfinite(mu) else np.nan

    biomass = model.reactions.get_by_id('BIOMASS')
    biomass.lower_bound = 0.999 * mu

    obj = {}
    for rxn in model.exchanges:
        if rxn.id in SECRETABLE_EX:
            obj[rxn] = 1.0

    if obj:
        model.objective = obj
    else:
        model.objective = 'BIOMASS'

    sol = pfba(model)
    return sol, mu


def get_secretions(sol, model, threshold=1e-6):
    secs = {}
    for rxn in model.exchanges:
        f = sol.fluxes.get(rxn.id, 0.0)
        if f > threshold:
            met_id = list(rxn.metabolites.keys())[0].id
            secs[met_id] = f
    return secs


def get_importable_mets(model):
    subs = set()
    for ex in model.exchanges:
        met = list(ex.metabolites.keys())[0]
        importable = False
        for r2 in met.reactions:
            if r2.id == ex.id:
                continue
            if r2.metabolites.get(met, 0.0) < 0:
                importable = True
                break
        if importable:
            subs.add(met.id)
    return subs


def run_analysis(verbose=True, return_details=False):
    N = len(GUILD_ORDER)
    cross_matrix = np.zeros((N, N))
    comp_matrix = np.zeros((N, N), dtype=int)
    tox_matrix = np.zeros((N, N))

    models, secretions, substrates = {}, {}, {}
    for guild in GUILD_ORDER:
        m = GUILD_BUILDERS[guild]()
        apply_medium(m, guild)
        try:
            sol, mu = solve_growth_with_secretions(m)
            if sol is None:
                raise RuntimeError('infeasible')
        except Exception as e:
            print(f'  {guild}: infeasible ({e})')
            continue
        models[guild]     = m
        secretions[guild] = get_secretions(sol, m)
        substrates[guild] = get_importable_mets(m)
        if verbose:
            secs_str = ', '.join(f'{k}={v:.2f}' for k,v in sorted(secretions[guild].items(), key=lambda x:-x[1])[:4])
            print(f'  {guild:22s} μ={mu:.3f}  secretes: {secs_str}')

    print()
    for src in GUILD_ORDER:
        if src not in models:
            continue
        j = GUILD_ORDER.index(src)
        secs_j = secretions[src]
        subs_j = substrates[src]
        for tgt in GUILD_ORDER:
            if src == tgt or tgt not in models:
                continue
            i = GUILD_ORDER.index(tgt)
            subs_i = substrates[tgt]

            cross = sum(v for met, v in secs_j.items() if met in CROSSFEED_METS and met in subs_i)
            shared = (subs_j & subs_i) & PRIMARY_SUBSTRATES
            comp = len(shared)
            tox = 0.0
            if tgt in ANAEROBIC and 'h2o2_e' in secs_j:
                tox += float(secs_j['h2o2_e'])
            if tgt in H2S_SENSITIVE and 'h2s_e' in secs_j:
                tox += float(secs_j['h2s_e'])

            cross_matrix[i, j] = cross
            comp_matrix[i, j] = comp
            tox_matrix[i, j] = tox

    if return_details:
        return cross_matrix, comp_matrix, tox_matrix, list(models.keys()), secretions, substrates
    return cross_matrix, comp_matrix, list(models.keys())


def predict_signs(cross_matrix, comp_matrix, cross_pos=0.01, comp_neg=4, cross_neg_max=0.001, tox_matrix=None, tox_pos=0.1):
    N = cross_matrix.shape[0]
    sign_matrix = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cross = cross_matrix[i, j]
            comp = comp_matrix[i, j]
            tox = float(tox_matrix[i, j]) if tox_matrix is not None else 0.0
            if tox > tox_pos:
                sign_matrix[i, j] = -1
            elif cross > cross_pos:
                sign_matrix[i, j] = +1
            elif comp >= comp_neg and cross <= cross_neg_max:
                sign_matrix[i, j] = -1
    return sign_matrix


def _eval_agreement(sign_matrix, glv_path, absA_min=0.05, edge_mask=None):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']
    n_p = 0
    n_a = 0
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            if src not in guilds or tgt not in guilds:
                continue
            if edge_mask is not None and not bool(edge_mask[i_a, j_a]):
                continue
            s_min = sign_matrix[i_a, j_a]
            if s_min == 0:
                continue
            i_g = guilds.index(tgt)
            j_g = guilds.index(src)
            a_val = A[i_g, j_g]
            if abs(a_val) < absA_min:
                continue
            s_glv = int(np.sign(a_val)) if abs(a_val) > 1e-6 else 0
            if s_glv == 0:
                continue
            n_p += 1
            if s_min == s_glv:
                n_a += 1
    return n_p, n_a


def tune_thresholds(cross_matrix, comp_matrix, glv_path, absA_min=0.05, edge_mask=None):
    cross_pos_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    comp_neg_list = [999, 3, 2, 1]
    cross_neg_max_list = [0.0, 1e-4, 1e-3, 1e-2, 0.05]

    best = None
    for cross_pos in cross_pos_list:
        for comp_neg in comp_neg_list:
            for cross_neg_max in cross_neg_max_list:
                sign = predict_signs(cross_matrix, comp_matrix, cross_pos=cross_pos, comp_neg=comp_neg, cross_neg_max=cross_neg_max)
                n_p, n_a = _eval_agreement(sign, glv_path, absA_min=absA_min, edge_mask=edge_mask)
                if n_p < 4:
                    continue
                rate = n_a / max(n_p, 1)
                key = (rate, n_a, n_p, -cross_pos, -comp_neg, -cross_neg_max)
                if best is None or key > best[0]:
                    best = (key, cross_pos, comp_neg, cross_neg_max, n_p, n_a, rate)

    if best is None:
        return 0.01, 4, 0.001
    _, cross_pos, comp_neg, cross_neg_max, n_p, n_a, rate = best
    print(f'Tuned thresholds (|A|≥{absA_min}): cross_pos={cross_pos}, comp_neg={comp_neg}, cross_neg_max={cross_neg_max}  (agree {n_a}/{n_p}={100*rate:.1f}%)')
    return cross_pos, comp_neg, cross_neg_max


def fit_linear_A(comp_matrix, glv_path, secretions, substrates, absA_min=0.05, edge_mask=None):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']

    met_list = sorted(CROSSFEED_METS)
    X_rows = []
    y = []
    w = []
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            if src not in guilds or tgt not in guilds:
                continue
            if edge_mask is not None and not bool(edge_mask[i_a, j_a]):
                continue
            i_g = guilds.index(tgt)
            j_g = guilds.index(src)
            a_val = float(A[i_g, j_g])
            if abs(a_val) < absA_min:
                continue
            comp = float(comp_matrix[i_a, j_a])
            feats = []
            if src in secretions and tgt in substrates:
                for met in met_list:
                    if met in substrates[tgt]:
                        v = float(secretions[src].get(met, 0.0))
                        feats.append(float(np.log1p(v)) if v > 0 else 0.0)
                    else:
                        feats.append(0.0)
            else:
                feats = [0.0] * len(met_list)
            X_rows.append([1.0] + feats + [comp])
            y.append(a_val)
            w.append(abs(a_val))

    if not X_rows:
        return np.zeros((len(GUILD_ORDER), len(GUILD_ORDER))), {'intercept': 0.0, **{m: 0.0 for m in met_list}, 'comp': 0.0}

    X = np.asarray(X_rows, dtype=float)
    yv = np.asarray(y, dtype=float)
    W = np.asarray(w, dtype=float)
    sw = np.sqrt(W)
    Xw = X * sw[:, None]
    yw = yv * sw
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    N = len(GUILD_ORDER)
    A_hat = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            comp = float(comp_matrix[i, j])
            src = GUILD_ORDER[j]
            tgt = GUILD_ORDER[i]
            feats = []
            if src in secretions and tgt in substrates:
                for met in met_list:
                    if met in substrates[tgt]:
                        v = float(secretions[src].get(met, 0.0))
                        feats.append(float(np.log1p(v)) if v > 0 else 0.0)
                    else:
                        feats.append(0.0)
            else:
                feats = [0.0] * len(met_list)
            row = np.asarray([1.0] + feats + [comp], dtype=float)
            A_hat[i, j] = float(row @ beta)

    coefs = {'intercept': float(beta[0])}
    for k, met in enumerate(met_list, start=1):
        coefs[met] = float(beta[k])
    coefs['comp'] = float(beta[-1])
    return A_hat, coefs


def compare_with_glv(sign_matrix, glv_path, absA_min=0.05, cross_matrix=None, comp_matrix=None, secretions=None, substrates=None):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']

    rows = []
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt: continue
            if src not in guilds or tgt not in guilds: continue
            i_g = guilds.index(tgt)
            j_g = guilds.index(src)
            a_val = A[i_g, j_g]
            strong = abs(a_val) >= absA_min
            s_glv = int(np.sign(a_val)) if strong and abs(a_val) > 1e-6 else 0
            s_min   = sign_matrix[i_a, j_a]
            cross = float(cross_matrix[i_a, j_a]) if cross_matrix is not None else None
            comp = int(comp_matrix[i_a, j_a]) if comp_matrix is not None else None
            drivers = ''
            if secretions is not None and substrates is not None and src in secretions and tgt in substrates:
                if s_min == +1:
                    feats = [(m, float(v)) for m, v in secretions[src].items() if m in CROSSFEED_METS and m in substrates[tgt]]
                    feats.sort(key=lambda x: -x[1])
                    drivers = ', '.join(f'{m}={v:.3g}' for m, v in feats[:3])
                elif s_min == -1:
                    shared = sorted(((substrates.get(src, set()) & substrates.get(tgt, set())) & PRIMARY_SUBSTRATES))
                    if shared:
                        drivers = ', '.join(shared)
            rows.append({
                'src': src, 'tgt': tgt,
                'A': a_val, 'absA': abs(a_val),
                'sign_glv': s_glv, 'sign_minimal': s_min,
                'absA_min': absA_min,
                'strongA': strong,
                'agree': (s_min != 0 and s_glv != 0 and s_min == s_glv),
                'has_pred': s_min != 0,
                'cross_score': cross,
                'comp_shared': comp,
                'drivers': drivers,
            })

    df = pd.DataFrame(rows).sort_values('absA', ascending=False)
    predicted = df[df['has_pred'] & df['strongA'] & (df['sign_glv'] != 0)]
    n_p = len(predicted); n_a = int(predicted['agree'].sum())
    print(f'Minimal-model predictions (|A|≥{absA_min}): {n_p} pairs, {n_a}/{n_p} agree with gLV ({100*n_a//max(n_p,1)}%)')
    print('\nTop disagreements:')
    disagree = predicted[~predicted['agree']].head(8)
    for _, r in disagree.iterrows():
        print(f'  {r["src"]:22s} → {r["tgt"]:22s}  A={r["A"]:+.3f}  minimal={r["sign_minimal"]:+d}')
    return df


def fit_linear_A_with_bias(comp_matrix, glv_path, secretions, substrates, absA_min=0.05, edge_mask=None):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']

    met_list = sorted(CROSSFEED_METS)
    N = len(GUILD_ORDER)
    src_index = {g: i for i, g in enumerate(GUILD_ORDER)}
    tgt_index = {g: i for i, g in enumerate(GUILD_ORDER)}

    X_rows = []
    y = []
    w = []
    idx_pairs = []
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            if src not in guilds or tgt not in guilds:
                continue
            if edge_mask is not None and not bool(edge_mask[i_a, j_a]):
                continue
            i_g = guilds.index(tgt)
            j_g = guilds.index(src)
            a_val = float(A[i_g, j_g])
            if abs(a_val) < absA_min:
                continue

            feats = []
            if src in secretions and tgt in substrates:
                for met in met_list:
                    if met in substrates[tgt]:
                        v = float(secretions[src].get(met, 0.0))
                        feats.append(float(np.log1p(v)) if v > 0 else 0.0)
                    else:
                        feats.append(0.0)
            else:
                feats = [0.0] * len(met_list)

            comp = float(comp_matrix[i_a, j_a])
            src_oh = [0.0] * (N - 1)
            tgt_oh = [0.0] * (N - 1)
            j_idx = src_index[src]
            i_idx = tgt_index[tgt]
            if j_idx > 0:
                src_oh[j_idx - 1] = 1.0
            if i_idx > 0:
                tgt_oh[i_idx - 1] = 1.0

            X_rows.append([1.0] + feats + [comp] + src_oh + tgt_oh)
            y.append(a_val)
            w.append(abs(a_val))
            idx_pairs.append((i_a, j_a))

    if not X_rows:
        return np.zeros((N, N), dtype=float), {'intercept': 0.0}, {}

    X = np.asarray(X_rows, dtype=float)
    yv = np.asarray(y, dtype=float)
    W = np.sqrt(np.asarray(w, dtype=float))
    beta, _, _, _ = np.linalg.lstsq(X * W[:, None], yv * W, rcond=None)

    A_hat = np.zeros((N, N), dtype=float)
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            feats = []
            if src in secretions and tgt in substrates:
                for met in met_list:
                    if met in substrates[tgt]:
                        v = float(secretions[src].get(met, 0.0))
                        feats.append(float(np.log1p(v)) if v > 0 else 0.0)
                    else:
                        feats.append(0.0)
            else:
                feats = [0.0] * len(met_list)

            comp = float(comp_matrix[i_a, j_a])
            src_oh = [0.0] * (N - 1)
            tgt_oh = [0.0] * (N - 1)
            j_idx = src_index[src]
            i_idx = tgt_index[tgt]
            if j_idx > 0:
                src_oh[j_idx - 1] = 1.0
            if i_idx > 0:
                tgt_oh[i_idx - 1] = 1.0

            row = np.asarray([1.0] + feats + [comp] + src_oh + tgt_oh, dtype=float)
            A_hat[i_a, j_a] = float(row @ beta)

    p = 0
    coefs = {'intercept': float(beta[p])}
    p += 1
    for k, met in enumerate(met_list):
        coefs[met] = float(beta[p + k])
    p += len(met_list)
    coefs['comp'] = float(beta[p]); p += 1
    src_bias = {'_baseline': 0.0}
    for j in range(1, N):
        src_bias[GUILD_ORDER[j]] = float(beta[p + (j - 1)])
    p += (N - 1)
    tgt_bias = {'_baseline': 0.0}
    for i in range(1, N):
        tgt_bias[GUILD_ORDER[i]] = float(beta[p + (i - 1)])

    meta = {'src_bias': src_bias, 'tgt_bias': tgt_bias}
    return A_hat, coefs, meta


def compare_with_glv_continuous(A_hat, glv_path, absA_min=0.05, mask=None, label='A_hat'):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']

    xs = []
    ys = []
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            if src not in guilds or tgt not in guilds:
                continue
            if mask is not None and not bool(mask[i_a, j_a]):
                continue
            i_g = guilds.index(tgt)
            j_g = guilds.index(src)
            a_val = float(A[i_g, j_g])
            if abs(a_val) < absA_min:
                continue
            xs.append(a_val)
            ys.append(float(A_hat[i_a, j_a]))

    if len(xs) < 3:
        print(f'{label} fit (|A|≥{absA_min}): not enough pairs ({len(xs)})')
        return {'n': len(xs), 'corr': None, 'sign_agree': None}

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        corr = None
    s_agree = float(np.mean(np.sign(x) == np.sign(y)))
    corr_str = f'{corr:.3f}' if corr is not None else 'nan'
    print(f'{label} fit (|A|≥{absA_min}): n={len(x)}, corr={corr_str}, sign_agree={100*s_agree:.1f}%')
    return {'n': int(len(x)), 'corr': corr, 'sign_agree': s_agree}


def build_edge_mask(glv_path, absA_min=0.05):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']
    N = len(GUILD_ORDER)
    mask = np.zeros((N, N), dtype=bool)
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            if src not in guilds or tgt not in guilds:
                continue
            a_val = float(A[guilds.index(tgt), guilds.index(src)])
            if abs(a_val) >= absA_min:
                mask[i_a, j_a] = True
    return mask


def kfold_edge_masks(edge_mask, k=5, seed=0):
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(edge_mask.shape[0]) for j in range(edge_mask.shape[1]) if bool(edge_mask[i, j])]
    rng.shuffle(pairs)
    folds = [pairs[i::k] for i in range(k)]
    masks = []
    for fold in folds:
        test = np.zeros_like(edge_mask, dtype=bool)
        for i, j in fold:
            test[i, j] = True
        train = edge_mask & (~test)
        masks.append((train, test))
    return masks


def evaluate_sign_on_mask(sign_matrix, glv_path, absA_min=0.05, edge_mask=None):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']
    n = 0
    n_a = 0
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            if edge_mask is not None and not bool(edge_mask[i_a, j_a]):
                continue
            if src not in guilds or tgt not in guilds:
                continue
            a_val = float(A[guilds.index(tgt), guilds.index(src)])
            if abs(a_val) < absA_min:
                continue
            s_glv = int(np.sign(a_val)) if abs(a_val) > 1e-12 else 0
            if s_glv == 0:
                continue
            s = int(sign_matrix[i_a, j_a])
            if s == 0:
                continue
            n += 1
            if s == s_glv:
                n_a += 1
    return {'n_pred': int(n), 'n_agree': int(n_a), 'agree': (n_a / n) if n else None}


def fit_magnitude_from_features(cross_matrix, comp_matrix, tox_matrix, sign_matrix, glv_path, absA_min=0.05, absA_min_fit=0.01, use_agree_only=True):
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']

    X_rows = []
    y = []
    w = []
    for i_a, tgt in enumerate(GUILD_ORDER):
        for j_a, src in enumerate(GUILD_ORDER):
            if src == tgt:
                continue
            if src not in guilds or tgt not in guilds:
                continue
            s = int(sign_matrix[i_a, j_a])
            if s == 0:
                continue
            i_g = guilds.index(tgt)
            j_g = guilds.index(src)
            a_val = float(A[i_g, j_g])
            if abs(a_val) < absA_min_fit:
                continue
            s_glv = int(np.sign(a_val)) if abs(a_val) > 1e-12 else 0
            if s_glv == 0:
                continue
            if use_agree_only and s != s_glv:
                continue

            cross = float(cross_matrix[i_a, j_a])
            comp = float(comp_matrix[i_a, j_a])
            tox = float(tox_matrix[i_a, j_a]) if tox_matrix is not None else 0.0
            X_rows.append([1.0, np.log1p(cross), comp, tox])
            y.append(float(np.log1p(abs(a_val))))
            w.append(abs(a_val))

    if len(X_rows) < 3:
        N = len(GUILD_ORDER)
        return np.zeros((N, N), dtype=float), {'intercept': 0.0, 'log_cross': 0.0, 'comp': 0.0, 'tox': 0.0}, {'train_n': len(X_rows), 'n': 0, 'corr': None, 'sign_agree': None}

    X = np.asarray(X_rows, dtype=float)
    yv = np.asarray(y, dtype=float)
    W = np.sqrt(np.asarray(w, dtype=float))
    beta, _, _, _ = np.linalg.lstsq(X * W[:, None], yv * W, rcond=None)

    N = len(GUILD_ORDER)
    mag_hat = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if int(sign_matrix[i, j]) == 0:
                continue
            cross = float(cross_matrix[i, j])
            comp = float(comp_matrix[i, j])
            tox = float(tox_matrix[i, j]) if tox_matrix is not None else 0.0
            z = float(beta[0] + beta[1] * np.log1p(cross) + beta[2] * comp + beta[3] * tox)
            mag_hat[i, j] = float(np.expm1(max(z, 0.0)))

    A_hat2 = mag_hat * sign_matrix.astype(float)
    stats = compare_with_glv_continuous(A_hat2, glv_path, absA_min=absA_min, mask=(sign_matrix != 0), label='A_hat2')
    stats['train_n'] = int(len(X_rows))
    coefs = {'intercept': float(beta[0]), 'log_cross': float(beta[1]), 'comp': float(beta[2]), 'tox': float(beta[3])}
    return A_hat2, coefs, stats


def plot_results(sign_matrix, cross_matrix, df_cmp, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from guild_replicator_dieckow import GUILD_SHORT

    N = len(GUILD_ORDER)
    labels = [GUILD_SHORT.get(g, g[:5]) for g in GUILD_ORDER]
    cmap = mcolors.LinearSegmentedColormap.from_list('rwg', ['#d62728','#f5f5f5','#2ca02c'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9})

    # Panel A: minimal model sign predictions
    ax = axes[0]
    ax.imshow(sign_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('source j'); ax.set_ylabel('target i')
    ax.set_title('A.  Minimal-model sign predictions\n(green=+, red=−, grey=uncertain)')

    # Panel B: cross-feeding flux
    ax = axes[1]
    log_cross = np.log1p(cross_matrix)
    im = ax.imshow(log_cross, cmap='Greens', aspect='auto')
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title('B.  Cross-feeding score\nlog(1 + Σ importable flux, j→i)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel C: gLV A vs minimal-model (sign or A-hat)
    ax = axes[2]
    if df_cmp is not None:
        pred = df_cmp[df_cmp['has_pred']]
        if 'A_hat' in pred.columns:
            use_col = 'A_hat'
            if 'A_hat2' in pred.columns:
                use_col = 'A_hat2'
            if 'A_hat_bias' in pred.columns:
                use_col = 'A_hat_bias'
            strong = pred[pred['strongA'] & (pred['sign_glv'] != 0)]
            colors = ['#2ca02c' if r else '#d62728' for r in strong['agree']]
            ax.scatter(strong['A'], strong[use_col], c=colors, alpha=0.85, s=55, edgecolors='k', linewidths=0.4, zorder=3)
            lim = max(float(np.max(np.abs(strong['A']))), float(np.max(np.abs(strong[use_col]))), 0.1)
            ax.plot([-lim, lim], [-lim, lim], color='k', lw=0.7, ls='--')
            ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.axhline(0, color='k', lw=0.5, ls=':')
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel('gLV A'); ax.set_ylabel(use_col)
            n_a = int(strong['agree'].sum()); n_t = int(len(strong))
            ax.set_title(f'C.  gLV A vs {use_col}\n{n_a}/{n_t} sign agree ({100*n_a//max(n_t,1)}%)')
        else:
            jitter = np.random.default_rng(42).normal(0, 0.04, len(pred))
            colors = ['#2ca02c' if r else '#d62728' for r in pred['agree']]
            ax.scatter(pred['A'], pred['sign_minimal'] + jitter, c=colors,
                       alpha=0.8, s=50, edgecolors='k', linewidths=0.4, zorder=3)
            ax.axvline(0, color='k', lw=0.7, ls='--')
            ax.set_xlabel('gLV A value'); ax.set_ylabel('Minimal-model sign')
            ax.set_yticks([-1, 0, 1]); ax.set_yticklabels([r'$-$', '0', r'$+$'])
            n_a = pred['agree'].sum(); n_t = len(pred)
            ax.set_title(f'C.  gLV A vs minimal-model sign\n{n_a}/{n_t} agree ({100*n_a//max(n_t,1)}%)')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#2ca02c', label='Agree'),
                            Patch(color='#d62728', label='Disagree')], fontsize=8, loc='upper left')
    ax.spines[['top','right']].set_visible(False)

    fig.suptitle('Guild-level sign validation: manually curated minimal metabolic models\n'
                 '(HOMD + Kolenbrander 2010 literature basis; oral-fluid pFBA)',
                 fontsize=9.5, y=1.01)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = out_dir / f'fig_minimal_model_signs.{ext}'
        fig.savefig(p, bbox_inches='tight', dpi=300)
        print(f'Saved: {p}')
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--glv_fit', default='results/dieckow_cr/fit_glv_8pat_kegg_prior.json')
    parser.add_argument('--out_dir', default='/home/nishioka/IKM_Hiwi/docs/figures/dieckow')
    parser.add_argument('--cross_pos', type=float, default=0.01)
    parser.add_argument('--comp_neg', type=int, default=999)
    parser.add_argument('--cross_neg_max', type=float, default=0.001)
    parser.add_argument('--absA_min', type=float, default=0.05)
    parser.add_argument('--use_toxins', action='store_true')
    parser.add_argument('--tox_pos', type=float, default=0.1)
    parser.add_argument('--regress_A', action='store_true')
    parser.add_argument('--regress_A_bias', action='store_true')
    parser.add_argument('--regress_mag', action='store_true')
    parser.add_argument('--mag_use_all', action='store_true')
    parser.add_argument('--absA_min_mag', type=float, default=0.01)
    parser.add_argument('--cv_edges', type=int, default=0)
    parser.add_argument('--cv_seed', type=int, default=0)
    parser.add_argument('--no_tune', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    print('=== Minimal-model guild sign validation ===')
    cross_matrix, comp_matrix, tox_matrix, present, secretions, substrates = run_analysis(return_details=True)

    glv_path = Path(args.glv_fit)
    if not glv_path.is_absolute():
        glv_path = Path(__file__).parent / glv_path

    cross_pos = args.cross_pos
    comp_neg = args.comp_neg
    cross_neg_max = args.cross_neg_max
    if glv_path.exists() and not args.no_tune:
        cross_pos, comp_neg, cross_neg_max = tune_thresholds(cross_matrix, comp_matrix, glv_path, absA_min=args.absA_min)

    tox_arg = tox_matrix if args.use_toxins else None
    sign_matrix = predict_signs(
        cross_matrix,
        comp_matrix,
        cross_pos=cross_pos,
        comp_neg=comp_neg,
        cross_neg_max=cross_neg_max,
        tox_matrix=tox_arg,
        tox_pos=args.tox_pos,
    )

    df_cmp = None
    if glv_path.exists():
        df_cmp = compare_with_glv(
            sign_matrix,
            glv_path,
            absA_min=args.absA_min,
            cross_matrix=cross_matrix,
            comp_matrix=comp_matrix,
            secretions=secretions,
            substrates=substrates,
        )

        if args.regress_A:
            A_hat, coefs = fit_linear_A(comp_matrix, glv_path, secretions, substrates, absA_min=args.absA_min)
            stats = compare_with_glv_continuous(A_hat, glv_path, absA_min=args.absA_min)
            nz = sorted(((k, v) for k, v in coefs.items() if k != 'intercept'), key=lambda x: -abs(x[1]))[:8]
            top = ', '.join(f'{k}={v:+.3f}' for k, v in nz)
            print(f'Linear coefficients: intercept={coefs["intercept"]:+.3f}; {top}')
            i_map = {g: i for i, g in enumerate(GUILD_ORDER)}
            j_map = {g: j for j, g in enumerate(GUILD_ORDER)}
            df_cmp['A_hat'] = df_cmp.apply(lambda r: float(A_hat[i_map[r['tgt']], j_map[r['src']]]) if (r['tgt'] in i_map and r['src'] in j_map) else np.nan, axis=1)
            df_cmp['sign_hat'] = df_cmp['A_hat'].apply(lambda v: int(np.sign(v)) if np.isfinite(v) and abs(v) > 1e-12 else 0)
            df_cmp.attrs['A_hat_coefs'] = coefs
            df_cmp.attrs['A_hat_stats'] = stats

        if args.regress_A_bias:
            A_hat_b, coefs_b, meta_b = fit_linear_A_with_bias(comp_matrix, glv_path, secretions, substrates, absA_min=args.absA_min)
            stats_b = compare_with_glv_continuous(A_hat_b, glv_path, absA_min=args.absA_min, label='A_hat_bias')
            i_map = {g: i for i, g in enumerate(GUILD_ORDER)}
            j_map = {g: j for j, g in enumerate(GUILD_ORDER)}
            df_cmp['A_hat_bias'] = df_cmp.apply(lambda r: float(A_hat_b[i_map[r['tgt']], j_map[r['src']]]) if (r['tgt'] in i_map and r['src'] in j_map) else np.nan, axis=1)
            df_cmp['sign_hat_bias'] = df_cmp['A_hat_bias'].apply(lambda v: int(np.sign(v)) if np.isfinite(v) and abs(v) > 1e-12 else 0)
            df_cmp.attrs['A_hat_bias_coefs'] = coefs_b
            df_cmp.attrs['A_hat_bias_meta'] = meta_b
            df_cmp.attrs['A_hat_bias_stats'] = stats_b

        if args.regress_mag:
            A_hat2, coefs2, stats2 = fit_magnitude_from_features(
                cross_matrix,
                comp_matrix,
                tox_matrix if args.use_toxins else None,
                sign_matrix,
                glv_path,
                absA_min=args.absA_min,
                absA_min_fit=args.absA_min_mag,
                use_agree_only=(not args.mag_use_all),
            )
            i_map = {g: i for i, g in enumerate(GUILD_ORDER)}
            j_map = {g: j for j, g in enumerate(GUILD_ORDER)}
            df_cmp['A_hat2'] = df_cmp.apply(lambda r: float(A_hat2[i_map[r['tgt']], j_map[r['src']]]) if (r['tgt'] in i_map and r['src'] in j_map) else np.nan, axis=1)
            df_cmp['sign_hat2'] = df_cmp['A_hat2'].apply(lambda v: int(np.sign(v)) if np.isfinite(v) and abs(v) > 1e-12 else 0)
            df_cmp.attrs['A_hat2_coefs'] = coefs2
            df_cmp.attrs['A_hat2_stats'] = stats2

        out_path = Path(args.out_dir) / 'minimal_model_sign_comparison.json'
        df_cmp.to_json(out_path, orient='records', indent=2)
        print(f'Saved: {out_path}')

        if args.cv_edges and args.cv_edges > 1:
            base_mask = build_edge_mask(glv_path, absA_min=args.absA_min)
            folds = kfold_edge_masks(base_mask, k=args.cv_edges, seed=args.cv_seed)
            rows = []
            for f, (train_mask, test_mask) in enumerate(folds, start=1):
                cross_pos_f, comp_neg_f, cross_neg_max_f = tune_thresholds(
                    cross_matrix,
                    comp_matrix,
                    glv_path,
                    absA_min=args.absA_min,
                    edge_mask=train_mask,
                )
                tox_arg_f = tox_matrix if args.use_toxins else None
                sign_f = predict_signs(
                    cross_matrix,
                    comp_matrix,
                    cross_pos=cross_pos_f,
                    comp_neg=comp_neg_f,
                    cross_neg_max=cross_neg_max_f,
                    tox_matrix=tox_arg_f,
                    tox_pos=args.tox_pos,
                )
                s_train = evaluate_sign_on_mask(sign_f, glv_path, absA_min=args.absA_min, edge_mask=train_mask)
                s_test = evaluate_sign_on_mask(sign_f, glv_path, absA_min=args.absA_min, edge_mask=test_mask)
                row = {'fold': f, 'sign_train': s_train, 'sign_test': s_test}

                if args.regress_A:
                    A_hat_f, _ = fit_linear_A(comp_matrix, glv_path, secretions, substrates, absA_min=args.absA_min, edge_mask=train_mask)
                    row['A_hat_test'] = compare_with_glv_continuous(A_hat_f, glv_path, absA_min=args.absA_min, mask=test_mask, label=f'A_hat fold{f}')

                if args.regress_A_bias:
                    A_hat_b_f, _, _ = fit_linear_A_with_bias(comp_matrix, glv_path, secretions, substrates, absA_min=args.absA_min, edge_mask=train_mask)
                    row['A_hat_bias_test'] = compare_with_glv_continuous(A_hat_b_f, glv_path, absA_min=args.absA_min, mask=test_mask, label=f'A_hat_bias fold{f}')

                rows.append(row)

            test_agree = [r['sign_test']['agree'] for r in rows if r['sign_test']['agree'] is not None]
            if test_agree:
                print(f'CV sign test mean agree (k={args.cv_edges}, |A|≥{args.absA_min}): {100*float(np.mean(test_agree)):.1f}%')

            if args.regress_A_bias:
                corrs = [r['A_hat_bias_test']['corr'] for r in rows if r.get('A_hat_bias_test') and r['A_hat_bias_test']['corr'] is not None]
                sags = [r['A_hat_bias_test']['sign_agree'] for r in rows if r.get('A_hat_bias_test') and r['A_hat_bias_test']['sign_agree'] is not None]
                if corrs:
                    print(f'CV A_hat_bias test mean corr (k={args.cv_edges}, |A|≥{args.absA_min}): {float(np.mean(corrs)):.3f}')
                if sags:
                    print(f'CV A_hat_bias test mean sign_agree (k={args.cv_edges}, |A|≥{args.absA_min}): {100*float(np.mean(sags)):.1f}%')

    if args.plot or df_cmp is not None:
        plot_results(sign_matrix, cross_matrix, df_cmp, args.out_dir)
