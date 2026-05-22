#!/usr/bin/env python3
"""
AGORA2-based sign validation for the 11-guild gLV A matrix.

Pipeline:
  1. Load one representative AGORA2 GEM per guild
  2. Apply oral-fluid medium (glucose + AAs + vitamins + heme)
  3. For each ordered pair (j→i):
       - pFBA on guild j  → secretion profile S_j (mmol/gDW/h)
       - Check if guild i has uptake exchange rxn for each metabolite in S_j
       - cross-feed score  = sum of importable fluxes  (→ A[i,j] > 0)
       - competition score = shared substrate uptake overlap (→ A[i,j] < 0)
       predicted sign: + if cross-feed > competition, else - (or 0 if neither)
  4. Compare sign(A_AGORA) vs sign(A_gLV) and vs SF1 sign predictions

Usage:
  python guild_agora_signs.py --agora_dir /path/to/AGORA2/xml
  python guild_agora_signs.py --agora_dir /path/to/AGORA2/xml --plot

AGORA2 models: download from https://www.vmh.life (AGORA2 section)
  or Zenodo record DOI:10.5281/zenodo.7050029
  XML files (~2GB total); only ~10 files needed for oral guilds.
"""

#!/usr/bin/env python3
"""
11のギルド（菌群）で構成される一般化ロトカ・ヴォルテラ（gLV）モデルの相互作用行列（A行列）に対する、AGORA2ベースの符号検証スクリプト。

パイプライン（処理の流れ）:
  1. ギルドごとに、それを代表するAGORA2のゲノム規模代謝モデル（GEM）を1つずつ読み込む。
  2. 唾液ベースの培地条件（グルコース ＋ アミノ酸 ＋ ビタミン ＋ ヘム）を適用する。
  3. 各ギルドの順序対 (j → i) について以下を計算する:
       - ギルド j の pFBA（プロトタイプ流速バランス解析）を実行 → 分泌プロファイル S_j (mmol/gDW/h) を取得。
       - ギルド i が、S_j に含まれる各代謝物質の取り込み（Uptake）交換反応を持っているかを確認。
       - クロスフィード（相互栄養）スコア ＝ 取り込み可能な流速の総和（ → A[i,j] > 0 の要因）
       - 競争スコア ＝ 共有する基質の取り込みの重複度（ → A[i,j] < 0 の要因）
       予測符号: クロスフィード ＞ 競争 ならば「＋」、それ以外は「ー」（どちらも無ければ「0」）
  4. 予測された符号（sign(A_AGORA)）を、実際の gLV モデルの符号（sign(A_gLV)）および SF1（Surrogate Function 1）による符号予測と比較する。

使用方法:
  python guild_agora_signs.py --agora_dir /path/to/AGORA2/xml
  python guild_agora_signs.py --agora_dir /path/to/AGORA2/xml --plot

AGORA2モデルについて: https://www.vmh.life (AGORA2 セクション) 
または Zenodo（DOI:10.5281/zenodo.7050029）からダウンロードしてください。
XMLファイルの総容量は約2GBですが、口腔内ギルドに必要なのはそのうち10ファイル程度のみです。
"""

import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

# ── Guild definitions ──────────────────────────────────────────────────────────
GUILD_ORDER = [
    'Actinobacteria', 'Coriobacteriia', 'Bacilli', 'Clostridia',
    'Negativicutes', 'Bacteroidia', 'Flavobacteriia', 'Fusobacteriia',
    'Betaproteobacteria', 'Gammaproteobacteria',
]
# 'Other' is excluded (taxonomically too diverse for single representative)

# Representative strains — AGORA2 filename keywords (case-insensitive)
# Priority order: best-characterised oral isolate first
GUILD_REPS = {
    'Actinobacteria':    ['Actinomyces_naeslundii',       'Rothia_dentocariosa',    'Rothia_mucilaginosa'],
    'Coriobacteriia':   ['Atopobium_parvulum',            'Olsenella_uli'],
    'Bacilli':          ['Streptococcus_gordonii',        'Streptococcus_mutans',   'Streptococcus_salivarius'],
    'Clostridia':       ['Parvimonas_micra',              'Peptostreptococcus_stomatis'],
    'Negativicutes':    ['Veillonella_parvula',           'Veillonella_dispar'],
    'Bacteroidia':      ['Prevotella_melaninogenica',     'Prevotella_intermedia',  'Porphyromonas_gingivalis'],
    'Flavobacteriia':   ['Capnocytophaga_gingivalis',     'Capnocytophaga_sputigena'],
    'Fusobacteriia':    ['Fusobacterium_nucleatum',       'Leptotrichia_buccalis'],
    'Betaproteobacteria': ['Eikenella_corrodens',         'Neisseria_mucosa',        'Neisseria_sicca'],
    'Gammaproteobacteria': ['Haemophilus_parainfluenzae', 'Haemophilus_influenzae'],
}

# ── Oral-fluid medium (AGORA2 exchange reaction IDs, mmol/gDW/h upper bounds) ──
# AGORA2 uses old BiGG format: EX_{met}(e)  e.g. EX_glc_D(e), EX_ala_L(e)
# Based on unstimulated whole saliva composition (Dawes 2008, Amerongen & Veerman 2002)
ORAL_MEDIUM = {
    # Sugars
    'EX_glc_D(e)':  10.0,   # glucose
    'EX_fru(e)':     5.0,   # fructose
    'EX_sucr(e)':    5.0,   # sucrose
    'EX_lac_L(e)':   3.0,   # L-lactate (Veillonella source)
    # Amino acids
    'EX_ala_L(e)':   2.0,
    'EX_arg_L(e)':   1.5,
    'EX_asn_L(e)':   1.0,
    'EX_asp_L(e)':   1.0,
    'EX_gln_L(e)':   2.0,
    'EX_glu_L(e)':   2.0,
    'EX_gly(e)':     1.5,
    'EX_his_L(e)':   0.5,
    'EX_ile_L(e)':   0.8,
    'EX_leu_L(e)':   0.8,
    'EX_lys_L(e)':   0.8,
    'EX_met_L(e)':   0.5,
    'EX_phe_L(e)':   0.5,
    'EX_pro_L(e)':   1.0,
    'EX_ser_L(e)':   1.0,
    'EX_thr_L(e)':   0.8,
    'EX_trp_L(e)':   0.2,
    'EX_tyr_L(e)':   0.5,
    'EX_val_L(e)':   0.8,
    # Nucleotides / bases
    'EX_ade(e)':     0.5,
    'EX_gua(e)':     0.5,
    # Vitamins & cofactors
    'EX_thm(e)':     0.1,   # thiamine
    'EX_ribflv(e)':  0.1,   # riboflavin
    'EX_nac(e)':     0.1,   # nicotinate
    'EX_pnto_R(e)':  0.1,   # pantothenate
    'EX_fol(e)':     0.1,   # folate
    'EX_pydam(e)':   0.1,   # pyridoxamine (B6 form)
    'EX_pydxn(e)':   0.1,   # pyridoxine (B6 form)
    'EX_pydx(e)':    0.1,   # pyridoxal (B6 form)
    'EX_cbl1(e)':    0.01,  # cobalamin (B12)
    'EX_btn(e)':     0.05,  # biotin
    # Heme / menaquinone (for anaerobes)
    'EX_pheme(e)':   0.5,   # protoheme (Porphyromonas, Prevotella)
    'EX_mqn7(e)':    0.3,   # menaquinone-7
    'EX_mqn8(e)':    0.3,
    # Inorganic
    'EX_h2o(e)':  1000.0,
    'EX_h(e)':    1000.0,
    'EX_pi(e)':    10.0,   # phosphate
    'EX_so4(e)':    5.0,   # sulfate
    'EX_nh4(e)':   10.0,   # ammonium
    'EX_na1(e)':   50.0,
    'EX_k(e)':     10.0,
    'EX_mg2(e)':    2.0,
    'EX_ca2(e)':    2.0,
    'EX_fe2(e)':    0.5,
    'EX_fe3(e)':    0.5,
    'EX_cl(e)':    50.0,
    'EX_zn2(e)':    0.1,
    'EX_mn2(e)':    0.1,
    'EX_cobalt2(e)': 0.05,
    'EX_cu2(e)':    0.05,  # copper — essential for cytochrome oxidase
    # Sulfur amino acid / polyamines (trace, present in saliva)
    'EX_cys_L(e)':   0.5,
    'EX_ptrc(e)':    0.05,  # putrescine
    'EX_spmd(e)':    0.05,  # spermidine
    # Quinones / cofactors (AGORA models require explicit uptake)
    'EX_2dmmq8(e)':  0.1,
    'EX_sheme(e)':   0.1,
    'EX_adocbl(e)':  0.01,  # adenosylcobalamin
    'EX_q8(e)':      0.1,
    # Cell wall / fatty acid precursors (required by AGORA Gram+ / anaerobe models)
    'EX_26dap_M(e)': 0.1,   # meso-2,6-diaminopimelate (peptidoglycan)
    'EX_ocdca(e)':   0.1,   # octadecanoate (stearic acid C18)
    'EX_ttdca(e)':   0.1,   # tetradecanoate (myristic acid C14)
    'EX_ddca(e)':    0.1,   # dodecanoate (lauric acid C12)
    'EX_4hbz(e)':    0.1,   # 4-hydroxybenzoate (ubiquinone precursor)
    # Glutathione (oxidized/reduced, required by Actinobacteria / Bacteroidia)
    'EX_gthrd(e)':   0.1,   # glutathione (reduced)
    'EX_gthox(e)':   0.1,   # glutathione (oxidized)
    # Amino acid derivatives (required by Coriobacteriia / Flavobacteriia)
    'EX_orn(e)':     0.5,   # ornithine (arginine catabolism, present in saliva)
    'EX_cgly(e)':    0.1,   # cys-gly dipeptide
    # Pyrimidine nucleosides (required by Gammaproteobacteria)
    'EX_cytd(e)':    0.1,   # cytidine
    # Gases (partial anaerobic — low O2)
    'EX_o2(e)':      2.0,   # microaerophilic (0 for strict anaerobes — set per model)
    'EX_co2(e)':     5.0,
}

# ── ORAL_MEDIUM v2: realistic saliva concentrations ──────────────────────────
# v1 was ~100× too high (blood-plasma scale).
# v2 based on Dawes 2008, Tenovuo 1998 (unstimulated whole saliva, mM → mmol/gDW/h
# conversion assuming ~2 gDW/L biofilm cell density, 0.1 /h growth rate).
# Scarce carbon forces competition signals to emerge in pFBA.
ORAL_MEDIUM_V2 = {
    # Sugars — salivary glucose ~0.08 mM; others trace
    'EX_glc_D(e)':  0.10,   # glucose   (0.08 mM → ~0.10 mmol/gDW/h)
    'EX_fru(e)':    0.04,   # fructose
    'EX_sucr(e)':   0.04,   # sucrose
    'EX_lac_L(e)':  0.15,   # L-lactate (0.1-0.25 mM; Veillonella source)
    # Amino acids — total ~2-5 mM, ~0.1-0.3 mM per species
    'EX_ala_L(e)':  0.10,
    'EX_arg_L(e)':  0.08,
    'EX_asn_L(e)':  0.05,
    'EX_asp_L(e)':  0.05,
    'EX_gln_L(e)':  0.10,
    'EX_glu_L(e)':  0.10,
    'EX_gly(e)':    0.08,
    'EX_his_L(e)':  0.02,
    'EX_ile_L(e)':  0.03,
    'EX_leu_L(e)':  0.03,
    'EX_lys_L(e)':  0.03,
    'EX_met_L(e)':  0.02,
    'EX_phe_L(e)':  0.02,
    'EX_pro_L(e)':  0.05,
    'EX_ser_L(e)':  0.05,
    'EX_thr_L(e)':  0.03,
    'EX_trp_L(e)':  0.01,
    'EX_tyr_L(e)':  0.02,
    'EX_val_L(e)':  0.03,
    'EX_cys_L(e)':  0.02,
    'EX_orn(e)':    0.02,
    # Nucleotides / bases — trace in saliva
    'EX_ade(e)':    0.005,
    'EX_gua(e)':    0.005,
    # Vitamins & cofactors — nM–μM range → very small FBA bounds
    'EX_thm(e)':    0.002,
    'EX_ribflv(e)': 0.002,
    'EX_nac(e)':    0.002,
    'EX_pnto_R(e)': 0.002,
    'EX_fol(e)':    0.001,
    'EX_pydam(e)':  0.001,
    'EX_pydxn(e)':  0.001,
    'EX_pydx(e)':   0.001,
    'EX_cbl1(e)':   0.0005,
    'EX_btn(e)':    0.001,
    # Heme / quinones — keep small but nonzero for anaerobes
    'EX_pheme(e)':  0.02,
    'EX_mqn7(e)':   0.01,
    'EX_mqn8(e)':   0.01,
    # Inorganic — abundant in saliva, keep proportional
    'EX_h2o(e)':  1000.0,
    'EX_h(e)':    1000.0,
    'EX_pi(e)':      2.0,
    'EX_so4(e)':     1.0,
    'EX_nh4(e)':     2.0,
    'EX_na1(e)':    50.0,
    'EX_k(e)':      10.0,
    'EX_mg2(e)':     0.5,
    'EX_ca2(e)':     0.5,
    'EX_fe2(e)':     0.02,
    'EX_fe3(e)':     0.02,
    'EX_cl(e)':     50.0,
    'EX_zn2(e)':    0.005,
    'EX_mn2(e)':    0.005,
    'EX_cobalt2(e)': 0.002,
    'EX_cu2(e)':    0.002,
    # Polyamines
    'EX_ptrc(e)':   0.002,
    'EX_spmd(e)':   0.002,
    # Cofactors required by AGORA models (keep small)
    'EX_2dmmq8(e)': 0.005,
    'EX_sheme(e)':  0.005,
    'EX_adocbl(e)': 0.0005,
    'EX_q8(e)':     0.005,
    'EX_26dap_M(e)': 0.005,
    'EX_ocdca(e)':  0.005,
    'EX_ttdca(e)':  0.005,
    'EX_ddca(e)':   0.005,
    'EX_4hbz(e)':   0.005,
    'EX_gthrd(e)':  0.005,
    'EX_gthox(e)':  0.005,
    'EX_cgly(e)':   0.005,
    'EX_cytd(e)':   0.005,
    # Gases
    'EX_o2(e)':     0.5,    # reduced O2 (biofilm is more anoxic)
    'EX_co2(e)':    2.0,
}

# Guilds that are strict anaerobes (set O2 to 0)
ANAEROBIC_GUILDS = {'Clostridia', 'Bacteroidia', 'Fusobacteriia', 'Negativicutes'}

# ── Helpers ────────────────────────────────────────────────────────────────────
def find_model_path(agora_dir: Path, candidates: list[str]) -> Path | None:
    """Return first AGORA2 XML file matching any candidate keyword."""
    for cand in candidates:
        parts = cand.split('_')
        genus = parts[0]
        patterns = [f"*{cand}*.xml", f"*{cand}*.json"]
        if len(parts) > 1:
            patterns.append(f"*{genus}*{parts[1]}*.xml")
        for pattern in patterns:
            try:
                found = sorted(agora_dir.glob(pattern))
            except ValueError:
                continue
            if found:
                return found[0]
    return None


def load_model(path: Path):
    from cobra.io import read_sbml_model, load_json_model
    if path.suffix == '.json':
        return load_json_model(str(path))
    return read_sbml_model(str(path))


def apply_medium(model, guild: str, medium_dict=None):
    """Close all exchange reactions, then open medium_dict ones.

    medium_dict defaults to ORAL_MEDIUM (v1). Pass ORAL_MEDIUM_V2 for v2.
    """
    if medium_dict is None:
        medium_dict = ORAL_MEDIUM
    model_rxn_ids = {rxn.id for rxn in model.exchanges}
    medium = {}
    for rxn in model.exchanges:
        rxn_id = rxn.id
        if rxn_id in medium_dict:
            medium[rxn_id] = medium_dict[rxn_id]
        else:
            medium[rxn_id] = 0.0
    # Strict anaerobes: shut off O2 (only if the rxn exists in this model)
    if guild in ANAEROBIC_GUILDS and 'EX_o2(e)' in model_rxn_ids:
        medium['EX_o2(e)'] = 0.0
    model.medium = medium


def run_pfba(model):
    """Return parsimonious FBA solution (None on infeasible)."""
    from cobra.flux_analysis import pfba
    try:
        sol = pfba(model)
        if sol.status != 'optimal':
            return None
        return sol
    except Exception:
        return None


def get_secretions(sol, model, threshold=1e-6) -> dict[str, float]:
    """Return dict of metabolite_id → flux for secreted metabolites (flux > 0 out)."""
    secs = {}
    for rxn in model.exchanges:
        f = sol.fluxes.get(rxn.id, 0.0)
        # Exchange rxn convention: positive = secretion
        if f > threshold:
            # get metabolite BiGG id (strip _e suffix)
            met_id = list(rxn.metabolites.keys())[0].id
            secs[met_id] = f
    return secs


def get_substrates(model) -> set[str]:
    """Return set of metabolite IDs that model can consume from medium."""
    subs = set()
    for rxn in model.exchanges:
        lb = rxn.lower_bound
        if lb < 0:  # can take up
            met_id = list(rxn.metabolites.keys())[0].id
            subs.add(met_id)
    return subs


# ── Main analysis ──────────────────────────────────────────────────────────────
def build_agora_sign_matrix(agora_dir: Path, verbose=True):
    """
    Returns (sign_matrix, cross_matrix, comp_matrix, guild_list, model_map).
    sign_matrix[i,j] = predicted sign of A[i,j] (effect of j on i).
    """
    N = len(GUILD_ORDER)
    sign_matrix = np.zeros((N, N), dtype=int)
    cross_matrix = np.zeros((N, N))  # cross-feeding score (j secretes → i consumes)
    comp_matrix  = np.zeros((N, N))  # competition score (shared substrates)

    # Load models
    models = {}
    secretions = {}
    substrates = {}
    for guild in GUILD_ORDER:
        path = find_model_path(agora_dir, GUILD_REPS[guild])
        if path is None:
            print(f'  [MISSING] {guild}: no AGORA2 model found in {agora_dir}')
            continue
        print(f'  Loading {guild}: {path.name}')
        try:
            m = load_model(path)
            apply_medium(m, guild)
            sol = run_pfba(m)
            if sol is None:
                print(f'    infeasible — skipping {guild}')
                continue
            models[guild]     = m
            secretions[guild] = get_secretions(sol, m)
            substrates[guild] = get_substrates(m)
            mu = sol.objective_value
            print(f'    μ = {mu:.4f}  secretions: {len(secretions[guild])} metabolites')
        except Exception as e:
            print(f'    ERROR loading {guild}: {e}')

    # Pairwise scores
    present = [g for g in GUILD_ORDER if g in models]
    for src in present:
        j = GUILD_ORDER.index(src)
        secs_j   = secretions[src]
        subs_j   = substrates[src]
        for tgt in present:
            if src == tgt:
                continue
            i = GUILD_ORDER.index(tgt)
            subs_i = substrates[tgt]

            # Cross-feeding: src secretes → tgt can import
            cross = sum(
                flux for met, flux in secs_j.items()
                if (met in subs_i or f"{met}_e" in {r.id.replace('EX_','').replace('_e','') for r in models[tgt].exchanges})
            )

            # Competition: both guilds can consume same substrates
            shared = subs_j & subs_i
            comp = len(shared)  # count shared substrates (proxy)

            cross_matrix[i, j] = cross
            comp_matrix[i, j]  = comp

            if cross > 0.01:
                sign_matrix[i, j] = +1
            elif comp > 5 and cross < 0.001:
                sign_matrix[i, j] = -1
            # else 0 = uncertain

    return sign_matrix, cross_matrix, comp_matrix, present, models


def get_agora_phi_matrix(agora_dir: Path, verbose=False) -> tuple[np.ndarray, np.ndarray]:
    """
    MacArthur consumer-resource prior matrix from AGORA2 FBA.

    Returns
    -------
    phi_net : (N, N) float
        Phi[i,j] = normalised cross-feeding(j→i) − normalised competition(i,j)
        Range roughly (−1, +1).  Used as prior mean for A[i,j].
    mask    : (N, N) bool
        True for pairs where at least one FBA model is available.

    Derivation (MacArthur 1970, Marsland et al. 2019 PLOS CB):
        A[i,j] ≈  Σ_α  s_{jα} · c_{iα}   (cross-feeding, +)
                 − Σ_α  c_{iα} · c_{jα}   (competition for shared resources, −)
    where s_{jα} = secretion flux (FBA), c_{iα} = max uptake flux (FBA bounds).
    """
    N = len(GUILD_ORDER)
    cross_raw = np.zeros((N, N))   # Σ_α s_{jα} · c_{iα}
    comp_raw  = np.zeros((N, N))   # Σ_α c_{iα} · c_{jα}  (shared resource overlap)

    # Load models & run pFBA
    secretions = {}   # guild → {met_id: flux}
    uptake_cap = {}   # guild → {met_id: max_uptake}  (from lb of exchange rxns)
    present = []

    for guild in GUILD_ORDER:
        path = find_model_path(agora_dir, GUILD_REPS[guild])
        if path is None:
            continue
        try:
            m = load_model(path)
            apply_medium(m, guild)
            sol = run_pfba(m)
            if sol is None or sol.objective_value < 1e-6:
                continue
            sec, cap = {}, {}
            for rxn in m.exchanges:
                met_id = list(rxn.metabolites.keys())[0].id
                f = sol.fluxes.get(rxn.id, 0.0)
                if f > 1e-6:
                    sec[met_id] = f           # secretion (positive FBA flux)
                elif f < -1e-6:
                    cap[met_id] = abs(f)      # actual uptake flux (not raw lb)
            secretions[guild] = sec
            uptake_cap[guild] = cap
            present.append(guild)
            if verbose:
                print(f'  Phi: {guild}  μ={sol.objective_value:.2f}  '
                      f'sec={len(sec)}  cap={len(cap)}')
        except Exception as e:
            if verbose:
                print(f'  Phi: {guild} ERROR {e}')

    # Exclude non-informative exchange metabolites from cross-feeding signal
    EXCLUDE_METS = {
        'h2o[e]', 'h[e]', 'co2[e]', 'o2[e]', 'na1[e]', 'k[e]',
        'cl[e]', 'pi[e]', 'so4[e]', 'nh4[e]', 'ca2[e]', 'mg2[e]',
    }

    # Pairwise MacArthur terms
    for src in present:                    # src = j (producer)
        j = GUILD_ORDER.index(src)
        secs_j = {m: f for m, f in secretions[src].items()
                  if m not in EXCLUDE_METS}
        cap_j  = uptake_cap[src]
        for tgt in present:                # tgt = i (consumer)
            if src == tgt:
                continue
            i = GUILD_ORDER.index(tgt)
            cap_i = uptake_cap[tgt]

            # Cross-feeding: Σ_α s_{jα} · min(c_{iα}, s_{jα})
            # min() caps the benefit at what j actually produces
            cf = sum(
                secs_j[m] * min(cap_i.get(m, 0.0), secs_j[m])
                for m in secs_j if m in cap_i and m not in EXCLUDE_METS
            )
            cross_raw[i, j] = cf

            # Competition: cosine similarity of uptake vectors (resource niche overlap)
            # α_{ij} = (c_i · c_j) / (|c_i| |c_j|) — stays in [0, 1]
            shared = set(cap_i) & set(cap_j)
            if shared:
                dot = sum(cap_i[m] * cap_j[m] for m in shared)
                norm_i = sum(v**2 for v in cap_i.values()) ** 0.5
                norm_j = sum(v**2 for v in cap_j.values()) ** 0.5
                comp_raw[i, j] = dot / (norm_i * norm_j + 1e-12)

    # Normalise cross-feeding to [0, 1]; competition already in [0, 1]
    mx_cf = cross_raw.max()
    phi_cf   = cross_raw / mx_cf if mx_cf > 1e-12 else cross_raw
    phi_comp = comp_raw   # already cosine similarity ∈ [0, 1]

    # Net MacArthur Phi: cross-feeding positive, competition negative
    phi_net = phi_cf - phi_comp

    # Mask: pair has FBA support if both guilds have models
    mask = np.zeros((N, N), dtype=bool)
    for src in present:
        for tgt in present:
            if src != tgt:
                mask[GUILD_ORDER.index(tgt), GUILD_ORDER.index(src)] = True

    if verbose:
        pos_pairs = int((phi_net > 0).sum())
        neg_pairs = int((phi_net < 0).sum())
        print(f'  Phi net: {pos_pairs} positive (cross-feeding), '
              f'{neg_pairs} negative (competition)  '
              f'range=[{phi_net.min():.3f}, {phi_net.max():.3f}]')

    return phi_net, mask


def compute_micom_signals(agora_dir: Path, medium_dict=None,
                          flux_threshold=0.01, fraction=0.5, verbose=False):
    """
    Community-aware cross-feeding signals via MICOM cooperative tradeoff.

    Unlike single-species pFBA (which checks whether j CAN secrete what i
    needs), MICOM runs a joint community FBA so that each species' secretion
    profile is resolved in the presence of competitors.  We extract the sign
    of the metabolite exchange, not the growth rate change, because the
    cooperative tradeoff's max-min fairness objective distorts growth rates.

    Algorithm
    ---------
    1. Build a 10-guild community model from AGORA2 SBML files.
    2. Set the shared medium and run ``cooperative_tradeoff(fluxes=True)``.
    3. For each exchange reaction r:
       Cross-feeding (non-toxin):
         secretors (flux > threshold) → consumers (flux < −threshold)
         pos[consumer, secretor] += min(src_flux, |tgt_flux|)   [flux magnitude]
       Toxins (H2O2, H2S):
         secretors harm ALL other guilds in community (not just metabolic consumers)
         neg[any_guild, secretor] += src_flux

    Using flux magnitude (instead of binary count) gives stronger weight to
    high-flux cross-feeding (e.g., lactate 97 mmol/gDW/h) over low-flux amino
    acid transfers (0.5 mmol/gDW/h).  fraction=0.5 is Diener 2020's default.

    Returns
    -------
    pos_cf : (N_G, N_G) float
        pos_cf[i, j] = total flux magnitude transferred from j to i.
    neg_tox : (N_G, N_G) float
        neg_tox[i, j] = total toxin flux secreted by j (harming i).
    present : list[str]
        Guilds for which a model was loaded.
    """
    try:
        import micom
    except ImportError:
        if verbose:
            print('  MICOM not installed; skipping community FBA')
        return np.zeros((len(GUILD_ORDER), len(GUILD_ORDER))), \
               np.zeros((len(GUILD_ORDER), len(GUILD_ORDER))), []

    if medium_dict is None:
        medium_dict = ORAL_MEDIUM

    N = len(GUILD_ORDER)
    pos_cf  = np.zeros((N, N))
    neg_tox = np.zeros((N, N))
    TOXIN_EXCH = {'EX_h2o2(e)', 'EX_h2s(e)'}

    def _to_micom_id(rxn_id):
        return rxn_id[:-3] + '_m' if rxn_id.endswith('(e)') else rxn_id

    # Build community taxonomy table
    rows = []
    for guild in GUILD_ORDER:
        path = find_model_path(agora_dir, GUILD_REPS[guild])
        if path is not None:
            rows.append({'id': guild, 'file': str(path), 'abundance': 1.0})

    if not rows:
        return pos_cf, neg_tox, []

    import pandas as pd
    tax = pd.DataFrame(rows)
    present = [r['id'] for r in rows]

    if verbose:
        print(f'  MICOM: building {len(present)}-guild community ...')

    try:
        com = micom.Community(tax, progress=False)
    except Exception as e:
        if verbose:
            print(f'  MICOM: Community build failed ({e})')
        return pos_cf, neg_tox, present

    # Set medium (convert IDs: EX_glc_D(e) → EX_glc_D_m)
    exch_set = {r.id for r in com.exchanges}
    med = {_to_micom_id(k): v for k, v in medium_dict.items()
           if _to_micom_id(k) in exch_set}
    com.medium = pd.Series(med)

    if verbose:
        print(f'  MICOM: medium {len(med)}/{len(medium_dict)} components matched')

    # Run cooperative tradeoff (fraction=0.5 is Diener 2020 default)
    try:
        sol = com.cooperative_tradeoff(fraction=fraction, fluxes=True)
    except Exception as e:
        if verbose:
            print(f'  MICOM: cooperative_tradeoff failed ({e})')
        return pos_cf, neg_tox, present

    if sol.status != 'optimal':
        if verbose:
            print(f'  MICOM: infeasible ({sol.status})')
        return pos_cf, neg_tox, present

    fluxes = sol.fluxes   # DataFrame: rows=species(+medium), cols=reactions

    exch_rxn_cols = [c for c in fluxes.columns
                     if c.startswith('EX_') and c.endswith('(e)')]

    cf_pairs = 0
    for rxn_id in exch_rxn_cols:
        is_toxin = rxn_id in TOXIN_EXCH
        for src in present:
            if src not in fluxes.index:
                continue
            src_flux = fluxes.loc[src, rxn_id] if rxn_id in fluxes.columns else 0.0
            if not np.isfinite(src_flux) or src_flux <= flux_threshold:
                continue
            j = GUILD_ORDER.index(src)

            if is_toxin:
                # Toxins diffuse and harm all co-occurring guilds, not just
                # those that metabolically consume them.
                for tgt in present:
                    if tgt == src:
                        continue
                    i = GUILD_ORDER.index(tgt)
                    neg_tox[i, j] += src_flux
                    cf_pairs += 1
            else:
                # Cross-feeding: only guilds that actually consume this metabolite.
                # Weight by actual flux transfer (min of secretion and uptake).
                # Guard against NaN: species lacking a reaction have NaN flux.
                for tgt in present:
                    if tgt == src or tgt not in fluxes.index:
                        continue
                    tgt_flux = fluxes.loc[tgt, rxn_id] if rxn_id in fluxes.columns else 0.0
                    if not np.isfinite(tgt_flux) or tgt_flux >= -flux_threshold:
                        continue
                    i = GUILD_ORDER.index(tgt)
                    transfer = min(src_flux, abs(tgt_flux))
                    pos_cf[i, j] += transfer
                    cf_pairs += 1

    if verbose:
        n_pos = int((pos_cf > 0).sum())
        n_neg = int((neg_tox > 0).sum())
        print(f'  MICOM: {cf_pairs} community flux events  '
              f'({n_pos} pos pairs, {n_neg} toxin pairs)  '
              f'fraction={fraction}')

    return pos_cf, neg_tox, present


def compute_growth_suppression(agora_dir: Path, medium_dict=None,
                               threshold=0.05, verbose=False):
    """
    Growth-rate suppression competition matrix.

    competition[i, j] = max(0, (μ_i(full) − μ_i(depleted_by_j)) / μ_i(full))

    For each pair (i, j): build a depleted medium by subtracting guild j's
    pFBA uptake fluxes from the shared medium, then re-run pFBA for guild i.
    A positive value means guild j reduces guild i's growth rate by competing
    for the same limiting resources.

    Returns
    -------
    comp : (N_G, N_G) float  — relative growth suppression (0 = no competition)
    present : list[str]      — guilds for which a model was found and solved
    """
    from cobra.flux_analysis import pfba as _pfba

    if medium_dict is None:
        medium_dict = ORAL_MEDIUM

    N = len(GUILD_ORDER)

    # ── Phase 1: run pFBA for all guilds on full medium ───────────────────────
    mu_full       = {}   # guild → growth rate
    uptake_fluxes = {}   # guild → {exchange_rxn_id: |flux|}
    model_paths   = {}   # guild → Path

    for guild in GUILD_ORDER:
        path = find_model_path(agora_dir, GUILD_REPS[guild])
        if path is None:
            continue
        try:
            model = load_model(path)
            apply_medium(model, guild, medium_dict=medium_dict)
            sol = _pfba(model)
            if sol.status != 'optimal' or sol.objective_value < 1e-6:
                continue
            mu_full[guild] = sol.objective_value
            upt = {}
            for rxn in model.exchanges:
                f = sol.fluxes.get(rxn.id, 0.0)
                if f < -threshold:
                    upt[rxn.id] = abs(f)
            uptake_fluxes[guild] = upt
            model_paths[guild]   = path
            if verbose:
                print(f'  GRS phase1 {guild}: μ={sol.objective_value:.3f}  '
                      f'upt={len(upt)}')
        except Exception as e:
            if verbose:
                print(f'  GRS phase1 {guild}: ERROR {e}')

    present = list(mu_full.keys())

    # ── Phase 2: for each (i, j), deplete medium by j and re-run i ───────────
    comp = np.zeros((N, N))

    for guild_j in present:
        j = GUILD_ORDER.index(guild_j)
        # medium after guild j has consumed its share
        depleted = dict(medium_dict)
        for rxn_id, flux_mag in uptake_fluxes[guild_j].items():
            if rxn_id in depleted:
                depleted[rxn_id] = max(0.0, depleted[rxn_id] - flux_mag)

        for guild_i in present:
            if guild_i == guild_j:
                continue
            i = GUILD_ORDER.index(guild_i)
            try:
                model_i = load_model(model_paths[guild_i])
                apply_medium(model_i, guild_i, medium_dict=depleted)
                sol_d = _pfba(model_i)
                mu_d = (sol_d.objective_value
                        if sol_d.status == 'optimal' and sol_d.objective_value > 1e-8
                        else 0.0)
            except Exception:
                mu_d = 0.0
            mu_f = mu_full[guild_i]
            comp[i, j] = max(0.0, (mu_f - mu_d) / (mu_f + 1e-12))

    if verbose:
        n_nonzero = int((comp > 0.01).sum())
        print(f'  GRS: {n_nonzero} pairs with >1% growth suppression  '
              f'max={comp.max():.3f}')

    return comp, present


def compare_with_glv(sign_agora, present_guilds, glv_path: Path):
    """Compare AGORA sign predictions with gLV A matrix signs."""
    d = json.load(open(glv_path))
    A = np.array(d['A'])
    guilds = d['guilds']

    rows = []
    for i_a, src in enumerate(GUILD_ORDER):
        if src not in present_guilds: continue
        for j_a, tgt in enumerate(GUILD_ORDER):
            if src == tgt or tgt not in present_guilds: continue
            if src not in guilds or tgt not in guilds: continue
            i_g = guilds.index(tgt)
            j_g = guilds.index(src)
            a_val = A[i_g, j_g]
            s_glv   = int(np.sign(a_val))
            s_agora = sign_agora[GUILD_ORDER.index(tgt), GUILD_ORDER.index(src)]
            rows.append({
                'src': src, 'tgt': tgt,
                'A': a_val,
                'sign_glv': s_glv,
                'sign_agora': s_agora,
                'agree': (s_agora != 0 and s_agora == s_glv),
                'has_agora_pred': s_agora != 0,
            })

    df = pd.DataFrame(rows)
    predicted = df[df['has_agora_pred']]
    n_pred  = len(predicted)
    n_agree = predicted['agree'].sum()
    print(f'\nAGORA sign predictions: {n_pred} pairs predicted, {n_agree}/{n_pred} agree with gLV A ({100*n_agree/max(n_pred,1):.0f}%)')
    return df


def plot_comparison(sign_agora, cross_matrix, df_cmp, out_dir: Path):
    GUILD_SHORT = {
        'Actinobacteria': 'Actin.', 'Coriobacteriia': 'Coriob.', 'Bacilli': 'Bacil.',
        'Clostridia': 'Clost.', 'Negativicutes': 'Negat.', 'Bacteroidia': 'Bact.',
        'Flavobacteriia': 'Flavo.', 'Fusobacteriia': 'Fusob.',
        'Betaproteobacteria': 'β-Prot.', 'Gammaproteobacteria': 'γ-Prot.',
    }
    N = len(GUILD_ORDER)
    labels = [GUILD_SHORT.get(g, g[:5]) for g in GUILD_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: AGORA sign matrix
    ax = axes[0]
    cmap = mcolors.LinearSegmentedColormap.from_list('rwg', ['#d62728','#ffffff','#2ca02c'])
    im = ax.imshow(sign_agora, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title('AGORA2-predicted sign(A)\n(green=+, red=−, white=uncertain)', fontsize=9)
    ax.set_xlabel('source guild (j)'); ax.set_ylabel('target guild (i)')

    # Panel B: cross-feeding score (log scale)
    ax = axes[1]
    log_cross = np.log1p(cross_matrix)
    im2 = ax.imshow(log_cross, cmap='Greens')
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title('Cross-feeding score log(1+Σflux)\n(j→i secretion importable by i)', fontsize=9)
    plt.colorbar(im2, ax=ax, fraction=0.046)

    # Panel C: agreement scatter
    ax = axes[2]
    if df_cmp is not None:
        pred = df_cmp[df_cmp['has_agora_pred']]
        colors = ['#2ca02c' if r else '#d62728' for r in pred['agree']]
        ax.scatter(pred['A'], pred['sign_agora'] + np.random.normal(0, 0.03, len(pred)),
                   c=colors, alpha=0.8, s=50, edgecolors='k', linewidths=0.4)
        ax.axvline(0, color='k', lw=0.7, ls='--')
        ax.set_xlabel('gLV A value'); ax.set_ylabel('AGORA predicted sign')
        ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(['−', '0', '+'])
        n_ag = pred['agree'].sum(); n_tot = len(pred)
        ax.set_title(f'gLV A vs AGORA sign\n{n_ag}/{n_tot} agree ({100*n_ag//max(n_tot,1)}%)', fontsize=9)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#2ca02c', label='Agree'),
                            Patch(color='#d62728', label='Disagree')], fontsize=8)

    fig.suptitle('AGORA2 genome-scale metabolic model sign validation\n(oral-fluid medium, pFBA)', fontsize=10)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = out_dir / f'fig_agora_sign_validation.{ext}'
        fig.savefig(p, bbox_inches='tight', dpi=300)
        print(f'Saved: {p}')
    plt.close()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument('--agora_dir', default='/home/nishioka/IKM_Hiwi/nife/data/agora2_xml',
                        help='Directory containing AGORA2 XML files')
    parser.add_argument('--glv_fit',   default='results/dieckow_cr/fit_glv_8pat_kegg_prior.json')
    parser.add_argument('--out_dir',   default='/home/nishioka/IKM_Hiwi/docs/figures/dieckow')
    parser.add_argument('--plot',      action='store_true')
    args = parser.parse_args()

    agora_dir = Path(args.agora_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not agora_dir.exists():
        print(f'AGORA2 directory not found: {agora_dir}')
        print('Download AGORA2 XML files from VMH (vmh.life) or Zenodo DOI:10.5281/zenodo.7050029')
        print('Place XML files in:', agora_dir)
        agora_dir.mkdir(parents=True, exist_ok=True)
        print('\nRequired species (one representative per guild):')
        for g, cands in GUILD_REPS.items():
            print(f'  {g:22s}: {cands[0]}')
        raise SystemExit(1)

    print('=== AGORA2 sign validation ===')
    sign_agora, cross_mat, comp_mat, present, _ = build_agora_sign_matrix(agora_dir)

    df_cmp = None
    glv_path = Path(args.glv_fit)
    if not glv_path.is_absolute():
        glv_path = Path(__file__).parent / glv_path
    if glv_path.exists():
        df_cmp = compare_with_glv(sign_agora, present, glv_path)
        # Save comparison
        out_json = out_dir / 'agora_sign_comparison.json'
        df_cmp.to_json(out_json, orient='records', indent=2)
        print(f'Saved: {out_json}')
    else:
        print(f'gLV fit not found: {glv_path}')

    if args.plot or df_cmp is not None:
        plot_comparison(sign_agora, cross_mat, df_cmp, out_dir)
