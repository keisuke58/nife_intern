#!/usr/bin/env python3
"""
build_net_flow_expanded.py — multi-source sign-prior flow matrix.

Layers
------
L1  Szafranski Suppl. File 1, experimental evidence
      KEGG/HMDB-annotated metabolite: weight 2.0
      unannotated metabolite:         weight 1.5
L2  Szafranski Suppl. File 1, prediction evidence
      weight 1.0  (lower confidence)
L3  AGORA genome-scale metabolic models — FBA cross-feeding
      weight 0.5  (computational)

Model-specific entry points
----------------------------
Hamilton model  — A is symmetric (A[i,j] = A[j,i]):
    net = net_flow_hamilton(...)   # symmetrized, amensalism → unconstrained

gLV model       — A is a full N×N matrix (no symmetry constraint):
    net = net_flow_glv(...)        # directed, preserves amensalism

Low-level (full control):
    net = build_net_flow_expanded(..., symmetrize=True/False)

Differences from original build_net_flow() in loo_cv_kegg_prior.py
-------------------------------------------------------------------
- experimental vs prediction evidence weighted separately
- Additional relationship types: RELEASES → PRODUCES, HYDROLYSES/DEGRADES → USES
- Expanded genus→guild mapping (typos + additional genera)
- Optional AGORA FBA layer
- Exploitative competition term (guilds consuming the same substrate)
- Environmental variables (O₂, CO₂, H₂O₂) excluded from competition
"""
from pathlib import Path
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent

from guild_replicator_dieckow import GUILD_ORDER, N_G

# Excel (2026-04-16 update, 351 rows) is the latest version of Szafranski Suppl. File 1.
# Falls back to the TSV if the Excel is absent.
_SUPPFILE_XL  = _here / 'Datasets' / \
    '20260416_AbutmentPapernpjBiofilmsDieckow_SI_Relationships.xlsx'
_SUPPFILE_TSV = _here / 'Szafranski_Published_Work' / 'Szafranski_Published_Work' / \
    'public_data' / 'Dieckow' / \
    'Supplementary_File_1_microbe_metabolite_enzyme_interactions.tsv'

def _load_suppfile():
    if _SUPPFILE_XL.exists():
        return pd.read_excel(_SUPPFILE_XL)
    return pd.read_csv(_SUPPFILE_TSV, sep='\t')

# ── Original genus → guild mapping (from loo_cv_kegg_prior.py) ───────────────
GENUS_GUILD = {
    'Actinomyces': 'Actinobacteria', 'Bifidobacterium': 'Actinobacteria',
    'Rothia': 'Actinobacteria', 'Schaalia': 'Actinobacteria',
    'Streptococcus': 'Bacilli', 'Gemella': 'Bacilli', 'Granulicatella': 'Bacilli',
    'Abiotrophia': 'Bacilli', 'Lactiplantibacillus': 'Bacilli',
    'Prevotella': 'Bacteroidia', 'Porphyromonas': 'Bacteroidia',
    'Tannerella': 'Bacteroidia', 'Alloprevotella': 'Bacteroidia',
    'Capnocytophaga': 'Flavobacteriia',
    'Neisseria': 'Betaproteobacteria', 'Eikenella': 'Betaproteobacteria',
    'Aggregatibacter': 'Betaproteobacteria',
    'Fusobacterium': 'Fusobacteriia', 'Leptotrichia': 'Fusobacteriia',
    'Haemophilus': 'Gammaproteobacteria', 'Pseudomonas': 'Gammaproteobacteria',
    'Veillonella': 'Negativicutes', 'Dialister': 'Negativicutes',
    'Parvimonas': 'Clostridia', 'Peptostreptococcus': 'Clostridia',
    'Eggerthella': 'Coriobacteriia', 'Atopobium': 'Coriobacteriia',
    'Olsenella': 'Coriobacteriia',
}

# ── Extended genus → guild mapping (typos + additional genera) ────────────────
GENUS_GUILD_EXTRA = {
    # Szafranski typos / synonyms
    'Actinomaces':         'Actinobacteria',    # typo for Actinomyces
    'Eikenalla':           'Betaproteobacteria', # typo for Eikenella
    'Streptocccus':        'Bacilli',            # typo for Streptococcus
    # Additional genera with clear guild assignments
    'Lactobacillus':       'Bacilli',
    'Limosilactobacillus': 'Bacilli',
    'Lancefieldella':      'Bacilli',            # formerly Streptococcus
    'Corynebacterium':     'Actinobacteria',
    'Alloscardovia':       'Actinobacteria',     # Bifidobacteriales
    'Arachnia':            'Actinobacteria',     # Propionibacteriales
    'Kingella':            'Betaproteobacteria',
    'Lautropia':           'Betaproteobacteria',
    'Mogibacterium':       'Clostridia',
    'Selenomonas':         'Negativicutes',
}

def _guild_of(taxon_str):
    genus = str(taxon_str).split()[0]
    return GENUS_GUILD.get(genus) or GENUS_GUILD_EXTRA.get(genus)


# Metabolites excluded from the competition term.
# These are environmental variables or detoxification substrates, not limiting
# nutrients — treating them as contested resources would misrepresent ecology.
_COMPETITION_EXCLUDE = {
    'oxygen',            # environmental; anaerobe sensitivity handled via IS_INHIBITED_BY
    'carbon dioxide',    # capnophilic requirement, not a carbon-source competition
    'hydrogen peroxide', # detoxification (catalase), not nutrient competition
}


def build_net_flow_expanded(use_agora=True, verbose=False, agora_weight=1.0,
                            competition_weight=0.5, symmetrize=True,
                            agora_medium='v1', agora_comp_weight=0.5,
                            micom_fraction=0.5):
    """Multi-source flow matrix. See module docstring for details.

    competition_weight : float
        Scaling factor for exploitative competition from Szafranski L1/L2 terms.
        Environmental variables (O₂, CO₂, H₂O₂) are excluded automatically.
        Set to 0.0 to reproduce the original (cross-feeding only) behaviour.

    symmetrize : bool (default True)
        If True, return (net + net.T) / 2 before returning.
        The Hamilton model constrains A to be symmetric (A[i,j] = A[j,i]),
        so the sign prior must also be symmetric.

    agora_medium : str ('v1', 'v2', or 'micom')
        'v1': original ORAL_MEDIUM (~blood-plasma scale).
        'v2': realistic saliva concentrations (Dawes 2008).  Under v2,
              growth-rate suppression competition is also computed.
        'micom': community FBA via MICOM cooperative tradeoff.  Cross-feeding
              signals are extracted from actual community fluxes (not single-
              species pFBA), giving more realistic cross-feeding detection.
              Requires the micom package: pip install micom.

    agora_comp_weight : float (default 0.5)
        Scaling for AGORA growth-rate-suppression competition term (v2 only).
        Set to 0.0 to disable AGORA competition entirely.

    micom_fraction : float (default 0.5)
        Cooperative tradeoff fraction τ passed to MICOM (micom mode only).
        τ = fraction of max-growth each species must achieve.
        Diener 2020 default is 0.5.  Lower values allow more specialisation.
    """
    gi  = {g: idx for idx, g in enumerate(GUILD_ORDER)}
    pos = np.zeros((N_G, N_G))
    neg = np.zeros((N_G, N_G))

    # ── L1 + L2: Szafranski (Excel 2026-04-16 update, falls back to TSV) ────
    df = _load_suppfile()

    def szaf_weight(row):
        is_exp = str(row.get('EVIDENCE', '')).strip() == 'experimental'
        kegg   = str(row.get('KEGG', ''))
        hmdb   = str(row.get('HMDB_ID', ''))
        has_db = kegg not in ('n/a', '', 'nan', 'NaN') or 'HMDB' in hmdb
        if is_exp:
            return 2.0 if has_db else 1.5   # L1
        else:
            return 1.0                       # L2

    prod_rel  = {'PRODUCES', 'RELEASES'}
    cons_rel  = {'USES', 'DEPENDS_ON', 'HYDROLYSES', 'DEGRADES'}
    inhib_rel = {'IS_INHIBITED_BY'}

    for met in df['OBJECT'].unique():
        mdf  = df[df['OBJECT'] == met]
        w    = float(mdf.apply(szaf_weight, axis=1).max())
        prod, cons, inhib = set(), set(), set()
        for _, row in mdf.iterrows():
            g   = _guild_of(row['TAXON'])
            rel = str(row['RELATIONSHIP'])
            if g is None or g not in gi:
                continue
            if rel in prod_rel:
                prod.add(g)
            elif rel in cons_rel:
                cons.add(g)
            elif rel in inhib_rel:
                inhib.add(g)

        # cross-feeding: producer → consumer (positive)
        for src in prod:
            for tgt in cons:
                if src != tgt:
                    pos[gi[tgt], gi[src]] += w
            for tgt in inhib:
                if src != tgt:
                    neg[gi[tgt], gi[src]] += w

        # exploitative competition: two consumers fight for the same substrate
        # (skip environmental / detoxification metabolites)
        if competition_weight > 0 and met not in _COMPETITION_EXCLUDE:
            cons_list = sorted(cons)
            for ii, gi_a in enumerate(cons_list):
                for gi_b in cons_list[ii + 1:]:
                    neg[gi[gi_a], gi[gi_b]] += w * competition_weight
                    neg[gi[gi_b], gi[gi_a]] += w * competition_weight

    net = pos - neg
    if verbose:
        n_dir = int((net != 0).sum() - np.count_nonzero(np.diag(net)))
        net_sym = (net + net.T) / 2
        n_und = int(((net_sym != 0).sum() - np.count_nonzero(np.diag(net_sym))) // 2)
        print(f'  L1+L2 Szafranski: {n_dir} directed pairs ({n_und} undirected)')

    # ── L3: AGORA2 FBA cross-feeding ─────────────────────────────────────────
    agora_dir = _here / 'data' / 'homd_db' / 'agora_gems'
    if use_agora and agora_dir.exists():
        try:
            W_AGORA = agora_weight

            if agora_medium == 'micom':
                # ── MICOM community FBA path ───────────────────────────────
                # Cross-feeding signals from actual community fluxes.
                # More realistic than single-species pFBA because each guild's
                # secretion profile is resolved in the presence of competitors.
                from guild_agora_signs import compute_micom_signals, ORAL_MEDIUM
                pos_cf, neg_tox, present_m = compute_micom_signals(
                    agora_dir, medium_dict=ORAL_MEDIUM,
                    fraction=micom_fraction, verbose=verbose)
                for guild_i in present_m:
                    if guild_i not in gi:
                        continue
                    for guild_j in present_m:
                        if guild_j not in gi or guild_i == guild_j:
                            continue
                        pos[gi[guild_i], gi[guild_j]] += W_AGORA * pos_cf[
                            GUILD_ORDER.index(guild_i), GUILD_ORDER.index(guild_j)]
                        neg[gi[guild_i], gi[guild_j]] += W_AGORA * neg_tox[
                            GUILD_ORDER.index(guild_i), GUILD_ORDER.index(guild_j)]

            else:
                # ── Single-species pFBA path (v1 or v2) ───────────────────
                import cobra
                from cobra.flux_analysis import pfba
                from guild_agora_signs import (ORAL_MEDIUM, ORAL_MEDIUM_V2,
                                                ANAEROBIC_GUILDS, apply_medium,
                                                GUILD_REPS, find_model_path)

                medium_dict = ORAL_MEDIUM_V2 if agora_medium == 'v2' else ORAL_MEDIUM
                THRESHOLD = 0.05
                TOXINS    = {'EX_h2o2(e)', 'EX_h2s(e)'}

                guild_models = {}
                for guild in GUILD_ORDER:
                    if guild not in gi:
                        continue
                    path = find_model_path(agora_dir, GUILD_REPS.get(guild, [guild]))
                    if path is None:
                        continue
                    guild_models[guild] = cobra.io.read_sbml_model(str(path))

                if verbose:
                    print(f'  L3 AGORA2: loaded {len(guild_models)} guild models '
                          f'(medium={agora_medium}, pFBA)')

                secretions, uptakes = {}, {}
                for guild, model in guild_models.items():
                    apply_medium(model, guild, medium_dict=medium_dict)
                    try:
                        sol = pfba(model)
                    except Exception:
                        continue
                    if sol.objective_value < 1e-6:
                        continue
                    sec, upt = {}, {}
                    for rxn in model.exchanges:
                        f = sol.fluxes.get(rxn.id, 0.0)
                        if f >  THRESHOLD:
                            sec[rxn.id] = f
                        elif f < -THRESHOLD:
                            upt[rxn.id] = abs(f)
                    secretions[guild] = sec
                    uptakes[guild]    = upt
                    if verbose:
                        print(f'    {guild}: μ={sol.objective_value:.2f}  '
                              f'sec={len(sec)}  upt={len(upt)}')

                # Cross-feeding and toxin signals
                cf_pairs = 0
                for j, sec_j in secretions.items():
                    for ex_id in sec_j:
                        for i, upt_i in uptakes.items():
                            if i == j or ex_id not in upt_i:
                                continue
                            if ex_id in TOXINS:
                                neg[gi[i], gi[j]] += W_AGORA
                            else:
                                pos[gi[i], gi[j]] += W_AGORA
                            cf_pairs += 1
                if verbose:
                    print(f'  L3 AGORA2 {agora_medium}: {cf_pairs} cross-feeding signals')

                # Competition via growth-rate suppression (v2 only, experimental)
                if agora_medium == 'v2' and agora_comp_weight > 0:
                    from guild_agora_signs import compute_growth_suppression
                    comp_mat, present_grs = compute_growth_suppression(
                        agora_dir, medium_dict=medium_dict, verbose=verbose)
                    comp_pairs = 0
                    for guild_i in present_grs:
                        if guild_i not in gi:
                            continue
                        for guild_j in present_grs:
                            if guild_j not in gi or guild_i == guild_j:
                                continue
                            v = comp_mat[GUILD_ORDER.index(guild_i),
                                         GUILD_ORDER.index(guild_j)]
                            if v > 0:
                                neg[gi[guild_i], gi[guild_j]] += agora_comp_weight * v
                                comp_pairs += 1
                    if verbose:
                        print(f'  L3 AGORA2 v2 GRS: {comp_pairs} competition pairs')

            net = pos - neg
            if verbose:
                n_dir = int((net != 0).sum() - np.count_nonzero(np.diag(net)))
                print(f'  L3 AGORA2: → {n_dir} directed pairs total')

        except ImportError:
            if verbose:
                print('  L3 AGORA2: cobra not available, skipping')
        except Exception as e:
            if verbose:
                print(f'  L3 AGORA2: skipped ({e})')

    net = pos - neg
    if symmetrize:
        net = (net + net.T) / 2
    return net


def net_flow_hamilton(use_agora=True, competition_weight=0.5,
                      agora_medium='v1', agora_comp_weight=0.5,
                      micom_fraction=0.5, **kwargs):
    """Sign-prior matrix for the Hamilton model (symmetric A).

    Returns a symmetric (N_G × N_G) matrix. Amensalism signals (+/-)
    from opposite directions are averaged and cancel to 0 (unconstrained),
    which is the correct behaviour when A[i,j] = A[j,i] is enforced.
    """
    return build_net_flow_expanded(
        use_agora=use_agora,
        competition_weight=competition_weight,
        symmetrize=True,
        agora_medium=agora_medium,
        agora_comp_weight=agora_comp_weight,
        micom_fraction=micom_fraction,
        **kwargs,
    )


def net_flow_glv(use_agora=True, competition_weight=0.5,
                 agora_medium='v1', agora_comp_weight=0.5,
                 micom_fraction=0.5, **kwargs):
    """Sign-prior matrix for the gLV model (full asymmetric A).

    Returns a directed (N_G × N_G) matrix. net[i,j] encodes the sign
    constraint on A[i,j] independently of A[j,i], preserving amensalism
    (+/-) interactions that are biologically meaningful in gLV dynamics.
    """
    return build_net_flow_expanded(
        use_agora=use_agora,
        competition_weight=competition_weight,
        symmetrize=False,
        agora_medium=agora_medium,
        agora_comp_weight=agora_comp_weight,
        micom_fraction=micom_fraction,
        **kwargs,
    )


if __name__ == '__main__':
    from guild_replicator_dieckow import GUILD_SHORT_LIST
    import json

    print('=== Original (loo_cv_kegg_prior.build_net_flow) ===')
    from loo_cv_kegg_prior import build_net_flow
    nf_orig = build_net_flow()
    ns_orig = (nf_orig + nf_orig.T) / 2
    n_orig  = int(((ns_orig != 0).sum() - np.count_nonzero(np.diag(ns_orig))) // 2)
    print(f'  Undirected constrained pairs: {n_orig}')

    print('\n=== Expanded (this file) ===')
    nf_exp = build_net_flow_expanded(use_agora=True, verbose=True)
    ns_exp = (nf_exp + nf_exp.T) / 2
    n_exp  = int(((ns_exp != 0).sum() - np.count_nonzero(np.diag(ns_exp))) // 2)
    print(f'  Undirected constrained pairs: {n_exp}  (was {n_orig})')

    GS = GUILD_SHORT_LIST
    print('\n  New pairs vs original:')
    for i in range(N_G):
        for j in range(i+1, N_G):
            if ns_exp[i,j] != 0 and ns_orig[i,j] == 0:
                sign = '+' if ns_exp[i,j] > 0 else '-'
                print(f'    NEW  {GS[i]} ↔ {GS[j]}  flow={ns_exp[i,j]:.2f} ({sign})')
            elif ns_exp[i,j] != ns_orig[i,j] and ns_orig[i,j] != 0:
                print(f'    CHG  {GS[i]} ↔ {GS[j]}  {ns_orig[i,j]:.2f} → {ns_exp[i,j]:.2f}')
