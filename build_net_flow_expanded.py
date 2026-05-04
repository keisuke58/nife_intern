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

Differences from original build_net_flow() in loo_cv_kegg_prior.py
-------------------------------------------------------------------
- experimental vs prediction evidence weighted separately
- Additional relationship types: RELEASES → PRODUCES, HYDROLYSES/DEGRADES → USES
- Expanded genus→guild mapping (typos + additional genera)
- Optional AGORA FBA layer

Usage
-----
    from build_net_flow_expanded import build_net_flow_expanded
    net_flow = build_net_flow_expanded(use_agora=True, verbose=True)
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


def build_net_flow_expanded(use_agora=True, verbose=False):
    """Multi-source flow matrix. See module docstring for details."""
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
        for src in prod:
            for tgt in cons:
                if src != tgt:
                    pos[gi[tgt], gi[src]] += w
            for tgt in inhib:
                if src != tgt:
                    neg[gi[tgt], gi[src]] += w

    net = pos - neg
    if verbose:
        n_dir = int((net != 0).sum() - np.count_nonzero(np.diag(net)))
        net_sym = (net + net.T) / 2
        n_und = int(((net_sym != 0).sum() - np.count_nonzero(np.diag(net_sym))) // 2)
        print(f'  L1+L2 Szafranski: {n_dir} directed pairs ({n_und} undirected)')

    # ── L3: AGORA FBA cross-feeding (weight 0.5) ─────────────────────────────
    agora_dir = _here / 'data' / 'homd_db' / 'agora_gems'
    if use_agora and agora_dir.exists():
        try:
            import cobra
            from cobra.flux_analysis import pfba

            W_AGORA   = 0.5
            THRESHOLD = 0.05

            CARBON_CAPS = {
                'EX_glc_D(e)': 10.0, 'EX_fru(e)': 5.0, 'EX_lac_L(e)': 8.0,
                'EX_ala_L(e)': 2.0,  'EX_glu_L(e)': 2.0, 'EX_arg_L(e)': 1.5,
                'EX_pro_L(e)': 1.0,  'EX_nh4(e)': 10.0,  'EX_pi(e)': 10.0,
                'EX_o2(e)': 2.0,     'EX_h2o(e)': 1000.0, 'EX_h(e)': 1000.0,
            }
            BLOCK_KW = ['inulin', 'chtbs', '12dgr', 'acald', '2obut']
            TOXINS   = {'EX_h2o2(e)', 'EX_h2s(e)'}

            guild_models = {}
            for xml in sorted(agora_dir.glob('*.xml')):
                guild = xml.stem.split('_')[0]
                if guild in gi and guild not in guild_models:
                    guild_models[guild] = cobra.io.read_sbml_model(str(xml))

            if verbose:
                print(f'  L3 AGORA: loaded {len(guild_models)} guild models')

            secretions, uptakes = {}, {}
            for guild, model in guild_models.items():
                with model:
                    for rxn in model.exchanges:
                        if rxn.id in CARBON_CAPS:
                            rxn.lower_bound = max(rxn.lower_bound,
                                                   -CARBON_CAPS[rxn.id])
                        if any(kw in rxn.id.lower() for kw in BLOCK_KW):
                            rxn.lower_bound = 0.0
                    try:
                        sol = pfba(model)
                    except Exception:
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

            agora_pairs = 0
            for j, sec_j in secretions.items():
                for ex_id in sec_j:
                    for i, upt_i in uptakes.items():
                        if i == j or ex_id not in upt_i:
                            continue
                        if ex_id in TOXINS:
                            neg[gi[i], gi[j]] += W_AGORA
                        else:
                            pos[gi[i], gi[j]] += W_AGORA
                        agora_pairs += 1

            net = pos - neg
            if verbose:
                n_dir = int((net != 0).sum() - np.count_nonzero(np.diag(net)))
                print(f'  L3 AGORA: +{agora_pairs} signals → total {n_dir} directed pairs')

        except ImportError:
            if verbose:
                print('  L3 AGORA: cobra not available, skipping')
        except Exception as e:
            if verbose:
                print(f'  L3 AGORA: skipped ({e})')

    return pos - neg


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
