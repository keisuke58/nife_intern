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


def apply_medium(model, guild: str):
    """Close all exchange reactions, then open ORAL_MEDIUM ones."""
    model_rxn_ids = {rxn.id for rxn in model.exchanges}
    medium = {}
    for rxn in model.exchanges:
        rxn_id = rxn.id
        if rxn_id in ORAL_MEDIUM:
            medium[rxn_id] = ORAL_MEDIUM[rxn_id]
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
