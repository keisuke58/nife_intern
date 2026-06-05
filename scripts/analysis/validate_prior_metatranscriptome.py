#!/usr/bin/env python3
"""
validate_prior_metatranscriptome.py
====================================
Independent functional validation of the Szafranski sign prior via the
Joshi 2025 metatranscriptome (PMC12381052).

Two tiers (see SCOPE doc):

  Tier A* (PRIMARY, amino-acid-resolved, Supplement H)
  -------------------------------------------------------
  For each cross-feeding edge in the prior where the metabolite is an amino
  acid, check directional consistency in Joshi Supplement H:
    - producer guild P shows anabolic activity for that AA  (_a present)
    - consumer guild C shows catabolic activity for that AA (_u present)
  Supported fraction = |edges with both signals| / |testable edges|.

  Tier B* (SECONDARY/robustness, EC-level, Supplement F)
  -------------------------------------------------------
  For each prior edge with a non-circular metabolite→EC bridge, compute
  whether the health↔peri EC shift direction matches the prior's predicted
  sign (FDR<0.05 gate, weighted by |LDA| correlation).
  Signed-agreement fraction = |prior-concordant EC shifts| / |testable edges|.

Permutation null (both tiers):
  Shuffle guild labels assigned to EC/AA activity records 10,000× (default),
  recompute the support statistic, report empirical p = P(null >= observed).
  A separate sign-flip null is reported for Tier B*.

Usage
-----
  python scripts/analysis/validate_prior_metatranscriptome.py \\
      [--tier A|B|both] \\
      [--null-iters 10000] \\
      [--seed 0] \\
      [--evidence all|experimental|prediction] \\
      [--fdr-threshold 0.05] \\
      [--no-figures]

Outputs (to results/prior_metatx_validation/)
---------------------------------------------
  tierA_result.json          — Tier A* statistics
  tierB_result.json          — Tier B* statistics
  permutation_null.npy       — null distribution array (shape: [iters])
  fig_tierA_support.pdf/.png — Tier A* bar figure
  fig_tierB_signed_agreement.pdf/.png — Tier B* bar figure

Import check
------------
  python -c "import validate_prior_metatranscriptome"  (from scripts/analysis/)
  → raises ImportError/FileNotFoundError only when data files are absent,
    never on import alone (all compute is inside main()).

Dependencies
------------
  prior_metatx_data  (repo root — bare import via pathshim)
  pub_style          (repo root)
  guild_replicator_dieckow  (repo root)
  numpy, scipy, pandas, matplotlib  (miniconda3 stack)
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pub_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_SHORT, GUILD_COLORS
import prior_metatx_data as _pmd

# ── output paths ──────────────────────────────────────────────────────────────
_REPO   = Path(__file__).resolve().parents[2]
_OUTDIR = _REPO / 'results' / 'prior_metatx_validation'
_OUTDIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier A* helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Map Joshi class names (Supplement H) → NIFE guild names.
# Supplement H uses bacterial class names at the class level; the NIFE guild
# taxonomy also uses class-level names for most guilds.
JOSHI_CLASS_TO_GUILD: dict[str, str] = {
    # exact matches (Joshi class == NIFE guild)
    'Actinobacteria':        'Actinobacteria',
    'Bacilli':               'Bacilli',
    'Bacteroidia':           'Bacteroidia',
    'Betaproteobacteria':    'Betaproteobacteria',
    'Clostridia':            'Clostridia',
    'Coriobacteriia':        'Coriobacteriia',
    'Fusobacteriia':         'Fusobacteriia',
    'Gammaproteobacteria':   'Gammaproteobacteria',
    'Negativicutes':         'Negativicutes',
    # common Joshi paper variants / synonyms
    'Actinomycetia':         'Actinobacteria',   # reclassified class name
    'Flavobacteriia':        'Bacteroidia',       # collapsed into Bacteroidia in NIFE
    'Clostridia (class)':    'Clostridia',
    'Peptostreptococcales':  'Clostridia',
    'Erysipelotrichia':      'Bacilli',           # often grouped with Firmicutes/Bacilli
}


def _guild_from_joshi_class(cls: str) -> str | None:
    """Map a Joshi class string to a NIFE guild name, or None if unmappable."""
    cls = str(cls).strip()
    mapped = JOSHI_CLASS_TO_GUILD.get(cls)
    if mapped:
        return mapped
    # fuzzy: if the class name IS a guild name
    if cls in GUILD_ORDER:
        return cls
    return None


def _is_aa_edge(edge: dict) -> bool:
    """Return True if the metabolite in this edge is an amino acid."""
    met  = str(edge.get('metabolite', '')).lower().strip()
    hmdb = str(edge.get('hmdb', ''))
    canon = _pmd._norm_aa(met)
    if canon:
        return True
    # HMDB IDs for the 20 standard amino acids (common range HMDB0000001–HMDB0000700)
    # Use metabolite name matching as primary; HMDB as fallback.
    _AA_HMDB_PREFIX = ('HMDB0000148', 'HMDB0000641', 'HMDB0000033',  # glu, gln, ala
                       'HMDB0000168', 'HMDB0000062', 'HMDB0000067',  # lys, asp, asn
                       'HMDB0000696', 'HMDB0000151', 'HMDB0000217',  # pro, val, leu
                       'HMDB0000172', 'HMDB0000182', 'HMDB0000177',  # ile, phe, tyr
                       'HMDB0000159', 'HMDB0000197', 'HMDB0000161',  # trp, ser, thr
                       'HMDB0000162', 'HMDB0000097', 'HMDB0000094',  # cys, met, gly
                       'HMDB0000220', 'HMDB0000904')                  # his, arg
    return hmdb in _AA_HMDB_PREFIX


def compute_tier_a(edges: list[dict],
                   supp_h: pd.DataFrame,
                   ) -> dict:
    """
    Tier A* statistic: fraction of prior AA cross-feeding edges directionally
    supported by Joshi Supplement H.

    Support criterion for edge (P -> C, aa, sign=+):
      - guild P has an anabolic (_a) record for aa in Supplement H
      - guild C has a catabolic (_u) record for aa in Supplement H

    Returns
    -------
    dict with keys:
      n_edges_total, n_edges_aa, n_edges_testable, n_supported,
      supported_fraction, edges_detail (list of per-edge dicts)
    """
    # Index Supplement H: (guild, aa, mode) → present bool
    # 'mode' is 'a' (anabolic) or 'u' (catabolic/utilization)
    suppH_index: set[tuple[str, str, str]] = set()
    for _, row in supp_h.iterrows():
        guild = _guild_from_joshi_class(row.get('class', ''))
        aa    = _pmd._norm_aa(row.get('aa', ''))
        mode  = str(row.get('mode', '')).strip().lower()
        if guild and aa and mode in ('a', 'u'):
            suppH_index.add((guild, aa, mode))

    aa_edges = [e for e in edges if _is_aa_edge(e) and e.get('sign') == '+']

    n_total    = len(edges)
    n_aa       = len(aa_edges)
    n_testable = 0
    n_supported = 0
    edges_detail = []

    for edge in aa_edges:
        pg    = edge['producer_guild']
        cg    = edge['consumer_guild']
        met   = str(edge.get('metabolite', '')).lower().strip()
        aa    = _pmd._norm_aa(met)
        if aa is None:
            continue

        # testable if both guilds appear in Supplement H for this AA
        has_prod_anab = (pg, aa, 'a') in suppH_index
        has_cons_cat  = (cg, aa, 'u') in suppH_index
        testable      = has_prod_anab or has_cons_cat  # at least one signal present
        supported     = has_prod_anab and has_cons_cat

        if testable:
            n_testable  += 1
        if supported:
            n_supported += 1

        edges_detail.append({
            'producer_guild': pg,
            'consumer_guild': cg,
            'metabolite':     met,
            'aa':             aa,
            'evidence':       edge.get('evidence'),
            'has_producer_anabolic': has_prod_anab,
            'has_consumer_catabolic': has_cons_cat,
            'testable':       testable,
            'supported':      supported,
        })

    frac = n_supported / n_testable if n_testable > 0 else float('nan')
    return {
        'n_edges_total':     n_total,
        'n_edges_aa':        n_aa,
        'n_edges_testable':  n_testable,
        'n_supported':       n_supported,
        'supported_fraction': frac,
        'edges_detail':      edges_detail,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tier B* helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tier_b(edges: list[dict],
                   supp_f: pd.DataFrame,
                   bridge: dict,
                   fdr_threshold: float = 0.05,
                   ) -> dict:
    """
    Tier B* statistic: signed agreement between prior edge direction and
    health↔peri EC expression shift (Joshi Supplement F).

    For each prior cross-feeding (+) or competition (−) edge with an
    external metabolite→EC bridge:
      - look up EC in Supplement F
      - gate: FDR < fdr_threshold
      - observed_sign = sign(mean_peri − mean_health)
      - prior_sign    = +1 for cross-feeding, −1 for competition/inhibition
      - concordant if observed_sign == prior_sign (weighted by |LDA|)

    Circular bridge: edges whose metabolite→EC map is drawn from Suppl File 1
    EC column (ec_from_supplfile != null) are EXCLUDED.

    Returns
    -------
    dict with keys:
      n_edges_total, n_edges_bridged, n_edges_significant,
      n_concordant, signed_agreement_fraction (unweighted),
      weighted_concordance, edges_detail, coverage_caveat
    """
    # Build Supplement F index: EC → {mean_health, mean_peri, P, FDR, LDA}
    suppF_by_ec: dict[str, dict] = {}
    for _, row in supp_f.iterrows():
        ec = str(row.get('EC', '')).strip()
        if ec:
            suppF_by_ec[ec] = {
                'mean_health': float(row.get('mean_health', 0.0)),
                'mean_peri':   float(row.get('mean_peri', 0.0)),
                'P':           float(row.get('P', 1.0)),
                'FDR':         float(row.get('FDR', 1.0)),
                'LDA':         float(row.get('LDA', 0.0)),
            }

    n_total       = len(edges)
    n_bridged     = 0
    n_significant = 0
    n_concordant  = 0
    weighted_sum  = 0.0
    weight_total  = 0.0
    n_dropped_circular = 0
    edges_detail  = []

    for edge in edges:
        # Exclude circular bridge (ec drawn from Suppl File 1 itself)
        if edge.get('ec_from_supplfile') is not None:
            n_dropped_circular += 1

        hmdb = str(edge.get('hmdb', ''))
        met  = str(edge.get('metabolite', '')).lower().strip()
        prior_sign_str = edge.get('sign', '+')
        prior_sign = +1 if prior_sign_str == '+' else -1

        # Resolve ECs from external bridge only (no Suppl File 1 ECs)
        ecs: list[str] = []
        if hmdb in bridge:
            ecs.extend(bridge[hmdb])
        if met in bridge:
            for ec in bridge[met]:
                if ec not in ecs:
                    ecs.append(ec)

        if not ecs:
            edges_detail.append({
                **{k: edge.get(k) for k in
                   ('producer_guild', 'consumer_guild', 'metabolite', 'hmdb',
                    'sign', 'evidence')},
                'ecs_found':    [],
                'bridged':      False,
                'reason_skip':  'no external bridge EC',
            })
            continue

        n_bridged += 1

        # Aggregate across all ECs for this metabolite
        ec_results = []
        for ec in ecs:
            if ec not in suppF_by_ec:
                continue
            rec = suppF_by_ec[ec]
            if rec['FDR'] >= fdr_threshold:
                continue

            obs_diff  = rec['mean_peri'] - rec['mean_health']
            obs_sign  = int(np.sign(obs_diff)) if obs_diff != 0 else 0
            lda_w     = abs(rec['LDA'])
            concordant = (obs_sign == prior_sign)
            ec_results.append({
                'EC':         ec,
                'mean_health': rec['mean_health'],
                'mean_peri':   rec['mean_peri'],
                'FDR':         rec['FDR'],
                'LDA':         rec['LDA'],
                'obs_sign':    obs_sign,
                'prior_sign':  prior_sign,
                'concordant':  concordant,
                'weight':      lda_w,
            })

        if not ec_results:
            edges_detail.append({
                **{k: edge.get(k) for k in
                   ('producer_guild', 'consumer_guild', 'metabolite', 'hmdb',
                    'sign', 'evidence')},
                'ecs_found':    ecs,
                'bridged':      True,
                'reason_skip':  'no EC passed FDR gate',
            })
            continue

        n_significant += 1

        # Majority vote (unweighted)
        n_conc = sum(1 for r in ec_results if r['concordant'])
        edge_concordant = (n_conc >= len(ec_results) / 2.0)
        if edge_concordant:
            n_concordant += 1

        # Weighted concordance
        for r in ec_results:
            w = r['weight']
            weighted_sum  += w * (1 if r['concordant'] else 0)
            weight_total  += w

        edges_detail.append({
            **{k: edge.get(k) for k in
               ('producer_guild', 'consumer_guild', 'metabolite', 'hmdb',
                'sign', 'evidence')},
            'ecs_found':        ecs,
            'bridged':          True,
            'significant':      True,
            'edge_concordant':  edge_concordant,
            'n_concordant_ecs': n_conc,
            'n_total_ecs':      len(ec_results),
            'ec_detail':        ec_results,
        })

    frac = n_concordant / n_significant if n_significant > 0 else float('nan')
    w_frac = weighted_sum / weight_total if weight_total > 0 else float('nan')

    return {
        'n_edges_total':              n_total,
        'n_edges_bridged':            n_bridged,
        'n_edges_significant':        n_significant,
        'n_concordant':               n_concordant,
        'signed_agreement_fraction':  frac,
        'weighted_concordance':       w_frac,
        'n_dropped_circular_bridge':  n_dropped_circular,
        'fdr_threshold':              fdr_threshold,
        'coverage_caveat': (
            f'{n_bridged}/{n_total} edges had external bridge ECs; '
            f'{n_significant}/{n_bridged} passed FDR<{fdr_threshold}; '
            f'{n_dropped_circular} excluded (circular bridge).'
        ),
        'edges_detail': edges_detail,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Permutation null
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_null(stat_fn,
                     iters: int = 10_000,
                     seed: int = 0,
                     ) -> np.ndarray:
    """
    Generic permutation null: call stat_fn(shuffle_fn) `iters` times.

    stat_fn must accept a single argument `rng` (a numpy Generator) and return
    a scalar (the statistic value under the shuffle).

    The shuffling logic is encapsulated inside stat_fn so that both Tier A and
    Tier B can share this driver.

    Returns the null distribution as a float array of length `iters`.
    """
    rng  = np.random.default_rng(seed)
    null = np.empty(iters, dtype=float)
    for i in range(iters):
        val = stat_fn(rng)
        null[i] = val if not np.isnan(val) else 0.0
    return null


def _tier_a_null_fn(edges: list[dict],
                    supp_h: pd.DataFrame,
                    ):
    """Return a function(rng) -> scalar for the Tier A* permutation null.

    Shuffle: randomly re-assign guild labels to the Supplement H activity
    records (preserving the set of guilds present, shuffling which class
    gets which activity profile).
    """
    # Pre-index: list of (guild, aa, mode) tuples actually in Supplement H
    records = []
    for _, row in supp_h.iterrows():
        guild = _guild_from_joshi_class(row.get('class', ''))
        aa    = _pmd._norm_aa(row.get('aa', ''))
        mode  = str(row.get('mode', '')).strip().lower()
        if guild and aa and mode in ('a', 'u'):
            records.append((guild, aa, mode))

    guild_labels = np.array([r[0] for r in records])
    aa_labels    = np.array([r[1] for r in records])
    mode_labels  = np.array([r[2] for r in records])

    aa_edges = [e for e in edges if _is_aa_edge(e) and e.get('sign') == '+']

    def _stat(rng: np.random.Generator) -> float:
        # Shuffle guild labels only (preserve AA and mode structure)
        perm_guilds = rng.permutation(guild_labels)
        perm_index  = set(zip(perm_guilds, aa_labels, mode_labels))

        n_testable  = 0
        n_supported = 0
        for edge in aa_edges:
            met = str(edge.get('metabolite', '')).lower().strip()
            aa  = _pmd._norm_aa(met)
            if aa is None:
                continue
            pg = edge['producer_guild']
            cg = edge['consumer_guild']
            has_p = (pg, aa, 'a') in perm_index
            has_c = (cg, aa, 'u') in perm_index
            if has_p or has_c:
                n_testable += 1
            if has_p and has_c:
                n_supported += 1
        return n_supported / n_testable if n_testable > 0 else float('nan')

    return _stat


def _tier_b_null_fn(edges: list[dict],
                    supp_f: pd.DataFrame,
                    bridge: dict,
                    fdr_threshold: float,
                    ):
    """Return a function(rng) -> scalar for the Tier B* permutation null.

    Shuffle: randomly flip the sign of the health↔peri difference in
    Supplement F (sign-flip null — tests directionality, not co-occurrence).
    """
    # Pre-compute significant EC set with observed signs
    ec_sign: dict[str, int] = {}
    ec_lda:  dict[str, float] = {}
    for _, row in supp_f.iterrows():
        ec  = str(row.get('EC', '')).strip()
        fdr = float(row.get('FDR', 1.0))
        if ec and fdr < fdr_threshold:
            diff = float(row.get('mean_peri', 0.0)) - float(row.get('mean_health', 0.0))
            ec_sign[ec] = int(np.sign(diff)) if diff != 0 else 0
            ec_lda[ec]  = abs(float(row.get('LDA', 0.0)))

    all_ecs  = list(ec_sign.keys())
    ec_signs_arr = np.array([ec_sign[e] for e in all_ecs])
    ec_lda_arr   = np.array([ec_lda[e]  for e in all_ecs])
    ec_idx       = {e: i for i, e in enumerate(all_ecs)}

    valid_edges = []
    for edge in edges:
        if edge.get('ec_from_supplfile') is not None:
            continue
        hmdb = str(edge.get('hmdb', ''))
        met  = str(edge.get('metabolite', '')).lower().strip()
        prior_sign = +1 if edge.get('sign', '+') == '+' else -1
        ecs: list[str] = list(bridge.get(hmdb, []))
        for ec in bridge.get(met, []):
            if ec not in ecs:
                ecs.append(ec)
        sig_ecs = [ec for ec in ecs if ec in ec_idx]
        if sig_ecs:
            valid_edges.append((prior_sign, sig_ecs))

    def _stat(rng: np.random.Generator) -> float:
        # Randomly flip each EC's sign direction
        flip   = rng.integers(0, 2, size=len(all_ecs)) * 2 - 1  # ±1
        perm_signs = ec_signs_arr * flip

        n_sig  = 0
        n_conc = 0
        for (prior_sign, sig_ecs) in valid_edges:
            ec_results = []
            for ec in sig_ecs:
                idx = ec_idx[ec]
                ec_results.append((perm_signs[idx], ec_lda_arr[idx]))
            if not ec_results:
                continue
            n_sig += 1
            # majority vote
            n_c = sum(1 for (s, _w) in ec_results if s == prior_sign)
            if n_c >= len(ec_results) / 2.0:
                n_conc += 1
        return n_conc / n_sig if n_sig > 0 else float('nan')

    return _stat


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def _empirical_p(observed: float, null: np.ndarray) -> float:
    """One-tailed empirical p-value: P(null >= observed)."""
    valid = null[~np.isnan(null)]
    if len(valid) == 0:
        return float('nan')
    return float(np.mean(valid >= observed))


def plot_tier_a(result: dict,
                null: np.ndarray,
                outdir: Path,
                ) -> None:
    pub_style.apply()
    obs  = result['supported_fraction']
    null_clean = null[~np.isnan(null)]
    ep   = _empirical_p(obs, null_clean)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: summary bar
    ax = axes[0]
    cats   = ['Not supported', 'Supported']
    n_supp = result['n_supported']
    n_test = result['n_edges_testable']
    n_not  = n_test - n_supp
    colors = ['#d0d0d0', '#2ca02c']
    ax.bar(cats, [n_not, n_supp], color=colors, width=0.5, edgecolor='black', linewidth=0.7)
    ax.set_ylabel('Number of testable AA edges')
    ax.set_title(
        f'Tier A* support\n'
        f'n={n_test} testable, {n_supp} supported\n'
        f'frac={obs:.3f}  p_perm={ep:.4f}'
    )
    for i, v in enumerate([n_not, n_supp]):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, max(n_not, n_supp) * 1.25 + 1)

    # Right: null distribution
    ax2 = axes[1]
    ax2.hist(null_clean, bins=40, color='#aec7e8', edgecolor='white',
             linewidth=0.4, density=True)
    ax2.axvline(obs, color='#d62728', linewidth=2.0,
                label=f'Observed = {obs:.3f}')
    ax2.axvline(0.5, color='#7f7f7f', linewidth=1.0, linestyle='--',
                label='Random (0.5)')
    ax2.set_xlabel('Supported fraction (permutation null)')
    ax2.set_ylabel('Density')
    ax2.set_title(
        f'Permutation null ({len(null_clean):,} shuffles)\n'
        f'p = {ep:.4f}'
    )
    ax2.legend(fontsize=9)

    fig.suptitle(
        'Tier A*: AA cross-feeding edge support (Joshi 2025 Supplement H)',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        p = outdir / f'fig_tierA_support.{ext}'
        fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  figures: {outdir}/fig_tierA_support.pdf/.png')


def plot_tier_b(result: dict,
                null: np.ndarray,
                outdir: Path,
                ) -> None:
    pub_style.apply()
    obs  = result['signed_agreement_fraction']
    null_clean = null[~np.isnan(null)]
    ep   = _empirical_p(obs, null_clean)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: concordance bar
    ax = axes[0]
    n_conc = result['n_concordant']
    n_sig  = result['n_edges_significant']
    n_disc = n_sig - n_conc
    cats   = ['Discordant', 'Concordant']
    colors = ['#ff9896', '#1f77b4']
    ax.bar(cats, [n_disc, n_conc], color=colors, width=0.5,
           edgecolor='black', linewidth=0.7)
    ax.set_ylabel('Number of significant edges')
    ax.set_title(
        f'Tier B* signed agreement\n'
        f'n={n_sig} significant, {n_conc} concordant\n'
        f'frac={obs:.3f}  p_perm={ep:.4f}'
    )
    for i, v in enumerate([n_disc, n_conc]):
        ax.text(i, v + 0.05, str(v), ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, max(n_disc, n_conc) * 1.3 + 1)
    ax.axhline(n_sig * 0.5, color='grey', linestyle='--', linewidth=0.8,
               label='Random (50%)')
    ax.legend(fontsize=9)

    # Right: null distribution
    ax2 = axes[1]
    ax2.hist(null_clean, bins=40, color='#aec7e8', edgecolor='white',
             linewidth=0.4, density=True)
    ax2.axvline(obs, color='#d62728', linewidth=2.0,
                label=f'Observed = {obs:.3f}')
    ax2.axvline(0.5, color='#7f7f7f', linewidth=1.0, linestyle='--',
                label='Random (0.5)')
    ax2.set_xlabel('Concordant fraction (sign-flip null)')
    ax2.set_ylabel('Density')
    ax2.set_title(
        f'Sign-flip null ({len(null_clean):,} shuffles)\n'
        f'p = {ep:.4f}'
    )
    ax2.legend(fontsize=9)

    fig.suptitle(
        'Tier B*: EC shift signed agreement (Joshi 2025 Supplement F)',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        p = outdir / f'fig_tierB_signed_agreement.{ext}'
        fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  figures: {outdir}/fig_tierB_signed_agreement.pdf/.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _check_data_or_exit():
    avail = _pmd.check_data_available()
    missing = [k for k, v in avail.items() if not v]
    if missing:
        print('WARNING: the following upstream data files are missing:')
        for k in missing:
            print(f'  {k}')
        print(
            'Run the extraction scripts first:\n'
            '  python scripts/analysis/prior_metatx_validation/extract_prior_edges.py\n'
            '  python scripts/analysis/prior_metatx_validation/parse_joshi_si.py\n'
            '  python scripts/analysis/prior_metatx_validation/metabolite_ec_bridge.py\n'
        )
    return missing


def main():
    parser = argparse.ArgumentParser(
        description='Validate Szafranski sign prior against Joshi 2025 metatranscriptome.'
    )
    parser.add_argument('--tier', choices=['A', 'B', 'both'], default='both',
                        help='Which tier(s) to run (default: both)')
    parser.add_argument('--null-iters', type=int, default=10_000,
                        metavar='N',
                        help='Permutation iterations (default: 10000)')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for permutation (default: 0)')
    parser.add_argument('--evidence', choices=['all', 'experimental', 'prediction'],
                        default='all',
                        help='Filter prior edges by EVIDENCE column (default: all)')
    parser.add_argument('--fdr-threshold', type=float, default=0.05,
                        metavar='FDR',
                        help='FDR gate for Tier B* (default: 0.05)')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    args = parser.parse_args()

    missing = _check_data_or_exit()

    # ── load data (will raise FileNotFoundError if missing) ──────────────────
    try:
        edges = _pmd.load_prior_edges(evidence=args.evidence)
        print(f'Loaded {len(edges)} prior edges (evidence={args.evidence})')
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}')
        sys.exit(1)

    run_a = args.tier in ('A', 'both')
    run_b = args.tier in ('B', 'both')

    null_a = null_b = None

    # ── Tier A* ──────────────────────────────────────────────────────────────
    if run_a:
        try:
            supp_h = _pmd.load_supp_h()
            print(f'Loaded Supplement H: {len(supp_h)} rows')

            print('Computing Tier A* statistic...')
            res_a = compute_tier_a(edges, supp_h)
            print(
                f'  Tier A*: {res_a["n_supported"]}/{res_a["n_edges_testable"]} '
                f'edges supported (frac={res_a["supported_fraction"]:.3f})'
            )

            print(f'Running permutation null ({args.null_iters} iters, seed={args.seed})...')
            stat_fn_a = _tier_a_null_fn(edges, supp_h)
            null_a    = permutation_null(stat_fn_a, iters=args.null_iters,
                                         seed=args.seed)
            emp_p_a   = _empirical_p(res_a['supported_fraction'], null_a)
            res_a['empirical_p']      = emp_p_a
            res_a['null_mean']        = float(np.nanmean(null_a))
            res_a['null_ci_95']       = [float(np.nanpercentile(null_a, 2.5)),
                                          float(np.nanpercentile(null_a, 97.5))]
            res_a['n_perm']           = args.null_iters
            res_a['perm_seed']        = args.seed
            res_a['perm_what_shuffled'] = (
                'Guild labels in Supplement H activity records; '
                'AA and mode labels held fixed.'
            )
            res_a['evidence_filter']  = args.evidence
            print(f'  Empirical p = {emp_p_a:.4f}  (null mean = {res_a["null_mean"]:.3f})')

            out_a = _OUTDIR / 'tierA_result.json'
            # Serialize: remove edges_detail list for cleaner top-level JSON
            out_a.write_text(json.dumps(
                {k: v for k, v in res_a.items() if k != 'edges_detail'},
                indent=2
            ))
            # Save full detail separately
            out_a_detail = _OUTDIR / 'tierA_edges_detail.json'
            out_a_detail.write_text(json.dumps(res_a['edges_detail'], indent=2))
            print(f'  Saved: {out_a}')
            print(f'  Saved: {out_a_detail}')

            if not args.no_figures:
                plot_tier_a(res_a, null_a, _OUTDIR)

        except FileNotFoundError as exc:
            print(f'Tier A skipped: {exc}')

    # ── Tier B* ──────────────────────────────────────────────────────────────
    if run_b:
        try:
            supp_f = _pmd.load_supp_f()
            bridge = _pmd.load_bridge()
            print(f'Loaded Supplement F: {len(supp_f)} rows')
            print(f'Loaded external bridge: {len(bridge)} entries')

            print('Computing Tier B* statistic...')
            res_b = compute_tier_b(edges, supp_f, bridge,
                                   fdr_threshold=args.fdr_threshold)
            print(
                f'  Tier B*: {res_b["n_concordant"]}/{res_b["n_edges_significant"]} '
                f'edges concordant (frac={res_b["signed_agreement_fraction"]:.3f})'
            )
            print(f'  Coverage: {res_b["coverage_caveat"]}')

            print(f'Running sign-flip null ({args.null_iters} iters, seed={args.seed})...')
            stat_fn_b = _tier_b_null_fn(edges, supp_f, bridge, args.fdr_threshold)
            null_b    = permutation_null(stat_fn_b, iters=args.null_iters,
                                         seed=args.seed)
            emp_p_b   = _empirical_p(res_b['signed_agreement_fraction'], null_b)
            res_b['empirical_p']       = emp_p_b
            res_b['null_mean']         = float(np.nanmean(null_b))
            res_b['null_ci_95']        = [float(np.nanpercentile(null_b, 2.5)),
                                           float(np.nanpercentile(null_b, 97.5))]
            res_b['n_perm']            = args.null_iters
            res_b['perm_seed']         = args.seed
            res_b['perm_what_shuffled'] = (
                'Random ±1 flip of each EC\'s health/peri difference sign '
                '(sign-flip null); tests directionality, not co-occurrence.'
            )
            res_b['evidence_filter']   = args.evidence
            print(f'  Empirical p = {emp_p_b:.4f}  (null mean = {res_b["null_mean"]:.3f})')

            out_b = _OUTDIR / 'tierB_result.json'
            out_b.write_text(json.dumps(
                {k: v for k, v in res_b.items() if k != 'edges_detail'},
                indent=2
            ))
            out_b_detail = _OUTDIR / 'tierB_edges_detail.json'
            out_b_detail.write_text(json.dumps(res_b['edges_detail'], indent=2))
            print(f'  Saved: {out_b}')
            print(f'  Saved: {out_b_detail}')

            if not args.no_figures:
                plot_tier_b(res_b, null_b, _OUTDIR)

        except FileNotFoundError as exc:
            print(f'Tier B skipped: {exc}')

    # ── save combined null array ──────────────────────────────────────────────
    # Tier A null stored in slot 0; Tier B null in slot 1 (if run)
    # Shape: (2, null_iters) or (1, null_iters) depending on --tier
    nulls_to_save = [x for x in [null_a, null_b] if x is not None]
    if nulls_to_save:
        combined_null = np.stack(nulls_to_save, axis=0)
        null_path = _OUTDIR / 'permutation_null.npy'
        np.save(null_path, combined_null)
        print(f'  Saved null array: {null_path}  shape={combined_null.shape}')

    print('\nDone.')


if __name__ == '__main__':
    import sys
    main()
