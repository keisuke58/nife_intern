#!/usr/bin/env python3
"""
prior_metatx_data.py — shared data loader for the Joshi 2025 metatranscriptome
validation of the Szafranski sign prior.

Consumed by:
  scripts/analysis/validate_prior_metatranscriptome.py  (main CLI)
  scripts/analysis/prior_metatx_validation/run_validation.py

Data sources
------------
Prior edges  : results/prior_metatx_validation/prior_edges.json
               (produced by prior_metatx_validation/extract_prior_edges.py)
Joshi Supp H : results/prior_metatx_validation/joshi_suppH_taxon_aa.csv
               (produced by prior_metatx_validation/parse_joshi_si.py)
Joshi Supp F : results/prior_metatx_validation/joshi_suppF_ec.csv
               (produced by prior_metatx_validation/parse_joshi_si.py)
External EC  : results/prior_metatx_validation/bridge_external.json
               (produced by prior_metatx_validation/metabolite_ec_bridge.py)

This module exposes lazy-loading helpers that raise FileNotFoundError with a
clear message when an upstream script has not yet been run.

No heavy compute here — just I/O + light normalisation.
"""
from pathlib import Path
import json
import pandas as pd

_here = Path(__file__).resolve().parent
_RESULTS = _here / 'results' / 'prior_metatx_validation'

# ── canonical paths ──────────────────────────────────────────────────────────
PRIOR_EDGES_JSON    = _RESULTS / 'prior_edges.json'
SUPP_H_CSV          = _RESULTS / 'joshi_suppH_taxon_aa.csv'
SUPP_F_CSV          = _RESULTS / 'joshi_suppF_ec.csv'
CLASS_ABUND_CSV     = _RESULTS / 'joshi_class_abund.csv'
BRIDGE_JSON         = _RESULTS / 'bridge_external.json'

# ── Amino-acid name normalisation ─────────────────────────────────────────────
# Supplement H uses full names; prior xlsx uses various spellings.
# Map everything to a canonical lower-case short form.
AA_CANONICAL = {
    # full name → short
    'alanine': 'ala', 'arginine': 'arg', 'asparagine': 'asn',
    'aspartate': 'asp', 'aspartic acid': 'asp', 'cysteine': 'cys',
    'glutamate': 'glu', 'glutamic acid': 'glu', 'glutamine': 'gln',
    'glycine': 'gly', 'histidine': 'his', 'isoleucine': 'ile',
    'leucine': 'leu', 'lysine': 'lys', 'methionine': 'met',
    'phenylalanine': 'phe', 'proline': 'pro', 'serine': 'ser',
    'threonine': 'thr', 'tryptophan': 'trp', 'tyrosine': 'tyr',
    'valine': 'val',
    # short already
    'ala': 'ala', 'arg': 'arg', 'asn': 'asn', 'asp': 'asp', 'cys': 'cys',
    'glu': 'glu', 'gln': 'gln', 'gly': 'gly', 'his': 'his', 'ile': 'ile',
    'leu': 'leu', 'lys': 'lys', 'met': 'met', 'phe': 'phe', 'pro': 'pro',
    'ser': 'ser', 'thr': 'thr', 'trp': 'trp', 'tyr': 'tyr', 'val': 'val',
}

def _norm_aa(name: str) -> str | None:
    """Return canonical 3-letter code or None if not an amino-acid."""
    return AA_CANONICAL.get(str(name).strip().lower())


# ── loaders ──────────────────────────────────────────────────────────────────

def load_prior_edges(evidence: str = 'all') -> list[dict]:
    """Load prior edges from JSON.

    Parameters
    ----------
    evidence : {'all', 'experimental', 'prediction'}
        Filter by the EVIDENCE column of Szafranski Supp File 1.

    Returns
    -------
    list of dicts with keys:
        producer_guild, consumer_guild, metabolite, hmdb, relationship,
        sign ('+' or '-'), evidence, ec_from_supplfile (str | null)
    """
    if not PRIOR_EDGES_JSON.exists():
        raise FileNotFoundError(
            f'{PRIOR_EDGES_JSON} not found — run '
            'scripts/analysis/prior_metatx_validation/extract_prior_edges.py first.'
        )
    with open(PRIOR_EDGES_JSON) as fh:
        edges = json.load(fh)
    if evidence != 'all':
        edges = [e for e in edges if e.get('evidence', '').lower() == evidence.lower()]
    return edges


def load_supp_h() -> pd.DataFrame:
    """Load Joshi Supplement H (class × amino-acid × mode).

    Columns: class, aa (canonical 3-letter), mode ('a'=anabolic/'u'=catabolic),
             mean_health, mean_peri, P, FDR
    """
    if not SUPP_H_CSV.exists():
        raise FileNotFoundError(
            f'{SUPP_H_CSV} not found — run '
            'scripts/analysis/prior_metatx_validation/parse_joshi_si.py first.'
        )
    df = pd.read_csv(SUPP_H_CSV)
    if 'aa' in df.columns:
        df['aa'] = df['aa'].apply(lambda x: _norm_aa(x) or str(x).lower().strip())
    return df


def load_supp_f() -> pd.DataFrame:
    """Load Joshi Supplement F (EC-level differential expression).

    Columns: EC, mean_health, mean_peri, P, FDR, LDA
    """
    if not SUPP_F_CSV.exists():
        raise FileNotFoundError(
            f'{SUPP_F_CSV} not found — run '
            'scripts/analysis/prior_metatx_validation/parse_joshi_si.py first.'
        )
    return pd.read_csv(SUPP_F_CSV)


def load_class_abund() -> pd.DataFrame:
    """Load Joshi class-level abundance (Suppl E + G combined)."""
    if not CLASS_ABUND_CSV.exists():
        raise FileNotFoundError(
            f'{CLASS_ABUND_CSV} not found — run '
            'scripts/analysis/prior_metatx_validation/parse_joshi_si.py first.'
        )
    return pd.read_csv(CLASS_ABUND_CSV)


def load_bridge() -> dict:
    """Load external metabolite→EC bridge (KEGG/Rhea, no Suppl File 1 ECs).

    Returns dict keyed by HMDB ID → list of EC strings.
    Also keyed by metabolite name (lower) → list of EC strings.
    """
    if not BRIDGE_JSON.exists():
        raise FileNotFoundError(
            f'{BRIDGE_JSON} not found — run '
            'scripts/analysis/prior_metatx_validation/metabolite_ec_bridge.py first.'
        )
    with open(BRIDGE_JSON) as fh:
        return json.load(fh)


def check_data_available() -> dict[str, bool]:
    """Return availability status of all upstream data files."""
    return {
        'prior_edges':   PRIOR_EDGES_JSON.exists(),
        'supp_h':        SUPP_H_CSV.exists(),
        'supp_f':        SUPP_F_CSV.exists(),
        'class_abund':   CLASS_ABUND_CSV.exists(),
        'bridge':        BRIDGE_JSON.exists(),
    }
