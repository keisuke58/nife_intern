#!/usr/bin/env python3
# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
prior_metatx_data.py — DATA LAYER for independent metatranscriptome validation
of the Szafranski sign prior.

Two public functions:
  load_prior_edges()   -> list[dict]   prior cross-feeding / competition edges
  load_joshi_si()      -> dict of pd.DataFrame  (suppF, suppH, suppE_class, suppG)

Both write their outputs to
  results/prior_metatx_validation/data/
on first call and reload from cache on subsequent calls (force_reparse=True to
override).

Scope reference: SCOPE.md (frozen 2026-06-04), section 2 and 3.

Edge schema (prior_edges.json):
  producer_guild  : str | null   (null = ambiguous or no guild match)
  consumer_guild  : str | null
  metabolite      : str          (OBJECT column value)
  hmdb            : str | null
  relationship    : str          (original RELATIONSHIP value)
  sign            : int          (+1 cross-feeding/consumer-benefit, -1 competition/inhibition)
  evidence        : str          ('experimental' | 'prediction')
  ec_from_supplfile: str | null  (EC from Suppl File 1's own EC column; NULL for leakage-free run)

DataFrame schemas:
  suppF_ec.csv    : EC, mean_abund, std_dev, mean_health, mean_peri, p_value, fdr, lda
  suppH_taxon_aa.csv : class_group, group, taxon_class, aa, mode (a/u),
                       mean_abund, mean_health, mean_peri, p_value, fdr
  class_abund_16S.csv: taxon_class, mean_abund, std_dev, mean_health, mean_peri, p_value, fdr
                       (Supplement E — full-16S class level)
  class_abund_tx.csv : taxon_class, mean_abund, std_dev, mean_health, mean_peri, p_value, fdr
                       (Supplement G — metatranscriptome class level)
"""

import json
import re
import warnings

import numpy as np
import pandas as pd

# ── repo layout ──────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]

# Joshi 2025 SI PDF (canonical location; falls back to /tmp/sm1.pdf)
_PDF_CANONICAL = (_REPO / 'Datasets' / 'joshi2025_SI'
                  / '41522_2025_807_MOESM1_ESM.pdf')
_PDF_FALLBACK  = Path('/tmp/sm1.pdf')

_OUT = _REPO / 'results' / 'prior_metatx_validation' / 'data'
_OUT.mkdir(parents=True, exist_ok=True)

# ── Springer static URL for the SI PDF ───────────────────────────────────────
_PDF_URL = (
    'https://static-content.springer.com/esm/'
    'art%3A10.1038%2Fs41522-025-00807-6/MediaObjects/'
    '41522_2025_807_MOESM1_ESM.pdf'
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PRIOR-EDGE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def load_prior_edges(force_reparse: bool = False) -> list[dict]:
    """Extract signed prior edges from Szafranski Suppl. File 1.

    Returns a list of edge dicts (see module docstring for schema).
    Writes/reads  results/prior_metatx_validation/data/prior_edges.json.

    Edge types:
      +1  cross-feeding:   producer P × consumer C on same metabolite M (P≠C)
      -1  competition:     two consumers of the same M (excluded: O2, CO2, H2O2)
      -1  inhibition:      IS_INHIBITED_BY (producer → inhibited guild)

    ec_from_supplfile:  EC number from the Suppl File 1 EC column where available
                        (present only on enzyme-OBJECT rows).  Flag these as
                        potentially circular when used as a metabolite→EC bridge.
    """
    _out_path = _OUT / 'prior_edges.json'
    if _out_path.exists() and not force_reparse:
        with open(_out_path) as fh:
            return json.load(fh)

    # ── import helpers from repo root ─────────────────────────────────────────
    from build_net_flow_expanded import (
        _load_suppfile, _guild_of, _COMPETITION_EXCLUDE,
    )
    from guild_replicator_dieckow import GUILD_ORDER

    valid_guilds = set(GUILD_ORDER)

    prod_rel  = {'PRODUCES', 'RELEASES'}
    cons_rel  = {'USES', 'DEPENDS_ON', 'HYDROLYSES', 'DEGRADES'}
    inhib_rel = {'IS_INHIBITED_BY'}

    df = _load_suppfile()

    # Normalise column names to upper-case stripped strings
    df.columns = [c.strip().upper() for c in df.columns]

    def _get(row, col, default=None):
        val = row.get(col, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        s = str(val).strip()
        return s if s not in ('', 'nan', 'NaN', 'N/A', 'n/a') else default

    edges: list[dict] = []

    for met, mdf in df.groupby('OBJECT'):
        met_str = str(met).strip().lower()

        # Collect producers, consumers, inhibited guilds per metabolite
        # and any EC values from the EC column (enzyme rows only)
        producers:  list[tuple[str, str]] = []   # (guild, evidence)
        consumers:  list[tuple[str, str]] = []
        inhibited:  list[tuple[str, str]] = []

        # EC numbers associated with this OBJECT from Suppl File 1
        ec_set: set[str] = set()

        for _, row in mdf.iterrows():
            taxon = _get(row, 'TAXON', '')
            g     = _guild_of(taxon) if taxon else None
            if g not in valid_guilds:
                g = None

            rel  = str(_get(row, 'RELATIONSHIP', '')).strip()
            evid = str(_get(row, 'EVIDENCE', 'unknown')).strip().lower()
            evid = 'experimental' if 'exp' in evid else 'prediction'

            # Collect EC from EC column (only non-null)
            ec_val = _get(row, 'EC', None)
            if ec_val and re.match(r'^\d+\.\d+\.\d+[\.\-]\d+', ec_val):
                ec_set.add(ec_val.strip())

            if g is None:
                continue
            if rel in prod_rel:
                producers.append((g, evid))
            elif rel in cons_rel:
                consumers.append((g, evid))
            elif rel in inhib_rel:
                inhibited.append((g, evid))

        hmdb_vals = mdf['HMDB_ID'].dropna().unique().tolist() if 'HMDB_ID' in mdf.columns else []
        hmdb_str  = '; '.join(str(h) for h in hmdb_vals if str(h).strip() not in ('', 'nan'))
        hmdb_out  = hmdb_str if hmdb_str else None
        ec_list   = sorted(ec_set) if ec_set else None  # may be circular; flagged in schema

        # Cross-feeding edges: producer → consumer (+1)
        for (pg, pevid) in producers:
            for (cg, cevid) in consumers:
                if pg == cg:
                    continue
                evid_final = 'experimental' if (
                    'experimental' in (pevid, cevid)) else 'prediction'
                edges.append(dict(
                    producer_guild=pg,
                    consumer_guild=cg,
                    metabolite=str(met).strip(),
                    hmdb=hmdb_out,
                    relationship='cross_feeding',
                    sign=+1,
                    evidence=evid_final,
                    ec_from_supplfile=ec_list,
                ))

        # Inhibition edges: producer → inhibited (-1)
        for (pg, pevid) in producers:
            for (ig, ievid) in inhibited:
                if pg == ig:
                    continue
                evid_final = 'experimental' if (
                    'experimental' in (pevid, ievid)) else 'prediction'
                edges.append(dict(
                    producer_guild=pg,
                    consumer_guild=ig,
                    metabolite=str(met).strip(),
                    hmdb=hmdb_out,
                    relationship='inhibition',
                    sign=-1,
                    evidence=evid_final,
                    ec_from_supplfile=ec_list,
                ))

        # Competition edges: consumer_A × consumer_B on same metabolite (-1)
        if met_str not in _COMPETITION_EXCLUDE:
            con_guilds = [(g, e) for (g, e) in consumers]
            for i, (ga, ea) in enumerate(con_guilds):
                for (gb, eb) in con_guilds[i+1:]:
                    if ga == gb:
                        continue
                    evid_final = 'experimental' if (
                        'experimental' in (ea, eb)) else 'prediction'
                    # Emit both directions (asymmetric: A harms B and B harms A)
                    for src, tgt, es, et in [(ga, gb, ea, eb), (gb, ga, eb, ea)]:
                        evid_f = 'experimental' if 'experimental' in (es, et) else 'prediction'
                        edges.append(dict(
                            producer_guild=src,
                            consumer_guild=tgt,
                            metabolite=str(met).strip(),
                            hmdb=hmdb_out,
                            relationship='competition',
                            sign=-1,
                            evidence=evid_f,
                            ec_from_supplfile=ec_list,
                        ))

    with open(_out_path, 'w') as fh:
        json.dump(edges, fh, indent=2)

    _summary = (
        f"prior_edges: {len(edges)} edges  "
        f"(+1={sum(1 for e in edges if e['sign']==+1)}  "
        f"-1={sum(1 for e in edges if e['sign']==-1)}  "
        f"exp={sum(1 for e in edges if e['evidence']=='experimental')}  "
        f"pred={sum(1 for e in edges if e['evidence']=='prediction')})"
    )
    print(_summary)
    (_OUT / 'prior_edges_summary.txt').write_text(_summary + '\n')

    return edges


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  JOSHI 2025 SI PDF PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def _get_pdf_reader():
    """Return a pypdf.PdfReader for the Joshi 2025 SI PDF.

    Search order: canonical Datasets path → /tmp fallback → download from Springer.
    """
    import pypdf

    for candidate in (_PDF_CANONICAL, _PDF_FALLBACK):
        if candidate.exists():
            return pypdf.PdfReader(str(candidate), strict=False)

    # Attempt download
    print(f'Downloading Joshi 2025 SI PDF from Springer …')
    try:
        import urllib.request
        _PDF_CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_PDF_URL, str(_PDF_CANONICAL))
        print(f'Saved to {_PDF_CANONICAL}')
        return pypdf.PdfReader(str(_PDF_CANONICAL), strict=False)
    except Exception as exc:
        raise FileNotFoundError(
            f'Cannot find or download Joshi 2025 SI PDF.\n'
            f'Expected at {_PDF_CANONICAL} or {_PDF_FALLBACK}.\n'
            f'Download URL: {_PDF_URL}\n'
            f'Error: {exc}'
        )


# ── numeric parsing helper ─────────────────────────────────────────────────────

_NUM = re.compile(r'^-?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?$')


def _to_float(s: str):
    """Parse a string to float; return NaN on failure (incl. 'NA', '0' for NA)."""
    s = s.strip()
    if s in ('', 'NA', 'N/A', 'na'):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ── Supplement F parser ───────────────────────────────────────────────────────

_EC_RE = re.compile(r'^\d+\.\d+\.\d+[\.\-]\d+')

# Supp F header line pattern — we need to identify the header token to skip it
_SUPPF_HEADER_TOKENS = {'EC', 'Mean', 'abundance', 'Std_deviation', 'health',
                         'peri-', 'peri', 'implantitis', 'FDR', 'LDA',
                         'corrected', 'correlation', 'value', 'deviation',
                         'P', 'CAP', '(CAP)'}


def _parse_suppF(reader) -> pd.DataFrame:
    """Parse Supplement F (EC metatranscriptome, pp 31-62, 0-indexed 30-61).

    Returns DataFrame with columns:
      EC, mean_abund, std_dev, mean_health, mean_peri, p_value, fdr, lda
    """
    records = []
    # Pages 31-62 inclusive (0-indexed 30-61); page 63 starts Supp G
    for p_idx in range(30, 62):
        text = reader.pages[p_idx].extract_text() or ''
        # pypdf extracts the table as one long run-on line per data row
        # because the PDF uses flowing text boxes.
        # Strategy: tokenise by whitespace, then reassemble rows by
        # recognising an EC-number token as the row start.
        tokens = text.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if _EC_RE.match(tok):
                # Collect 7 numeric tokens following the EC
                nums = []
                j = i + 1
                while j < len(tokens) and len(nums) < 7:
                    # skip repeated header tokens that bleed into data pages
                    if tokens[j] in _SUPPF_HEADER_TOKENS:
                        j += 1
                        continue
                    nums.append(tokens[j])
                    j += 1
                if len(nums) >= 7:
                    try:
                        records.append({
                            'EC':         tok,
                            'mean_abund': _to_float(nums[0]),
                            'std_dev':    _to_float(nums[1]),
                            'mean_health':_to_float(nums[2]),
                            'mean_peri':  _to_float(nums[3]),
                            'p_value':    _to_float(nums[4]),
                            'fdr':        _to_float(nums[5]),
                            'lda':        _to_float(nums[6]),
                        })
                    except Exception:
                        pass
                i = j
            else:
                i += 1

    df = pd.DataFrame(records)
    # Deduplicate: keep first occurrence (pages repeat header on continuation pages)
    df = df.drop_duplicates(subset='EC', keep='first').reset_index(drop=True)
    return df


# ── Supplement H parser ───────────────────────────────────────────────────────

# Class_Group format: "{Class}_{AA}_{mode}"  e.g. "Actinomycetia_Arg_a"
# Group format:       "{AA}_{mode}"          e.g. "Arg_a"
# Class format:       "Actinomycetia"
# Note: Thr and Trp have a trailing space in the PDF (e.g. "Thr _a")

_SUPPF_AMINO_HEADER_TOKENS = {'Class_Group', 'Group', 'Class', 'Mean',
                               'abundance', 'health', 'peri-', 'peri',
                               'implantitis', 'P', 'FDR', 'corrected', 'value'}

# Known amino acid prefixes in the data
_AA_CODES = {
    'Arg', 'Cys', 'His', 'Lys', 'Met', 'phe', 'Phe', 'Pro', 'Thr', 'Trp', 'Tyr',
}

# Known class → guild mapping (Joshi taxon classes → GUILD_ORDER)
# This is the bridge used in Tier A validation.
JOSHI_CLASS_TO_GUILD: dict[str, str] = {
    # Joshi metatranscriptome class → nife GUILD_ORDER guild
    'Actinomycetia':        'Actinobacteria',   # Actinomyces, Rothia, Schaalia
    'Bacilli':              'Bacilli',
    'Bacteroidia':          'Bacteroidia',
    'Bacteroidales':        'Bacteroidia',       # order-level in some rows
    'Betaproteobacteria':   'Betaproteobacteria',
    'BetaProteobacteria':   'Betaproteobacteria',
    'Clostridia':           'Clostridia',
    'Coriobacteriia':       'Coriobacteriia',
    'Fusobacteriia':        'Fusobacteriia',
    'Gammaproteobacteria':  'Gammaproteobacteria',
    'GammaProteobacteria':  'Gammaproteobacteria',
    'Negativicutes':        'Negativicutes',
    # Classes without a direct guild mapping (mapped to 'Other' or None)
    'AlphaProteobacteria':  None,
    'Alphaproteobacteria':  None,
    'Anaerolineae':         None,
    'Campylobacteria':      None,
    'DeltaProteobacteria':  None,
    'Deltaproteobacteria':  None,
    'Desulfovibrionia':     None,
    'EpsilonProteobacteria':None,
    'Epsilonproteobacteria':None,
    'Erysipelotrichia':     None,
    'Flavobacteriia':       None,   # Capnocytophaga → excluded from GUILD_ORDER
    'Spirochaetia':         None,
    'Synergistia':          None,
    'TM7 [C-1]':            None,
    'SR1 [C-1]':            None,
}


def _parse_suppH(reader) -> pd.DataFrame:
    """Parse Supplement H (amino-acid metabolism, pp 64-68, 0-indexed 63-67).

    Returns DataFrame with columns:
      class_group, group, taxon_class, aa, mode, mean_abund, mean_health,
      mean_peri, p_value, fdr
    """
    records = []
    # Pages 64-68 (0-indexed 63-67); page 69 starts Supp I
    for p_idx in range(63, 68):
        text = reader.pages[p_idx].extract_text() or ''
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Each data row in Supp H starts with the Class_Group token
            # e.g. "Actinomycetia_Arg_a Arg_a Actinomycetia17.80 19.53 16.07 0.28 0.40"
            # pypdf merges some columns without spaces; use regex to split
            # Pattern: word_word[_word]+ [optional space] numerics
            m = re.match(
                r'^([A-Za-z][A-Za-z0-9 \[\]\-]+?_[A-Za-z]{2,4}_[au])'
                r'\s+'
                r'([A-Za-z]{2,4}_[au]|[A-Za-z]{2,4}\s_[au])'  # Group col (may have space)
                r'\s+'
                r'([A-Za-z][A-Za-z0-9 \[\]\-]+?)'             # Class (no trailing number)
                r'(\d.*)$',                                     # numerics
                line,
            )
            if m is None:
                continue

            cg_raw  = m.group(1).strip()   # Class_Group
            grp_raw = m.group(2).strip()   # Group
            cls_raw = m.group(3).strip()   # Class
            num_str = m.group(4).strip()

            # Normalise Thr _a → Thr_a etc.
            grp_raw = re.sub(r'\s+_', '_', grp_raw)

            # Parse group into aa + mode
            grp_parts = grp_raw.rsplit('_', 1)
            if len(grp_parts) != 2:
                continue
            aa_raw, mode = grp_parts
            aa_raw = aa_raw.strip()
            mode   = mode.strip()
            if mode not in ('a', 'u'):
                continue

            # Parse 5 numeric values: mean_abund, mean_health, mean_peri, p_value, fdr
            num_tokens = num_str.split()
            # Filter out any class name bleed-through
            num_tokens = [t for t in num_tokens if _NUM.match(t) or t in ('NA',)]
            if len(num_tokens) < 5:
                continue

            try:
                records.append({
                    'class_group':  cg_raw,
                    'group':        grp_raw,
                    'taxon_class':  cls_raw,
                    'aa':           aa_raw,
                    'mode':         mode,
                    'mean_abund':   _to_float(num_tokens[0]),
                    'mean_health':  _to_float(num_tokens[1]),
                    'mean_peri':    _to_float(num_tokens[2]),
                    'p_value':      _to_float(num_tokens[3]),
                    'fdr':          _to_float(num_tokens[4]),
                })
            except Exception:
                continue

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset='class_group', keep='first').reset_index(drop=True)
    return df


# ── Supplement E class-level 16S parser ───────────────────────────────────────

def _parse_class_abund(reader, page_start_0idx: int, page_end_0idx: int,
                       section_header: str) -> pd.DataFrame:
    """Generic parser for class-level abundance tables (Supp E and Supp G).

    Columns: taxon_class, mean_abund, std_dev, mean_health, mean_peri, p_value, fdr
    """
    records = []
    header_skip = {
        'Class', 'Mean', 'abundance', 'Std_deviation', 'health', 'peri-', 'peri',
        'implantitis', 'FDR', 'corrected', 'value', 'P', 'deviation',
        section_header.split()[0],  # e.g. 'E.' or 'G.'
    }
    for p_idx in range(page_start_0idx, page_end_0idx + 1):
        text = reader.pages[p_idx].extract_text() or ''
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Class name = first token(s) before a numeric run
            m = re.match(
                r'^([A-Za-z][A-Za-z0-9 \[\]\-]+?)\s+([\d\-][\d\.Ee+\- ]+)$',
                line,
            )
            if m is None:
                continue
            cls_raw  = m.group(1).strip()
            num_part = m.group(2).strip()
            if cls_raw in header_skip:
                continue
            num_tokens = [t for t in num_part.split() if _NUM.match(t) or t == 'NA']
            if len(num_tokens) < 6:
                continue
            try:
                records.append({
                    'taxon_class': cls_raw,
                    'mean_abund':  _to_float(num_tokens[0]),
                    'std_dev':     _to_float(num_tokens[1]),
                    'mean_health': _to_float(num_tokens[2]),
                    'mean_peri':   _to_float(num_tokens[3]),
                    'p_value':     _to_float(num_tokens[4]),
                    'fdr':         _to_float(num_tokens[5]),
                })
            except Exception:
                continue

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset='taxon_class', keep='first').reset_index(drop=True)
    return df


# ── Master loader ──────────────────────────────────────────────────────────────

def load_joshi_si(force_reparse: bool = False) -> dict[str, pd.DataFrame]:
    """Parse Joshi 2025 SI PDF into clean pandas DataFrames.

    Returns dict with keys: 'suppF', 'suppH', 'suppE_class', 'suppG'.
    Writes CSVs to results/prior_metatx_validation/data/.

    PDF page ranges (1-indexed, matching scope):
      Supp F : 31-62  (EC metatranscriptome)
      Supp G : 63     (metatranscriptome class level)
      Supp H : 64-68  (amino-acid metabolism, taxon-resolved)
      Supp E : 30     (full-16S class level)
    """
    paths = {
        'suppF':      _OUT / 'joshi_suppF_ec.csv',
        'suppH':      _OUT / 'joshi_suppH_taxon_aa.csv',
        'suppE_class':_OUT / 'joshi_class_abund_16S.csv',
        'suppG':      _OUT / 'joshi_class_abund_tx.csv',
    }

    # Load from cache if all files present
    if not force_reparse and all(p.exists() for p in paths.values()):
        return {k: pd.read_csv(v) for k, v in paths.items()}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reader = _get_pdf_reader()

    print(f'Parsing Joshi 2025 SI PDF ({len(reader.pages)} pages) …')

    # ── Supplement F (EC table, pp 31-62, 0-indexed 30-61) ───────────────────
    suppF = _parse_suppF(reader)
    suppF.to_csv(paths['suppF'], index=False)
    print(f'  Supp F: {len(suppF)} EC rows → {paths["suppF"].name}')

    # ── Supplement G (metatranscriptome class level, p 63, 0-indexed 62) ─────
    suppG = _parse_class_abund(reader, 62, 62, 'G.')
    suppG.to_csv(paths['suppG'], index=False)
    print(f'  Supp G: {len(suppG)} class rows → {paths["suppG"].name}')

    # ── Supplement H (amino-acid metabolism, pp 64-68, 0-indexed 63-67) ──────
    suppH = _parse_suppH(reader)
    suppH.to_csv(paths['suppH'], index=False)
    print(f'  Supp H: {len(suppH)} class×AA rows → {paths["suppH"].name}')

    # ── Supplement E (full-16S class level, p 30, 0-indexed 29) ─────────────
    suppE = _parse_class_abund(reader, 29, 30, 'E.')
    suppE.to_csv(paths['suppE_class'], index=False)
    print(f'  Supp E: {len(suppE)} class rows → {paths["suppE_class"].name}')

    return {
        'suppF':       suppF,
        'suppH':       suppH,
        'suppE_class': suppE,
        'suppG':       suppG,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  QUICK DIAGNOSTICS (run as script)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='DATA layer: extract prior edges + parse Joshi SI PDF')
    parser.add_argument('--force', action='store_true',
                        help='Re-parse even if cached files exist')
    parser.add_argument('--skip-prior', action='store_true',
                        help='Skip prior edge extraction (xlsx may be missing)')
    parser.add_argument('--skip-joshi', action='store_true',
                        help='Skip Joshi SI PDF parsing')
    args = parser.parse_args()

    print('=' * 60)
    print('DATA LAYER — prior_metatx_data.py')
    print('=' * 60)

    if not args.skip_prior:
        print('\n[1] Prior edges …')
        edges = load_prior_edges(force_reparse=args.force)
        cf   = [e for e in edges if e['sign'] == +1]
        comp = [e for e in edges if e['sign'] == -1]
        expt = [e for e in edges if e['evidence'] == 'experimental']
        pred = [e for e in edges if e['evidence'] == 'prediction']
        ec_edges = [e for e in edges if e['ec_from_supplfile'] is not None]
        print(f'  Total edges   : {len(edges)}')
        print(f'  Cross-feeding : {len(cf)}')
        print(f'  Competition   : {len(comp)}')
        print(f'  Experimental  : {len(expt)}')
        print(f'  Prediction    : {len(pred)}')
        print(f'  Edges w/ EC from Suppl File 1 (circular risk): {len(ec_edges)}')
        print(f'  Output: {_OUT / "prior_edges.json"}')

        # Guild coverage
        from guild_replicator_dieckow import GUILD_ORDER
        guilds_seen = set()
        for e in edges:
            if e['producer_guild']:
                guilds_seen.add(e['producer_guild'])
            if e['consumer_guild']:
                guilds_seen.add(e['consumer_guild'])
        missing = set(GUILD_ORDER) - guilds_seen
        print(f'  Guilds covered: {sorted(guilds_seen)}')
        if missing:
            print(f'  Guilds absent from edges: {sorted(missing)}')

    if not args.skip_joshi:
        print('\n[2] Joshi 2025 SI …')
        dfs = load_joshi_si(force_reparse=args.force)

        suppF = dfs['suppF']
        suppH = dfs['suppH']
        suppE = dfs['suppE_class']
        suppG = dfs['suppG']

        print(f'\n  Supp F (EC metatranscriptome):')
        print(f'    {len(suppF)} ECs, columns: {list(suppF.columns)}')
        sig_F = suppF[suppF['fdr'].notna() & (suppF['fdr'] < 0.05)]
        print(f'    FDR<0.05: {len(sig_F)} ECs')
        print(f'    Sample rows:')
        print(suppF.head(3).to_string(index=False))

        print(f'\n  Supp H (amino-acid metabolism, taxon-resolved):')
        print(f'    {len(suppH)} rows, columns: {list(suppH.columns)}')
        print(f'    Classes: {sorted(suppH["taxon_class"].unique())}')
        print(f'    AAs: {sorted(suppH["aa"].unique())}')
        print(f'    Sample rows:')
        print(suppH.head(3).to_string(index=False))

        print(f'\n  Supp E (16S class abundance):')
        print(f'    {len(suppE)} classes: {list(suppE["taxon_class"])}')

        print(f'\n  Supp G (metatranscriptome class abundance):')
        print(f'    {len(suppG)} classes: {list(suppG["taxon_class"])}')

        # EC coverage: prior's 14 unique ECs vs Supp F universe
        if not args.skip_prior:
            edges = load_prior_edges()
            prior_ecs: set[str] = set()
            for e in edges:
                if e.get('ec_from_supplfile'):
                    for ec in e['ec_from_supplfile']:
                        prior_ecs.add(ec)
            suppF_ecs = set(suppF['EC'].str.strip())
            overlap = prior_ecs & suppF_ecs
            print(f'\n  EC overlap check:')
            print(f'    Prior ECs (from Suppl File 1 EC column): {len(prior_ecs)}')
            print(f'    Supp F universe: {len(suppF_ecs)}')
            print(f'    Overlap (circular): {len(overlap)}')
            print(f'    Prior ECs not in Supp F: {prior_ecs - suppF_ecs}')

    print('\nDone. Output dir:', _OUT)
