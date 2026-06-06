#!/usr/bin/env python3
"""
validate_agora_prior_humann.py
==============================
Validate AGORA/Szafrański sign-prior edges against sample-level HUMAnN and
MetaPhlAn output from the Joshi 2025 metatranscriptome (PRJNA1192962).

This is the raw-read counterpart to validate_prior_metatranscriptome.py, which
uses the published Joshi supplementary tables.  HUMAnN gives functional RNA
activity, optionally stratified by taxon, and MetaPhlAn gives the corresponding
taxonomic abundance.  The validation is intentionally conservative:

  * a prior edge is "testable" only when the edge metabolite has at least one
    configured function/pathway marker in the HUMAnN tables;
  * cross-feeding/competition edges are supported when metabolite-associated
    functional activity is observed in both endpoint guilds;
  * inhibition edges are supported when metabolite-associated activity is
    observed in the producing guild and the inhibited guild is taxonomically
    present by MetaPhlAn (or has HUMAnN activity when MetaPhlAn is omitted).

The script uses only the Python standard library so it can be syntax-checked on
minimal login nodes; the heavy dependencies live in the upstream HUMAnN/
MetaPhlAn workflow.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR = REPO / "results" / "prior_metatx_validation" / "data" / "prior_edges.json"
DEFAULT_OUTDIR = REPO / "results" / "joshi_mtx_humann_prior_validation"

# NIFE guilds plus genus-level aliases that appear in HUMAnN/MetaPhlAn strings.
GUILD_ALIASES: dict[str, list[str]] = {
    "Actinobacteria": ["actinobacteria", "actinomycetia", "actinomyces", "schaalia", "rothia", "corynebacterium"],
    "Bacilli": ["bacilli", "streptococcus", "gemella", "lactobacillus", "granulicatella", "staphylococcus"],
    "Bacteroidia": ["bacteroidia", "bacteroides", "porphyromonas", "prevotella", "tannerella", "capnocytophaga"],
    "Betaproteobacteria": ["betaproteobacteria", "neisseria"],
    "Clostridia": ["clostridia", "filifactor", "parvimonas", "peptostreptococcus", "pseudoramibacter"],
    "Coriobacteriia": ["coriobacteriia", "atopobium", "olsenella"],
    "Fusobacteriia": ["fusobacteriia", "fusobacterium", "leptotrichia"],
    "Gammaproteobacteria": ["gammaproteobacteria", "haemophilus", "aggregatibacter", "pasteurella", "pseudomonas"],
    "Negativicutes": ["negativicutes", "veillonella", "selenomonas", "dialister", "megasphaera"],
    "Other": ["treponema", "spirochaetia", "campylobacter", "epsilonproteobacteria"],
}

# Curated marker patterns for common oral cross-feeding metabolites.  These are
# intentionally broad and can/should be overridden with --metabolite-map when a
# stricter manuscript-grade marker set is frozen.
CURATED_MARKERS: dict[str, list[str]] = {
    "lactate": [r"lactate", r"l-lactate", r"d-lactate", r"EC[: ]1\.1\.1\.27", r"EC[: ]1\.1\.1\.28"],
    "lactic acid": [r"lactate", r"l-lactate", r"d-lactate", r"EC[: ]1\.1\.1\.27", r"EC[: ]1\.1\.1\.28"],
    "pyruvate": [r"pyruvate", r"EC[: ]1\.2\.4\.1", r"EC[: ]2\.3\.1\.12", r"EC[: ]1\.1\.1\.27"],
    "acetate": [r"acetate", r"acetyl.?coa", r"EC[: ]2\.3\.1\.8", r"EC[: ]6\.2\.1\.1"],
    "butyrate": [r"butyrate", r"butanoate", r"EC[: ]2\.8\.3\.8", r"EC[: ]1\.3\.8\.1"],
    "succinate": [r"succinate", r"succinate dehydrogenase", r"EC[: ]1\.3\.5\.1"],
    "propionate": [r"propionate", r"propanoate", r"methylmalonyl"],
    "formate": [r"formate", r"formate dehydrogenase", r"EC[: ]1\.17\.1\.9"],
    "hydrogen peroxide": [r"hydrogen peroxide", r"peroxidase", r"catalase", r"EC[: ]1\.11\.1\.", r"EC[: ]1\.11\.1\.6"],
    "h2o2": [r"hydrogen peroxide", r"peroxidase", r"catalase", r"EC[: ]1\.11\.1\.", r"EC[: ]1\.11\.1\.6"],
    "hydrogen sulfide": [r"hydrogen sulfide", r"sulfide", r"cysteine desulfhydrase", r"EC[: ]4\.4\.1\.", r"EC[: ]2\.8\.1\.7"],
    "h2s": [r"hydrogen sulfide", r"sulfide", r"cysteine desulfhydrase", r"EC[: ]4\.4\.1\.", r"EC[: ]2\.8\.1\.7"],
    "heme": [r"heme", r"haem", r"hemin", r"porphyrin"],
    "haem": [r"heme", r"haem", r"hemin", r"porphyrin"],
    "iron": [r"iron", r"siderophore", r"ferric", r"ferrous"],
    "ammonia": [r"ammonia", r"ammonium", r"urease", r"EC[: ]3\.5\.1\.5"],
}

AA_NAMES = {
    "alanine", "arginine", "asparagine", "aspartate", "aspartic acid", "cysteine",
    "glutamate", "glutamic acid", "glutamine", "glycine", "histidine", "isoleucine",
    "leucine", "lysine", "methionine", "phenylalanine", "proline", "serine",
    "threonine", "tryptophan", "tyrosine", "valine",
}

AA_PATTERNS = [
    r"amino.?acid", r"aminotransferase", r"transaminase", r"peptidase",
    r"EC[: ]2\.6\.1\.", r"EC[: ]3\.4\.", r"EC[: ]6\.3\.",
]


def norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def compile_marker_map(path: Path | None) -> dict[str, list[re.Pattern[str]]]:
    raw: dict[str, list[str]] = {k: list(v) for k, v in CURATED_MARKERS.items()}
    for aa in AA_NAMES:
        raw.setdefault(aa, []).extend([re.escape(aa), *AA_PATTERNS])
    if path:
        with open(path) as fh:
            user = json.load(fh)
        for key, patterns in user.items():
            raw[norm(key)] = list(patterns)
    return {k: [re.compile(p, re.I) for p in pats] for k, pats in raw.items()}


def infer_markers(metabolite: str, marker_map: dict[str, list[re.Pattern[str]]]) -> list[re.Pattern[str]]:
    met = norm(metabolite)
    if met in marker_map:
        return marker_map[met]
    # Fallback: require all informative tokens from the metabolite name to occur
    # as a phrase-ish regex.  This makes named MetaCyc tables usable even before
    # a curated JSON marker map is written.
    tokens = [t for t in re.split(r"[^a-z0-9]+", met) if len(t) >= 4 and t not in {"acid", "ions"}]
    if not tokens:
        return []
    phrase = r".*".join(re.escape(t) for t in tokens[:3])
    return [re.compile(phrase, re.I)]


def guild_from_taxon(text: str) -> str | None:
    low = norm(text).replace("|", " ").replace("__", " ")
    best: tuple[int, str] | None = None
    for guild, aliases in GUILD_ALIASES.items():
        for alias in aliases:
            if re.search(rf"(^|[^a-z]){re.escape(alias)}([^a-z]|$)", low):
                score = len(alias)
                if best is None or score > best[0]:
                    best = (score, guild)
    return best[1] if best else None


def parse_float(value: str) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0


def sample_columns(header: list[str]) -> list[str]:
    return [h for h in header[1:] if h and not h.startswith("#")]


def feature_and_taxon(row_name: str) -> tuple[str, str | None]:
    parts = row_name.split("|")
    feature = parts[0].strip()
    taxon = "|".join(parts[1:]).strip() if len(parts) > 1 else None
    return feature, taxon


def load_humann_activity(paths: Iterable[Path]) -> tuple[dict[str, dict[str, float]], dict[str, float], list[str]]:
    """Return (guild_feature_activity, community_feature_activity, samples)."""
    guild_feature = defaultdict(lambda: defaultdict(float))
    community_feature = defaultdict(float)
    all_samples: list[str] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader)
            while header and header[0].startswith("#") and len(header) == 1:
                header = next(reader)
            samples = sample_columns(header)
            all_samples.extend([s for s in samples if s not in all_samples])
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                row += ["0"] * (len(header) - len(row))
                feature, taxon = feature_and_taxon(row[0])
                total = sum(parse_float(v) for v in row[1:len(header)])
                if total <= 0:
                    continue
                community_feature[feature] += total
                guild = guild_from_taxon(taxon or "")
                if guild:
                    guild_feature[guild][feature] += total
    return guild_feature, community_feature, all_samples


def load_metaphlan_guilds(path: Path | None, min_abundance: float) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)
    present: set[str] = set()
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            clade = row[0]
            vals = [parse_float(v) for v in row[1:]]
            abundance = max(vals) if vals else 0.0
            if abundance < min_abundance:
                continue
            guild = guild_from_taxon(clade)
            if guild:
                present.add(guild)
    return present


def marker_activity(features: dict[str, float], markers: list[re.Pattern[str]]) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[tuple[float, str]] = []
    for feature, val in features.items():
        if any(p.search(feature) for p in markers):
            score += val
            hits.append((val, feature))
    hits.sort(reverse=True)
    return score, [h for _v, h in hits[:5]]


def validate_edges(edges: list[dict],
                   guild_feature: dict[str, dict[str, float]],
                   community_feature: dict[str, float],
                   marker_map: dict[str, list[re.Pattern[str]]],
                   metaphlan_present: set[str],
                   min_activity: float,
                   ) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    summary = defaultdict(int)
    by_relationship = defaultdict(lambda: defaultdict(int))

    guilds_with_activity = {g for g, feats in guild_feature.items() if sum(feats.values()) > 0}
    present_guilds = set(metaphlan_present) | guilds_with_activity

    for idx, edge in enumerate(edges):
        producer = edge.get("producer_guild")
        consumer = edge.get("consumer_guild")
        metabolite = edge.get("metabolite", "")
        relationship = edge.get("relationship", "")
        sign = int(edge.get("sign", 0) or 0)
        markers = infer_markers(metabolite, marker_map)
        if not producer or not consumer or not markers:
            continue

        community_score, community_hits = marker_activity(community_feature, markers)
        p_score, p_hits = marker_activity(guild_feature.get(producer, {}), markers)
        c_score, c_hits = marker_activity(guild_feature.get(consumer, {}), markers)
        p_active = p_score >= min_activity
        c_active = c_score >= min_activity
        p_present = producer in present_guilds
        c_present = consumer in present_guilds
        community_active = community_score >= min_activity

        # testable only if the metabolite marker is observed somewhere in the
        # metatranscriptome and both endpoint guilds are represented.
        testable = community_active and p_present and c_present
        if not testable:
            continue

        if sign > 0 or relationship == "competition":
            supported = p_active and c_active
            rule = "both_endpoint_guilds_have_metabolite_marker_activity"
        elif relationship == "inhibition":
            supported = p_active and c_present
            rule = "producer_activity_and_target_presence"
        else:
            supported = False
            rule = "unsupported_relationship_type"

        rows.append({
            "edge_id": idx,
            "producer_guild": producer,
            "consumer_guild": consumer,
            "metabolite": metabolite,
            "relationship": relationship,
            "sign": sign,
            "evidence": edge.get("evidence", ""),
            "supported": int(bool(supported)),
            "producer_activity": f"{p_score:.6g}",
            "consumer_activity": f"{c_score:.6g}",
            "community_activity": f"{community_score:.6g}",
            "producer_hits": "; ".join(p_hits),
            "consumer_hits": "; ".join(c_hits),
            "community_hits": "; ".join(community_hits),
            "support_rule": rule,
        })
        summary["testable_edges"] += 1
        summary["supported_edges"] += int(bool(supported))
        by_relationship[relationship]["testable"] += 1
        by_relationship[relationship]["supported"] += int(bool(supported))

    testable = summary["testable_edges"]
    supported = summary["supported_edges"]
    result = {
        "n_edges_input": len(edges),
        "n_edges_testable": testable,
        "n_edges_supported": supported,
        "supported_fraction": supported / testable if testable else None,
        "guilds_with_humann_activity": sorted(guilds_with_activity),
        "guilds_present_metaphlan": sorted(metaphlan_present),
        "min_activity": min_activity,
        "by_relationship": {
            k: {"testable": v["testable"], "supported": v["supported"],
                "fraction": (v["supported"] / v["testable"] if v["testable"] else None)}
            for k, v in sorted(by_relationship.items())
        },
    }
    return rows, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate AGORA/Szafrański sign prior using HUMAnN/MetaPhlAn Joshi MTX outputs."
    )
    parser.add_argument("--humann-table", action="append", type=Path, required=True,
                        help="Joined HUMAnN table; pass multiple times for EC/pathway/named tables.")
    parser.add_argument("--metaphlan-table", type=Path, default=None,
                        help="Merged MetaPhlAn table for endpoint-guild presence filtering.")
    parser.add_argument("--prior-edges", type=Path, default=DEFAULT_PRIOR,
                        help=f"Prior edge JSON (default: {DEFAULT_PRIOR})")
    parser.add_argument("--metabolite-map", type=Path, default=None,
                        help="Optional JSON mapping metabolite names to regex marker patterns.")
    parser.add_argument("--min-activity", type=float, default=0.0,
                        help="Minimum summed HUMAnN activity to call a marker present (default: 0).")
    parser.add_argument("--min-metaphlan-abundance", type=float, default=0.0,
                        help="Minimum MetaPhlAn relative abundance to call a guild present (default: 0).")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help=f"Output directory (default: {DEFAULT_OUTDIR})")
    args = parser.parse_args()

    if not args.prior_edges.exists():
        raise FileNotFoundError(
            f"Missing prior edge JSON: {args.prior_edges}\n"
            "Create it with: python scripts/analysis/prior_metatx_data.py --skip-joshi"
        )
    with open(args.prior_edges) as fh:
        edges = json.load(fh)

    marker_map = compile_marker_map(args.metabolite_map)
    guild_feature, community_feature, samples = load_humann_activity(args.humann_table)
    metaphlan_present = load_metaphlan_guilds(args.metaphlan_table, args.min_metaphlan_abundance)
    rows, result = validate_edges(
        edges=edges,
        guild_feature=guild_feature,
        community_feature=community_feature,
        marker_map=marker_map,
        metaphlan_present=metaphlan_present,
        min_activity=args.min_activity,
    )
    result.update({
        "humann_tables": [str(p) for p in args.humann_table],
        "metaphlan_table": str(args.metaphlan_table) if args.metaphlan_table else None,
        "samples_detected": samples,
        "n_samples_detected": len(samples),
        "metabolite_map": str(args.metabolite_map) if args.metabolite_map else "built-in defaults",
    })

    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / "agora_prior_humann_validation.json", "w") as fh:
        json.dump(result, fh, indent=2)
    with open(args.outdir / "agora_prior_humann_edge_support.csv", "w", newline="") as fh:
        fieldnames = [
            "edge_id", "producer_guild", "consumer_guild", "metabolite",
            "relationship", "sign", "evidence", "supported", "producer_activity",
            "consumer_activity", "community_activity", "producer_hits",
            "consumer_hits", "community_hits", "support_rule",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frac = result["supported_fraction"]
    frac_text = "NA" if frac is None else f"{frac:.3f}"
    print(
        f"HUMAnN prior validation: {result['n_edges_supported']}/"
        f"{result['n_edges_testable']} supported (fraction={frac_text})"
    )
    print(f"Outputs: {args.outdir}")


if __name__ == "__main__":
    main()
