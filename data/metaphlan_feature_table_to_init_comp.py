#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InitComp:
    Str: float
    Act: float
    Vel: float
    Hae: float
    Rot: float
    Fus: float
    Por: float


def _match_group(clade: str) -> str | None:
    c = clade.lower()
    if "streptococcus" in c:
        return "Str"
    if "actinomyces" in c or "schaalia" in c:
        return "Act"
    if "veillonella" in c:
        return "Vel"
    if "haemophilus" in c:
        return "Hae"
    if "rothia" in c:
        return "Rot"
    if "fusobacterium" in c:
        return "Fus"
    if "porphyromonas" in c:
        return "Por"
    return None


def _read_feature_table(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) < 2:
            raise ValueError("Invalid feature_table.tsv header")
        clade_col = header[0]
        samples = header[1:]
        table: dict[str, dict[str, float]] = {s: {} for s in samples}
        for row in reader:
            if not row:
                continue
            clade = row[0]
            if clade.startswith("#") or clade == clade_col:
                continue
            for i, s in enumerate(samples, start=1):
                v = float(row[i]) if i < len(row) and row[i] else 0.0
                table[s][clade] = v
    return samples, table


def _aggregate_sample(clade_to_val: dict[str, float], tax_lev: str) -> dict[str, float]:
    out = {k: 0.0 for k in ["Str", "Act", "Vel", "Hae", "Rot", "Fus", "Por"]}
    for clade, v in clade_to_val.items():
        if tax_lev:
            last = clade.split("|")[-1]
            if not last.startswith(f"{tax_lev}__"):
                continue
        g = _match_group(clade)
        if g is None:
            continue
        out[g] += v
    return out


def _normalize(d: dict[str, float], *, fallback_equal: bool = False) -> dict[str, float]:
    s = sum(d.values())
    if s <= 0.0:
        if fallback_equal:
            import warnings
            n = len(d)
            warnings.warn(
                "None of the 7 target genera detected in this sample. "
                "Falling back to equal fractions (1/7 each). "
                "Check that MetaPhlAn ran successfully and the sample name is correct.",
                stacklevel=3,
            )
            return {k: 1.0 / n for k in d}
        raise ValueError(
            "Sum of selected taxa is 0 — none of the 7 target genera "
            "(Streptococcus, Actinomyces, Veillonella, Haemophilus, Rothia, "
            "Fusobacterium, Porphyromonas) were found in this sample. "
            "Re-run with --tax-lev g (genus) or --tax-lev '' (all levels) "
            "to debug, or verify the MetaPhlAn profile is non-empty."
        )
    return {k: v / s for k, v in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Convert MetaPhlAn 4 feature_table.tsv to COMETS init_comp.json. "
            "Aggregates species-level abundances into 7 genus groups "
            "(Str/Act/Vel/Hae/Rot/Fus/Por) and normalises to sum=1."
        )
    )
    ap.add_argument("--feature-table", type=Path, required=True,
                    help="Path to merged MetaPhlAn profile (merge_metaphlan_tables.py output).")
    ap.add_argument("--sample", type=str, required=True,
                    help="Sample column name in feature_table.tsv (e.g. ERR13166576_A_3).")
    ap.add_argument("--tax-lev", type=str, default="s",
                    help="MetaPhlAn tax level prefix to filter (default: 's' for species). "
                         "Use 'g' for genus or '' to include all levels.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON to file (stdout if omitted).")
    ap.add_argument("--fallback-equal", action="store_true",
                    help="If no target genera are detected, fall back to equal fractions "
                         "instead of raising an error.")
    ap.add_argument("--list-samples", action="store_true",
                    help="Print all sample names in the feature table and exit.")
    args = ap.parse_args()

    samples, table = _read_feature_table(args.feature_table)

    if args.list_samples:
        print("\n".join(samples))
        return 0

    if args.sample not in table:
        raise SystemExit(
            f"Unknown sample '{args.sample}'. Available ({len(samples)} total):\n"
            + "\n".join(f"  {s}" for s in samples[:20])
            + ("\n  ..." if len(samples) > 20 else "")
        )

    agg = _aggregate_sample(table[args.sample], args.tax_lev)

    # Show detected abundances for transparency
    detected = {k: v for k, v in agg.items() if v > 0}
    if detected:
        import sys as _sys
        print(f"[info] Detected genera: { {k: round(v,2) for k,v in detected.items()} }",
              file=_sys.stderr)
    else:
        import sys as _sys
        print(f"[warn] No target genera detected at tax_lev='{args.tax_lev}'. "
              "Try --tax-lev g or --tax-lev ''", file=_sys.stderr)

    init = _normalize(agg, fallback_equal=args.fallback_equal)
    out_obj = InitComp(**{k: float(init[k]) for k in init})
    s = json.dumps(out_obj.__dict__, indent=2, sort_keys=True)
    if args.out is None:
        print(s)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(s + "\n")
        import sys as _sys
        print(f"[info] Wrote {args.out}", file=_sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

