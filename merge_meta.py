#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS_RE.sub("", s)
    return s


CANONICAL_COLUMNS = [
    "sample_alias",
    "project_accession",
    "run_accession",
    "biosample_accession",
    "in_vivo_in_vitro",
    "material",
    "day",
    "donor",
    "patient",
    "week",
    "title",
    "collection_date",
    "env_local_scale",
    "env_medium",
    "geo_loc_name",
]


REQUIRED_COLUMNS = [
    "sample_alias",
    "project_accession",
    "in_vivo_in_vitro",
    "material",
    "day",
    "donor",
]


def _force_columns(df: pd.DataFrame, columns: list[str], fill_value: str = "not_provided") -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = fill_value
    df = df[columns]
    for c in df.columns:
        df[c] = df[c].map(_sanitize_text)
        df[c] = df[c].where(df[c].astype(str).str.len() > 0, fill_value)
    return df


def _assert_unique(df: pd.DataFrame, key: str) -> None:
    dup = df[key][df[key].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"Duplicate {key}: {dup[:10]} (total {len(dup)})")


def load_prjeb71108_from_manifest(manifest_path: Path) -> pd.DataFrame:
    m = pd.read_csv(manifest_path, sep="\t")
    need = {"sample", "patient", "timepoint"}
    if not need.issubset(set(m.columns)):
        raise ValueError(f"Manifest missing required columns: {sorted(need - set(m.columns))}")
    run_col = "err_accession" if "err_accession" in m.columns else None
    if run_col is None:
        raise ValueError("Manifest missing err_accession column")

    df = pd.DataFrame(
        {
            "sample_alias": m["sample"].astype(str),
            "project_accession": "PRJEB71108",
            "run_accession": m[run_col].astype(str),
            "biosample_accession": "not_provided",
            "in_vivo_in_vitro": "in_vivo",
            "material": "not_provided",
            "day": "not_provided",
            "donor": m["patient"].astype(str),
            "patient": m["patient"].astype(str),
            "week": m["timepoint"].astype(str),
            "title": "not_provided",
            "collection_date": "not_provided",
            "env_local_scale": "not_provided",
            "env_medium": "not_provided",
            "geo_loc_name": "not_provided",
        }
    )
    return df


def load_prjna1159109_meta(filereport_path: Path, biosample_attributes_path: Path) -> pd.DataFrame:
    fr = pd.read_csv(filereport_path, sep="\t")
    need_fr = {"run_accession", "sample_alias"}
    if not need_fr.issubset(set(fr.columns)):
        raise ValueError(f"PRJNA filereport missing required columns: {sorted(need_fr - set(fr.columns))}")
    fr = fr[list(need_fr)].copy()
    fr["project_accession"] = "PRJNA1159109"

    bs = pd.read_csv(biosample_attributes_path, sep="\t")
    need_bs = {"biosample_accession", "sample_alias", "in_vivo_in_vitro", "material", "day", "donor"}
    if not need_bs.issubset(set(bs.columns)):
        raise ValueError(f"biosample_attributes missing required columns: {sorted(need_bs - set(bs.columns))}")

    merged = fr.merge(bs, how="left", on="sample_alias", suffixes=("", "_bs"))
    missing = merged["biosample_accession"].isna().sum()
    if missing:
        raise ValueError(f"Missing biosample attributes for {missing} PRJNA samples")

    out = pd.DataFrame(
        {
            "sample_alias": merged["sample_alias"].astype(str),
            "project_accession": merged["project_accession"].astype(str),
            "run_accession": merged["run_accession"].astype(str),
            "biosample_accession": merged["biosample_accession"].astype(str),
            "in_vivo_in_vitro": merged["in_vivo_in_vitro"].astype(str),
            "material": merged["material"].astype(str),
            "day": merged["day"].astype(str),
            "donor": merged["donor"].astype(str),
            "patient": "not_provided",
            "week": "not_provided",
            "title": merged.get("title", "not_provided").astype(str),
            "collection_date": merged.get("collection_date", "not_provided").astype(str),
            "env_local_scale": merged.get("env_local_scale", "not_provided").astype(str),
            "env_medium": merged.get("env_medium", "not_provided").astype(str),
            "geo_loc_name": merged.get("geo_loc_name", "not_provided").astype(str),
        }
    )
    return out


def schema_diff(left: list[str], right: list[str]) -> dict[str, Any]:
    return {
        "missing_in_output": sorted(set(left) - set(right)),
        "extra_in_output": sorted(set(right) - set(left)),
    }


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prjeb-manifest", type=Path, default=Path(__file__).parent / "dieckow_manifest.tsv")
    ap.add_argument("--prjna-filereport", type=Path, default=Path(__file__).parent / "data" / "PRJNA1159109_filereport.tsv")
    ap.add_argument("--prjna-biosample", type=Path, default=Path(__file__).parent / "data" / "biosample_attributes.tsv")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "combined_meta.tsv")
    ap.add_argument("--schema-log", type=Path, default=Path(__file__).parent / "data" / "schema_diff.json")
    args = ap.parse_args()

    prjeb = load_prjeb71108_from_manifest(args.prjeb_manifest)
    prjna = load_prjna1159109_meta(args.prjna_filereport, args.prjna_biosample)

    _assert_unique(prjeb, "sample_alias")
    _assert_unique(prjna, "sample_alias")

    combined = pd.concat([prjeb, prjna], ignore_index=True)
    _assert_unique(combined, "sample_alias")

    combined = _force_columns(combined, CANONICAL_COLUMNS, fill_value="not_provided")

    missing_required = {c: int((combined[c].isna() | (combined[c] == "")).sum()) for c in REQUIRED_COLUMNS}
    if any(v != 0 for v in missing_required.values()):
        raise RuntimeError(f"Required columns contain missing values: {missing_required}")

    args.schema_log.parent.mkdir(parents=True, exist_ok=True)
    args.schema_log.write_text(json.dumps(schema_diff(CANONICAL_COLUMNS, list(combined.columns)), indent=2) + "\n")

    write_tsv(combined, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
