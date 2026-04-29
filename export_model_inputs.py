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

GUILD_MAP = {
    "Actinomyces": "Actinobacteria",
    "Bifidobacterium": "Actinobacteria",
    "Corynebacterium": "Actinobacteria",
    "Rothia": "Actinobacteria",
    "Slackia": "Actinobacteria",
    "Abiotrophia": "Bacilli",
    "Aerococcus": "Bacilli",
    "Gemella": "Bacilli",
    "Granulicatella": "Bacilli",
    "Lacticaseibacillus": "Bacilli",
    "Lactiplantibacillus": "Bacilli",
    "Limosilactobacillus": "Bacilli",
    "Streptococcus": "Bacilli",
    "Alloprevotella": "Bacteroidia",
    "Porphyromonas": "Bacteroidia",
    "Prevotella": "Bacteroidia",
    "Prevotella_7": "Bacteroidia",
    "Tannerella": "Bacteroidia",
    "Aggregatibacter": "Betaproteobacteria",
    "Cardiobacterium": "Betaproteobacteria",
    "Eikenella": "Betaproteobacteria",
    "Kingella": "Betaproteobacteria",
    "Neisseria": "Betaproteobacteria",
    "Anaerococcus": "Clostridia",
    "Catonella": "Clostridia",
    "Finegoldia": "Clostridia",
    "Johnsonella": "Clostridia",
    "Lachnoanaerobaculum": "Clostridia",
    "Mogibacterium": "Clostridia",
    "Oribacterium": "Clostridia",
    "Parvimonas": "Clostridia",
    "Peptoniphilus": "Clostridia",
    "Peptostreptococcus": "Clostridia",
    "Solobacterium": "Clostridia",
    "Stomatobaculum": "Clostridia",
    "Atopobium": "Coriobacteriia",
    "Cryptobacterium": "Coriobacteriia",
    "Olsenella": "Coriobacteriia",
    "Fusobacterium": "Fusobacteriia",
    "Leptotrichia": "Fusobacteriia",
    "Capnocytophaga": "Flavobacteriia",
    "Bergeyella": "Flavobacteriia",
    "Riemerella": "Flavobacteriia",
    "Haemophilus": "Gammaproteobacteria",
    "Pseudomonas": "Gammaproteobacteria",
    "Centipeda": "Negativicutes",
    "Dialister": "Negativicutes",
    "Megasphaera": "Negativicutes",
    "Selenomonas": "Negativicutes",
    "Veillonella": "Negativicutes",
    "Campylobacter": "Other",
    "Treponema": "Other",
    "Shuttleworthia": "Other",
    "Acanthostaurus": "Other",
    "P5D1-392": "Other",
}


class ModelInputError(RuntimeError):
    pass


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS_RE.sub("", s)
    return s.strip()


def read_feature_table_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ModelInputError(f"feature-table.tsv not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(lines[:200]):
        if line.startswith("#OTU ID") or line.startswith("OTU ID") or line.startswith("#OTU\t") or line.startswith("#OTU_ID"):
            header_idx = i
            break
        if line.startswith("#OTU"):
            header_idx = i
            break
        if line.startswith("#") and "\t" in line and "OTU" in line:
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(lines[:50]):
            if not line.startswith("#") and "\t" in line:
                header_idx = i
                break
    if header_idx is None:
        raise ModelInputError(f"Could not find header line in feature table: {path}")

    header = lines[header_idx].lstrip("#").split("\t")
    if not header or header[0].strip().lower() not in {"otu id", "feature id", "feature-id", "otu_id"}:
        header[0] = "feature-id"
    data_lines = lines[header_idx + 1 :]
    rows = []
    for line in data_lines:
        if not line.strip() or line.startswith("#"):
            continue
        rows.append(line.split("\t"))
    df = pd.DataFrame(rows, columns=header)
    fid_col = header[0]
    df = df.rename(columns={fid_col: "feature-id"})
    df = df.set_index("feature-id")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def read_qiime2_taxonomy_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ModelInputError(f"taxonomy.tsv not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    need = {"Feature ID", "Taxon"}
    if not need.issubset(set(df.columns)):
        raise ModelInputError(f"taxonomy.tsv missing required columns: {sorted(need - set(df.columns))}")
    return df


def parse_genus_from_taxon(taxon: str) -> str:
    t = _sanitize_text(taxon)
    if not t or t.lower() in {"unassigned", "na"}:
        return "Unassigned"
    parts = [p.strip() for p in t.split(";") if p.strip()]
    genus = None
    for p in parts:
        if p.startswith("g__"):
            genus = p[3:].strip()
    if genus and genus not in {"", "uncultured", "unidentified"}:
        return genus
    for p in reversed(parts):
        if "__" in p:
            val = p.split("__", 1)[1].strip()
            if val:
                return val
    return "Unassigned"


def build_feature_to_genus(taxonomy_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, r in taxonomy_df.iterrows():
        fid = _sanitize_text(r.get("Feature ID"))
        genus = parse_genus_from_taxon(r.get("Taxon", ""))
        if fid:
            out[fid] = genus
    return out


def collapse_table(table: pd.DataFrame, feature_to_group: dict[str, str], group_name: str) -> pd.DataFrame:
    groups: dict[str, pd.Series] = {}
    for fid, row in table.iterrows():
        g = feature_to_group.get(str(fid), "Unassigned")
        if g not in groups:
            groups[g] = row.copy()
        else:
            groups[g] = groups[g] + row
    out = pd.DataFrame(groups).T
    out.index.name = group_name
    return out


def compute_time_days(meta: pd.DataFrame) -> pd.Series:
    day_col = "day" if "day" in meta.columns else None
    week_col = "week" if "week" in meta.columns else None

    t_days = pd.Series(["NA"] * meta.shape[0], index=meta.index, dtype=object)

    def _to_int(v: str) -> int | None:
        v = _sanitize_text(v)
        if v in {"", "NA", "not_provided"}:
            return None
        try:
            return int(float(v))
        except Exception:
            return None

    if day_col is not None:
        for i, v in enumerate(meta[day_col].astype(str).tolist()):
            iv = _to_int(v)
            if iv is not None:
                t_days.iat[i] = str(iv)

    if week_col is not None:
        for i, v in enumerate(meta[week_col].astype(str).tolist()):
            if t_days.iat[i] != "NA":
                continue
            iv = _to_int(v)
            if iv is not None:
                t_days.iat[i] = str(iv * 7)

    return t_days


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n", na_rep="NA", quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-table", type=Path, required=True, help="QIIME2 exported feature-table.tsv (biom converted).")
    ap.add_argument("--taxonomy", type=Path, required=True, help="QIIME2 exported taxonomy.tsv.")
    ap.add_argument("--metadata", type=Path, required=True, help="QIIME2 sample-metadata.tsv.")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "data" / "model_inputs")
    ap.add_argument("--split-test", type=str, default="in_vivo", help="Use in_vivo_in_vitro==this as test split.")
    args = ap.parse_args()

    ft = read_feature_table_tsv(args.feature_table)
    tax = read_qiime2_taxonomy_tsv(args.taxonomy)
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, keep_default_na=False)
    if "sample-id" not in meta.columns:
        raise ModelInputError("metadata missing sample-id")
    meta = meta.copy()
    meta["sample-id"] = meta["sample-id"].map(_sanitize_text)
    if meta["sample-id"].duplicated().any():
        dup = meta.loc[meta["sample-id"].duplicated(), "sample-id"].unique().tolist()
        raise ModelInputError(f"duplicate sample-id in metadata: {dup[:10]}")

    samples_in_table = list(ft.columns.astype(str))
    if set(samples_in_table) != set(meta["sample-id"].astype(str).tolist()):
        only_table = sorted(set(samples_in_table) - set(meta["sample-id"]))
        only_meta = sorted(set(meta["sample-id"]) - set(samples_in_table))
        raise ModelInputError(f"sample-id mismatch between feature table and metadata: only_table={len(only_table)} only_meta={len(only_meta)}")

    meta = meta.set_index("sample-id").loc[samples_in_table].reset_index()
    meta["t_days"] = compute_time_days(meta)

    feature_to_genus = build_feature_to_genus(tax)
    genus_table = collapse_table(ft, feature_to_genus, group_name="genus")

    genus_to_guild = {g: GUILD_MAP.get(g, "Other") for g in genus_table.index.astype(str)}
    guild_table = collapse_table(genus_table, genus_to_guild, group_name="guild")

    long_rows = []
    for sample in samples_in_table:
        total = float(ft[sample].sum())
        if total <= 0:
            continue
        trow = meta[meta["sample-id"] == sample].iloc[0].to_dict()
        for genus, cnt in genus_table[sample].items():
            c = float(cnt)
            if c <= 0:
                continue
            long_rows.append(
                {
                    "sample-id": sample,
                    "t_days": trow.get("t_days", "NA"),
                    "taxon_level": "genus",
                    "taxon": str(genus),
                    "count": c,
                    "rel_abundance": c / total,
                    "project_accession": trow.get("project_accession", "NA"),
                    "in_vivo_in_vitro": trow.get("in_vivo_in_vitro", "NA"),
                    "material": trow.get("material", "NA"),
                    "day": trow.get("day", "NA"),
                    "week": trow.get("week", "NA"),
                    "donor": trow.get("donor", "NA"),
                    "patient": trow.get("patient", "NA"),
                    "run_accession": trow.get("run_accession", "NA"),
                }
            )

    trajectories = pd.DataFrame(long_rows)
    if trajectories.empty:
        raise ModelInputError("no trajectories generated (empty feature table?)")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    write_tsv(outdir / "trajectories.tsv", trajectories)

    obs_map = {
        "feature_to_genus": feature_to_genus,
        "genus_to_guild": genus_to_guild,
        "taxon_level": "genus",
        "guilds": sorted(set(genus_to_guild.values())),
    }
    (outdir / "observation_map.json").write_text(json.dumps(obs_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split = []
    if "in_vivo_in_vitro" in meta.columns:
        for _, r in meta.iterrows():
            sid = r["sample-id"]
            viv = _sanitize_text(r.get("in_vivo_in_vitro", "NA"))
            split.append({"sample-id": sid, "split": "test" if viv == args.split_test else "train"})
    else:
        for sid in samples_in_table:
            split.append({"sample-id": sid, "split": "train"})
    split_df = pd.DataFrame(split)
    write_tsv(outdir / "split_specs.tsv", split_df)
    (outdir / "split_specs.json").write_text(
        json.dumps(
            {
                "rule": {"test_if_in_vivo_in_vitro_equals": args.split_test},
                "counts": split_df["split"].value_counts().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "n_samples": int(len(samples_in_table)),
        "n_features": int(ft.shape[0]),
        "n_genera": int(genus_table.shape[0]),
        "n_guilds": int(guild_table.shape[0]),
        "trajectories_rows": int(trajectories.shape[0]),
        "metadata_columns": int(meta.shape[1]),
    }
    (outdir / "export_model_inputs_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
