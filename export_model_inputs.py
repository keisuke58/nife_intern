#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


class ExportModelInputsError(RuntimeError):
    pass


_GUILD_MAP: dict[str, str] = {
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


def read_qiime2_feature_table_tsv(path: Path) -> pd.DataFrame:
    path = path.resolve()
    if not path.exists():
        raise ExportModelInputsError(f"feature-table.tsv not found: {path}")
    header_i = None
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#OTU ID") or s.startswith("OTU ID") or s.startswith("#Feature ID") or s.startswith("Feature ID"):
                header_i = i
                break
    if header_i is None:
        raise ExportModelInputsError(f"could not find header row in feature-table.tsv: {path}")
    df = pd.read_csv(path, sep="\t", skiprows=header_i, dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise ExportModelInputsError(f"unexpected feature-table.tsv shape: {df.shape}")
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "feature-id"})
    if "taxonomy" in [c.lower() for c in df.columns]:
        drop = [c for c in df.columns if c.lower() == "taxonomy"]
        df = df.drop(columns=drop)
    df = df.set_index("feature-id")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _clean_rank_token(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    if "__" in token:
        token = token.split("__", 1)[1]
    token = token.strip()
    if token in {"", "uncultured", "unidentified", "unassigned"}:
        return ""
    return token


def genus_from_taxon(taxon: str) -> str:
    s = "" if taxon is None else str(taxon)
    parts = [p.strip() for p in s.split(";")]
    if any(p.startswith("g__") for p in parts):
        for p in parts:
            if p.startswith("g__"):
                g = _clean_rank_token(p)
                return g if g else "Unassigned"
        return "Unassigned"
    for p in parts:
        if p.startswith("D_5__"):
            g = _clean_rank_token(p)
            return g if g else "Unassigned"
    return "Unassigned"


def read_qiime2_taxonomy_tsv(path: Path) -> pd.DataFrame:
    path = path.resolve()
    if not path.exists():
        raise ExportModelInputsError(f"taxonomy.tsv not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    cols = {c.lower(): c for c in df.columns}
    if "feature id" not in cols:
        raise ExportModelInputsError(f"taxonomy.tsv missing 'Feature ID' column: {path}")
    if "taxon" not in cols:
        raise ExportModelInputsError(f"taxonomy.tsv missing 'Taxon' column: {path}")
    df = df.rename(columns={cols["feature id"]: "feature-id", cols["taxon"]: "taxon"})
    df["genus"] = df["taxon"].map(genus_from_taxon)
    return df[["feature-id", "genus"]]


def _relative_abundance(counts: pd.DataFrame) -> pd.DataFrame:
    col_sums = counts.sum(axis=0)
    col_sums = col_sums.replace(0.0, 1.0)
    return counts.divide(col_sums, axis=1)


def _to_long(abund: pd.DataFrame, metadata: pd.DataFrame, time_col: str, group_cols: list[str], tax_col: str) -> pd.DataFrame:
    md = metadata.copy()
    keep_cols = ["sample-id", time_col] + [c for c in group_cols if c in md.columns]
    md = md[keep_cols].copy()
    md[time_col] = pd.to_numeric(md[time_col], errors="coerce")

    present_group_cols = [c for c in group_cols if c in md.columns]
    if present_group_cols:
        group_series = md[present_group_cols].astype(str).apply(
            lambda r: "|".join([f"{c}={r[c]}" for c in present_group_cols]), axis=1
        )
    else:
        group_series = pd.Series(["group=NA"] * md.shape[0], index=md.index)
    md = md.assign(group=group_series)

    melted = abund.reset_index().melt(id_vars=[tax_col], var_name="sample-id", value_name="rel_abundance")
    out = melted.merge(md, on="sample-id", how="left")
    out = out.rename(columns={tax_col: "taxon"})
    out = out[["sample-id", time_col, "group", "taxon", "rel_abundance"] + present_group_cols]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-table-tsv", type=Path, default=Path(__file__).parent / "data" / "qiime2_out" / "exported_table" / "feature-table.tsv")
    ap.add_argument("--taxonomy-tsv", type=Path, default=Path(__file__).parent / "data" / "qiime2_taxonomy" / "exported_taxonomy" / "taxonomy.tsv")
    ap.add_argument("--metadata", type=Path, default=Path(__file__).parent / "data" / "sample-metadata.tsv")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "data" / "model_inputs")
    ap.add_argument("--time-col", type=str, default="")
    ap.add_argument("--group-cols", type=str, default="material,in_vivo_in_vitro,project_accession")
    args = ap.parse_args()

    feature_table = read_qiime2_feature_table_tsv(args.feature_table_tsv)
    taxonomy = read_qiime2_taxonomy_tsv(args.taxonomy_tsv)

    md_path = args.metadata.resolve()
    if not md_path.exists():
        raise ExportModelInputsError(f"sample-metadata.tsv not found: {md_path}")
    metadata = pd.read_csv(md_path, sep="\t", dtype=str, keep_default_na=False)
    if "sample-id" not in metadata.columns:
        raise ExportModelInputsError("sample-metadata.tsv missing 'sample-id' column")

    time_col = args.time_col.strip()
    if not time_col:
        if "day" in metadata.columns:
            time_col = "day"
        elif "week" in metadata.columns:
            time_col = "week"
        else:
            raise ExportModelInputsError("no time column found (expected 'day' or 'week', or pass --time-col)")
    if time_col not in metadata.columns:
        raise ExportModelInputsError(f"time column not found in metadata: {time_col}")

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    sample_cols = [c for c in feature_table.columns]
    md_samples = set(metadata["sample-id"].astype(str))
    missing_md = [s for s in sample_cols if s not in md_samples]
    if missing_md:
        raise ExportModelInputsError(f"metadata missing {len(missing_md)} samples present in feature table (e.g. {missing_md[:5]})")

    tax_map = taxonomy.set_index("feature-id")["genus"].to_dict()
    if not set(feature_table.index).intersection(set(tax_map.keys())):
        raise ExportModelInputsError("no overlapping features between feature table and taxonomy")

    genus = [tax_map.get(fid, "Unassigned") for fid in feature_table.index]
    genus_counts = feature_table.copy()
    genus_counts["genus"] = genus
    genus_counts = genus_counts.groupby("genus", sort=False).sum(numeric_only=True)

    guild_counts = genus_counts.copy()
    guild_counts["guild"] = [(_GUILD_MAP.get(g, "Other")) for g in guild_counts.index]
    guild_counts = guild_counts.groupby("guild", sort=False).sum(numeric_only=True)

    genus_rel = _relative_abundance(genus_counts)
    guild_rel = _relative_abundance(guild_counts)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    genus_rel_path = outdir / "genus_relative_abundance.wide.tsv"
    guild_rel_path = outdir / "guild_relative_abundance.wide.tsv"
    genus_rel.reset_index().rename(columns={"genus": "taxon"}).to_csv(genus_rel_path, sep="\t", index=False)
    guild_rel.reset_index().rename(columns={"guild": "taxon"}).to_csv(guild_rel_path, sep="\t", index=False)

    genus_long = _to_long(genus_rel.reset_index().rename(columns={"genus": "taxon"}).set_index("taxon"), metadata, time_col, group_cols, "taxon")
    guild_long = _to_long(guild_rel.reset_index().rename(columns={"guild": "taxon"}).set_index("taxon"), metadata, time_col, group_cols, "taxon")
    genus_long_path = outdir / "genus_relative_abundance.long.tsv"
    guild_long_path = outdir / "guild_relative_abundance.long.tsv"
    genus_long.to_csv(genus_long_path, sep="\t", index=False)
    guild_long.to_csv(guild_long_path, sep="\t", index=False)

    report = {
        "inputs": {
            "feature_table_tsv": str(args.feature_table_tsv.resolve()),
            "taxonomy_tsv": str(args.taxonomy_tsv.resolve()),
            "metadata_tsv": str(md_path),
        },
        "params": {"time_col": time_col, "group_cols": group_cols},
        "counts": {
            "n_samples": int(len(sample_cols)),
            "n_features": int(feature_table.shape[0]),
            "n_genera": int(genus_counts.shape[0]),
            "n_guilds": int(guild_counts.shape[0]),
        },
        "outputs": {
            "genus_wide": str(genus_rel_path),
            "guild_wide": str(guild_rel_path),
            "genus_long": str(genus_long_path),
            "guild_long": str(guild_long_path),
        },
    }
    (outdir / "model_inputs.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
