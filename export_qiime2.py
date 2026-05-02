#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


class ExportQiime2Error(RuntimeError):
    pass


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_cell(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS_RE.sub("", s)
    s = s.strip()
    if s in {"", "not_provided"}:
        return "NA"
    if "\n" in s or "\t" in s:
        s = s.replace("\t", " ").replace("\n", " ")
    return s


def _sanitize_column_name(name: str) -> str:
    name = _sanitize_cell(name)
    if name == "NA":
        name = "col"
    name = name.lower()
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"c_{name}"
    return name


def _infer_numeric_series(values: list[str]) -> list[str] | None:
    non_na = [v for v in values if v != "NA"]
    if not non_na:
        return values
    ints_ok = True
    floats_ok = True
    for v in non_na:
        try:
            f = float(v)
        except Exception:
            ints_ok = False
            floats_ok = False
            break
        if abs(f - int(f)) > 1e-9:
            ints_ok = False
    if not floats_ok:
        return None
    out: list[str] = []
    for v in values:
        if v == "NA":
            out.append("NA")
        else:
            f = float(v)
            out.append(str(int(f)) if ints_ok else str(f))
    return out


def _find_paired_fastq(fastq_dir: Path, run: str, sample: str) -> tuple[Path, Path]:
    f1 = fastq_dir / f"{run}_{sample}_1.fastq.gz"
    f2 = fastq_dir / f"{run}_{sample}_2.fastq.gz"
    if f1.exists() and f2.exists():
        return f1, f2

    cands = list(fastq_dir.glob(f"{run}_{sample}_*.fastq.gz"))
    if not cands:
        cands = list(fastq_dir.glob(f"{run}_*{sample}*_*.fastq.gz"))
    fwd = None
    rev = None
    for p in cands:
        name = p.name
        if name.endswith("_1.fastq.gz") or "_R1_" in name or name.endswith("_R1.fastq.gz"):
            fwd = p
        if name.endswith("_2.fastq.gz") or "_R2_" in name or name.endswith("_R2.fastq.gz"):
            rev = p
    if fwd is None or rev is None:
        raise ExportQiime2Error(f"paired FASTQ not found for run={run} sample={sample} in {fastq_dir}")
    return fwd, rev


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        na_rep="NA",
        quoting=csv.QUOTE_MINIMAL,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-meta", type=Path, default=Path(__file__).parent / "data" / "combined_meta.tsv")
    ap.add_argument("--fastq-dir", type=Path, required=True)
    ap.add_argument("--project", type=str, default="PRJNA1159109")
    ap.add_argument("--out-manifest", type=Path, default=Path(__file__).parent / "data" / "qiime2_manifest.tsv")
    ap.add_argument("--out-metadata", type=Path, default=Path(__file__).parent / "data" / "sample-metadata.tsv")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    combined = args.combined_meta.resolve()
    fastq_dir = args.fastq_dir.resolve()
    if not combined.exists():
        raise ExportQiime2Error(f"combined_meta.tsv not found: {combined}")
    if not fastq_dir.exists():
        raise ExportQiime2Error(f"fastq-dir not found: {fastq_dir}")

    df = pd.read_csv(combined, sep="\t", dtype=str, keep_default_na=False)
    required = {"sample_alias", "project_accession", "run_accession"}
    missing = required - set(df.columns)
    if missing:
        raise ExportQiime2Error(f"combined_meta.tsv missing required columns: {sorted(missing)}")

    df = df[df["project_accession"].astype(str) == args.project].copy()
    if df.empty:
        raise ExportQiime2Error(f"no rows for project={args.project} in {combined}")

    df["sample-id"] = df["sample_alias"].map(_sanitize_cell)
    if df["sample-id"].duplicated().any():
        dups = df.loc[df["sample-id"].duplicated(), "sample-id"].unique().tolist()
        raise ExportQiime2Error(f"duplicate sample-id detected: {dups[:10]}")

    manifest_rows = []
    missing_fastq = []
    for _, r in df.iterrows():
        run = _sanitize_cell(r["run_accession"])
        sid = _sanitize_cell(r["sample-id"])
        if run == "NA" or sid == "NA":
            if args.strict:
                raise ExportQiime2Error("run_accession or sample-id is NA")
            continue
        try:
            fwd, rev = _find_paired_fastq(fastq_dir, run=run, sample=sid)
        except ExportQiime2Error:
            missing_fastq.append({"sample-id": sid, "run_accession": run})
            continue
        manifest_rows.append(
            {
                "sample-id": sid,
                "forward-absolute-filepath": str(fwd.resolve()),
                "reverse-absolute-filepath": str(rev.resolve()),
            }
        )

    if missing_fastq and args.strict:
        raise ExportQiime2Error(f"missing FASTQ for {len(missing_fastq)} samples (e.g. {missing_fastq[:3]})")

    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.empty:
        raise ExportQiime2Error("manifest is empty (no paired FASTQ found)")
    if manifest_df["sample-id"].duplicated().any():
        raise ExportQiime2Error("manifest sample-id is not unique")

    write_tsv(args.out_manifest.resolve(), manifest_df)

    meta = df.copy()
    drop_cols = {"sample_alias"}
    meta = meta[[c for c in meta.columns if c not in drop_cols]]

    cols = list(meta.columns)
    renamed = {}
    for c in cols:
        if c == "sample-id":
            continue
        renamed[c] = _sanitize_column_name(c)
    meta = meta.rename(columns=renamed)
    ordered_cols = ["sample-id"] + [c for c in meta.columns if c != "sample-id"]
    meta = meta[ordered_cols]

    for c in meta.columns:
        meta[c] = meta[c].map(_sanitize_cell)

    for c in meta.columns:
        if c == "sample-id":
            continue
        inferred = _infer_numeric_series(meta[c].astype(str).tolist())
        if inferred is not None:
            meta[c] = inferred

    if set(manifest_df["sample-id"]) != set(meta["sample-id"]):
        only_manifest = sorted(set(manifest_df["sample-id"]) - set(meta["sample-id"]))
        only_meta = sorted(set(meta["sample-id"]) - set(manifest_df["sample-id"]))
        if args.strict:
            raise ExportQiime2Error(
                f"sample mismatch manifest vs metadata: only_manifest={len(only_manifest)} only_meta={len(only_meta)}"
            )

    write_tsv(args.out_metadata.resolve(), meta)

    report = {
        "combined_meta": str(combined),
        "fastq_dir": str(fastq_dir),
        "project": args.project,
        "n_project_rows": int(df.shape[0]),
        "n_manifest_samples": int(manifest_df.shape[0]),
        "n_metadata_samples": int(meta.shape[0]),
        "n_missing_fastq": int(len(missing_fastq)),
        "missing_fastq_examples": missing_fastq[:5],
        "out_manifest": str(args.out_manifest.resolve()),
        "out_metadata": str(args.out_metadata.resolve()),
    }
    report_path = args.out_manifest.resolve().with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

