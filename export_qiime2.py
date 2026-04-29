#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ExportError(RuntimeError):
    pass


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SAFE_COL_RE = re.compile(r"[^A-Za-z0-9_\\-]+")


def _sanitize_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS_RE.sub("", s)
    return s


def _sanitize_column_name(name: str) -> str:
    name = _sanitize_text(name).strip()
    name = name.replace(" ", "_")
    name = _SAFE_COL_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"c_{name}"
    return name


@dataclass(frozen=True)
class PairedFastq:
    sample_id: str
    forward: Path
    reverse: Path


def _detect_prjna_pair(fastq_dir: Path, run_accession: str, sample_id: str) -> PairedFastq | None:
    f1 = fastq_dir / f"{run_accession}_{sample_id}_1.fastq.gz"
    f2 = fastq_dir / f"{run_accession}_{sample_id}_2.fastq.gz"
    if f1.exists() and f2.exists():
        return PairedFastq(sample_id=sample_id, forward=f1, reverse=f2)
    return None


def _assert_utf8_no_control(df: pd.DataFrame) -> None:
    for c in df.columns:
        if _CONTROL_CHARS_RE.search(str(c)):
            raise ExportError(f"Control character found in column name: {c!r}")
    for c in df.columns:
        ser = df[c].astype(str)
        bad = ser[ser.str.contains(_CONTROL_CHARS_RE, regex=True)]
        if not bad.empty:
            ex = bad.iloc[0]
            raise ExportError(f"Control character found in column {c!r}, example value: {ex!r}")


def _write_manifest(path: Path, pairs: list[PairedFastq]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["sample-id", "absolute-filepath", "direction"])
        for p in pairs:
            w.writerow([p.sample_id, str(p.forward.resolve()), "forward"])
            w.writerow([p.sample_id, str(p.reverse.resolve()), "reverse"])


def _write_metadata(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n", na_rep="NA", quoting=csv.QUOTE_MINIMAL)


def _run_qiime_validate(metadata_path: Path) -> dict[str, Any]:
    qiime = shutil.which("qiime")
    if qiime is None:
        return {"status": "skipped", "reason": "qiime_not_found"}
    out_qzv = Path("/tmp") / f"metadata_{os.getpid()}.qzv"
    cmd = [
        qiime,
        "metadata",
        "tabulate",
        "--m-input-file",
        str(metadata_path),
        "--o-visualization",
        str(out_qzv),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if out_qzv.exists():
        out_qzv.unlink(missing_ok=True)
    if p.returncode != 0:
        return {"status": "error", "returncode": p.returncode, "stderr": p.stderr[-2000:], "stdout": p.stdout[-2000:]}
    return {"status": "ok"}


def _run_dada2_smoke(manifest_path: Path) -> dict[str, Any]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {"status": "skipped", "reason": "Rscript_not_found"}
    script = r"""
args <- commandArgs(trailingOnly=TRUE)
man <- read.delim(args[1], sep="\t", header=TRUE, stringsAsFactors=FALSE)
stopifnot(all(c("sample-id","absolute-filepath","direction") %in% colnames(man)))
fwd <- man[man$direction=="forward","absolute-filepath"][1]
rev <- man[man$direction=="reverse","absolute-filepath"][1]
stopifnot(file.exists(fwd), file.exists(rev))
suppressWarnings({
  ok1 <- FALSE
  ok2 <- FALSE
  if (requireNamespace("ShortRead", quietly=TRUE)) {
    fq1 <- ShortRead::readFastq(fwd, n=1)
    fq2 <- ShortRead::readFastq(rev, n=1)
    ok1 <- length(fq1) == 1
    ok2 <- length(fq2) == 1
  }
  if (!ok1 || !ok2) {
    quit(status=2)
  }
})
cat("ok\n")
"""
    p = subprocess.run([rscript, "-e", script, str(manifest_path)], capture_output=True, text=True)
    if p.returncode != 0:
        return {"status": "error", "returncode": p.returncode, "stderr": p.stderr[-2000:], "stdout": p.stdout[-2000:]}
    return {"status": "ok"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--combined-meta",
        type=Path,
        default=Path(__file__).parent / "data" / "combined_meta.tsv",
    )
    ap.add_argument(
        "--fastq-prjna-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "PRJNA1159109_fastq",
    )
    ap.add_argument(
        "--out-manifest",
        type=Path,
        default=Path(__file__).parent / "data" / "qiime2_manifest.tsv",
    )
    ap.add_argument(
        "--out-metadata",
        type=Path,
        default=Path(__file__).parent / "data" / "sample-metadata.tsv",
    )
    ap.add_argument("--project", type=str, default="PRJNA1159109", help="Filter to this project_accession.")
    ap.add_argument("--allow-missing-fastq", action="store_true")
    ap.add_argument("--external-validate", action="store_true")
    ap.add_argument("--strict-external", action="store_true")
    ap.add_argument("--report-json", type=Path, default=Path(__file__).parent / "data" / "export_qiime2_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.combined_meta, sep="\t", dtype=str, keep_default_na=False)
    required = {"sample_alias", "project_accession", "run_accession"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ExportError(f"combined_meta.tsv missing required columns: {missing_cols}")

    df = df.copy()
    df["sample_alias"] = df["sample_alias"].map(_sanitize_text).str.strip()
    df["project_accession"] = df["project_accession"].map(_sanitize_text).str.strip()
    df["run_accession"] = df["run_accession"].map(_sanitize_text).str.strip()

    df = df[df["project_accession"] == args.project].copy()
    if df.empty:
        raise ExportError(f"No records found for project_accession={args.project}")

    if df["sample_alias"].duplicated().any():
        dup = df.loc[df["sample_alias"].duplicated(), "sample_alias"].unique().tolist()
        raise ExportError(f"Duplicate sample_alias detected: {dup[:10]}")

    pairs: list[PairedFastq] = []
    missing_fastq: list[dict[str, str]] = []
    for _, row in df.iterrows():
        sample_id = row["sample_alias"]
        run = row["run_accession"]
        p = _detect_prjna_pair(args.fastq_prjna_dir, run, sample_id)
        if p is None:
            missing_fastq.append({"sample_id": sample_id, "run_accession": run, "fastq_dir": str(args.fastq_prjna_dir)})
            continue
        pairs.append(p)

    if missing_fastq and not args.allow_missing_fastq:
        raise ExportError(f"Missing paired FASTQ for {len(missing_fastq)} samples (use --allow-missing-fastq to drop).")

    kept_ids = {p.sample_id for p in pairs}
    df_kept = df[df["sample_alias"].isin(kept_ids)].copy()

    if df_kept.empty:
        raise ExportError("No samples with paired FASTQ found; manifest would be empty.")

    col_map: dict[str, str] = {}
    out_cols: list[str] = ["sample-id"]
    for c in df_kept.columns:
        if c == "sample_alias":
            continue
        out_c = _sanitize_column_name(c)
        if out_c == "sample-id":
            out_c = "sample_id_meta"
        i = 2
        base = out_c
        while out_c in out_cols:
            out_c = f"{base}_{i}"
            i += 1
        col_map[c] = out_c
        out_cols.append(out_c)

    meta = pd.DataFrame()
    meta["sample-id"] = df_kept["sample_alias"].map(_sanitize_text).str.strip()
    for c, out_c in col_map.items():
        meta[out_c] = df_kept[c].map(_sanitize_text).str.strip()

    for c in meta.columns:
        meta[c] = meta[c].replace("", "NA")
        meta[c] = meta[c].fillna("NA")

    _assert_utf8_no_control(meta)

    numeric_candidates = {"day", "week"}
    for src, out_c in col_map.items():
        if src in numeric_candidates:
            v = meta[out_c].replace("NA", pd.NA)
            as_num = pd.to_numeric(v, errors="coerce")
            if as_num.notna().any():
                meta[out_c] = as_num

    _write_manifest(args.out_manifest, pairs)
    _write_metadata(args.out_metadata, meta)

    missing_manifest_paths = []
    for p in pairs:
        if not p.forward.exists():
            missing_manifest_paths.append(str(p.forward))
        if not p.reverse.exists():
            missing_manifest_paths.append(str(p.reverse))
    if missing_manifest_paths:
        raise ExportError(f"Manifest contains missing file paths (count={len(missing_manifest_paths)}), example={missing_manifest_paths[0]}")

    report: dict[str, Any] = {
        "input_combined_meta": str(args.combined_meta),
        "project": args.project,
        "n_input_rows_project": int(df.shape[0]),
        "n_kept_for_manifest": int(df_kept.shape[0]),
        "n_dropped_missing_fastq": int(len(missing_fastq)),
        "manifest_path": str(args.out_manifest),
        "metadata_path": str(args.out_metadata),
        "metadata_columns": int(meta.shape[1]),
        "missing_fastq_samples": missing_fastq[:20],
        "external_validation": {},
    }

    if args.external_validate:
        qiime_res = _run_qiime_validate(args.out_metadata)
        dada2_res = _run_dada2_smoke(args.out_manifest)
        report["external_validation"] = {"qiime2": qiime_res, "dada2": dada2_res}
        if args.strict_external:
            if qiime_res.get("status") != "ok":
                raise ExportError(f"QIIME2 validation failed: {qiime_res}")
            if dada2_res.get("status") != "ok":
                raise ExportError(f"DADA2 smoke test failed: {dada2_res}")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ["project", "n_input_rows_project", "n_kept_for_manifest", "n_dropped_missing_fastq", "metadata_columns"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
