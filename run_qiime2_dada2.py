#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class Qiime2PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    forward: Path
    reverse: Path


def _read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        need = {"sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise Qiime2PipelineError(f"manifest missing required columns: {sorted(need - set(reader.fieldnames or []))}")
        for r in reader:
            sid = (r.get("sample-id") or "").strip()
            fwd = (r.get("forward-absolute-filepath") or "").strip()
            rev = (r.get("reverse-absolute-filepath") or "").strip()
            if not sid or not fwd or not rev:
                raise Qiime2PipelineError(f"invalid manifest row: {r}")
            pf = Path(fwd)
            pr = Path(rev)
            if not pf.is_absolute() or not pr.is_absolute():
                raise Qiime2PipelineError(f"manifest paths must be absolute: {fwd} {rev}")
            rows.append(ManifestRow(sample_id=sid, forward=pf, reverse=pr))
    if not rows:
        raise Qiime2PipelineError("manifest has no rows")
    return rows


def _validate_manifest_files(rows: list[ManifestRow]) -> dict[str, Any]:
    missing_paths: list[str] = []
    sample_ids: set[str] = set()
    for r in rows:
        if r.sample_id in sample_ids:
            raise Qiime2PipelineError(f"duplicate sample-id in manifest: {r.sample_id}")
        sample_ids.add(r.sample_id)
        if not r.forward.exists():
            missing_paths.append(str(r.forward))
        if not r.reverse.exists():
            missing_paths.append(str(r.reverse))
    return {
        "n_rows": len(rows),
        "n_samples": len(sample_ids),
        "missing_paths": missing_paths,
        "missing_pairs": [],
    }


def _read_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "sample-id" not in df.columns:
        raise Qiime2PipelineError("sample-metadata.tsv missing sample-id column")
    if df["sample-id"].duplicated().any():
        dup = df.loc[df["sample-id"].duplicated(), "sample-id"].unique().tolist()
        raise Qiime2PipelineError(f"duplicate sample-id in metadata: {dup[:10]}")
    return df


def _validate_metadata_against_manifest(df: pd.DataFrame, manifest_samples: set[str]) -> dict[str, Any]:
    meta_samples = set(df["sample-id"].astype(str).tolist())
    only_in_manifest = sorted(manifest_samples - meta_samples)
    only_in_meta = sorted(meta_samples - manifest_samples)
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "only_in_manifest": only_in_manifest[:50],
        "only_in_meta": only_in_meta[:50],
        "only_in_manifest_count": len(only_in_manifest),
        "only_in_meta_count": len(only_in_meta),
    }


def _docker_run(workdir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    if docker is None:
        raise Qiime2PipelineError("docker not found")
    cmd = [docker, "run", "--rm", "-v", f"{workdir}:/data", "-w", "/data"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_checked(workdir: Path, container: str, cmd: list[str]) -> None:
    p = _docker_run(workdir, [container] + cmd)
    if p.returncode != 0:
        raise Qiime2PipelineError(
            "command failed: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + (p.stdout[-4000:] if p.stdout else "")
            + "\nstderr:\n"
            + (p.stderr[-4000:] if p.stderr else "")
        )


def _rewrite_manifest_for_container(workdir: Path, manifest_rows: list[ManifestRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="\n", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"])
        for r in manifest_rows:
            try:
                f_rel = r.forward.resolve().relative_to(workdir)
                r_rel = r.reverse.resolve().relative_to(workdir)
            except Exception as e:
                raise Qiime2PipelineError(f"FASTQ path is not under workdir={workdir}: {r.forward} {r.reverse}") from e
            w.writerow([r.sample_id, f"/data/{f_rel.as_posix()}", f"/data/{r_rel.as_posix()}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "data" / "qiime2_manifest.tsv")
    ap.add_argument("--metadata", type=Path, default=Path(__file__).parent / "data" / "sample-metadata.tsv")
    ap.add_argument("--container", type=str, default="quay.io/qiime2/amplicon:2024.2")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "data" / "qiime2_out")
    ap.add_argument("--trim-left-f", type=int, default=0)
    ap.add_argument("--trim-left-r", type=int, default=0)
    ap.add_argument("--trunc-len-f", type=int, default=0)
    ap.add_argument("--trunc-len-r", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    workdir = args.workdir.resolve()
    manifest = args.manifest.resolve()
    metadata = args.metadata.resolve()
    outdir = args.outdir.resolve()

    man_rows = _read_manifest(manifest)
    man_val = _validate_manifest_files(man_rows)
    if man_val["missing_paths"]:
        raise Qiime2PipelineError(f"manifest has missing file paths (count={len(man_val['missing_paths'])})")

    meta = _read_metadata(metadata)
    meta = meta.replace("", "NA").fillna("NA")
    meta_val = _validate_metadata_against_manifest(meta, {r.sample_id for r in man_rows})
    if meta_val["only_in_manifest_count"] or meta_val["only_in_meta_count"]:
        raise Qiime2PipelineError(
            f"metadata/manifest sample-id mismatch: only_in_manifest={meta_val['only_in_manifest_count']} only_in_meta={meta_val['only_in_meta_count']}"
        )

    report: dict[str, Any] = {
        "manifest": str(manifest),
        "metadata": str(metadata),
        "manifest_validation": man_val,
        "metadata_validation": meta_val,
        "container": args.container,
        "outdir": str(outdir),
        "commands": [],
        "results": {},
    }

    rel_manifest = os.path.relpath(manifest, workdir)
    rel_metadata = os.path.relpath(metadata, workdir)
    rel_outdir = os.path.relpath(outdir, workdir)

    outdir.mkdir(parents=True, exist_ok=True)
    demux_qza = f"{rel_outdir}/demux-paired-end.qza"
    table_qza = f"{rel_outdir}/table.qza"
    repseq_qza = f"{rel_outdir}/rep-seqs.qza"
    stats_qza = f"{rel_outdir}/denoising-stats.qza"
    meta_qzv = f"{rel_outdir}/sample-metadata.qzv"

    cmds = [
        ["qiime", "metadata", "tabulate", "--m-input-file", rel_metadata, "--o-visualization", meta_qzv],
        [
            "qiime",
            "tools",
            "import",
            "--type",
            "SampleData[PairedEndSequencesWithQuality]",
            "--input-path",
            rel_manifest,
            "--output-path",
            demux_qza,
            "--input-format",
            "PairedEndFastqManifestPhred33V2",
        ],
        [
            "qiime",
            "dada2",
            "denoise-paired",
            "--i-demultiplexed-seqs",
            demux_qza,
            "--p-trim-left-f",
            str(args.trim_left_f),
            "--p-trim-left-r",
            str(args.trim_left_r),
            "--p-trunc-len-f",
            str(args.trunc_len_f),
            "--p-trunc-len-r",
            str(args.trunc_len_r),
            "--o-table",
            table_qza,
            "--o-representative-sequences",
            repseq_qza,
            "--o-denoising-stats",
            stats_qza,
        ],
        ["qiime", "feature-table", "summarize", "--i-table", table_qza, "--m-sample-metadata-file", rel_metadata, "--o-visualization", f"{rel_outdir}/table.qzv"],
        ["qiime", "feature-table", "tabulate-seqs", "--i-data", repseq_qza, "--o-visualization", f"{rel_outdir}/rep-seqs.qzv"],
        ["qiime", "tools", "export", "--input-path", table_qza, "--output-path", f"{rel_outdir}/exported_table"],
        ["qiime", "tools", "export", "--input-path", repseq_qza, "--output-path", f"{rel_outdir}/exported_rep_seqs"],
    ]

    if args.n_threads and args.n_threads > 0:
        cmds[2].extend(["--p-n-threads", str(args.n_threads)])

    report["commands"] = cmds

    report_path = outdir / "qiime2_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "report": str(report_path)}))
        return 0

    container_manifest = outdir / "manifest_container.tsv"
    _rewrite_manifest_for_container(workdir, man_rows, container_manifest)
    rel_container_manifest = os.path.relpath(container_manifest, workdir)
    cmds[1][cmds[1].index("--input-path") + 1] = rel_container_manifest

    for cmd in cmds:
        _run_checked(workdir, args.container, cmd)

    biom_path = outdir / "exported_table" / "feature-table.biom"
    tsv_path = outdir / "exported_table" / "feature-table.tsv"
    if biom_path.exists():
        _run_checked(
            workdir,
            args.container,
            [
                "bash",
                "-lc",
                f"biom convert -i {os.path.relpath(biom_path, workdir)} -o {os.path.relpath(tsv_path, workdir)} --to-tsv",
            ],
        )

    report2 = json.loads(report_path.read_text(encoding="utf-8"))
    report2["results"] = {
        "demux_qza": str((workdir / demux_qza).resolve()),
        "table_qza": str((workdir / table_qza).resolve()),
        "repseq_qza": str((workdir / repseq_qza).resolve()),
        "stats_qza": str((workdir / stats_qza).resolve()),
        "table_tsv": str(tsv_path.resolve()) if tsv_path.exists() else "not_provided",
    }
    report_path.write_text(json.dumps(report2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
