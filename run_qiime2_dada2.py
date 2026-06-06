#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


class Qiime2RunError(RuntimeError):
    pass


def _docker() -> str:
    d = shutil.which("docker")
    if d is None:
        raise Qiime2RunError("docker not found")
    return d


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_checked(cmd: list[str], label: str) -> None:
    p = _run(cmd)
    if p.returncode != 0:
        raise Qiime2RunError(
            f"{label} failed\ncmd: {' '.join(cmd)}\nstdout:\n{(p.stdout or '')[-4000:]}\nstderr:\n{(p.stderr or '')[-4000:]}"
        )


def _write_container_manifest(host_manifest: Path, workdir: Path, out_path: Path) -> None:
    lines = host_manifest.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise Qiime2RunError(f"empty manifest: {host_manifest}")
    header = lines[0].split("\t")
    required = {"sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"}
    if set(header) != required:
        raise Qiime2RunError(
            f"unexpected manifest header: {header} (expected exactly {sorted(required)})"
        )
    idx = {h: i for i, h in enumerate(header)}
    out_lines = ["\t".join(header)]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        fwd = Path(parts[idx["forward-absolute-filepath"]]).resolve()
        rev = Path(parts[idx["reverse-absolute-filepath"]]).resolve()
        for p in (fwd, rev):
            if not p.exists():
                raise Qiime2RunError(f"FASTQ not found: {p}")
            try:
                rel = os.path.relpath(p, workdir)
            except Exception as e:
                raise Qiime2RunError(f"FASTQ path must be under workdir={workdir}: {p}") from e
            if rel.startswith(".."):
                raise Qiime2RunError(f"FASTQ path must be under workdir={workdir}: {p}")
        parts[idx["forward-absolute-filepath"]] = f"/data/{os.path.relpath(fwd, workdir)}"
        parts[idx["reverse-absolute-filepath"]] = f"/data/{os.path.relpath(rev, workdir)}"
        out_lines.append("\t".join(parts))
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "data" / "qiime2_manifest.tsv")
    ap.add_argument("--metadata", type=Path, default=Path(__file__).parent / "data" / "sample-metadata.tsv")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "data" / "qiime2_out")
    ap.add_argument("--container", type=str, default="quay.io/qiime2/amplicon:2024.2")
    ap.add_argument("--trunc-len-f", type=int, default=0)
    ap.add_argument("--trunc-len-r", type=int, default=0)
    ap.add_argument("--trim-left-f", type=int, default=0)
    ap.add_argument("--trim-left-r", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    workdir = args.workdir.resolve()
    manifest = args.manifest.resolve()
    metadata = args.metadata.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise Qiime2RunError(f"manifest not found: {manifest}")
    if not metadata.exists():
        raise Qiime2RunError(f"metadata not found: {metadata}")

    try:
        rel_metadata = os.path.relpath(metadata, workdir)
        if rel_metadata.startswith(".."):
            raise Qiime2RunError(f"metadata must be under workdir={workdir}: {metadata}")
    except Exception as e:
        raise Qiime2RunError(f"metadata must be under workdir={workdir}: {metadata}") from e

    man_df = pd.read_csv(manifest, sep="\t", dtype=str)
    meta_df = pd.read_csv(metadata, sep="\t", dtype=str)
    if "sample-id" not in man_df.columns or "sample-id" not in meta_df.columns:
        raise Qiime2RunError("manifest and metadata must contain a sample-id column")
    man_ids = set(man_df["sample-id"].astype(str))
    meta_ids = set(meta_df["sample-id"].astype(str))
    if man_ids != meta_ids:
        only_manifest = sorted(man_ids - meta_ids)
        only_meta = sorted(meta_ids - man_ids)
        raise Qiime2RunError(
            f"sample-id mismatch: only_in_manifest={only_manifest[:5]} only_in_metadata={only_meta[:5]}"
        )

    container_manifest = outdir / "qiime2_manifest.container.tsv"
    _write_container_manifest(manifest, workdir=workdir, out_path=container_manifest)

    rel_container_manifest = os.path.relpath(container_manifest, workdir)
    rel_outdir = os.path.relpath(outdir, workdir)

    report: dict[str, Any] = {
        "workdir": str(workdir),
        "container": args.container,
        "inputs": {"manifest": str(manifest), "metadata": str(metadata)},
        "outdir": str(outdir),
        "container_manifest": str(container_manifest),
        "params": {
            "trim_left_f": args.trim_left_f,
            "trim_left_r": args.trim_left_r,
            "trunc_len_f": args.trunc_len_f,
            "trunc_len_r": args.trunc_len_r,
            "n_threads": args.n_threads,
        },
    }
    report_path = outdir / "qiime2_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        _docker()
        print(json.dumps({"status": "dry_run", "report": str(report_path)}))
        return 0

    docker = _docker()
    base = [docker, "run", "--rm", "-v", f"{workdir}:/data", "-w", "/data", args.container]

    _run_checked(
        base
        + [
            "qiime",
            "metadata",
            "tabulate",
            "--m-input-file",
            f"/data/{rel_metadata}",
            "--o-visualization",
            f"/data/{rel_outdir}/sample-metadata.qzv",
        ],
        "qiime metadata tabulate",
    )

    demux_qza = f"/data/{rel_outdir}/demux.qza"
    _run_checked(
        base
        + [
            "qiime",
            "tools",
            "import",
            "--type",
            "SampleData[PairedEndSequencesWithQuality]",
            "--input-path",
            f"/data/{rel_container_manifest}",
            "--output-path",
            demux_qza,
            "--input-format",
            "PairedEndFastqManifestPhred33V2",
        ],
        "qiime import",
    )
    _run_checked(base + ["qiime", "tools", "peek", demux_qza], "qiime tools peek (demux)")
    _run_checked(
        base
        + [
            "qiime",
            "demux",
            "summarize",
            "--i-data",
            demux_qza,
            "--o-visualization",
            f"/data/{rel_outdir}/demux.qzv",
        ],
        "qiime demux summarize",
    )

    dada2_cmd = [
        "qiime",
        "dada2",
        "denoise-paired",
        "--i-demultiplexed-seqs",
        demux_qza,
        "--p-trunc-len-f",
        str(args.trunc_len_f),
        "--p-trunc-len-r",
        str(args.trunc_len_r),
        "--o-table",
        f"/data/{rel_outdir}/table.qza",
        "--o-representative-sequences",
        f"/data/{rel_outdir}/rep-seqs.qza",
        "--o-denoising-stats",
        f"/data/{rel_outdir}/denoising-stats.qza",
    ]
    if args.trim_left_f:
        dada2_cmd += ["--p-trim-left-f", str(args.trim_left_f)]
    if args.trim_left_r:
        dada2_cmd += ["--p-trim-left-r", str(args.trim_left_r)]
    if args.n_threads:
        dada2_cmd += ["--p-n-threads", str(args.n_threads)]
    _run_checked(base + dada2_cmd, "qiime dada2 denoise-paired")

    _run_checked(
        base
        + [
            "qiime",
            "tools",
            "export",
            "--input-path",
            f"/data/{rel_outdir}/table.qza",
            "--output-path",
            f"/data/{rel_outdir}/exported_table",
        ],
        "qiime export table",
    )
    _run_checked(
        base
        + [
            "qiime",
            "tools",
            "export",
            "--input-path",
            f"/data/{rel_outdir}/rep-seqs.qza",
            "--output-path",
            f"/data/{rel_outdir}/exported_rep_seqs",
        ],
        "qiime export rep-seqs",
    )
    _run_checked(
        base
        + [
            "qiime",
            "tools",
            "export",
            "--input-path",
            f"/data/{rel_outdir}/denoising-stats.qza",
            "--output-path",
            f"/data/{rel_outdir}/exported_denoising_stats",
        ],
        "qiime export denoising-stats",
    )

    biom_path = outdir / "exported_table" / "feature-table.biom"
    if biom_path.exists():
        p = _run(
            base
            + [
                "bash",
                "-lc",
                f"biom convert -i /data/{rel_outdir}/exported_table/feature-table.biom -o /data/{rel_outdir}/exported_table/feature-table.tsv --to-tsv",
            ]
        )
        if p.returncode != 0:
            (outdir / "biom_convert.stderr.txt").write_text((p.stderr or "")[-8000:], encoding="utf-8")

    report2 = json.loads(report_path.read_text(encoding="utf-8"))
    report2["results"] = {
        "demux_qza": str((outdir / "demux.qza").resolve()),
        "table_qza": str((outdir / "table.qza").resolve()),
        "rep_seqs_qza": str((outdir / "rep-seqs.qza").resolve()),
        "denoising_stats_qza": str((outdir / "denoising-stats.qza").resolve()),
        "feature_table_biom": str(biom_path.resolve()) if biom_path.exists() else "NA",
        "feature_table_tsv": str((outdir / "exported_table" / "feature-table.tsv").resolve())
        if (outdir / "exported_table" / "feature-table.tsv").exists()
        else "NA",
        "rep_seqs_fasta": str((outdir / "exported_rep_seqs" / "dna-sequences.fasta").resolve())
        if (outdir / "exported_rep_seqs" / "dna-sequences.fasta").exists()
        else "NA",
    }
    report_path.write_text(json.dumps(report2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
