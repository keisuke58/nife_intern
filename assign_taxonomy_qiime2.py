#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class Qiime2TaxonomyError(RuntimeError):
    pass


def _docker() -> str:
    d = shutil.which("docker")
    if d is None:
        raise Qiime2TaxonomyError("docker not found")
    return d


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_checked(cmd: list[str], label: str) -> None:
    p = _run(cmd)
    if p.returncode != 0:
        raise Qiime2TaxonomyError(
            f"{label} failed\ncmd: {' '.join(cmd)}\nstdout:\n{(p.stdout or '')[-4000:]}\nstderr:\n{(p.stderr or '')[-4000:]}"
        )


def _require_under_workdir(workdir: Path, p: Path, label: str) -> str:
    p = p.resolve()
    if not p.exists():
        raise Qiime2TaxonomyError(f"{label} not found: {p}")
    try:
        rel = os.path.relpath(p, workdir)
    except Exception as e:
        raise Qiime2TaxonomyError(f"{label} must be under workdir={workdir}: {p}") from e
    if rel.startswith(".."):
        raise Qiime2TaxonomyError(f"{label} must be under workdir={workdir}: {p}")
    return rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--rep-seqs", type=Path, default=Path(__file__).parent / "data" / "qiime2_out" / "rep-seqs.qza")
    ap.add_argument(
        "--classifier",
        type=Path,
        default=Path(__file__).parent / "data" / "classifiers" / "classifier.qza",
    )
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "data" / "qiime2_taxonomy")
    ap.add_argument("--container", type=str, default="quay.io/qiime2/amplicon:2024.2")
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    workdir = args.workdir.resolve()
    rep_seqs = args.rep_seqs.resolve()
    classifier = args.classifier.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rel_rep_seqs = _require_under_workdir(workdir, rep_seqs, "rep-seqs")
    rel_classifier = _require_under_workdir(workdir, classifier, "classifier")
    docker = _docker()

    report: dict[str, Any] = {
        "workdir": str(workdir),
        "container": args.container,
        "inputs": {"rep_seqs": str(rep_seqs), "classifier": str(classifier)},
        "outdir": str(outdir),
        "params": {"n_jobs": args.n_jobs},
    }
    report_path = outdir / "qiime2_taxonomy_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "report": str(report_path)}))
        return 0

    rel_outdir = os.path.relpath(outdir, workdir)
    base = [docker, "run", "--rm", "-v", f"{workdir}:/data", "-w", "/data", args.container]

    _run_checked(
        base + ["qiime", "tools", "peek", f"/data/{rel_rep_seqs}"],
        "qiime tools peek (rep-seqs)",
    )
    _run_checked(
        base + ["qiime", "tools", "peek", f"/data/{rel_classifier}"],
        "qiime tools peek (classifier)",
    )

    taxonomy_qza = f"/data/{rel_outdir}/taxonomy.qza"
    cmd = [
        "qiime",
        "feature-classifier",
        "classify-sklearn",
        "--i-classifier",
        f"/data/{rel_classifier}",
        "--i-reads",
        f"/data/{rel_rep_seqs}",
        "--o-classification",
        taxonomy_qza,
    ]
    if args.n_jobs:
        cmd += ["--p-n-jobs", str(args.n_jobs)]
    _run_checked(base + cmd, "qiime classify-sklearn")

    exported = f"/data/{rel_outdir}/exported_taxonomy"
    _run_checked(
        base + ["qiime", "tools", "export", "--input-path", taxonomy_qza, "--output-path", exported],
        "qiime tools export (taxonomy)",
    )

    taxonomy_tsv = outdir / "exported_taxonomy" / "taxonomy.tsv"
    report2 = json.loads(report_path.read_text(encoding="utf-8"))
    report2["results"] = {
        "taxonomy_qza": str((outdir / "taxonomy.qza").resolve()),
        "taxonomy_tsv": str(taxonomy_tsv.resolve()) if taxonomy_tsv.exists() else "NA",
    }
    report_path.write_text(json.dumps(report2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
