#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class TaxonomyError(RuntimeError):
    pass


def _docker_run(workdir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    if docker is None:
        raise TaxonomyError("docker not found")
    cmd = [docker, "run", "--rm", "-v", f"{workdir}:/data", "-w", "/data"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_checked(workdir: Path, container: str, cmd: list[str]) -> None:
    p = _docker_run(workdir, [container] + cmd)
    if p.returncode != 0:
        raise TaxonomyError(
            "command failed: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + (p.stdout[-4000:] if p.stdout else "")
            + "\nstderr:\n"
            + (p.stderr[-4000:] if p.stderr else "")
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--rep-seqs-qza", type=Path, required=True)
    ap.add_argument("--classifier-qza", type=Path, required=True)
    ap.add_argument("--container", type=str, default="quay.io/qiime2/amplicon:2024.2")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "data" / "qiime2_taxonomy")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    workdir = args.workdir.resolve()
    rep = args.rep_seqs_qza.resolve()
    clf = args.classifier_qza.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not rep.exists():
        raise TaxonomyError(f"rep-seqs.qza not found: {rep}")
    if not clf.exists():
        raise TaxonomyError(f"classifier.qza not found: {clf}")

    try:
        rel_rep = os.path.relpath(rep, workdir)
        rel_clf = os.path.relpath(clf, workdir)
        rel_outdir = os.path.relpath(outdir, workdir)
    except Exception as e:
        raise TaxonomyError(f"inputs must be under workdir={workdir}") from e

    tax_qza = f"{rel_outdir}/taxonomy.qza"
    tax_qzv = f"{rel_outdir}/taxonomy.qzv"
    export_dir = f"{rel_outdir}/exported_taxonomy"

    cmds = [
        [
            "qiime",
            "feature-classifier",
            "classify-sklearn",
            "--i-classifier",
            rel_clf,
            "--i-reads",
            rel_rep,
            "--o-classification",
            tax_qza,
            "--p-n-jobs",
            str(args.n_jobs),
        ],
        ["qiime", "metadata", "tabulate", "--m-input-file", tax_qza, "--o-visualization", tax_qzv],
        ["qiime", "tools", "export", "--input-path", tax_qza, "--output-path", export_dir],
    ]

    report: dict[str, Any] = {
        "rep_seqs_qza": str(rep),
        "classifier_qza": str(clf),
        "container": args.container,
        "outdir": str(outdir),
        "commands": cmds,
    }
    report_path = outdir / "taxonomy_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "report": str(report_path)}))
        return 0

    for c in cmds:
        _run_checked(workdir, args.container, c)

    taxonomy_tsv = outdir / "exported_taxonomy" / "taxonomy.tsv"
    if not taxonomy_tsv.exists():
        raise TaxonomyError(f"QIIME2 export did not produce taxonomy.tsv at {taxonomy_tsv}")

    report2 = json.loads(report_path.read_text(encoding="utf-8"))
    report2["results"] = {
        "taxonomy_qza": str((workdir / tax_qza).resolve()),
        "taxonomy_qzv": str((workdir / tax_qzv).resolve()),
        "taxonomy_tsv": str(taxonomy_tsv.resolve()),
    }
    report_path.write_text(json.dumps(report2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "taxonomy_tsv": str(taxonomy_tsv.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

