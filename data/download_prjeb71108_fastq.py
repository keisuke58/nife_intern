#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    run_accession: str
    sample_alias: str
    fastq_bytes: int
    fastq_url: str


def _to_https(url: str) -> str:
    url = url.strip()
    if url.startswith("ftp."):
        return "https://" + url
    if url.startswith("ftp://"):
        return "https://" + url[len("ftp://") :]
    return url


def _read_entries(tsv_path: Path) -> list[Entry]:
    entries: list[Entry] = []
    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run = (row.get("run_accession") or "").strip()
            alias = (row.get("sample_alias") or "").strip()
            b = int((row.get("fastq_bytes") or "0").strip())
            url = _to_https((row.get("fastq_ftp") or "").strip())
            if not run or not url:
                continue
            entries.append(Entry(run_accession=run, sample_alias=alias, fastq_bytes=b, fastq_url=url))
    return entries


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=Path(__file__).with_name("PRJEB71108_filereport.tsv"))
    ap.add_argument("--outdir", type=Path, default=Path(__file__).with_name("PRJEB71108_fastq"))
    ap.add_argument("--limit", type=int, default=0, help="Download only first N runs (0 = no limit).")
    ap.add_argument("--samples", type=str, default="", help="Comma-separated sample_alias filter (e.g., A_1,B_2).")
    ap.add_argument("--runs", type=str, default="", help="Comma-separated run_accession filter (e.g., ERR13166574).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--download", action="store_true", help="Actually download files.")
    args = ap.parse_args()

    entries = _read_entries(args.tsv)
    if not entries:
        print(f"No entries found in: {args.tsv}", file=sys.stderr)
        return 2

    sample_filter = {s.strip() for s in args.samples.split(",") if s.strip()}
    run_filter = {s.strip() for s in args.runs.split(",") if s.strip()}
    if sample_filter:
        entries = [e for e in entries if e.sample_alias in sample_filter]
    if run_filter:
        entries = [e for e in entries if e.run_accession in run_filter]
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    total_bytes = sum(e.fastq_bytes for e in entries)
    print(f"Runs: {len(entries)}")
    print(f"Total size: {_format_bytes(total_bytes)}")
    print(f"Output dir: {args.outdir}")

    manifest_path = args.outdir / "manifest.tsv"
    args.outdir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["run_accession", "sample_alias", "fastq_bytes", "fastq_url", "dest_path"])
        for e in entries:
            dest = args.outdir / f"{e.run_accession}_{e.sample_alias}.fastq.gz"
            w.writerow([e.run_accession, e.sample_alias, str(e.fastq_bytes), e.fastq_url, str(dest)])
    print(f"Wrote manifest: {manifest_path}")

    if args.dry_run or not args.download:
        print("Dry run. Use --download to fetch files.")
        for e in entries[: min(5, len(entries))]:
            print(f"- {e.run_accession} {e.sample_alias} {_format_bytes(e.fastq_bytes)} {e.fastq_url}")
        return 0

    for i, e in enumerate(entries, start=1):
        dest = args.outdir / f"{e.run_accession}_{e.sample_alias}.fastq.gz"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{i}/{len(entries)}] skip (exists): {dest.name}")
            continue
        print(f"[{i}/{len(entries)}] download: {dest.name} ({_format_bytes(e.fastq_bytes)})")
        _download(e.fastq_url, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

