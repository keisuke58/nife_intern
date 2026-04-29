import gzip
from pathlib import Path

import pandas as pd
import pytest


def _write_fastq_gz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        f.write("@r1\n")
        f.write("ACGT\n")
        f.write("+\n")
        f.write("####\n")


def test_run_qiime2_dry_run_validates_and_writes_report(tmp_path, monkeypatch):
    from nife.run_qiime2_dada2 import main

    fq1 = tmp_path / "SRR1_NU_01_1.fastq.gz"
    fq2 = tmp_path / "SRR1_NU_01_2.fastq.gz"
    _write_fastq_gz(fq1)
    _write_fastq_gz(fq2)

    manifest = tmp_path / "qiime2_manifest.tsv"
    manifest.write_text(
        "sample-id\tforward-absolute-filepath\treverse-absolute-filepath\n"
        f"NU_01\t{fq1}\t{fq2}\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "sample-metadata.tsv"
    metadata.write_text(
        "sample-id\tmaterial\tday\n"
        "NU_01\tTi6Al4V\t2\n",
        encoding="utf-8",
    )

    argv = [
        "run_qiime2_dada2.py",
        "--workdir",
        str(tmp_path),
        "--manifest",
        str(manifest),
        "--metadata",
        str(metadata),
        "--outdir",
        str(tmp_path / "qiime2_out"),
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    report = tmp_path / "qiime2_out" / "qiime2_run_report.json"
    assert report.exists()


def test_run_qiime2_rejects_mismatched_sample_ids(tmp_path, monkeypatch):
    from nife.run_qiime2_dada2 import main

    fq1 = tmp_path / "SRR1_NU_01_1.fastq.gz"
    fq2 = tmp_path / "SRR1_NU_01_2.fastq.gz"
    _write_fastq_gz(fq1)
    _write_fastq_gz(fq2)

    manifest = tmp_path / "qiime2_manifest.tsv"
    manifest.write_text(
        "sample-id\tforward-absolute-filepath\treverse-absolute-filepath\n"
        f"NU_01\t{fq1}\t{fq2}\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "sample-metadata.tsv"
    metadata.write_text(
        "sample-id\tmaterial\n"
        "NU_XX\tTi6Al4V\n",
        encoding="utf-8",
    )

    argv = [
        "run_qiime2_dada2.py",
        "--workdir",
        str(tmp_path),
        "--manifest",
        str(manifest),
        "--metadata",
        str(metadata),
        "--outdir",
        str(tmp_path / "qiime2_out"),
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(Exception):
        main()
