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


def test_export_qiime2_creates_manifest_and_metadata(tmp_path):
    from nife.export_qiime2 import main as export_main

    combined = tmp_path / "combined_meta.tsv"
    combined.write_text(
        "sample_alias\tproject_accession\trun_accession\tin_vivo_in_vitro\tmaterial\tday\tdonor\n"
        "NU_01\tPRJNA1159109\tSRR1\tsaliva\tnot_provided\tnot_provided\t1\n"
        "NU_02\tPRJNA1159109\tSRR2\tin_vivo\tTi6Al4V\t2\tnot_provided\n",
        encoding="utf-8",
    )

    fqdir = tmp_path / "PRJNA1159109_fastq"
    _write_fastq_gz(fqdir / "SRR1_NU_01_1.fastq.gz")
    _write_fastq_gz(fqdir / "SRR1_NU_01_2.fastq.gz")
    _write_fastq_gz(fqdir / "SRR2_NU_02_1.fastq.gz")
    _write_fastq_gz(fqdir / "SRR2_NU_02_2.fastq.gz")

    manifest = tmp_path / "qiime2_manifest.tsv"
    metadata = tmp_path / "sample-metadata.tsv"
    report = tmp_path / "report.json"

    argv = [
        "export_qiime2.py",
        "--combined-meta",
        str(combined),
        "--fastq-prjna-dir",
        str(fqdir),
        "--out-manifest",
        str(manifest),
        "--out-metadata",
        str(metadata),
        "--report-json",
        str(report),
        "--allow-missing-fastq",
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", argv)
        assert export_main() == 0

    man = pd.read_csv(manifest, sep="\t")
    assert list(man.columns) == ["sample-id", "absolute-filepath", "direction"]
    assert man.shape[0] == 4
    assert set(man["direction"].tolist()) == {"forward", "reverse"}
    assert all(Path(p).is_absolute() for p in man["absolute-filepath"].tolist())

    meta = pd.read_csv(metadata, sep="\t")
    assert meta["sample-id"].is_unique
    assert meta.shape[0] == 2


def test_export_qiime2_errors_on_missing_fastq_by_default(tmp_path):
    from nife.export_qiime2 import main as export_main

    combined = tmp_path / "combined_meta.tsv"
    combined.write_text(
        "sample_alias\tproject_accession\trun_accession\n"
        "NU_01\tPRJNA1159109\tSRR1\n",
        encoding="utf-8",
    )
    fqdir = tmp_path / "PRJNA1159109_fastq"
    fqdir.mkdir(parents=True, exist_ok=True)

    argv = [
        "export_qiime2.py",
        "--combined-meta",
        str(combined),
        "--fastq-prjna-dir",
        str(fqdir),
        "--out-manifest",
        str(tmp_path / "m.tsv"),
        "--out-metadata",
        str(tmp_path / "s.tsv"),
        "--report-json",
        str(tmp_path / "r.json"),
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", argv)
        with pytest.raises(Exception):
            export_main()
