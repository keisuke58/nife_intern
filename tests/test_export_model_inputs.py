from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from nife import export_model_inputs as m


def test_genus_from_taxon_parses_qiime2_styles() -> None:
    assert m.genus_from_taxon("k__Bacteria; p__Firmicutes; c__Bacilli; g__Streptococcus; s__") == "Streptococcus"
    assert m.genus_from_taxon("D_0__Bacteria; D_1__Firmicutes; D_5__Veillonella; D_6__sp") == "Veillonella"
    assert m.genus_from_taxon("Unassigned") == "Unassigned"


def test_read_qiime2_feature_table_tsv_handles_biom_convert_header(tmp_path: Path) -> None:
    p = tmp_path / "feature-table.tsv"
    p.write_text(
        "\n".join(
            [
                "# Constructed from biom file",
                "#OTU ID\tS1\tS2",
                "f1\t10\t0",
                "f2\t0\t5",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    df = m.read_qiime2_feature_table_tsv(p)
    assert list(df.columns) == ["S1", "S2"]
    assert set(df.index) == {"f1", "f2"}
    assert float(df.loc["f1", "S1"]) == 10.0


def test_export_model_inputs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    feature_table = tmp_path / "feature-table.tsv"
    taxonomy = tmp_path / "taxonomy.tsv"
    metadata = tmp_path / "sample-metadata.tsv"
    outdir = tmp_path / "out"

    feature_table.write_text(
        "\n".join(
            [
                "# Constructed from biom file",
                "#OTU ID\tS1\tS2",
                "f1\t10\t0",
                "f2\t0\t5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    taxonomy.write_text(
        "\n".join(
            [
                "Feature ID\tTaxon\tConfidence",
                "f1\tk__Bacteria; p__Firmicutes; c__Bacilli; g__Streptococcus; s__\t0.99",
                "f2\tD_0__Bacteria; D_1__Firmicutes; D_5__Veillonella; D_6__sp\t0.95",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata.write_text(
        "\n".join(
            [
                "sample-id\tday\tmaterial\tin_vivo_in_vitro\tproject_accession",
                "S1\t2\tTi6Al4V\tin_vitro\tPRJNA1159109",
                "S2\t4\tY-TZP\tin_vitro\tPRJNA1159109",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    argv = [
        "export_model_inputs.py",
        "--feature-table-tsv",
        str(feature_table),
        "--taxonomy-tsv",
        str(taxonomy),
        "--metadata",
        str(metadata),
        "--outdir",
        str(outdir),
        "--time-col",
        "day",
        "--group-cols",
        "material,in_vivo_in_vitro,project_accession",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert m.main() == 0

    genus_wide = pd.read_csv(outdir / "genus_relative_abundance.wide.tsv", sep="\t")
    assert set(genus_wide["taxon"]) == {"Streptococcus", "Veillonella"}

    genus_long = pd.read_csv(outdir / "genus_relative_abundance.long.tsv", sep="\t")
    assert set(genus_long["sample-id"]) == {"S1", "S2"}
    assert set(genus_long["taxon"]) == {"Streptococcus", "Veillonella"}

    report = json.loads((outdir / "model_inputs.report.json").read_text(encoding="utf-8"))
    assert report["counts"]["n_samples"] == 2

