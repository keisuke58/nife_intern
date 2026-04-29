import pandas as pd
import pytest


def test_parse_genus_from_taxon():
    from nife.export_model_inputs import parse_genus_from_taxon

    assert parse_genus_from_taxon("k__Bacteria; p__Firmicutes; g__Streptococcus; s__") == "Streptococcus"
    assert parse_genus_from_taxon("Unassigned") == "Unassigned"
    assert parse_genus_from_taxon("") == "Unassigned"


def test_export_model_inputs_end_to_end(tmp_path, monkeypatch):
    from nife.export_model_inputs import main

    ft = tmp_path / "feature-table.tsv"
    ft.write_text(
        "# Constructed from biom file\n"
        "#OTU ID\tNU_01\tNU_02\n"
        "F1\t10\t0\n"
        "F2\t5\t5\n",
        encoding="utf-8",
    )
    tax = tmp_path / "taxonomy.tsv"
    tax.write_text(
        "Feature ID\tTaxon\tConfidence\n"
        "F1\tk__Bacteria; p__Firmicutes; g__Streptococcus; s__oralis\t0.9\n"
        "F2\tk__Bacteria; p__Firmicutes; g__Veillonella; s__\t0.8\n",
        encoding="utf-8",
    )
    meta = tmp_path / "sample-metadata.tsv"
    meta.write_text(
        "sample-id\tproject_accession\tin_vivo_in_vitro\tmaterial\tday\tweek\tdonor\tpatient\trun_accession\n"
        "NU_01\tPRJNA1159109\tsaliva\tnot_provided\tNA\tNA\t1\tNA\tSRR1\n"
        "NU_02\tPRJNA1159109\tin_vivo\tTi6Al4V\t2\tNA\tnot_provided\tNA\tSRR2\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "model_inputs"

    argv = [
        "export_model_inputs.py",
        "--feature-table",
        str(ft),
        "--taxonomy",
        str(tax),
        "--metadata",
        str(meta),
        "--outdir",
        str(outdir),
        "--split-test",
        "in_vivo",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0

    traj = pd.read_csv(outdir / "trajectories.tsv", sep="\t")
    assert set(traj["sample-id"].unique()) == {"NU_01", "NU_02"}
    assert set(traj["taxon"].unique()) == {"Streptococcus", "Veillonella"}

    splits = pd.read_csv(outdir / "split_specs.tsv", sep="\t")
    assert splits.shape[0] == 2
    assert splits.loc[splits["sample-id"] == "NU_02", "split"].item() == "test"


def test_export_model_inputs_rejects_sample_mismatch(tmp_path, monkeypatch):
    from nife.export_model_inputs import main

    ft = tmp_path / "feature-table.tsv"
    ft.write_text(
        "# Constructed from biom file\n"
        "#OTU ID\tNU_01\n"
        "F1\t10\n",
        encoding="utf-8",
    )
    tax = tmp_path / "taxonomy.tsv"
    tax.write_text("Feature ID\tTaxon\nF1\tk__Bacteria; g__X\n", encoding="utf-8")
    meta = tmp_path / "sample-metadata.tsv"
    meta.write_text("sample-id\nNU_XX\n", encoding="utf-8")
    outdir = tmp_path / "out"

    argv = [
        "export_model_inputs.py",
        "--feature-table",
        str(ft),
        "--taxonomy",
        str(tax),
        "--metadata",
        str(meta),
        "--outdir",
        str(outdir),
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(Exception):
        main()

