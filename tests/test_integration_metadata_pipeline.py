import subprocess
from pathlib import Path

import pandas as pd
import pytest


def test_merge_meta_and_extract_supp_integration(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out_combined = tmp_path / "combined_meta.tsv"
    out_schema = tmp_path / "schema_diff.json"
    r = subprocess.run(
        [
            "python3",
            str(root / "merge_meta.py"),
            "--prjeb-manifest",
            str(root / "dieckow_manifest.tsv"),
            "--prjna-filereport",
            str(root / "data" / "PRJNA1159109_filereport.tsv"),
            "--prjna-biosample",
            str(root / "data" / "biosample_attributes.tsv"),
            "--out",
            str(out_combined),
            "--schema-log",
            str(out_schema),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out_combined, sep="\t")
    assert df["sample_alias"].is_unique
    assert (df["project_accession"] == "PRJNA1159109").sum() == 64

    out_struct = tmp_path / "structured_supp.tsv"
    out_manual = tmp_path / "manual_review.tsv"
    pdf_path = root / "docs" / "ZJOM_16_2424227.pdf"
    docx_path = root / "docs" / "ZJOM_A_2424227_SM9092.docx"
    if not pdf_path.exists() or not docx_path.exists():
        pytest.skip("ZJOM PDF/DOCX not present in repository clone")
    r2 = subprocess.run(
        [
            "python3",
            str(root / "extract_supp.py"),
            "--pdf",
            str(pdf_path),
            "--docx",
            str(docx_path),
            "--combined-meta",
            str(out_combined),
            "--out",
            str(out_struct),
            "--manual-review",
            str(out_manual),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stderr
    df_struct = pd.read_csv(out_struct, sep="\t")
    assert df_struct["sample_alias"].nunique() == 64


def test_export_model_inputs_integration_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    qiime_out = root / "data" / "qiime2_out"
    ft = qiime_out / "exported_table" / "feature-table.tsv"
    tax = root / "data" / "qiime2_taxonomy" / "exported_taxonomy" / "taxonomy.tsv"
    meta = root / "data" / "sample-metadata.tsv"
    if not ft.exists() or not tax.exists() or not meta.exists():
        pytest.skip("QIIME2 exports not present; requires running QIIME2 + taxonomy first")
    outdir = tmp_path / "model_inputs"
    r = subprocess.run(
        [
            "python3",
            str(root / "export_model_inputs.py"),
            "--feature-table",
            str(ft),
            "--taxonomy",
            str(tax),
            "--metadata",
            str(meta),
            "--outdir",
            str(outdir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (outdir / "trajectories.tsv").exists()
