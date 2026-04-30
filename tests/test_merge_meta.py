import pandas as pd
import pytest


def test_merge_meta_builds_combined_meta(tmp_path):
    from nife.merge_meta import CANONICAL_COLUMNS, _force_columns, load_prjeb71108_from_manifest, load_prjna1159109_meta

    prjeb_manifest = tmp_path / "dieckow_manifest.tsv"
    prjeb_manifest.write_text(
        "sample\tpatient\ttimepoint\terr_accession\tfastq\n"
        "A_1\tA\t1\tERR1\tERR1.fastq.gz\n"
        "A_2\tA\t2\tERR2\tERR2.fastq.gz\n",
        encoding="utf-8",
    )
    prjna_filereport = tmp_path / "PRJNA1159109_filereport.tsv"
    prjna_filereport.write_text(
        "run_accession\tsample_alias\tfastq_ftp\tfastq_md5\tfastq_bytes\tlibrary_layout\tlibrary_strategy\tlibrary_source\tlibrary_selection\n"
        "SRR1\tNU_01\tftp://x;ftp://y\tm1;m2\t1;2\tPAIRED\tAMPLICON\tGENOMIC\tRT-PCR\n",
        encoding="utf-8",
    )
    prjna_biosample = tmp_path / "biosample_attributes.tsv"
    prjna_biosample.write_text(
        "biosample_accession\tsample_alias\tin_vivo_in_vitro\tmaterial\tday\tdonor\ttitle\tcollection_date\tenv_local_scale\tenv_medium\tgeo_loc_name\n"
        "SAMN1\tNU_01\tsaliva\tnot_provided\tnot_provided\t1\tBatch 1 Saliva 1\t2023-11-21\toral\tOral Cavity\tGermany: Aachen\n",
        encoding="utf-8",
    )

    prjeb = load_prjeb71108_from_manifest(prjeb_manifest)
    prjna = load_prjna1159109_meta(prjna_filereport, prjna_biosample)
    combined = pd.concat([prjeb, prjna], ignore_index=True)
    combined = _force_columns(combined, CANONICAL_COLUMNS)
    assert combined.shape[0] == 3
    assert list(combined.columns) == CANONICAL_COLUMNS
    assert combined.loc[combined["sample_alias"] == "A_1", "in_vivo_in_vitro"].item() == "in_vivo"
    assert combined.loc[combined["sample_alias"] == "NU_01", "biosample_accession"].item() == "SAMN1"


def test_merge_meta_rejects_duplicate_sample_alias(tmp_path):
    from nife.merge_meta import _assert_unique

    df = pd.DataFrame({"sample_alias": ["X", "X"]})
    with pytest.raises(ValueError):
        _assert_unique(df, "sample_alias")
