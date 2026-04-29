import pandas as pd


def test_extract_fields_finds_expected_patterns():
    from nife.extract_supp import _extract_fields

    text = "2d biofilm In Vivo on Ti6Al4V oral splint BHI 50 mM PIPES Illumina MiSeq V3-V4 DADA2"
    found, missing = _extract_fields(text, source="x")
    d = {f.field: f.value for f in found}
    assert d["material"] == "Ti6Al4V"
    assert "2" in d["timepoints_days"]
    assert d["device_oral_splint"] == "oral splint"
    assert d["medium_BHI"] == "BHI"
    assert d["buffer_PIPES"] == "PIPES"
    assert d["sequencing_platform"].lower().find("illumina") != -1 or d["sequencing_platform"].lower().find("miseq") != -1
    assert d["amplicon_region"].upper().replace("–", "-") in {"V3-V4"}


def test_write_kv_tsv_is_pandas_readable(tmp_path):
    from nife.extract_supp import write_kv_tsv

    out = tmp_path / "structured_supp.tsv"
    write_kv_tsv(
        out,
        [
            {
                "sample_alias": "NU_01",
                "field": "sequencing_platform",
                "value": "MiSeq",
                "source_file": "a.pdf",
                "evidence": "MiSeq",
            }
        ],
    )
    df = pd.read_csv(out, sep="\t")
    assert df.shape == (1, 5)
