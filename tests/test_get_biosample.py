import pandas as pd


def _xml(accession: str, alias: str, title: str, attrs: dict[str, str]) -> bytes:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<SAMPLE_SET>",
        f'<SAMPLE accession="{accession}" alias="{alias}">',
        f"<TITLE>{title}</TITLE>",
        "<SAMPLE_ATTRIBUTES>",
    ]
    for k, v in attrs.items():
        parts.append("<SAMPLE_ATTRIBUTE>")
        parts.append(f"<TAG>{k}</TAG>")
        parts.append(f"<VALUE>{v}</VALUE>")
        parts.append("</SAMPLE_ATTRIBUTE>")
    parts += ["</SAMPLE_ATTRIBUTES>", "</SAMPLE>", "</SAMPLE_SET>"]
    return "\n".join(parts).encode("utf-8")


def test_parse_biosample_xml_and_extract_required_in_vitro():
    from nife.get_biosample import extract_required, parse_biosample_xml

    xml = _xml(
        accession="SAMN43561083",
        alias="NU_61",
        title="7d biofilm In Vitro IV Ti6Al4V 1",
        attrs={
            "collection_date": "2023-11-21",
            "env_local_scale": "24 well plate",
            "env_medium": "BHI + 0.2% Sucrose + 50 mM PIPES",
            "geo_loc_name": "Germany: Aachen",
        },
    )
    parsed = parse_biosample_xml(xml)
    row = extract_required(parsed)
    assert row.biosample_accession == "SAMN43561083"
    assert row.sample_alias == "NU_61"
    assert row.in_vivo_in_vitro == "in_vitro"
    assert row.material == "Ti6Al4V"
    assert row.day == "7"
    assert row.donor == "not_provided"
    assert row.collection_date == "2023-11-21"
    assert row.env_local_scale == "24 well plate"


def test_parse_biosample_xml_and_extract_required_in_vivo():
    from nife.get_biosample import extract_required, parse_biosample_xml

    xml = _xml(
        accession="SAMN43561028",
        alias="NU_06",
        title="2d biofilm In Vivo on Ti6Al4V 2",
        attrs={
            "collection_date": "2023-11-21",
            "env_local_scale": "oral splint",
            "env_medium": "Oral Cavity",
            "geo_loc_name": "Germany: Aachen",
        },
    )
    parsed = parse_biosample_xml(xml)
    row = extract_required(parsed)
    assert row.in_vivo_in_vitro == "in_vivo"
    assert row.material == "Ti6Al4V"
    assert row.day == "2"
    assert row.donor == "not_provided"


def test_parse_biosample_xml_and_extract_required_saliva_has_donor_number():
    from nife.get_biosample import extract_required, parse_biosample_xml

    xml = _xml(
        accession="SAMN43561025",
        alias="NU_01",
        title="Batch 1 Saliva 1",
        attrs={
            "collection_date": "2023-11-21",
            "env_local_scale": "oral",
            "env_medium": "Oral Cavity",
            "geo_loc_name": "Germany: Aachen",
        },
    )
    parsed = parse_biosample_xml(xml)
    row = extract_required(parsed)
    assert row.in_vivo_in_vitro == "saliva"
    assert row.day == "not_provided"
    assert row.donor == "1"


def test_write_tsv_is_pandas_readable(tmp_path):
    from nife.get_biosample import ParsedBioSample, write_tsv

    rows = [
        ParsedBioSample(
            biosample_accession="SAMN1",
            sample_alias="NU_01",
            in_vivo_in_vitro="saliva",
            material="not_provided",
            day="not_provided",
            donor="1",
            title="Batch 1 Saliva 1",
            collection_date="2023-11-21",
            env_local_scale="oral",
            env_medium="Oral Cavity",
            geo_loc_name="Germany: Aachen",
        )
    ]
    out = tmp_path / "biosample_attributes.tsv"
    write_tsv(out, rows)
    df = pd.read_csv(out, sep="\t")
    assert df.shape[0] == 1
    assert list(df.columns)[0] == "biosample_accession"
