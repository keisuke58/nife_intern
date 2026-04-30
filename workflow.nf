nextflow.enable.dsl = 2

params.filereport = "${projectDir}/data/PRJNA1159109_filereport.tsv"
params.study = "PRJNA1159109"
params.biosample_out = "${projectDir}/data/biosample_attributes.tsv"
params.run_log = "${projectDir}/data/run.log"

params.prjeb_manifest = "${projectDir}/dieckow_manifest.tsv"
params.combined_out = "${projectDir}/data/combined_meta.tsv"
params.schema_log = "${projectDir}/data/schema_diff.json"

params.pdf = "${projectDir}/docs/ZJOM_16_2424227.pdf"
params.docx = "${projectDir}/docs/ZJOM_A_2424227_SM9092.docx"
params.structured_out = "${projectDir}/data/structured_supp.tsv"
params.manual_review_out = "${projectDir}/data/manual_review.tsv"

process GET_BIOSAMPLE {
  tag "get_biosample"
  publishDir "${projectDir}/data", mode: 'copy', overwrite: true
  input:
    path filereport
  output:
    path "biosample_attributes.tsv"
    path "run.log"
  script:
    """
    python3 ${projectDir}/get_biosample.py \\
      --filereport ${filereport} \\
      --study ${params.study} \\
      --out ${projectDir}/data/biosample_attributes.tsv \\
      --log ${projectDir}/data/run.log
    cp ${projectDir}/data/biosample_attributes.tsv .
    cp ${projectDir}/data/run.log .
    """
}

process MERGE_META {
  tag "merge_meta"
  publishDir "${projectDir}/data", mode: 'copy', overwrite: true
  input:
    path biosample_tsv
    path filereport
    path prjeb_manifest
  output:
    path "combined_meta.tsv"
    path "schema_diff.json"
  script:
    """
    python3 ${projectDir}/merge_meta.py \\
      --prjeb-manifest ${prjeb_manifest} \\
      --prjna-filereport ${filereport} \\
      --prjna-biosample ${biosample_tsv} \\
      --out ${projectDir}/data/combined_meta.tsv \\
      --schema-log ${projectDir}/data/schema_diff.json
    cp ${projectDir}/data/combined_meta.tsv .
    cp ${projectDir}/data/schema_diff.json .
    """
}

process EXTRACT_SUPP {
  tag "extract_supp"
  publishDir "${projectDir}/data", mode: 'copy', overwrite: true
  input:
    path combined_meta
    path pdf
    path docx
  output:
    path "structured_supp.tsv"
    path "manual_review.tsv"
  script:
    """
    python3 ${projectDir}/extract_supp.py \\
      --pdf ${pdf} \\
      --docx ${docx} \\
      --combined-meta ${combined_meta} \\
      --out ${projectDir}/data/structured_supp.tsv \\
      --manual-review ${projectDir}/data/manual_review.tsv
    cp ${projectDir}/data/structured_supp.tsv .
    cp ${projectDir}/data/manual_review.tsv .
    """
}

process SUMMARY {
  tag "summary"
  input:
    path combined_meta
    path biosample_tsv
    path structured_supp
    path manual_review
  output:
    stdout
  script:
    """
    python3 - <<'PY'
    import pandas as pd
    cm=pd.read_csv('${combined_meta}', sep='\\t')
    bs=pd.read_csv('${biosample_tsv}', sep='\\t')
    ss=pd.read_csv('${structured_supp}', sep='\\t')
    mr=pd.read_csv('${manual_review}', sep='\\t')
    print('combined_meta_rows', cm.shape[0])
    print('combined_projects', cm['project_accession'].value_counts().to_dict())
    print('biosample_rows', bs.shape[0])
    print('biosample_in_vivo_in_vitro', bs['in_vivo_in_vitro'].value_counts().to_dict())
    print('structured_supp_rows', ss.shape[0], 'unique_alias', ss['sample_alias'].nunique())
    print('manual_review_rows', mr.shape[0])
    print('handoff_files', ['${projectDir}/data/biosample_attributes.tsv','${projectDir}/data/combined_meta.tsv','${projectDir}/data/structured_supp.tsv','${projectDir}/data/manual_review.tsv'])
    PY
    """
}

workflow {
  filereport_ch = Channel.fromPath(params.filereport)
  prjeb_manifest_ch = Channel.fromPath(params.prjeb_manifest)
  pdf_ch = Channel.fromPath(params.pdf)
  docx_ch = Channel.fromPath(params.docx)

  bios = GET_BIOSAMPLE(filereport_ch)
  merged = MERGE_META(bios.out[0], filereport_ch, prjeb_manifest_ch)
  supp = EXTRACT_SUPP(merged.out[0], pdf_ch, docx_ch)
  SUMMARY(merged.out[0], bios.out[0], supp.out[0], supp.out[1])
}
