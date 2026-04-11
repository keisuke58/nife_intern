#!/bin/bash
#PBS -N minimap2_16s
#PBS -l nodes=1:ppn=8
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/data/minimap2_16s.log

# ============================================================
# PacBio full-length 16S rRNA → genus-level init_comp.json
# Tools: minimap2 + SILVA 138.1 SSU NR99 (in metaphlan4 env)
# ============================================================

set -euo pipefail

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

NIFE_DIR="/home/nishioka/IKM_Hiwi/nife"
DATA_DIR="${NIFE_DIR}/data"
SILVA_DIR="${DATA_DIR}/silva_db"
FASTQ="${DATA_DIR}/PRJEB71108_fastq/ERR13166576_A_3.fastq.gz"
SAMPLE="ERR13166576_A_3"
OUTDIR="${DATA_DIR}/metaphlan_profiles"
NCPU=8

SILVA_FA="${SILVA_DIR}/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz"
SILVA_URL="https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz"
SILVA_IDX="${SILVA_DIR}/silva138_ssu_nr99.mmi"
SAM_OUT="${OUTDIR}/${SAMPLE}.silva.sam"
INIT_COMP="${OUTDIR}/init_comp_${SAMPLE}_minimap2.json"

mkdir -p "${SILVA_DIR}" "${OUTDIR}"
export SAM_OUT INIT_COMP SAMPLE

# ---------- 1. Download SILVA (skip if present) ----------
if [[ ! -f "${SILVA_FA}" ]]; then
    echo "[$(date)] Downloading SILVA 138.1 SSU NR99 (~380MB)..."
    wget -q --show-progress --timeout=120 --tries=5 --retry-connrefused \
        -O "${SILVA_FA}" "${SILVA_URL}" \
    || { rm -f "${SILVA_FA}"; echo "[error] SILVA download failed"; exit 1; }
    echo "[$(date)] Download complete."
else
    echo "[$(date)] SILVA already present, skipping download."
fi

# ---------- 2. Build minimap2 index (skip if present) ----------
if [[ ! -f "${SILVA_IDX}" ]]; then
    echo "[$(date)] Building minimap2 index (map-hifi for PacBio CCS)..."
    minimap2 -x map-hifi -d "${SILVA_IDX}" "${SILVA_FA}" -t "${NCPU}"
    echo "[$(date)] Index built: ${SILVA_IDX}"
else
    echo "[$(date)] Index already present, skipping."
fi

# ---------- 3. Map reads ----------
echo "[$(date)] Mapping ${SAMPLE} against SILVA..."
minimap2 -ax map-hifi -t "${NCPU}" --secondary=no \
    "${SILVA_IDX}" "${FASTQ}" \
    > "${SAM_OUT}"
echo "[$(date)] Mapping done: ${SAM_OUT}"

# ---------- 4. Parse SAM → genus counts → init_comp.json ----------
echo "[$(date)] Parsing alignments → genus-level composition..."
python3 - <<'PYEOF'
import re, json, sys, os
from collections import defaultdict
from pathlib import Path

SAM  = os.environ.get("SAM_OUT")
OUT  = os.environ.get("INIT_COMP")
SAMPLE = os.environ.get("SAMPLE")

TARGET = {
    "streptococcus": "Str",
    "actinomyces":   "Act",
    "schaalia":      "Act",   # renamed Actinomyces
    "veillonella":   "Vel",
    "haemophilus":   "Hae",
    "rothia":        "Rot",
    "fusobacterium": "Fus",
    "porphyromonas": "Por",
}

counts = defaultdict(int)
total_mapped = 0

with open(SAM) as f:
    for line in f:
        if line.startswith("@"):
            continue
        fields = line.split("\t")
        flag = int(fields[1])
        if flag & 4:          # unmapped
            continue
        rname = fields[2]     # SILVA ref name contains full taxonomy
        # SILVA header format: AccNr.Range Taxonomy
        # e.g. "AB016645.1.1486 Bacteria;Firmicutes;...;Streptococcus;S. mutans"
        rname_lower = rname.lower()
        total_mapped += 1
        for keyword, group in TARGET.items():
            if keyword in rname_lower:
                counts[group] += 1
                break

print(f"[info] Total mapped reads: {total_mapped}", file=sys.stderr)
print(f"[info] Target genus counts: {dict(counts)}", file=sys.stderr)

groups = ["Str", "Act", "Vel", "Hae", "Rot", "Fus", "Por"]
total = sum(counts[g] for g in groups)

if total == 0:
    print("[warn] No target genera found — using equal fallback.", file=sys.stderr)
    comp = {g: round(1.0/7, 6) for g in groups}
else:
    comp = {g: round(counts[g] / total, 6) for g in groups}
    # renormalize to exactly 1
    s = sum(comp.values())
    comp = {g: round(comp[g]/s, 6) for g in groups}

comp["_source"] = f"minimap2 v2.30 + SILVA 138.1 SSU NR99; map-hifi; {total_mapped} mapped reads"
out = json.dumps(comp, indent=2, sort_keys=True)
Path(OUT).write_text(out + "\n")
print(f"[info] Wrote {OUT}", file=sys.stderr)
print(out)
PYEOF

echo "[$(date)] Done. init_comp: ${INIT_COMP}"
