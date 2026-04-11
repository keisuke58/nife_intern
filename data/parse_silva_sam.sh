#!/bin/bash
#PBS -N parse_silva
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/data/parse_silva.log

# Re-parse existing SAM using accession→taxonomy map from SILVA FASTA
# SAM already exists; just rebuild init_comp.json correctly

set -euo pipefail

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

NIFE_DIR="/home/nishioka/IKM_Hiwi/nife"
DATA_DIR="${NIFE_DIR}/data"
SILVA_DIR="${DATA_DIR}/silva_db"
SAMPLE="ERR13166576_A_3"
OUTDIR="${DATA_DIR}/metaphlan_profiles"

SILVA_FA="${SILVA_DIR}/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz"
SAM_OUT="${OUTDIR}/${SAMPLE}.silva.sam"
INIT_COMP="${OUTDIR}/init_comp_${SAMPLE}_minimap2.json"

export SAM_OUT INIT_COMP SAMPLE SILVA_FA

echo "[$(date)] Building accession→taxonomy map from SILVA FASTA and parsing SAM..."

python3 - <<'PYEOF'
import gzip, json, sys, os
from collections import defaultdict
from pathlib import Path

SAM     = os.environ["SAM_OUT"]
OUT     = os.environ["INIT_COMP"]
SAMPLE  = os.environ["SAMPLE"]
SILVA   = os.environ["SILVA_FA"]

TARGET = {
    "streptococcus": "Str",
    "actinomyces":   "Act",
    "schaalia":      "Act",
    "veillonella":   "Vel",
    "haemophilus":   "Hae",
    "rothia":        "Rot",
    "fusobacterium": "Fus",
    "porphyromonas": "Por",
}

# ---------- 1. Build accession → group map from SILVA headers ----------
print("[info] Reading SILVA headers...", file=sys.stderr)
acc2group: dict[str, str] = {}
open_fn = gzip.open if SILVA.endswith(".gz") else open
with open_fn(SILVA, "rt") as f:
    for line in f:
        if not line.startswith(">"):
            continue
        # e.g. ">AB016645.1.1486 Bacteria;Firmicutes;...;Streptococcus;S. mutans"
        header = line[1:].rstrip()
        acc = header.split()[0]           # "AB016645.1.1486"
        taxonomy = header[len(acc):].strip().lower()
        for keyword, group in TARGET.items():
            if keyword in taxonomy:
                acc2group[acc] = group
                break

print(f"[info] Accessions with target genus: {len(acc2group)}", file=sys.stderr)

# ---------- 2. Parse SAM ----------
counts: dict[str, int] = defaultdict(int)
total_mapped = 0

with open(SAM) as f:
    for line in f:
        if line.startswith("@"):
            continue
        fields = line.split("\t")
        flag = int(fields[1])
        if flag & 4:
            continue
        rname = fields[2]
        total_mapped += 1
        g = acc2group.get(rname)
        if g:
            counts[g] += 1

print(f"[info] Total mapped reads: {total_mapped}", file=sys.stderr)
print(f"[info] Target genus counts: {dict(counts)}", file=sys.stderr)

groups = ["Str", "Act", "Vel", "Hae", "Rot", "Fus", "Por"]
total = sum(counts[g] for g in groups)

if total == 0:
    print("[warn] No target genera found — using equal fallback.", file=sys.stderr)
    comp = {g: round(1.0/7, 6) for g in groups}
else:
    comp = {g: round(counts[g] / total, 6) for g in groups}
    s = sum(comp.values())
    comp = {g: round(comp[g] / s, 6) for g in groups}

comp["_source"] = (
    f"minimap2 v2.30 + SILVA 138.1 SSU NR99; map-hifi; "
    f"{total_mapped} mapped reads; accession-based taxonomy lookup"
)
out = json.dumps(comp, indent=2, sort_keys=True)
Path(OUT).write_text(out + "\n")
print(f"[info] Wrote {OUT}", file=sys.stderr)
print(out)
PYEOF

echo "[$(date)] Done. init_comp: ${INIT_COMP}"
