#!/bin/bash
#PBS -N silva_genus_idx
#PBS -l nodes=1:ppn=8
#PBS -l walltime=02:00:00
#PBS -q default
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/data/silva_db/build_genus_index.log

# Build SILVA genus-named FASTA + minimap2 index.
# Renames each sequence: ">genus|class|flag|accession"
# so SAM RNAME contains the genus directly.

DATA_DIR="/home/nishioka/IKM_Hiwi/nife/data/silva_db"
SILVA_FA="${DATA_DIR}/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz"
GENUS_FA="${DATA_DIR}/silva138_genus_named.fasta.gz"
GENUS_IDX="${DATA_DIR}/silva138_genus_named.mmi"
NCPU=8

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

echo "[$(date)] Building genus-named FASTA from SILVA..."

python3 - <<'PYEOF'
import gzip, re, sys
from pathlib import Path

# genus → (class, flag) for our oral microbiome TARGET
GENUS_MAP = {
    "streptococcus":      ("Bacilli",              "com"),
    "actinomyces":        ("Actinobacteria",        "com"),
    "schaalia":           ("Actinobacteria",        "com"),
    "rothia":             ("Actinobacteria",        "com"),
    "veillonella":        ("Negativicutes",         "com"),
    "dialister":          ("Negativicutes",         "com"),
    "haemophilus":        ("Pasteurellota",         "com"),
    "aggregatibacter":    ("Pasteurellota",         "com"),
    "porphyromonas":      ("Bacteroidia",           "dys"),
    "tannerella":         ("Bacteroidia",           "dys"),
    "prevotella":         ("Bacteroidia",           "dys"),
    "fusobacterium":      ("Fusobacteriia",         "dys"),
    "treponema":          ("Spirochaetia",          "dys"),
    "campylobacter":      ("Epsilonproteobacteria", "dys"),
    "peptostreptococcus": ("Clostridia",            "dys"),
    "filifactor":         ("Clostridia",            "dys"),
    "parvimonas":         ("Clostridia",            "dys"),
}

SILVA_FA  = "/home/nishioka/IKM_Hiwi/nife/data/silva_db/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz"
GENUS_FA  = "/home/nishioka/IKM_Hiwi/nife/data/silva_db/silva138_genus_named.fasta.gz"

kept = 0
total = 0

with gzip.open(SILVA_FA, "rt") as fin, gzip.open(GENUS_FA, "wt") as fout:
    acc = None; seq_lines = []
    for line in fin:
        if line.startswith(">"):
            # flush previous
            if acc and seq_lines:
                total += 1
                for kw, (cls, flag) in GENUS_MAP.items():
                    if kw in acc[1]:   # acc = (accession, taxonomy_lower)
                        new_name = f"{kw}|{cls}|{flag}|{acc[0]}"
                        fout.write(f">{new_name}\n")
                        fout.write("".join(seq_lines))
                        kept += 1
                        break
            # parse new header: ">AccNr Taxonomy"
            parts = line[1:].rstrip().split(" ", 1)
            accession = parts[0]
            taxonomy  = parts[1].lower() if len(parts) > 1 else ""
            acc = (accession, taxonomy)
            seq_lines = []
        else:
            seq_lines.append(line)
    # flush last
    if acc and seq_lines:
        total += 1
        for kw, (cls, flag) in GENUS_MAP.items():
            if kw in acc[1]:
                new_name = f"{kw}|{cls}|{flag}|{acc[0]}"
                fout.write(f">{new_name}\n")
                fout.write("".join(seq_lines))
                kept += 1
                break

print(f"[done] Kept {kept}/{total} sequences matching TARGET genera.")
print(f"[done] Written to: {GENUS_FA}")
PYEOF

echo "[$(date)] Building minimap2 index from genus-named FASTA..."
minimap2 -x map-hifi -d "${GENUS_IDX}" "${GENUS_FA}" -t "${NCPU}"
echo "[$(date)] Index written to: ${GENUS_IDX}"
echo "[$(date)] Done."
