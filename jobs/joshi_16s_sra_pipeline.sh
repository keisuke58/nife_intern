#!/bin/bash
#PBS -N joshi_16s
#PBS -l nodes=1:ppn=8
#PBS -l walltime=120:00:00
#PBS -q default
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/joshi_16s_sra_pipeline.log

# ============================================================
# PRJNA1192962 full-length 16S (PacBio CCS) → genus profiles
# 177 AMPLICON runs → minimap2 (SILVA) → expanded genus JSON
# ============================================================

NIFE_DIR="/home/nishioka/IKM_Hiwi/nife"
DATA_DIR="${NIFE_DIR}/data"
SILVA_DIR="${DATA_DIR}/silva_db"
OUT_DIR="${NIFE_DIR}/results/joshi_16s_profiles"
RUNS_JSON="${DATA_DIR}/prjna1192962_amplicon_runs.json"
SILVA_FA="${SILVA_DIR}/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz"
SILVA_IDX="${SILVA_DIR}/silva138_genus_named.mmi"  # genus-named: RNAME=genus|class|flag|acc
SRA_TMP="/tmp/sra_joshi_$$"
NCPU=8

mkdir -p "${OUT_DIR}" "${SILVA_DIR}" "${SRA_TMP}"
trap "rm -rf ${SRA_TMP}" EXIT

echo "[$(date)] Starting Joshi 16S pipeline on $(hostname)"

# ---------- 1. conda setup ----------
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

# ---------- 2. Install sra-tools if missing ----------
if ! fasterq-dump --version &>/dev/null 2>&1; then
    echo "[$(date)] sra-tools not found, installing..."
    CONDA_NO_PLUGINS=true conda install -n metaphlan4 -c bioconda -c conda-forge \
        sra-tools -y --quiet 2>&1 | tail -5 || true
    conda activate metaphlan4
fi

# Last resort: download pre-built binary
if ! fasterq-dump --version &>/dev/null 2>&1; then
    echo "[$(date)] conda install failed, downloading sra-tools binary..."
    SRA_BIN="${DATA_DIR}/sratoolkit/bin"
    if [[ ! -f "${SRA_BIN}/fasterq-dump" ]]; then
        mkdir -p "${DATA_DIR}/sratoolkit"
        wget -q --timeout=120 --tries=3 \
            "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-centos_linux64.tar.gz" \
            -O "${DATA_DIR}/sratoolkit.tar.gz" \
        && tar -xzf "${DATA_DIR}/sratoolkit.tar.gz" -C "${DATA_DIR}/sratoolkit" \
               --strip-components=1 \
        && rm "${DATA_DIR}/sratoolkit.tar.gz" \
        && echo "[$(date)] sra-tools binary installed." \
        || echo "[$(date)] WARNING: sra-tools install failed."
    fi
    export PATH="${SRA_BIN}:${PATH}"
fi

if ! fasterq-dump --version &>/dev/null 2>&1; then
    echo "[$(date)] FATAL: fasterq-dump not available. Cannot continue."
    exit 1
fi

echo "[$(date)] fasterq-dump: $(fasterq-dump --version 2>&1 | head -1)"

# ---------- 3. SILVA index ----------
SILVA_URL="https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz"

if [[ ! -f "${SILVA_FA}" ]]; then
    echo "[$(date)] Downloading SILVA 138.1 SSU NR99 (~380MB)..."
    wget -q --show-progress --timeout=300 --tries=5 \
        -O "${SILVA_FA}" "${SILVA_URL}" \
    || { echo "[$(date)] SILVA download failed."; exit 1; }
fi

if [[ ! -f "${SILVA_IDX}" ]]; then
    echo "[$(date)] Building minimap2 index..."
    minimap2 -x map-hifi -d "${SILVA_IDX}" "${SILVA_FA}" -t "${NCPU}" \
    || { echo "[$(date)] minimap2 index build failed."; exit 1; }
fi

echo "[$(date)] Setup complete. Starting per-sample processing..."

# ---------- 4. Export vars for Python ----------
export RUNS_JSON OUT_DIR SILVA_IDX SRA_TMP NCPU

# ---------- 5. Download + map each sample ----------
python3 - <<'PYEOF'
import json, subprocess, os, sys
from pathlib import Path
from collections import defaultdict

RUNS_JSON = os.environ["RUNS_JSON"]
OUT_DIR   = os.environ["OUT_DIR"]
SILVA_IDX = os.environ["SILVA_IDX"]
SRA_TMP   = os.environ["SRA_TMP"]
NCPU      = int(os.environ["NCPU"])

TARGET = {
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

def parse_sam(sam_path):
    counts = defaultdict(int)
    total  = 0
    with open(sam_path) as f:
        for line in f:
            if line.startswith("@"):
                continue
            parts = line.split("\t")
            if int(parts[1]) & 4:
                continue
            rname = parts[2].lower()
            total += 1
            for kw, (cls, flag) in TARGET.items():
                if kw in rname:
                    counts[(kw, cls, flag)] += 1
                    break
    return counts, total

# Group runs by sample, pick largest
runs = json.load(open(RUNS_JSON))
by_sample = defaultdict(list)
for r in runs:
    by_sample[r['sample']].append(r)
best = {s: max(rs, key=lambda x: x['size_mb']) for s, rs in by_sample.items()}

print(f"[info] {len(best)} unique samples to process.", flush=True)
errors = []

for samp, run in sorted(best.items()):
    srr   = run['srr']
    out_f = Path(OUT_DIR) / f"{samp}.json"
    if out_f.exists():
        print(f"  {samp}: skip (done)", flush=True)
        continue

    print(f"\n[{samp}] {srr} ({run['size_mb']:.0f} MB)...", flush=True)
    sra_dir  = Path(SRA_TMP) / srr
    fastq_f  = Path(SRA_TMP) / f"{srr}.fastq"
    sam_f    = Path(SRA_TMP) / f"{srr}.sam"
    sra_dir.mkdir(exist_ok=True)

    try:
        # prefetch
        r1 = subprocess.run(
            ["prefetch", "--output-directory", SRA_TMP, srr],
            capture_output=True, text=True, timeout=1800)
        if r1.returncode != 0:
            raise RuntimeError(f"prefetch: {r1.stderr[-300:]}")

        # find the actual .sra or .sralite file
        sra_files = list(sra_dir.glob("*.sra")) + list(sra_dir.glob("*.sralite"))
        if not sra_files:
            raise RuntimeError(f"No .sra/.sralite file found in {sra_dir}")
        sra_file = sra_files[0]

        # fasterq-dump
        r2 = subprocess.run(
            ["fasterq-dump", "--outfile", str(fastq_f),
             "--skip-technical", "--threads", str(NCPU), str(sra_file)],
            capture_output=True, text=True, timeout=1800)
        if r2.returncode != 0:
            raise RuntimeError(f"fasterq-dump: {r2.stderr[-300:]}")
        if not fastq_f.exists():
            raise RuntimeError("fasterq-dump produced no output")

        # minimap2
        with open(sam_f, "w") as sam_out:
            r3 = subprocess.run(
                ["minimap2", "-ax", "map-hifi", "-t", str(NCPU),
                 "--secondary=no", SILVA_IDX, str(fastq_f)],
                stdout=sam_out, stderr=subprocess.PIPE,
                text=True, timeout=3600)
        if r3.returncode != 0:
            raise RuntimeError(f"minimap2: {r3.stderr[-300:]}")

        counts, total = parse_sam(str(sam_f))
        profile = {
            "sample": samp, "srr": srr, "total_mapped": total,
            "genera": {f"{kw}|{cls}|{flag}": int(cnt)
                       for (kw, cls, flag), cnt in counts.items()},
        }
        out_f.write_text(json.dumps(profile, indent=2))
        dys = sum(v for (kw,cls,flag),v in counts.items() if flag=="dys")
        com = sum(v for (kw,cls,flag),v in counts.items() if flag=="com")
        print(f"  OK: {total} mapped reads, dys={dys} com={com}", flush=True)

    except Exception as e:
        print(f"  ERROR [{samp}]: {e}", flush=True)
        errors.append(samp)

    finally:
        import shutil
        shutil.rmtree(sra_dir, ignore_errors=True)
        fastq_f.unlink(missing_ok=True)
        sam_f.unlink(missing_ok=True)

print(f"\n[done] Processed {len(best)-len(errors)}/{len(best)} samples.", flush=True)
if errors:
    print(f"[warn] Failed samples: {errors}", flush=True)
PYEOF

# ---------- 6. GDI comparison ----------
echo "[$(date)] Running GDI comparison..."
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
    "${NIFE_DIR}/scripts/analysis/joshi_gdi_expanded_check.py" \
|| echo "[$(date)] GDI script failed (check manually)"

echo "[$(date)] Pipeline complete."
