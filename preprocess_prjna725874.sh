#!/bin/bash
#PBS -N preprocess_16s_725874
#PBS -l nodes=frontale02:ppn=12
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/data/prjna725874/preprocess.log

set -euo pipefail

NIFE=/home/nishioka/IKM_Hiwi/nife
DATA=$NIFE/data/prjna725874
SILVA_FA=$NIFE/data/silva_db/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz
VSEARCH=$(which vsearch)
PYTHON=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python
THREADS=12
MANIFEST=$DATA/runs_parsed.tsv

mkdir -p $DATA/fastq $DATA/merged $DATA/filtered

echo "=== PRJNA725874 16S preprocessing  $(date) ==="
echo "  vsearch: $VSEARCH"
echo "  SILVA:   $SILVA_FA"

# ── Step 1: Download FASTQs ────────────────────────────────────────────────────
echo ""
echo "=== Step 1: Download FASTQs ==="
download_one() {
    run=$1; ftp1=$2; ftp2=$3
    f1=$DATA/fastq/${run}_1.fastq.gz
    f2=$DATA/fastq/${run}_2.fastq.gz
    [[ ! -f $f1 ]] && wget -q -O $f1 "ftp://$ftp1" || true
    [[ ! -f $f2 && -n $ftp2 ]] && wget -q -O $f2 "ftp://$ftp2" || true
}
export -f download_one
export DATA

tail -n +2 $MANIFEST | \
    awk '{print $1"\t"$6"\t"$7}' | \
    xargs -P $THREADS -I{} bash -c 'download_one {}'
echo "Download done: $(date)"

# ── Step 2: Merge PE + quality filter per sample (parallel) ───────────────────
echo ""
echo "=== Step 2: Merge + QC ==="
process_sample() {
    run=$1
    DATA=$2
    VSEARCH=$3
    f1=$DATA/fastq/${run}_1.fastq.gz
    f2=$DATA/fastq/${run}_2.fastq.gz
    out=$DATA/filtered/${run}.fasta.gz
    [[ -f $out ]] && return
    [[ ! -f $f1 ]] && return
    tmp_merged=$(mktemp /tmp/merged_XXXX.fastq.gz)
    $VSEARCH --fastq_mergepairs $f1 --reverse $f2 \
        --fastqout $tmp_merged \
        --fastq_minovlen 50 --fastq_maxdiffs 5 \
        --quiet 2>/dev/null || true
    $VSEARCH --fastq_filter $tmp_merged \
        --fastq_maxee 1.0 --fastq_minlen 200 --fastq_maxlen 650 \
        --fastaout - \
        --relabel "${run}." \
        --quiet 2>/dev/null | gzip > $out || true
    rm -f $tmp_merged
}
export -f process_sample
export DATA VSEARCH

tail -n +2 $MANIFEST | awk '{print $1}' | \
    xargs -P $THREADS -I{} bash -c 'process_sample {} "$DATA" "$VSEARCH"'
echo "QC done: $(date)"

# ── Step 3: Concatenate all filtered reads ─────────────────────────────────────
echo ""
echo "=== Step 3: Concatenate ==="
ALL_READS=$DATA/all_reads.fasta.gz
if [[ ! -f $ALL_READS ]]; then
    cat $DATA/filtered/*.fasta.gz > $ALL_READS
fi
n_reads=$(zcat $ALL_READS | grep -c "^>" || true)
echo "  Total reads: $n_reads"

# ── Step 4: Chimera removal ───────────────────────────────────────────────────
echo ""
echo "=== Step 4: Chimera removal ==="
NOCHIM=$DATA/all_reads_nochim.fasta.gz
if [[ ! -f $NOCHIM ]]; then
    $VSEARCH --uchime_ref $ALL_READS \
        --db $SILVA_FA \
        --nonchimeras - \
        --quiet \
        --threads $THREADS 2>/dev/null | gzip > $NOCHIM
fi
n_clean=$(zcat $NOCHIM | grep -c "^>" || true)
echo "  Reads after chimera removal: $n_clean"

# ── Step 5: Classify all reads against SILVA ──────────────────────────────────
echo ""
echo "=== Step 5: Classify against SILVA ==="
TAX_OUT=$DATA/all_taxonomy.blast6
if [[ ! -f $TAX_OUT ]]; then
    $VSEARCH --usearch_global $NOCHIM \
        --db $SILVA_FA \
        --id 0.80 \
        --maxaccepts 1 \
        --maxrejects 32 \
        --blast6out $TAX_OUT \
        --notmatched /dev/null \
        --threads $THREADS \
        --quiet 2>/dev/null
fi
echo "  Classified: $(wc -l < $TAX_OUT) reads"

# ── Step 6: Build guild phi array ─────────────────────────────────────────────
echo ""
echo "=== Step 6: Build guild phi array ==="
$PYTHON $NIFE/build_guild_phi_725874.py --blast6 $TAX_OUT
echo ""
echo "=== All done: $(date) ==="
