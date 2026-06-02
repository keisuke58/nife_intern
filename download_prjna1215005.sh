#!/bin/bash
# download_prjna1215005.sh
# PRJNA1215005: Peri-implantitis 7 patients × 3 timepoints (baseline, 3m, 6m post-treatment)
# Method: prefetch + fasterq-dump (SRA tools required)

set -e
OUTDIR="data/prjna1215005"
mkdir -p "$OUTDIR"

echo "Fetching SRA run list for PRJNA1215005..."
# Download run list via Entrez
esearch -db sra -query "PRJNA1215005[BioProject]" \
    | efetch -format runinfo \
    | grep -v "^Run" \
    | awk -F',' '{print $1}' > "$OUTDIR/run_list.txt"

N=$(wc -l < "$OUTDIR/run_list.txt")
echo "Found $N runs"

# Download FASTQs
mkdir -p "$OUTDIR/fastq"
while read RUN; do
    echo "  Downloading $RUN..."
    prefetch "$RUN" -O "$OUTDIR/prefetch"
    fasterq-dump "$OUTDIR/prefetch/$RUN" -O "$OUTDIR/fastq" --split-files --threads 4
    gzip "$OUTDIR/fastq/${RUN}"*.fastq 2>/dev/null || true
done < "$OUTDIR/run_list.txt"

echo "Done. FASTQs in $OUTDIR/fastq/"
echo "Next: run vsearch pipeline → build_guild_phi.py"
