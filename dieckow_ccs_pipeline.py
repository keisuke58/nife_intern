#!/usr/bin/env python3
"""
Dieckow 2024 PacBio subread → CCS-like consensus → SILVA taxonomy pipeline

Usage:
  python dieckow_ccs_pipeline.py --fastq ERR13166576.fastq.gz --sample A_3 \
      --out results/A_3_taxonomy.tsv [--n-zmw 500]

Method:
  1. Group subreads by ZMW ID (from FASTQ header)
  2. Align all subreads to SILVA with minimap2 (--secondary=no, map-pb)
     to determine strand orientation per subread
  3. For each ZMW: orient all subreads to majority strand, run SPOA consensus
  4. VSEARCH --usearch_global at 97% identity against SILVA NR99
  5. Aggregate genus-level taxonomy
"""

import argparse, gzip, subprocess, os, collections, tempfile, json
from pathlib import Path

MINIMAP2 = '/home/nishioka/miniconda3/envs/metaphlan4/bin/minimap2'
MMI      = '/home/nishioka/IKM_Hiwi/nife/data/silva_db/silva138_ssu_nr99.mmi'
SILVA_FA = '/home/nishioka/IKM_Hiwi/nife/data/silva_db/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz'
VSEARCH  = 'vsearch'
SPOA     = 'spoa'

def revcomp(seq):
    comp = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
    return ''.join(comp.get(b,'N') for b in reversed(seq.upper()))


def load_zmw_reads(fastq_path, n_zmw):
    """Load subreads grouped by ZMW from PacBio FASTQ."""
    zmw_reads = collections.OrderedDict()
    opener = gzip.open if str(fastq_path).endswith('.gz') else open
    with opener(fastq_path, 'rt') as f:
        while True:
            h = f.readline()
            if not h: break
            seq = f.readline().strip()
            f.readline(); f.readline()
            parts = h.strip().split()
            zmw_id = parts[1].split('/')[1] if len(parts) >= 2 else 'unk'
            if zmw_id not in zmw_reads:
                if len(zmw_reads) >= n_zmw:
                    break
                zmw_reads[zmw_id] = []
            zmw_reads[zmw_id].append(seq)
    return zmw_reads


def get_subread_strands(zmw_reads, tmp_dir):
    """Map all subreads to SILVA to determine strand orientation."""
    all_sub_fa = os.path.join(tmp_dir, 'all_subs.fasta')
    with open(all_sub_fa, 'w') as out:
        for zmw_id, reads in zmw_reads.items():
            for i, seq in enumerate(reads):
                out.write(f">{zmw_id}_{i}\n{seq}\n")

    result = subprocess.run(
        [MINIMAP2, '-a', '--secondary=no', '-t', '8', MMI, all_sub_fa],
        capture_output=True, text=True
    )
    sub_strand = {}
    for line in result.stdout.split('\n'):
        if not line.strip() or line.startswith('@'): continue
        fields = line.split('\t')
        flag = int(fields[1])
        qname = fields[0]
        sub_strand[qname] = -1 if (flag & 4) else (1 if (flag & 16) else 0)
    return sub_strand


def compute_ccs_per_zmw(zmw_reads, sub_strand, tmp_dir):
    """Orient subreads and run SPOA consensus for each ZMW."""
    ccs_seqs = []
    for zmw_id, reads in zmw_reads.items():
        mapped = [(i, seq) for i, seq in enumerate(reads)
                  if sub_strand.get(f"{zmw_id}_{i}", -1) != -1]
        if not mapped:
            continue
        fwd = sum(1 for i, _ in mapped if sub_strand.get(f"{zmw_id}_{i}", -1) == 0)
        majority = 0 if fwd >= (len(mapped) - fwd) else 1

        oriented = []
        for i, seq in mapped:
            strand = sub_strand.get(f"{zmw_id}_{i}", -1)
            if strand == -1:
                continue
            oriented.append(revcomp(seq) if strand != majority else seq)
        if not oriented:
            continue

        tmp_fa = os.path.join(tmp_dir, f'zmw_{zmw_id}.fasta')
        with open(tmp_fa, 'w') as f:
            for k, seq in enumerate(oriented):
                f.write(f">s{k}\n{seq}\n")
        r = subprocess.run([SPOA, tmp_fa], capture_output=True, text=True)
        if r.returncode == 0:
            lines = r.stdout.strip().split('\n')
            if len(lines) >= 2:
                ccs_seqs.append((zmw_id, len(oriented), lines[1]))
        os.unlink(tmp_fa)
    return ccs_seqs  # list of (zmw_id, n_passes, consensus_seq)


def run_vsearch_taxonomy(ccs_seqs, tmp_dir):
    """Run VSEARCH and return genus-level taxonomy counts."""
    ccs_fa = os.path.join(tmp_dir, 'ccs.fasta')
    with open(ccs_fa, 'w') as f:
        for zmw_id, n, seq in ccs_seqs:
            f.write(f">{zmw_id}\n{seq}\n")

    blast6 = os.path.join(tmp_dir, 'ccs.blast6')
    subprocess.run([VSEARCH, '--usearch_global', ccs_fa,
        '--db', SILVA_FA, '--id', '0.97', '--strand', 'both',
        '--top_hits_only', '--blast6out', blast6, '--threads', '8'],
        capture_output=True, check=False)

    # parse hits
    hits = {}
    if os.path.exists(blast6):
        with open(blast6) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 3:
                    hits[p[0]] = (p[1], float(p[2]))

    # SILVA taxonomy lookup
    rnames_needed = {r for r, _ in hits.values()}
    silva_tax = {}
    with gzip.open(SILVA_FA, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                p = line[1:].strip().split(' ', 1)
                if p[0] in rnames_needed:
                    silva_tax[p[0]] = p[1] if len(p) > 1 else 'Unknown'

    genus_counts = collections.Counter()
    identities = []
    for zmw_id, (rname, pident) in hits.items():
        tax = silva_tax.get(rname, '')
        parts = [x.strip() for x in tax.rstrip(';').split(';')]
        # SILVA: Domain;Phylum;Class;Order;Family;Genus;Species
        genus = parts[-2] if len(parts) >= 2 else parts[0] if parts else 'Unknown'
        genus_counts[genus] += 1
        identities.append(pident)

    return genus_counts, hits, identities


def main():
    parser = argparse.ArgumentParser(description='Dieckow PacBio subread → taxonomy')
    parser.add_argument('--fastq', required=True)
    parser.add_argument('--sample', default='unknown')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-zmw', type=int, default=500)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[{args.sample}] Loading {args.n_zmw} ZMWs...")
        zmw_reads = load_zmw_reads(args.fastq, args.n_zmw)
        print(f"[{args.sample}] Loaded {len(zmw_reads)} ZMWs, "
              f"{sum(len(v) for v in zmw_reads.values())} subreads")

        print(f"[{args.sample}] Determining subread strands via minimap2...")
        sub_strand = get_subread_strands(zmw_reads, tmp_dir)
        fwd = sum(v == 0 for v in sub_strand.values())
        rev = sum(v == 1 for v in sub_strand.values())
        unm = sum(v == -1 for v in sub_strand.values())
        print(f"[{args.sample}] Strand: {fwd} fwd, {rev} rev, {unm} unmapped")

        print(f"[{args.sample}] Computing SPOA consensus per ZMW...")
        ccs_seqs = compute_ccs_per_zmw(zmw_reads, sub_strand, tmp_dir)
        print(f"[{args.sample}] Got {len(ccs_seqs)} CCS consensus sequences")

        print(f"[{args.sample}] VSEARCH taxonomy at 97% identity...")
        genus_counts, hits, identities = run_vsearch_taxonomy(ccs_seqs, tmp_dir)
        if identities:
            print(f"[{args.sample}] Identity: mean={sum(identities)/len(identities):.1f}%, "
                  f"min={min(identities):.1f}%")

    # Write output
    os.makedirs(Path(args.out).parent, exist_ok=True)
    total = sum(genus_counts.values())
    rows = []
    for genus, cnt in genus_counts.most_common():
        rows.append(f"{args.sample}\t{genus}\t{cnt}\t{cnt/total*100:.2f}")

    with open(args.out, 'w') as f:
        f.write("sample\tgenus\tcount\tpercent\n")
        f.write('\n'.join(rows) + '\n')

    print(f"\n=== {args.sample} Top 15 genera ===")
    for genus, cnt in genus_counts.most_common(15):
        print(f"  {genus:45s}: {cnt:4d} ({cnt/total*100:.1f}%)")
    print(f"\nOutput: {args.out}")


if __name__ == '__main__':
    main()
