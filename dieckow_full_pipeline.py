#!/usr/bin/env python3
"""
Optimized Dieckow pipeline:
  1. Map ALL subreads to SILVA → whitelist oral genera (eHOMD) vs contaminants
  2. Group non-contaminant subreads by ZMW
  3. SPOA consensus only for real ZMWs (~3% of total)
  4. VSEARCH taxonomy at 97% identity

Usage (PBS job or local):
  python dieckow_full_pipeline.py \
      --fastq /path/to/ERR13166576.fastq.gz \
      --sample A_3 \
      --out results/ \
      --threads 12
"""
import argparse, gzip, subprocess, os, collections, tempfile, json
from pathlib import Path

MINIMAP2 = '/home/nishioka/miniconda3/envs/metaphlan4/bin/minimap2'
MMI      = '/home/nishioka/IKM_Hiwi/nife/data/silva_db/silva138_ssu_nr99.mmi'
SILVA_FA = '/home/nishioka/IKM_Hiwi/nife/data/silva_db/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz'
VSEARCH  = 'vsearch'
SPOA     = 'spoa'

# Oral microbiome whitelist (eHOMD-based) — only these genera are kept
ORAL_GENERA = {
    # Firmicutes
    'Streptococcus', 'Veillonella', 'Gemella', 'Granulicatella', 'Abiotrophia',
    'Lachnoanaerobaculum', 'Oribacterium', 'Shuttleworthia', 'Solobacterium',
    'Catonella', 'Johnsonella', 'Mogibacterium', 'Filifactor', 'Parvimonas',
    'Peptostreptococcus', 'Peptoniphilus', 'Finegoldia', 'Anaerococcus',
    'Selenomonas', 'Centipeda', 'Megasphaera', 'Dialister', 'Mitsuokella',
    'Eubacterium', 'Lancefieldella', 'Aerococcus', 'Stomatobaculum',
    'Limosilactobacillus', 'Lactiplantibacillus', 'Lacticaseibacillus',
    # Actinobacteria
    'Actinomyces', 'Rothia', 'Schaalia', 'Bifidobacterium', 'Slackia',
    'Atopobium', 'Olsenella', 'Cryptobacterium', 'Corynebacterium',
    # Bacteroidetes
    'Prevotella', 'Capnocytophaga', 'Porphyromonas', 'Tannerella',
    'Alloprevotella', 'Odoribacter', 'Paraprevotella',
    # Fusobacteria
    'Fusobacterium', 'Leptotrichia',
    # Proteobacteria (oral only)
    'Haemophilus', 'Aggregatibacter', 'Eikenella', 'Neisseria', 'Kingella',
    'Simonsiella', 'Cardiobacterium', 'Campylobacter', 'Wolinella',
    # Spirochaetes
    'Treponema',
    # Synergistetes
    'Jonquetella', 'Pyramidobacter',
}


def revcomp(seq):
    comp = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
    return ''.join(comp.get(b,'N') for b in reversed(seq.upper()))


def load_all_subreads(fastq_path):
    """Load all subreads grouped by ZMW."""
    zmw_reads = collections.defaultdict(list)
    opener = gzip.open if str(fastq_path).endswith('.gz') else open
    with opener(fastq_path, 'rt') as f:
        while True:
            h = f.readline()
            if not h: break
            seq = f.readline().strip()
            f.readline(); f.readline()
            parts = h.strip().split()
            zmw_id = parts[1].split('/')[1] if len(parts) >= 2 else 'unk'
            zmw_reads[zmw_id].append(seq)
    return zmw_reads


def align_subreads_to_silva(zmw_reads, tmp_dir, threads):
    """Write all subreads to FASTA and align to SILVA."""
    all_sub_fa = os.path.join(tmp_dir, 'all_subs.fasta')
    n_seqs = 0
    with open(all_sub_fa, 'w') as out:
        for zmw_id, reads in zmw_reads.items():
            for i, seq in enumerate(reads):
                out.write(f">{zmw_id}_{i}\n{seq}\n")
                n_seqs += 1
    print(f"  Written {n_seqs} subreads to FASTA")

    sam_out = os.path.join(tmp_dir, 'all_subs.sam')
    print(f"  Aligning to SILVA ({threads} threads)...")
    subprocess.run(
        [MINIMAP2, '-a', '--secondary=no', f'-t{threads}', MMI, all_sub_fa,
         '-o', sam_out],
        check=True, capture_output=True
    )
    return sam_out


def classify_zmw_contaminant(sam_path, silva_fa):
    """
    For each subread, get SILVA taxonomy from alignment.
    Return: dict zmw_id → {'contaminant': bool, 'strands': list}
    """
    # Parse SAM → per-subread (rname, strand)
    sub_info = {}  # zmw_subid → (rname, strand)
    with open(sam_path) as f:
        for line in f:
            if line.startswith('@'): continue
            fields = line.split('\t')
            flag = int(fields[1])
            qname = fields[0]
            if flag & 4:
                sub_info[qname] = ('*', -1)
            else:
                rname = fields[2]
                strand = 1 if (flag & 16) else 0
                sub_info[qname] = (rname, strand)

    # Get taxonomy for all reference names
    rnames_needed = {r for r, _ in sub_info.values() if r != '*'}
    silva_tax = {}
    print(f"  Looking up {len(rnames_needed)} SILVA references...")
    with gzip.open(silva_fa, 'rt') as f:
        for line in f:
            if not rnames_needed: break
            if line.startswith('>'):
                p = line[1:].strip().split(' ', 1)
                acc = p[0]
                if acc in rnames_needed:
                    silva_tax[acc] = p[1] if len(p) > 1 else ''
                    rnames_needed.discard(acc)

    # Per-ZMW: determine if contaminant based on majority vote
    zmw_data = collections.defaultdict(lambda: {'rnames': [], 'strands': []})
    for sub_id, (rname, strand) in sub_info.items():
        zmw_id = sub_id.rsplit('_', 1)[0]
        zmw_data[zmw_id]['rnames'].append(rname)
        zmw_data[zmw_id]['strands'].append(strand)

    zmw_classification = {}
    for zmw_id, data in zmw_data.items():
        # Count oral vs non-oral reads (whitelist approach)
        oral_count = 0
        nonoral_count = 0
        for rname, strand in zip(data['rnames'], data['strands']):
            if rname == '*': continue
            tax = silva_tax.get(rname, '')
            parts = [x.strip() for x in tax.rstrip(';').split(';')]
            genus = parts[-2] if len(parts) >= 2 else ''
            if genus in ORAL_GENERA:
                oral_count += 1
            else:
                nonoral_count += 1
        is_contam = oral_count <= nonoral_count  # require majority oral
        zmw_classification[zmw_id] = {
            'contaminant': is_contam,
            'strands': data['strands'],
            'n_mapped': oral_count + nonoral_count,
        }

    return zmw_classification


def compute_ccs_for_real_zmws(zmw_reads, zmw_classification, tmp_dir):
    """Run SPOA only on non-contaminant ZMWs."""
    ccs_seqs = []
    for zmw_id, reads in zmw_reads.items():
        cls = zmw_classification.get(zmw_id, {'contaminant': True})
        if cls['contaminant']:
            continue

        strands = cls.get('strands', [])
        # Determine majority strand
        mapped_with_strand = [(i, seq, s) for i, (seq, s) in enumerate(zip(reads, strands))
                               if s != -1]
        if not mapped_with_strand:
            continue
        fwd = sum(1 for _, _, s in mapped_with_strand if s == 0)
        majority = 0 if fwd >= (len(mapped_with_strand) - fwd) else 1

        oriented = [revcomp(seq) if s != majority else seq
                    for _, seq, s in mapped_with_strand]

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
    return ccs_seqs


def run_vsearch_taxonomy(ccs_seqs, tmp_dir, threads):
    """VSEARCH taxonomy at 97% identity."""
    ccs_fa = os.path.join(tmp_dir, 'ccs_real.fasta')
    with open(ccs_fa, 'w') as f:
        for zmw_id, n, seq in ccs_seqs:
            f.write(f">{zmw_id}\n{seq}\n")

    blast6 = os.path.join(tmp_dir, 'ccs_real.blast6')
    subprocess.run([VSEARCH, '--usearch_global', ccs_fa,
        '--db', SILVA_FA, '--id', '0.97', '--strand', 'both',
        '--top_hits_only', '--blast6out', blast6,
        '--threads', str(threads)],
        capture_output=True, check=False)

    hits = {}
    if os.path.exists(blast6):
        with open(blast6) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 3:
                    hits[p[0]] = (p[1], float(p[2]))

    rnames_needed = {r for r, _ in hits.values()}
    silva_tax = {}
    with gzip.open(SILVA_FA, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                p = line[1:].strip().split(' ', 1)
                if p[0] in rnames_needed:
                    silva_tax[p[0]] = p[1] if len(p) > 1 else ''

    genus_counts = collections.Counter()
    identities = []
    for zmw_id, (rname, pident) in hits.items():
        tax = silva_tax.get(rname, '')
        parts = [x.strip() for x in tax.rstrip(';').split(';')]
        # SILVA: Domain;Phylum;Class;Order;Family;Genus;Species
        genus = parts[-2] if len(parts) >= 2 else (parts[0] if parts else 'Unknown')
        genus_counts[genus] += 1
        identities.append(pident)
    return genus_counts, identities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastq', required=True)
    parser.add_argument('--sample', default='unknown')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[{args.sample}] Loading subreads...")
        zmw_reads = load_all_subreads(args.fastq)
        total_zmw = len(zmw_reads)
        total_sub = sum(len(v) for v in zmw_reads.values())
        print(f"[{args.sample}] {total_zmw} ZMWs, {total_sub} subreads")

        print(f"[{args.sample}] Aligning to SILVA...")
        sam_path = align_subreads_to_silva(zmw_reads, tmp_dir, args.threads)

        print(f"[{args.sample}] Classifying ZMWs (contaminant vs real)...")
        zmw_cls = classify_zmw_contaminant(sam_path, SILVA_FA)
        n_real = sum(1 for v in zmw_cls.values() if not v['contaminant'])
        n_contam = sum(1 for v in zmw_cls.values() if v['contaminant'])
        print(f"[{args.sample}] Real: {n_real}, Contaminant: {n_contam} "
              f"({n_contam/total_zmw*100:.1f}% contamination)")

        print(f"[{args.sample}] Computing SPOA consensus for {n_real} real ZMWs...")
        ccs_seqs = compute_ccs_for_real_zmws(zmw_reads, zmw_cls, tmp_dir)
        print(f"[{args.sample}] Got {len(ccs_seqs)} CCS sequences")

        print(f"[{args.sample}] VSEARCH taxonomy...")
        genus_counts, identities = run_vsearch_taxonomy(ccs_seqs, tmp_dir, args.threads)
        if identities:
            print(f"[{args.sample}] Identity: mean={sum(identities)/len(identities):.1f}%")

    # Write output
    total = sum(genus_counts.values())
    tsv = os.path.join(args.out, f"{args.sample}_taxonomy.tsv")
    with open(tsv, 'w') as f:
        f.write("sample\tgenus\tcount\tpercent\n")
        for genus, cnt in genus_counts.most_common():
            f.write(f"{args.sample}\t{genus}\t{cnt}\t{cnt/total*100:.2f}\n")

    # JSON summary
    summary = {
        'sample': args.sample,
        'total_zmw': total_zmw,
        'contamination_fraction': n_contam / total_zmw if total_zmw else 0,
        'real_zmw': n_real,
        'ccs_sequences': len(ccs_seqs),
        'classified_reads': total,
        'mean_identity': sum(identities)/len(identities) if identities else 0,
        'top10_genera': dict(genus_counts.most_common(10)),
    }
    with open(os.path.join(args.out, f"{args.sample}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.sample}: {total} classified reads (after decontam) ===")
    for genus, cnt in genus_counts.most_common(15):
        print(f"  {genus:40s}: {cnt:4d} ({cnt/total*100:.1f}%)")
    print(f"Written: {tsv}")


if __name__ == '__main__':
    main()
