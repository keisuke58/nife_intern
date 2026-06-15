#!/usr/bin/env python3
"""
build_guild_phi_1215005.py
Convert vsearch SILVA blast6 output → guild-level phi array for PRJNA1215005
(the "Next:" step in jobs/download_prjna1215005.sh).

Cohort: Anuntakarun et al. 2025, Int Dent J 75(5):100951; PRJNA1215005 — 7 peri-
implantitis patients, paired peri-implantitis + healthy-implant sites, subgingival
biofilm, 16S V3-V4 (Bakt_341F/805R), at baseline / 3 mo / 6 mo post-treatment.
The matched-design (peri-implant, longitudinal) counterpart to Dieckow.

PREREQUISITE — a manifest data/prjna1215005/runs_parsed.tsv with columns
    run    patient    timepoint    site
must exist. Build it from the SRA runinfo once the reads are released:
    esearch -db sra -query "PRJNA1215005[BioProject]" | efetch -format runinfo
then parse SampleName/title for patient + visit (baseline/3mo/6mo) + site
(peri-implantitis / healthy). timepoint values must be one of TIMEPOINTS below.

NOTE (2026-06-16): PRJNA1215005 was NOT yet public on SRA or ENA — run this only
after the reads are released (re-check ENA: ena/browser/view/PRJNA1215005).

Usage:
    python scripts/preprocessing/build_guild_phi_1215005.py \
        --blast6 data/prjna1215005/all_taxonomy.blast6 [--site peri-implantitis]
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import argparse, json
import numpy as np
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--blast6', required=True,
                    help='vsearch blast6out file (all samples, reads labeled run.N)')
parser.add_argument('--site', default='peri-implantitis',
                    help="site to keep (manifest col 4); 'all' aggregates sites. "
                         "default keeps the diseased sites for the disease-dynamics comparison")
args = parser.parse_args()

DATA     = Path(__file__).resolve().parents[2] / 'data' / 'prjna1215005'
MANIFEST = DATA / 'runs_parsed.tsv'

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
N_G = len(GUILD_ORDER)
GUILD_IDX = {g: i for i, g in enumerate(GUILD_ORDER)}

# SILVA class → guild mapping (identical to the Duran-Pinedo build, the canonical map)
CLASS_TO_GUILD = {
    'Actinobacteria': 'Actinobacteria', 'Actinobacteria (class)': 'Actinobacteria',
    'Actinomycetia': 'Actinobacteria', 'Actinomycetes': 'Actinobacteria',
    'Bacilli': 'Bacilli',
    'Bacteroidia': 'Bacteroidia', 'Bacteroidetes': 'Bacteroidia',
    'Bacteroidia (class)': 'Bacteroidia',
    'Betaproteobacteria': 'Betaproteobacteria',
    'Clostridia': 'Clostridia', 'Erysipelotrichia': 'Clostridia',
    'Coriobacteriia': 'Coriobacteriia', 'Coriobacteriales': 'Coriobacteriia',
    'Fusobacteriia': 'Fusobacteriia', 'Fusobacteriales': 'Fusobacteriia',
    'Gammaproteobacteria': 'Gammaproteobacteria',
    'Negativicutes': 'Negativicutes', 'Selenomonadales': 'Negativicutes',
    'Veillonellales-Selenomonadales': 'Negativicutes', 'Veillonellales': 'Negativicutes',
}


def silva_to_guild(subject: str) -> str:
    """Extract guild from SILVA subject field (blast6 col 2)."""
    parts = subject.split(None, 1)
    if len(parts) < 2:
        return 'Other'
    ranks = [r.strip() for r in parts[1].split(';')]
    for r in ranks[2:5]:                      # class (2) / order (3) first
        if r in CLASS_TO_GUILD:
            return CLASS_TO_GUILD[r]
    for r in ranks:
        if r in CLASS_TO_GUILD:
            return CLASS_TO_GUILD[r]
    return 'Other'


# ── Timepoints: the three peri-implant visits ─────────────────────────────────
TIMEPOINTS = ['baseline', '3mo', '6mo']
TP_IDX = {tp: i for i, tp in enumerate(TIMEPOINTS)}

# ── Load manifest: run → (patient, timepoint, site) ───────────────────────────
if not MANIFEST.exists():
    raise SystemExit(
        f"manifest not found: {MANIFEST}\n"
        "Build it from the SRA runinfo (run\\tpatient\\ttimepoint\\tsite); see the\n"
        "module docstring. PRJNA1215005 must be released first (SRA/ENA).")

run_to_meta = {}
for line in open(MANIFEST).read().strip().split('\n')[1:]:
    parts = line.split('\t')
    if len(parts) < 4:
        continue
    run, pid, tp, site = parts[0], parts[1], parts[2], parts[3]
    if args.site != 'all' and site != args.site:
        continue
    if tp not in TP_IDX:
        continue
    run_to_meta[run] = (pid, tp)

PATIENTS = sorted(set(m[0] for m in run_to_meta.values()))
N_P = len(PATIENTS)
PID_IDX = {p: i for i, p in enumerate(PATIENTS)}
print(f"Patients: {PATIENTS}  N={N_P}  | site filter: {args.site}")
print(f"Timepoints: {TIMEPOINTS}")

# ── Parse blast6 output (read label: 'SRRxxxxxxx.1234' = run + '.' + read no) ──
counts = defaultdict(lambda: defaultdict(lambda: np.zeros(N_G)))
print(f"Reading {args.blast6} ...")
with open(args.blast6) as f:
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 2:
            continue
        query, subject = cols[0], cols[1]
        run = query.rsplit('.', 1)[0]
        if run not in run_to_meta:
            continue
        pid, tp = run_to_meta[run]
        counts[pid][tp][GUILD_IDX[silva_to_guild(subject)]] += 1

# ── Build phi array (N_P × 3 × 10), keyed to GUILD_ORDER ──────────────────────
phi = np.zeros((N_P, len(TIMEPOINTS), N_G))
for pid, pi in PID_IDX.items():
    for tp, ti in TP_IDX.items():
        c = counts[pid][tp]; total = c.sum()
        if total > 0:
            phi[pi, ti] = c / total
        else:
            phi[pi, ti] = np.ones(N_G) / N_G
            print(f"  WARNING: P{pid} {tp}: 0 reads → uniform")

print(f"\nphi shape: {phi.shape}")
print(f"Row sums (≈1.0): min={phi.sum(2).min():.4f}  max={phi.sum(2).max():.4f}")

np.save(DATA / 'phi_guild.npy', phi)
meta = {
    'n_patients': N_P, 'n_timepoints': len(TIMEPOINTS), 'n_guilds': N_G,
    'patients': PATIENTS, 'timepoints': TIMEPOINTS, 'guilds': GUILD_ORDER,
    'site_filter': args.site,
    'source': 'PRJNA1215005 — Anuntakarun et al. 2025, Int Dent J 75(5):100951',
    'disease': 'peri-implantitis (subgingival biofilm, paired diseased/healthy sites)',
    'sampling': 'baseline / 3 mo / 6 mo post-treatment; 16S V3-V4',
    'notes': 'Matched-design (peri-implant, longitudinal) cross-cohort partner to Dieckow.',
}
json.dump(meta, open(DATA / 'metadata.json', 'w'), indent=2)
print(f"\nSaved: {DATA}/phi_guild.npy  +  metadata.json")
print("\nTop guild per patient at baseline:")
for pi, pid in enumerate(PATIENTS):
    print(f"  P{pid}: {GUILD_ORDER[phi[pi,0].argmax()]} ({phi[pi,0].max():.2%})")
