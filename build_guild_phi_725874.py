#!/usr/bin/env python3
"""
build_guild_phi_725874.py
Convert vsearch SILVA blast6 output → guild-level phi array for PRJNA725874

Usage:
    python build_guild_phi_725874.py --blast6 data/prjna725874/all_taxonomy.blast6
"""
import argparse, re, json
import numpy as np
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--blast6', required=True,
                    help='vsearch blast6out file (all samples, reads labeled run.N)')
args = parser.parse_args()

DATA     = Path(__file__).parent / 'data' / 'prjna725874'
MANIFEST = DATA / 'runs_parsed.tsv'

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
N_G = len(GUILD_ORDER)
GUILD_IDX = {g: i for i, g in enumerate(GUILD_ORDER)}

# SILVA class → guild mapping
CLASS_TO_GUILD = {
    # Actinobacteria
    'Actinobacteria': 'Actinobacteria', 'Actinobacteria (class)': 'Actinobacteria',
    'Actinomycetia': 'Actinobacteria', 'Actinomycetes': 'Actinobacteria',
    # Bacilli
    'Bacilli': 'Bacilli',
    # Bacteroidia
    'Bacteroidia': 'Bacteroidia', 'Bacteroidetes': 'Bacteroidia',
    'Bacteroidia (class)': 'Bacteroidia',
    # Betaproteobacteria
    'Betaproteobacteria': 'Betaproteobacteria',
    # Clostridia
    'Clostridia': 'Clostridia', 'Erysipelotrichia': 'Clostridia',
    # Coriobacteriia
    'Coriobacteriia': 'Coriobacteriia', 'Coriobacteriales': 'Coriobacteriia',
    # Fusobacteriia
    'Fusobacteriia': 'Fusobacteriia', 'Fusobacteriales': 'Fusobacteriia',
    # Gammaproteobacteria
    'Gammaproteobacteria': 'Gammaproteobacteria',
    # Negativicutes
    'Negativicutes': 'Negativicutes', 'Selenomonadales': 'Negativicutes',
    'Veillonellales-Selenomonadales': 'Negativicutes',
    'Veillonellales': 'Negativicutes',
}


def silva_to_guild(subject: str) -> str:
    """Extract guild from SILVA subject field (blast6 col 2)."""
    # SILVA subject: "AB000123.1.1500 Bacteria;Firmicutes;Bacilli;Lactobacillales;..."
    parts = subject.split(None, 1)
    if len(parts) < 2:
        return 'Other'
    ranks = [r.strip() for r in parts[1].split(';')]
    # Check class (index 2) and order (index 3) first
    for r in ranks[2:5]:
        if r in CLASS_TO_GUILD:
            return CLASS_TO_GUILD[r]
    for r in ranks:
        if r in CLASS_TO_GUILD:
            return CLASS_TO_GUILD[r]
    return 'Other'


# ── Load manifest: run → (pid, tp) ────────────────────────────────────────────
TIMEPOINTS = ['00', '02', '04', '06', '08', '10', '12']
TP_IDX = {tp: i for i, tp in enumerate(TIMEPOINTS)}

run_to_meta = {}
lines = open(MANIFEST).read().strip().split('\n')
for line in lines[1:]:
    parts = line.split('\t')
    if len(parts) < 5: continue
    run, pid, tooth, stype, tp = parts[0], parts[1], parts[2], parts[3], parts[4]
    run_to_meta[run] = (pid, tp)

PATIENTS = sorted(set(m[0] for m in run_to_meta.values()), key=int)
N_P = len(PATIENTS)
PID_IDX = {p: i for i, p in enumerate(PATIENTS)}

print(f"Patients: {PATIENTS}  N={N_P}")
print(f"Timepoints: {TIMEPOINTS}")

# ── Parse blast6 output ────────────────────────────────────────────────────────
# read label format: "SRR14643440.1234"  (run + "." + read_number)
counts = defaultdict(lambda: defaultdict(lambda: np.zeros(N_G)))

print(f"Reading {args.blast6} ...")
with open(args.blast6) as f:
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 2: continue
        query, subject = cols[0], cols[1]
        # Extract run ID from query label
        run = query.rsplit('.', 1)[0]
        if run not in run_to_meta: continue
        pid, tp = run_to_meta[run]
        if tp not in TP_IDX: continue  # skip Post
        guild = silva_to_guild(subject)
        counts[pid][tp][GUILD_IDX[guild]] += 1

# ── Build phi array ───────────────────────────────────────────────────────────
phi = np.zeros((N_P, len(TIMEPOINTS), N_G))
for pid, pi in PID_IDX.items():
    for tp, ti in TP_IDX.items():
        c = counts[pid][tp]
        total = c.sum()
        if total > 0:
            phi[pi, ti] = c / total
        else:
            phi[pi, ti] = np.ones(N_G) / N_G
            print(f"  WARNING: P{pid} T{tp}: 0 reads → uniform")

print(f"\nphi shape: {phi.shape}")
print(f"Row sums (should be 1.0): min={phi.sum(2).min():.4f}  max={phi.sum(2).max():.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
np.save(DATA / 'phi_guild.npy', phi)
meta = {
    'n_patients': N_P, 'n_timepoints': len(TIMEPOINTS), 'n_guilds': N_G,
    'patients': PATIENTS,
    'timepoints': TIMEPOINTS,
    'guilds': GUILD_ORDER,
    'source': 'PRJNA725874 — Botelho et al. 2021 BMC Biology',
    'disease': 'periodontitis (subgingival plaque)',
    'interval_months': 2,
    'notes': 'Aggregated across all teeth and site types (Stable/Progressing/Fluctuant).',
}
json.dump(meta, open(DATA / 'metadata.json', 'w'), indent=2)
print(f"\nSaved: {DATA}/phi_guild.npy  +  metadata.json")

# Top guild summary
print("\nTop guild per patient at T00:")
for pi, pid in enumerate(PATIENTS):
    top = GUILD_ORDER[phi[pi, 0].argmax()]
    print(f"  P{pid}: {top} ({phi[pi,0].max():.2%})")
