#!/usr/bin/env python3
"""Aggregate AGORA W=1.0 LOO-CV folds and compare with L1+L2 baseline."""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json
import numpy as np
from pathlib import Path

CR = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr')
OTU = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_otu')
PATIENTS = list('ABCDEFGHKL')

# Load CT labels
ct_map = json.load(open(OTU / 'community_types.json'))['patient_ct']

# Load AGORA W=1.0 LOO folds
agora_folds = {}
missing = []
for i, p in enumerate(PATIENTS):
    f = CR / f'loo_expanded_agora_w1p0_fold{i}.json'
    if f.exists():
        d = json.load(open(f))
        agora_folds[p] = d['rmse']
    else:
        missing.append(p)

if missing:
    print(f'Missing folds: {missing}')
    print(f'Completed: {list(agora_folds.keys())}')

if not agora_folds:
    print('No AGORA LOO folds found.')
    exit(1)

# Load L1+L2 baseline LOO
l12_folds = {}
for i, p in enumerate(PATIENTS):
    f = CR / f'loo_expanded_fold{i}.json'
    if f.exists():
        d = json.load(open(f))
        l12_folds[p] = d['rmse']

common_pats = sorted(set(agora_folds) & set(l12_folds))
all_pats = [p for p in PATIENTS if p in agora_folds]

agora_rmses = [agora_folds[p] for p in all_pats]
l12_rmses   = [l12_folds[p]   for p in all_pats if p in l12_folds]

print(f'\n=== AGORA W=1.0 LOO-CV ({len(agora_folds)}/10 folds) ===')
print(f'{"Patient":>8}  {"CT":>3}  {"AGORA LOO":>10}  {"L1+L2 LOO":>10}  {"Δ":>8}')
for p in all_pats:
    ct = ct_map[p]
    a = agora_folds.get(p, float('nan'))
    b = l12_folds.get(p, float('nan'))
    d = a - b if not np.isnan(b) else float('nan')
    print(f'{p:>8}  CT{ct}  {a:10.4f}  {b:10.4f}  {d:+8.4f}')

print(f'\nAGORA W=1.0 LOO: mean={np.mean(agora_rmses):.4f} ± {np.std(agora_rmses):.4f}')
if l12_rmses:
    print(f'L1+L2 LOO:       mean={np.mean([l12_folds[p] for p in all_pats if p in l12_folds]):.4f}')

# Save summary
summary = {
    'loo_rmse_mean': float(np.mean(agora_rmses)),
    'loo_rmse_std':  float(np.std(agora_rmses)),
    'n_folds': len(agora_folds),
    'per_patient': {p: agora_folds[p] for p in all_pats},
    'model': 'Hamilton expanded AGORA W=1.0 LOO-CV (10 pat, 10 guilds)'
}
out = CR / 'loo_expanded_agora_w1p0_summary.json'
json.dump(summary, open(out, 'w'), indent=2)
print(f'\nSaved {out}')
