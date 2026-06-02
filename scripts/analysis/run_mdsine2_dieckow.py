"""
MDSINE2 on Dieckow 2024 guild-level data.
Input: results/dieckow_otu/guild_summary.json
Output: results/mdsine2/
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, numpy as np, pandas as pd
from pathlib import Path
import mdsine2 as md2
from mdsine2.names import STRNAMES

# ── load data ────────────────────────────────────────────
with open('results/dieckow_otu/guild_summary.json') as f:
    gs = json.load(f)

GUILDS   = gs['guilds']          # 10 guild names
PATIENTS = list('ABCDEFGHKL')
N_G = len(GUILDS)

# Build (patient, week, guild) arrays
samples = gs['samples']  # list of dicts
obs = {}
for s in samples:
    obs[(s['patient'], s['week'])] = np.array([s[g] for g in GUILDS])

# ── build MDSINE2 Study ──────────────────────────────────
study = md2.Study(taxa=md2.TaxaSet())

# Add taxa (guilds)
for g in GUILDS:
    study.taxa.add_taxon(name=g)

# Add subjects and timepoints
# MDSINE2 expects absolute counts; we use relative × 10000 as pseudo-counts
SCALE = 10000
for pat in PATIENTS:
    subj = study.add_subject(name=pat)
    for wk in [1, 2, 3]:
        if (pat, wk) not in obs:
            continue
        phi = obs[(pat, wk)]
        counts = (phi * SCALE).astype(int)
        # reads = total pseudo-reads (SCALE)
        subj.add_reads(
            timepoint=float(wk),
            data=counts,
            qpcr=np.array([SCALE])  # total abundance proxy
        )

OUT = Path('results/mdsine2')
OUT.mkdir(exist_ok=True)

# ── MCMC parameters (short run for test) ────────────────
params = md2.config.MDSINE2ModelConfig(
    basepath=str(OUT),
    seed=0,
    burnin=500,
    n_samples=1000,
    checkpoint=100,
)

# ── run inference ─────────────────────────────────────────
print("Running MDSINE2 MCMC...")
mcmc = md2.run_inference(params=params, study=study)

# ── extract A matrix posterior ───────────────────────────
interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ]
A_samples = interactions.get_trace_from_disk()  # shape: (n_samples, N_G, N_G)
A_mean = A_samples.mean(axis=0)
A_std  = A_samples.std(axis=0)

df_A = pd.DataFrame(A_mean, index=GUILDS, columns=GUILDS).round(4)
df_A.to_csv(OUT / 'A_matrix_mean.csv')

df_A_std = pd.DataFrame(A_std, index=GUILDS, columns=GUILDS).round(4)
df_A_std.to_csv(OUT / 'A_matrix_std.csv')

print(f"A matrix saved to {OUT}")
print("Mean A diagonal:", np.diag(A_mean).round(4))
