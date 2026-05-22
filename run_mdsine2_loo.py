#!/usr/bin/env python3
"""
run_mdsine2_loo.py — LOO-CV for MDSINE2 on Dieckow 2024 guild-level data.

Relative abundances are scaled to pseudo-counts (×10000) with constant qPCR.
One fold at a time; results saved to results/mdsine2/loo_fold{FOLD}.json.

Usage:
    python run_mdsine2_loo.py --fold 0
    for i in $(seq 0 9); do python run_mdsine2_loo.py --fold $i; done
"""
import argparse, json, logging, numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING)   # suppress MDSINE2 debug spam

import mdsine2 as md2

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, required=True, help='held-out patient index (0-9)')
parser.add_argument('--burnin',    type=int, default=500)
parser.add_argument('--n_samples', type=int, default=2000)
parser.add_argument('--seed',      type=int, default=0)
args = parser.parse_args()

PATIENTS = list('ABCDEFGHKL')
HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / 'results' / 'dieckow_otu'
OUT_DIR  = HERE / 'results' / 'mdsine2'
OUT_DIR.mkdir(exist_ok=True)

SCALE    = 10000          # pseudo-count scale for relative abundances
QPCR_VAL = float(SCALE)  # constant total abundance proxy

# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_DIR / 'guild_summary.json') as f:
    gs = json.load(f)

GUILDS = gs['guilds']
N_G    = len(GUILDS)

obs = {}
for s in gs['samples']:
    obs[(s['patient'], s['week'])] = np.array([s[g] for g in GUILDS])

# ── Build MDSINE2 Study for training patients ──────────────────────────────────
hold      = args.fold
hold_pat  = PATIENTS[hold]
train_pat = [p for p in PATIENTS if p != hold_pat]

print(f'LOO fold {hold}: held={hold_pat}  train={train_pat}', flush=True)

taxa = md2.TaxaSet()
for idx, g in enumerate(GUILDS):
    taxa.add(md2.Taxon(name=g, idx=idx))

study = md2.Study(taxa=taxa, name=f'dieckow_fold{hold}')

for pat in train_pat:
    study.add_subject(name=pat)          # returns self (Study)
    subj = study[pat]                    # get actual Subject object
    for wk in [1, 2, 3]:
        if (pat, wk) not in obs:
            continue
        phi    = obs[(pat, wk)]
        counts = np.round(phi * SCALE).astype(int)
        subj.add_reads(timepoints=float(wk), reads=counts)
        subj.add_qpcr(timepoints=float(wk), qpcr=np.array([QPCR_VAL]))

# ── MCMC parameters ────────────────────────────────────────────────────────────
basepath = str(OUT_DIR / f'fold{hold}')
Path(basepath).mkdir(exist_ok=True)

# negbin_a0, negbin_a1: NegBin overdispersion (typical MDSINE2 defaults)
params = md2.MDSINE2ModelConfig(
    basepath=basepath,
    seed=args.seed,
    burnin=args.burnin,
    n_samples=args.n_samples,
    negbin_a0=1e-10,
    negbin_a1=0.001,
    checkpoint=200,
)

# ── Run inference ──────────────────────────────────────────────────────────────
print(f'Running MDSINE2 MCMC (burnin={args.burnin}, n_samples={args.n_samples})...', flush=True)
# 3 timepoints per patient — LOESS initialization fails; use 'linear-interpolation'
from mdsine2.names import STRNAMES as _SN
params.INITIALIZATION_KWARGS[_SN.FILTERING]['x_value_option'] = 'coupling'
# 10 taxa → n_clusters must be ≤ 10
from mdsine2.names import STRNAMES as _SN2
params.INITIALIZATION_KWARGS[_SN2.CLUSTERING]['n_clusters'] = 10

mcmc = md2.initialize_graph(
    params=params,
    graph_name=f'dieckow_fold{hold}',
    subjset=study,
)
mcmc = md2.run.run_graph(mcmc, crash_if_error=False)
print('MCMC done.', flush=True)

# ── Extract posterior A matrix directly from HDF5 ─────────────────────────────
import h5py
from mdsine2.names import STRNAMES

trace_file = Path(basepath) / 'traces.hdf5'
with h5py.File(trace_file, 'r') as hf:
    A_samples = hf['Interactions object'][:]      # (n_samples, N_G, N_G)
    b_samples = hf['Growth parameter'][:]         # (n_samples, N_G)

# use posterior samples only (discard burnin fraction at front)
n_post = len(A_samples)
A_mean = A_samples.mean(axis=0)
b_mean = b_samples.mean(axis=0)
print(f'Posterior samples used: {n_post}', flush=True)

# ── LOO prediction for held-out patient ───────────────────────────────────────
from scipy.integrate import solve_ivp

def _replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)

def _step(phi0, b, A):
    sol = solve_ivp(_replicator_rhs, [0, 1.0], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0.copy()

phi_ho = [obs[(hold_pat, wk)] for wk in [1, 2, 3]]

# fit b for held-out patient (A fixed at posterior mean)
from scipy.optimize import minimize

def _ho_obj(b_p):
    p2 = _step(phi_ho[0], b_p, A_mean)
    p3 = _step(p2,        b_p, A_mean)
    return float(np.sqrt((np.mean((p2 - phi_ho[1])**2) +
                          np.mean((p3 - phi_ho[2])**2)) / 2))

res_b = minimize(_ho_obj, b_mean.copy(), method='L-BFGS-B',
                 options={'maxiter': 500, 'ftol': 1e-12})
b_ho_opt = res_b.x

p2 = _step(phi_ho[0], b_ho_opt, A_mean)
p3 = _step(p2,        b_ho_opt, A_mean)
rmse = float(np.sqrt((np.mean((p2 - phi_ho[1])**2) +
                      np.mean((p3 - phi_ho[2])**2)) / 2))

print(f'Fold {hold} ({hold_pat}): LOO-RMSE={rmse:.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'patient':   hold_pat,
    'hold_idx':  hold,
    'rmse':      rmse,
    'A_mean':    A_mean.tolist(),
    'b_ho':      b_ho_opt.tolist(),
    'burnin':    args.burnin,
    'n_samples': args.n_samples,
}
json.dump(out, open(OUT_DIR / f'loo_fold{hold}.json', 'w'), indent=2)
print(f'Saved loo_fold{hold}.json', flush=True)
