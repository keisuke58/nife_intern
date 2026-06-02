#!/usr/bin/env python3
"""
robustness_strong_pairs.py — Robustness of the prior-free cross-dataset sign agreement
between Dieckow(no-prior) and Botelho(no-prior) gLV A matrices.

Tests whether the "strong pairs agree 89%" result is an artifact of the |A|>0.1 cutoff
and whether it survives a proper null (guild-label permutation), not just a binomial-0.5.

Outputs:
  (a) Threshold sweep:  fraction & binomial-p as a function of |A| cutoff theta.
  (b) Magnitude-weighted concordance W (threshold-free) + permutation p.
  (c) Strong-pair (theta=0.1) permutation test (guild relabeling null).
  (d) Bootstrap CI on the strong-pair agreement fraction.

Both SYMMETRIZED A (primary; convention-free) and RAW upper-triangle A (secondary;
the version that gave 8/9) are reported.

Run:  python robustness_strong_pairs.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

OUT = _here / 'results' / 'botelho_validation'
A_dk  = np.array(json.load(open(OUT / 'dieckow_A_noprior_comparison.json'))['A_dieckow_noprior'])
A_bot = np.array(json.load(open(OUT / 'botelho_A_comparison_noprior.json'))['A_botelho'])

rng = np.random.default_rng(0)
N_PERM = 20000
N_BOOT = 20000
pairs = [(i, j) for i in range(N_G) for j in range(i + 1, N_G)]   # 45 unordered

def sym(A):
    return (A + A.T) / 2.0

def upper_vals(A):
    """value per unordered pair (upper triangle as stored)."""
    return np.array([A[i, j] for i, j in pairs])

# ── core statistics on a pair-value vector ───────────────────────────────────
def strong_agreement(a, b, theta):
    """sign agreement among pairs with |a|>theta AND |b|>theta."""
    sel = (np.abs(a) > theta) & (np.abs(b) > theta)
    n = int(sel.sum())
    if n == 0:
        return 0, 0, float('nan')
    agree = int((np.sign(a[sel]) == np.sign(b[sel])).sum())
    return agree, n, agree / n

def weighted_concordance(a, b):
    """W = sum |a||b| sign(a)sign(b) / sum |a||b|  in [-1,1], threshold-free."""
    w = np.abs(a) * np.abs(b)
    denom = w.sum()
    if denom == 0:
        return 0.0
    return float((w * np.sign(a) * np.sign(b)).sum() / denom)

# ── guild-label permutation null ─────────────────────────────────────────────
# Apply a random permutation pi to the guild indices of the Botelho MATRIX
# (symmetric relabel of rows+cols), recompute the statistic. This breaks the
# cross-dataset guild alignment while preserving each matrix's internal structure.
def permuted_bot_vals(Bmat, mode):
    perm = rng.permutation(N_G)
    Bp = Bmat[np.ix_(perm, perm)]
    return (upper_vals(sym(Bp)) if mode == 'sym' else upper_vals(Bp))

def run_block(mode):
    print(f'\n{"="*66}\n  MODE = {mode.upper()}   '
          f'({"symmetrized (A+A^T)/2, convention-free" if mode=="sym" else "raw A[i,j] upper-triangle"})\n{"="*66}')
    Bm = sym(A_bot) if mode == 'sym' else A_bot   # not used directly; vals below
    a = upper_vals(sym(A_dk)) if mode == 'sym' else upper_vals(A_dk)
    b = upper_vals(sym(A_bot)) if mode == 'sym' else upper_vals(A_bot)

    # (a) threshold sweep
    print('\n  (a) |A| threshold sweep:')
    print('       theta   n_strong  agree   frac    binom_p(>50%)')
    sweep = []
    for theta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        agree, n, frac = strong_agreement(a, b, theta)
        bp = stats.binomtest(agree, n, 0.5, alternative='greater').pvalue if n > 0 else float('nan')
        print(f'       {theta:4.2f}   {n:6d}    {agree:4d}   '
              f'{frac if n else float("nan"):.3f}   {bp if n else float("nan"):.4f}')
        sweep.append({'theta': theta, 'n': n, 'agree': agree,
                      'frac': None if n == 0 else frac,
                      'binom_p': None if n == 0 else bp})

    # (b) weighted concordance + permutation
    W_obs = weighted_concordance(a, b)
    null_W = np.array([weighted_concordance(a, permuted_bot_vals(A_bot, mode)) for _ in range(N_PERM)])
    p_W = (1 + np.sum(null_W >= W_obs)) / (1 + N_PERM)
    print(f'\n  (b) Weighted concordance W = {W_obs:+.3f}   (null mean {null_W.mean():+.3f}, '
          f'sd {null_W.std():.3f})')
    print(f'      permutation p (W_perm >= W_obs) = {p_W:.4f}   [{N_PERM} guild relabelings]')

    # (c) strong-pair permutation at theta=0.1 (statistic = excess agreement #agree-#disagree)
    theta0 = 0.10
    agree0, n0, frac0 = strong_agreement(a, b, theta0)
    excess_obs = agree0 - (n0 - agree0)            # = 2*agree - n
    null_excess, null_frac = [], []
    for _ in range(N_PERM):
        bp_ = permuted_bot_vals(A_bot, mode)
        ag, n_, fr = strong_agreement(a, bp_, theta0)
        null_excess.append(ag - (n_ - ag))
        if n_ > 0:
            null_frac.append(fr)
    null_excess = np.array(null_excess)
    p_excess = (1 + np.sum(null_excess >= excess_obs)) / (1 + N_PERM)
    p_frac = (1 + np.sum(np.array(null_frac) >= frac0)) / (1 + len(null_frac)) if null_frac else float('nan')
    print(f'\n  (c) Strong pairs (|A|>{theta0}): {agree0}/{n0} = {frac0:.1%}')
    print(f'      excess agreement (agree-disagree) = {excess_obs}  '
          f'(null mean {null_excess.mean():+.2f})')
    print(f'      permutation p (excess) = {p_excess:.4f}')
    print(f'      permutation p (fraction, among perms with n>0) = {p_frac:.4f}')

    # (d) bootstrap CI on strong-pair fraction (resample the strong pairs)
    sel = (np.abs(a) > theta0) & (np.abs(b) > theta0)
    sa, sb = np.sign(a[sel]), np.sign(b[sel])
    match = (sa == sb).astype(float)
    if len(match) > 0:
        boot = np.array([rng.choice(match, size=len(match), replace=True).mean()
                         for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f'\n  (d) Bootstrap 95% CI on strong-pair fraction: '
              f'[{lo:.1%}, {hi:.1%}]  (median {np.median(boot):.1%})')
    else:
        lo = hi = float('nan')

    return {
        'sweep': sweep,
        'W': W_obs, 'W_perm_p': p_W, 'W_null_mean': float(null_W.mean()),
        'strong_theta0.1': {'agree': agree0, 'n': n0, 'frac': frac0,
                            'perm_p_excess': p_excess, 'perm_p_frac': p_frac},
        'bootstrap_ci': [None if np.isnan(lo) else lo, None if np.isnan(hi) else hi],
    }

results = {'n_perm': N_PERM, 'n_boot': N_BOOT}
for mode in ('sym', 'raw'):
    results[mode] = run_block(mode)

json.dump(results, open(OUT / 'robustness_strong_pairs.json', 'w'), indent=2)
print(f'\nSaved → {OUT}/robustness_strong_pairs.json')
