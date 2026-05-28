#!/usr/bin/env python3
"""
guild_network_analysis.py — A matrix credibility & guild-level network analysis.

Four analyses:
  1. Influence vs Vulnerability (column-sum vs row-sum of |A|)
     → who drives the community vs who is most affected
  2. Net effect per guild (Σ_j A_ij) — mutualist vs competitor
  3. LOO stability: mean ± SD of A_ij across 10 LOO folds
     → which interactions are confidently estimated
  4. Permutation test: shuffle patient labels → compare real A structure
     to null distribution → p-value per interaction

Usage:
    python guild_network_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import minimize

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / 'results' / 'guild_network'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_JSON     = HERE / 'results' / 'dieckow_cr' / 'fit_guild.json'
PHI_NPY      = HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy'
LOO_JSON_DIR = HERE / 'results' / 'dieckow_cr'   # LOO fold files if present

GUILD_SHORT = {
    'Actinobacteria':      'Acti',
    'Bacilli':             'Baci',
    'Bacteroidia':         'Bact',
    'Betaproteobacteria':  'Beta',
    'Clostridia':          'Clos',
    'Coriobacteriia':      'Cori',
    'Fusobacteriia':       'Fuso',
    'Gammaproteobacteria': 'Gamm',
    'Negativicutes':       'Nega',
    'Other':               'Othr',
}
CMAP = plt.get_cmap('tab10')


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data():
    d       = json.loads(FIT_JSON.read_text())
    A       = np.array(d['A'])
    b_all   = np.array(d['b_all'])
    guilds  = d['guilds']
    patients = d['patients']
    short   = [GUILD_SHORT.get(g, g[:4]) for g in guilds]
    colors  = [CMAP(i / len(guilds)) for i in range(len(guilds))]
    phi_all = np.load(PHI_NPY)
    return A, b_all, guilds, short, colors, patients, phi_all


def load_loo_matrices():
    """Load per-fold A matrices if available."""
    matrices = []
    for fold_file in sorted(LOO_JSON_DIR.glob('fit_guild_loo_fold*.json')):
        d = json.loads(fold_file.read_text())
        matrices.append(np.array(d['A']))
    return matrices if matrices else None


# ── Analysis 1: Influence vs Vulnerability ────────────────────────────────────

def influence_vulnerability(A, short):
    """
    influence[i]    = Σ_j |A_ji|  (column i: how much guild i affects all others)
    vulnerability[i] = Σ_j |A_ij|  (row i: how much guild i is affected by others)
    net_effect[i]   = Σ_j A_ij     (positive = net beneficiary from community)
    """
    n = len(short)
    influence    = np.array([np.sum(np.abs(A[:, i])) - np.abs(A[i, i]) for i in range(n)])
    vulnerability = np.array([np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) for i in range(n)])
    net_effect   = np.array([np.sum(A[i, :]) - A[i, i] for i in range(n)])
    return influence, vulnerability, net_effect


def plot_influence_vulnerability(influence, vulnerability, net_effect, short, colors,
                                  out_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: scatter influence vs vulnerability
    ax = axes[0]
    for i, (inf, vul) in enumerate(zip(influence, vulnerability)):
        ax.scatter(inf, vul, color=colors[i], s=80, zorder=3)
        ax.annotate(short[i], (inf, vul), fontsize=7,
                    xytext=(4, 2), textcoords='offset points')
    ax.set_xlabel('Influence  Σ|A_ji|  (drives others)', fontsize=9)
    ax.set_ylabel('Vulnerability  Σ|A_ij|  (affected by others)', fontsize=9)
    ax.set_title('Influence vs Vulnerability', fontsize=9)
    ax.axline((0, 0), slope=1, color='grey', lw=0.8, ls='--', label='equal')
    ax.legend(fontsize=7)

    # Panel B: stacked bar influence + vulnerability
    ax = axes[1]
    x = np.arange(len(short))
    ax.bar(x, influence,    color=colors, alpha=0.85, label='Influence (drives)')
    ax.bar(x, vulnerability, color=colors, alpha=0.40,
           bottom=0, linewidth=0.8, edgecolor='k', linestyle='--',
           fill=False, label='Vulnerability (affected)')
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, fontsize=8)
    ax.set_ylabel('Σ|A| (off-diagonal)', fontsize=9)
    ax.set_title('Interaction strength per guild', fontsize=9)
    ax.legend(fontsize=7)

    # Panel C: net effect (mutualist vs competitor)
    ax = axes[2]
    bar_colors = ['#2166ac' if v >= 0 else '#d6604d' for v in net_effect]
    bars = ax.bar(range(len(short)), net_effect, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, fontsize=8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('Net effect  Σ A_ij  (excl. self)', fontsize=9)
    ax.set_title('Net community effect\n(blue=mutualist, red=competitor)', fontsize=9)
    for bar, val in zip(bars, net_effect):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01 * np.sign(val + 1e-9),
                f'{val:+.2f}', ha='center', va='bottom', fontsize=6.5)

    fig.suptitle('Guild-level network roles in 10-guild Dieckow gLV', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Analysis 2: LOO stability of A_ij ────────────────────────────────────────

def compute_loo_stability_from_data(A_full, b_all, phi_all, guilds, short,
                                     n_folds=10, lam=1e-4):
    """
    Re-fit A on 9/10 patients (LOO) to get fold-specific A matrices.
    Fast: uses L-BFGS-B with scipy, no JAX needed.
    """
    from scipy.integrate import solve_ivp

    n_g = len(guilds)
    n_pat = len(b_all)
    phi_obs = phi_all  # (n_pat, n_weeks, n_guilds)

    def glv_rhs(t, phi, A, b):
        phi = np.maximum(phi, 0)
        return phi * (b + A @ phi)

    def simulate_weeks(A, b, phi0, t_weeks=[0, 7, 14, 21]):
        sol = solve_ivp(glv_rhs, [0, 21], phi0, args=(A, b),
                        t_eval=t_weeks, method='RK45', rtol=1e-6, atol=1e-8)
        return sol.y.T  # (4, n_g)

    def loss_fn(params, train_patients):
        n_off = n_g * (n_g - 1) // 2
        A_tri = params[:n_off]
        # Build symmetric A
        A = np.zeros((n_g, n_g))
        idx = np.triu_indices(n_g, k=1)
        A[idx] = A_tri
        A = A + A.T
        # Diagonal (self-limitation, negative)
        diag_raw = params[n_off:n_off + n_g]
        np.fill_diagonal(A, -np.exp(diag_raw))

        b_mat = params[n_off + n_g:].reshape(len(train_patients), n_g)
        loss = 0.0
        for pi, p in enumerate(train_patients):
            phi0 = phi_obs[p, 0, :]
            phi0 = phi0 / (phi0.sum() + 1e-12)
            try:
                traj = simulate_weeks(A, b_mat[pi], phi0)
                for t in [1, 2]:
                    diff = traj[t] - phi_obs[p, t, :]
                    loss += np.mean(diff**2)
            except Exception:
                loss += 1e3
        loss += lam * np.sum(A_tri**2)
        return loss

    A_folds = []
    for ho in range(n_folds):
        train = [p for p in range(n_pat) if p != ho]
        n_tr  = len(train)
        n_off = n_g * (n_g - 1) // 2

        # Warm start from full-data A
        A_tri0 = A_full[np.triu_indices(n_g, k=1)]
        diag0  = np.log(np.maximum(-np.diag(A_full), 1e-4))
        b0     = b_all[train].ravel()
        x0     = np.concatenate([A_tri0, diag0, b0])

        result = minimize(loss_fn, x0, args=(train,),
                          method='L-BFGS-B',
                          options={'maxiter': 800, 'ftol': 1e-10})

        # Extract A
        params  = result.x
        A_fold  = np.zeros((n_g, n_g))
        idx     = np.triu_indices(n_g, k=1)
        A_fold[idx] = params[:n_off]
        A_fold  = A_fold + A_fold.T
        diag_raw = params[n_off:n_off + n_g]
        np.fill_diagonal(A_fold, -np.exp(diag_raw))
        A_folds.append(A_fold)
        print(f'  Fold {ho+1}/{n_folds} done (loss={result.fun:.4f})')

    return np.array(A_folds)  # (n_folds, n_g, n_g)


def plot_loo_stability(A_full, A_folds, short, colors, out_path=None):
    n_g  = len(short)
    mean = A_folds.mean(axis=0)
    std  = A_folds.std(axis=0)
    # Sign consistency
    signs_full = np.sign(A_full)
    sc = np.mean(np.sign(A_folds) == signs_full[None], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Heatmap: mean A
    ax = axes[0]
    vmax = np.abs(mean).max()
    im = ax.imshow(mean, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_g)); ax.set_xticklabels(short, rotation=45, fontsize=7)
    ax.set_yticks(range(n_g)); ax.set_yticklabels(short, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('LOO mean A_ij', fontsize=9)

    # Heatmap: std A
    ax = axes[1]
    im2 = ax.imshow(std, cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n_g)); ax.set_xticklabels(short, rotation=45, fontsize=7)
    ax.set_yticks(range(n_g)); ax.set_yticklabels(short, fontsize=7)
    for i in range(n_g):
        for j in range(n_g):
            if i != j and std[i, j] > 0.01:
                ax.text(j, i, f'{std[i,j]:.2f}', ha='center', va='center',
                        fontsize=5, color='white' if std[i,j] > 0.5*std.max() else 'black')
    fig.colorbar(im2, ax=ax, fraction=0.046)
    ax.set_title('LOO std of A_ij\n(low = stable)', fontsize=9)

    # Heatmap: sign consistency
    ax = axes[2]
    im3 = ax.imshow(sc, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_g)); ax.set_xticklabels(short, rotation=45, fontsize=7)
    ax.set_yticks(range(n_g)); ax.set_yticklabels(short, fontsize=7)
    for i in range(n_g):
        for j in range(n_g):
            if i != j:
                ax.text(j, i, f'{sc[i,j]:.1f}', ha='center', va='center',
                        fontsize=5.5,
                        color='white' if sc[i,j] < 0.5 else 'black')
    fig.colorbar(im3, ax=ax, label='Sign consistency (0–1)', fraction=0.046)
    ax.set_title('Sign consistency across LOO folds\n(1.0 = always same sign)', fontsize=9)

    n_stable = int((sc[~np.eye(n_g, dtype=bool)] >= 0.9).sum())
    fig.suptitle(f'A matrix LOO stability  ({n_stable}/{n_g*(n_g-1)} off-diag pairs SC≥0.9)',
                 fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig, mean, std, sc


# ── Analysis 3: Permutation test ─────────────────────────────────────────────

def _perm_one(args):
    """Top-level worker for joblib: fit A on one permuted dataset."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import minimize
    A_full, b_all, phi_all, n_g, lam, perm_idx = args
    n_pat = len(b_all)
    phi_shuffled = phi_all[perm_idx]

    def glv_rhs(t, phi, A, b):
        phi = np.maximum(phi, 0)
        return phi * (b + A @ phi)

    def loss_fn(params):
        n_off = n_g * (n_g - 1) // 2
        A = np.zeros((n_g, n_g))
        idx = np.triu_indices(n_g, k=1)
        A[idx] = params[:n_off]
        A = A + A.T
        np.fill_diagonal(A, -np.exp(params[n_off:n_off + n_g]))
        b_mat = params[n_off + n_g:].reshape(n_pat, n_g)
        loss = 0.0
        for p in range(n_pat):
            phi0 = phi_shuffled[p, 0, :]
            phi0 = phi0 / (phi0.sum() + 1e-12)
            try:
                sol = solve_ivp(glv_rhs, [0, 21], phi0,
                                args=(A, b_mat[p]), t_eval=[7, 14, 21],
                                method='RK45', rtol=1e-4, atol=1e-6)
                traj = sol.y.T
                for t in range(min(2, len(traj))):
                    diff = traj[t] - phi_shuffled[p, t + 1, :]
                    loss += np.mean(diff**2)
            except Exception:
                loss += 1e3
        return loss + lam * np.sum(params[:n_off]**2)

    n_off = n_g * (n_g - 1) // 2
    A_tri0 = A_full[np.triu_indices(n_g, k=1)]
    diag0  = np.log(np.maximum(-np.diag(A_full), 1e-4))
    b0     = b_all.ravel()
    x0     = np.concatenate([A_tri0, diag0, b0])
    result = minimize(loss_fn, x0, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-6})
    A_out = np.zeros((n_g, n_g))
    idx   = np.triu_indices(n_g, k=1)
    A_out[idx] = result.x[:n_off]
    A_out = A_out + A_out.T
    np.fill_diagonal(A_out, -np.exp(result.x[n_off:n_off + n_g]))
    return A_out


def permutation_test(A_full, b_all, phi_all, guilds, short,
                      n_perm=100, lam=1e-4, seed=42):
    """
    Shuffle patient labels (phi_all rows) → fit A → get null distribution of |A_ij|.
    p-value[i,j] = fraction of permutations where |A_perm[i,j]| >= |A_real[i,j]|
    Parallelised with joblib across permutations.
    """
    from joblib import Parallel, delayed
    rng   = np.random.default_rng(seed)
    n_g   = len(guilds)
    n_pat = len(b_all)

    perm_indices = [rng.permutation(n_pat) for _ in range(n_perm)]
    jobs = [(A_full, b_all, phi_all, n_g, lam, idx) for idx in perm_indices]

    n_jobs = min(24, n_perm)
    print(f'  Running {n_perm} permutations on {n_jobs} cores …')
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_perm_one)(j) for j in jobs)

    null_A = np.array(results)
    pval   = np.mean(np.abs(null_A) >= np.abs(A_full)[None], axis=0)
    return null_A, pval


def plot_permutation_results(A_full, null_A, pval, short, out_path=None):
    n_g  = len(short)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Heatmap: -log10(p-value)
    ax = axes[0]
    log_pval = -np.log10(np.maximum(pval, 1 / (null_A.shape[0] + 1)))
    np.fill_diagonal(log_pval, 0)
    im = ax.imshow(log_pval, cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n_g)); ax.set_xticklabels(short, rotation=45, fontsize=7)
    ax.set_yticks(range(n_g)); ax.set_yticklabels(short, fontsize=7)
    for i in range(n_g):
        for j in range(n_g):
            if i != j and log_pval[i, j] > 0.5:
                ax.text(j, i, f'{pval[i,j]:.2f}', ha='center', va='center',
                        fontsize=5.5,
                        color='white' if log_pval[i,j] > log_pval.max()*0.7 else 'black')
    fig.colorbar(im, ax=ax, label='-log₁₀(p)', fraction=0.046)
    ax.set_title('Permutation test: -log₁₀(p-value)\n'
                 '(high = significantly different from random)', fontsize=8)

    # Scatter: |A_real| vs permutation null distribution
    ax = axes[1]
    null_means = np.abs(null_A).mean(axis=0)
    null_stds  = np.abs(null_A).std(axis=0)
    abs_A      = np.abs(A_full)

    # off-diagonal only
    mask = ~np.eye(n_g, dtype=bool)
    x = null_means[mask]
    y = abs_A[mask]
    sig = pval[mask] < 0.05

    ax.scatter(x[~sig], y[~sig], color='grey',  alpha=0.5, s=30, label='p≥0.05')
    ax.scatter(x[sig],  y[sig],  color='#d6604d', alpha=0.8, s=50, label='p<0.05')
    ax.axline((0, 0), slope=1, color='k', lw=0.8, ls='--', label='y=x')
    ax.set_xlabel('Mean |A_ij| under permutation (null)', fontsize=9)
    ax.set_ylabel('|A_ij| real data', fontsize=9)
    ax.set_title('Real vs permutation A magnitude', fontsize=9)
    ax.legend(fontsize=8)

    n_sig = int(sig.sum())
    fig.suptitle(f'Permutation test (n={null_A.shape[0]} permutations)  '
                 f'{n_sig}/{n_g*(n_g-1)} off-diag pairs significant (p<0.05)',
                 fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Summary figure ────────────────────────────────────────────────────────────

def plot_guild_summary(A_full, influence, vulnerability, net_effect, short, colors,
                        out_path=None):
    """One-panel ranked summary: who matters most."""
    n_g = len(short)
    total_involvement = influence + vulnerability  # overall network centrality

    order = np.argsort(total_involvement)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_g)
    width = 0.35

    bars1 = ax.bar(x - width/2, influence[order],    width,
                   color=[colors[i] for i in order], alpha=0.85, label='Influence (drives others)')
    bars2 = ax.bar(x + width/2, vulnerability[order], width,
                   color=[colors[i] for i in order], alpha=0.40,
                   edgecolor=[colors[i] for i in order], linewidth=1.2,
                   label='Vulnerability (affected by others)')

    # Net effect markers
    for xi, i in enumerate(order):
        ne = net_effect[i]
        marker = '▲' if ne > 0 else '▼'
        color  = '#2166ac' if ne > 0 else '#d6604d'
        ax.text(xi, max(influence[i], vulnerability[i]) + 0.05,
                f'{marker}{abs(ne):.1f}', ha='center', va='bottom',
                fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([short[i] for i in order], rotation=45, fontsize=9)
    ax.set_ylabel('Σ|A| (off-diagonal)', fontsize=9)
    ax.set_title('Guild network centrality  (▲=net mutualist, ▼=net competitor)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.axhline(0, color='k', lw=0.5)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-loo',   action='store_true',
                        help='Skip LOO re-fitting (use precomputed if available)')
    parser.add_argument('--skip-perm',  action='store_true',
                        help='Skip permutation test')
    parser.add_argument('--n-perm',     type=int, default=100)
    args = parser.parse_args()

    A, b_all, guilds, short, colors, patients, phi_all = load_data()
    n_g = len(guilds)

    # ── Analysis 1: Influence / Vulnerability / Net effect ─────────────────────
    print('=== Analysis 1: Influence / Vulnerability ===')
    influence, vulnerability, net_effect = influence_vulnerability(A, short)

    print(f'\n{"Guild":<6} {"Influence":>10} {"Vulnerab.":>10} {"Net effect":>11} {"Role"}')
    for i in np.argsort(influence + vulnerability)[::-1]:
        role = 'mutualist' if net_effect[i] > 0 else 'competitor'
        print(f'{short[i]:<6} {influence[i]:10.3f} {vulnerability[i]:10.3f} '
              f'{net_effect[i]:11.3f}  {role}')

    plot_influence_vulnerability(influence, vulnerability, net_effect, short, colors,
                                  out_path=OUT_DIR / 'influence_vulnerability.png')
    plot_guild_summary(A, influence, vulnerability, net_effect, short, colors,
                       out_path=OUT_DIR / 'guild_centrality_summary.png')

    # ── Analysis 2: LOO stability ──────────────────────────────────────────────
    print('\n=== Analysis 2: LOO stability ===')
    loo_cache = OUT_DIR / 'loo_A_folds.npy'

    if args.skip_loo and not loo_cache.exists():
        print('  Skipped (--skip-loo).')
        A_folds = None
    elif loo_cache.exists():
        print(f'  Loading cached LOO matrices from {loo_cache}')
        A_folds = np.load(str(loo_cache))
    else:
        existing = load_loo_matrices()
        if existing is not None and len(existing) == 10:
            print(f'  Found {len(existing)} pre-computed LOO fold files.')
            A_folds = np.array(existing)
        else:
            print('  Re-fitting A on 9/10 patients for each fold...')
            A_folds = compute_loo_stability_from_data(A, b_all, phi_all, guilds, short)
            np.save(str(loo_cache), A_folds)
            print(f'  Saved to {loo_cache}')

    if A_folds is not None:
        fig_loo, A_mean, A_std, sc = plot_loo_stability(
            A, A_folds, short, colors,
            out_path=OUT_DIR / 'loo_stability.png')
        n_stable = int((sc[~np.eye(n_g, dtype=bool)] >= 0.9).sum())
        print(f'  Sign consistency >= 0.9: {n_stable}/{n_g*(n_g-1)} pairs')
        stable_pairs = [(sc[i,j], short[i], short[j], A_mean[i,j])
                        for i in range(n_g) for j in range(n_g)
                        if i != j and sc[i,j] >= 0.99]
        for s, gi, gj, val in sorted(stable_pairs, key=lambda x: abs(x[3]), reverse=True)[:10]:
            print(f'    {gi} -> {gj}: A={val:+.3f}  SC={s:.2f}')
    else:
        sc = None

    # ── Analysis 3: Permutation test ───────────────────────────────────────────
    if not args.skip_perm:
        print(f'\n=== Analysis 3: Permutation test (n={args.n_perm}) ===')
        perm_cache = OUT_DIR / f'perm_null_A_{args.n_perm}.npy'
        if perm_cache.exists():
            print(f'  Loading cached permutation results.')
            null_A = np.load(str(perm_cache))
            pval   = np.mean(np.abs(null_A) >= np.abs(A)[None], axis=0)
        else:
            null_A, pval = permutation_test(A, b_all, phi_all, guilds, short,
                                             n_perm=args.n_perm)
            np.save(str(perm_cache), null_A)

        n_sig = int((pval[~np.eye(n_g, dtype=bool)] < 0.05).sum())
        print(f'  Significant pairs (p<0.05): {n_sig}/{n_g*(n_g-1)}')
        print(f'  Most significant:')
        pairs = [(pval[i,j], short[i], short[j], A[i,j])
                 for i in range(n_g) for j in range(n_g) if i != j]
        for pv, gi, gj, val in sorted(pairs, key=lambda x: x[0])[:10]:
            print(f'    {gi} → {gj}: A={val:+.3f}  p={pv:.3f}')

        plot_permutation_results(A, null_A, pval, short,
                                  out_path=OUT_DIR / 'permutation_test.png')
    else:
        print('\n=== Permutation test skipped ===')
        pval = None

    # ── Save summary JSON ──────────────────────────────────────────────────────
    summary = {
        'influence':     dict(zip(short, influence.tolist())),
        'vulnerability': dict(zip(short, vulnerability.tolist())),
        'net_effect':    dict(zip(short, net_effect.tolist())),
    }
    if sc is not None:
        summary['loo_sign_consistency'] = {
            f'{short[i]}->{short[j]}': float(sc[i, j])
            for i in range(n_g) for j in range(n_g) if i != j
        }
    (OUT_DIR / 'network_summary.json').write_text(
        __import__('json').dumps(summary, indent=2))
    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
