#!/usr/bin/env python3
"""
guild_tipping_point.py — Tipping point / bifurcation analysis for 10-guild Dieckow gLV.

Question: how much do growth rates b need to shift (from CT2 dysbiotic → CT1 commensal
environment) before the community flips to a healthy state?

Analysis:
  1. Cluster patients into CT1 / CT2 using k-means on b_all (PC1 of b).
  2. Linear interpolation: b(α) = (1-α)·b_CT2 + α·b_CT1,  α ∈ [0,1].
  3. For each α, simulate gLV from every patient's week-0 phi0; record
     week-3 R/G ratio = log(φ_dys) - log(φ_com).
  4. Find tipping α* where mean R/G ratio crosses zero.
  5. Single-guild b sweep: scan b_i from b_CT2[i] to b_CT1[i] one at a time
     to identify which guild's growth rate drives the tipping point.
  6. 2D phase diagram for the two most critical guilds.

R/G ratio definition (consistent with Joshi validation):
  φ_dys = φ_Bact + φ_Fuso + φ_Clos
  φ_com = φ_Baci + φ_Acti + φ_Nega
  R/G ratio < 0 → CT1 (commensal), R/G ratio > 0 → CT2 (dysbiotic)
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
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.cluster import KMeans

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / 'results' / 'guild_tipping'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_JSON = HERE / 'results' / 'dieckow_cr' / 'fit_guild.json'
PHI_NPY  = HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy'

T_END      = 21.0
T_END_LONG = 150.0            # long integration to find equilibrium
T_EVAL     = np.linspace(0, T_END, 300)
T_EVAL_LONG = np.linspace(0, T_END_LONG, 400)

# CT1 = commensal (A,D,E,G,K,L), CT2 = dysbiotic (B,C,F,H)
# Determined by GMM on W1 guild fractions in Dieckow 2024.
CT_LABELS_KNOWN = {
    'A': 2, 'B': 2, 'C': 2, 'D': 1, 'E': 1,
    'F': 2, 'G': 1, 'H': 2, 'K': 1, 'L': 1,
}

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
    b_all   = np.array(d['b_all'])       # (n_pat, n_guilds)
    guilds  = d['guilds']
    patients = d['patients']
    short   = [GUILD_SHORT.get(g, g[:4]) for g in guilds]
    colors  = [CMAP(i / len(guilds)) for i in range(len(guilds))]

    phi_all = np.load(PHI_NPY)           # (n_pat, n_weeks, n_guilds)
    phi0_pp = phi_all[:, 0, :]
    sums    = phi0_pp.sum(axis=1, keepdims=True)
    phi0_pp = np.where(sums > 1e-12, phi0_pp / sums, phi0_pp)

    return A, b_all, phi0_pp, guilds, short, colors, patients


# ── gLV ───────────────────────────────────────────────────────────────────────

def glv_rhs(t, phi, A, b):
    phi = np.maximum(phi, 0)
    return phi * (b + A @ phi)


def simulate(A, b, phi0, t_end=T_END, t_eval=None, rtol=1e-6):
    if t_eval is None:
        t_eval = np.linspace(0, t_end, max(100, int(t_end * 10)))
    sol = solve_ivp(glv_rhs, [0, t_end], phi0,
                    args=(A, b), t_eval=t_eval,
                    method='RK45', rtol=rtol, atol=rtol * 1e-2)
    traj = sol.y.T
    if len(traj) < len(t_eval):
        pad  = np.tile(traj[-1], (len(t_eval) - len(traj), 1))
        traj = np.vstack([traj, pad])
    return traj


def week3(traj):
    """Return composition at t=21 days."""
    return traj[-1]


def equilibrium(A, b, phi0):
    """Integrate to t=150 to approximate equilibrium."""
    traj = simulate(A, b, phi0, t_end=T_END_LONG, rtol=1e-5)
    return traj[-1]


def gdi(phi, short):
    """guild-level R/G ratio (log ratio)."""
    dys_idx = [short.index(s) for s in ['Bact', 'Fuso', 'Clos'] if s in short]
    com_idx = [short.index(s) for s in ['Baci', 'Acti', 'Nega'] if s in short]
    phi_dys = phi[dys_idx].sum() + 1e-9
    phi_com = phi[com_idx].sum() + 1e-9
    return float(np.log(phi_dys) - np.log(phi_com))


# ── CT clustering ─────────────────────────────────────────────────────────────

def assign_ct(b_all, patients):
    """Use known CT labels from Dieckow 2024."""
    ct = np.array([CT_LABELS_KNOWN.get(p, 0) for p in patients])
    return ct   # 1 = CT1 commensal, 2 = CT2 dysbiotic


# ── Interpolation scan ────────────────────────────────────────────────────────

def alpha_scan(A, b_all, phi0_pp, short, patients, n_alpha=60):
    ct       = assign_ct(b_all, patients)
    b_CT1    = b_all[ct == 1].mean(axis=0)
    b_CT2    = b_all[ct == 2].mean(axis=0)
    alphas   = np.linspace(0, 1, n_alpha)

    # Per-patient trajectories at each alpha
    n_pat   = len(b_all)
    gdi_mat = np.zeros((n_pat, n_alpha))  # (patient, alpha)

    for ai, alpha in enumerate(alphas):
        b_interp = (1 - alpha) * b_CT2 + alpha * b_CT1
        for p in range(n_pat):
            traj = simulate(A, b_interp, phi0_pp[p])
            gdi_mat[p, ai] = gdi(week3(traj), short)

    gdi_mean = gdi_mat.mean(axis=0)
    gdi_std  = gdi_mat.std(axis=0)

    # Find tipping point: where mean R/G ratio crosses zero
    sign_changes = np.where(np.diff(np.sign(gdi_mean)))[0]
    alpha_star   = float(alphas[sign_changes[0]]) if len(sign_changes) else None

    return alphas, gdi_mean, gdi_std, gdi_mat, alpha_star, b_CT1, b_CT2


# ── Single-guild b sweep ──────────────────────────────────────────────────────

def _sweep_one(args):
    """Worker for single_guild_sweep (joblib-safe top-level function)."""
    A, b_CT2, phi0_pp, short, i, bv = args
    b_test    = b_CT2.copy()
    b_test[i] = bv
    gdi_vals  = []
    for p in range(len(phi0_pp)):
        traj = simulate(A, b_test, phi0_pp[p])
        gdi_vals.append(gdi(week3(traj), short))
    return np.mean(gdi_vals)


def single_guild_sweep(A, b_all, phi0_pp, short, patients, n_steps=20):
    """
    For each guild i, sweep b_i from b_CT2[i] to b_CT1[i] while keeping
    all other b components at b_CT2.  Record mean R/G ratio at week-3.
    Returns: effect_sizes (n_guilds,) = |R/G ratio(b_CT1[i]) - R/G ratio(b_CT2[i])|
    Uses joblib parallel across all (guild, step) combinations.
    """
    from joblib import Parallel, delayed
    ct    = assign_ct(b_all, patients)
    b_CT1 = b_all[ct == 1].mean(axis=0)
    b_CT2 = b_all[ct == 2].mean(axis=0)
    n_g   = len(short)

    # Build flat job list: (guild_idx, b_value)
    jobs = [(A, b_CT2, phi0_pp, short, i, bv)
            for i in range(n_g)
            for bv in np.linspace(b_CT2[i], b_CT1[i], n_steps)]

    n_jobs = min(24, len(jobs))
    results = Parallel(n_jobs=n_jobs)(delayed(_sweep_one)(j) for j in jobs)

    effect_sizes = np.zeros(n_g)
    sweep_curves = {}
    idx = 0
    for i in range(n_g):
        b_vals  = np.linspace(b_CT2[i], b_CT1[i], n_steps)
        gdi_arr = np.array(results[idx: idx + n_steps])
        idx    += n_steps
        effect_sizes[i]      = abs(gdi_arr[-1] - gdi_arr[0])
        sweep_curves[short[i]] = (b_vals, gdi_arr)

    return effect_sizes, sweep_curves, b_CT1, b_CT2


# ── 2D phase diagram ──────────────────────────────────────────────────────────

def _phase2d_one(args):
    """Worker for phase_diagram_2d (joblib-safe)."""
    A, b_CT2, phi0_pp, short, gi, gj, bvi, bvj = args
    b_test     = b_CT2.copy()
    b_test[gi] = bvi
    b_test[gj] = bvj
    gdi_list = [gdi(week3(simulate(A, b_test, phi0_pp[p])), short)
                for p in range(len(phi0_pp))]
    return np.mean(gdi_list)


def phase_diagram_2d(A, b_all, phi0_pp, short, patients, gi, gj, n_grid=20):
    """
    Scan b_i × b_j on a 2D grid. Returns: b_i_vals, b_j_vals, gdi_grid (n_grid, n_grid).
    Uses joblib parallel.
    """
    from joblib import Parallel, delayed
    ct    = assign_ct(b_all, patients)
    b_CT1 = b_all[ct == 1].mean(axis=0)
    b_CT2 = b_all[ct == 2].mean(axis=0)

    margin = 0.25
    def rng(idx):
        lo = min(b_CT2[idx], b_CT1[idx])
        hi = max(b_CT2[idx], b_CT1[idx])
        d  = max(hi - lo, 0.5)           # at least 0.5 width
        return np.linspace(lo - margin * d, hi + margin * d, n_grid)

    b_i_range = rng(gi)
    b_j_range = rng(gj)

    jobs = [(A, b_CT2, phi0_pp, short, gi, gj, bvi, bvj)
            for bvi in b_i_range for bvj in b_j_range]

    n_jobs = min(24, len(jobs))
    results = Parallel(n_jobs=n_jobs)(delayed(_phase2d_one)(j) for j in jobs)

    gdi_grid = np.array(results).reshape(n_grid, n_grid)
    return b_i_range, b_j_range, gdi_grid


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_alpha_scan(alphas, gdi_mean, gdi_std, gdi_mat, alpha_star,
                    patients, ct_labels, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    # Per-patient lines (thin, coloured by CT)
    ct_colors = {1: '#2166ac', 2: '#d6604d'}
    for p, pat in enumerate(patients):
        ct = ct_labels[p]
        ax.plot(alphas, gdi_mat[p], color=ct_colors[ct], alpha=0.3, lw=1)

    # Mean + std band
    ax.fill_between(alphas, gdi_mean - gdi_std, gdi_mean + gdi_std,
                    color='grey', alpha=0.2, label='mean ± SD')
    ax.plot(alphas, gdi_mean, 'k-', lw=2, label='Mean R/G ratio')

    ax.axhline(0, color='grey', lw=1, ls='--')
    ax.axvline(0, color='#d6604d', lw=1, ls=':', label='CT2 env (α=0)')
    ax.axvline(1, color='#2166ac', lw=1, ls=':', label='CT1 env (α=1)')

    if alpha_star is not None:
        ax.axvline(alpha_star, color='black', lw=1.5, ls='--',
                   label=f'Tipping α*={alpha_star:.2f}')
        ax.annotate(f'α* = {alpha_star:.2f}',
                    xy=(alpha_star, 0), xytext=(alpha_star + 0.05, 0.5),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=9)

    # CT legend patches
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color='#2166ac', alpha=0.5, label='CT1 patients'),
                Patch(color='#d6604d', alpha=0.5, label='CT2 patients')]
    ax.legend(handles=handles, fontsize=7, loc='upper right')

    ax.set_xlabel('Interpolation parameter α  (0 = CT2 environment, 1 = CT1 environment)',
                  fontsize=9)
    ax.set_ylabel('guild-level R/G ratio (R/G ratio)', fontsize=9)
    ax.set_title('Community tipping point: dysbiotic → commensal transition\n'
                 'b(α) = (1-α)·b_CT2 + α·b_CT1', fontsize=10)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_single_guild_effects(effect_sizes, short, colors, b_CT1, b_CT2, out_path=None):
    order = np.argsort(effect_sizes)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Bar chart of effect sizes
    ax = axes[0]
    bars = ax.bar(range(len(short)),
                  effect_sizes[order],
                  color=[colors[i] for i in order], alpha=0.85)
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels([short[i] for i in order], rotation=45, fontsize=8)
    ax.set_ylabel('|ΔR/G|  (b_CT2[i] → b_CT1[i])', fontsize=9)
    ax.set_title('Which guild\'s growth rate drives the tipping?', fontsize=9)
    for bar, val in zip(bars, effect_sizes[order]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # b_CT1 vs b_CT2 comparison
    ax = axes[1]
    x = np.arange(len(short))
    width = 0.35
    ax.bar(x - width/2, b_CT2, width, color='#d6604d', alpha=0.75, label='CT2 (dysbiotic)')
    ax.bar(x + width/2, b_CT1, width, color='#2166ac', alpha=0.75, label='CT1 (commensal)')
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, fontsize=8)
    ax.set_ylabel('Growth rate b_i', fontsize=9)
    ax.set_title('CT1 vs CT2 growth rate vectors', fontsize=9)
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=8)

    fig.suptitle('Single-guild growth rate tipping contribution', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_phase_diagram(b_i_vals, b_j_vals, gdi_grid, short, gi, gj,
                       b_CT1, b_CT2, out_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(gdi_grid).max(), 0.1)
    im = ax.contourf(b_j_vals, b_i_vals, gdi_grid,
                     levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(b_j_vals, b_i_vals, gdi_grid, levels=[0],
               colors='black', linewidths=1.5, linestyles='--')

    # Mark CT1 and CT2 points
    ax.plot(b_CT2[gj], b_CT2[gi], 'v', color='#d6604d', ms=10,
            label='CT2 mean', zorder=5)
    ax.plot(b_CT1[gj], b_CT1[gi], '^', color='#2166ac', ms=10,
            label='CT1 mean', zorder=5)

    fig.colorbar(im, ax=ax, label='Mean R/G ratio (week-3)')
    ax.set_xlabel(f'b_{short[gj]}  (growth rate of {short[gj]})', fontsize=9)
    ax.set_ylabel(f'b_{short[gi]}  (growth rate of {short[gi]})', fontsize=9)
    ax.set_title(f'2D phase diagram: {short[gi]} × {short[gj]}\n'
                 'Black dashed: R/G ratio = 0 (tipping boundary)', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_top_sweep_curves(sweep_curves, short, colors, alpha_star, b_CT1, b_CT2,
                          top_n=4, out_path=None):
    """Show R/G ratio vs b_i for the top-N most influential guilds."""
    fig, axes = plt.subplots(1, top_n, figsize=(3.5 * top_n, 4), sharey=True)

    effect_sizes = {s: abs(v[1][-1] - v[1][0]) for s, v in sweep_curves.items()}
    top_guilds   = sorted(effect_sizes, key=effect_sizes.get, reverse=True)[:top_n]

    for ax, gname in zip(axes, top_guilds):
        b_vals, gdi_vals = sweep_curves[gname]
        gi = short.index(gname)
        ax.plot(b_vals, gdi_vals, color=colors[gi], lw=2)
        ax.axhline(0, color='grey', lw=1, ls='--')
        # Mark CT1 and CT2 b values
        ax.axvline(b_CT2[gi], color='#d6604d', lw=1, ls=':',
                   label=f'CT2: b={b_CT2[gi]:.2f}')
        ax.axvline(b_CT1[gi], color='#2166ac', lw=1, ls=':',
                   label=f'CT1: b={b_CT1[gi]:.2f}')

        # Mark zero crossing if any
        sc = np.where(np.diff(np.sign(gdi_vals)))[0]
        if len(sc):
            bstar = float(b_vals[sc[0]])
            ax.axvline(bstar, color='black', lw=1.2, ls='--',
                       label=f'b*={bstar:.2f}')

        ax.set_xlabel(f'b_{gname}', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Mean R/G ratio (week-3)', fontsize=9)
        ax.set_title(gname, fontsize=9, color=colors[gi])
        ax.legend(fontsize=6)

    fig.suptitle('R/G ratio tipping curves: sweep one guild\'s b at a time', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def plot_convergence_trajectories(A, b_all, phi0_pp, short, colors, patients, ct,
                                   out_path=None):
    """CT1 & CT2 patients converge to the same attractor over 150 days."""
    ct_colors_map = {1: '#2166ac', 2: '#d6604d'}
    ct1_pat = [i for i, c in enumerate(ct) if c == 1][:3]
    ct2_pat = [i for i, c in enumerate(ct) if c == 2][:3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A: R/G ratio over time
    ax = axes[0]
    for p in ct1_pat + ct2_pat:
        traj = simulate(A, b_all[p], phi0_pp[p], t_end=T_END_LONG, rtol=1e-5)
        t    = np.linspace(0, T_END_LONG, len(traj))
        gdi_t = [gdi(traj[ti], short) for ti in range(len(traj))]
        c = ct[p]
        ax.plot(t, gdi_t, color=ct_colors_map[c], lw=1.5, alpha=0.8)
    ax.axhline(0, color='grey', lw=1, ls='--', label='R/G=0 boundary')
    ax.axvline(21, color='k', lw=1, ls=':', alpha=0.5, label='Week 3 (data end)')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#2166ac', label='CT1 commensal'),
        Patch(color='#d6604d', label='CT2 dysbiotic'),
        plt.Line2D([0],[0], color='grey', ls='--', label='R/G=0'),
        plt.Line2D([0],[0], color='k', ls=':', label='Week 3'),
    ], fontsize=7)
    ax.set_xlabel('Days', fontsize=9)
    ax.set_ylabel('R/G ratio', fontsize=9)
    ax.set_title('All patients converge to same attractor\n(R/G ratio < 0 = commensal)', fontsize=9)

    # Panel B: week-0 vs equilibrium composition
    ax = axes[1]
    eq_comp = np.array([equilibrium(A, b_all[p], phi0_pp[p]) for p in range(len(b_all))])
    x = np.arange(len(short))
    width = 0.35
    ax.bar(x - width/2, phi0_pp.mean(axis=0), width, color=colors, alpha=0.5, label='Week 0')
    ax.bar(x + width/2, eq_comp.mean(axis=0),  width, color=colors, alpha=0.9, label='Equilibrium')
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, fontsize=8)
    ax.set_ylabel('Mean guild fraction', fontsize=9)
    ax.set_title('Week-0 vs long-time equilibrium\n(all patients converge)', fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle('Single-attractor structure of fitted A matrix', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig, eq_comp


def plot_b_vs_equilibrium(A, b_all, phi0_pp, short, colors, patients, ct,
                           out_path=None):
    """Scatter b_i vs equilibrium phi*_i for top-6 variable guilds."""
    eq_comp = np.array([equilibrium(A, b_all[p], phi0_pp[p]) for p in range(len(b_all))])
    ct_colors_map = {1: '#2166ac', 2: '#d6604d'}
    top6 = np.argsort(eq_comp.var(axis=0))[::-1][:6]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, gi in zip(axes.ravel(), top6):
        for p in range(len(b_all)):
            ax.scatter(b_all[p, gi], eq_comp[p, gi],
                       color=ct_colors_map[ct[p]], s=60, zorder=3)
        r = float(np.corrcoef(b_all[:, gi], eq_comp[:, gi])[0, 1])
        ax.set_xlabel(f'b_{short[gi]}', fontsize=8)
        ax.set_ylabel(f'phi*_{short[gi]}', fontsize=8)
        ax.set_title(f'{short[gi]}  r={r:+.2f}', fontsize=9, color=colors[gi])
        ax.tick_params(labelsize=7)

    from matplotlib.patches import Patch
    axes[0,0].legend(handles=[Patch(color='#2166ac', label='CT1'),
                               Patch(color='#d6604d', label='CT2')], fontsize=7)
    fig.suptitle('Growth rate b_i vs equilibrium abundance phi*_i\n'
                 '(top 6 variable guilds)', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def main():
    A, b_all, phi0_pp, guilds, short, colors, patients = load_data()
    ct    = assign_ct(b_all, patients)
    b_CT1 = b_all[ct == 1].mean(axis=0)
    b_CT2 = b_all[ct == 2].mean(axis=0)
    print(f'CT1: {[patients[i] for i in range(len(patients)) if ct[i]==1]}')
    print(f'CT2: {[patients[i] for i in range(len(patients)) if ct[i]==2]}')

    # 1. Alpha scan
    print('\n=== Alpha scan ===')
    alphas, gdi_mean, gdi_std, gdi_mat, alpha_star, b_CT1, b_CT2 = \
        alpha_scan(A, b_all, phi0_pp, short, patients)
    print(f'  R/G ratio at alpha=0 (CT2): {gdi_mean[0]:.3f}')
    print(f'  R/G ratio at alpha=1 (CT1): {gdi_mean[-1]:.3f}')
    if alpha_star is None:
        print('  No crossing: single commensal attractor (A matrix dominant)')
    else:
        print(f'  Tipping alpha* = {alpha_star:.2f}')
    plot_alpha_scan(alphas, gdi_mean, gdi_std, gdi_mat, alpha_star,
                    patients, ct, out_path=OUT_DIR / 'tipping_alpha_scan.png')

    # 2. Long-time convergence
    print('\n=== Convergence trajectories ===')
    fig_conv, eq_comp = plot_convergence_trajectories(
        A, b_all, phi0_pp, short, colors, patients, ct,
        out_path=OUT_DIR / 'tipping_convergence.png')
    eq_gdi = [gdi(eq_comp[p], short) for p in range(len(b_all))]
    print(f'  All equilibrium R/G ratio < 0: {all(g < 0 for g in eq_gdi)}')

    # 3. b_i vs equilibrium
    print('\n=== b vs equilibrium ===')
    plot_b_vs_equilibrium(A, b_all, phi0_pp, short, colors, patients, ct,
                          out_path=OUT_DIR / 'tipping_b_vs_eq.png')

    # 4. Single-guild sweep
    print('\n=== Single-guild b sweep ===')
    effect_sizes, sweep_curves, _, _ = \
        single_guild_sweep(A, b_all, phi0_pp, short, patients, n_steps=15)
    order = np.argsort(effect_sizes)[::-1]
    print(f'  {"Guild":<6} {"ΔR/G":>7}  {"Δb":>7}')
    for i in order:
        print(f'  {short[i]:<6} {effect_sizes[i]:7.3f}  {b_CT1[i]-b_CT2[i]:+7.3f}')
    plot_single_guild_effects(effect_sizes, short, colors, b_CT1, b_CT2,
                               out_path=OUT_DIR / 'tipping_single_guild.png')
    plot_top_sweep_curves(sweep_curves, short, colors, alpha_star, b_CT1, b_CT2,
                          top_n=4, out_path=OUT_DIR / 'tipping_sweep_curves.png')

    # 5. 2D phase diagram
    top2 = list(np.argsort(effect_sizes)[::-1][:2])
    gi, gj = top2[0], top2[1]
    print(f'\n=== 2D phase diagram: {short[gi]} x {short[gj]} ===')
    b_i_vals, b_j_vals, gdi_grid = \
        phase_diagram_2d(A, b_all, phi0_pp, short, patients, gi, gj)
    plot_phase_diagram(b_i_vals, b_j_vals, gdi_grid, short, gi, gj, b_CT1, b_CT2,
                       out_path=OUT_DIR / 'tipping_phase_diagram_2d.png')

    summary = {
        'alpha_star': alpha_star,
        'b_CT1': dict(zip(short, b_CT1.tolist())),
        'b_CT2': dict(zip(short, b_CT2.tolist())),
        'effect_sizes': dict(zip(short, effect_sizes.tolist())),
        'equilibrium_gdi': dict(zip(patients, [float(g) for g in eq_gdi])),
        'all_commensal_eq': all(g < 0 for g in eq_gdi),
    }
    (OUT_DIR / 'tipping_summary.json').write_text(
        __import__('json').dumps(summary, indent=2))
    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
