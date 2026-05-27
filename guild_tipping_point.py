#!/usr/bin/env python3
"""
guild_tipping_point.py — Tipping point / bifurcation analysis for 10-guild Dieckow gLV.

Question: how much do growth rates b need to shift (from CT2 dysbiotic → CT1 commensal
environment) before the community flips to a healthy state?

Analysis:
  1. Cluster patients into CT1 / CT2 using k-means on b_all (PC1 of b).
  2. Linear interpolation: b(α) = (1-α)·b_CT2 + α·b_CT1,  α ∈ [0,1].
  3. For each α, simulate gLV from every patient's week-0 phi0; record
     week-3 GDI = log(φ_dys) - log(φ_com).
  4. Find tipping α* where mean GDI crosses zero.
  5. Single-guild b sweep: scan b_i from b_CT2[i] to b_CT1[i] one at a time
     to identify which guild's growth rate drives the tipping point.
  6. 2D phase diagram for the two most critical guilds.

GDI definition (consistent with Joshi validation):
  φ_dys = φ_Bact + φ_Fuso + φ_Clos
  φ_com = φ_Baci + φ_Acti + φ_Nega
  GDI < 0 → CT1 (commensal), GDI > 0 → CT2 (dysbiotic)
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

T_END  = 21.0
T_EVAL = np.linspace(0, T_END, 300)

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


def simulate(A, b, phi0):
    sol = solve_ivp(glv_rhs, [0, T_END], phi0,
                    args=(A, b), t_eval=T_EVAL,
                    method='RK45', rtol=1e-8, atol=1e-10)
    traj = sol.y.T
    if len(traj) < len(T_EVAL):
        pad  = np.tile(traj[-1], (len(T_EVAL) - len(traj), 1))
        traj = np.vstack([traj, pad])
    return traj


def week3(traj):
    """Return composition at t=21 days."""
    return traj[-1]


def gdi(phi, short):
    """Guild Dysbiosis Index (log ratio)."""
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

    # Find tipping point: where mean GDI crosses zero
    sign_changes = np.where(np.diff(np.sign(gdi_mean)))[0]
    alpha_star   = float(alphas[sign_changes[0]]) if len(sign_changes) else None

    return alphas, gdi_mean, gdi_std, gdi_mat, alpha_star, b_CT1, b_CT2


# ── Single-guild b sweep ──────────────────────────────────────────────────────

def single_guild_sweep(A, b_all, phi0_pp, short, patients, n_steps=40):
    """
    For each guild i, sweep b_i from b_CT2[i] to b_CT1[i] while keeping
    all other b components at b_CT2.  Record mean GDI at week-3.
    Returns: effect_sizes (n_guilds,) = |GDI(b_CT1[i]) - GDI(b_CT2[i])|
    """
    ct    = assign_ct(b_all, patients)
    b_CT1 = b_all[ct == 1].mean(axis=0)
    b_CT2 = b_all[ct == 2].mean(axis=0)
    n_g   = len(short)
    n_pat = len(b_all)

    effect_sizes = np.zeros(n_g)
    sweep_curves = {}   # guild → (steps, gdi_mean)

    for i in range(n_g):
        b_vals = np.linspace(b_CT2[i], b_CT1[i], n_steps)
        gdi_vals = []
        for bv in b_vals:
            b_test     = b_CT2.copy()
            b_test[i]  = bv
            gdi_list = []
            for p in range(n_pat):
                traj = simulate(A, b_test, phi0_pp[p])
                gdi_list.append(gdi(week3(traj), short))
            gdi_vals.append(np.mean(gdi_list))
        gdi_arr = np.array(gdi_vals)
        effect_sizes[i] = abs(gdi_arr[-1] - gdi_arr[0])
        sweep_curves[short[i]] = (b_vals, gdi_arr)

    return effect_sizes, sweep_curves, b_CT1, b_CT2


# ── 2D phase diagram ──────────────────────────────────────────────────────────

def phase_diagram_2d(A, b_all, phi0_pp, short, patients, gi, gj, n_grid=20):
    """
    Scan b_i and b_j on a 2D grid (from b_CT2[i/j] to b_CT1[i/j] + 20% overshoot).
    Returns: b_i_vals, b_j_vals, gdi_grid (n_grid, n_grid)
    """
    ct    = assign_ct(b_all, patients)
    b_CT1 = b_all[ct == 1].mean(axis=0)
    b_CT2 = b_all[ct == 2].mean(axis=0)
    n_pat = len(b_all)

    margin = 0.20
    b_i_range = np.linspace(min(b_CT2[gi], b_CT1[gi]) - margin * abs(b_CT1[gi] - b_CT2[gi]),
                             max(b_CT2[gi], b_CT1[gi]) + margin * abs(b_CT1[gi] - b_CT2[gi]),
                             n_grid)
    b_j_range = np.linspace(min(b_CT2[gj], b_CT1[gj]) - margin * abs(b_CT1[gj] - b_CT2[gj]),
                             max(b_CT2[gj], b_CT1[gj]) + margin * abs(b_CT1[gj] - b_CT2[gj]),
                             n_grid)

    gdi_grid = np.zeros((n_grid, n_grid))
    for ii, bvi in enumerate(b_i_range):
        for jj, bvj in enumerate(b_j_range):
            b_test     = b_CT2.copy()
            b_test[gi] = bvi
            b_test[gj] = bvj
            gdi_list = []
            for p in range(n_pat):
                traj = simulate(A, b_test, phi0_pp[p])
                gdi_list.append(gdi(week3(traj), short))
            gdi_grid[ii, jj] = np.mean(gdi_list)

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
    ax.plot(alphas, gdi_mean, 'k-', lw=2, label='Mean GDI')

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
    ax.set_ylabel('Guild Dysbiosis Index (GDI)', fontsize=9)
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
    ax.set_ylabel('|ΔGDI|  (b_CT2[i] → b_CT1[i])', fontsize=9)
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

    fig.colorbar(im, ax=ax, label='Mean GDI (week-3)')
    ax.set_xlabel(f'b_{short[gj]}  (growth rate of {short[gj]})', fontsize=9)
    ax.set_ylabel(f'b_{short[gi]}  (growth rate of {short[gi]})', fontsize=9)
    ax.set_title(f'2D phase diagram: {short[gi]} × {short[gj]}\n'
                 'Black dashed: GDI = 0 (tipping boundary)', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_top_sweep_curves(sweep_curves, short, colors, alpha_star, b_CT1, b_CT2,
                          top_n=4, out_path=None):
    """Show GDI vs b_i for the top-N most influential guilds."""
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
            ax.set_ylabel('Mean GDI (week-3)', fontsize=9)
        ax.set_title(gname, fontsize=9, color=colors[gi])
        ax.legend(fontsize=6)

    fig.suptitle('GDI tipping curves: sweep one guild\'s b at a time', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    A, b_all, phi0_pp, guilds, short, colors, patients = load_data()
    ct = assign_ct(b_all, patients)
    n_g = len(guilds)

    print('CT labels:', dict(zip(patients, ct)))
    print('CT1 patients:', [patients[i] for i in range(len(patients)) if ct[i] == 1])
    print('CT2 patients:', [patients[i] for i in range(len(patients)) if ct[i] == 2])

    # ── 1. Alpha scan ──────────────────────────────────────────────────────────
    print('\nAlpha scan (b interpolation CT2 → CT1) ...')
    alphas, gdi_mean, gdi_std, gdi_mat, alpha_star, b_CT1, b_CT2 = \
        alpha_scan(A, b_all, phi0_pp, short, patients)

    print(f'  Tipping point α* = {alpha_star}')
    if alpha_star is not None:
        b_star = (1 - alpha_star) * b_CT2 + alpha_star * b_CT1
        print(f'  b at tipping: {dict(zip(short, b_star.round(3)))}')
        delta_b = b_star - b_CT2
        print(f'  Δb from CT2 baseline: {dict(zip(short, delta_b.round(3)))}')

    plot_alpha_scan(alphas, gdi_mean, gdi_std, gdi_mat, alpha_star,
                    patients, ct,
                    out_path=OUT_DIR / 'tipping_alpha_scan.png')

    # ── 2. Single-guild sweep ──────────────────────────────────────────────────
    print('\nSingle-guild b sweep ...')
    effect_sizes, sweep_curves, _, _ = \
        single_guild_sweep(A, b_all, phi0_pp, short, patients)

    print('\nGuild tipping contributions (|ΔGDI|):')
    order = np.argsort(effect_sizes)[::-1]
    for i in order:
        print(f'  {short[i]:<6}  ΔGDI={effect_sizes[i]:.3f}  '
              f'b_CT2={b_CT2[i]:.3f} → b_CT1={b_CT1[i]:.3f}  '
              f'Δb={b_CT1[i]-b_CT2[i]:+.3f}')

    plot_single_guild_effects(effect_sizes, short, colors, b_CT1, b_CT2,
                               out_path=OUT_DIR / 'tipping_single_guild.png')

    plot_top_sweep_curves(sweep_curves, short, colors, alpha_star, b_CT1, b_CT2,
                          top_n=4, out_path=OUT_DIR / 'tipping_sweep_curves.png')

    # ── 3. 2D phase diagram (top 2 guilds) ────────────────────────────────────
    top2 = list(np.argsort(effect_sizes)[::-1][:2])
    gi, gj = top2[0], top2[1]
    if not args.skip_2d:
        print(f'\n2D phase diagram: {short[gi]} x {short[gj]} ...')
        b_i_vals, b_j_vals, gdi_grid = \
            phase_diagram_2d(A, b_all, phi0_pp, short, patients, gi, gj)
        plot_phase_diagram(b_i_vals, b_j_vals, gdi_grid, short, gi, gj, b_CT1, b_CT2,
                           out_path=OUT_DIR / 'tipping_phase_diagram_2d.png')
    else:
        print('\n2D phase diagram skipped (--skip-2d).')

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        'alpha_star':    alpha_star,
        'b_CT1':         dict(zip(short, b_CT1.tolist())),
        'b_CT2':         dict(zip(short, b_CT2.tolist())),
        'effect_sizes':  dict(zip(short, effect_sizes.tolist())),
        'top2_guilds':   [short[gi], short[gj]],
        'ct_labels':     dict(zip(patients, ct.tolist())),
    }
    (OUT_DIR / 'tipping_summary.json').write_text(
        __import__('json').dumps(summary, indent=2))
    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
