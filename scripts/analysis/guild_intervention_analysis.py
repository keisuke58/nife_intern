#!/usr/bin/env python3
"""
guild_intervention_analysis.py — In-silico intervention targeting for 10-guild Dieckow gLV.

Simulates two types of interventions applied at t=0 (before week 1):
  - Suppression : φ_i(0) ← φ_i(0) × (1 − f),  f ∈ [0,1]   (antibiotic / phage)
  - Boost       : φ_i(0) ← φ_i(0) + δ × φ_total(0)         (probiotic / transplant)

After intervention, the community is renormalised and propagated forward with the
fitted gLV A matrix and patient-specific b vectors.

"Success" is measured as Bray-Curtis distance to a healthy reference composition
(mean observed week-3 across all patients — post-caries-treatment).

Outputs:
  - Heatmap: BC-to-target improvement per (guild, intervention strength)
  - Bar chart: best single-guild intervention ranked by improvement
  - Trajectory plots: baseline vs best intervention per patient
  - JSON summary

Usage:
    python guild_intervention_analysis.py
    python guild_intervention_analysis.py --mode suppress --frac 0.9
    python guild_intervention_analysis.py --mode boost    --delta 0.3
    python guild_intervention_analysis.py --mode both
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse
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

HERE    = Path(__file__).resolve().parents[2]
OUT_DIR = HERE / 'results' / 'guild_intervention'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_JSON = HERE / 'results' / 'dieckow_cr' / 'fit_guild.json'
PHI_NPY  = HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy'

T_END  = 21.0
T_EVAL = np.linspace(0, T_END, 300)
T_WEEK = [0.0, 7.0, 14.0, 21.0]   # weeks 0-3 in days

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


# ── Load ──────────────────────────────────────────────────────────────────

def load_data():
    d      = json.loads(FIT_JSON.read_text())
    A      = np.array(d['A'])
    b_all  = np.array(d['b_all'])          # (n_pat, n_guilds)
    guilds = d['guilds']
    short  = [GUILD_SHORT.get(g, g[:4]) for g in guilds]
    colors = [CMAP(i / len(guilds)) for i in range(len(guilds))]

    phi_all = np.load(PHI_NPY)             # (n_pat, n_weeks, n_guilds)
    phi0_pp = phi_all[:, 0, :]             # week-0 per patient
    sums    = phi0_pp.sum(axis=1, keepdims=True)
    phi0_pp = np.where(sums > 1e-12, phi0_pp / sums, phi0_pp)

    # Healthy reference: mean week-3 composition across patients
    phi_ref = phi_all[:, -1, :].mean(axis=0)
    phi_ref /= phi_ref.sum() + 1e-12

    return A, b_all, phi0_pp, phi_all, phi_ref, guilds, short, colors


# ── gLV ──────────────────────────────────────────────────────────────────

def glv_rhs(t, phi, A, b):
    phi = np.maximum(phi, 0)
    return phi * (b + A @ phi)


def simulate(A, b, phi0, t_eval=T_EVAL):
    sol = solve_ivp(glv_rhs, [0, t_eval[-1]], phi0,
                    args=(A, b), t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10,
                    dense_output=False)
    traj = sol.y.T
    # pad to full length if solver stopped early
    if len(traj) < len(t_eval):
        pad  = np.tile(traj[-1], (len(t_eval) - len(traj), 1))
        traj = np.vstack([traj, pad])
    return traj


def phi_at_week(traj, week=3):
    """Interpolate trajectory at given week (days = week*7)."""
    t_target = week * 7.0
    idx = np.searchsorted(T_EVAL, t_target)
    idx = min(idx, len(traj) - 1)
    return traj[idx]


def bray_curtis(p, q):
    return np.sum(np.abs(p - q)) / (np.sum(p) + np.sum(q) + 1e-12)


# ── Intervention ──────────────────────────────────────────────────────────

def apply_suppression(phi0, guild_idx, fraction):
    """Suppress guild i by fraction f: φ_i ← φ_i × (1-f), renormalise."""
    phi = phi0.copy()
    phi[guild_idx] *= (1.0 - fraction)
    total = phi.sum()
    return phi / total if total > 1e-12 else phi


def apply_boost(phi0, guild_idx, delta):
    """Boost guild i by delta × φ_total, renormalise."""
    phi = phi0.copy()
    phi[guild_idx] += delta * phi0.sum()
    total = phi.sum()
    return phi / total if total > 1e-12 else phi


# ── Scan all guilds × strengths ───────────────────────────────────────────

def scan_interventions(A, b_all, phi0_pp, phi_ref, guilds,
                       fracs=None, deltas=None):
    """
    For each guild and each intervention strength, compute
    mean BC improvement across patients.

    Returns:
        suppress_grid : (n_guilds, n_fracs)   BC improvement (positive = better)
        boost_grid    : (n_guilds, n_deltas)
    """
    if fracs  is None: fracs  = np.linspace(0.1, 0.9, 9)
    if deltas is None: deltas = np.linspace(0.05, 0.5, 10)

    n_g   = len(guilds)
    n_pat = len(b_all)

    suppress_grid = np.zeros((n_g, len(fracs)))
    boost_grid    = np.zeros((n_g, len(deltas)))

    for p in range(n_pat):
        phi0 = phi0_pp[p]
        b    = b_all[p]

        # Baseline BC at week 3
        base_traj = simulate(A, b, phi0)
        base_w3   = phi_at_week(base_traj, 3)
        base_bc   = bray_curtis(base_w3, phi_ref)

        for i in range(n_g):
            # Suppression
            for fi, f in enumerate(fracs):
                phi_int  = apply_suppression(phi0, i, f)
                traj_int = simulate(A, b, phi_int)
                w3_int   = phi_at_week(traj_int, 3)
                bc_int   = bray_curtis(w3_int, phi_ref)
                suppress_grid[i, fi] += (base_bc - bc_int) / n_pat  # improvement

            # Boost
            for di, d in enumerate(deltas):
                phi_int  = apply_boost(phi0, i, d)
                traj_int = simulate(A, b, phi_int)
                w3_int   = phi_at_week(traj_int, 3)
                bc_int   = bray_curtis(w3_int, phi_ref)
                boost_grid[i, di] += (base_bc - bc_int) / n_pat

    return suppress_grid, boost_grid, fracs, deltas


def best_interventions(suppress_grid, boost_grid, short, fracs, deltas, top_n=5):
    """Return ranked list of best interventions."""
    results = []
    for i, s in enumerate(short):
        best_s_val = suppress_grid[i].max()
        best_s_f   = fracs[suppress_grid[i].argmax()]
        best_b_val = boost_grid[i].max()
        best_b_d   = deltas[boost_grid[i].argmax()]
        results.append({
            'guild': s,
            'suppress_improvement': float(best_s_val),
            'suppress_fraction':    float(best_s_f),
            'boost_improvement':    float(best_b_val),
            'boost_delta':          float(best_b_d),
        })
    results.sort(key=lambda x: max(x['suppress_improvement'], x['boost_improvement']),
                 reverse=True)
    return results[:top_n]


# ── Trajectory plots ──────────────────────────────────────────────────────

def plot_best_trajectory(A, b_all, phi0_pp, phi_ref, best, guilds, short, colors,
                         out_path=None):
    """Show baseline vs best suppression intervention (top guild) for 3 patients."""
    top = best[0]
    gi  = short.index(top['guild'])
    f   = top['suppress_fraction']

    n_show = min(3, len(b_all))
    fig, axes = plt.subplots(n_show, len(short),
                             figsize=(2.0 * len(short), 2.5 * n_show))

    for p in range(n_show):
        phi0 = phi0_pp[p]
        b    = b_all[p]

        base_traj = simulate(A, b, phi0)
        phi_int   = apply_suppression(phi0, gi, f)
        int_traj  = simulate(A, b, phi_int)

        for s in range(len(short)):
            ax = axes[p, s]
            ax.plot(T_EVAL, base_traj[:, s], color=colors[s], lw=1.5, label='Baseline')
            ax.plot(T_EVAL, int_traj[:, s],  color=colors[s], lw=1.5,
                    ls='--', alpha=0.7, label=f'Suppress {top["guild"]} {f:.0%}')
            ax.axvline(0, color='grey', lw=0.5, ls=':')
            ax.set_ylim(0, 1)
            if p == 0:
                ax.set_title(short[s], fontsize=8, color=colors[s])
            if s == 0:
                ax.set_ylabel(f'P{p+1}', fontsize=8)
            if p == n_show - 1:
                ax.set_xlabel('Day', fontsize=7)
            ax.tick_params(labelsize=6)

    # Legend
    axes[0, -1].legend(fontsize=6, loc='upper right')

    bc_base = np.mean([bray_curtis(phi_at_week(simulate(A, b_all[p], phi0_pp[p]), 3), phi_ref)
                       for p in range(len(b_all))])
    bc_int  = np.mean([bray_curtis(
        phi_at_week(simulate(A, b_all[p], apply_suppression(phi0_pp[p], gi, f)), 3), phi_ref)
        for p in range(len(b_all))])

    fig.suptitle(
        f'Best intervention: suppress {top["guild"]} by {f:.0%}\n'
        f'BC to healthy ref: {bc_base:.3f} → {bc_int:.3f} '
        f'(Δ={bc_base-bc_int:+.3f})',
        fontsize=9,
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Heatmap ───────────────────────────────────────────────────────────────

def plot_intervention_heatmap(suppress_grid, boost_grid, short, fracs, deltas,
                               out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, grid, x_vals, x_label, title in [
        (axes[0], suppress_grid, fracs,  'Suppression fraction f',
         'Suppression: BC improvement (Δ > 0 = closer to healthy)'),
        (axes[1], boost_grid,    deltas, 'Boost delta δ',
         'Boost: BC improvement'),
    ]:
        vmax = max(abs(grid).max(), 1e-6)
        im   = ax.imshow(grid, aspect='auto', cmap='RdYlGn',
                         vmin=-vmax, vmax=vmax, origin='upper')
        ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=9)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f'{v:.2f}' for v in x_vals], rotation=45, fontsize=7)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, label='ΔBC (mean over patients)', fraction=0.046)

        # annotate best per guild
        for i in range(len(short)):
            best_j = grid[i].argmax()
            ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor='black', lw=1.5))

    fig.suptitle('In-silico intervention landscape (10-guild Dieckow gLV)', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_best_bar(best, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    guilds_s = [r['guild'] for r in best]
    impr_s   = [r['suppress_improvement'] for r in best]
    impr_b   = [r['boost_improvement'] for r in best]
    frac_s   = [r['suppress_fraction'] for r in best]
    delta_b  = [r['boost_delta'] for r in best]
    colors   = plt.get_cmap('tab10')(np.linspace(0, 1, len(best)))

    for ax, vals, labels, title, param_label, params in [
        (axes[0], impr_s, guilds_s, 'Best suppression per guild',
         'f=', frac_s),
        (axes[1], impr_b, guilds_s, 'Best boost per guild',
         'δ=', delta_b),
    ]:
        bars = ax.bar(range(len(best)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(best)))
        ax.set_xticklabels([f'{g}\n({param_label}{p:.2f})'
                            for g, p in zip(labels, params)],
                           fontsize=8)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('Mean BC improvement (Δ)', fontsize=9)
        ax.set_title(title, fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Top intervention targets — Dieckow 10-guild', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  choices=['suppress','boost','both'], default='both')
    parser.add_argument('--frac',  type=float, default=None,
                        help='Single suppression fraction (skips scan)')
    parser.add_argument('--delta', type=float, default=None,
                        help='Single boost delta (skips scan)')
    args = parser.parse_args()

    A, b_all, phi0_pp, phi_all, phi_ref, guilds, short, colors = load_data()
    n_pat = len(b_all)

    print(f'Loaded: {len(guilds)} guilds, {n_pat} patients')
    print(f'Healthy reference (week-3 mean): {dict(zip(short, phi_ref.round(3)))}')

    # Baseline BC across patients
    bc_baselines = [
        bray_curtis(phi_at_week(simulate(A, b_all[p], phi0_pp[p]), 3), phi_ref)
        for p in range(n_pat)
    ]
    print(f'Baseline BC to healthy ref: {np.mean(bc_baselines):.4f} ± {np.std(bc_baselines):.4f}')

    fracs  = np.linspace(0.1, 0.9, 9)
    deltas = np.linspace(0.05, 0.5, 10)

    print('\nScanning interventions ...')
    suppress_grid, boost_grid, fracs, deltas = scan_interventions(
        A, b_all, phi0_pp, phi_ref, guilds, fracs=fracs, deltas=deltas
    )

    best = best_interventions(suppress_grid, boost_grid, short, fracs, deltas)

    print('\nTop interventions (ranked by max improvement):')
    print(f'  {"Guild":<6} {"Suppress Δ":>12} {"  at f":>6}  {"Boost Δ":>10} {"  at δ":>6}')
    for r in best:
        print(f'  {r["guild"]:<6} {r["suppress_improvement"]:12.5f} '
              f'{r["suppress_fraction"]:6.2f}  '
              f'{r["boost_improvement"]:10.5f} {r["boost_delta"]:6.2f}')

    plot_intervention_heatmap(suppress_grid, boost_grid, short, fracs, deltas,
                              out_path=OUT_DIR / 'intervention_heatmap.png')
    plot_best_bar(best, out_path=OUT_DIR / 'intervention_best_bar.png')
    plot_best_trajectory(A, b_all, phi0_pp, phi_ref, best, guilds, short, colors,
                         out_path=OUT_DIR / 'intervention_trajectory.png')

    # Save
    summary = {
        'baseline_bc_mean':  float(np.mean(bc_baselines)),
        'baseline_bc_std':   float(np.std(bc_baselines)),
        'healthy_reference': dict(zip(short, phi_ref.tolist())),
        'top_interventions': best,
        'suppress_grid':     suppress_grid.tolist(),
        'boost_grid':        boost_grid.tolist(),
        'fracs':             fracs.tolist(),
        'deltas':            deltas.tolist(),
        'guilds':            short,
    }
    (OUT_DIR / 'intervention_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
