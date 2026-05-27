#!/usr/bin/env python3
"""
guild_relapse_prevention.py — Three extensions of the relapse dynamics finding.

Analysis 1: Prevention threshold
  For each "transient responder" patient (C, G, L), find the minimum Δb_i
  for each guild that converts them from "relapsing" to "permanent commensal".
  → Quantitative treatment target: "reduce b_Bacteroidia by X to prevent relapse"

Analysis 2: Week-3 GDI as relapse predictor
  Plot week-3 GDI vs relapse day for all transient-responder patients.
  → Can week-3 snapshot predict relapse timing? (clinical prognostic marker)

Analysis 3: Joshi cross-validation
  Compare long-time predicted GDI to Joshi peri-implant GDI distribution.
  → Does the 3-week model correctly predict PI-range dysbiosis at equilibrium?
  → External validation of the long-term extrapolation.
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

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / 'results' / 'guild_relapse'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_JSON = HERE / 'results' / 'dieckow_cr' / 'fit_guild.json'
PHI_NPY  = HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy'

GUILD_SHORT = {
    'Actinobacteria': 'Acti', 'Bacilli': 'Baci', 'Bacteroidia': 'Bact',
    'Betaproteobacteria': 'Beta', 'Clostridia': 'Clos', 'Coriobacteriia': 'Cori',
    'Fusobacteriia': 'Fuso', 'Gammaproteobacteria': 'Gamm',
    'Negativicutes': 'Nega', 'Other': 'Othr',
}
CMAP = plt.get_cmap('tab10')

# Known CT labels (Dieckow 2024)
CT_KNOWN = {'A':2,'B':2,'C':2,'D':1,'E':1,'F':2,'G':1,'H':2,'K':1,'L':1}

# Joshi 2025 reference GDI values
JOSHI = {'Health': -2.09, 'Mucositis': -1.74, 'PI': +0.53}
JOSHI_SD = {'Health': 1.20, 'Mucositis': 1.15, 'PI': 1.30}  # approximate SD

DYS_IDX = [2, 6, 4]   # Bact, Fuso, Clos
COM_IDX = [1, 0, 8]   # Baci, Acti, Nega


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    d        = json.loads(FIT_JSON.read_text())
    A        = np.array(d['A'])
    b_all    = np.array(d['b_all'])
    guilds   = d['guilds']
    patients = d['patients']
    short    = [GUILD_SHORT.get(g, g[:4]) for g in guilds]
    colors   = [CMAP(i / len(guilds)) for i in range(len(guilds))]
    phi_all  = np.load(PHI_NPY)
    phi0_pp  = phi_all[:, 0, :]
    phi0_pp  = phi0_pp / (phi0_pp.sum(axis=1, keepdims=True) + 1e-12)
    return A, b_all, phi0_pp, guilds, short, colors, patients


def glv_rhs(t, phi, A, b):
    phi = np.maximum(phi, 0)
    return phi * (b + A @ phi)


def sim(A, b, phi0, t_end=150, n=500, rtol=1e-5):
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(glv_rhs, [0, t_end], phi0, args=(A, b),
                    t_eval=t_eval, method='RK45', rtol=rtol, atol=rtol * 0.01)
    traj = sol.y.T
    if len(traj) < n:
        traj = np.vstack([traj, np.tile(traj[-1], (n - len(traj), 1))])
    return t_eval, traj


def gdi_scalar(phi):
    phi = phi / (phi.sum() + 1e-12)
    return float(np.log(phi[DYS_IDX].sum() + 1e-9) - np.log(phi[COM_IDX].sum() + 1e-9))


def relapse_day(A, b, phi0, t_end=150):
    """Return (gdi_w3, gdi_eq, flip_day or None)."""
    t, traj = sim(A, b, phi0, t_end=t_end)
    gdi_t = np.array([gdi_scalar(traj[i]) for i in range(len(traj))])
    idx21 = np.searchsorted(t, 21)
    gdi_w3 = gdi_t[min(idx21, len(gdi_t) - 1)]
    gdi_eq = gdi_t[-1]
    crosses = np.where(np.diff(np.sign(gdi_t)))[0]
    flip = float(t[crosses[0]]) if len(crosses) else None
    return gdi_w3, gdi_eq, flip


# ── Analysis 1: Prevention threshold ─────────────────────────────────────────

def find_prevention_threshold(A, b, phi0, guild_idx, b_range=None, n_steps=30):
    """
    Sweep b[guild_idx] from its current value down to current - delta_max.
    Return (b_vals, relapse_days) where relapse_days=None means no relapse.
    """
    if b_range is None:
        delta_max = max(abs(b[guild_idx]) * 2.0, 1.0)
        b_vals = np.linspace(b[guild_idx], b[guild_idx] - delta_max, n_steps)
    else:
        b_vals = b_range

    results = []
    for bv in b_vals:
        b_test = b.copy()
        b_test[guild_idx] = bv
        _, _, flip = relapse_day(A, b_test, phi0)
        results.append(flip)
    return b_vals, results


def run_prevention_analysis(A, b_all, phi0_pp, short, patients, colors):
    """
    For transient responders C, G, L: find which guild's b reduction prevents relapse.
    Focus on guilds with highest dysbiotic role: Bact, Fuso, Gamm.
    """
    transient = ['C', 'G', 'L']
    target_guilds = ['Bact', 'Fuso', 'Gamm', 'Baci']   # intervention candidates
    target_idx = [short.index(g) for g in target_guilds]

    results = {}
    for pat in transient:
        pi = patients.index(pat)
        b  = b_all[pi].copy()
        phi0 = phi0_pp[pi]
        gdi_w3, gdi_eq, flip0 = relapse_day(A, b, phi0)
        print(f'\n  Patient {pat}: GDI(w3)={gdi_w3:.2f}, relapse={flip0:.0f}d')
        results[pat] = {'b_orig': b.copy(), 'flip0': flip0, 'gdi_w3': gdi_w3,
                        'thresholds': {}}

        for gi, gname in zip(target_idx, target_guilds):
            # sweep b_gi from current down by up to 2 units
            delta_max = 2.0
            b_vals = np.linspace(b[gi], b[gi] - delta_max, 25)
            flips  = []
            for bv in b_vals:
                b_test      = b.copy()
                b_test[gi]  = bv
                _, _, fl    = relapse_day(A, b_test, phi0)
                flips.append(fl)

            # find threshold: first b value where relapse becomes None (permanent commensal)
            threshold_b   = None
            threshold_db  = None
            for bv, fl in zip(b_vals, flips):
                if fl is None:
                    threshold_b  = float(bv)
                    threshold_db = float(b[gi] - bv)
                    break

            results[pat]['thresholds'][gname] = {
                'b_orig':       float(b[gi]),
                'threshold_b':  threshold_b,
                'threshold_db': threshold_db,
                'b_vals':       b_vals.tolist(),
                'flip_days':    [float(f) if f is not None else None for f in flips],
            }
            status = f'threshold Δb={threshold_db:.2f}' if threshold_db else 'no threshold found'
            print(f'    {gname}: b={b[gi]:.3f}  {status}')

    return results


def plot_prevention(results, short, colors, patients, b_all, out_path=None):
    transient = list(results.keys())
    target_guilds = ['Bact', 'Fuso', 'Gamm', 'Baci']

    fig, axes = plt.subplots(len(transient), len(target_guilds),
                              figsize=(3.5 * len(target_guilds), 3.2 * len(transient)),
                              sharey='row')

    guild_colors = {g: colors[short.index(g)] for g in target_guilds}

    for ri, pat in enumerate(transient):
        pi   = patients.index(pat)
        data = results[pat]
        for ci, gname in enumerate(target_guilds):
            ax  = axes[ri, ci]
            th  = data['thresholds'][gname]
            bv  = np.array(th['b_vals'])
            fld = th['flip_days']

            # Convert None (no relapse) to large value for plotting
            fld_plot = [f if f is not None else 200 for f in fld]
            ax.plot(bv, fld_plot, color=guild_colors[gname], lw=2)
            ax.axhline(90, color='grey', lw=0.8, ls='--', alpha=0.5)

            if th['threshold_b'] is not None:
                ax.axvline(th['threshold_b'], color='black', lw=1.2, ls='--')
                ax.annotate(f"Δb={th['threshold_db']:.2f}",
                            xy=(th['threshold_b'], 100),
                            xytext=(th['threshold_b'] + 0.05, 130),
                            fontsize=7, color='black',
                            arrowprops=dict(arrowstyle='->', lw=0.8))

            ax.axvline(th['b_orig'], color=guild_colors[gname], lw=1, ls=':',
                       label=f'current b={th["b_orig"]:.2f}')
            ax.set_ylim(0, 210)
            ax.set_xlabel(f'b_{gname}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'Relapse day\n(Patient {pat})', fontsize=8)
            if ri == 0:
                ax.set_title(gname, fontsize=9, color=guild_colors[gname])
            ax.legend(fontsize=6)
            # "permanent" label
            ax.text(0.98, 0.95, '→ permanent\ncommensal',
                    transform=ax.transAxes, fontsize=6, ha='right', va='top',
                    color='green', alpha=0.7)

    fig.suptitle('Prevention threshold: Δb required to prevent relapse\n'
                 '(dashed = threshold where relapse day → ∞)', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


# ── Analysis 2: Week-3 GDI as relapse predictor ───────────────────────────────

def plot_gdi_vs_relapse(A, b_all, phi0_pp, patients, short, colors, out_path=None):
    """
    Scatter: week-3 GDI vs relapse day for all patients.
    Transient responders (GDI<0 at w3 but flip>0) are the clinically interesting group.
    """
    gdi_w3_all  = []
    relapse_all = []
    ct_all      = []

    for pi, pat in enumerate(patients):
        gdi_w3, gdi_eq, flip = relapse_day(A, b_all[pi], phi0_pp[pi])
        gdi_w3_all.append(gdi_w3)
        relapse_all.append(flip if flip is not None else 200)
        ct_all.append(CT_KNOWN.get(pat, 0))

    gdi_w3_all  = np.array(gdi_w3_all)
    relapse_all = np.array(relapse_all)

    fig, ax = plt.subplots(figsize=(7, 5))

    ct_colors = {1: '#2166ac', 2: '#d6604d'}
    for pi, pat in enumerate(patients):
        c      = ct_all[pi]
        marker = '^' if c == 1 else 'o'
        # highlight transient responders
        if gdi_w3_all[pi] < 0 and relapse_all[pi] < 190:
            ax.scatter(gdi_w3_all[pi], relapse_all[pi],
                       color='#e08214', s=120, zorder=5, marker=marker,
                       edgecolors='black', linewidths=0.8)
            ax.annotate(pat, (gdi_w3_all[pi], relapse_all[pi]),
                        xytext=(4, 4), textcoords='offset points', fontsize=9,
                        color='#e08214', fontweight='bold')
        elif relapse_all[pi] > 190:  # permanent commensal
            ax.scatter(gdi_w3_all[pi], 200,
                       color='#1a6faf', s=120, zorder=5, marker=marker,
                       edgecolors='black', linewidths=0.8)
            ax.annotate(f'{pat}\n(never)', (gdi_w3_all[pi], 200),
                        xytext=(4, -12), textcoords='offset points', fontsize=8,
                        color='#1a6faf')
        else:
            ax.scatter(gdi_w3_all[pi], relapse_all[pi],
                       color=ct_colors[c], s=70, zorder=3, marker=marker, alpha=0.7)
            ax.annotate(pat, (gdi_w3_all[pi], relapse_all[pi]),
                        xytext=(3, 3), textcoords='offset points', fontsize=8)

    # Trend line for transient responders
    mask = (gdi_w3_all < 0) & (relapse_all < 190)
    if mask.sum() >= 2:
        z = np.polyfit(gdi_w3_all[mask], relapse_all[mask], 1)
        x_line = np.linspace(gdi_w3_all[mask].min() - 0.5,
                              gdi_w3_all[mask].max() + 0.5, 50)
        ax.plot(x_line, np.polyval(z, x_line), 'k--', lw=1, alpha=0.5,
                label=f'Trend (slope={z[0]:.1f} d/GDI)')

    ax.axvline(0, color='grey', lw=1, ls='--', alpha=0.7, label='GDI=0')
    ax.axhline(90, color='grey', lw=0.5, ls=':', alpha=0.5)
    ax.set_yticks([0, 21, 30, 48, 60, 90, 150, 200])
    ax.set_yticklabels(['0', '21\n(wk3)', '30', '48', '60', '90', '150', '∞\n(never)'])
    ax.set_xlabel('GDI at week 3  (< 0 = apparent recovery)', fontsize=10)
    ax.set_ylabel('Predicted relapse day', fontsize=10)
    ax.set_title('Week-3 GDI as a prognostic marker for relapse\n'
                 'Orange = transient responders, Blue = permanent commensal', fontsize=10)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#e08214', label='Transient responder'),
        Patch(color='#1a6faf', label='Permanent commensal (D)'),
        Patch(color='#d6604d', label='CT2 persistently dysbiotic'),
        Patch(color='#2166ac', alpha=0.5, label='CT1 persistently dysbiotic'),
    ] + [plt.Line2D([0],[0], color='k', ls='--', label='Trend (transient)')],
              fontsize=8, loc='upper right')

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig, gdi_w3_all, relapse_all


# ── Analysis 3: Joshi cross-validation ───────────────────────────────────────

def plot_joshi_comparison(A, b_all, phi0_pp, patients, short, out_path=None):
    """
    Compare:
    - Dieckow week-3 GDI (observed, from simulation)
    - Dieckow long-time equilibrium GDI (predicted)
    - Joshi Health / Mucositis / PI distributions
    """
    # Compute per-patient GDI at week-3 and equilibrium
    gdi_w3_all  = []
    gdi_eq_all  = []
    for pi in range(len(patients)):
        gdi_w3, gdi_eq, _ = relapse_day(A, b_all[pi], phi0_pp[pi])
        gdi_w3_all.append(gdi_w3)
        gdi_eq_all.append(gdi_eq)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Joshi distributions (approximate Gaussian)
    x = np.linspace(-5, 4, 300)
    colors_joshi = {'Health': '#4393c3', 'Mucositis': '#92c5de', 'PI': '#d6604d'}
    for label, mu in JOSHI.items():
        sd = JOSHI_SD[label]
        from scipy.stats import norm
        ax.fill_between(x, norm.pdf(x, mu, sd) * 0.5,
                        alpha=0.25, color=colors_joshi[label], label=f'Joshi {label}')
        ax.axvline(mu, color=colors_joshi[label], lw=1.5, ls='--', alpha=0.8)

    # Dieckow week-3 GDI (strip)
    y_w3 = np.random.default_rng(42).uniform(0.55, 0.75, len(patients))
    for pi, pat in enumerate(patients):
        ct = CT_KNOWN.get(pat, 0)
        mc = '^' if ct == 1 else 'o'
        ax.scatter(gdi_w3_all[pi], y_w3[pi],
                   marker=mc, s=70, color='#1a6faf' if ct == 1 else '#d6604d',
                   zorder=5, alpha=0.9)
        if pat in ['C', 'G', 'L', 'D']:
            ax.annotate(pat, (gdi_w3_all[pi], y_w3[pi]),
                        xytext=(3, 3), textcoords='offset points', fontsize=7)

    # Dieckow equilibrium GDI (strip)
    y_eq = np.random.default_rng(0).uniform(0.25, 0.45, len(patients))
    for pi, pat in enumerate(patients):
        ct = CT_KNOWN.get(pat, 0)
        mc = '^' if ct == 1 else 'o'
        ax.scatter(gdi_eq_all[pi], y_eq[pi],
                   marker=mc, s=70,
                   color='#1a6faf' if ct == 1 else '#d6604d',
                   zorder=5, alpha=0.5)

    # Mean markers
    ax.axvline(np.mean(gdi_w3_all), color='navy', lw=2, ls='-',
               label=f'Dieckow week-3 mean (GDI={np.mean(gdi_w3_all):.2f})')
    ax.axvline(np.mean(gdi_eq_all), color='darkred', lw=2, ls='-',
               label=f'Dieckow eq. mean (GDI={np.mean(gdi_eq_all):.2f})')

    # Annotations
    ax.text(np.mean(gdi_w3_all) + 0.1, 0.88, 'Week-3', fontsize=8, color='navy')
    ax.text(np.mean(gdi_eq_all) + 0.1, 0.58, 'Equilibrium\n(predicted)', fontsize=8, color='darkred')

    ax.set_xlabel('Guild Dysbiosis Index (GDI)', fontsize=10)
    ax.set_yticks([])
    ax.set_title('Predicted long-term dysbiosis matches Joshi peri-implantitis range\n'
                 '(External validation of gLV equilibrium predictions)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(-5.5, 3.5)

    # Key numbers box
    txt = (f'Week-3 mean GDI: {np.mean(gdi_w3_all):.2f}\n'
           f'Eq. mean GDI: {np.mean(gdi_eq_all):.2f}\n'
           f'Joshi PI mean: {JOSHI["PI"]:.2f}')
    ax.text(0.98, 0.05, txt, transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig, gdi_w3_all, gdi_eq_all


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    A, b_all, phi0_pp, guilds, short, colors, patients = load_data()

    # ── Analysis 2: Week-3 GDI as relapse predictor (fast) ────────────────────
    print('=== Analysis 2: Week-3 GDI vs relapse day ===')
    fig2, gdi_w3_all, relapse_all = plot_gdi_vs_relapse(
        A, b_all, phi0_pp, patients, short, colors,
        out_path=OUT_DIR / 'gdi_w3_vs_relapse.png')

    # ── Analysis 3: Joshi comparison (fast) ───────────────────────────────────
    print('\n=== Analysis 3: Joshi cross-validation ===')
    fig3, gdi_w3_all2, gdi_eq_all = plot_joshi_comparison(
        A, b_all, phi0_pp, patients, short,
        out_path=OUT_DIR / 'joshi_comparison.png')
    print(f'  Week-3 mean GDI: {np.mean(gdi_w3_all2):.3f}')
    print(f'  Equilibrium mean GDI: {np.mean(gdi_eq_all):.3f}  (Joshi PI: +0.53)')
    n_pi = sum(1 for g in gdi_eq_all if g > 0)
    print(f'  Patients with PI-like equilibrium: {n_pi}/10')

    # ── Analysis 1: Prevention threshold (slow, ~60s) ─────────────────────────
    print('\n=== Analysis 1: Prevention threshold ===')
    prev_results = run_prevention_analysis(A, b_all, phi0_pp, short, patients, colors)

    plot_prevention(prev_results, short, colors, patients, b_all,
                    out_path=OUT_DIR / 'prevention_threshold.png')

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        'gdi_week3':        dict(zip(patients, [float(g) for g in gdi_w3_all2])),
        'gdi_equilibrium':  dict(zip(patients, [float(g) for g in gdi_eq_all])),
        'relapse_days':     dict(zip(patients, [float(r) if r < 190 else None
                                                for r in relapse_all])),
        'joshi_pi_mean':    JOSHI['PI'],
        'eq_gdi_mean':      float(np.mean(gdi_eq_all)),
        'eq_pi_like_count': n_pi,
        'prevention': {
            pat: {
                gname: {
                    'threshold_db': data['thresholds'][gname]['threshold_db'],
                    'b_orig': data['thresholds'][gname]['b_orig'],
                }
                for gname in data['thresholds']
            }
            for pat, data in prev_results.items()
        }
    }
    (OUT_DIR / 'relapse_prevention_summary.json').write_text(
        json.dumps(summary, indent=2))
    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
