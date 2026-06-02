#!/usr/bin/env python3
"""
guild_importance_analysis.py — Guild-level importance for nife 10-guild Dieckow model.

Same analyses as species_importance_analysis.py but for the 10-guild gLV
fitted to Dieckow 2024 (10 patients, 3 weekly timepoints, caries recovery).

  1. Interaction importance   : rank |A_ij| across 10 guilds
  2. Guild knockout           : remove guild i → BC divergence, productivity drop,
                                volume growth drop
  3. Volume growth contribution: ∫ φ_i(t) dt per guild

b is patient-specific (b_all, 10 patients × 10 guilds).
Analysis uses b_mean (across patients) for the community-level summary,
plus per-patient spread shown as error bars.

Usage:
    python guild_importance_analysis.py
    python guild_importance_analysis.py --fit best   # use best LOO model (α=0.25)
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
OUT_DIR = HERE / 'results' / 'guild_importance'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_FILES = {
    'base':    HERE / 'results' / 'dieckow_cr' / 'fit_guild.json',
    'best':    HERE / 'results' / 'dieckow_cr' / 'fit_guild_jaxlbfgs_best.json',
    'kegg':    HERE / 'results' / 'dieckow_cr' / 'fit_guild_glv_struct.json',
}
PHI_NPY = HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy'

T_END  = 21.0
T_EVAL = np.linspace(0, T_END, 300)

# Short guild labels (class-level, same order as fit JSON)
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

def load_fit(fit_key='base'):
    path = FIT_FILES[fit_key]
    d    = json.loads(path.read_text())
    A    = np.array(d['A'])
    b_all = np.array(d['b_all'])         # (n_patients, n_guilds)
    b_mean = b_all.mean(axis=0)
    guilds = d['guilds']
    short  = [GUILD_SHORT.get(g, g[:4]) for g in guilds]
    colors = [CMAP(i / len(guilds)) for i in range(len(guilds))]
    return A, b_mean, b_all, guilds, short, colors


def get_phi0_dieckow():
    """Week-0 mean composition across patients from phi_guild.npy."""
    try:
        phi_all = np.load(PHI_NPY)   # (n_patients, n_weeks, n_guilds)
        phi0    = phi_all[:, 0, :].mean(axis=0)
        phi0    = phi0 / (phi0.sum() + 1e-12)
        return phi0
    except Exception:
        return None


def get_phi0_per_patient():
    """Week-0 composition per patient — for per-patient spread."""
    try:
        phi_all = np.load(PHI_NPY)
        phi0    = phi_all[:, 0, :]
        sums    = phi0.sum(axis=1, keepdims=True)
        return np.where(sums > 1e-12, phi0 / sums, phi0)
    except Exception:
        return None


# ── gLV ODE ───────────────────────────────────────────────────────────────

def glv_rhs(t, phi, A, b, mask):
    phi = np.maximum(phi * mask, 0)
    return phi * (b + A @ phi) * mask


def simulate_glv(A, b, phi0, mask=None):
    n = len(b)
    if mask is None:
        mask = np.ones(n)
    phi0m = phi0 * mask
    sol = solve_ivp(glv_rhs, [0, T_END], phi0m,
                    args=(A, b, mask), t_eval=T_EVAL,
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y.T   # (T, n)


def bray_curtis(p, q):
    return np.sum(np.abs(p - q)) / (np.sum(p) + np.sum(q) + 1e-12)


# ── Knockout metrics ───────────────────────────────────────────────────────

def run_knockout(A, b, phi0):
    n         = len(b)
    full_traj = simulate_glv(A, b, phi0)
    knockouts = {}
    for i in range(n):
        mask = np.ones(n); mask[i] = 0.0
        knockouts[i] = simulate_glv(A, b, phi0, mask=mask)
    return full_traj, knockouts


def compute_metrics(full_traj, knockouts, b):
    nf       = len(full_traj)
    t_nf     = T_EVAL[:nf]
    full_prod = np.maximum(full_traj @ b, 0)
    full_vol  = np.trapezoid(full_traj.sum(axis=1), t_nf)
    metrics   = {}
    for i, traj in knockouts.items():
        n        = min(nf, len(traj))
        bc_vals  = np.array([bray_curtis(full_traj[t], traj[t]) for t in range(n)])
        ko_prod  = np.maximum(traj[:n] @ b, 0)
        ko_vol   = np.trapezoid(traj[:n].sum(axis=1), T_EVAL[:n])
        fp_mean  = full_prod[:n].mean()
        metrics[i] = {
            'bc_mean':   float(bc_vals.mean()),
            'bc_max':    float(bc_vals.max()),
            'prod_drop': float((fp_mean - ko_prod.mean()) / (fp_mean + 1e-12)),
            'vol_drop':  float((full_vol - ko_vol) / (full_vol + 1e-12)),
        }
    return metrics


def volume_contributions(full_traj):
    nf = len(full_traj)
    integ = np.array([np.trapezoid(full_traj[:, i], T_EVAL[:nf])
                      for i in range(full_traj.shape[1])])
    return integ / (integ.sum() + 1e-12), integ


# ── Per-patient spread ─────────────────────────────────────────────────────

def per_patient_metrics(A, b_all, phi0_mean, n_guilds):
    """Run knockout for each patient's b and phi0 → metric arrays (n_patients, n_guilds)."""
    phi0_pp = get_phi0_per_patient()
    n_pat   = len(b_all)
    bc_arr  = np.zeros((n_pat, n_guilds))
    vol_arr = np.zeros((n_pat, n_guilds))
    for p, b_p in enumerate(b_all):
        phi0_p = phi0_pp[p] if phi0_pp is not None else phi0_mean
        full_traj, knockouts = run_knockout(A, b_p, phi0_p)
        m = compute_metrics(full_traj, knockouts, b_p)
        for i in range(n_guilds):
            bc_arr[p, i]  = m[i]['bc_mean']
            vol_arr[p, i] = m[i]['vol_drop']
    return bc_arr, vol_arr


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_A_heatmap(A, short, colors, out_path=None):
    n = len(short)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.abs(A), cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel('source j', fontsize=9)
    ax.set_ylabel('target i', fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                    fontsize=5.5,
                    color='white' if abs(A[i,j]) > 0.5 * np.abs(A).max() else 'black')
    fig.colorbar(im, ax=ax, label='|A_ij|', fraction=0.046)
    ax.set_title('10-guild interaction matrix |A_ij|', fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_knockout_bars(metrics, bc_arr, vol_arr, short, colors, out_path=None):
    n   = len(short)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metric_keys   = ['bc_mean', 'prod_drop', 'vol_drop']
    metric_labels = ['BC divergence (mean)', 'Productivity drop', 'Volume growth drop']

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        vals = [metrics[i][key] for i in range(n)]
        bars = ax.bar(range(n), vals, color=colors, alpha=0.85)

        # per-patient error bars for bc and vol
        if key == 'bc_mean':
            err = bc_arr.std(axis=0)
            ax.errorbar(range(n), vals, yerr=err, fmt='none', color='k', capsize=3, lw=1)
        elif key == 'vol_drop':
            err = vol_arr.std(axis=0)
            ax.errorbar(range(n), vals, yerr=err, fmt='none', color='k', capsize=3, lw=1)

        ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.axhline(0, color='k', lw=0.5)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005 * np.sign(val + 1e-9),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6)

    fig.suptitle('10-guild knockout effects (Dieckow gLV)', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def plot_volume_contributions(frac, integ, short, colors, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        np.maximum(frac, 0), labels=short, colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        startangle=90, pctdistance=0.8,
    )
    for t in texts: t.set_fontsize(8)
    ax.set_title('Volume growth contribution\n(∫ φ_i dt / ∫ φ_total dt)', fontsize=9)

    # Bar chart with absolute integrals
    ax = axes[1]
    ax.bar(range(len(short)), integ, color=colors, alpha=0.85)
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, fontsize=8)
    ax.set_ylabel('∫ φ_i(t) dt  (days)', fontsize=9)
    ax.set_title('Absolute volume integral per guild', fontsize=9)

    fig.suptitle('10-guild volume growth contributions (Dieckow gLV)', fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    return fig


def rank_interactions(A, short, top_n=10):
    n = len(short)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((abs(A[i, j]), i, j, A[i, j]))
    pairs.sort(reverse=True)
    return pairs[:top_n]


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', choices=list(FIT_FILES.keys()), default='base')
    args = parser.parse_args()

    A, b_mean, b_all, guilds, short, colors = load_fit(args.fit)
    n = len(guilds)

    phi0_loaded = get_phi0_dieckow()
    phi0 = phi0_loaded if phi0_loaded is not None else np.ones(n) / n

    print(f'Fit: {args.fit}  ({n} guilds)')
    print(f'phi0: {dict(zip(short, phi0.round(3)))}')

    # Main simulation with b_mean
    full_traj, knockouts = run_knockout(A, b_mean, phi0)
    metrics = compute_metrics(full_traj, knockouts, b_mean)
    frac, integ = volume_contributions(full_traj)

    # Per-patient spread
    print('Computing per-patient spread ...')
    bc_arr, vol_arr = per_patient_metrics(A, b_all, phi0, n)

    # Print summary
    print(f'\n{"Guild":<20} {"BC div":>7} {"Prod drop":>10} {"Vol drop":>9} {"Vol frac":>9}')
    for i in range(n):
        m = metrics[i]
        print(f'{guilds[i]:<20} {m["bc_mean"]:7.4f} {m["prod_drop"]:10.4f} '
              f'{m["vol_drop"]:9.4f} {frac[i]:9.1%}')

    print('\nTop 10 interactions:')
    for rank, (mag, i, j, val) in enumerate(rank_interactions(A, short), 1):
        print(f'  {rank:2}. {short[j]} → {short[i]}  A={val:+.3f}')

    # Plots
    plot_A_heatmap(A, short, colors,
                   out_path=OUT_DIR / f'A_heatmap_{args.fit}.png')
    plot_knockout_bars(metrics, bc_arr, vol_arr, short, colors,
                       out_path=OUT_DIR / f'knockout_bars_{args.fit}.png')
    plot_volume_contributions(frac, integ, short, colors,
                              out_path=OUT_DIR / f'volume_contributions_{args.fit}.png')

    # Save JSON
    summary = {
        'fit': args.fit,
        'guilds': guilds,
        'top_interactions': [
            {'source': short[j], 'target': short[i], 'A_ij': float(val)}
            for _, i, j, val in rank_interactions(A, short)
        ],
        'knockout_metrics': {short[i]: metrics[i] for i in range(n)},
        'volume_fractions':  {short[i]: float(frac[i]) for i in range(n)},
        'bc_std_per_patient': {short[i]: float(bc_arr[:, i].std()) for i in range(n)},
    }
    (OUT_DIR / f'guild_importance_{args.fit}.json').write_text(json.dumps(summary, indent=2))
    print(f'\nAll outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
