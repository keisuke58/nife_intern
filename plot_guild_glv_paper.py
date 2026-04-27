#!/usr/bin/env python3
"""
Paper-quality figure: class-level gLV fit (Dieckow data).
Panels:
  a  — A-matrix heatmap (N×N guild interactions)
  b  — predicted vs observed scatter (W2 + W3, all patients)
  c  — per-patient stacked bars (obs W1 | obs W2 / pred W2 | obs W3 / pred W3)
"""

import json, sys, argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from guild_replicator_dieckow import (
    GUILD_ORDER,
    GUILD_SHORT,
    GUILD_COLORS,
)

# ── paper rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.titlesize':    9,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'axes.linewidth':    0.7,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ── prediction helpers ────────────────────────────────────────────────────────
def replicator_rhs(t, phi, b, A):
    f    = b + A @ phi
    fbar = phi @ f
    return phi * (f - fbar)


def integrate_step(phi0, b, A, dt=1.0):
    sol  = solve_ivp(replicator_rhs, [0, dt], phi0, args=(b, A),
                     method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s    = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


def predict_all(A, b_all, phi_obs):
    """Return pred_W2 and pred_W3 arrays (n_p, N_G)."""
    n_p, _, n_g = phi_obs.shape
    p_W2 = np.zeros((n_p, n_g))
    p_W3 = np.zeros((n_p, n_g))
    for i in range(n_p):
        p_W2[i] = integrate_step(phi_obs[i, 0], b_all[i], A)
        p_W3[i] = integrate_step(p_W2[i],       b_all[i], A)
    return p_W2, p_W3


# ── figure ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-json', default='/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/fit_guild.json')
    ap.add_argument('--phi-npy', default='/home/nishioka/IKM_Hiwi/nife/results/dieckow_otu/phi_guild.npy')
    ap.add_argument('--out-dir', default='/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr')
    ap.add_argument('--prefix', default='guild_glv_paper')
    ap.add_argument('--model', default='replicator', choices=['replicator', 'hamilton'])
    ap.add_argument('--hamilton-n-steps', type=int, default=200)
    ap.add_argument('--hamilton-dt', type=float, default=1e-4)
    args = ap.parse_args()

    d       = json.load(open(args.fit_json))
    A       = np.array(d['A'])
    b_all   = np.array(d['b_all'])
    phi_all = np.load(args.phi_npy)
    rmse    = d['rmse']

    guilds = d.get('guilds', GUILD_ORDER)
    n_g = int(len(guilds))
    short = [GUILD_SHORT.get(g, str(g)[:6]) for g in guilds]
    colors = [GUILD_COLORS.get(g, '#000000') for g in guilds]

    patients_fit = d.get('patients')
    if isinstance(patients_fit, list) and len(patients_fit) > 0 and phi_all.shape[0] == 10:
        order = list('ABCDEFGHKL')
        idx_map = {p: i for i, p in enumerate(order)}
        rows = [idx_map.get(p) for p in patients_fit]
        if all(r is not None for r in rows):
            phi_obs = phi_all[rows, :, :n_g]
            patients = patients_fit
        else:
            phi_obs = phi_all[:, :, :n_g]
            patients = list(patients_fit)
    else:
        phi_obs = phi_all[:, :, :n_g]
        patients = list(patients_fit) if isinstance(patients_fit, list) else [str(i) for i in range(phi_obs.shape[0])]

    present = phi_obs.sum(axis=2) > 1e-9
    keep = present[:, 0]
    phi_obs = phi_obs[keep]
    present = present[keep]
    patients = [p for k, p in zip(keep.tolist(), patients) if k]

    if args.model == 'replicator':
        pred_W2, pred_W3 = predict_all(A, b_all, phi_obs)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / 'Tmcmc202601' / 'data_5species' / 'main'))
        import jax.numpy as jnp
        from hamilton_ode_jax_nsp import simulate_0d_nsp

        A_upper = d.get('A_upper')
        if A_upper is None:
            upper = []
            for j in range(n_g):
                for i in range(j + 1):
                    upper.append(A[i, j])
            A_upper = upper
        A_upper = jnp.array(np.array(A_upper, dtype=float))

        def eq_one(phi0, b):
            s0 = float(np.sum(phi0))
            phi0n = phi0 / s0 if s0 > 1e-12 else np.ones(n_g) / n_g
            theta = jnp.concatenate([A_upper, jnp.array(b, dtype=float)])
            tr = simulate_0d_nsp(
                theta, n_sp=n_g, n_steps=args.hamilton_n_steps,
                dt=args.hamilton_dt, phi_init=jnp.array(phi0n, dtype=float),
                c_const=25.0, alpha_const=100.0
            )
            eq = np.array(tr[-1])
            s = float(eq.sum())
            return eq / s if s > 1e-12 else np.ones(n_g) / n_g

        n_p = phi_obs.shape[0]
        pred_W2 = np.zeros((n_p, n_g))
        pred_W3 = np.zeros((n_p, n_g))
        for i in range(n_p):
            pred_W2[i] = eq_one(phi_obs[i, 0], b_all[i])
            pred_W3[i] = eq_one(pred_W2[i], b_all[i])

    fig = plt.figure(figsize=(7.2, 8.0))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1.15, 0.85],
        height_ratios=[1.0, 0.85],
        hspace=0.52, wspace=0.38,
    )

    # ── (a) A-matrix heatmap ──────────────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])
    vmax = np.abs(A).max()
    im   = ax_A.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')

    ax_A.set_xticks(range(n_g))
    ax_A.set_xticklabels(short, rotation=45, ha='right', fontsize=7.5)
    ax_A.set_yticks(range(n_g))
    ax_A.set_yticklabels(short, fontsize=7.5)
    ax_A.tick_params(length=0)
    ax_A.set_xlabel('Source guild', labelpad=4)
    ax_A.set_ylabel('Target guild', labelpad=4)

    thresh = 0.55 * vmax
    for i in range(n_g):
        for j in range(n_g):
            c = 'white' if abs(A[i, j]) > thresh else '#222222'
            ax_A.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                      fontsize=6.0, color=c)

    cbar = plt.colorbar(im, ax=ax_A, fraction=0.040, pad=0.03, shrink=0.85)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Interaction strength $A_{ij}$', fontsize=8)

    ax_A.text(-0.14, 1.04, 'a', transform=ax_A.transAxes,
              fontsize=11, fontweight='bold', va='top')
    ax_A.set_title('Guild interaction matrix', pad=6)

    # ── (b) predicted vs observed scatter ─────────────────────────────────────
    ax_sc = fig.add_subplot(gs[0, 1])

    for g in range(n_g):
        obs_parts = []
        pred_parts = []
        if present[:, 1].any():
            obs_parts.append(phi_obs[present[:, 1], 1, g])
            pred_parts.append(pred_W2[present[:, 1], g])
        if present[:, 2].any():
            obs_parts.append(phi_obs[present[:, 2], 2, g])
            pred_parts.append(pred_W3[present[:, 2], g])
        if not obs_parts:
            continue
        obs = np.concatenate(obs_parts)
        pred = np.concatenate(pred_parts)
        ax_sc.scatter(obs, pred, color=colors[g], s=24, alpha=0.85,
                      linewidths=0.4, edgecolors='white', zorder=3)

    lim = 1.02
    ax_sc.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.6)
    ax_sc.set_xlim(-0.01, lim); ax_sc.set_ylim(-0.01, lim)
    ax_sc.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_sc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_sc.set_xlabel('Observed relative abundance')
    ax_sc.set_ylabel('Predicted relative abundance')
    ax_sc.set_title('Predicted vs. observed', pad=6)
    ax_sc.set_aspect('equal')

    # compute Pearson r
    obs_all_parts = []
    pred_all_parts = []
    if present[:, 1].any():
        obs_all_parts.append(phi_obs[present[:, 1], 1, :].ravel())
        pred_all_parts.append(pred_W2[present[:, 1], :].ravel())
    if present[:, 2].any():
        obs_all_parts.append(phi_obs[present[:, 2], 2, :].ravel())
        pred_all_parts.append(pred_W3[present[:, 2], :].ravel())
    obs_all = np.concatenate(obs_all_parts) if obs_all_parts else np.array([])
    pred_all = np.concatenate(pred_all_parts) if pred_all_parts else np.array([])
    r_val    = np.corrcoef(obs_all, pred_all)[0, 1]
    ax_sc.text(0.04, 0.94,
               f'RMSE = {rmse:.4f}\n$r$ = {r_val:.3f}',
               transform=ax_sc.transAxes, fontsize=7.5,
               va='top', linespacing=1.5,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', lw=0.6))

    legend_patches = [Patch(color=colors[g], label=short[g]) for g in range(n_g)]
    ax_sc.legend(handles=legend_patches, ncol=2, fontsize=7,
                 frameon=False, loc='lower right',
                 handlelength=1.0, handletextpad=0.4, labelspacing=0.25)

    ax_sc.text(-0.18, 1.04, 'b', transform=ax_sc.transAxes,
               fontsize=11, fontweight='bold', va='top')

    # ── (c) per-patient stacked bars ──────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])

    n_p   = len(patients)
    group = 5          # bars per patient: W1 | W2obs | W2pred | W3obs | W3pred
    gap   = 0.4        # gap between patients
    bw    = 0.55       # bar width
    inner = 0.05       # small gap within pairs

    x_ticks = []
    x_labels = []

    for p_idx, pat in enumerate(patients):
        x0 = p_idx * (group * bw + gap)

        # positions: W1, [W2obs, W2pred], [W3obs, W3pred]
        xs = [
            x0,
            x0 + 1.2 * bw,
            x0 + 1.2 * bw + bw + inner,
            x0 + 2.6 * bw + 2 * inner + 0.2,
            x0 + 2.6 * bw + 3 * inner + 0.2 + bw,
        ]
        bars_data = [
            phi_obs[p_idx, 0],   # W1 obs
            phi_obs[p_idx, 1] if present[p_idx, 1] else np.zeros(n_g),   # W2 obs
            pred_W2[p_idx],      # W2 pred
            phi_obs[p_idx, 2] if present[p_idx, 2] else np.zeros(n_g),   # W3 obs
            pred_W3[p_idx],      # W3 pred
        ]
        hatch_list = ['', '', '///', '', '///']

        for xi, data, hatch in zip(xs, bars_data, hatch_list):
            bottom = 0.0
            for g in range(n_g):
                ax_bar.bar(xi, data[g], bottom=bottom, width=bw,
                           color=colors[g], hatch=hatch,
                           linewidth=0.3, edgecolor='white')
                bottom += data[g]

        # patient label centred under the 5 bars
        x_center = (xs[0] + xs[-1]) / 2
        x_ticks.append(x_center)
        x_labels.append(f'P{pat}')

        # week labels below bars
        for xi, lbl in zip(xs, ['W1', 'W2\nobs', 'W2\npred', 'W3\nobs', 'W3\npred']):
            ax_bar.text(xi, -0.07, lbl, ha='center', va='top',
                        fontsize=6.5, transform=ax_bar.get_xaxis_transform(),
                        color='#444444')

    ax_bar.set_xlim(-0.5, xs[-1] + bw)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_xticks(x_ticks)
    ax_bar.set_xticklabels(x_labels, fontsize=8)
    ax_bar.set_ylabel('Relative abundance')
    ax_bar.set_title('Per-patient community composition: observed vs. predicted', pad=6)
    ax_bar.tick_params(axis='x', length=0)
    ax_bar.spines['bottom'].set_visible(False)

    # legend: guild colours + hatching
    leg_patches = [Patch(color=colors[g], label=short[g]) for g in range(n_g)]
    leg_extra   = [
        Patch(facecolor='#cccccc', label='Observed', linewidth=0.5, edgecolor='white'),
        Patch(facecolor='#cccccc', hatch='///', label='Predicted', linewidth=0.5, edgecolor='white'),
    ]
    ax_bar.legend(handles=leg_patches + leg_extra, ncol=6, fontsize=7,
                  frameon=False, loc='upper right',
                  handlelength=1.1, handletextpad=0.4, labelspacing=0.3,
                  bbox_to_anchor=(1.0, 1.12))

    ax_bar.text(-0.05, 1.04, 'c', transform=ax_bar.transAxes,
                fontsize=11, fontweight='bold', va='top')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f'{args.prefix}.pdf'
    out_png = out_dir / f'{args.prefix}.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
