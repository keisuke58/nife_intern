#!/usr/bin/env python3
"""Visualise the guild A matrix and prediction vs observation."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from guild_replicator_dieckow import (
    predict_trajectory, GUILD_ORDER, GUILD_COLORS, GUILD_SHORT
)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-json', default=str(Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild.json'))
    ap.add_argument('--phi-npy', default=str(Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'))
    ap.add_argument('--out-dir', default=str(Path(__file__).parent / 'results' / 'dieckow_otu'))
    ap.add_argument('--prefix', default='guild_Amatrix')
    ap.add_argument('--linthresh', type=float, default=0.003)
    ap.add_argument('--model', default='replicator', choices=['replicator', 'hamilton'])
    ap.add_argument('--hamilton-n-steps', type=int, default=200)
    ap.add_argument('--hamilton-dt', type=float, default=1e-4)
    args = ap.parse_args()

    with open(args.fit_json) as f:
        fit = json.load(f)
    A     = np.array(fit['A'])
    b_all = np.array(fit['b_all'])
    phi_all = np.load(args.phi_npy)

    guilds = fit.get('guilds', GUILD_ORDER)
    n_g = int(len(guilds))
    short = [GUILD_SHORT.get(g, str(g)[:6]) for g in guilds]
    colors = [GUILD_COLORS.get(g, '#000000') for g in guilds]

    patients_fit = fit.get('patients')
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    vmax = float(np.abs(A).max())
    norm = mcolors.SymLogNorm(linthresh=args.linthresh, linscale=0.5,
                              vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(A, cmap='RdBu_r', norm=norm, aspect='equal')

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_g))
    ax.set_yticklabels(short, fontsize=8)

    for spine in ('left', 'bottom'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    for k, col in enumerate(colors):
        ax.get_xticklabels()[k].set_color(col)
        ax.get_yticklabels()[k].set_color(col)

    ax.set_xlabel('Source guild', fontsize=9)
    ax.set_ylabel('Target guild', fontsize=9)
    ax.set_title(f'Guild interaction matrix A\n(RMSE={fit["rmse"]:.4f}, SymLog scale)', fontsize=10)

    cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label('$A_{ij}$', fontsize=8)

    threshold_annot = args.linthresh
    for i in range(n_g):
        for j in range(n_g):
            v = A[i, j]
            if abs(v) >= threshold_annot:
                bg = abs(norm(v) - 0.5) / 0.5
                tc = 'white' if bg > 0.6 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=5.5, color=tc)

    ax = axes[1]
    all_obs, all_pred, all_colors = [], [], []
    if args.model == 'replicator':
        for idx, _patient in enumerate(patients):
            p2, p3 = predict_trajectory(phi_obs[idx, 0], b_all[idx], A)
            if present[idx, 1]:
                for g in range(n_g):
                    all_obs.append(phi_obs[idx, 1, g])
                    all_pred.append(p2[g])
                    all_colors.append(colors[g])
            if present[idx, 2]:
                for g in range(n_g):
                    all_obs.append(phi_obs[idx, 2, g])
                    all_pred.append(p3[g])
                    all_colors.append(colors[g])
    else:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / 'Tmcmc202601' / 'data_5species' / 'main'))
        import jax.numpy as jnp
        from hamilton_ode_jax_nsp import simulate_0d_nsp

        A_upper = fit.get('A_upper')
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

        for idx, _patient in enumerate(patients):
            p2 = eq_one(phi_obs[idx, 0], b_all[idx])
            p3 = eq_one(p2, b_all[idx])
            if present[idx, 1]:
                for g in range(n_g):
                    all_obs.append(phi_obs[idx, 1, g])
                    all_pred.append(p2[g])
                    all_colors.append(colors[g])
            if present[idx, 2]:
                for g in range(n_g):
                    all_obs.append(phi_obs[idx, 2, g])
                    all_pred.append(p3[g])
                    all_colors.append(colors[g])

    all_obs  = np.array(all_obs)
    all_pred = np.array(all_pred)

    ax.scatter(all_obs, all_pred, s=12, alpha=0.55, c=all_colors, linewidths=0)
    lim = max(float(all_obs.max()), float(all_pred.max())) * 1.05 if all_obs.size else 1.0
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.7)
    ax.set_xlabel('Observed φ')
    ax.set_ylabel('Predicted φ')
    ax.set_title('Observed vs predicted\n(W2 + W3)', fontsize=10)
    corr = np.corrcoef(all_obs, all_pred)[0, 1] if all_obs.size else float('nan')
    rmse_scatter = float(np.sqrt(np.mean((all_obs - all_pred)**2))) if all_obs.size else float('nan')
    ax.text(0.05, 0.93, f'r = {corr:.3f}\nRMSE = {rmse_scatter:.4f}',
            transform=ax.transAxes, fontsize=9, va='top')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=6, label=g[:8])
               for c, g in zip(colors, guilds)]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc='lower right',
              frameon=False, handletextpad=0.3, labelspacing=0.3)

    fig.suptitle('gLV replicator model — Dieckow 2024', fontsize=11)
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = out_dir / f'{args.prefix}.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=300)
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
