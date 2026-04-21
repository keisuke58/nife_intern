#!/usr/bin/env python3
"""
generate_dieckow_paper_figures.py
Dieckow 独立論文用 publication-quality figures.

Fig 1: Per-patient week1→2→3 predictions (MAP, 10 patients grid)
Fig 2: Shared A matrix heatmap (Dieckow vs 4 Heine)
Fig 3: Cross-prediction RMSE comparison
Fig 4: A matrix correlation matrix (Heine × Dieckow)
Fig 5: Sign pattern comparison (5 conditions)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
from hamilton_ode_jax_nsp import simulate_0d_nsp, theta_to_matrices

RUNS_DIR = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs')
FITS_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
OBS_JSON = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_obs_matrix_5sp.json')
OUT_DIR  = Path('/home/nishioka/IKM_Hiwi/docs/figures/dieckow')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP = 5; N_A = 15; DT = 1e-4; C = 25.0; ALPHA = 100.0; N_STEPS = 2500
SPECIES_COLORS = ['#2166ac', '#4dac26', '#f1a340', '#7b3294', '#d7191c']
HEINE_CONDS = {'CS': 'commensal_static', 'CH': 'commensal_hobic',
               'DS': 'dysbiotic_static', 'DH': 'dh_baseline'}

RC = {
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif',
}
plt.rcParams.update(RC)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_obs():
    raw = json.load(open(OBS_JSON))
    phi_list, valid = [], []
    for p in PATIENTS:
        obs_p = np.array(raw['obs'][p])
        phi_w = obs_p.T
        if np.any(np.isnan(phi_w)) or np.any(phi_w.sum(axis=1) < 0.01):
            continue
        phi_w = np.clip(phi_w, 1e-6, 1.0)
        phi_w /= phi_w.sum(axis=1, keepdims=True)
        phi_list.append(phi_w)
        valid.append(p)
    return np.array(phi_list), valid

def load_theta_map(run_name):
    d = json.load(open(RUNS_DIR / run_name / 'theta_MAP.json'))
    if isinstance(d, list): return np.array(d)
    for k in ['theta_MAP', 'theta_map', 'MAP']:
        if k in d: return np.array(d[k])
    return np.array(list(d.values())[0])

def build_A(theta_15):
    A = np.zeros((N_SP, N_SP))
    idx = 0
    for j in range(N_SP):
        for i in range(j+1):
            A[i,j] = A[j,i] = theta_15[idx]; idx += 1
    return A

def run_week(theta_20, phi_start):
    traj = simulate_0d_nsp(jnp.array(theta_20), n_sp=N_SP, n_steps=N_STEPS,
                           dt=DT, phi_init=jnp.array(phi_start),
                           c_const=C, alpha_const=ALPHA)
    raw = traj[-1]
    return np.array(raw / jnp.maximum(raw.sum(), 1e-12))

run_week_jit = jax.jit(lambda t, p: simulate_0d_nsp(
    t, n_sp=N_SP, n_steps=N_STEPS, dt=DT, phi_init=p,
    c_const=C, alpha_const=ALPHA)[-1])


# ── Fig 1: per-patient week1→2→3 predictions ─────────────────────────────────
def fig1_patient_predictions(phi_obs, valid_patients, theta_map):
    N = len(valid_patients)
    ncols = 5; nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.8),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    weeks = [1, 2, 3]

    # JIT warm-up
    _ = run_week_jit(jnp.array(theta_map[0]), jnp.array(phi_obs[0, 0]))

    for pi, p in enumerate(valid_patients):
        pidx = PATIENTS.index(p)
        theta_p = theta_map[pidx]
        phi_ic = phi_obs[pi, 0]

        phi_w2_j = run_week_jit(jnp.array(theta_p), jnp.array(phi_ic))
        phi_w2 = np.array(phi_w2_j / jnp.maximum(phi_w2_j.sum(), 1e-12))
        phi_w3_j = run_week_jit(jnp.array(theta_p), phi_w2_j)
        phi_w3 = np.array(phi_w3_j / jnp.maximum(phi_w3_j.sum(), 1e-12))
        pred = np.stack([phi_obs[pi, 0], phi_w2, phi_w3])

        ax = axes[pi]
        x = np.arange(N_SP)
        width = 0.3
        for w, (obs_w, pred_w) in enumerate(zip(phi_obs[pi], pred)):
            offset = (w - 1) * width
            ax.bar(x + offset, obs_w, width=width, color=SPECIES_COLORS,
                   alpha=0.35, label='obs' if w == 0 else None)
            ax.plot(x + offset, pred_w, 'x', ms=6, color='k',
                    mew=1.5, label='MAP' if (pi == 0 and w == 0) else None)

        ax.set_title(f'Patient {p}', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT, fontsize=7)
        ax.set_ylim(0, 1)
        if pi % ncols == 0:
            ax.set_ylabel('φ fraction', fontsize=8)

        rmse_p = float(np.sqrt(np.mean(
            (np.stack([phi_w2, phi_w3]) - phi_obs[pi, 1:]) ** 2)))
        ax.text(0.97, 0.97, f'RMSE={rmse_p:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='gray')

    for ax in axes[N:]: ax.set_visible(False)

    # week indicator: shade week columns
    from matplotlib.patches import Patch
    week_labels = [Patch(color='gray', alpha=0.3, label=f'Wk{w+1} bar offset')
                   for w in range(3)]
    sp_patches  = [Patch(color=c, alpha=0.6, label=s)
                   for c, s in zip(SPECIES_COLORS, SHORT)]
    fig.legend(handles=sp_patches, loc='lower center', ncol=N_SP,
               bbox_to_anchor=(0.5, 0), frameon=False, fontsize=8)
    fig.text(0.5, 0.01, 'Week1 (left bar) → Week2 (mid) → Week3 (right bar); × = MAP prediction',
             ha='center', fontsize=7, color='gray')
    fig.suptitle('Per-patient MAP predictions: Dieckow 10-patient in-vivo\n'
                 '(shared A, patient-specific b; TMCMC N_p=1000)',
                 fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / 'fig1_patient_predictions.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Fig 2: A matrix heatmaps (Dieckow + 4 Heine) ─────────────────────────────
def fig2_A_heatmaps(theta_dieckow_flat, theta_dieckow_sign):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))

    def plot_heatmap(ax, A, title, vmax=None):
        vm = vmax or max(abs(A).max(), 0.5)
        im = ax.imshow(A, cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='equal')
        ax.set_xticks(range(N_SP)); ax.set_xticklabels(SHORT, fontsize=8)
        ax.set_yticks(range(N_SP)); ax.set_yticklabels(SHORT, fontsize=8)
        for i in range(N_SP):
            for j in range(N_SP):
                color = 'white' if abs(A[i,j]) > 0.6*vm else 'black'
                ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                        fontsize=7.5, color=color)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return im

    A_dieckow_flat = build_A(theta_dieckow_flat[:N_A])
    A_dieckow_sign = build_A(theta_dieckow_sign[:N_A])
    vm_global = max(abs(A_dieckow_flat).max(),
                    *[abs(build_A(load_theta_map(v)[:N_A])).max()
                      for v in HEINE_CONDS.values()])

    # Row 0: Dieckow flat, sign, diff
    im = plot_heatmap(axes[0,0], A_dieckow_flat,
                      'Dieckow in-vivo\n(flat prior)', vm_global)
    plot_heatmap(axes[0,1], A_dieckow_sign,
                 'Dieckow in-vivo\n(Bergey sign prior)', vm_global)
    diff = A_dieckow_sign - A_dieckow_flat
    vd = max(abs(diff).max(), 0.1)
    axes[0,2].imshow(diff, cmap='RdBu_r', vmin=-vd, vmax=vd, aspect='equal')
    axes[0,2].set_xticks(range(N_SP)); axes[0,2].set_xticklabels(SHORT, fontsize=8)
    axes[0,2].set_yticks(range(N_SP)); axes[0,2].set_yticklabels(SHORT, fontsize=8)
    for i in range(N_SP):
        for j in range(N_SP):
            c = 'white' if abs(diff[i,j]) > 0.6*vd else 'black'
            axes[0,2].text(j, i, f'{diff[i,j]:.2f}', ha='center', va='center',
                           fontsize=7.5, color=c)
    axes[0,2].set_title('Δ A (sign − flat)', fontsize=10, fontweight='bold')

    # Row 1: CS, CH, DS; DH inlined as text note (or show 3 + label 4th)
    heine_items = list(HEINE_CONDS.items())
    for k, (label, run) in enumerate(heine_items[:3]):
        A_h = build_A(load_theta_map(run)[:N_A])
        plot_heatmap(axes[1, k], A_h, f'Heine {label} (in-vitro)', vm_global)
    # Hide 4th slot — show DH values as text annotation on DS panel
    if len(heine_items) > 3:
        label4, run4 = heine_items[3]
        A_h4 = build_A(load_theta_map(run4)[:N_A])
        # Draw DH as small panel overlaid on bottom-right corner
        ax_dh = fig.add_axes([0.685, 0.04, 0.28, 0.35])
        plot_heatmap(ax_dh, A_h4, f'Heine {label4} (in-vitro)', vm_global)

    plt.colorbar(im, ax=axes[:, :3], orientation='vertical', shrink=0.55,
                 label='Interaction coefficient $A_{ij}$', pad=0.02)
    fig.suptitle('Interaction matrices A: Dieckow in-vivo vs Heine in-vitro\n'
                 'Red = cooperative ($A_{ij}>0$); Blue = competitive ($A_{ij}<0$)',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    out = OUT_DIR / 'fig2_A_heatmaps.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Fig 3 + 4: cross-prediction (loaded from summary if available) ────────────
def fig3_cross_prediction():
    cp_json = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cross_prediction/summary.json')
    if not cp_json.exists():
        print('  cross_prediction/summary.json not ready yet — skipping Fig3/4')
        return

    d = json.load(open(cp_json))
    results = d['results']
    corr_labels = d['A_corr_labels']
    corr_mat = np.array(d['A_corr_matrix'])

    # Fig 3: RMSE bar
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    keys   = list(results.keys())
    values = list(results.values())
    palette = []
    for k in keys:
        if 'self' in k: palette.append('#2166ac')
        elif 'fixed' in k: palette.append('#f4a582')
        else: palette.append('#d6604d')

    ax = axes[0]
    bars = ax.barh(range(len(keys)), values, color=palette, alpha=0.85, height=0.65)
    rmse_self = results.get('Dieckow self (TMCMC Np=1000)', None)
    if rmse_self:
        ax.axvline(rmse_self, color='#2166ac', lw=1.5, ls='--', alpha=0.8,
                   label='Dieckow self-fit')
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlabel('RMSE (weeks 2+3)', fontsize=9)
    ax.set_title('Cross-prediction: Heine A on Dieckow in-vivo', fontsize=10)
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis='x', alpha=0.3)

    # Fig 4: A-matrix correlation
    ax2 = axes[1]
    im = ax2.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
    ax2.set_xticks(range(len(corr_labels)))
    ax2.set_xticklabels(corr_labels, fontsize=9)
    ax2.set_yticks(range(len(corr_labels)))
    ax2.set_yticklabels(corr_labels, fontsize=9)
    ax2.set_title('A matrix correlation (Heine × Dieckow)', fontsize=10)
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            c = 'white' if abs(corr_mat[i,j]) > 0.7 else 'black'
            ax2.text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                     fontsize=9, color=c)
    plt.colorbar(im, ax=ax2, label='Pearson r', shrink=0.8)

    plt.tight_layout()
    out = OUT_DIR / 'fig3_cross_prediction.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Fig 4: Metabolic network (bipartite: species ↔ metabolites) ──────────────
def fig4_metabolic_network():
    """
    Bipartite graph showing the metabolite-mediated interactions among 5 species.
    Based on Dieckow Supplementary File 1.
    Left: species (colored circles); Right: metabolites (gray boxes)
    Edges: PRODUCES (blue), USES/consumes (orange), IS_INHIBITED_BY (red dashed)
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    SPECIES = ['So', 'An', 'Vd', 'Fn', 'Pg']
    METS    = ['Lactic acid', 'Menaquinone', 'H₂O₂', 'O₂', 'L-Arginine', 'Nitrate']
    SP_COL  = SPECIES_COLORS

    # x positions: species at x=0, metabolites at x=1
    sp_y  = {s: 1.0 - i * 0.22 for i, s in enumerate(SPECIES)}
    met_y = {m: 1.0 - i * 0.19 for i, m in enumerate(METS)}

    # (species, metabolite, relation): 'prod'=produces, 'use'=uses/consumes, 'inh'=inhibited_by
    edges = [
        ('So', 'Lactic acid',  'prod'),
        ('An', 'Lactic acid',  'prod'),
        ('Vd', 'Lactic acid',  'use'),
        ('An', 'Menaquinone',  'prod'),
        ('Vd', 'Menaquinone',  'prod'),
        ('Pg', 'Menaquinone',  'use'),
        ('So', 'H₂O₂',        'prod'),
        ('Pg', 'H₂O₂',        'inh'),
        ('So', 'O₂',           'inh'),
        ('An', 'O₂',           'inh'),
        ('Vd', 'O₂',           'inh'),
        ('Fn', 'O₂',           'inh'),
        ('So', 'L-Arginine',   'use'),
        ('An', 'Nitrate',      'use'),
        ('Vd', 'Nitrate',      'use'),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.15)
    ax.axis('off')

    # draw species nodes
    for i, sp in enumerate(SPECIES):
        y = sp_y[sp]
        c = ax.add_patch(plt.Circle((0.0, y), 0.055, color=SP_COL[i], zorder=4))
        ax.text(-0.12, y, sp, ha='right', va='center', fontsize=10, fontweight='bold')

    # draw metabolite nodes
    for m in METS:
        y = met_y[m]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.92, y - 0.035), 0.32, 0.07,
            boxstyle='round,pad=0.01', facecolor='#f0f0f0',
            edgecolor='#888', linewidth=0.8, zorder=4))
        ax.text(1.08, y, m, ha='center', va='center', fontsize=8.5)

    # draw edges
    style_map = {
        'prod': dict(color='#2166ac', lw=1.5, ls='-',  label='Produces'),
        'use':  dict(color='#d97c00', lw=1.5, ls='-',  label='Uses / consumes'),
        'inh':  dict(color='#b2182b', lw=1.2, ls='--', label='Inhibited by'),
    }
    drawn_labels = set()
    for sp, met, rel in edges:
        xs = 0.055
        xe = 0.92
        ys = sp_y[sp]
        ye = met_y[met]
        kw = style_map[rel]
        lbl = kw['label'] if kw['label'] not in drawn_labels else '_'
        drawn_labels.add(kw['label'])
        ax.annotate('', xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle='->', color=kw['color'],
                                   lw=kw['lw'], linestyle=kw['ls']),
                    zorder=2)
        # invisible proxy for legend
    legend_elems = [
        mpatches.Patch(facecolor='#2166ac', label='Produces'),
        mpatches.Patch(facecolor='#d97c00', label='Uses / consumes'),
        mpatches.Patch(facecolor='#b2182b', label='Inhibited by'),
    ]
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color='#2166ac', lw=2, label='Produces'),
        Line2D([0],[0], color='#d97c00', lw=2, label='Uses / consumes'),
        Line2D([0],[0], color='#b2182b', lw=2, ls='--', label='Inhibited by'),
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=8, frameon=True)

    # column headers
    ax.text(0.0,  1.12, 'Species', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.08, 1.12, 'Metabolites', ha='center', fontsize=10, fontweight='bold')

    ax.set_title('Metabolite-mediated interactions among 5 implant biofilm species\n'
                 '(Dieckow Supplementary File 1)', fontsize=10)

    plt.tight_layout()
    out = OUT_DIR / 'fig4_metabolic_network.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Fig 5: Sign pattern comparison (5 conditions) ────────────────────────────
def fig5_sign_comparison(theta_dieckow_flat):
    cond_thetas = {}
    for label, run in HEINE_CONDS.items():
        cond_thetas[label] = load_theta_map(run)[:N_A]
    cond_thetas['Dieckow'] = theta_dieckow_flat[:N_A]

    cond_names = list(cond_thetas.keys())   # CS CH DS DH Dieckow
    pairs = [(i, j, f'{SHORT[i]}–{SHORT[j]}')
             for j in range(N_SP) for i in range(j+1)]

    sign_mat = np.zeros((len(pairs), len(cond_names)))
    for ci, cname in enumerate(cond_names):
        t = cond_thetas[cname]
        idx = 0
        for pi, (i, j, _) in enumerate(pairs):
            sign_mat[pi, ci] = np.sign(t[idx])
            idx += 1

    # Identify conserved pairs
    conserved_pos = np.all(sign_mat > 0, axis=1)
    conserved_neg = np.all(sign_mat < 0, axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = matplotlib.colors.ListedColormap(['#d7191c', '#ffffbf', '#2c7bb6'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(sign_mat, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, fontsize=9, fontweight='bold')
    pair_labels = [lbl for _, _, lbl in pairs]
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=8)

    # mark conserved rows
    for pi in range(len(pairs)):
        if conserved_pos[pi] or conserved_neg[pi]:
            ax.add_patch(FancyBboxPatch((-0.5, pi-0.5), len(cond_names), 1,
                         boxstyle='round,pad=0.05', linewidth=1.5,
                         edgecolor='gold', facecolor='none', zorder=3))

    ax.set_title('Sign pattern of A matrix entries across 5 conditions\n'
                 '(red=negative, blue=positive, gold border=conserved)',
                 fontsize=10)
    ax.set_xlabel('Condition', fontsize=9)
    ax.set_ylabel('Interaction pair', fontsize=9)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#2c7bb6', label='Positive (+)'),
        Patch(facecolor='#d7191c', label='Negative (−)'),
        Patch(facecolor='none', edgecolor='gold', linewidth=2,
              label='Conserved (all 5 cond.)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', frameon=True, fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / 'fig5_sign_comparison.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    phi_obs, valid_patients = load_obs()
    d_flat = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    d_sign = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p_meta.json'))
    theta_flat = np.array(d_flat['theta_map'])
    theta_sign = np.array(d_sign['theta_map'])

    # per-patient theta: A shared + b patient-specific
    b_flat = theta_flat[N_A:].reshape(len(PATIENTS), N_SP)
    theta_per_patient = np.array([
        np.concatenate([theta_flat[:N_A], b_flat[PATIENTS.index(p)]])
        for p in valid_patients
    ])

    print('Generating Fig1: per-patient predictions...')
    fig1_patient_predictions(phi_obs, valid_patients, theta_per_patient)

    print('Generating Fig2: A matrix heatmaps...')
    fig2_A_heatmaps(theta_flat, theta_sign)

    print('Generating Fig3: cross-prediction...')
    fig3_cross_prediction()

    print('Generating Fig4: metabolic network...')
    fig4_metabolic_network()

    print('Generating Fig5: sign comparison...')
    fig5_sign_comparison(theta_flat)

    print(f'\nAll figures → {OUT_DIR}')


if __name__ == '__main__':
    main()
