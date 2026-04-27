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
from guild_replicator_dieckow import GUILD_COLORS as DIECKOW_GUILD_COLORS, GUILD_SHORT

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HAS_JAX = False
try:
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
    from hamilton_ode_jax_nsp import simulate_0d_nsp, theta_to_matrices

    HAS_JAX = True
except Exception:
    HAS_JAX = False

RUNS_DIR = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs')
FITS_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
OBS_JSON = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_obs_matrix_5sp.json')
GUILD_FIT_JSON = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/fit_guild.json')
OUT_DIR  = Path('/home/nishioka/IKM_Hiwi/docs/figures/dieckow')
CR_DIR   = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CR_DIR.mkdir(parents=True, exist_ok=True)

SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP = 5; N_A = 15; DT = 1e-4; C = 25.0; ALPHA = 100.0; N_STEPS = 2500
SPECIES_COLORS = ['#2166ac', '#4dac26', '#f1a340', '#7b3294', '#d7191c']
HEINE_CONDS = {'CS': 'commensal_static', 'CH': 'commensal_hobic',
               'DS': 'dysbiotic_static', 'DH': 'dh_baseline'}

RC = {
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '0.8',
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
    'lines.linewidth':   1.5,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
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

def theta15_to_pairs(theta_15):
    pairs = []
    idx = 0
    for j in range(N_SP):
        for i in range(j + 1):
            pairs.append((i, j, float(theta_15[idx])))
            idx += 1
    return pairs

def build_A_samples(theta_samples):
    theta_samples = np.asarray(theta_samples)
    A_s = np.zeros((theta_samples.shape[0], N_SP, N_SP), dtype=float)
    for k in range(theta_samples.shape[0]):
        A_s[k] = build_A(theta_samples[k, :N_A])
    return A_s

def plot_signed_network(ax, node_labels, A_mean, sign_certainty, node_colors, title):
    n = len(node_labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[np.cos(ang), np.sin(ang)]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    vmax = max(1e-12, float(np.max(np.abs(A_mean))))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(A_mean[i, j])
            if abs(w) < 1e-6:
                continue
            lw = 0.8 + 5.0 * (abs(w) / vmax)
            a = float(np.clip(sign_certainty[i, j], 0.05, 1.0))
            col = '#d7191c' if w > 0 else '#2166ac'
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    color=col, lw=lw, alpha=a, solid_capstyle='round', zorder=1)

    for i, lbl in enumerate(node_labels):
        ax.scatter([xy[i, 0]], [xy[i, 1]], s=420, c=[node_colors[i]],
                   edgecolors='white', linewidths=1.2, zorder=3)
        col = str(node_colors[i]).lstrip('#')
        if len(col) == 6:
            r = int(col[0:2], 16)
            g = int(col[2:4], 16)
            b = int(col[4:6], 16)
            lum = 0.2126 * (r / 255.0) + 0.7152 * (g / 255.0) + 0.0722 * (b / 255.0)
            txt_col = 'black' if lum > 0.62 else 'white'
        else:
            txt_col = 'white'
        ax.text(xy[i, 0], xy[i, 1], lbl, ha='center', va='center',
                fontsize=12, fontweight='bold', color=txt_col, zorder=4)

def run_week(theta_20, phi_start):
    if not HAS_JAX:
        raise RuntimeError('JAX is required for ODE prediction (run_week).')
    traj = simulate_0d_nsp(jnp.array(theta_20), n_sp=N_SP, n_steps=N_STEPS,
                           dt=DT, phi_init=jnp.array(phi_start),
                           c_const=C, alpha_const=ALPHA)
    raw = traj[-1]
    return np.array(raw / jnp.maximum(raw.sum(), 1e-12))

run_week_jit = None
if HAS_JAX:
    run_week_jit = jax.jit(lambda t, p: simulate_0d_nsp(
        t, n_sp=N_SP, n_steps=N_STEPS, dt=DT, phi_init=p,
        c_const=C, alpha_const=ALPHA)[-1])


# ── Fig 1: per-patient week1→2→3 predictions ─────────────────────────────────
def fig1_patient_predictions(phi_obs, valid_patients, theta_map):
    if not HAS_JAX:
        raise RuntimeError('JAX is required for Fig1 patient predictions.')
    N = len(valid_patients)
    ncols = 5; nrows = (N + ncols - 1) // ncols

    # ── Compute predictions ───────────────────────────────────────────────────
    _ = run_week_jit(jnp.array(theta_map[0]), jnp.array(phi_obs[0, 0]))  # warm-up

    phi_pred = []  # list of (3, N_SP) arrays: [W1-obs, W2-pred, W3-pred]
    rmse_list = []
    for pi in range(N):
        theta_p = theta_map[pi]
        phi_ic  = jnp.array(phi_obs[pi, 0])
        raw2 = run_week_jit(jnp.array(theta_p), phi_ic)
        phi_w2 = np.array(raw2 / jnp.maximum(raw2.sum(), 1e-12))
        raw3 = run_week_jit(jnp.array(theta_p), raw2)
        phi_w3 = np.array(raw3 / jnp.maximum(raw3.sum(), 1e-12))
        phi_pred.append(np.stack([phi_obs[pi, 0], phi_w2, phi_w3]))
        rmse_list.append(float(np.sqrt(np.mean(
            (np.stack([phi_w2, phi_w3]) - phi_obs[pi, 1:]) ** 2))))

    # ── Layout: 2×5 grid, each panel = stacked-bar composition timeline ───────
    # 5 bars per panel: W1-obs | W2-obs | W2-pred | W3-obs | W3-pred
    # Species stacked with consistent colors; clear obs/pred pairing
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, nrows * 3.2),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.15})
    axes = axes.flatten()

    bw = 0.72          # bar width
    # x-positions: W1 | gap | W2-obs, W2-pred | gap | W3-obs, W3-pred
    x_w1   = 0.0
    x_w2_o = 1.6;  x_w2_p = x_w2_o + bw + 0.06
    x_w3_o = 3.4;  x_w3_p = x_w3_o + bw + 0.06
    xs_obs  = [x_w1,   x_w2_o, x_w3_o]
    xs_pred = [x_w2_p, x_w3_p]

    HATCHES = ['////', '////']  # predicted bars use diagonal hatch

    for pi, p in enumerate(valid_patients):
        ax   = axes[pi]
        obs  = phi_obs[pi]       # (3, N_SP)  W1, W2, W3 observed
        pred = phi_pred[pi]      # (3, N_SP)  W1, W2-pred, W3-pred

        # draw stacked obs bars (W1, W2, W3)
        for wi, x0 in enumerate(xs_obs):
            bottom = 0.0
            for si in range(N_SP):
                ax.bar(x0, obs[wi, si], bw, bottom=bottom,
                       color=SPECIES_COLORS[si],
                       alpha=0.88, linewidth=0)
                bottom += obs[wi, si]

        # draw stacked pred bars (W2-pred, W3-pred) — hatched, lighter
        for wi, x0 in enumerate(xs_pred):
            bottom = 0.0
            for si in range(N_SP):
                h = pred[wi + 1, si]   # W2-pred (wi=0) or W3-pred (wi=1)
                ax.bar(x0, h, bw, bottom=bottom,
                       color=SPECIES_COLORS[si],
                       alpha=0.42, linewidth=0.6,
                       edgecolor=SPECIES_COLORS[si],
                       hatch='////')
                bottom += h

        # x-axis ticks and labels
        tick_x = [x_w1, (x_w2_o + x_w2_p) / 2, (x_w3_o + x_w3_p) / 2]
        ax.set_xticks(tick_x)
        ax.set_xticklabels(['W1', 'W2', 'W3'], fontsize=9)
        ax.set_xlim(-0.55, x_w3_p + 0.55)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        if pi % ncols == 0:
            ax.set_ylabel('Relative abundance', fontsize=9)
        else:
            ax.set_yticklabels([])

        # thin vertical dividers between week pairs
        for xd in [1.25, 3.05]:
            ax.axvline(xd, color='#cccccc', lw=0.7, ls='--', zorder=0)

        # RMSE annotation
        ax.text(0.97, 0.97, f'RMSE={rmse_list[pi]:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

        ax.set_title(f'Patient {p}', fontsize=10, fontweight='bold', pad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[N:]: ax.set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    sp_patches = [Patch(color=c, alpha=0.88, label=s)
                  for c, s in zip(SPECIES_COLORS, SHORT)]
    obs_patch  = Patch(color='#888888', alpha=0.88,   label='Observed')
    pred_patch = Patch(color='#888888', alpha=0.42,   label='Predicted (MAP)',
                       hatch='////')
    fig.legend(handles=sp_patches + [obs_patch, pred_patch],
               loc='lower center', ncol=N_SP + 2,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9,
               handlelength=1.4)
    fig.text(0.5, -0.025,
             'Solid = observed, hatched = MAP prediction.  '
             'W1 = initial condition (not predicted).',
             ha='center', fontsize=7.5, color='#666666')

    fig.suptitle('Per-patient MAP predictions — Dieckow 10-patient in-vivo\n'
                 '(Hamilton ODE, shared A, patient-specific b; TMCMC N_p=1 000)',
                 fontsize=12, fontweight='bold', y=1.01)

    for d in (OUT_DIR, CR_DIR):
        out = d / 'fig1b_patient_predictions_stacked.pdf'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        fig.savefig(str(out).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig1b_patient_predictions_stacked (docs + results/dieckow_cr)')


# ── Fig 2: A matrix heatmaps (Dieckow + 4 Heine) ─────────────────────────────
def fig2_A_heatmaps(theta_dieckow_flat, theta_dieckow_sign):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    A_dieckow_flat = build_A(theta_dieckow_flat[:N_A])
    A_dieckow_sign = build_A(theta_dieckow_sign[:N_A])
    heine_As = {lbl: build_A(load_theta_map(run)[:N_A])
                for lbl, run in HEINE_CONDS.items()}
    vm_global = max(abs(A_dieckow_flat).max(), abs(A_dieckow_sign).max(),
                    *[abs(A).max() for A in heine_As.values()])

    def plot_heatmap(ax, A, title, vmax=None):
        vm = vmax or max(abs(A).max(), 0.5)
        im = ax.imshow(A, cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='equal')
        ax.set_xticks(range(N_SP)); ax.set_xticklabels(SHORT, fontsize=10)
        ax.set_yticks(range(N_SP)); ax.set_yticklabels(SHORT, fontsize=10)
        for i in range(N_SP):
            for j in range(N_SP):
                color = 'white' if abs(A[i, j]) > 0.6 * vm else 'black'
                ax.text(j, i, f'{A[i, j]:.2f}', ha='center', va='center',
                        fontsize=7.5, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return im

    # Layout: top row = Dieckow flat | sign | Δ | colorbar
    #         bot row = CS | CH | DS | DH  (all 4 Heine conditions)
    fig = plt.figure(figsize=(15, 7.5))
    outer = GridSpec(2, 1, figure=fig, hspace=0.38)
    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0],
                                  width_ratios=[1, 1, 1, 0.07], wspace=0.30)
    bot = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.30)

    # Top row
    im = plot_heatmap(fig.add_subplot(top[0]), A_dieckow_flat,
                      'Dieckow in-vivo\n(flat prior)', vm_global)
    plot_heatmap(fig.add_subplot(top[1]), A_dieckow_sign,
                 'Dieckow in-vivo\n(sign prior)', vm_global)
    diff = A_dieckow_sign - A_dieckow_flat
    vd = max(abs(diff).max(), 0.1)
    ax_d = fig.add_subplot(top[2])
    im_d = ax_d.imshow(diff, cmap='RdBu_r', vmin=-vd, vmax=vd, aspect='equal')
    ax_d.set_xticks(range(N_SP)); ax_d.set_xticklabels(SHORT, fontsize=10)
    ax_d.set_yticks(range(N_SP)); ax_d.set_yticklabels(SHORT, fontsize=10)
    for i in range(N_SP):
        for j in range(N_SP):
            c = 'white' if abs(diff[i, j]) > 0.6 * vd else 'black'
            ax_d.text(j, i, f'{diff[i, j]:.2f}', ha='center', va='center',
                      fontsize=7.5, color=c)
    ax_d.set_title('Δ A (sign − flat)', fontsize=12, fontweight='bold')
    cbar_ax = fig.add_subplot(top[3])
    fig.colorbar(im, cax=cbar_ax, label='$A_{ij}$')

    # Bottom row: all 4 Heine conditions
    for k, (label, run) in enumerate(HEINE_CONDS.items()):
        plot_heatmap(fig.add_subplot(bot[k]), heine_As[label],
                     f'Heine {label} (in-vitro)', vm_global)

    fig.suptitle('Interaction matrices $A$: Dieckow in-vivo vs Heine in-vitro\n'
                 'Red = cooperative ($A_{ij}>0$);  Blue = competitive ($A_{ij}<0$)',
                 fontsize=13)
    out = OUT_DIR / 'fig2_A_heatmaps.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

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
    ax.set_yticklabels(keys, fontsize=10)
    ax.set_xlabel('RMSE (weeks 2+3)', fontsize=11)
    ax.set_title('Cross-prediction: Heine A on Dieckow in-vivo', fontsize=12)
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=10)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis='x', alpha=0.3)

    # Fig 4: A-matrix correlation
    ax2 = axes[1]
    im = ax2.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
    ax2.set_xticks(range(len(corr_labels)))
    ax2.set_xticklabels(corr_labels, fontsize=11)
    ax2.set_yticks(range(len(corr_labels)))
    ax2.set_yticklabels(corr_labels, fontsize=11)
    ax2.set_title('A matrix correlation (Heine × Dieckow)', fontsize=12)
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            c = 'white' if abs(corr_mat[i,j]) > 0.7 else 'black'
            ax2.text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                     fontsize=11, color=c)
    plt.colorbar(im, ax=ax2, label='Pearson r', shrink=0.8)

    plt.tight_layout()
    out = OUT_DIR / 'fig3_cross_prediction.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=300, bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.15)
    ax.axis('off')

    # draw species nodes
    for i, sp in enumerate(SPECIES):
        y = sp_y[sp]
        c = ax.add_patch(plt.Circle((0.0, y), 0.055, color=SP_COL[i], zorder=4))
        ax.text(-0.12, y, sp, ha='right', va='center', fontsize=12, fontweight='bold')

    # draw metabolite nodes
    for m in METS:
        y = met_y[m]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.92, y - 0.035), 0.32, 0.07,
            boxstyle='round,pad=0.01', facecolor='#f0f0f0',
            edgecolor='#888', linewidth=0.8, zorder=4))
        ax.text(1.08, y, m, ha='center', va='center', fontsize=10.5, zorder=5)

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
    ax.legend(handles=legend_elems, loc='lower left', fontsize=10, frameon=True)

    # column headers
    ax.text(0.0,  1.12, 'Species', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.08, 1.12, 'Metabolites', ha='center', fontsize=12, fontweight='bold')

    ax.set_title('Metabolite-mediated interactions among 5 implant biofilm species\n'
                 '(Dieckow Supplementary File 1)', fontsize=12)

    plt.tight_layout()
    out = OUT_DIR / 'fig4_metabolic_network.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Fig 5: Sign pattern comparison (5 conditions) ────────────────────────────
def fig5_sign_comparison(theta_dieckow_flat):
    cond_thetas = {}
    for label, run in HEINE_CONDS.items():
        cond_thetas[label] = load_theta_map(run)[:N_A]
    cond_thetas['Dieckow'] = theta_dieckow_flat[:N_A]

    cond_names = list(cond_thetas.keys())   # CS CH DS DH Dieckow
    # Off-diagonal pairs only (exclude self-interactions)
    pairs = [(i, j, f'{SHORT[i]}–{SHORT[j]}')
             for j in range(N_SP) for i in range(j)]

    sign_mat = np.zeros((len(pairs), len(cond_names)))
    for ci, cname in enumerate(cond_names):
        A = build_A(cond_thetas[cname])
        for pi, (i, j, _) in enumerate(pairs):
            sign_mat[pi, ci] = np.sign(A[i, j])

    # Identify conserved pairs
    conserved_pos = np.all(sign_mat > 0, axis=1)
    conserved_neg = np.all(sign_mat < 0, axis=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = matplotlib.colors.ListedColormap(['#d7191c', '#ffffbf', '#2c7bb6'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(sign_mat, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_names, fontsize=11, fontweight='bold')
    pair_labels = [lbl for _, _, lbl in pairs]
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=10)

    # mark conserved rows
    for pi in range(len(pairs)):
        if conserved_pos[pi] or conserved_neg[pi]:
            ax.add_patch(FancyBboxPatch((-0.5, pi-0.5), len(cond_names), 1,
                         boxstyle='round,pad=0.05', linewidth=1.5,
                         edgecolor='gold', facecolor='none', zorder=3))

    ax.set_title('Sign pattern of A matrix entries across 5 conditions\n'
                 '(red=negative, blue=positive, gold border=conserved)',
                 fontsize=12)
    ax.set_xlabel('Condition', fontsize=11)
    ax.set_ylabel('Interaction pair', fontsize=11)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#2c7bb6', label='Positive (+)'),
        Patch(facecolor='#d7191c', label='Negative (−)'),
        Patch(facecolor='none', edgecolor='gold', linewidth=2,
              label='Conserved (all 5 cond.)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', frameon=True, fontsize=10)

    plt.tight_layout()
    out = OUT_DIR / 'fig5_sign_comparison.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def fig6_dieckow_A_uncertainty(theta_samples_5sp):
    A_s = build_A_samples(theta_samples_5sp)
    A_mean = A_s.mean(axis=0)
    A_lo = np.percentile(A_s, 2.5, axis=0)
    A_hi = np.percentile(A_s, 97.5, axis=0)
    p_pos = (A_s > 0).mean(axis=0)
    sign_cert = 2.0 * np.abs(p_pos - 0.5)
    sign_cert = np.clip(sign_cert, 0.0, 1.0)

    fig = plt.figure(figsize=(11, 4.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.35)

    def heat(ax, M, title, cmap, vmin, vmax, fmt=None):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks(range(N_SP)); ax.set_xticklabels(SHORT, fontsize=10)
        ax.set_yticks(range(N_SP)); ax.set_yticklabels(SHORT, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if fmt is not None:
            for i in range(N_SP):
                for j in range(N_SP):
                    ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center',
                            fontsize=7, color='black')
        return im

    ax0 = fig.add_subplot(gs[0, 0])
    vm = max(1e-12, float(np.max(np.abs(A_mean))))
    im0 = heat(ax0, A_mean, 'Dieckow A mean', 'RdBu_r', -vm, vm, fmt='{:.2f}')

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = heat(ax1, p_pos, 'P(A>0)', 'RdBu_r', 0.0, 1.0, fmt='{:.2f}')

    ax2 = fig.add_subplot(gs[0, 2])
    plot_signed_network(
        ax2,
        node_labels=SHORT,
        A_mean=A_mean,
        sign_certainty=sign_cert,
        node_colors=SPECIES_COLORS,
        title='Signed network\n(width=|mean|, alpha=sign certainty)',
    )

    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label='A mean')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Probability')

    out = OUT_DIR / 'fig6_dieckow_A_uncertainty.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    summary = []
    for i in range(N_SP):
        for j in range(i + 1, N_SP):
            summary.append({
                'pair': f'{SHORT[i]}–{SHORT[j]}',
                'mean': float(A_mean[i, j]),
                'ci_low': float(A_lo[i, j]),
                'ci_high': float(A_hi[i, j]),
                'p_pos': float(p_pos[i, j]),
                'sign_certainty': float(sign_cert[i, j]),
            })
    summary = sorted(summary, key=lambda r: (r['sign_certainty'], abs(r['mean'])), reverse=True)

    out_json = OUT_DIR / 'fig6_dieckow_A_uncertainty_summary.json'
    with open(out_json, 'w') as f:
        json.dump({'pairs': summary}, f, indent=2)
    print(f'Saved: {out_json}')


def fig7_guild_A_network():
    if not GUILD_FIT_JSON.exists():
        print('  fit_guild.json not found — skipping Fig7')
        return
    d = json.load(open(GUILD_FIT_JSON))
    A = np.array(d['A'], dtype=float)
    labels = d.get('guilds', [f'G{i+1}' for i in range(A.shape[0])])
    short = [GUILD_SHORT.get(x, x[:8]) for x in labels]

    fig, ax = plt.subplots(figsize=(6.3, 6.0))
    cert = np.ones_like(A, dtype=float)
    colors = [DIECKOW_GUILD_COLORS.get(x, '#aaaaaa') for x in labels]
    plot_signed_network(ax, short, A, cert, colors, 'Guild gLV: signed A network')
    out = OUT_DIR / 'fig7_guild_A_network.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
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

    if HAS_JAX:
        print('Generating Fig1: per-patient predictions...')
        fig1_patient_predictions(phi_obs, valid_patients, theta_per_patient)
    else:
        print('  JAX not available — skipping Fig1 patient predictions')

    print('Generating Fig2: A matrix heatmaps...')
    fig2_A_heatmaps(theta_flat, theta_sign)

    print('Generating Fig3: cross-prediction...')
    fig3_cross_prediction()

    print('Generating Fig4: metabolic network...')
    fig4_metabolic_network()

    print('Generating Fig5: sign comparison...')
    fig5_sign_comparison(theta_flat)

    if 'samples' in d_flat and len(d_flat['samples']) > 0:
        print('Generating Fig6: Dieckow A uncertainty (posterior samples)...')
        fig6_dieckow_A_uncertainty(np.array(d_flat['samples']))
    else:
        print('  No posterior samples found in fit_joint_5sp_1000p.json — skipping Fig6')

    print('Generating Fig7: guild A network...')
    fig7_guild_A_network()

    print(f'\nAll figures → {OUT_DIR}')


if __name__ == '__main__':
    main()
