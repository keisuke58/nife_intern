#!/usr/bin/env python3
"""
dieckow_analysis.py — Post-fit analysis (3 parts)
  Part 1: In vivo attractor re-estimation (MAP + posterior samples)
  Part 2: Per-patient fit visualization (obs vs MAP)
  Part 3: Sign-prior comparison (sign_prior=OFF vs ON)

Usage:
  python dieckow_analysis.py
  python dieckow_analysis.py --no-jax   # skip Part 1 (no ODE needed)
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# JAX imported lazily in init_jax() — only called when Part 1 runs
jax = None
jnp = None
simulate_0d_nsp = None

def init_jax():
    global jax, jnp, simulate_0d_nsp
    if jax is not None:
        return
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    try:
        import jax_cuda12_plugin  # noqa
    except ImportError:
        pass
    import jax as _jax
    import jax.numpy as _jnp
    _jax.config.update('jax_enable_x64', True)
    sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
    sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/core')
    from hamilton_ode_jax_nsp import simulate_0d_nsp as _sim
    jax = _jax; jnp = _jnp; simulate_0d_nsp = _sim
    print(f'  JAX devices: {jax.devices()}')

# ── Constants ─────────────────────────────────────────────────────────────────
GENERA   = ['Streptococcus', 'Actinomyces', 'Veillonella', 'Fusobacterium', 'Porphyromonas']
SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP     = 5
N_A      = 15
N_B      = 50
N_PARAMS = 65
N_STEPS  = 2500
DT       = 1e-4
C_CONST  = 25.0
ALPHA    = 100.0

FITS_DIR   = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
OUT_DIR    = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_analysis')
DATA_CACHE = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_obs_matrix_5sp.json')
GMM_CSV    = Path('/home/nishioka/IKM_Hiwi/nife/results/gmm_attractor_analysis.csv')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sign-prior index map (upper-triangle index → expected sign)
META_SIGN = {
    1: +1,   # A[So,An]
    2: +1,   # A[So,Vd]
    3: +1,   # A[So,Fn]
    6: +1,   # A[An,Vd]
    8: +1,   # A[An,Pg]
   11: +1,   # A[Vd,Pg]
}

A_NAMES = []
for r in range(N_SP):
    for c in range(r, N_SP):
        A_NAMES.append(f'A[{SHORT[r]},{SHORT[c]}]')

PATIENT_CMAP = plt.cm.tab10(np.linspace(0, 1, 10))
SP_COLORS    = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#264653']

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    d1  = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    d2  = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p_meta.json'))
    raw = json.load(open(DATA_CACHE))
    obs = {p: np.array(v) for p, v in raw['obs'].items()}
    return d1, d2, obs


def extract_theta_patient(theta, idx):
    A = theta[:N_A]
    b = theta[N_A + idx*N_SP: N_A + (idx+1)*N_SP]
    return np.concatenate([A, b])


def run_ode(theta20, phi0, n_weeks):
    """Run ODE n_weeks consecutive weeks; return (n_weeks, N_SP) array."""
    theta_j = jnp.array(theta20, dtype=jnp.float64)
    phi_cur  = jnp.array(phi0,   dtype=jnp.float64)
    out = []
    for _ in range(n_weeks):
        traj  = simulate_0d_nsp(theta_j, n_sp=N_SP, n_steps=N_STEPS, dt=DT,
                                 phi_init=phi_cur, c_const=C_CONST, alpha_const=ALPHA)
        phi_n = np.array(traj[-1])
        s = phi_n.sum()
        phi_n = phi_n / s if s > 1e-12 else phi_n
        out.append(phi_n)
        phi_cur = jnp.array(phi_n)
    return np.array(out)

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — In vivo attractor re-estimation
# ═══════════════════════════════════════════════════════════════════════════════

def part1_attractors(d1, obs):
    init_jax()
    print('\n=== Part 1: In vivo attractor re-estimation ===')
    theta_map = np.array(d1['theta_map'])
    samples   = np.array(d1['samples'])   # (1000, 65)

    # ── Per-patient MAP attractor (run 20 weeks from week-1 IC) ───────────────
    print('  Running per-patient 20-week ODE from week-1 IC (MAP θ)...')
    pat_att = {}
    for i, p in enumerate(PATIENTS):
        if p not in obs:
            continue
        theta_p = extract_theta_patient(theta_map, i)
        phi0    = obs[p][:, 0]
        traj    = run_ode(theta_p, phi0, n_weeks=20)
        pat_att[p] = traj[-1]
        print(f'    Patient {p}:  So={traj[-1][0]:.3f}  Fn={traj[-1][3]:.3f}  Pg={traj[-1][4]:.3f}')

    # ── Posterior attractor spread (50 random samples, avg-b theta) ───────────
    print('  Running 50 posterior samples (avg-b A) × 30 random ICs for attractor cloud...')
    rng = np.random.default_rng(42)
    n_post_samples = 50
    post_idx = rng.choice(len(samples), n_post_samples, replace=False)

    all_endpoints = []
    for pi in post_idx:
        theta_s  = samples[pi]
        # avg b across patients
        b_avg    = theta_s[N_A:].reshape(len(PATIENTS), N_SP).mean(axis=0)
        theta_20 = np.concatenate([theta_s[:N_A], b_avg])
        ics = rng.dirichlet(np.ones(N_SP), size=5)
        for ic in ics:
            traj = run_ode(theta_20, ic, n_weeks=20)
            all_endpoints.append(traj[-1])
    all_endpoints = np.array(all_endpoints)  # (250, 5)

    # ── Szafranski clinical centroids (from ANALYSIS_NOTES) ───────────────────
    szaf = {
        'Health (PIH)':    np.array([0.591, 0.056, 0.140, 0.140, 0.072]),
        'Mucositis (PIM)': np.array([0.625, 0.035, 0.104, 0.171, 0.065]),
        'Peri-impl. (PI)': np.array([0.311, 0.024, 0.076, 0.352, 0.237]),
    }
    szaf_colors = ['#1565C0', '#2E7D32', '#B71C1C']

    # ── Figure 1: 2-panel So-Pg / So-Fn ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax_i, (xi, yi, xl, yl) in enumerate([
        (0, 4, 'Streptococcus (So)', 'Porphyromonas (Pg)'),
        (0, 3, 'Streptococcus (So)', 'Fusobacterium (Fn)'),
    ]):
        ax = axes[ax_i]
        # Posterior cloud
        ax.scatter(all_endpoints[:, xi], all_endpoints[:, yi],
                   c='#B0BEC5', s=18, alpha=0.5, label='Posterior attractors (250)')
        # Per-patient MAP
        for i, (p, att) in enumerate(pat_att.items()):
            ax.scatter(att[xi], att[yi], c=[PATIENT_CMAP[i]], s=130,
                       zorder=6, edgecolors='k', linewidths=0.5)
            ax.annotate(p, (att[xi], att[yi]), fontsize=8,
                        ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')
        # Szafranski
        for (lbl, comp), c in zip(szaf.items(), szaf_colors):
            ax.scatter(comp[xi], comp[yi], marker='*', s=350, c=c,
                       zorder=10, edgecolors='k', linewidths=0.5, label=lbl)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Patient legend
    pat_handles = [mpatches.Patch(color=PATIENT_CMAP[i], label=f'Patient {p}')
                   for i, p in enumerate(PATIENTS) if p in pat_att]
    szaf_handles = [mpatches.Patch(color=c, label=lbl)
                    for lbl, c in zip(szaf.keys(), szaf_colors)]
    cloud_handle = mpatches.Patch(color='#B0BEC5', label='Posterior attractors (250)')
    axes[0].legend(handles=pat_handles + [cloud_handle], fontsize=7, loc='upper right', ncol=2)
    axes[1].legend(handles=szaf_handles, fontsize=8, loc='upper right')

    axes[0].set_title('In vivo attractors: MAP per-patient\nvs Szafranski clinical centroids  [So–Pg]', fontsize=10)
    axes[1].set_title('In vivo attractors  [So–Fn]', fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'part1_invivo_attractors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUT_DIR}/part1_invivo_attractors.png')

    # ── Figure 2: composition bar of per-patient attractors ───────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    x = np.arange(len(pat_att))
    labels_p = list(pat_att.keys())
    bot = np.zeros(len(labels_p))
    for sp in range(N_SP):
        vals = [pat_att[p][sp] for p in labels_p]
        ax2.bar(x, vals, bottom=bot, color=SP_COLORS[sp], label=SHORT[sp])
        bot += vals

    # Also add Szafranski as dashed outlines
    offset = len(labels_p)
    ax2.bar(offset,   szaf['Health (PIH)'][0],    color=SP_COLORS[0], alpha=0.3)
    for sp in range(N_SP):
        sz_vals = [szaf[k][sp] for k in szaf]
        ax2.bar(np.arange(offset, offset+3), sz_vals,
                color=SP_COLORS[sp], alpha=0.3)
    ax2.set_xticks(list(x) + list(np.arange(offset, offset+3)))
    ax2.set_xticklabels(labels_p + ['PIH', 'PIM', 'PI'], fontsize=10)
    ax2.axvline(offset - 0.5, color='gray', ls='--', lw=1.2)
    ax2.text(offset + 1, 1.02, 'Szafranski', ha='center', fontsize=9, color='gray')
    ax2.text(len(labels_p)/2 - 0.5, 1.02, 'Dieckow MAP attractors', ha='center', fontsize=9)
    ax2.set_ylabel('Relative abundance', fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='lower right', ncol=5, fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_title('Per-patient MAP attractors vs Szafranski clinical means', fontsize=11)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'part1_attractor_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUT_DIR}/part1_attractor_bars.png')

    # Save JSON
    out_json = {'genera': SHORT,
                'patient_attractors_map': {p: v.tolist() for p, v in pat_att.items()},
                'posterior_cloud': all_endpoints.tolist()}
    with open(OUT_DIR / 'patient_attractors.json', 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f'  Saved: {OUT_DIR}/patient_attractors.json')

    return pat_att, all_endpoints


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Per-patient fit visualization
# ═══════════════════════════════════════════════════════════════════════════════

def part2_patient_fits(d1, obs, with_predictions=True):
    print('\n=== Part 2: Per-patient fit visualization ===')
    theta_map = np.array(d1['theta_map'])
    weeks_lab = ['Wk1\n(IC)', 'Wk2', 'Wk3']

    if with_predictions:
        init_jax()

    # ── Panel grid: 2×5 ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    per_rmse = {}

    for ax_i, (i, p) in enumerate(zip(range(len(PATIENTS)), PATIENTS)):
        ax = axes[ax_i]
        if p not in obs:
            ax.set_visible(False)
            continue

        phi_obs = obs[p]                         # (5, 3)
        n_wk    = int((phi_obs.sum(axis=0) > 0).sum())
        bar_w   = 0.35 if with_predictions else 0.6
        x_all   = np.arange(n_wk)
        x_pred  = np.arange(1, min(3, n_wk))

        # Observed bars
        bot_obs = np.zeros(n_wk)
        for sp in range(N_SP):
            ax.bar(x_all - (bar_w/2 if with_predictions else 0),
                   phi_obs[sp, :n_wk], bottom=bot_obs,
                   width=bar_w, color=SP_COLORS[sp],
                   label=SHORT[sp] if ax_i == 0 else None, alpha=0.9)
            bot_obs += phi_obs[sp, :n_wk]

        if with_predictions:
            theta_p = extract_theta_patient(theta_map, i)
            pred2   = run_ode(theta_p, phi_obs[:, 0], n_weeks=2)
            # RMSE
            rmse_list = []
            for t in range(1, n_wk):
                rmse_list.append(np.sqrt(np.mean((pred2[t-1] - phi_obs[:, t])**2)))
            per_rmse[p] = float(np.mean(rmse_list)) if rmse_list else float('nan')
            # Predicted bars
            n_pred   = min(2, n_wk - 1)
            bot_pred = np.zeros(n_pred)
            for sp in range(N_SP):
                ax.bar(x_pred[:n_pred] + bar_w/2, pred2[:n_pred, sp],
                       bottom=bot_pred, width=bar_w, color=SP_COLORS[sp],
                       alpha=0.45, hatch='//')
                bot_pred += pred2[:n_pred, sp]
            title_str = f'Patient {p}  RMSE={per_rmse[p]:.3f}'
        else:
            per_rmse[p] = float('nan')
            title_str = f'Patient {p}'

        ax.set_xticks(x_all)
        ax.set_xticklabels(weeks_lab[:n_wk], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(title_str, fontsize=10)
        ax.set_ylabel('Rel. abund.' if ax_i % 5 == 0 else '', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    # Legend
    sp_patches = [mpatches.Patch(color=c, label=s) for c, s in zip(SP_COLORS, SHORT)]
    legend_handles = sp_patches
    if with_predictions:
        legend_handles += [
            mpatches.Patch(color='gray', alpha=0.9, label='Observed'),
            mpatches.Patch(color='gray', alpha=0.45, label='Predicted (MAP)', hatch='//'),
        ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    pred_note = '' if with_predictions else ' (observed only)'
    fig.suptitle(f'Dieckow 5-sp Hamilton ODE — per-patient fit (sign_prior=OFF){pred_note}\n'
                 f'Overall RMSE = {d1["rmse"]:.4f}', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'part2_patient_fits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUT_DIR}/part2_patient_fits.png')

    # ── Per-patient RMSE bar (only if predictions were made) ─────────────────
    valid_rmse = {p: v for p, v in per_rmse.items() if not np.isnan(v)}
    if valid_rmse:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ps   = [p for p in PATIENTS if p in valid_rmse]
        vals = [valid_rmse[p] for p in ps]
        bars = ax2.bar(ps, vals, color=[PATIENT_CMAP[PATIENTS.index(p)] for p in ps])
        mean_rmse = float(np.mean(vals))
        ax2.axhline(mean_rmse, color='red', ls='--', lw=1.5, label=f'Mean = {mean_rmse:.3f}')
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.set_xlabel('Patient', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('Per-patient RMSE — Dieckow MAP fit', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, max(vals)*1.25)
        ax2.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        fig2.savefig(OUT_DIR / 'part2_rmse_per_patient.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {OUT_DIR}/part2_rmse_per_patient.png')
        print(f'  Per-patient RMSE: { {p: round(v,4) for p, v in valid_rmse.items()} }')
        print(f'  Worst: {max(valid_rmse, key=valid_rmse.get)} = {max(valid_rmse.values()):.4f}')
        print(f'  Best:  {min(valid_rmse, key=valid_rmse.get)} = {min(valid_rmse.values()):.4f}')
    else:
        print('  [skip] RMSE bar (no predictions in --no-jax mode)')

    return per_rmse


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Sign-prior comparison
# ═══════════════════════════════════════════════════════════════════════════════

def part3_sign_prior(d1, d2):
    print('\n=== Part 3: Sign-prior comparison ===')
    s1 = np.array(d1['samples'])   # (1000, 65) sign_prior=OFF
    s2 = np.array(d2['samples'])   # (1000, 65) sign_prior=ON
    t1 = np.array(d1['theta_map'])
    t2 = np.array(d2['theta_map'])

    con_idx = sorted(META_SIGN.keys())
    n_con   = len(con_idx)

    # ── Compliance stats ───────────────────────────────────────────────────────
    print(f'\n  {"k":>3}  {"param":15}  {"exp":>4}  {"MAP_OFF":>8}  {"MAP_ON":>8}  {"comply_OFF":>10}  {"comply_ON":>10}')
    print('  ' + '-'*68)
    for k in con_idx:
        exp = META_SIGN[k]
        v1  = s1[:, k]; v2 = s2[:, k]
        c1  = (v1 * exp > 0).mean()
        c2  = (v2 * exp > 0).mean()
        print(f'  {k:3d}  {A_NAMES[k]:15}  {"+"if exp>0 else"-":>4}  {t1[k]:8.3f}  {t2[k]:8.3f}  {c1:10.1%}  {c2:10.1%}')

    print(f'\n  RMSE:  sign_prior=OFF = {d1["rmse"]:.4f},  sign_prior=ON = {d2["rmse"]:.4f}'
          f'  (Δ = {d2["rmse"]-d1["rmse"]:+.4f})')

    # Bhattacharyya overlap for all A params
    overlaps = []
    for k in range(N_A):
        v1, v2  = s1[:, k], s2[:, k]
        mu1, s_1 = v1.mean(), max(v1.std(), 1e-9)
        mu2, s_2 = v2.mean(), max(v2.std(), 1e-9)
        bc = np.sqrt(2*s_1*s_2/(s_1**2+s_2**2)) * np.exp(-0.25*(mu1-mu2)**2/(s_1**2+s_2**2))
        overlaps.append(bc)
    print(f'  Posterior Bhattacharyya overlap (A params):  mean={np.mean(overlaps):.3f}'
          f'  min={np.min(overlaps):.3f}  (A[{A_NAMES[int(np.argmin(overlaps))]}])')

    # ── Figure 3a: violin plots of sign-constrained params ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    run_configs = [
        ('sign_prior=OFF', s1, t1, '#2196F3'),
        ('sign_prior=ON',  s2, t2, '#F44336'),
    ]
    for ax_i, (lbl, samp, theta, col) in enumerate(run_configs):
        ax = axes[ax_i]
        data = [samp[:, k] for k in con_idx]
        vp = ax.violinplot(data, positions=range(n_con), showmedians=True,
                           widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor(col); body.set_alpha(0.55)
        vp['cmedians'].set_color('black')
        ax.scatter(range(n_con), [theta[k] for k in con_idx],
                   c='black', s=70, zorder=8, label='MAP', marker='D')
        ax.axhline(0, color='gray', ls='--', lw=1)
        # Shade violation region per param
        for j, k in enumerate(con_idx):
            exp = META_SIGN[k]
            lo  = -5.5 if exp < 0 else -5.5
            hi  =  5.5
            shade_lo = -5.5 if exp > 0 else 0.0
            shade_hi =  0.0 if exp > 0 else 5.5
            ax.fill_betweenx([shade_lo, shade_hi], j-0.4, j+0.4,
                             color='red', alpha=0.08)
        ax.set_xticks(range(n_con))
        ax.set_xticklabels([A_NAMES[k] for k in con_idx], rotation=30,
                           ha='right', fontsize=9)
        ax.set_ylabel('Parameter value', fontsize=11)
        ax.set_ylim(-5.5, 5.5)
        ax.set_title(f'{lbl}\nRMSE = {d1["rmse"] if ax_i==0 else d2["rmse"]:.4f}', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle('Sign-constrained A parameters: posterior violins\n'
                 '(red shading = sign-violation region)', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'part3_sign_prior_violin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {OUT_DIR}/part3_sign_prior_violin.png')

    # ── Figure 3b: MAP A matrix heatmaps ─────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    A_mats = {}
    for lbl, theta in [('OFF', t1), ('ON', t2)]:
        A = np.zeros((N_SP, N_SP))
        ki = 0
        for r in range(N_SP):
            for c in range(r, N_SP):
                A[r, c] = A[c, r] = theta[ki]; ki += 1
        A_mats[lbl] = A

    vmax = max(abs(A_mats['OFF']).max(), abs(A_mats['ON']).max())
    titles = ['sign_prior=OFF (MAP)', 'sign_prior=ON (MAP)', 'Difference (ON − OFF)']
    mats   = [A_mats['OFF'], A_mats['ON'], A_mats['ON'] - A_mats['OFF']]
    vmaxs  = [vmax, vmax, 2.0]

    for ax_i, (mat, ttl, vm) in enumerate(zip(mats, titles, vmaxs)):
        ax = axes2[ax_i]
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vm, vmax=vm)
        ax.set_xticks(range(N_SP)); ax.set_xticklabels(SHORT, fontsize=11)
        ax.set_yticks(range(N_SP)); ax.set_yticklabels(SHORT, fontsize=11)
        ax.set_title(ttl, fontsize=11)
        for r in range(N_SP):
            for c in range(N_SP):
                txt_c = 'white' if abs(mat[r, c]) > vm * 0.55 else 'black'
                fmt   = f'{mat[r,c]:+.2f}' if ax_i == 2 else f'{mat[r,c]:.2f}'
                ax.text(c, r, fmt, ha='center', va='center', fontsize=8, color=txt_c)
        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.suptitle('MAP A interaction matrix: sign_prior ON vs OFF', fontsize=12)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'part3_A_matrix_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUT_DIR}/part3_A_matrix_heatmap.png')

    # ── Figure 3c: compliance bar chart ───────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    x = np.arange(n_con)
    c_off = [(s1[:, k] * META_SIGN[k] > 0).mean() for k in con_idx]
    c_on  = [(s2[:, k] * META_SIGN[k] > 0).mean() for k in con_idx]
    ax3.bar(x - 0.18, c_off, 0.35, color='#2196F3', label='sign_prior=OFF', alpha=0.8)
    ax3.bar(x + 0.18, c_on,  0.35, color='#F44336', label='sign_prior=ON',  alpha=0.8)
    ax3.axhline(0.95, color='black', ls=':', lw=1, label='95% threshold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([A_NAMES[k] for k in con_idx], rotation=20, ha='right', fontsize=9)
    ax3.set_ylabel('Fraction of posterior satisfying sign', fontsize=11)
    ax3.set_ylim(0, 1.1)
    ax3.set_title('Sign compliance: posterior fraction', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig3.savefig(OUT_DIR / 'part3_sign_compliance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUT_DIR}/part3_sign_compliance.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-jax', action='store_true', help='Skip Part 1 (no ODE simulation)')
    args = parser.parse_args()

    print('Loading results...')
    d1, d2, obs = load_data()
    print(f'  1000p (sign_prior=OFF): RMSE={d1["rmse"]:.4f}, n_stages={d1["n_stages"]}')
    print(f'  1000p (sign_prior=ON):  RMSE={d2["rmse"]:.4f}, n_stages={d2["n_stages"]}')
    print(f'  Patients: {sorted(obs.keys())}')

    if not args.no_jax:
        part1_attractors(d1, obs)
        part2_patient_fits(d1, obs, with_predictions=True)
    else:
        print('\n[skip] Part 1 (--no-jax)')
        part2_patient_fits(d1, obs, with_predictions=False)

    part3_sign_prior(d1, d2)

    print(f'\nAll outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()
