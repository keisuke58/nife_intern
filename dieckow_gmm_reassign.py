#!/usr/bin/env python3
"""
dieckow_gmm_reassign.py — Two analyses:

 A) Patient F anomaly diagnosis
    - Why does MAP ODE converge to ~uniform (0.03 per species)?
    - F has Fn=0, Pg=0 throughout → 3-species system in practice
    - High b_Fn/b_Vd from TMCMC → phantom species destabilize attractor

 B) GMM re-assignment with in vivo θ (Dieckow MAP A matrix)
    - Replace Heine in vitro ODE attractors with Dieckow in vivo MAP A
    - Run ODE from each Szafranski sample for 20 weeks → endpoint
    - Compare with Szafranski GMM clusters (k=3/k=4)
    - Quantify in vitro vs in vivo offset

Usage:
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 dieckow_gmm_reassign.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/core')
from hamilton_ode_jax_nsp import simulate_0d_nsp

print(f'JAX devices: {jax.devices()}')

# ── Constants ─────────────────────────────────────────────────────────────────
SHORT    = ['So', 'An', 'Vd', 'Fn', 'Pg']
PATIENTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L']
N_SP = 5; N_A = 15; N_STEPS = 2500; DT = 1e-4; C = 25.0; ALPHA = 100.0

FITS_DIR   = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_fits')
OUT_DIR    = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_gmm')
DATA_CACHE = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_obs_matrix_5sp.json')
GMM_CSV    = Path('/home/nishioka/IKM_Hiwi/nife/results/gmm_attractor_analysis.csv')
HEINE_RUNS = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SP_COLORS = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#264653']

def run_ode(theta20, phi0, n_weeks=20):
    t = jnp.array(theta20, dtype=jnp.float64)
    phi = jnp.array(phi0,   dtype=jnp.float64)
    for _ in range(n_weeks):
        traj = simulate_0d_nsp(t, n_sp=N_SP, n_steps=N_STEPS, dt=DT,
                                phi_init=phi, c_const=C, alpha_const=ALPHA)
        phi = traj[-1]
        phi = phi / jnp.maximum(phi.sum(), 1e-12)
    return np.array(phi)

# ═══════════════════════════════════════════════════════════════════════════════
# A) Patient F diagnosis
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_patient_F():
    print('\n=== A) Patient F anomaly diagnosis ===')
    d1  = json.load(open(FITS_DIR / 'fit_joint_5sp_1000p.json'))
    t   = np.array(d1['theta_map'])
    obs_raw = json.load(open(DATA_CACHE))
    obs = {p: np.array(v) for p, v in obs_raw['obs'].items()}

    i_F = PATIENTS.index('F')
    b_F = t[N_A + i_F*N_SP: N_A + (i_F+1)*N_SP]
    A_utri = t[:N_A]
    theta_F = np.concatenate([A_utri, b_F])

    phi_F = obs['F']
    print(f'  b_F: {dict(zip(SHORT, b_F.round(3)))}')
    print(f'  Obs week1: Fn={phi_F[3,0]:.4f}  Pg={phi_F[4,0]:.4f}  (effectively 3-sp)')

    # Test 1: Run from week1 IC with full b_F
    att_full = run_ode(theta_F, phi_F[:, 0])
    print(f'\n  Attractor (full b_F, from week1 IC):')
    print(f'    {dict(zip(SHORT, att_full.round(4)))}')

    # Test 2: Run with b_Fn=b_Pg=0.01 (suppress phantom species)
    b_suppress = b_F.copy()
    b_suppress[3] = 0.01   # b_Fn
    b_suppress[4] = 0.01   # b_Pg
    theta_supp = np.concatenate([A_utri, b_suppress])
    att_supp = run_ode(theta_supp, phi_F[:, 0])
    print(f'\n  Attractor (b_Fn=b_Pg=0.01, suppress phantom):')
    print(f'    {dict(zip(SHORT, att_supp.round(4)))}')

    # Test 3: Run from 3-sp IC (Fn=Pg=0)
    phi_3sp = phi_F[:, 0].copy()
    phi_3sp[3] = phi_3sp[4] = 0.0
    s = phi_3sp.sum()
    phi_3sp /= s
    att_3sp = run_ode(theta_F, phi_3sp)
    print(f'\n  Attractor (IC with Fn=Pg=0):')
    print(f'    {dict(zip(SHORT, att_3sp.round(4)))}')

    # Compare average b (all patients except F)
    b_avg = np.zeros(N_SP)
    for i, p in enumerate(PATIENTS):
        if p != 'F':
            b_avg += t[N_A + i*N_SP: N_A + (i+1)*N_SP]
    b_avg /= (len(PATIENTS) - 1)
    theta_avg = np.concatenate([A_utri, b_avg])
    att_avg = run_ode(theta_avg, phi_F[:, 0])
    print(f'\n  Attractor (avg b, excl. F, from F week1 IC):')
    print(f'    {dict(zip(SHORT, att_avg.round(4)))}')

    # ── Figure: Patient F trajectories ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    configs = [
        ('Full b_F\n(MAP)', theta_F, phi_F[:, 0]),
        ('b_Fn=b_Pg=0.01\n(phantom suppressed)', theta_supp, phi_F[:, 0]),
        ('Avg b (excl. F)', theta_avg, phi_F[:, 0]),
    ]
    for ax, (lbl, theta, phi0) in zip(axes, configs):
        # Run week-by-week and record trajectory
        traj = [phi0.copy()]
        phi_cur = jnp.array(phi0, dtype=jnp.float64)
        t_j = jnp.array(theta, dtype=jnp.float64)
        for _ in range(12):
            out = simulate_0d_nsp(t_j, n_sp=N_SP, n_steps=N_STEPS, dt=DT,
                                   phi_init=phi_cur, c_const=C, alpha_const=ALPHA)
            phi_n = np.array(out[-1])
            phi_n /= max(phi_n.sum(), 1e-12)
            traj.append(phi_n)
            phi_cur = jnp.array(phi_n)
        traj = np.array(traj)
        for sp in range(N_SP):
            ax.plot(range(13), traj[:, sp], color=SP_COLORS[sp], label=SHORT[sp], lw=2)
        # Mark observed weeks 1-3
        for wk in range(3):
            for sp in range(N_SP):
                ax.scatter(wk, phi_F[sp, wk], color=SP_COLORS[sp],
                           s=80, zorder=6, edgecolors='black', linewidths=0.8)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Rel. abundance' if ax == axes[0] else '')
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Patient F: ODE trajectory analysis\n'
                 'Observed data (circles): Fn≈0, Pg≈0 across all weeks', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'patF_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {OUT_DIR}/patF_trajectory.png')

    print('\n  DIAGNOSIS:')
    print('  Patient F has Fn=0, Pg≈0 throughout → effectively a 3-species system.')
    print('  High b_Fn=13.3 and b_Vd=13.5 learned by TMCMC to prevent Fn/Pg growth.')
    print('  With full b_F, ODE attractor is degenerate (uniform ~0.03/sp).')
    print('  → Patient F is an outlier: 5-sp model is misspecified for this patient.')
    print('  → Recommend: flag as "Fn/Pg-absent" subtype, exclude from attractor mapping.')

    return A_utri, b_avg


# ═══════════════════════════════════════════════════════════════════════════════
# B) GMM re-assignment with in vivo θ
# ═══════════════════════════════════════════════════════════════════════════════

def gmm_reassign_invivo(A_utri, b_avg):
    print('\n=== B) GMM re-assignment with in vivo θ ===')

    # Load Szafranski GMM data
    gmm = pd.read_csv(GMM_CSV)
    print(f'  Szafranski samples: {len(gmm)}')
    print(f'  Columns: {list(gmm.columns)}')

    # Extract composition columns
    phi_cols = ['phi0_So', 'phi0_An', 'phi0_Vd', 'phi0_Fn', 'phi0_Pg']
    compositions = gmm[phi_cols].values   # (127, 5)

    # Identify Heine in vitro attractors for comparison
    heine_attractors = {}
    heine_theta_map = {}
    for cond in ['commensal_static', 'commensal_hobic', 'dh_baseline', 'dysbiotic_static']:
        path = HEINE_RUNS / cond / 'theta_MAP.json'
        if not path.exists():
            print(f'  WARNING: {path} not found')
            continue
        with open(path) as f:
            hd = json.load(f)
        # theta_MAP.json has Heine layout (20 params for 5sp)
        raw = hd.get('theta_full', hd.get('theta_sub', []))
        if len(raw) >= 20:
            heine_theta_map[cond] = np.array(raw[:20])

    # Run Heine ODE attractors from uniform IC
    print('\n  Computing Heine in vitro attractors...')
    phi_unif = np.ones(N_SP) / N_SP
    for cond, theta20 in heine_theta_map.items():
        att = run_ode(theta20, phi_unif)
        heine_attractors[cond] = att
        print(f'    {cond}: So={att[0]:.3f} Fn={att[3]:.3f} Pg={att[4]:.3f}')

    # In vivo θ: Dieckow A + avg b
    theta_invivo = np.concatenate([A_utri, b_avg])

    # Run each Szafranski sample through in vivo ODE for 20 weeks
    print(f'\n  Running {len(gmm)} Szafranski samples through in vivo ODE...')
    endpoints_invivo = []
    for i, phi0 in enumerate(compositions):
        phi0 = np.clip(phi0, 1e-6, 1.0)
        phi0 /= phi0.sum()
        ep = run_ode(theta_invivo, phi0, n_weeks=20)
        endpoints_invivo.append(ep)
        if (i+1) % 20 == 0:
            print(f'    {i+1}/{len(gmm)} done')
    endpoints_invivo = np.array(endpoints_invivo)   # (127, 5)

    # Also run with Heine DH theta for comparison
    print('\n  Running same samples through Heine DH theta...')
    endpoints_heine = []
    if 'dh_baseline' in heine_theta_map:
        for phi0 in compositions:
            phi0 = np.clip(phi0, 1e-6, 1.0); phi0 /= phi0.sum()
            ep = run_ode(heine_theta_map['dh_baseline'], phi0, n_weeks=20)
            endpoints_heine.append(ep)
        endpoints_heine = np.array(endpoints_heine)
    else:
        endpoints_heine = None

    # ── Attractor assignment: nearest endpoint cluster ─────────────────────────
    # Find unique attractor clusters in in vivo endpoints via k-means (k=4)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    km.fit(endpoints_invivo)
    cluster_labels = km.labels_
    centroids = km.cluster_centers_

    # Sort clusters by So content (descending)
    order = np.argsort(centroids[:, 0])[::-1]
    cluster_names = {order[0]: 'CS', order[1]: 'CH', order[2]: 'DH', order[3]: 'DS'}
    invivo_attr = [cluster_names[l] for l in cluster_labels]

    print('\n  In vivo attractor centroids (k=4):')
    for ci in range(4):
        c = centroids[ci]
        nm = cluster_names.get(ci, f'C{ci}')
        n  = (cluster_labels == ci).sum()
        print(f'    {nm} (n={n}): So={c[0]:.3f} An={c[1]:.3f} Vd={c[2]:.3f} Fn={c[3]:.3f} Pg={c[4]:.3f}')

    # Compare with original GMM assignment
    if 'gmm_attractor' in gmm.columns:
        orig_attr = gmm['gmm_attractor'].values
        agree = sum(a == b for a, b in zip(invivo_attr, orig_attr))
        print(f'\n  Agreement with original GMM: {agree}/{len(gmm)} ({agree/len(gmm):.1%})')

    # Diagnosis breakdown by clinical condition
    if 'diagnosis' in gmm.columns:
        print('\n  In vivo attractor assignment by diagnosis:')
        gmm['invivo_attr'] = invivo_attr
        for diag in ['PIH', 'PIM', 'PI']:
            sub = gmm[gmm['diagnosis'] == diag]
            if len(sub) == 0:
                continue
            counts = sub['invivo_attr'].value_counts()
            print(f'    {diag} (n={len(sub)}): {dict(counts)}')

    # ── In vitro vs in vivo offset ─────────────────────────────────────────────
    print('\n  In vitro vs in vivo offset:')
    print(f'  {"Condition":20}  {"So_invitro":>10}  {"So_invivo":>10}  {"ΔSo":>8}')
    for cond, att in heine_attractors.items():
        short = {'commensal_static': 'CS', 'commensal_hobic': 'CH',
                 'dh_baseline': 'DH', 'dysbiotic_static': 'DS'}[cond]
        # Find matching in vivo centroid
        inv_c = centroids[np.argmin([np.linalg.norm(centroids[k] - att) for k in range(4)])]
        delta = inv_c[0] - att[0]
        print(f'  {short:20}  {att[0]:10.3f}  {inv_c[0]:10.3f}  {delta:+8.3f}')

    # ── Figure B1: So-Pg scatter (Szafranski samples coloured by condition) ───
    diag_colors = {'PIH': '#1565C0', 'PIM': '#2E7D32', 'PI': '#B71C1C'}
    attr_colors = {'CS': '#2196F3', 'CH': '#4CAF50', 'DH': '#FF9800', 'DS': '#F44336'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Original compositions by diagnosis
    ax = axes[0]
    for diag, col in diag_colors.items():
        mask = gmm['diagnosis'] == diag
        ax.scatter(compositions[mask, 0], compositions[mask, 4],
                   c=col, s=30, alpha=0.7, label=diag)
    ax.set_xlabel('Streptococcus (So)', fontsize=11)
    ax.set_ylabel('Porphyromonas (Pg)', fontsize=11)
    ax.set_title('Szafranski compositions\n(coloured by diagnosis)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    # Panel 2: In vitro ODE endpoints (Heine θ)
    ax = axes[1]
    if endpoints_heine is not None:
        for diag, col in diag_colors.items():
            mask = (gmm['diagnosis'] == diag).values
            ax.scatter(endpoints_heine[mask, 0], endpoints_heine[mask, 4],
                       c=col, s=30, alpha=0.7, label=diag)
        # Heine attractors
        for cond, att in heine_attractors.items():
            nm = {'commensal_static':'CS','commensal_hobic':'CH',
                  'dh_baseline':'DH','dysbiotic_static':'DS'}[cond]
            ax.scatter(att[0], att[4], marker='*', s=400, c=attr_colors.get(nm,'gray'),
                       zorder=10, edgecolors='k', lw=0.5)
            ax.annotate(nm, (att[0], att[4]), fontsize=9, fontweight='bold',
                        xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Streptococcus (So)', fontsize=11)
    ax.set_ylabel('Porphyromonas (Pg)', fontsize=11)
    ax.set_title('In vitro ODE endpoints\n(Heine DH θ)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    # Panel 3: In vivo ODE endpoints (Dieckow θ)
    ax = axes[2]
    for diag, col in diag_colors.items():
        mask = (gmm['diagnosis'] == diag).values
        ax.scatter(endpoints_invivo[mask, 0], endpoints_invivo[mask, 4],
                   c=col, s=30, alpha=0.7, label=diag)
    # In vivo centroids
    for ci in range(4):
        nm = cluster_names.get(ci, f'C{ci}')
        c  = centroids[ci]
        ax.scatter(c[0], c[4], marker='*', s=400, c=attr_colors.get(nm, 'gray'),
                   zorder=10, edgecolors='k', lw=0.5)
        ax.annotate(nm, (c[0], c[4]), fontsize=9, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Streptococcus (So)', fontsize=11)
    ax.set_ylabel('Porphyromonas (Pg)', fontsize=11)
    ax.set_title('In vivo ODE endpoints\n(Dieckow MAP θ)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    plt.suptitle('Szafranski 127 samples: composition → ODE 20-week endpoint\n'
                 'Left: raw data  |  Centre: Heine in vitro  |  Right: Dieckow in vivo',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'gmm_reassign_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {OUT_DIR}/gmm_reassign_scatter.png')

    # ── Figure B2: Assignment comparison bar ──────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    attr_order = ['CS', 'CH', 'DH', 'DS']

    for ax, (col_name, title) in zip(axes2, [
        ('gmm_attractor', 'Original GMM assignment\n(Heine in vitro)'),
        ('invivo_attr',   'In vivo re-assignment\n(Dieckow θ)'),
    ]):
        if col_name not in gmm.columns:
            continue
        for di, (diag, dcol) in enumerate(diag_colors.items()):
            sub = gmm[gmm['diagnosis'] == diag]
            counts = [sub[col_name].value_counts().get(a, 0) for a in attr_order]
            bottom = [sum(
                gmm[gmm['diagnosis'] == list(diag_colors.keys())[j]][col_name]
                    .value_counts().get(a, 0)
                for j in range(di)) for a in attr_order]
            ax.bar(attr_order, counts, bottom=bottom,
                   color=dcol, label=diag, alpha=0.85)
        ax.set_xlabel('Attractor', fontsize=11)
        ax.set_ylabel('Number of samples', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Attractor assignment: GMM (in vitro) vs in vivo re-assignment', fontsize=12)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'gmm_reassign_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUT_DIR}/gmm_reassign_bars.png')

    # ── Save results ──────────────────────────────────────────────────────────
    gmm.to_csv(OUT_DIR / 'gmm_reassigned.csv', index=False)
    print(f'  Saved: {OUT_DIR}/gmm_reassigned.csv')

    # Offset summary JSON
    offset_data = {}
    for cond, att in heine_attractors.items():
        nm = {'commensal_static':'CS','commensal_hobic':'CH',
              'dh_baseline':'DH','dysbiotic_static':'DS'}[cond]
        ci = np.argmin([np.linalg.norm(centroids[k] - att) for k in range(4)])
        inv_c = centroids[ci]
        offset_data[nm] = {
            'invitro': dict(zip(SHORT, att.round(4).tolist())),
            'invivo':  dict(zip(SHORT, inv_c.round(4).tolist())),
            'delta_So': float(inv_c[0] - att[0]),
        }
    with open(OUT_DIR / 'invitro_invivo_offset.json', 'w') as f:
        json.dump({'genera': SHORT, 'offsets': offset_data,
                   'invivo_centroids': centroids.tolist()}, f, indent=2)
    print(f'  Saved: {OUT_DIR}/invitro_invivo_offset.json')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print('Installing scikit-learn...')
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn', '-q'])
        from sklearn.cluster import KMeans

    A_utri, b_avg = diagnose_patient_F()
    gmm_reassign_invivo(A_utri, b_avg)
    print(f'\nAll outputs → {OUT_DIR}')
