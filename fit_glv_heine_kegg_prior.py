#!/usr/bin/env python3
"""
gLV fit to Heine 2025 4-condition data with KEGG/HMDB-weighted sign prior
from Dieckow Supplementary File 1 (same prior as guild gLV).

5 focal species: So, An, Vd/Vp, Fn, Pg.

Outputs:
  results/heine2025/fit_glv_heine_kegg_prior.json
  docs/figures/dieckow/fig_heine_kegg_sign_comparison.pdf/.png
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_here = Path(__file__).parent
SUPPFILE = (_here / 'Szafranski_Published_Work' / 'Szafranski_Published_Work'
            / 'public_data' / 'Dieckow'
            / 'Supplementary_File_1_microbe_metabolite_enzyme_interactions.tsv')
DATA_CSV = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data'
                '/fig3_species_distribution_replicates.csv')
FLAT_JSON = _here / 'results' / 'heine2025' / 'fit_glv_heine.json'
OUT_JSON  = _here / 'results' / 'heine2025' / 'fit_glv_heine_kegg_prior.json'
FIG_DIR   = _here.parent / 'docs' / 'figures' / 'dieckow'

SPECIES = ['So', 'An', 'Vd', 'Fn', 'Pg']
N_SP    = 5
SIGMA   = 0.15   # same as guild gLV
N_STARTS = 5
DAYS    = [1, 3, 6, 10, 15, 21]

CONDITIONS = [
    ('Commensal', 'Static',  'CS'),
    ('Commensal', 'HOBIC',   'CH'),
    ('Dysbiotic', 'Static',  'DS'),
    ('Dysbiotic', 'HOBIC',   'DH'),
]
# Heine species names per condition type
SPECIES_NAMES = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',  'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic':  ['S. oralis', 'A. naeslundii', 'V. parvula', 'F. nucleatum', 'P. gingivalis_W83'],
}

# Genus → species index (0=So,1=An,2=Vd,3=Fn,4=Pg)
GENUS_SP = {
    'Streptococcus': 0, 'Schaalia': 0,       # broad Streptococcus/Actinomyces group → So/An
    'Actinomyces': 1,
    'Veillonella': 2, 'Lancefieldella': 2, 'Selenomonas': 2,
    'Fusobacterium': 3, 'Leptotrichia': 3,
    'Porphyromonas': 4, 'Prevotella': 4, 'Tannerella': 4,
}


def build_net_flow_5sp() -> np.ndarray:
    """Build 5×5 KEGG/HMDB-weighted net-flow matrix from Dieckow SF1."""
    df = pd.read_csv(SUPPFILE, sep='\t')

    def met_weight(row):
        kegg = str(row.get('KEGG', ''))
        hmdb = str(row.get('HMDB_ID', ''))
        if kegg not in ('n/a', '', 'nan', 'NaN'):
            return 2.0
        if 'HMDB' in hmdb:
            return 2.0
        return 1.0

    pos = np.zeros((N_SP, N_SP))
    neg = np.zeros((N_SP, N_SP))
    for met in df['OBJECT'].unique():
        mdf = df[df['OBJECT'] == met]
        w = float(mdf.apply(met_weight, axis=1).max())
        prod, cons, inhib = set(), set(), set()
        for _, row in mdf.iterrows():
            genus = str(row['TAXON']).split()[0]
            idx = GENUS_SP.get(genus)
            if idx is None:
                continue
            rel = row['RELATIONSHIP']
            if rel == 'PRODUCES':
                prod.add(idx)
            elif rel == 'USES':
                cons.add(idx)
            elif rel == 'IS_INHIBITED_BY':
                inhib.add(idx)
        for src in prod:
            for tgt in cons:
                if src != tgt:
                    pos[tgt, src] += w
            for tgt in inhib:
                if src != tgt:
                    neg[tgt, src] += w
    # eHOMD/literature supplement for 4 pairs not covered by Dieckow SF1
    ehomD_w = 1.0  # eHOMD-only weight (vs KEGG/HMDB = 2.0)
    ehomD_pos = [
        (0, 1),  # So-An: lactate/nitrite cross-feeding (Dieckow SI Neo4j)
        (1, 3),  # An-Fn: early colonizer bridge synergy (eHOMD)
        (2, 3),  # Vd-Fn: metabolite cross-feeding (eHOMD)
        (3, 4),  # Fn-Pg: coaggregation + cross-feeding (Kapatral 2002, Periasamy 2011)
    ]
    for i, j in ehomD_pos:
        pos[j, i] += ehomD_w
        pos[i, j] += ehomD_w
    return pos - neg


def replicator_rhs(t, phi, A, b):
    phi = np.maximum(phi, 1e-10)
    phi = phi / phi.sum()
    f = A @ phi + b
    return phi * (f - (phi @ f))


def integrate_glv(A, b, phi0, days):
    sol = solve_ivp(replicator_rhs, [days[0], days[-1]], phi0,
                    t_eval=days, args=(A, b), method='RK45',
                    rtol=1e-6, atol=1e-9, max_step=0.5)
    traj = np.maximum(sol.y.T, 0)
    s = traj.sum(axis=1, keepdims=True)
    return np.where(s > 0, traj / s, traj)


def sign_penalty(A, net_flow):
    pen = 0.0
    for i in range(N_SP):
        for j in range(N_SP):
            if i == j:
                continue
            f = net_flow[i, j]
            if f == 0:
                continue
            w = abs(f)
            v = max(0.0, -np.sign(f) * A[i, j])
            pen += w * v * v / (2 * SIGMA * SIGMA)
    return pen


def pack_A(A):
    return A.ravel()


def unpack_A(x):
    return x.reshape(N_SP, N_SP)


def fit_condition(phi_obs, net_flow, A_init=None, b_init=None):
    """phi_obs: (T, N_SP) with T = len(DAYS)."""
    days = np.array(DAYS, dtype=float)

    def obj(x):
        A = unpack_A(x[:N_SP * N_SP])
        b = x[N_SP * N_SP:]
        # force diagonal ≤ 0
        A_use = A.copy()
        np.fill_diagonal(A_use, np.minimum(np.diag(A_use), 0))
        try:
            traj = integrate_glv(A_use, b, phi_obs[0], days)
        except Exception:
            return 1e6
        rmse = np.sqrt(np.mean((traj[1:] - phi_obs[1:]) ** 2))
        pen  = sign_penalty(A_use, net_flow)
        return rmse + pen

    rng = np.random.default_rng(42)
    best_x, best_val = None, np.inf
    for s in range(N_STARTS):
        if A_init is not None and s == 0:
            A0 = A_init.copy()
        else:
            A0 = rng.normal(0, 0.1, (N_SP, N_SP))
            np.fill_diagonal(A0, -0.1)
        b0 = b_init.copy() if (b_init is not None and s == 0) else rng.normal(0, 0.1, N_SP)
        x0 = np.concatenate([A0.ravel(), b0])
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-7})
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x

    A_opt = unpack_A(best_x[:N_SP * N_SP])
    np.fill_diagonal(A_opt, np.minimum(np.diag(A_opt), 0))
    b_opt = best_x[N_SP * N_SP:]
    traj  = integrate_glv(A_opt, b_opt, phi_obs[0], np.array(DAYS, dtype=float))
    rmse  = float(np.sqrt(np.mean((traj[1:] - phi_obs[1:]) ** 2)))
    return A_opt, b_opt, rmse


def load_heine_data():
    df = pd.read_csv(DATA_CSV)
    data = {}
    for cond_type, flow_type, label in CONDITIONS:
        sp_names = SPECIES_NAMES[cond_type]
        sub = df[(df['condition'] == cond_type) & (df['cultivation'] == flow_type)]
        phi = []
        for day in DAYS:
            rows = sub[sub['day'] == day]
            vals = []
            for sp in sp_names:
                sp_rows = rows[rows['species'] == sp]['distribution_pct']
                vals.append(float(sp_rows.median()) if len(sp_rows) > 0 else 0.0)
            vals = np.array(vals, dtype=float) / 100.0  # pct → fraction
            vals = np.maximum(vals, 0)
            s = vals.sum()
            vals = vals / s if s > 0 else np.ones(N_SP) / N_SP
            phi.append(vals)
        data[label] = np.array(phi)  # (6, 5)
    return data


def sign_agree(A, net_flow, tol=0.02):
    agree = total = 0
    for i in range(N_SP):
        for j in range(N_SP):
            if i == j:
                continue
            f = net_flow[i, j]
            if f == 0 or abs(A[i, j]) < tol:
                continue
            total += 1
            if np.sign(f) == np.sign(A[i, j]):
                agree += 1
    return agree, total


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    print('Building 5-species KEGG/HMDB net-flow matrix ...')
    net_flow = build_net_flow_5sp()
    nz = int((net_flow != 0).sum())
    print(f'  Non-zero off-diagonal pairs: {nz}, max |flow|={np.abs(net_flow).max():.1f}')
    print('  Net-flow matrix (rows=tgt, cols=src):')
    print(pd.DataFrame(net_flow, index=SPECIES, columns=SPECIES).to_string())

    print('\nLoading Heine data ...')
    data = load_heine_data()

    # Load flat-prior warm-start
    flat = json.load(open(FLAT_JSON)) if FLAT_JSON.exists() else {}

    results = {}
    for cond_type, flow_type, label in CONDITIONS:
        phi_obs = data[label]
        print(f'\n--- {label} ({cond_type}/{flow_type}) ---')

        # warm-start from flat-prior fit
        A_init, b_init = None, None
        if label in flat and 'A' in flat[label]:
            A_init = np.array(flat[label]['A'])
            b_init = np.array(flat[label]['b'])

        t0 = time.time()
        A_opt, b_opt, rmse = fit_condition(phi_obs, net_flow, A_init, b_init)
        dt = time.time() - t0

        agree, total = sign_agree(A_opt, net_flow)
        pct = 100 * agree / total if total > 0 else 0
        flat_rmse = flat.get(label, {}).get('rmse', float('nan'))
        print(f'  RMSE (prior): {rmse:.4f}  (flat: {flat_rmse:.4f})  '
              f'sign agree: {agree}/{total} ({pct:.0f}%)  [{dt:.1f}s]')

        results[label] = {
            'rmse': rmse,
            'flat_rmse': flat_rmse,
            'sign_agree': agree,
            'sign_total': total,
            'sign_pct': round(pct, 1),
            'A': A_opt.tolist(),
            'b': b_opt.tolist(),
        }

    # Save JSON
    out = {
        'net_flow': net_flow.tolist(),
        'species': SPECIES,
        'sigma': SIGMA,
        'model': 'gLV Heine 5-sp KEGG/HMDB-prior (Dieckow SF1)',
        'conditions': results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, 'w'), indent=2)
    print(f'\nSaved: {OUT_JSON}')

    # Figure: A matrix comparison (flat vs KEGG-prior) for all 4 conditions
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    vmax = 0.4
    cmap = plt.cm.RdBu_r

    for col, (cond_type, flow_type, label) in enumerate(CONDITIONS):
        A_flat  = np.array(flat[label]['A']) if label in flat else np.zeros((N_SP, N_SP))
        A_prior = np.array(results[label]['A'])
        for row_idx, (A, title) in enumerate([(A_flat, f'{label} flat'), (A_prior, f'{label} KEGG-prior')]):
            ax = axes[row_idx, col]
            im = ax.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_xticks(range(N_SP)); ax.set_xticklabels(SPECIES, fontsize=7)
            ax.set_yticks(range(N_SP)); ax.set_yticklabels(SPECIES, fontsize=7)
            rmse_val = flat[label]['rmse'] if row_idx == 0 else results[label]['rmse']
            ax.set_title(f'{title}\nRMSE={rmse_val:.4f}', fontsize=7.5)
            for i in range(N_SP):
                for j in range(N_SP):
                    ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                            fontsize=5.5, color='k' if abs(A[i,j]) < vmax * 0.7 else 'w')

    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02,
                 label='A[i,j]')
    fig.suptitle('Heine 5-species gLV: flat prior (top) vs KEGG/HMDB prior (bottom)', fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.97, 1])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'fig_heine_kegg_sign_comparison.{ext}', dpi=300, bbox_inches='tight')
    print(f'Figure saved to {FIG_DIR}/fig_heine_kegg_sign_comparison.*')


if __name__ == '__main__':
    main()
