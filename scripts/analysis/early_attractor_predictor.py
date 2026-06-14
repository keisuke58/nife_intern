"""Early attractor predictor for oral biofilm community dynamics.

Simulates 2000 gLV trajectories from random initial conditions using the
fitted Dieckow interaction matrix, clusters converged states into attractors,
then trains a Random Forest to predict attractor label from early (t=0,1)
community composition.

Usage:
    python scripts/analysis/early_attractor_predictor.py [--n-sim 2000] [--k 3]

Outputs:
    results/attractor_predictor/results.json
    results/attractor_predictor/fig_attractor_predictor.pdf
"""
# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from guild_replicator_dieckow import GUILD_ORDER, GUILD_SHORT, GUILD_COLORS, N_G
GUILD_LABELS = [GUILD_SHORT.get(g, g) for g in GUILD_ORDER]
from pub_style import apply as apply_pub_style

_here = Path(__file__).resolve().parents[2]
FIT_FILE = _here / 'results' / 'dieckow_cr' / 'fit_glv_hamilton_kegg_prior.json'
PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
DURANPINEDO_NPY = _here / 'data' / 'prjna725874' / 'phi_guild.npy'
OUT_DIR  = _here / 'results' / 'attractor_predictor'

T_CONV   = 150.0  # weeks to integrate for convergence
DT_EARLY = [0.0, 1.0]  # early timepoints used as features


def replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)


def integrate_to(phi0, b, A, t_end, t_eval=None):
    sol = solve_ivp(replicator_rhs, [0, t_end], phi0, args=(b, A),
                    method='RK45', rtol=1e-7, atol=1e-9,
                    t_eval=t_eval, dense_output=False)
    y = sol.y
    # renormalize
    norms = y.sum(axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return y / norms


def simulate_trajectories(A, b_mean, n_sim, rng):
    """Return (phi_early, phi_final) arrays of shape (n_sim, 2*N_G) and (n_sim, N_G)."""
    phi_early = np.zeros((n_sim, 2 * N_G))
    phi_final = np.zeros((n_sim, N_G))

    for i in range(n_sim):
        alpha = rng.exponential(1.0, N_G) + 0.01
        phi0 = alpha / alpha.sum()

        t_eval = np.array([0.0, 1.0, T_CONV])
        traj = integrate_to(phi0, b_mean, A, T_CONV, t_eval=t_eval)

        phi_early[i, :N_G]  = traj[:, 0]  # t=0
        phi_early[i, N_G:]  = traj[:, 1]  # t=1
        phi_final[i]        = traj[:, 2]  # converged

        if (i + 1) % 200 == 0:
            print(f'  simulated {i+1}/{n_sim}', flush=True)

    return phi_early, phi_final


def cluster_attractors(phi_final, k, rng):
    """KMeans-style clustering with multiple restarts."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_score, best_labels, best_centers = -1, None, None
    for _ in range(10):
        km = KMeans(n_clusters=k, n_init=1, random_state=rng.integers(9999))
        labels = km.fit_predict(phi_final)
        score = silhouette_score(phi_final, labels)
        if score > best_score:
            best_score, best_labels, best_centers = score, labels, km.cluster_centers_
    print(f'  k={k}  silhouette={best_score:.3f}')
    return best_labels, best_centers, best_score


def assign_real_patients(phi_final_real, centers):
    """Assign each real patient to nearest attractor centroid."""
    dists = cdist(phi_final_real, centers)
    return dists.argmin(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sim', type=int, default=2000)
    ap.add_argument('--k',     type=int, default=3)
    ap.add_argument('--seed',  type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load fitted gLV ──
    with open(FIT_FILE) as f:
        fit = json.load(f)
    A      = np.array(fit['A'])
    b_all  = np.array(fit['b_all'])
    b_mean = b_all.mean(axis=0)

    # ── Simulate trajectories ──
    print(f'Simulating {args.n_sim} trajectories (k={args.k})...')
    phi_early, phi_final = simulate_trajectories(A, b_mean, args.n_sim, rng)

    # ── Cluster into attractors ──
    print('Clustering converged states...')
    labels, centers, sil = cluster_attractors(phi_final, args.k, rng)

    # ── Train RF classifier ──
    print('Training Random Forest...')
    clf = RandomForestClassifier(n_estimators=300, random_state=args.seed)
    clf.fit(phi_early, labels)
    train_acc = accuracy_score(labels, clf.predict(phi_early))

    # LOO-CV on simulated data (sample 200 for speed)
    idx = rng.choice(args.n_sim, size=min(200, args.n_sim), replace=False)
    loo_preds = []
    for j, test_i in enumerate(idx):
        train_mask = np.ones(len(idx), dtype=bool); train_mask[j] = False
        train_idx  = idx[train_mask]
        c = RandomForestClassifier(n_estimators=100, random_state=args.seed)
        c.fit(phi_early[train_idx], labels[train_idx])
        loo_preds.append(c.predict(phi_early[[test_i]])[0])
    loo_acc = accuracy_score(labels[idx], loo_preds)

    # ── Project real Dieckow patients ──
    phi_dieckow = np.load(PHI_NPY)   # (10, 3, 10)
    phi_real_early = np.hstack([phi_dieckow[:, 0, :], phi_dieckow[:, 1, :]])
    phi_real_final = phi_dieckow[:, 2, :]
    real_pred  = clf.predict(phi_real_early)
    real_truth = assign_real_patients(phi_real_final, centers)
    real_acc   = accuracy_score(real_truth, real_pred)

    print(f'Train accuracy:     {train_acc:.3f}')
    print(f'LOO-CV accuracy:    {loo_acc:.3f}')
    print(f'Real patient match: {real_acc:.3f}  ({real_acc*10:.0f}/10 patients)')

    # ── Feature importance ──
    feat_names = (
        [f't0_{g}' for g in GUILD_SHORT] +
        [f't1_{g}' for g in GUILD_SHORT]
    )
    importance = clf.feature_importances_

    # ── PCA visualization ──
    apply_pub_style()
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(phi_final)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: PCA of simulated converged states
    ax = axes[0]
    cmap = plt.colormaps['tab10'].resampled(args.k)
    for c in range(args.k):
        mask = labels == c
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=4, alpha=0.3,
                   color=cmap(c), label=f'Attractor {c}')
    # overlay real patients
    pcs_real = pca.transform(phi_real_final)
    for i in range(len(phi_dieckow)):
        ax.scatter(pcs_real[i, 0], pcs_real[i, 1], s=60,
                   color=cmap(real_truth[i]), edgecolors='k', linewidths=0.8,
                   zorder=5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
    ax.set_title('Converged states (PCA)')
    ax.legend(fontsize=7, markerscale=3)

    # Panel B: Attractor centroids (guild composition)
    ax = axes[1]
    x = np.arange(N_G)
    width = 0.8 / args.k
    for c in range(args.k):
        ax.bar(x + c * width, centers[c], width=width,
               color=cmap(c), label=f'Attractor {c}', alpha=0.85)
    ax.set_xticks(x + width * (args.k - 1) / 2)
    ax.set_xticklabels(GUILD_LABELS, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean composition')
    ax.set_title('Attractor guild composition')
    ax.legend(fontsize=7)

    # Panel C: Feature importance (top 10)
    ax = axes[2]
    order = np.argsort(importance)[::-1][:10]
    ax.barh(range(10), importance[order][::-1],
            color='steelblue', alpha=0.8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([feat_names[o] for o in order[::-1]], fontsize=7)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top-10 features\nLOO acc={loo_acc:.2f}')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_attractor_predictor.pdf')
    print(f'Figure saved: {OUT_DIR}/fig_attractor_predictor.pdf')

    # ── Save JSON ──
    results = {
        'n_sim':       args.n_sim,
        'k':           args.k,
        'silhouette':  float(sil),
        'train_acc':   float(train_acc),
        'loo_acc':     float(loo_acc),
        'real_acc':    float(real_acc),
        'real_pred':   real_pred.tolist(),
        'real_truth':  real_truth.tolist(),
        'attractor_centers': centers.tolist(),
        'feature_importance': dict(zip(feat_names, importance.tolist())),
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {OUT_DIR}/results.json')


if __name__ == '__main__':
    main()
