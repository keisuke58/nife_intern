"""Symbolic regression: discover which metabolic features explain A_ij.

Uses the fitted gLV interaction matrix A (10×10) as the target and
metabolic network features (KEGG/AGORA cross-feeding flows, guild identity)
as predictors. Attempts gplearn SymbolicRegressor if available, otherwise
falls back to polynomial ElasticNet.

Key question: A_ij = f(net_flow[i,j], guild_i, guild_j, …)?

Usage:
    python scripts/analysis/symbolic_regression_A.py [--model glv|hamilton]

Outputs:
    results/symbolic_regression/results.json
    results/symbolic_regression/fig_symbolic_regression.pdf
"""
# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

from guild_replicator_dieckow import GUILD_ORDER, GUILD_SHORT, N_G
GUILD_LABELS = [GUILD_SHORT.get(g, g) for g in GUILD_ORDER]
from loo_cv_kegg_prior import build_net_flow
from pub_style import apply as apply_pub_style

_here = Path(__file__).resolve().parents[2]
FIT_FILES = {
    'glv':      _here / 'results' / 'dieckow_cr' / 'fit_glv_8pat_kegg_prior.json',
    'hamilton': _here / 'results' / 'dieckow_cr' / 'fit_glv_hamilton_kegg_prior.json',
}
OUT_DIR = _here / 'results' / 'symbolic_regression'

# ── guild-level metabolic traits (oxygen tolerance, gram stain, broad role) ──
# 0=anaerobic, 1=facultative, 2=aerobic
OXYGEN = {
    'Actinobacteria': 1, 'Bacilli': 1, 'Bacteroidia': 0,
    'Betaproteobacteria': 1, 'Clostridia': 0, 'Coriobacteriia': 0,
    'Fusobacteriia': 0, 'Gammaproteobacteria': 1, 'Negativicutes': 0, 'Other': 1,
}
# 0=gram-neg, 1=gram-pos
GRAM = {
    'Actinobacteria': 1, 'Bacilli': 1, 'Bacteroidia': 0,
    'Betaproteobacteria': 0, 'Clostridia': 1, 'Coriobacteriia': 1,
    'Fusobacteriia': 0, 'Gammaproteobacteria': 0, 'Negativicutes': 0, 'Other': 0,
}

OXY_VEC  = np.array([OXYGEN[g]  for g in GUILD_ORDER], dtype=float)
GRAM_VEC = np.array([GRAM[g]    for g in GUILD_ORDER], dtype=float)


def build_features(net_flow):
    """Build feature matrix X (N_G^2, n_features) and target y (N_G^2,).

    Features per (i,j) pair:
      0: net_flow[i,j]            (KEGG/AGORA signed flow)
      1: |net_flow[i,j]|          (flow magnitude)
      2: net_flow[j,i]            (reverse flow)
      3: oxy_i, gram_i            (guild i traits)
      4: oxy_j, gram_j            (guild j traits)
      5: oxy_i * oxy_j            (interaction of traits)
      6: gram_i * gram_j
      7: i == j (diagonal flag)
    """
    rows, feat_names = [], []
    for i in range(N_G):
        for j in range(N_G):
            f = [
                net_flow[i, j],
                abs(net_flow[i, j]),
                net_flow[j, i],
                OXY_VEC[i],
                GRAM_VEC[i],
                OXY_VEC[j],
                GRAM_VEC[j],
                OXY_VEC[i] * OXY_VEC[j],
                GRAM_VEC[i] * GRAM_VEC[j],
                float(i == j),
            ]
            rows.append(f)

    feat_names = [
        'flow_ij', '|flow_ij|', 'flow_ji',
        'oxy_i', 'gram_i', 'oxy_j', 'gram_j',
        'oxy_i*oxy_j', 'gram_i*gram_j', 'self',
    ]
    return np.array(rows), feat_names


def try_gplearn(X, y, feat_names):
    """Attempt gplearn symbolic regression. Returns (program_str, r2, ok)."""
    try:
        from gplearn.genetic import SymbolicRegressor
        est = SymbolicRegressor(
            population_size=500,
            generations=30,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            parsimony_coefficient=0.01,
            random_state=42,
        )
        est.fit(X, y)
        y_pred = est.predict(X)
        r2 = r2_score(y, y_pred)
        return str(est._program), r2, True
    except ImportError:
        return None, None, False


def polynomial_regression(X, y, degree=2):
    """Polynomial ElasticNet regression as fallback."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    Xp = scaler.fit_transform(poly.fit_transform(X))

    loo = LeaveOneOut()
    loo_preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(Xp):
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xp[train_idx], y[train_idx])
        loo_preds[test_idx] = ridge.predict(Xp[test_idx])
    loo_r2 = r2_score(y, loo_preds)

    # Full fit for coefficients
    net = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], max_iter=5000)
    net.fit(Xp, y)
    full_r2 = net.score(Xp, y)

    poly_names = poly.get_feature_names_out()
    coef_order = np.argsort(np.abs(net.coef_))[::-1][:10]
    top_terms = [(poly_names[k], float(net.coef_[k])) for k in coef_order]

    return loo_r2, full_r2, top_terms, net.predict(Xp)


def sign_agreement(A_flat, net_flow_flat):
    mask = net_flow_flat != 0
    if mask.sum() == 0:
        return 0.0
    agree = (np.sign(A_flat[mask]) == np.sign(net_flow_flat[mask])).mean()
    return float(agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['glv', 'hamilton'], default='hamilton')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load fitted A ──
    with open(FIT_FILES[args.model]) as f:
        fit = json.load(f)
    A = np.array(fit['A'])

    # ── Build metabolic sign-prior (net flow) ──
    print('Building net flow from Szafranski data...')
    net_flow = build_net_flow()

    # ── Build features ──
    X, feat_names = build_features(net_flow)
    y = A.flatten()

    # Exclude diagonal for off-diagonal analysis
    off_diag = ~np.eye(N_G, dtype=bool).flatten()
    X_off = X[off_diag]
    y_off = y[off_diag]

    print(f'Dataset: {X_off.shape[0]} guild pairs, {X_off.shape[1]} features')

    # ── Sign agreement (baseline) ──
    flow_flat = net_flow.flatten()
    sa_all    = sign_agreement(y, flow_flat)
    sa_off    = sign_agreement(y_off, flow_flat[off_diag])
    print(f'Sign agreement (off-diag): {sa_off:.3f}')

    # ── Try gplearn symbolic regression ──
    print('Attempting gplearn symbolic regression...')
    sym_str, sym_r2, sym_ok = try_gplearn(X_off, y_off, feat_names)
    if sym_ok:
        print(f'gplearn R²={sym_r2:.3f}  expression: {sym_str[:100]}')
    else:
        print('gplearn not available — using polynomial ElasticNet fallback')

    # ── Polynomial regression ──
    print('Running polynomial ElasticNet regression...')
    loo_r2, full_r2, top_terms, y_pred = polynomial_regression(X_off, y_off)
    print(f'Polynomial LOO-R²={loo_r2:.3f}  full-R²={full_r2:.3f}')
    print('Top regression terms:')
    for name, coef in top_terms[:5]:
        print(f'  {name:30s}  coef={coef:+.4f}')

    # ── Figure ──
    apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: A_ij vs net_flow_ij scatter
    ax = axes[0]
    ax.scatter(net_flow[~np.eye(N_G, dtype=bool)],
               A[~np.eye(N_G, dtype=bool)],
               s=30, alpha=0.7, color='steelblue')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('net_flow[i,j]  (metabolic signal)')
    ax.set_ylabel('A[i,j]  (fitted interaction)')
    ax.set_title(f'Sign agreement = {sa_off:.2f}')

    # Panel B: Predicted vs actual from polynomial regression
    ax = axes[1]
    ax.scatter(y_pred, y_off, s=30, alpha=0.7, color='tomato')
    lim = max(abs(y_off).max(), abs(y_pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.set_xlabel('Predicted A[i,j]')
    ax.set_ylabel('Observed A[i,j]')
    ax.set_title(f'Poly regression (LOO-R²={loo_r2:.2f})')

    # Panel C: A matrix heatmap with sign-prior overlay
    ax = axes[2]
    vmax = np.abs(A).max()
    im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    # overlay sign-prior arrows
    sp_sign = np.sign(net_flow)
    for i in range(N_G):
        for j in range(N_G):
            if sp_sign[i, j] != 0 and i != j:
                marker = '+' if sp_sign[i, j] > 0 else '×'
                color = 'lime' if np.sign(A[i, j]) == sp_sign[i, j] else 'red'
                ax.text(j, i, marker, ha='center', va='center',
                        fontsize=6, color=color, fontweight='bold')
    ax.set_xticks(range(N_G)); ax.set_xticklabels(GUILD_LABELS, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(N_G)); ax.set_yticklabels(GUILD_LABELS, fontsize=6)
    ax.set_title('A matrix (+/× = prior agree/disagree)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_symbolic_regression.pdf')
    print(f'Figure: {OUT_DIR}/fig_symbolic_regression.pdf')

    # ── Save results ──
    results = {
        'model':         args.model,
        'sign_agreement_off_diag': sa_off,
        'sym_regression': {
            'available': sym_ok,
            'r2':        sym_r2,
            'expression': sym_str,
        },
        'poly_regression': {
            'loo_r2':   loo_r2,
            'full_r2':  full_r2,
            'top_terms': top_terms,
        },
        'A': A.tolist(),
        'net_flow': net_flow.tolist(),
        'guilds': GUILD_ORDER,
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {OUT_DIR}/results.json')


if __name__ == '__main__':
    main()
