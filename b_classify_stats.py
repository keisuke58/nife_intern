#!/usr/bin/env python3
"""
b_classify_stats.py — Statistical testing + diagnostic classification from b̂

1. Mann-Whitney U + rank-biserial r for each species × diagnosis pair
2. Logistic regression + SVM (5-fold CV), AUROC (one-vs-rest)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve

SHORT   = ['So', 'An', 'Vd', 'Fn', 'Pg']
DIAGS   = ['PIH', 'PIM', 'PI']
COLORS  = {'PIH': '#1565C0', 'PIM': '#2E7D32', 'PI': '#B71C1C'}
B_DIR   = Path('/home/nishioka/IKM_Hiwi/nife/results/b_szafranski')
GMM_CSV = Path('/home/nishioka/IKM_Hiwi/nife/results/gmm_attractor_analysis.csv')
OUT_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/b_classify_stats')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rank_biserial(x, y):
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    n1, n2 = len(x), len(y)
    u, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    return 1 - 2 * u / (n1 * n2)


# ── 1. Mann-Whitney U ─────────────────────────────────────────────────────────
def stat_tests(b_hat, diag):
    pairs = [('PIH', 'PI'), ('PIM', 'PI'), ('PIH', 'PIM')]
    rows  = []
    for sp_i, sp in enumerate(SHORT):
        for d1, d2 in pairs:
            x = b_hat[diag == d1, sp_i]
            y = b_hat[diag == d2, sp_i]
            _, p = stats.mannwhitneyu(x, y, alternative='two-sided')
            r   = rank_biserial(x, y)
            rows.append({'species': sp, 'group1': d1, 'group2': d2,
                         'p_value': p, 'r_biserial': r,
                         'p_bonf': min(p * len(SHORT) * len(pairs), 1.0)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'mannwhitney.csv', index=False)
    print('\n── Mann-Whitney U (Bonferroni corrected) ──')
    sig = df[df['p_bonf'] < 0.05].copy()
    for _, row in sig.iterrows():
        print(f"  {row['species']} {row['group1']} vs {row['group2']}: "
              f"p={row['p_value']:.4f} (p_bonf={row['p_bonf']:.4f}), r={row['r_biserial']:.3f}")
    if sig.empty:
        print('  (none significant after Bonferroni)')
    return df


# ── 2. Classification ─────────────────────────────────────────────────────────
def classify(b_hat, diag):
    X    = np.log(b_hat + 1e-6)
    y    = diag
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'LogReg': Pipeline([('sc', StandardScaler()),
                            ('clf', LogisticRegression(max_iter=1000, C=1.0))]),
        'SVM-RBF': Pipeline([('sc', StandardScaler()),
                             ('clf', SVC(kernel='rbf', C=1.0, probability=True))]),
    }

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, model) in zip(axes, models.items()):
        acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f'\n{name} 5-fold CV accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}')

        # AUROC one-vs-rest
        y_bin = label_binarize(y, classes=DIAGS)
        auc_scores = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            model.fit(X[tr], y[tr])
            y_prob = model.predict_proba(X[te])
            try:
                auc = roc_auc_score(y_bin[te], y_prob, multi_class='ovr',
                                    average='macro')
                auc_scores.append(auc)
            except ValueError:
                pass

        mean_auc = np.mean(auc_scores)
        print(f'{name} macro AUROC (OVR): {mean_auc:.3f}')
        results[name] = {'accuracy': float(acc_scores.mean()),
                         'accuracy_std': float(acc_scores.std()),
                         'auroc_macro': float(mean_auc)}

        # Plot ROC per class (fit on all data for display)
        model.fit(X, y)
        y_prob_all = model.predict_proba(X)
        for ci, (d, col) in enumerate(COLORS.items()):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], y_prob_all[:, ci])
            auc_c = roc_auc_score(y_bin[:, ci], y_prob_all[:, ci])
            ax.plot(fpr, tpr, color=col, lw=1.5, label=f'{d} AUC={auc_c:.2f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f'{name}\nAcc={acc_scores.mean():.2f}±{acc_scores.std():.2f}  '
                     f'AUROC={mean_auc:.2f}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('Diagnostic classification from b̂ (log b̂, 5-fold CV)', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/roc_curves.png')
    return results


# ── Plot: p-value heatmap ─────────────────────────────────────────────────────
def plot_pvalue_heatmap(df_stats):
    pairs  = [('PIH', 'PI'), ('PIM', 'PI'), ('PIH', 'PIM')]
    labels = [f'{a}\nvs\n{b}' for a, b in pairs]
    mat_p  = np.zeros((len(SHORT), len(pairs)))
    mat_r  = np.zeros((len(SHORT), len(pairs)))
    for si, sp in enumerate(SHORT):
        for pi, (d1, d2) in enumerate(pairs):
            row = df_stats[(df_stats.species == sp) &
                           (df_stats.group1 == d1) & (df_stats.group2 == d2)]
            mat_p[si, pi] = row['p_bonf'].values[0]
            mat_r[si, pi] = row['r_biserial'].values[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax1.imshow(-np.log10(mat_p + 1e-10), cmap='Reds', aspect='auto',
                     vmin=0, vmax=4)
    ax1.set_xticks(range(len(pairs))); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticks(range(len(SHORT))); ax1.set_yticklabels(SHORT)
    ax1.set_title('-log₁₀(p_bonf)')
    plt.colorbar(im1, ax=ax1)
    for si in range(len(SHORT)):
        for pi in range(len(pairs)):
            ax1.text(pi, si, f'{mat_p[si,pi]:.3f}', ha='center', va='center',
                     fontsize=7, color='white' if mat_p[si, pi] < 0.05 else 'black')

    im2 = ax2.imshow(mat_r, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(pairs))); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_yticks(range(len(SHORT))); ax2.set_yticklabels(SHORT)
    ax2.set_title('Rank-biserial r (effect size)')
    plt.colorbar(im2, ax=ax2)
    for si in range(len(SHORT)):
        for pi in range(len(pairs)):
            ax2.text(pi, si, f'{mat_r[si,pi]:.2f}', ha='center', va='center',
                     fontsize=8)

    plt.suptitle('b̂ group differences: Mann-Whitney U', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'stat_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUT_DIR}/stat_heatmap.png')


def main():
    b_hat = np.load(B_DIR / 'b_hat.npy')
    gmm   = pd.read_csv(GMM_CSV)
    diag  = gmm['diagnosis'].values

    print(f'Loaded b_hat: {b_hat.shape}, diagnoses: {dict(zip(*np.unique(diag, return_counts=True)))}')

    df_stats = stat_tests(b_hat, diag)
    plot_pvalue_heatmap(df_stats)

    clf_results = classify(b_hat, diag)

    summary = {
        'n_samples': int(len(diag)),
        'classification': clf_results,
        'significant_pairs_bonf005': df_stats[df_stats['p_bonf'] < 0.05][
            ['species', 'group1', 'group2', 'p_value', 'r_biserial']].to_dict('records'),
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nAll outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()
