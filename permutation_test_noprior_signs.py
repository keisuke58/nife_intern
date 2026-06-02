#!/usr/bin/env python3
"""
permutation_test_noprior_signs.py
  --no-prior（sign penalty を切った純データフィット）の LOO fit に対し、
  代謝 sign prior セルが「prior の予測符号」に有意に richer かを permutation test で検定する。

背景（ANALYSIS_NOTES §5）:
  alpha=0 の prior は cross-feeding のみで全て正なので、"sign agreement" は
  「フィット A の prior セルが正である割合」と同義。多数決ベースラインは 100% になり無意味。
  正しい検定は「prior セルが（A 全体の符号分布から見て）予測符号に偏っているか」の enrichment。

設計:
  - fold 非独立性に対処するため、10 fold を **per-cell consensus 符号** に畳む
    （cons = sign(mean_folds A_c)）。検定単位は fold ではなく行列セル。
  - gLV は非対称 → off-diag 90 セルを candidate。Hamilton は対称 → 上三角 off-diag 45 セルを candidate。
  - observed = consensus 符号が prior 符号に一致する割合（prior マスク上）。
  - null = consensus 符号を candidate セル間でシャッフル（= prior 位置をランダム再配置）→ 一致割合。
  - p = P(perm 一致割合 >= observed)。
  - alpha を 0→1 に振り、prior に負（競争）を入れて検定を対称化（予測競争が実際に負かも見る）。

repo root から実行:
    python permutation_test_noprior_signs.py
    python permutation_test_noprior_signs.py --nperm 20000 --alphas 0,0.25,0.5,1.0
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from build_net_flow_expanded import net_flow_glv, net_flow_hamilton

_here = Path(__file__).resolve().parent
RES = _here / 'results' / 'dieckow_cr'

MODELS = {
    'glv': {
        'glob': 'loo_glv_noprior_a0p0_fold*.json',
        'net_fn': net_flow_glv,
        'directed': True,
        'label': 'gLV (asymmetric A, run_glv_loo.py)',
    },
    'hamilton_expanded': {
        'glob': 'loo_expanded_noprior_a0p0_fold*.json',
        'net_fn': net_flow_hamilton,
        'directed': False,
        'label': 'Hamilton (symmetric A, run_hamilton_expanded_loo.py)',
    },
}


def load_consensus(glob_pat):
    """10 fold の A を読み、per-cell consensus 符号 sign(mean_folds A) を返す。"""
    files = sorted(glob.glob(str(RES / glob_pat)))
    if not files:
        raise FileNotFoundError(glob_pat)
    As = np.stack([np.array(json.load(open(f))['A']) for f in files])  # (n_fold,N,N)
    return np.sign(As.mean(axis=0)), As, files


def candidate_mask(N, directed):
    """permutation の候補セル集合（直接=全off-diag / 対称=上三角off-diag）。"""
    off = ~np.eye(N, dtype=bool)
    if directed:
        return off
    return np.triu(np.ones((N, N), dtype=bool), k=1)  # i<j


def run_model(key, cfg, alphas, nperm, rng):
    cons, As, files = load_consensus(cfg['glob'])
    N = cons.shape[0]
    cand = candidate_mask(N, cfg['directed'])
    cand_idx = np.argwhere(cand)                      # (n_cand,2)
    cons_cand = cons[cand]                            # consensus 符号（候補セル上）

    # pooled fold×cell 正割合（記述統計; ANALYSIS_NOTES の 72/41 を再現）
    pooled_pos_oncell = (np.sign(As) > 0)[:, cand]    # (n_fold, n_cand)

    out = {'label': cfg['label'], 'n_folds': len(files), 'files': [Path(f).name for f in files],
           'directed': cfg['directed'], 'n_candidate_cells': int(cand.sum()),
           'alpha_results': []}

    for a in alphas:
        prior = np.sign(np.asarray(cfg['net_fn'](use_agora=False, competition_weight=a)))
        pmask = (prior != 0) & cand                   # prior セル（候補内）
        pmask_in_cand = pmask[cand]                   # candidate ベクトル上の bool
        prior_cand = prior[cand]
        n_prior = int(pmask_in_cand.sum())
        if n_prior == 0:
            continue

        # observed consensus 一致率
        agree_cand = (cons_cand == prior_cand)
        obs = float(agree_cand[pmask_in_cand].mean())

        # 予測 +/- 別の一致率（検定の対称性チェック）
        pos_sel = pmask_in_cand & (prior_cand > 0)
        neg_sel = pmask_in_cand & (prior_cand < 0)
        obs_pos = float(agree_cand[pos_sel].mean()) if pos_sel.any() else None
        obs_neg = float(agree_cand[neg_sel].mean()) if neg_sel.any() else None

        # pooled fold×cell 正割合（prior 内 / prior 外）— 記述
        pooled_on = float(pooled_pos_oncell[:, pmask_in_cand].mean())
        pooled_off = float(pooled_pos_oncell[:, ~pmask_in_cand].mean())

        # permutation null: consensus 符号を候補セル間でシャッフル
        prior_sub = prior_cand[pmask_in_cand]
        ge = 0
        perm_means = np.empty(nperm)
        for i in range(nperm):
            perm = cons_cand[rng.permutation(cons_cand.size)]
            am = float((perm[pmask_in_cand] == prior_sub).mean())
            perm_means[i] = am
            if am >= obs - 1e-12:
                ge += 1
        pval = (ge + 1) / (nperm + 1)                 # add-one（保守的）

        out['alpha_results'].append({
            'alpha': a,
            'n_prior_cells': n_prior,
            'prior_pos': int((prior_cand[pmask_in_cand] > 0).sum()),
            'prior_neg': int((prior_cand[pmask_in_cand] < 0).sum()),
            'observed_agreement': obs,
            'agreement_on_predicted_pos': obs_pos,
            'agreement_on_predicted_neg': obs_neg,
            'pooled_pos_frac_on_prior': pooled_on,
            'pooled_pos_frac_off_prior': pooled_off,
            'perm_mean': float(perm_means.mean()),
            'perm_std': float(perm_means.std()),
            'p_value': pval,
            'z_score': float((obs - perm_means.mean()) / (perm_means.std() + 1e-12)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nperm', type=int, default=10000)
    ap.add_argument('--alphas', type=str, default='0,0.25,0.5,1.0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default=str(RES / 'noprior_permutation_test.json'))
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(',')]
    rng = np.random.default_rng(args.seed)

    results = {'nperm': args.nperm, 'seed': args.seed, 'alphas': alphas, 'models': {}}
    for key, cfg in MODELS.items():
        results['models'][key] = run_model(key, cfg, alphas, args.nperm, rng)

    # ── print summary ──
    for key, r in results['models'].items():
        print(f"\n=== {key}: {r['label']} ===")
        print(f"  candidate cells={r['n_candidate_cells']} (directed={r['directed']}), folds={r['n_folds']}")
        hdr = ('  alpha  n(+/-)     SA     +cells  -cells   pooled(on/off)   perm   p-value   z')
        print(hdr)
        for ar in r['alpha_results']:
            ap_ = '%.2f' % ar['agreement_on_predicted_pos'] if ar['agreement_on_predicted_pos'] is not None else '  - '
            an_ = '%.2f' % ar['agreement_on_predicted_neg'] if ar['agreement_on_predicted_neg'] is not None else '  - '
            print('  %4.2f  %2d(%2d/%2d)  %.3f   %s    %s    %.2f/%.2f      %.3f  %.4f  %+.2f'
                  % (ar['alpha'], ar['n_prior_cells'], ar['prior_pos'], ar['prior_neg'],
                     ar['observed_agreement'], ap_, an_,
                     ar['pooled_pos_frac_on_prior'], ar['pooled_pos_frac_off_prior'],
                     ar['perm_mean'], ar['p_value'], ar['z_score']))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
