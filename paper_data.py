"""paper_data.py — 論文用データセットの正典（single source of truth）.

「どの run が論文用か」を毎回確認しなくて済むよう、ここに一元化する。
論文用フィットを使うスクリプトは、パスをハードコードせず必ずここを参照すること:

    from paper_data import PAPER_5SP_DIR, PAPER_5SP_STATES, paper_5sp_samples, paper_5sp_theta
    s = paper_5sp_samples('DH')          # (10000, 20) posterior samples
    th = paper_5sp_theta('CS')           # MAP theta_full (20,)

新しい論文用 run に差し替えるときは **このファイルの定数だけ**を直せばよい。
"""
from pathlib import Path
import json
import numpy as np

_here = Path(__file__).resolve().parent

# ── Heine 5-species アトラクター（CS/CH/DS/DH）── 論文用 = 10000-particle TMCMC
#
# 検証済み（2026-06-03）: これは論文の **Phase 2**（N_p=10000, free-ψ, full joint posterior;
# nishioka_heine_paper.tex §"Phase 2"）の posterior。論文図スクリプト plot_kegg_sign_comparison.py
# が読む P10K_RUNS = /home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs/{cs,ch,ds,dh}_10000p/samples.npy
# と byte-identical（md5 一致）であることを確認済み。ultimate_10000p はその repo 内コピー。
PAPER_5SP_DIR = _here / 'results' / 'ultimate_10000p'
PAPER_5SP_STATES = {        # 状態コード → サブディレクトリ名（DH は dh_baseline）
    'CS': 'commensal_static',
    'CH': 'commensal_hobic',
    'DS': 'dysbiotic_static',
    'DH': 'dh_baseline',
}
PAPER_5SP_SAMPLES = 10000   # particles per state


def paper_5sp_samples(state):
    """論文用 posterior samples (n, 20) for a state code in PAPER_5SP_STATES."""
    return np.load(PAPER_5SP_DIR / PAPER_5SP_STATES[state] / 'samples.npy')


def paper_5sp_theta(state):
    """論文用 MAP theta (20,) for a state code.
    10000p は {'0':v,..,'19':v} 形式、旧 run は {'theta_full':[..]} 形式の両対応。"""
    p = PAPER_5SP_DIR / PAPER_5SP_STATES[state] / 'theta_MAP.json'
    d = json.load(open(p))
    if 'theta_full' in d or 'theta_sub' in d:
        return np.array(d.get('theta_full', d.get('theta_sub')))
    # dict of index->value
    return np.array([d[str(i)] for i in range(len(d))])


if __name__ == '__main__':
    # 健全性チェック: python paper_data.py
    print('PAPER 5-species canonical run:', PAPER_5SP_DIR)
    for tag in PAPER_5SP_STATES:
        s = paper_5sp_samples(tag)
        print('  %-3s %-18s samples %s  theta_MAP %s' %
              (tag, PAPER_5SP_STATES[tag], s.shape, paper_5sp_theta(tag).shape))
