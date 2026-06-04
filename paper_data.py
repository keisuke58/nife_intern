"""paper_data.py — 論文用データセットの正典（single source of truth）.

「どの run が論文用か」を毎回確認しなくて済むよう、ここに一元化する。
論文用フィットを使うスクリプトは、パスをハードコードせず必ずここを参照すること:

    from paper_data import PAPER_5SP_DIR, PAPER_5SP_STATES, paper_5sp_samples, paper_5sp_theta
    s = paper_5sp_samples('DH')          # (10000, 20) posterior samples
    th = paper_5sp_theta('CS')           # MAP theta_full (20,)

新しい論文用 run に差し替えるときは **このファイルの定数だけ**を直せばよい。

────────────────────────────────────────────────────────────────────────────────
5-species Heine モデル全体像（混同しやすいので整理）:

  ① gLV  (非対称 A, L-BFGS-B, 点推定)
       スクリプト : scripts/fitting/fit_glv_heine.py
       結果       : results/heine2025/fit_glv_heine.json  (A 5×5 asymm + b)
       図         : scripts/figures/plot_glv_heine_thesis.py  ← 修論用
       RMSE       : CS=0.032 / CH=0.026 / DS=0.012 / DH=0.022   ← 最良

  ② Hamilton TMCMC  (対称 A, TMCMC 10000粒子, Bayesian posterior; **このファイルが指すもの**)
       場所       : results/ultimate_10000p/{commensal_static,commensal_hobic,dysbiotic_static,dh_baseline}/
       theta エンコード: 20パラメータ (tools/verify_symmetric_A.py の theta_to_A 参照)
                     A[i,j]=theta[k] 対称、b = theta[3,4,8,9,15]
       ODE        : 単純 replicator (Hamilton, symmetric A), scipy solve_ivp
       RMSE       : CS=0.23 / CH=0.22 / DS=0.27 / DH=0.27   ← 対称制約のため高い
       使いどころ : sign prior 検証 (no-prior enrichment)、network 解析の A source

  ③ Hamilton NUTS  (対称 A, NUTS + JAX + Hill速度論, 最高精度)
       場所       : Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_*/
       図生成     : Tmcmc202601/docs/generate_fig2_phi_version.py
                   (キャッシュ Tmcmc202601/docs/paper_figures/_cache/fig2_phi_{CS/CH/DS/DH}.npz)
       RMSE       : CS=0.119 / CH=0.104 / DS=0.033 / DH=0.087
       使いどころ : Tmcmc202601 論文 Fig.2 (修論でも並列比較に使用)

  比較まとめ:
    フィット精度 : gLV (①) > Hamilton NUTS (③) > Hamilton TMCMC (②)
    不確かさ定量 : ②③ あり (posterior)、① なし (点推定)
    対称制約     : ②③ あり (物理的根拠)、① なし (表現力高)
────────────────────────────────────────────────────────────────────────────────
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
    """TMCMC posterior samples (n=10000, 20) — sign解析・network解析用。
    Hamilton NUTS (精度優先) は paper_hamilton_nuts_samples() を使うこと。"""
    return np.load(PAPER_5SP_DIR / PAPER_5SP_STATES[state] / 'samples.npy')


def paper_5sp_theta(state):
    """TMCMC MAP theta (20,) — sign解析用。軌道・図には paper_hamilton_nuts_samples() 推奨。
    10000p は {'0':v,..,'19':v} 形式、旧 run は {'theta_full':[..]} 形式の両対応。"""
    p = PAPER_5SP_DIR / PAPER_5SP_STATES[state] / 'theta_MAP.json'
    d = json.load(open(p))
    if 'theta_full' in d or 'theta_sub' in d:
        return np.array(d.get('theta_full', d.get('theta_sub')))
    return np.array([d[str(i)] for i in range(len(d))])


# ── Hamilton NUTS (③ 最高精度): Tmcmc202601/data_5species/main/_runs ────────────
# RMSE CS=0.119 / CH=0.104 / DS=0.033 / DH=0.087  (gLV ① の次に精度良い)
# 軌道図・修論 Fig は generate_fig2_phi_version.py (キャッシュ使用)。
# サンプルのみ保存（theta_MAP.json なし）→ 軌道計算は JAX ODE 必須。
_NUTS_RUNS_DIR = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main/_runs')
PAPER_HAMILTON_NUTS_DIRS = {
    'CS': 'jax_ode_nuts_Commensal_Static_20260320_043505',
    'CH': 'jax_ode_nuts_Commensal_HOBIC_20260320_043812',
    'DS': 'jax_ode_nuts_Dysbiotic_Static_20260320_044847',
    'DH': 'jax_ode_nuts_Dysbiotic_HOBIC_20260320_052844',
}


def paper_hamilton_nuts_samples(state):
    """Hamilton NUTS posterior samples — 軌道・修論図用 (③ 最高精度).
    shape: (n, 20); ODE は JAX の simulate_0d (Hill速度論) で積分すること。
    図生成: PATH=~/texlive/2025/bin/x86_64-linux:$PATH python
            ~/IKM_Hiwi/Tmcmc202601/docs/generate_fig2_phi_version.py
    """
    return np.load(_NUTS_RUNS_DIR / PAPER_HAMILTON_NUTS_DIRS[state] / 'samples.npy')


# ── gLV MAP fit (① 最高精度 RMSE): results/heine2025/fit_glv_heine.json ──────
# RMSE CS=0.032 / CH=0.026 / DS=0.012 / DH=0.022
# 図生成: PATH=~/texlive/.../bin:$PATH python scripts/figures/plot_glv_heine_thesis.py
GLV_FIT_JSON = _here / 'results' / 'heine2025' / 'fit_glv_heine.json'


def paper_glv_fit(state=None):
    """gLV MAP fit dict (or single condition). Keys per condition: A, b, rmse.
    state=None → 全条件 dict; state='CS' など → その条件の dict。
    """
    d = json.load(open(GLV_FIT_JSON))
    return d[state] if state else d


if __name__ == '__main__':
    print('=== TMCMC (②) samples — sign解析用 ===')
    print('Dir:', PAPER_5SP_DIR)
    for tag in PAPER_5SP_STATES:
        s = paper_5sp_samples(tag)
        print(f'  {tag:<3} samples {s.shape}  theta_MAP {paper_5sp_theta(tag).shape}')
    print()
    print('=== Hamilton NUTS (③) samples — 軌道・修論図用 ===')
    for tag in PAPER_HAMILTON_NUTS_DIRS:
        s = paper_hamilton_nuts_samples(tag)
        print(f'  {tag:<3} {PAPER_HAMILTON_NUTS_DIRS[tag]}  samples {s.shape}')
    print()
    print('=== gLV MAP fit (①) — RMSE 最小 ===')
    for tag, v in paper_glv_fit().items():
        print(f'  {tag:<3} RMSE={v["rmse"]:.4f}')
