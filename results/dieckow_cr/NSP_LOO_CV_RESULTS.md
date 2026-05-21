# NSP Hamilton ODE LOO-CV 結果まとめ

Dieckow 10-guild 口腔微生物データセット（患者 A,B,C,D,E,F,G,H,K,L）での
Leave-One-Out Cross-Validation。予測対象は W3（3週目組成）。

---

## モデル比較 (10-fold LOO-CV)

| バージョン | RMSE (mean) | RMSE (std) | BC (mean) | 説明 |
|-----------|------------|-----------|----------|------|
| gLV 2-step | **0.0450** | — | **0.144** | ベースライン (参考) |
| NSP IFT v2 | **0.0906** | 0.0427 | — | 最良 NSP 結果 |
| NSP IFT v4 | 0.1135 | 0.0462 | — | joint A+b 最適化 |
| NSP IFT v5 | 0.1132 | 0.0559 | 0.342 | 1-step + A_pop prior |
| NSP-φ (replicator) | 0.1712 | 0.0563 | 0.524 | gLV replicator 化 |
| NSP IFT v3 | ~0.106 | — | — | alternating 3 rounds |

---

## v2 (最良 NSP) の詳細

**ファイル**: `nife/loo_nsp_ift_gpu.py`

### アルゴリズム

3ステップ交互最適化 (per fold):

```
Step A: 全 train 患者 W1→W2 で 共有 A_upper (55 params) を L-BFGS-B
Step B: A 固定、各 train 患者の b_train (9×10=90 params) を最適化
Step C: test 患者の W1→W2 で b_test (10 params) をフィット
        → W1→W2→W3 の 2-step 予測で W3 推定
```

### 勾配計算

JAX `custom_vjp` + IFT (Implicit Function Theorem):

- Newton ステップの不動点 `G(x, θ) = 0` に関する陰的微分
- `x_bar = -(∂G/∂x)^T v`、`v` は `J^T v = g_bar` の解
- NSP Hamilton ODE の `custom_vjp` を正しい符号で実装済み

### ハイパーパラメータ

```
n_steps = 100    # 1週間 = 1.0 time unit
dt      = 0.01
c       = 25.0   # NSP スケーリング定数
alpha   = 100.0  # NSP alpha
maxfun  = 2000   # L-BFGS-B
λ_A     = 1e-4
λ_b     = 1e-3
```

### per-patient 結果

| 患者 | LOO RMSE | Train RMSE |
|------|---------|-----------|
| A    | 0.1769  | 0.0767    |
| B    | 0.0339  | 0.0838    |
| C    | 0.0959  | 0.0841    |
| D    | 0.0979  | 0.0823    |
| E    | 0.0236  | 0.0748    |
| F    | 0.0888  | 0.0781    |
| G    | 0.0549  | 0.0815    |
| H    | 0.1176  | 0.0819    |
| K    | 0.0935  | 0.0700    |
| L    | 0.1225  | 0.0826    |
| **mean** | **0.0906** | **0.0796** |

---

## なぜ gLV に負けるか

NSP Hamilton ODE の根本的な制約:

1. **観測量は φ̄ = φ × ψ** (φ 単体ではない)
2. **b_diag は ψ 方程式にのみ登場** — `b_diag[i]*alpha/Eta[i]*psi_new[i]`
   → φ 方程式に b は存在しない
3. **潜在変数 ψ, φ₀, γ** が W1 から "ウォームアップ" 必要
   → W2_obs から直接スタートする 1-step 予測は精度悪化 (v5 で確認)
4. **c=25, alpha=100** は biofilm 向け設計、週次口腔マイクロバイオームには合わない可能性

## 試みた改善と結果

| 改善案 | バージョン | 結果 |
|-------|----------|------|
| alternating A/b 複数ラウンド | v3 | 悪化 (oscillation) |
| joint [A, b] 145-param 最適化 | v4 | 悪化 (0.113) |
| 1-step W3 training + A_pop prior | v5 | 悪化 (0.113) |
| NSP-φ: ψ=1 固定 → gLV replicator | phi | 大幅悪化 (0.171) |

**結論**: NSP IFT v2 (RMSE=0.091) が NSP variants の最良。
発表・論文では v2 を NSP の代表結果として使用する。

---

## 実行方法

```bash
# Stuttgart01 / Vancouver01 (~/miniforge3/envs/klempt_fem2/bin/python)
CUDA_VISIBLE_DEVICES=0 python loo_nsp_ift_gpu.py --fold-start 0 --fold-end 5
CUDA_VISIBLE_DEVICES=1 python loo_nsp_ift_gpu.py --fold-start 5 --fold-end 10
```

結果: `results/dieckow_cr/loo_nsp_ift_v2_all10.json`
