# FEM Pg biofilm model — Mukherjee 2025 fitting

## 概要

Mukherjee 2025 の *P. gingivalis* バイオフィルム実験（rougher / smoother Ti 表面）を
Q1 hex FEM で再現するプロトタイプ。`fem_pg_model.py` 実装。

論文参照: Mukherjee et al. 2025 (DHNA による Pg 抑制、Ti 表面粗さ依存性)

---

## モデル構造

**メッシュ**: Q1 hex (Galerkin), デフォルト 12×12×8 ノード

**場変数**:
- `c(x,t)` — 栄養濃度（上面 Dirichlet, 下面 Neumann）
- `φ(x,t)` — 相場（バイオフィルム体積分率、下面シード）
- `NH4(t)` — アンモニウム（0D ODE、成長の proxy）

**成長則** (logistic × dual-Monod):
```
∂φ/∂t = μ_max · μ_scale · φ(1−φ) · [c/(Km_c+c)] · [NH4/(Km_NH4+NH4)] − k_detach·φ
```

**DHNA モデリング**: `μ_scale = 1.2`（成長促進 proxy）

---

## キャリブレーション手順 (`--fit_fig2`)

1. **Step 1 — `fit_k_nh4` bisection**  
   baseline NH4(day6) = 5 mM (論文 Fig 1) に合わせて `k_nh4` を2分法で自動フィット  
   結果: `k_nh4 ≈ 2.4`

2. **Step 2 — per-surface 3D grid search**  
   `phi_seed × k_detach × km_nh4` を各 surface 独立に探索  
   目的関数: baseline + DHNA 両方の volume RMSE (最小二乗スケール込み)  
   degenerate scale (`> 1e4`) は自動 reject

3. **Step 3 — 最良パラメータで最終 run & overlay plot**

---

## Best params (v7, 2026-04-11)

| Surface | phi_seed | k_detach | km_nh4 (mM) | scale | RMSE baseline | RMSE DHNA |
|---------|----------|----------|-------------|-------|---------------|-----------|
| rougher  | 0.050 | 0.030 | 0.30 | 1.59 | 0.584 | 0.315 |
| smoother | 0.070 | 0.005 | 0.50 | 0.34 | 0.199 | 0.203 |

出力: `pipeline_results/fem_pg_fit_fig2_v7.png`

---

## RMSE 改善推移

| version | 変更 | rougher baseline RMSE |
|---------|------|-----------------------|
| v3 | mu_phi=0.25, k_detach=0.04 fixed | 0.690 |
| v4 | phi_seed × k_detach 2D grid search | 0.668 |
| v5 | + NH4 Monod (km_nh4=5.0 固定) | 0.775 ← 悪化 |
| v6 | + km_nh4 も grid search (3D), scale>1e4 reject | 0.626 |
| v7 | surface 別グリッド細分化 | **0.584** |

---

## 残課題

- **rougher の peak-then-decline 未再現**  
  論文 Fig 2: rougher baseline は day3 ピーク → day4-6 低下  
  現モデルは単調 sigmoid。km_nh4=0.30 が best → NH4 Monod は rougher に効いていない。  
  根本原因: logistic (1−φ) carrying capacity が時変でないと decline を作れない。

  将来案:
  - detachment を `k_detach · φ²` に変更（高密度で加速）
  - carrying capacity `φ_max(t)` を NH4 連動で時変化
  - EPS 蓄積 → 剥離のタイムラグモデル

- **DHNA の定量的モデリング**  
  現在は `μ_scale=1.2` の proxy。論文は DHNA が電子伝達鎖を阻害 → Pg 選択的抑制。
  将来: ATP 産生率を DHNA 濃度の関数として陽に入れる。

---

## ファイル構成

```
comets/
├── fem_pg_model.py          # FEM solver + calibration
├── fem_pg_grid_search.sh    # PBS job script
└── pipeline_results/
    └── fem_pg_fit_fig2_v7.png  # best overlay plot
```
