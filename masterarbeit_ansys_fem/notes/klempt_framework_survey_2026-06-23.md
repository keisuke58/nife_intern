# IKM FEM Biofilm Framework — Literature Survey
*作成: 2026-06-23, 修論§5.2 ANSYS統合準備*

---

## 1. Klempt et al. 2024 — 統合先の直接的な基盤論文（CC BY 4.0）

**タイトル**: A Hamilton principle-based model for diffusion-driven biofilm growth  
**著者**: Felix Klempt, Meisam Soleimani, Peter Wriggers, Philipp Junker  
**誌**: Biomechanics and Modeling in Mechanobiology (Springer)  
**受理**: 2024-08-05 / 公開: 2024-09-30  
**PMC**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11554842/  
**ライセンス**: CC BY 4.0 → 図・データ再利用自由（帰属表記のみ必要）

### モデル概要

| 項目 | 内容 |
|---|---|
| 支配原理 | 拡張Hamilton原理（Junker 2021） |
| 変数（節点） | 変位 **u** (3) + 栄養濃度 c (1) + バイオフィルム状態変数 ϕ (1) = 5 DOF/node |
| 内部変数（Gausspoint） | 膨張パラメータ α（節点DOFではない） |
| 要素 | 8ノードブリック (H1P0)、静水圧 p を静的縮約 (Schur complement) |
| 実装 | AceGen (Mathematica) で自動微分 → 線形化剛性行列 → FORTRAN UEL → ANSYS |
| 結合 | 完全陰的モノリシック (変位＋拡散＋成長) |
| 種数 | **単一種** (ϕ 1変数) |

### 支配方程式（Eq. 34–36から抽出）

```
バイオフィルム進化:
  ϕ̇ − β∇²ϕ − k_α·α + ∇ϕ · (rc/(k+c)) n_∇ϕ · n_∇c = 0

栄養拡散:
  ċ − d∇²c + g·ϕ = 0

膨張パラメータ (Gausspoint ローカル):
  α̇ − k_α·ϕ = 0   ← ここにPythonモデルを接続する箇所
```

### 自由エネルギー密度 (Eq. 20)

```
Ψ = ϕ·(μ/2)(I:C^e − 3) + (β̄/2)|∇_X ϕ|² + (d̄/2)|∇_X c|² − k̄_α·α·ϕ
```

### 運動学

- 乗法分解: **F** = **F**^e · **F**^g
- 等方成長: **F**^g = α**I**
- 非圧縮: det[**F**^e] = 1

### 材料パラメータ値（Table 2）

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| せん断弾性率 | μ | 3.3557 | Pa |
| Young率 | E | 10 | Pa |
| Poisson比 | ν | 0.49 | − |
| 栄養拡散率 | d | 10^10 | μm²·T*⁻¹ |
| 位相場正則化 | β | 2 | μm²·T*⁻¹ |
| 成長係数 | k_α | 10⁻³ | T*⁻¹ |
| 半速度定数 | k | 1 | − |
| 消費パラメータ | g | 10^8 | T*⁻¹ |
| 成長パラメータ | r | 100 | T*⁻¹ |

※ T* = t/t_ref（正規化時間）

> **重要**: 論文内で「パラメータは現象論的に選択。実データへのfittingは将来課題」と明記。
> E=10 Pa は "requires adjustment to fit real biofilm in future works" と figure caption に記載。

### ANSYSローカルGausspoint ループ（Table 1より）

```
1. 線形FEM形状関数で場変数を補間
2. Newton-Raphson初期化: F^e_(n+1) = F_(n+1) · (F^g_n)^{-1}
3. [Gausspoint ローカル] α_(n+1) を Eq.36 で解く  ← Pythonモデル挿入点
4. 残差 (Eq.26-28) を弱形式で計算
5. 剛性行列 K^e に内部変数感度 ∂α/∂D を組み込み
6. 静水圧 p を Schur 縮約: 41×41 → 40×40
7. 大域 Newton-Raphson（収束まで）
8. 自動時間刻み 2分割（substeps: 10³〜10⁶）
```

### 検証ケース（全て定性的のみ）

| ケース | ドメイン | 初期条件 | 主結果 |
|---|---|---|---|
| 方向成長 | 20³ μm | 中心5μm球にϕ=1 | 栄養源方向に卵形成長 |
| 寒天プレート (low g) | 20³ μm | 上面5μm円盤 | キノコ→液滴形状 |
| 寒天プレート (high g) | 20³ μm | 同上 | 基底層のみ（栄養枯渇で上昇なし） |
| 剛体障害物回避 | 30³ μm | 中心5μm球 | 4柱を回り込んで再接続 |
| 迷路（semi-2D） | 560μm径, 4μm厚 | — | 栄養源方向へ成長, 行き止まりから退縮 |
| 格子（semi-2D） | — | — | 障害物分岐・再合流, 内部圧縮/境界引張 |

### ⚠️ 実験検証なし（論文の明示的記述）

> "This is necessary when it comes to the experimental validation of the model and **is the direction of our future work**."

→ **修論の直接的な貢献ギャップ**: TMCMC推定済みパラメータ + FISH空間データによる定量検証がここに入る。

---

## 2. Klempt, Geisler, Soleimani, Junker 2025 — 修論の直接競合モデル

**タイトル**: A continuum multi-species biofilm model with a novel interaction scheme  
**著者**: Felix Klempt, **Hendrik Geisler**, Meisam Soleimani, Philipp Junker  
**arXiv**: 2509.01274  
**状態**: preprint（2025、査読前）

### 重要点

- **多種Hamilton FEM**：体積分率 φ_i (i=1..N) を複数追跡
- 「novel interaction scheme」= 種間相互作用の新定式化
- **修論の位置づけ**: このinteraction schemeを、TMCMC推定済みの相互作用行列 **A**（Heineデータ由来）で置き換えるのが本修論の貢献

### 修論との差分

| 項目 | Klempt 2025 | 本修論 |
|---|---|---|
| 相互作用行列 | 現象論的パラメータ | TMCMC posterior (A, 15パラメータ) |
| パラメータ同定 | フィッティング | Bayesian推定済み |
| 比較モデル | なし | Hamilton vs gLV 2軸 |
| 検証 | FEM内部整合 | HeineFISH / Dieckow空間データ（目標） |

---

## 3. Soleimani et al. 2023 — 多種共凝集の実験データ

**タイトル**: Numerical and experimental investigation of multi-species bacterial co-aggregation  
**誌**: Scientific Reports  
**DOI**: https://www.nature.com/articles/s41598-023-38806-2

### 実験データ（validationに使える可能性）

- 菌種: *Streptococcus gordonii* (So), *Eikenella corrodens* (Ec), *Candida albicans* (Ca)
- 測定: CLSM + LIVE/DEAD染色 (SYTO9 488nm / PI 590-680nm)
- 3D再構成: Imaris → 緑/赤/黄（共局在）体積分率
- **注意**: 5菌種(So/An/Vei/Fn/Pg)ではなく別組み合わせ

---

## 4. Fitting対象・Validationデータ候補まとめ

### 優先度A: 既存lab内データ（IKM/NIFE）

| データ | 物理量 | 修論での使い方 |
|---|---|---|
| CLSMバイオフィルム厚さ (84 FOV) | z方向成長プロファイル | §5.2のFEM成長fitting（corr 0.88確認済） |
| FISH深さプロファイル (Heine data) | φ_i(z)（5菌種空間分布） | **最重要**: FEM φ_i(x,z) と直接比較可能 |
| HOBIC reactor vs static 4条件 | 条件別バイオフィルム構造 | FEM境界条件設定の根拠 |

### 優先度B: 文献データ

| データ | 物理量 | 出典 |
|---|---|---|
| AFMインデンテーション | Young's modulus 20 Pa–2 kPa | 文献値でパラメータ範囲設定 |
| Soleimani 2023 CLSM | So/Ec/Ca 共凝集3D形態 | 代替spatialデータ（菌種異なる） |

### 優先度C: 要取得

| データ | 手段 |
|---|---|
| 5菌種FISHの3D空間データ | Felix/Szymonに確認（未公開データの可能性） |
| Klempt 2025のFEM検証ベンチマーク | preprint共著者=Felixに直接聞く |

---

## 5. 修論§5.2への接続

```
Klempt 2024 (単種FEM) ← Klempt 2025 (多種FEM, 現象論的相互作用)
                                        ↑
                        本修論: TMCMC済みA行列 (Hamilton / gLV) で置換
                                        ↓
                        Validation: FISH φ_i(z) vs FEM出力
```

### Hamilton vs gLV 2軸比較の意義

- Dieckow LOO-CV: gLV (α=0.25) > Hamilton（バルクODE精度）
- FEM空間予測: どちらが勝るか未知 → これが§5.2の核心的問い
- Hamilton優位仮説: 変分整合性 → 空間的な自由エネルギー最小化が物理的に正しいパターンを生成する

---

## 6. Felix Klemptへの確認事項（返信待ち 2026-06-22送信済み）

1. coupling手法: UPF / ctypes / IPC のどれか
2. コード共有（minimal example）
3. 検証ベンチマーク
4. **追加で確認したい**: Klempt 2025のFEM空間出力データ、FISH深さプロファイルデータの有無

---

*参照*: `INTEGRATION_PLAN.md`, `../heine_paper/nishioka_heine_paper.tex`, `project_masterarbeit` memory
