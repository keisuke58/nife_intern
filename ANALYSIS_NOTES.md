# nife/ データセット分析ノート
作成: 2026-04-17

---

## 0. 次のプラン（CT1/CT2予測・Consumer Resource比較・COMETS dFBA拡張）

### 0.1 CT1/CT2 予測（分類木：解釈重視）
- **目的**: Dieckow患者の community type（CT1/CT2）を予測し、「どの特徴が閾値で分かれるか」を明示的ルールとして得る。
- **ラベル定義（固定）**:
  - `nife/community_type_analysis.py` のクラスタ結果をラベルとして利用
  - 出力: `nife/results/dieckow_otu/community_types.json`（patient_ct と CT1/CT2 平均組成）
- **特徴量候補（同一患者で揃うものから）**:
  - Week1（または初期）組成（guild/属/OTU、相対存在量、CLRやlog比など）
  - DI などの集約指標（計算可能なら）
  - モデル由来特徴量（例: Dieckow fit の患者別 b、CR fit の b など）
- **モデル**: DecisionTreeClassifier（浅めの木 + 剪定）を第一候補。比較としてLogReg（L1）も並走すると解釈が安定。
- **評価**: 小標本のため LOO（leave-one-patient-out）または層化CV。AccuracyだけでなくCT2のrecallも確認。
- **成果物**: ルール（木の分岐）・混同行列・重要特徴量ランキング・外れ患者の診断。

### 0.2 Consumer-resource（CR）model 比較（CR vs Hamilton/GLV系）
- **目的**: 予測精度・解釈性・汎化の観点で、CRモデルが既存のA行列モデルに対して何を説明できるかを定量比較する。
- **CR実装（現状）**:
  - モデル: `nife/consumer_resource_dieckow.py`
  - フィット: `nife/fit_cr_dieckow.py` → `nife/results/dieckow_cr/fit_cr.json`
  - CRからの有効相互作用: `effective_A_from_cr(theta_cr)`（等組成点でのヤコビアン近似）
- **比較対象（A行列ベースの予測枠組み）**:
  - Dieckow week2/3予測RMSE比較の型: `nife/dieckow_cross_prediction.py`
- **比較軸（最低限）**:
  - Week1 → Week2/3 の予測RMSE（in-sample + LOO）
  - CR由来の A_effective と、A行列モデル推定（MAPなど）の相関・符号整合
  - パラメータ数・初期値依存・安定性（局所解/感度）

### 0.3 COMETS dFBA 拡張（CT差・CR仮説検証と接続）
- **目的**: GEM+dFBA（COMETS）で、healthy/diseasedの再現に加え、CT1/CT2差や代謝仮説（cross-feeding、O2制約など）を再現・検証できる状態にする。
- **入口（現状）**:
  - パイプライン実行: `nife/comets/run_comets_pipeline.py`
  - 5種モデル/媒体/交換制約: `nife/comets/oral_biofilm.py`
  - 2D Monod dFBA（COMETSとは別系の自前モデル）: `nife/comets/spatial_dfba.py`
- **拡張の優先順位**:
  - CT1/CT2ごとの「初期組成・媒体条件」のプリセットを用意し、0D/2Dで差が出るか検証
  - 出力を週スケールに合わせたサマリ（Week1/2/3相当、DI、主要代謝物）に統一
  - 空間（2D）で O2 勾配・乳酸勾配と Vp/Pg の棲み分けが成立するかを確認
- **接続の形**:
  - 分類木が拾う「閾値特徴（例: Week1の特定属の比率）」を、COMETS側の初期/媒体条件として再現可能かを探索
  - CR/Hamiltonで示唆される相互作用の符号（促進/阻害）を、dFBAの代謝フラックス（交換・分泌）として整合させる

## 1. Szafranski 2025 横断データ (5-genera subset)

**論文**: Szafranski et al. 2025, ecoPreprint  
**DOI**: https://doi.org/10.1101/2025.06.23.661096  
**ファイル**: `Datasets/20260416_mSystems_16S_5genera_all_profiles.txt`

### 論文概要（重要）
- 125 peri-implant biofilm samples, 48 individuals
- 4つの **community types (CTs)** を transcriptional activity で定義:
  - CT1 → PIH (peri-implant health)
  - CT2 → PI (peri-implantitis)
  - CT3, CT4 → PIM (peri-implant mucositis, 2種類)
  - CT3 (Neisseria-rich): pyruvate/lipoic acid 代謝亢進
- **論文の4 CT = Hamilton ODE の4アトラクター (CS/CH/DH/DS) と1:1対応の可能性**

### データ概要
| 項目 | 内容 |
|------|------|
| サンプル数 | 127（横断、時系列なし） |
| 条件 | Health=56, Mucositis=39, Peri-implantitis=32 |
| 菌属 (5種) | Actinomyces, Fusobacterium, Porphyromonas, Streptococcus, Veillonella |
| OTU数/属 | An=23, Fn=19, Pg=24, So=45, Vei=9 |
| リード数/サンプル | min=730, median=22891, max=44237 |

**重要**: これはTmcmc202601の5菌種モデルと完全に同じ5属 → Hamilton ODEアトラクターと直接比較可能

### 条件別平均組成
```
            Actinomyces  Fusobacterium  Porphyromonas  Streptococcus  Veillonella
Health          0.056        0.140          0.072          0.591          0.140
Mucositis       0.035        0.171          0.065          0.625          0.104
Peri-impl.      0.024        0.352          0.237          0.311          0.076
```

### アトラクター解析
```
Health ↔ Mucositis:        距離 0.062  ← 近い（遷移中間状態）
Health ↔ Peri-implantitis: 距離 0.395  ← 遠い（別アトラクター）
Peri-impl. 内分散:          0.159       ← 最大（不均一→複数サブアトラクター？）
```

**Porphyromonas 二峰性** (peri-implantitis内):
```
Pg < 5%:  10/32 サンプル  ← Fn優位アトラクター（Pg低、Fn ~50%）
Pg > 30%: 13/32 サンプル  ← Pg優位アトラクター（古典的ペリイン）
```
→ **peri-implantitis は単一アトラクターでなく2つのサブアトラクター**

### Hamilton ODEとの対応
| 臨床条件 | 推定アトラクター | 主要指標 | Szafranski CT |
|---------|----------------|---------|--------------|
| Health | commensal_static (CS) | Streptococcus ~59% 優位 | CT1 |
| Mucositis (aerotolerant) | commensal_hobic (CH) | Health に近い | CT3 or CT4 |
| Mucositis (Neisseria-rich) | commensal_hobic or CS | Neisseria 優位 | CT3 or CT4 |
| Peri-impl. (Pg-high) | dysbiotic_static (DS) | Pg 30-80% | CT2 |
| Peri-impl. (Fn-high) | dysbiotic_hobic (DH) | Fn 50%, Pg <5% | CT2? |

### GMM/EM解析結果 (2026-04-17)
**スクリプト**: `gmm_attractor_analysis.py`  
**出力**: `results/gmm_attractor_analysis.{png,csv}`, `results/gmm_summary.txt`

| GMM cluster | →attractor | n | So | Fn | Pg |
|-------------|-----------|---|-----|-----|-----|
| Cluster 0 | DH | 37 | 32% | 35% | 29% |
| Cluster 1 | CH | 75 | 75% | 10% | 1% |
| Cluster 2 | CS | 13 | 65% | 7% | 0% |
| Cluster 3 | DS |  2 | 4% | 25% | 70% |

BIC=515.5, converged=True, GMM vs NN agreement=48%

**主要知見**:
1. PIH/PIM → CH dominant (66-67%), PI → DH dominant (59%)
2. DS 2サンプルのみ → Pg極端高値(70%)の古典的ペリイン
3. GMM vs NNの48%不一致は「in vitroオフセット」を過小評価している:
   - Heine in vitro ODEアトラクター: CS=24% So, CH=29% So
   - in vivo GMM centroid:           CS=65% So, CH=75% So  ← 完全に別の空間
   - NN(φ-L2) vs NN(CLR) の一致率はわずか **30%** （同じアトラクターを使っても）
   - → ODEアトラクターが組成空間に「置かれた位置」自体が無意味
   - → **NN割り当てはHeine in vitro/peri-implant in vivoギャップにより全面的に信頼不可**
   - → **GMM分析のみが信頼できるin vivo attractor推定**
   - → Dieckow fitting は「改善」でなく **in vivoへの初回キャリブレーション** として必須

### GMM k 選択 (BIC, 2026-04-17)
**スクリプト**: `gmm_bic_curve.py` → `results/gmm_bic_curve.png`

| k | BIC | 備考 |
|---|-----|------|
| 2 | 524.1 | |
| **3** | **501.9** | **← BIC 最小（最適）** |
| 4 | 515.5 | Hamilton ODE 仮定 |
| 5 | 564.7 | |

k=3 クラスター:
- C0 (n=66): So=56%, Fn=19%, Pg=10% — PIH/PIM/PI 全体に混在
- C1 (n=59): So=51%, Fn=22%, Pg=11% — 同上（C0とほぼ同組成）
- C2 (n=2):  So=31%, Pg=50% — 極端 DS

**解釈**: BIC は「4アトラクター仮説は過分割」と判定。in vivo ではPIH→PI 遷移が**連続スペクトル**に見える可能性。ただし k=3 の C0/C1 重心が近すぎ（局所解疑い）→ Dieckow in vivo θ で再評価が必要。

### GMM re-assignment with in vivo θ (2026-04-19)
**スクリプト**: `dieckow_gmm_reassign.py`
**出力**: `results/dieckow_gmm/`

- Dieckow MAP A 行列 + 患者平均 b → Szafranski 127サンプル × 20週 ODE → k-means(k=4)
- Heine in vitro DH attractor: So=0.849 Fn=0.024 Pg=0.033 → **in vivo よりはるかに So 優位**
- 結果: `gmm_reassigned.csv`, `invitro_invivo_offset.json`

### 保留事項
- [x] EM/GMM で attractor basin assignment → 完了
- [x] ペリイン2クラスターが DH vs DS と対応するか検証 → DH(59%) + DS(3%)で確認
- [x] BIC k 選択 → k=3 最適（過分割疑い）
- [x] Dieckow n_sp=5 shared-A フィッティング後にin vivo アトラクター位置を再推定 → 完了(2026-04-19)
- [x] GMM re-assignment with in vivo θ → 完了(2026-04-19)
- [ ] Szafranski 論文のCT label (論文PDF本体から手動で取得が必要)
- [ ] Hamilton ODE のin vitro vs in vivo オフセットを定量化（全4条件 → gmm_reassign 再実行で取得）
- [ ] GMM 逆問題 (C) — ODE 評価がボトルネック → JAX で高速化してから再挑戦
- [ ] Posterior predictive check (Dieckow 1000p → Szafranski 127サンプル予測分布)

---

## 2. Hamilton ODE フィッティング実現可能性 (Dieckow 2025)

**対象**: PRJEB71108, 10患者×3週=30サンプル

### タイムスケール → **問題なし**
```python
# auto day_scale = (2500*1e-4 * 0.95) / 21 = 0.01131
# weeks 1,2,3 = days 7,14,21 → steps [792, 1583, 2375]
# Siddiqui後半3点と同じマッピング → convert_days_to_model_time() そのまま使える
```

### phibar正規化 → **問題なし**
```python
# compute_phibar(normalize=True): phibar_i / Σ_j phibar_j
# → relative abundance に変換、データと直接比較可能
```

### 識別可能性 → **設計次第**

| 設計 | params | data | 比 | 判定 |
|------|--------|------|----|------|
| n_sp=10, per-patient | 65 | 20 | 0.31× | ✗ 不可 |
| n_sp=10, shared A | 155 | 200 | 1.29× | △ ギリギリ |
| **n_sp=5, shared A + per-patient b** | 65 | 100 | 1.54× | ✓ 推奨 |
| n_sp=5, 全患者1θ | 20 | 100 | 5.0× | ✓ 理想 |
| Siddiqui参考 | 27 | 30 | 1.11× | ✓ 実績あり |

### 推奨設計
```
n_sp = 5  (mSystems 5-genera データと揃える → 直接比較可能)
A行列:  患者間共通 (15 params)  ← 生態系ルールは同じ
b_i:    患者ごと独立 (10×5=50 params)  ← 成長速度は患者差あり
total: 65 params / 100 data = 1.54×
```

**n_sp=5 の追加利点**: mSystems横断データとの比較が可能
- 横断データの各サンプル → ODEアトラクターに割り当て
- 縦断データ(Dieckow)でθを推定 → 横断データを予測

### 実装状況 (2026-04-19 更新)
- `dieckow_hamilton_fit.py` — n_sp=5 shared-A GPU TMCMC **完成・実行済み**
- `dieckow_manifest.tsv` — ERR ↔ 患者×週 対応表（完成）
- **taxonomy**: 全30サンプル完了 (`results/dieckow_taxonomy/`)
- **1000p fit**: RMSE=0.1006, 10 stages, MAP完了 (`results/dieckow_fits/fit_joint_5sp_1000p.json`)

### TMCMC結果サマリー (2026-04-18完了)
| 項目 | 値 |
|------|-----|
| RMSE (sign_prior=OFF) | 0.1006 |
| RMSE (sign_prior=ON)  | 0.1008 (Δ=+0.0001) |
| ステージ数 | 10 |
| Per-patient RMSE | best=G(0.050), worst=L(0.119) |
| Posterior samples | (1000, 65) |

### 符号プライア vs データ矛盾（重要知見, 2026-04-19）
Bergey/Dieckow 文献予測と in vivo TMCMC posterior の比較:

| パラメータ | 期待符号 | comply OFF | comply ON | 解釈 |
|-----------|---------|------------|-----------|------|
| A[So,An]  | + | 100% | 100% | ✓ 一致 |
| A[So,Vd]  | + | 97.7% | 100% | ✓ 一致 |
| A[So,Fn]  | + | 100% | 100% | ✓ 一致 |
| **A[An,Vd]** | + | **1.3%** | **0.0%** | ✗ **競合が共生を凌駕** |
| A[An,Pg]  | + | 93.4% | 100% | ✓ 一致 |
| **A[Vd,Pg]** | + | **0.0%** | **0.0%** | ✗ **Vd-Pg 空間競合** |

→ in vivo では An-Vd 間・Vd-Pg 間が負の相互作用 → 文献の in vitro 予測と乖離
→ 論文主張: **in vivo キャリブレーションにより生態系相互作用の再評価が可能**

### 患者F 異常診断 (2026-04-19)
- 観測データ: Fn≈0, Pg≈0 全3週 → 実質3菌種システム
- b_Fn=13.3, b_Vd=13.5 が高く、平均 b で回すと Health様 attractor (So=0.69) に正常収束
- MAP b_An=0.584（極端に低い）→ An支配 degenerate attractor (An=0.877)
- **推奨**: 患者F を "Fn/Pg-absent" サブタイプとして flag、attractor 解析から除外

### 保留事項
- [x] n_sp=5 shared-A TMCMC 実装・実行 → 完了
- [x] 全30サンプル taxonomy → 完了
- [x] mSystems 5属データとの attractor 比較 → dieckow_gmm_reassign.py で実施
- [ ] GMM re-assignment 全4条件 Heine θ で再実行（CS/CH/DS theta_MAP をvancouver01に転送済み）
- [ ] Posterior predictive check (Dieckow 1000p → Szafranski 127サンプル予測分布)

---

## 3. Dieckow ENA sample mapping

**ファイル**: `dieckow_manifest.tsv`

10患者 (A,B,C,D,E,F,G,H,K,L) × 3週 = 30サンプル
（I, J は存在しない）

---

## 4. ジョブ状況 (2026-04-19更新)

| Job | Sample | Status | 備考 |
|-----|--------|--------|------|
| 39804-39832 | dc_A_1〜dc_L_3 各種 | **PBS MOM断絶・自然完了待ち** | taxonomy TSV は全30サンプル完了済み |

- PBS サーバーがノードを `down` 認識 → qdel 不可 → 放置（実害なし）
- 全30サンプルの taxonomy TSV は既に `results/dieckow_taxonomy/` に存在
