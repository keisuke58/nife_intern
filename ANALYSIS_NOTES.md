# nife/ データセット分析ノート
作成: 2026-04-17

---

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

### 保留事項
- [x] EM/GMM で attractor basin assignment → 完了
- [x] ペリイン2クラスターが DH vs DS と対応するか検証 → DH(59%) + DS(3%)で確認
- [x] BIC k 選択 → k=3 最適（過分割疑い）
- [ ] Szafranski 論文のCT label (論文PDF本体から手動で取得が必要)
- [ ] Dieckow n_sp=5 shared-A フィッティング後にin vivo アトラクター位置を再推定
- [ ] Hamilton ODE のin vitro vs in vivo オフセットを定量化
- [ ] GMM 逆問題 (C) — ODE 評価がボトルネック → JAX で高速化してから再挑戦

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

### 実装状況
- `dieckow_hamilton_fit.py` — glue script骨格（n_sp=10で書いてある → n_sp=5に修正必要）
- `dieckow_batch_submit.sh` — 全30サンプル一括PBS投入スクリプト（ready）
- `dieckow_manifest.tsv` — ERR ↔ 患者×週 対応表（完成）

### 保留事項
- [ ] `dieckow_hamilton_fit.py` を n_sp=5, shared A 設計に修正
- [ ] job 39803 (A_3 whitelist fix) 完了後 Streptococcus 1位を確認
- [ ] 確認後 `bash dieckow_batch_submit.sh --skip-done` で残り29投入
- [ ] mSystems 5属データのアトラクター解析と照合

---

## 3. Dieckow ENA sample mapping

**ファイル**: `dieckow_manifest.tsv`

10患者 (A,B,C,D,E,F,G,H,K,L) × 3週 = 30サンプル
（I, J は存在しない）

---

## 4. ジョブ状況

| Job | Sample | Status | 備考 |
|-----|--------|--------|------|
| 39802 | A_3 | 完了(失敗) | blacklistアプローチ → Klebsiella 27% |
| **39803** | A_3 | **実行中** | whitelistアプローチ (eHOMD) |

完了後確認コマンド:
```bash
cat /home/nishioka/IKM_Hiwi/nife/results/dieckow_ccs_A_3.log | tail -20
```
期待: Streptococcus ~35% が1位
