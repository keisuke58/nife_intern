# In Vivo Microbiome Data Fitting — 主要論文まとめ

16S rRNA シーケンスによる相対量データに ODE/統計モデルを当てはめた代表的研究。

---

## 1. gLV 系 (Generalized Lotka-Volterra)

### Stein et al. 2013 — PLOS Computational Biology
- **タイトル**: Ecological Modeling from Time-Series Inference: Insight into Dynamics and Stability of Intestinal Microbiota
- **データ**: ヒト腸内 16S 時系列（健常 + 抗生剤投与）
- **モデル**: gLV + elastic net / LASSO 正則化
- **貢献**: in vivo 腸内細菌叢への gLV fitting の最初期論文、LOO-CV で予測精度評価
- **相対量の扱い**: CLR (centered log-ratio) 変換後に gLV

### Bucci et al. 2016 — PLOS Computational Biology (MDSINE)
- **タイトル**: MDSINE: Microbial Dynamical Systems INference Engine for microbiome time-series analyses
- **データ**: マウス腸内 16S（抗生剤 + *C. difficile* 投与）
- **モデル**: gLV + Bayesian spike-and-slab prior (MCMC)
- **貢献**: in vivo gLV fitting の **gold standard**。事後分布から相互作用ネットワーク推定、介入実験の予測
- **相対量の扱い**: count data → 正規化後 gLV

### Gibson et al. 2021 — PLOS Computational Biology (MDSINE2)
- **タイトル**: Robust and Scalable Models of Microbiome Dynamics
- **データ**: MDSINE と同データ + ヒト被験者データ
- **モデル**: gLV + Bayesian nonparametric (negative binomial likelihood)
- **貢献**: リードカウントを negative binomial で直接モデル化 → 疑似絶対量、MDSINE より安定
- **相対量の扱い**: NB モデルで count level まで降りる（実質最も丁寧な処理）

### Fisher & Mehta 2014 — PLOS Computational Biology (LIMITS)
- **タイトル**: Identifying Keystone Species in the Human Gut Microbiome from Metagenomic Time-Series Using Sparse Linear Regression
- **データ**: ヒト腸内メタゲノム時系列
- **モデル**: gLV + sparse regression (LIMITS)
- **貢献**: keystone species の同定、スパース A 行列推定

---

## 2. 口腔マイクロバイオーム

### Dieckow et al. (本研究の参照データセット)
- **データ**: 口腔フローラ 10 患者 × 3 週、16S guild レベル集計
- **モデル**: gLV replicator / NSP Hamilton ODE (本研究)
- **特徴**: 健常ヒト口腔、週次サンプリング

### Gao et al. 2016 — Scientific Reports
- **タイトル**: Community-wide transcriptomics reveals...
- **データ**: ヒト歯垢 16S + metatranscriptome
- **ポイント**: 代謝ネットワークと組成の同時解析

---

## 3. 絶対量定量 (Absolute Quantification)

### Vandeputte et al. 2017 — Nature
- **タイトル**: Quantitative microbiome profiling links gut community variation to microbial load
- **データ**: ヒト腸内 16S + flow cytometry による総菌数
- **手法**: flow cytometry × 16S = absolute quantification
- **貢献**: 相対量バイアスを補正した絶対定量 OTU プロファイリング
- **意義**: IBS 患者では相対量では見えない絶対的な変化が重要

### Tkacz et al. 2018 — Microbiome
- **タイトル**: Absolute quantitation of microbiota abundance in environmental samples
- **手法**: spike-in (合成 DNA 添加) + 16S
- **用途**: 土壌・環境微生物叢の絶対定量

---

## 4. ベイズ・機械学習系

### Ridaura et al. 2013 — Science
- **タイトル**: Gut Microbiota from Twins Discordant for Obesity Modulate Metabolism in Mice
- **データ**: ヒト双子便移植 → マウス
- **ポイント**: 組成 → 体重表現型の因果関係（fitting よりも移植実験）

### Buffie et al. 2015 — Nature
- **タイトル**: Precision microbiome modulation with discrete transfer functions
- **データ**: マウス腸内 16S（抗生剤 + C.diff）
- **モデル**: 相関ネットワーク + logistic regression
- **貢献**: *Clostridium scindens* が C.diff 抵抗性に keystone であることを予測・検証

### Venturelli et al. 2018 — Molecular Systems Biology
- **タイトル**: Deciphering microbial interactions in synthetic human gut microbiome communities
- **データ**: **in vitro** 12 種合成コミュニティ（腸内菌）× 全組み合わせ単培養〜12種混合
- **モデル**: gLV（相互作用行列の直接測定に近い）
- **意義**: 制御された環境で A 行列の実験的検証 → in vivo 研究の benchmark

---

## 5. Hamilton / NSP 系 (本研究関連)

### Marsland et al. 2019 — Cell Systems (Consumer-Resource Model)
- **モデル**: MacArthur consumer-resource、Hamilton ODE の源流
- **貢献**: 種プール・ニッチ幅から群集構造を予測

### Niehaus et al. 2019 — Nature Communications
- **タイトル**: Microbial coexistence through chemical-mediated interactions
- **関連**: 代謝産物経由の間接相互作用、NSP の設計思想に近い

---

## 相対量 vs 絶対量 まとめ

| 手法 | データ形式 | 主な補正法 |
|------|---------|-----------|
| 16S (standard) | **相対量** (Σφᵢ=1) | CLR変換 / replicator / simplex ODE |
| 16S + flow cytometry | **絶対量** (cells/mL) | 総菌数 × 相対比 |
| 16S + spike-in | **絶対量** | spike-in ratio 補正 |
| metagenomics | **相対量** | 同上 |

In vivo 研究の **95% 以上は相対量**。
絶対量は flow cytometry との組み合わせが必要で、手間とコストが高い。
