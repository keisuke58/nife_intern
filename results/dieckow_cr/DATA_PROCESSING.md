# Dieckow 2024 — データ処理パイプライン

## 概要

Dieckow et al. 2024 (*npj Biofilms and Microbiomes*) の peri-implant 16S アンプリコン
データから 10-guild gLV モデルをフィットするまでの全処理手順。

---

## 1. 元データ

| 項目 | 内容 |
|------|------|
| 論文 | Dieckow et al. 2025, *npj Biofilms Microbiomes* |
| データベース | ENA: PRJEB71108 |
| シーケンサー | PacBio (long-read 16S amplicon) |
| サンプル数 | 30 
= 10 患者 × 3 週 (W1/W2/W3) |
| Patient　患者 ID | A, B, C, D, E, F, G, H, K, L |
| ERR 番号 | ERR13166574–ERR13166603 (manifest: `dieckow_manifest.tsv`) |
| FASTQ 保存先 | `data/` (`.fastq.gz`) |

---

## 2. CCS コンセンサス生成 & 属レベル分類

**スクリプト**: `dieckow_ccs_pipeline.py`
**出力**: `results/dieckow_taxonomy/{sample}_taxonomy.tsv`

### 処理手順

```
PacBio FASTQ (subreads)
  │
  ├─ 1. ZMW ID でサブリードをグループ化
  │       (FASTQ ヘッダー: @m*/zmw/{id}/sub/{start}_{end})
  │
  ├─ 2. minimap2 (-a --secondary=no -t 8) → SILVA 138.1 NR99 (.mmi index)
  │       各サブリードのストランド方向を決定
  │
  ├─ 3. ZMW ごとにサブリードを多数決ストランドへ reverse-complement
  │       → SPOA でコンセンサス配列を生成
  │
  ├─ 4. VSEARCH --usearch_global (97% identity) → SILVA NR99
  │       属レベル分類を付与
  │
  └─ 5. 属ごとに集計 → TSV 出力
```

### TSV 形式 (例: `A_1_taxonomy.tsv`)

```
sample  genus           count   percent
A_1     Bifidobacterium  3258   42.90
A_1     Streptococcus    1091   14.37
A_1     Atopobium         565    7.44
...
```

- 1 サンプルあたり ~7,000–15,000 リード、~20–40 属
- **例外**: 患者 D の W1 (D_1) はヒト DNA コンタミ 97.6% のため bacterial reads = 452 のみ（他サンプルの ~1/15）

### 使用ツール & DB

| ツール | バージョン | 用途 |
|--------|-----------|------|
| minimap2 | 2.26 | サブリード → SILVA アライメント |
| SPOA | 4.1 | コンセンサス生成 |
| VSEARCH | 2.23 | OTU クラスタリング (97%) |
| SILVA | 138.1 NR99 | リファレンス DB |

---

## 3. 5-種 phi 行列の生成

**スクリプト**: `aggregate_dieckow_otu.py`
**出力**: `results/dieckow_otu/phi_obs_raw.npy` (10 × 3 × 5)

Hamilton ODE の 5 菌種に対応する属を抽出・集計:

| モデル変数 | 属 |
|-----------|-----|
| So | *Streptococcus* |
| An | *Actinomyces* |
| Vd | *Veillonella* |
| Fn | *Fusobacterium* |
| Pg | *Porphyromonas* |

5 種の合計が 1 になるように行正規化。

---

## 4. 10-guild phi 行列の生成

**スクリプト**: `aggregate_dieckow_guilds.py`
**出力**: `results/dieckow_otu/phi_guild.npy` (10 × 3 × 10)

Dieckow Fig. 4a の細菌クラス分類に従い、全属を 10 ギルドへマッピング:

| ギルド | 含まれる主な属 |
|--------|--------------|
| Actinobacteria | *Actinomyces*, *Rothia*, *Bifidobacterium*, *Corynebacterium*, *Slackia* |
| Bacilli | *Streptococcus*, *Gemella*, *Granulicatella*, *Abiotrophia* |
| Bacteroidia | *Prevotella*, *Porphyromonas*, *Capnocytophaga*, *Tannerella* |
| Betaproteobacteria | *Neisseria*, *Eikenella*, *Aggregatibacter*, *Kingella* |
| Clostridia | *Parvimonas*, *Peptostreptococcus*, *Oribacterium* など |
| Coriobacteriia | *Atopobium*, *Olsenella*, *Cryptobacterium* |
| Fusobacteriia | *Fusobacterium*, *Leptotrichia* |
| Gammaproteobacteria | *Haemophilus*, *Pseudomonas* |
| Negativicutes | *Veillonella*, *Dialister*, *Selenomonas* |
| Other | *Campylobacter*, *Treponema* など残余 |

各週のベクトルを合計 1 に正規化。

### 患者 W1 の支配ギルド

| 患者 | 支配ギルド | 割合 |
|------|-----------|------|
| A | Actinobacteria | 0.48 |
| B | Bacilli | 0.38 |
| C | Bacilli | 0.34 |
| D | Bacilli | 0.93 |
| E | Bacilli | 0.53 |
| F | Actinobacteria | 0.62 |
| G | Bacilli | 0.77 |
| H | Bacilli | 0.57 |
| K | Bacilli | 0.74 |
| L | Bacilli | 0.91 |

---

## 5. gLV フィッティング

**スクリプト**: `fit_guild_replicator.py`
**PBS ジョブ**: `guild_fit_job.sh`
**出力**: `results/dieckow_cr/fit_guild.json`

### モデル

Replicator 型 gLV (simplex 上で Σφ_i = 1 を保存):

```
dφ_i/dt = φ_i * (Σ_j A_ij φ_j + b_i − f̄)
f̄ = Σ_i φ_i (Σ_j A_ij φ_j + b_i)
```

### パラメータ

| パラメータ | 形状 | 説明 |
|-----------|------|------|
| A | (10, 10) | 非対称ギルド間相互作用行列 (対角 ≤ 0) |
| b_all | (10, 10) | 患者別固有成長率 |
| 合計 | 200 | 10 患者 × 2 予測 × 10 ギルド = 200 残差 |

### 最適化

| バージョン | 方法 | λ | RMSE | 備考 |
|-----------|------|---|------|------|
| v1 (fit_guild_jax.py) | JAX Adam (5000 epochs, lr=0.003) | 1e-3 | 0.0549 | 現在の fit_guild.json |
| v2 (fit_guild_replicator.py) | scipy L-BFGS-B | 1e-4 | 更新待ち (job 39876) | A の零要素が多い問題に対処 |

共通設定:
- **制約**: 対角 A_ii ≤ 0（自己抑制）
- **初期値**: 複数ランダムスタート + 対角 −0.1 warm start
- **予測**: W1 → W2 → W3 を RK45 で積分 (Δt = 1 week)

### 現在の結果 (v1, JAX Adam λ=1e-3)

| 指標 | 値 |
|------|-----|
| RMSE | 0.0549 |
| r (pred vs obs) | 0.954 |

---

## 関連研究（Faust / Thiele / COMETS）と本研究への接続

### 10-guild の色（使い回し）

- 10-guild の順序と色は [guild_replicator_dieckow.py](file:///home/nishioka/IKM_Hiwi/nife/guild_replicator_dieckow.py) に集約:
  - `GUILD_ORDER`
  - `GUILD_COLORS`（辞書）
  - `GUILD_COLORS_LIST`（`GUILD_ORDER`順のリスト）

### 論文・資料の取得先

- 保存先: `results/dieckow_cr/related_work_papers/`
  - `Faust_CoNet_app_F1000Research_2016.fulltext.xml`（Europe PMC fullTextXML）
  - `Wang_Microbiome_Modelling_Toolbox_2_0_Bioinformatics_2022.fulltext.xml`（Europe PMC fullTextXML）
  - `Heirendt_COBRA_Toolbox_v3_NatProtocols_2019.metadata.json`（Europe PMC metadata）

### 本研究での使い方（最短ルート）

- Faust/CoNet（association network）
  - 目的: 「候補辺集合（スパースなマスク）」を作り、10-guild gLV の A 行列推定の同定性を上げる（探索空間を狭める）。
  - 運用: 10-guild（または genus）×30サンプルの表から複数指標でエッジ候補を出し、合意したものだけを prior / constraint に反映。
- Thiele/COBRA・Microbiome Modelling Toolbox（constraint-based metabolic modeling）
  - 目的: gLV の “相互作用係数” を、代謝物の収支・交換（cross-feeding / competition）として機構的に裏づける。
  - 運用: Dieckow の代謝物ネットワーク（USES/PRODUCES 等）を起点に、交換される代謝物の符号（競合/相利）と A の符号比較を行う。
- COMETS（代謝モデル＋空間/離散時間）
  - 目的: 「代謝＋環境制約（拡散/空間）」まで含めたシミュレーションで、時系列・空間構造を説明できるか検証する。
  - 運用: COBRA系で整備したコミュニティモデルを入力にして、酸素など環境勾配や代謝物拡散を入れる。

---

## 6. ファイル構造まとめ

```
nife/
├── dieckow_manifest.tsv               # 30サンプル × ERR番号
├── data/
│   ├── ERR13166574.fastq.gz           # 生 PacBio FASTQ
│   └── ...
├── dieckow_ccs_pipeline.py            # Step 2: CCS → taxonomy
├── aggregate_dieckow_otu.py           # Step 3: 5-種 phi
├── aggregate_dieckow_guilds.py        # Step 4: 10-guild phi
├── fit_guild_replicator.py            # Step 5: gLV フィット
├── guild_fit_job.sh                   # PBS ジョブ
└── results/
    ├── dieckow_taxonomy/
    │   └── {P}_{W}_taxonomy.tsv       # 30ファイル (属 × リード数)
    ├── dieckow_otu/
    │   ├── otu_matrix.csv             # 30サンプル × 全属
    │   ├── phi_obs_raw.npy            # (10, 3, 5) 5-種
    │   ├── phi_guild.npy              # (10, 3, 10) 10-guild ★
    │   └── guild_summary.json
    └── dieckow_cr/
        ├── fit_guild.json             # A, b_all, RMSE
        ├── guild_glv_paper.pdf        # 論文図
        └── DATA_PROCESSING.md        # 本ファイル
```
