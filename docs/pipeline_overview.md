# Replicator ODE × Oral Microbiome — Pipeline Overview
**Date**: 2026-05-07  
**Project**: NIFE / Szafranski collaboration

---

## 概要

口腔マイクロバイオームの縦断16Sデータに対して、gLV（generalized Lotka-Volterra）および Hamilton replicator モデルを当てはめ、leave-one-patient-out (LOO) クロスバリデーションで予測性能を評価するパイプライン。

```
16S raw reads (SRA/ENA)
    ↓  vsearch: merge PE → QC → chimera removal → SILVA classification
Guild-level phi array  (N_patients × N_timepoints × 10_guilds)
    ↓  Hamilton / gLV LOO-CV with Szafranski sign prior
LOO-RMSE, Bray-Curtis
```

---

## 1. 16S前処理パイプライン

### 1.1 使用ツール
| ツール | バージョン | 用途 |
|--------|-----------|------|
| vsearch | 2.30.5 | 全ステップ（merge/QC/chimera/classify） |
| SILVA 138.1 SSU NR99 | — | 分類参照データベース |
| cutadapt | 5.2 | プライマートリミング（必要時） |

### 1.2 ステップ別説明

**Step 1: Paired-end merge**
```
vsearch --fastq_mergepairs R1.fastq.gz --reverse R2.fastq.gz
    --fastq_minovlen 50 --fastq_maxdiffs 5
```
- 2本のリードをオーバーラップ領域で1本に連結
- 16S V1-V3 (~500bp): 250bp × 2 で十分オーバーラップ

**Step 2: Quality filter**
```
vsearch --fastq_filter merged.fastq.gz
    --fastq_maxee 1.0   # 期待エラー数 ≤ 1
    --fastq_minlen 200  --fastq_maxlen 650
    --relabel "SRR_ID."  # サンプルIDをリード名に埋め込む
```
- Expected Error法で品質フィルタ
- `--relabel` でサンプルIDを各リードに付与 → 後で一括処理可能

**Step 3: 全サンプル結合**
```
cat filtered/*.fasta.gz > all_reads.fasta.gz
```
- 全サンプルのリードを1ファイルに集約 → 分類を一括実行（速い）

**Step 4: Chimera removal**
```
vsearch --uchime_ref all_reads.fasta.gz --db SILVA_138.fasta.gz
    --nonchimeras all_reads_nochim.fasta.gz
```
- SILVA参照ベースのキメラ検出

**Step 5: SILVA分類**
```
vsearch --usearch_global all_reads_nochim.fasta.gz
    --db SILVA_138.fasta.gz --id 0.80
    --blast6out all_taxonomy.blast6
```
- global alignmentで最も近いSILVA配列を検索（identity ≥ 80%）
- blast6out: `query_id  subject_id  identity  ...`
- subject_id にSILVA分類群が入っている例:
  `AB000123.1.1500 Bacteria;Firmicutes;Bacilli;Lactobacillales;...`

**Step 6: Guild集計**
```python
# query_id = "SRR14643440.1234"  → run_id = "SRR14643440"
# subject → class level → guild (10 guilds)
```
- run_id → (patient, timepoint) のマッピングをmanifestから参照
- Class→Guild マッピング例:
  - Bacilli, Lactobacillales → `Bacilli`
  - Fusobacteriia, Fusobacteriales → `Fusobacteriia`
  - Negativicutes, Selenomonadales → `Negativicutes`
- 各患者・タイムポイントで歯/部位を平均してcomposition (Σ=1) に正規化

**出力**: `phi_guild.npy`  shape = `(N_patients, N_timepoints, 10)`

---

## 2. Guild定義（10 guilds）

Dieckow et al. と共通のguild定義を全データセットで使用。

| Guild | 対応するClass/Order (SILVA) |
|-------|---------------------------|
| Actinobacteria | Actinobacteria, Actinomycetia |
| Bacilli | Bacilli |
| Bacteroidia | Bacteroidia |
| Betaproteobacteria | Betaproteobacteria |
| Clostridia | Clostridia, Erysipelotrichia |
| Coriobacteriia | Coriobacteriia |
| Fusobacteriia | Fusobacteriia, Fusobacteriales |
| Gammaproteobacteria | Gammaproteobacteria |
| Negativicutes | Negativicutes, Selenomonadales, Veillonellales |
| Other | 上記以外 |

---

## 3. LOO-CVパイプライン

### 3.1 モデル

**gLV replicator**（非対称A行列、scipy solve_ivp）
```
dφᵢ/dt = φᵢ ( bᵢ + Σⱼ Aᵢⱼφⱼ - φᵀ(b + Aφ) )
```
- 対角 Aᵢᵢ ≤ 0（自己制限）
- 1ステップ = 各データセットの採取間隔

**Hamilton replicator**（対称A行列、JAX ODE）
- A[i,j] = A[j,i]（無向相互作用）

### 3.2 Sign prior（Szafranski 2025 由来）

Szafranski et al. の実験的代謝フロー + KEGG/HMDB から推定した菌種間相互作用の符号を事前情報として使用。
```
penalty = w × max(0, -sign(net)[i,j] × A[i,j])² / (2σ²)
```
- w = |net_flow[i,j]|（確信度重み）
- σ = 0.15
- α = competition_weight（搾取競争の強度、0〜0.5）
- AGORA2 FBA（L3層）はオプション

### 3.3 LOO手順

1. N-1 患者でA, {bₖ} を最適化（L-BFGS-B）
2. held-out患者のb_holdを固定Aで最適化
3. t=0 から積分して t=1,...,T-1 を予測
4. RMSE = √(Σ‖φ̂ - φ‖² / ((T-1)×N_G))
5. BC = (1/(T-1)) × Σ (Σᵢ|φ̂ᵢ-φᵢ|/2)

---

## 4. データセット一覧

### 4.1 Dieckow et al. (Szafranski collaboration)
| 項目 | 値 |
|------|-----|
| 患者数 N | 10 |
| タイムポイント T | 3（W1, W2, W3） |
| 採取間隔 | 1週間 |
| 部位 | 上顎前歯部歯肉縁下プラーク |
| 疾患 | 初期う蝕（回復後） |
| phi array | `results/dieckow_otu/phi_guild.npy` |

**LOO-CV 完了結果（主要）**:
| モデル | α | RMSE | BC |
|--------|---|------|-----|
| Hamilton | 0.00 | 0.0567 | — |
| Hamilton+AGORA | 0.00 | 0.0562 | — |
| gLV | 0.25 | **0.0490** | 0.163 |

### 4.2 Botelho et al. 2021 (PRJNA725874, BMC Biology)
| 項目 | 値 |
|------|-----|
| 患者数 N | 15 |
| タイムポイント T | 7（0, 2, 4, 6, 8, 10, 12 ヶ月） |
| 採取間隔 | 2ヶ月 |
| 部位 | 歯周ポケット（Stable/Progressing/Fluctuant × 複数歯） |
| 疾患 | 慢性歯周炎 |
| phi array | `data/prjna725874/phi_guild.npy` （生成中）|

**前処理ジョブ**: PBS 40077 (frontale04, ppn=12)
- Step 1: ENA FTPからFASTQダウンロード（13.2 GB, 329サンプル）
- Step 2: vsearch merge + QC
- Step 3: SILVA分類（全サンプル一括）
- Step 4: guild phi配列生成

**LOO-CVスクリプト**: `run_glv_loo_725874.py`  
**投入スクリプト**: `submit_loo_725874.sh`（前処理完了後に実行）

---

## 5. スクリプト一覧

```
nife/
├── guild_replicator_dieckow.py      # 共通guild定義・モデルヘルパー
├── build_net_flow_expanded.py       # Szafranski sign prior構築
│
├── run_hamilton_expanded_loo.py     # Hamilton LOO-CV (Dieckow)
├── run_glv_loo.py                   # gLV LOO-CV (Dieckow)
├── run_glv_loo_725874.py            # gLV LOO-CV (PRJNA725874, 7tp)
│
├── submit_loo_alpha_scan.sh         # Dieckow Hamilton 投入
├── submit_glv_loo_alpha_scan.sh     # Dieckow gLV 投入
├── submit_baseline_loo.sh           # no-prior baseline 投入
├── submit_loo_725874.sh             # PRJNA725874 LOO 投入
│
├── collect_loo_alpha.py             # 結果集計・比較表
├── preprocess_prjna725874.sh        # 16S前処理PBSジョブ
├── build_guild_phi_725874.py        # SILVA出力→guild phi
│
└── data/silva_db/
    ├── SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz
    └── silva138_ssu_nr99.mmi        # minimap2インデックス（未使用）
```

---

## 6. 注意点・限界

### 前処理に関して
- vsearch の `--id 0.80` (80% identity) は genus レベル分類に適切。species levelには 97% が必要だが、guild (class) レベルには十分。
- 元論文 (Botelho 2021) は Kraken2 + 専用口腔マイクロバイオームDBを使用。vsearch+SILVAによる再解析は近似値。
- プライマー情報不明のため、プライマートリミングをスキップ。末端配列が多少ノイズになる可能性あり。

### モデルに関して
- gLV > Hamilton（RMSE差 -13.5%）: Hamiltonの対称A仮定が制限的
- Sign prior はDieckow（う蝕、歯肉縁下プラーク）由来 → PRJNA725874（歯周炎）への転用は近似
- α=0.25 が gLV の最適（Dieckow）→ PRJNA725874 でも確認が必要

### PRJNA575550 について
- 当初「第3候補」として検討したが、SRAで1子供1サンプル（横断データ）と判明 → 縦断LOO-CVには不適。除外。
