# 研究方針メモ（NIFE/SIIRI B02：NGS × COMETS/dFBA × Bayesian UQ）

## 0. TL;DR

- **目的**：インプラント上の口腔バイオフィルムの dysbiosis（健常→病的シフト）を、
  NGS・in vitro 実験・計算モデルを繋いで理解し、
  早期検知指標と介入（抗菌薬・栄養・表面）の計算根拠を作る
- **パイプライン**：

```
患者 FASTQ (PRJEB71108, Dieckow et al. 2024)
  ↓ MetaPhlAn 4 (菌叢プロファイリング)
init_comp.json (初期菌叢組成)
  ↓ COMETS dFBA (AGORA GEMs × 5 種)
代謝プロファイル (乳酸, succinate, pH proxy)
  ↓
DI (Dysbiosis Index) + 病態ステージ分類
  ↓
抗菌薬・栄養・表面介入シミュレーション
  ↓
実験設計提案 (次に何を測るべきか)
```

---

## 1. 背景と動機

### 1.1 なぜ計算モデルが必要か

口腔バイオフィルムは「多菌種・相互作用・空間構造・環境勾配（O₂/pH/栄養）」が複雑に絡む。
NGS で"誰がいるか"は分かっても、**なぜそうなるか（因果）** を説明するのが難しい。

Dieckow et al. 2024 (NPJ Biofilms Microbiomes, PMID 39719447) は
インプラント上の初期バイオフィルムが「複雑・多様・被験者特異的・動的」であることを示した。
この subject-specificity こそ、**患者個別の初期組成を入力できる COMETS パイプライン**の動機になる。

### 1.2 Szafrański グループの関連研究

| 論文 | ポイント | 本研究との接続 |
|------|---------|----------------|
| Dieckow et al. 2024 (NPJ Biofilms) | CLSM × NGS、インプラント初期バイオフィルムの動態 | `spatial_dfba.py` の検証データ；init_comp の元データ |
| Heine et al. 2025 (Front. Oral Health, PMID 40978156) | 5 種 × flow/static 条件 × dysbiosis in vitro | モデル校正・検証の基準データ；DI 定義の根拠 |
| Doll-Nikutta et al. 2025 (J Dent Res, PMID 39629932) | バイオフィルム-インプラント界面の漸進的 pH 酸性化 | COMETS 代謝出力（乳酸↑）→ pH 低下の実験検証先 |
| Frings et al. 2025 (Analyst, PMID 40539919) | ATR-FTIR による菌株レベル同定 | strain-level の代謝多様性を DI に組み込む手がかり |
| Mukherjee et al. 2025 (Microbiol Spectr, PMID 40815221) | Pg バイオフィルム × 1,4-DHNA（抗菌化合物）× Junker/Klempt (IKM) | B02×B07 連携の実例；COMETS で薬剤介入を in silico 再現 |
| Szafrański et al. 2021 (Periodontology 2000, PMID 33690937) | ヒト口腔ファージオーム | phage–bacteria 相互作用を DI に将来追加する可能性 |
| Grischke et al. 2021 (BMC Oral Health, PMID 33794847) | 義歯が peri-implantitis リスク → periodontopathogens 増加 | 患者リスク層別化の根拠；init_comp の臨床解釈 |

---

## 2. データ整理

### 2.1 Heine 系（in vitro、5 菌種、flow vs static）

- **強み**：条件を制御できる（表面・flow/static・期間・菌セット）→ モデル校正に適する
- **readout**：biofilm volume/area、菌比率（CLSM/NGS）、pH、gingipain 活性
- **本研究での役割**：Monod/COMETS パラメータの校正ターゲット

### 2.2 Dieckow/NGS 系（患者・メタゲノム・biomarkers）

- **強み**：臨床に近い；被験者個別の初期組成が取れる
- **readout**：16S/メタゲノム（相対存在量）、炎症マーカー、probing depth
- **本研究での役割**：init_comp の供給源、モデルの外部検証

### 2.3 PRJEB71108（ENA、Shotgun metagenomics）

現在実行中のパイプライン：

```bash
# Step 1: MetaPhlAn 4 プロファイリング（PBS job 29511 実行中）
#   → nife/data/metaphlan_profiles/ に profile.txt が出る

# Step 2: feature_table.tsv 作成
merge_metaphlan_tables.py profiles/*.profile.txt > feature_table.tsv

# Step 3: 5 種 init_comp.json に変換
python nife/data/metaphlan_feature_table_to_init_comp.py \
  --feature-table feature_table.tsv \
  --sample ERR13166576_A_3 \
  --out init_comp_ERR13166576_A_3.json

# Step 4: COMETS Step C（PBS job でスケジュール済み）
python nife/comets/run_comets_pipeline.py --step C --init-comp init_comp_ERR13166576_A_3.json
```

---

## 3. COMETS/dFBA の役割

### 3.1 何を計算するか

各菌に **GEM（Genome-Scale Metabolic Model, AGORA）** を持たせ、  
共有培地から栄養を取る・代謝物を分泌する・別の菌がそれを食べる を反復することで、

- **Cross-feeding chains**：So/An → 乳酸 → Vp/Fn；An → succinate → Pg
- **pH proxy**：乳酸・酢酸生産量 → 局所 pH（Doll-Nikutta 2025 と比較可能）
- **菌叢動態**：dysbiosis への移行タイムコース
- **DI（Dysbiosis Index）**：Shannon entropy 規格化 = 多様性指標

を、パラメータとして設定した培地・初期組成から**機構的に導出**する。

### 3.2 A 行列モデルとの違い

| | A 行列モデル（Tmcmc202601）| COMETS/dFBA |
|---|---|---|
| 相互作用の表現 | 係数 a_ij として直接推定 | 代謝ネットワークから導出 |
| 強み | 少数パラメータ、TMCMC で後験推定 | 機構的、代謝物レベルで予測 |
| NIFE 文脈 | IKM 力学論文用（本研究外） | pH・薬剤介入の in silico 実験に適合 |

### 3.3 いま動いているもの

- `spatial_dfba.py`：Monod dFBA（GEM なし）、Dieckow 2024 データとの Validation 済み
- `run_comets_pipeline.py Step B`：COMETS GEM、10×20 グリッド、80h シミュレーション（PBS job 29730 実行中）
- `run_comets_pipeline.py Step C`：患者 init_comp → COMETS（MetaPhlAn 完了待ち）

---

## 4. 介入シミュレーション（B02 の "in silico 実験"）

Mukherjee et al. 2025 は COMETS を使わず実験のみで 1,4-DHNA の Pg 抑制を示した。
本研究はこれを **COMETS で in silico 再現し、さらに予測に使う**ことを目指す。

### 4.1 抗菌薬介入のモデル化

```python
# 例：metronidazole → 嫌気性菌（Pg, Fn）の growth rate に阻害係数を乗算
inhibition = 1.0 / (1.0 + conc / MIC)  # Hill-type
# COMETS では上限取り込み速度 (Vmax) を inhibition でスケーリング
sp_model.change_vmax(reaction_id="EX_glc_D_e", new_vmax=Vmax * inhibition)
```

### 4.2 介入シナリオ比較

| シナリオ | 変更パラメータ | 評価指標 |
|---|---|---|
| ベースライン | − | DI, pH proxy, 菌叢タイムコース |
| 低グルコース培地 | `glc_D[e]` ↓ | Pg 比率, 乳酸生産 |
| metronidazole | Vmax(Pg, Fn) × (1−α) | DI 低下速度 |
| 1,4-DHNA 表面 | Pg 初期接着率 ↓ | 定着抑制期間 |
| 患者 A_3 実測 init_comp | MetaPhlAn 出力 | 個別 DI 軌跡 |

### 4.3 アウトカム指標

- **DI < 0.5**：健常維持と定義（healthy attractor）
- **DI ≥ 0.7**：病的シフト（介入要）
- **Δ(DI) / Δt**：移行速度（個体差の捉え方）
- **pH proxy**：[乳酸] + [酢酸] の重み付き和（Doll-Nikutta 2025 の実験値とキャリブレーション）

---

## 5. Bayesian UQ の役割

### 5.1 何を推定するか

- Monod パラメータ（Km, Vmax, 各種）の posterior distribution
- 患者間 variability の分離：被験者固定効果 vs. 菌叢由来の variability

### 5.2 何が分からないかを定量化する

- Parameter identifiability：どのパラメータが時系列データで同定できるか
- 予測区間：「この患者の 7 日後 DI は [0.4, 0.8]」のような信頼区間
- OED（Optimal Experimental Design）：次に何を測れば uncertainty が最も減るか

### 5.3 ツール

- **校正**：TMCMC（`data_5species/core/tmcmc.py`）または PyMC3
- **Sensitivity**：Sobol 指数（`sweep_comets_0d.py` を拡張）
- **検証**：LOO-CV，posterior predictive check

---

## 6. 12 週間スコープ（現実的）

| 週 | タスク | 成果物 |
|---|---|---|
| 1–2 | データ確認・モデルベースライン（Heine in vitro） | Monod fit, DI タイムコース再現 |
| 3–4 | COMETS Step B/C 稼働確認、患者 init_comp パイプライン | 患者別 DI 予測 1 例 |
| 5–7 | Bayesian 校正（Vmax/Km posterior） | 予測区間付き DI タイムコース |
| 8–9 | 介入シミュレーション（metronidazole / 低 glc） | 介入効果の in silico 比較図 |
| 10–11 | OED：次の実験提案 | "このタイムポイントを測れ"の定量的根拠 |
| 12 | 再現可能パイプライン整備 + 1 枚スライド | GitHub repo + 技術レポート |

---

## 7. 面接・説明用（英語）

> "Szafrański's group has shown that peri-implant biofilm composition is subject-specific and dynamic
> (Dieckow 2024), and that local acidification correlates with dysbiosis (Doll-Nikutta 2025).
> My contribution is to connect patient NGS data via MetaPhlAn to a COMETS dFBA simulation,
> producing patient-specific DI trajectories and in-silico intervention predictions—
> for example, how metronidazole or low-glucose conditions shift the community away from a dysbiotic attractor.
> Uncertainty quantification via Bayesian calibration will tell us how confident these predictions are
> and what the next most informative experiment would be."

---

## 8. 用語ミニ

| 用語 | 意味 |
|---|---|
| NGS | DNA を大量に読む（16S / メタゲノム / メタトランスクリプトーム） |
| GEM | Genome-Scale Metabolic Model（代謝ネットワーク全体） |
| dFBA | Dynamic Flux Balance Analysis（代謝ネットワークで時間発展） |
| COMETS | dFBA を空間グリッドで動かすフレームワーク |
| AGORA | 口腔・腸内細菌の GEM ライブラリ |
| DI | Dysbiosis Index = 規格化 Shannon entropy（0=均一, 1=単菌支配） |
| Monod | 栄養濃度に依存した成長速度式（μ = μmax·S/(Km+S)） |
| OED | Optimal Experimental Design（情報量最大の実験点を探す） |
