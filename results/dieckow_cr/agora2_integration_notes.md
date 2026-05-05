# AGORA2 Integration — Sign Prior Enhancement

**Date**: 2026-05-05  
**Status**: フィット実行中

---

## 背景

Dieckow et al. (2024) の口腔縦断データ（10患者×3週）に対して Hamilton ODE + 符号制約プライアーを適用している。
プライアーの符号は Szafrański (2025 preprint) Supplementary File 1（L1/L2 層）から構築してきたが、
AGORA2 ゲノムスケール代謝モデルを用いた FBA 由来シグナル（L3 層）を追加した。

## AGORA2 とは

- **Heinken et al. 2023** — *Nature Biotechnology* 41:1320–1331  
  DOI: [10.1038/s41587-022-01628-0](https://doi.org/10.1038/s41587-022-01628-0)
- 7,302 株のゲノムスケール代謝再構成（GEM）
- v1（773株・腸内のみ）から大幅拡張し、**口腔菌を含む複数部位**をカバー
- モデルファイル: `nife/data/homd_db/agora_gems/` に 10 ギルド分を収録

## 3層プライアー構成

| 層 | ソース | 重み | 根拠 |
|---|---|---|---|
| L1 | Szafrański Suppl. File 1（実験的エビデンス、KEGG/HMDB注釈あり） | 2.0 | 直接観測 |
| L1 | Szafrański Suppl. File 1（実験的エビデンス、注釈なし） | 1.5 | 直接観測 |
| L2 | Szafrański Suppl. File 1（予測エビデンス） | 1.0 | 計算的予測 |
| L3 | **AGORA2 FBA クロスフィーディング** | **1.0** | ゲノムスケール代謝モデル（92% gLV符号一致） |

## L3 実装：AGORA2 FBA パイプライン

### 手順
1. 各ギルドの代表 AGORA2 モデルを口腔液培地でロード
2. pFBA を実行し分泌プロファイルを取得
3. ギルド j が分泌する代謝物をギルド i が取り込める → `net_flow[i, j] += 0.5`（正の相互作用）
4. 毒素（H₂O₂, H₂S）の分泌 → `net_flow[i, j] -= 0.5`（負の相互作用）

### 口腔液培地の設定 (`guild_agora_signs.py`)

AGORA2 の交換反応 ID 形式: `EX_{met}(e)`（旧 BiGG 形式、`_e` サフィックスではない）

主な成分（Dawes 2008, Amerongen & Veerman 2002 に基づく）：

| カテゴリ | 代謝物 | 備考 |
|---|---|---|
| 糖質 | glucose, fructose, sucrose, lactate | 主炭素源 |
| アミノ酸 | 20種 + Cys + Orn | Cys はモデル必須 |
| ビタミン B | thiamine, riboflavin, niacin, pantothenate, folate, pyridoxine/pyridoxamine, B12 | B6 は 3 形態必要 |
| 微量元素 | Fe²⁺, Fe³⁺, Mg²⁺, Ca²⁺, Zn²⁺, Mn²⁺, Co²⁺, **Cu²⁺** | Cu 必須（シトクロム） |
| 細胞壁前駆体 | **meso-DAP**, stearate (C18), myristate (C14), laurate (C12) | Gram+ / 偏性嫌気性菌に必要 |
| グルタチオン | 酸化型 + 還元型 | Actinobacteria / Bacteroidia |
| アミノ酸誘導体 | ornithine, Cys-Gly ジペプチド, cytidine | Coriobacteriia / Flavobacteriia |
| キノン類 | menaquinone-7/8, ubiquinone-8, protoheme, siroheme, adenosylcobalamin | 電子伝達鎖 |
| ポリアミン | putrescine, spermidine | 微量 |

**全 10 ギルドで正の増殖率を確認**（μ = 0.11–1.66 h⁻¹、Fusobacteriia 最高）

### AGORA2 符号 vs gLV A 行列との一致率

```
72ペア予測 → 66/72 一致 (92%)
```

gLV A の符号と AGORA2 FBA 由来符号が 92% 一致 → 代謝ネットワーク情報が生態的相互作用の符号を説明できることを示す。

### プライアー規模の変化

| 構成 | 制約ペア数（無向） |
|---|---|
| L1+L2 のみ（旧） | 11 |
| L1+L2+L3 AGORA2 | **43** |

90 クロスフィーディングシグナル → 43 の有向ペアに集約

## ギルド代表株（AGORA2）

| ギルド | 代表株 |
|---|---|
| Actinobacteria | *Actinomyces naeslundii* str. Howell 279 |
| Coriobacteriia | *Atopobium parvulum* DSM 20469 |
| Bacilli | *Streptococcus gordonii* str. Challis CH1 |
| Clostridia | *Parvimonas micra* ATCC 33270 |
| Negativicutes | *Veillonella parvula* Te3 DSM 2008 |
| Bacteroidia | *Prevotella melaninogenica* ATCC 25845 |
| Flavobacteriia | *Capnocytophaga sputigena* ATCC 33612 |
| Fusobacteriia | *Fusobacterium nucleatum* subsp. nucleatum ATCC 25586 |
| Betaproteobacteria | *Eikenella corrodens* ATCC 23834 |
| Gammaproteobacteria | *Haemophilus parainfluenzae* T3T1 |

## 新規性

| 観点 | 評価 |
|---|---|
| AGORA2 自体 | 既存 (Heinken 2023) |
| AGORA2 を口腔ギルド符号プライアーとして使う | **新規** — FBA cross-feeding → 生態的 A 行列符号への変換 |
| Hamilton ODE + AGORA2 サイン制約 | **新規** |
| Szafrański ギルド分類 + Dieckow 口腔臨床データへの適用 | **新規** |

## フィット結果（更新予定）

| 構成 | 訓練 RMSE | Pearson r | SA | ペア数 | LOO-CV RMSE |
|---|---|---|---|---|---|
| Replicator + Dieckow prior (L1+L2) | 0.083 | — | — | 22 | 0.0770 |
| Replicator + expanded prior (L1+L2, 34 pairs) | **0.0631** | 0.938 | 64/68 (94%) | 34 | **0.0516** |
| **+ AGORA2 L3, W=0.5** | 0.0661 | 0.932 | 66/70 (94%) | 35 | — |
| **★ + AGORA2 L3, W=1.0** | **0.0565** | **0.951** | **70/70 (100%)** | 35 | running |
| **+ AGORA2 L3, W=1.5** | 0.0597 | 0.945 | 66/70 (94%) | 35 | — |
| **+ AGORA2 L3, W=2.0** | 0.0575 | 0.949 | 70/70 (100%) | 35 | — |

**W=1.0 が最適** — RMSE 10%改善、SA 100%達成。LOO-CV running (vancouver01/02)  
出力ファイル: `fit_glv_hamilton_kegg_expanded_agora_w{0p5|1p0|1p5|2p0}.json`

## MacArthur Consumer-Resource Prior（定量的拡張）

### 理論的根拠

MacArthur 1970 および Marsland et al. 2019 (*PLOS Comput Biol*) より、gLV の $A_{ij}$ は消費者-資源モデルから導出できる：

$$A_{ij} = \underbrace{\sum_\alpha s_{j\alpha} \cdot c_{i\alpha}}_{\text{cross-feeding (+)}} - \underbrace{\frac{\sum_\alpha c_{i\alpha} \cdot c_{j\alpha}}{|\mathbf{c}_i||\mathbf{c}_j|}}_{\text{niche overlap (−)}}$$

AGORA2 pFBA が $s_{j\alpha}$（分泌フラックス）と $c_{i\alpha}$（実測 uptake フラックス）を与える。

### Φ 行列の実装

```python
# 1. pFBA 分泌 (s_{jα}) — 実際に分泌される代謝物フラックス
secs_j = {met: flux for rxn in exchanges if flux > 1e-6}

# 2. pFBA 実測 uptake (c_{iα}) — 実際に取り込む代謝物フラックス
cap_i  = {met: |flux| for rxn in exchanges if flux < -1e-6}

# 3. Cross-feeding: Σ_α s_{jα} · min(c_{iα}, s_{jα})
phi_cf[i,j] = Σ_{α∈secs_j ∩ cap_i} secs_j[α] × min(cap_i[α], secs_j[α])

# 4. Competition: cosine similarity of uptake vectors
phi_comp[i,j] = (c_i · c_j) / (|c_i| × |c_j|)

# 5. Net MacArthur Phi
phi_net[i,j] = phi_cf_normalized[i,j] − phi_comp[i,j]
```

**除外代謝物**: h₂o, h, co₂, o₂, Na⁺, K⁺, Cl⁻, Pi, SO₄²⁻（非生態的に無情報）

### Φ 行列の結果（口腔液培地, pFBA）

| | 正（クロスフィーディング） | 負（競合） |
|---|---|---|
| ペア数 | **6** | 84 |
| Φ 範囲 | [0.11, 0.96] | [−1.00, −0.01] |

**主要クロスフィーディングペア**:
- Flavobacteriia → Bacilli: Φ = 0.955
- Flavobacteriia → Negativicutes: Φ = 0.708
- Flavobacteriia → Betaproteobacteria: Φ = 0.681
- Gammaproteobacteria → Bacilli: Φ = 0.283

### 損失関数

$$\mathcal{L}_\text{MacArthur} = \sum_{i \neq j, m_{ij}} \frac{(A_{ij} - c \cdot \Phi_{ij})^2}{2\sigma^2}$$

- $c$: スケーリング定数（FBA単位 → $A$ 単位変換）、グリッドサーチ: $c \in \{0.01, 0.05, 0.1, 0.5\}$
- $\sigma = 0.5$: 事前分布の広さ
- $m_{ij}$: FBAモデルが存在するペアのみ（90ペア）

### フィット結果（更新予定）

| 構成 | $c$ | 訓練 RMSE | Pearson $r$ | SA |
|---|---|---|---|---|
| L1+L2+L3 sign prior (W=1.0) | — | TBD | TBD | TBD |
| **MacArthur prior** | 0.01 | TBD | TBD | TBD |
| **MacArthur prior** | 0.05 | TBD | TBD | TBD |
| **MacArthur prior** | 0.1 | TBD | TBD | TBD |
| **MacArthur prior** | 0.5 | TBD | TBD | TBD |

### 文献

- MacArthur R (1970) *Theor Pop Biol* 1:1–11 — consumer-resource → gLV 導出
- Marsland R et al. (2019) *PLOS Comput Biol* 15:e1006793 — Community Simulator
- Friedman J et al. (2017) *PNAS* 114:E2149 — マイクロバイオームへの応用

## 関連ファイル

| ファイル | 説明 |
|---|---|
| `guild_agora_signs.py` | AGORA2 FBA 符号検証、`ORAL_MEDIUM`、`apply_medium` |
| `build_net_flow_expanded.py` | L1+L2+L3 プライアー行列構築 |
| `run_hamilton_kegg_expanded.py` | Hamilton ODE フィット（`--no-agora` で L3 無効化） |
| `data/homd_db/agora_gems/` | AGORA2 XML モデル（10 ギルド） |
| `results/dieckow_cr/agora_sign_comparison.json` | AGORA vs gLV 符号比較 |
| `results/dieckow_cr/fig_agora_sign_validation.png` | 検証図（クロスフィーディング行列） |
