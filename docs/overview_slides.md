---
title: "プロジェクト全体像：口腔バイオフィルムの群集動態モデリング"
subtitle: "生態 ODE 推定・代謝シミュレーション・空間拡張を束ねる傘"
author: "西岡佳祐 — NIFE"
date: "2026-06-03"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## 研究の目的と概要

ペリインプラント炎に関わる**口腔バイオフィルム群集の動態**をモデル化する
（SIIRI コンソーシアム）。

- 健常（commensal）から疾患（dysbiotic）への遷移を、群集組成の時間発展として記述する。
- 中心的命題：**ゲノムから計算した代謝が、生態的相互作用の符号を制約する**。

\vspace{0.4em}
本デッキは 3 つの緩結合したモデリングの柱と、それらを結ぶ
データ・モデルの共通契約を俯瞰する。

---

## 概念図（全体像）

![](results/figures/concept_overview_pub.png){ height=66% }

代謝が相互作用の**符号**を決め、dysbiosis は**リワイヤリング＋空間再編**（Pg の中心化・深部沈降）として現れる。

---

## 統一オブジェクト — 10 ギルド分類

全データ・全モデルが共有する正典 = **class レベルの 10 ギルド分類**
（`GUILD_ORDER`）。

- `.npy` 配列と JSON は、この順序で**位置インデックス**される。
- 各ギルドは代表 AGORA 株を 1 つ持つ（例：Bacilli = *S. gordonii*、
  Negativicutes = *V. parvula*、Bacteroidia = *P. melaninogenica*、
  Fusobacteriia = *F. nucleatum*）。
- 16S 組成 $\varphi$、相互作用行列 $A\in\mathbb{R}^{10\times10}$、符号 prior
  $P\in\{-1,0,+1\}^{10\times10}$ — すべて同じ 10 軸で揃う。

\vspace{0.3em}
この共通分類が、生態・代謝・前処理の 3 つの柱を 1 つの言語で結ぶ。

---

## 3 つの緩結合モデリングの柱

1. **生態 ODE 推定（論文本体）** — 縦断 16S から gLV / Hamilton
   相互作用行列を推定。**代謝符号 prior** で正則化し、
   leave-one-patient-out CV（LOO-CV）で検証。
2. **COMETS / dFBA** — ゲノムスケール（AGORA）の機械論的群集シミュレーション。
   推定された相互作用を独立に交差検証。
3. **16S / メタデータ前処理** — 生シーケンス + 論文補遺
   $\to$ ギルド存在量配列。

\vspace{0.4em}
3 本は独立に走るが、**10 ギルド分類**と
**`fit_*.json` という交換フォーマット**で接続される。

---

## データフロー（実データ）

![](results/figures/pipeline_overview_pub.png){ height=66% }

実 Dieckow データ：guild $\varphi$（10 患者 $\times$ 3 週）→ AGORA 符号 prior →
gLV/Hamilton フィット（$\hat A$, $r=0.95$, SA 70/70）→ LOO-CV 検証 → 4 アトラクター。
`fit_*.json` が交換フォーマット。

---

## コアモデル（簡潔版）

絶対量 $x_i$ の一般化 Lotka–Volterra：
$$\dot{x}_i = x_i\Big(b_i + \sum_{j=1}^{S} A_{ij}\,x_j\Big),\qquad S=10 .$$

単体 $\sum_i\varphi_i=1$ 上の replicator / Hamilton 形（16S は組成データ）：
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big].$$

**符号 prior**：AGORA cross-feeding フラックス $F_{ij}$ から
$$P_{ij}=\sgn(F_{ij})\in\{-1,0,+1\}$$
を作り、$A$ の**符号のみ**を制約（大きさは設計上捨てる）。

---

## データセット

| 名称 | Accession | 形 | 役割 |
|---|---|---|---|
| Dieckow 2024 | PRJEB71108 | 10 患者 $\times$ 3 週 | 主要な縦断 fit / LOO |
| Duran-Pinedo 2021 | PRJNA725874 | 15 患者 $\times$ 7 時点 | 長期時系列の検証 |
| Szafrański 2025 | mSystems 16S | 127 横断サンプル, 5 属 | アトラクター / 群集型 |
| Heine 2025 | in vitro ODE | — | 4 つの ODE アトラクター |

\vspace{0.3em}
Dieckow が一次フィット、Duran-Pinedo が独立コホート、Szafrański が横断的群集型、
Heine が in vitro の力学的基準を与える。

---

## 4 つの ODE アトラクター

Heine in vitro 系の 4 状態 = commensal/dysbiotic $\times$ static/HOBIC：

| コード | 群集 | 環境 |
|---|---|---|
| **CS** | commensal | static |
| **CH** | commensal | HOBIC |
| **DS** | dysbiotic | static |
| **DH** | dysbiotic | HOBIC |

\vspace{0.2em}
**HOBIC** = チタンインプラント流路チャンバー（flow chamber）。
これらのコードは `results/` とスクリプト全体で繰り返し現れる。

---

## ① 出発点：5 菌種 Bayesian ODE フィット（Heine in-vitro）

Heine 2025 の in-vitro 5 菌種（So/An/Vd-Vp/Fn/Pg）時系列に対し ODE フィットを実施。
**gLV**（非対称 $A$、RMSE 0.012–0.032）と **Hamilton NUTS**（対称 $A$、RMSE 0.033–0.119）
の 2 モデルを比較；4 アトラクター（CS/CH/DS/DH）を再現。

![](results/heine2025/glv_heine_fit_thesis.pdf){ height=40% }

- 図：gLV MAP 軌道 + 実験 IQR 箱ひげ図（RMSE は列タイトル）。
- Hamilton NUTS は GPU-TMCMC $N_p=10{,}000$；**独立検証**：pH $R^2=0.78$、Gingipain–Pg $r=0.90$。
- 本研究では①を AGORA prior・in-vivo Dieckow・空間 PDE/FISH（②）へ拡張する。

---

## 柱 1 — 相互作用推定と誠実な検証

![](dieckow_paper/figures/fig2_loo_comparison.png){ height=44% }

- ベストモデルの **LOO-RMSE $= 0.0504$**。
- cross-feeding 方向は独立に検証：**$p=4\times10^{-4}$**。
- 2 コホート（Dieckow $\times$ Duran-Pinedo）が prior 抜きで強ペア符号
  **89% 一致**（$p\approx0.02$）。

\textcolor{red}{prior の価値は予測精度でなく解釈可能性 — 誠実に位置づける。}

---

## 柱 1 — アトラクター構造の検証

![](dieckow_paper/figures/fig3_joshi_attractor.png){ height=58% }

推定した $A$ から再構成した固定点が、観測された commensal / dysbiotic
群集型に対応する（Joshi アトラクター解析）。動態は単なる回帰ではなく
力学系の構造を持つ。

---

## 柱 2 — COMETS / dFBA

![](comets/pipeline_results/sweep_crossfeeding.png){ height=50% }

同じ AGORA GEM が 5 種（So/An/Vp/Fn/Pg）の**動的 FBA**を前向きに駆動。
フォールバック連鎖：Java COMETS $\to$ **AGORA 較正 Monod dFBA（主経路）**
$\to$ mock logistic。cross-feeding の**符号**を構造的に検証（大きさは不問）。

---

## 柱 3 — 16S / メタデータ前処理

生シーケンス + 論文補遺から、モデルが消費する**ギルド $\varphi$ 配列**を作る：

- **vsearch + SILVA**（merge $\to$ QC $\to$ chimera $\to$ 分類）。
- **QIIME2 / DADA2**、メタゲノムは **MetaPhlAn**。
- サンプルメタデータは **Nextflow** パイプライン
  （`get_biosample.py` $\to$ `merge_meta.py` $\to$ `extract_supp.py`）。

\vspace{0.3em}
出力はすべて 10 ギルド分類にマップされ、$\varphi\;(N_{\text{pat}}\times T\times 10)$
として柱 1・柱 2 に渡る。

---

## 空間拡張（Heine HOBIC FISH）

![](results/diffusion_fit/zprofiles_all_ti.png){ height=46% }

CLSM-FISH から得た**深さ分解 5 種プロファイル**を反応拡散方程式で fit。
dysbiosis は単なる組成変化でなく**空間的再編成**：
*P. gingivalis* が深部へ沈む（嫌気ニッチ）。0D の gLV を
$z$ 方向の拡散項付き PDE に拡張する。

---

## サブデッキの接続

- **AGORA**（符号 prior）：$\sgn(F_{ij})\to P_{ij}$ を構築する代謝側の橋。
- **Dieckow（デッキ A）**：その prior 下で $A$ を推定し LOO で検証する本体。
- **Network（デッキ B）**：推定 $A$ の構造 — 中心性・concordant backbone・
  CS$\leftrightarrow$DH リワイヤリングを解析。
- **Spatial PDE（デッキ C）**：時間動態を深さ方向の反応拡散に拡張。
- **FISH パイプライン**：CLSM `.lif` $\to$ 深さプロファイル（PDE の観測量）。

---

## 現状

| 項目 | 状態 |
|---|---|
| 相互作用モデル + 検証 | **完了**（LOO-RMSE 0.0504, $p=4\times10^{-4}$） |
| 空間拡散フィット | HPC で**実行中** |
| GDI / Joshi 臨床検証 | メタデータ**待ち** |

\vspace{0.3em}
2 コホート再現と独立な機械論経路（COMETS）により、相互作用の主張は堅固になりつつある。
残る課題は空間 PDE の収束と臨床メタデータの統合である。

---

## Take-home

1. **代謝が生態的相互作用の符号を制約する**
   （$P_{ij}=\sgn(F_{ij})$；大きさは使わない）。
2. dysbiosis は**リワイヤリング + 空間的再編成**であり、
   *P. gingivalis* の深部への沈降を伴う。
3. その主張は **2 コホート**（Dieckow $\times$ Duran-Pinedo, 89%, $p\approx0.02$）と
   **独立な機械論経路**（COMETS dFBA）で再現される。
4. すべてを 1 つの **10 ギルド分類**と **`fit_*.json`** が束ねる。
