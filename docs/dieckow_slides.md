---
title: "Dieckow 縦断16Sからの生態相互作用推定と検証"
subtitle: "組成時系列から符号付き相互作用行列 $A$ へ — フィット・LOO-CV・誠実な検証"
author: "西岡佳祐 — NIFE / SFB TRR-298"
date: "2026-06-03"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## このデッキの位置づけ

本プロジェクト = 3 本柱 ＋ 空間拡張。データの流れ：

raw 16S → guild $\varphi$ → gLV/Hamilton（＋符号 prior）→ LOO 検証 → 空間 PDE

デッキ構成（◀ = 本デッキ）：

- **Overview**（傘）— 全体像と3本柱
- **AGORA** — 代謝 → 符号 prior（生態モデルの入力）
- **Dieckow** — in-vivo 相互作用の推定と検証（生態モデル本体）  ◀ **本デッキ**
- **Network** — 相互作用行列 $A$ の構造解析
- **Spatial-PDE** — FISH 深さプロファイルの反応拡散
- **FISH pipeline** — .lif → 5 種深さ組成

---

## 狙い

口腔バイオフィルムの**組成的 16S 時系列**から、**符号付き相互作用行列** $A$ を
推定し、その推定をどこまで信用できるかを検証する。

- モデル：一般化 Lotka–Volterra (gLV) / Hamilton replicator（単体上）
- 制約：AGORA2 由来の **代謝符号 prior**（別デッキ参照、ここでは簡潔に引用）
- 焦点：**フィット**・**LOO-CV**・モデル比較・**誠実な検証**

\vspace{0.4em}
**主張：** prior は予測精度を買うのではなく、符号整合性・解釈可能性を与える。
データから独立に裏づくのは **cross-feeding 方向のみ**（後述）。

---

## データと設計

Dieckow et al. 2024（ENA **PRJEB71108**）：**10 患者 $\times$ 3 週**の縦断
16S。class レベルの **10 ギルド**に集約（`GUILD_ORDER` を正典とする）。

![](dieckow_paper/figures/fig0_study_design.png){ height=58% }

ペリインプラント周囲の生着初期動態。各サンプルは組成 $\varphi\in\Delta^{9}$
（$\sum_i\varphi_i=1$）として扱う。

---

## モデル — gLV と replicator

絶対量 $x_i$ の一般化 Lotka–Volterra：
$$\dot{x}_i = x_i\Big(b_i + \sum_{j=1}^{S} A_{ij}\,x_j\Big),\qquad S=10 .$$

単体上の組成 $\varphi_i=x_i/\sum_k x_k$ に対する replicator / Hamilton 形：
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big],
\qquad \sum_i\varphi_i=1 .$$

- **Hamilton** は $A$ を**対称**に取る（線形ペイオフの replicator；Taylor & Jonker 1978）。
- **古典 gLV** は $A$ が**非対称**。$A_{ij}>0$：促進、$A_{ij}<0$：抑制。

---

## 推定は劣決定 $\Rightarrow$ 代謝符号 prior

パラメータ $\mathcal{O}(S^2)\approx100$ に対し観測は $10\times3$。存在量だけでは
$A$ の符号は識別不能 $\Rightarrow$ 片側ヒンジで**符号違反のみ**を罰する：
$$\mathcal{L}(A,b)=\frac{1}{2\sigma^{2}}\sum_{t}\big\lVert\varphi^{\text{obs}}_{t}-\varphi^{\text{pred}}_{t}\big\rVert^{2}
\;+\; W\!\!\sum_{(i,j)\in\mathcal{M}}\!\!\relu{-\,P_{ij}\,A_{ij}} .$$

- $P_{ij}=\sgn(F_{ij})$ は AGORA cross-feeding フラックス由来（別デッキで導出）。
- $\relu{-P_{ij}A_{ij}}$ は $\sgn(A_{ij})=P_{ij}$ のとき $0$ — **大きさは不問**。
- posterior は **TMCMC**（$10^{4}$ 粒子）でサンプリング。

---

## フィットした相互作用行列 $A$

![](dieckow_paper/figures/fig1_A_matrix.png){ height=60% }

対角は**自己抑制** $A_{ii}<0$（資源制限／飽和）。
オフ対角に Actinobacteria $\leftrightarrow$ Bacilli の**促進**が現れる。

---

## 軌道フィット（全 10 患者）

![](dieckow_paper/figures/fig_fitting_results.png){ height=66% }

観測（点）と予測軌道（線）。患者ごとに $b$ を、行列 $A$ は共有して推定。

---

## $A_{ij}$ と代謝 $F_{ij}$ の符号対応

![](dieckow_paper/figures/fig5b_aij_fij_scatter.png){ height=58% }

横軸 = 代謝フロー $F_{ij}$、縦軸 = 推定 $\hat A_{ij}$。
W=1.0 では制約セル $\mathcal{M}$ 上で $\sgn(\hat A_{ij})=\sgn(F_{ij})$ が一致。

---

## LOO-CV の定義

患者 $p$ を抜いて行列を再推定し、抜いた患者の軌道で評価：
$$\mathrm{RMSE}=\sqrt{\frac{1}{NT}\sum_{t}\big\lVert\varphi^{\text{obs}}_{t}-\varphi^{\text{pred}}_{t}\big\rVert^{2}},$$
$$\mathrm{BC}=\frac{\sum_i\lvert\varphi^{\text{obs}}_i-\varphi^{\text{pred}}_i\rvert}{\sum_i\big(\varphi^{\text{obs}}_i+\varphi^{\text{pred}}_i\big)}
\quad(\text{Bray–Curtis 非類似度}).$$

10 患者すべてを順に hold-out し、平均する（leave-one-patient-out）。

---

## LOO モデル比較

![](dieckow_paper/figures/fig2_loo_comparison.png){ height=44% }

![](dieckow_paper/figures/fig8_all_models_rmse_bc.png){ height=30% }

prior 層・モデル形・$W$ を変えて LOO-RMSE / LOO-BC を比較。

---

## 数値（要点）

ベストモデル **L1+L2+AGORA, $W=1.0$**：

| 指標 | 値 |
|---|---|
| train RMSE | $0.0565$ |
| Pearson $r$ | $0.951$ |
| sign agreement | $70/70$（$100\%$） |
| **LOO-RMSE** | $\mathbf{0.0504}$ |
| LOO-BC | $0.1468$ |

- L1+L2 のみ：LOO-RMSE $0.0516$。
- **prior-free gLV**：LOO-RMSE $0.0455$（より低い — パラメータが多く代謝非制約）。
- **MacArthur 量 prior は失敗**：SA $4\text{–}8/70$。

\vspace{0.2em}
$\Rightarrow$ prior の価値は予測精度ではなく**符号整合性・解釈可能性**。

---

## LOO での $A$ 行列の安定性

![](dieckow_paper/figures/fig7_loo_stability.png){ height=60% }

10 個の hold-out で再推定した行列間で、全ペアの**符号一致率 $\geq 0.70$**。
推定された符号構造は患者の入れ替えに対して頑健。

---

## 批判的検証（誠実に）

prior を **OFF**（$\alpha=0$）にすると prior は**全正**になり、naive な
sign-agreement は過大評価される。**ラベル permutation 検定**（$n=10^{4}$）で対照：

| モデル | cross-feeding 方向 | 競争方向 |
|---|---|---|
| **Hamilton $\alpha=0$** | $\mathbf{78.6\%}$ (11/14) vs random $37.7\%$, $p=4\times10^{-4}$, $z=+3.79$ | $\approx$ chance |
| gLV | $41\%$（null） | null |

![](results/dieckow_cr/loo_alpha_comparison_hamilton_noagora_glv_noagora.png){ height=30% }

**検証されるのは cross-feeding 方向のみ。競争方向は支持されない。**

---

## 2 コホート再現

Botelho 2021（**PRJNA725874**、15 患者 $\times$ 7 時点）を **prior 抜き**で
独立にフィットし、Dieckow と強ペアの符号を比較：

![](results/botelho_validation/fig_botelho_A_comparison_noprior.png){ height=46% }

強ペアの有向符号が **$89\%$**（$8/9$ 上三角）で一致（$p\approx0.02$）。
特に **Actinobacteria 軸**が両コホートで整合。

---

## 誠実な解釈

- **AGORA prior 自体は 16S 力学で再現されない** $\Rightarrow$ それは
  **モデリング選択**であり、データ確認済みの事実ではない。
- データから独立に裏づくのは：
  1. **cross-feeding 方向**（Hamilton, $p=4\times10^{-4}$）、
  2. **2 コホート強ペア**符号（Dieckow $\times$ Botelho, $89\%$）。
- 競争方向・prior の大きさ・全体行列は検証されていない。

\vspace{0.3em}
過大に語らない：符号の**一部**が支持され、残りは仮説に留まる。

---

## 外部臨床検証（進行中）

Guild Dysbiosis Index を Joshi 2025 コホート（**PRJNA1192962**）と対照する計画：

![](dieckow_paper/figures/fig3_joshi_attractor.png){ height=46% }

\textcolor{red}{\textbf{予備的}：臨床メタデータ待ち。本図は暫定であり、健常／疾患
ラベルとの突き合わせは未完了。}

---

## 結論

1. 組成 16S 時系列 $\to$ 符号付き $A$ を gLV / Hamilton replicator で推定。
2. 劣決定性は **AGORA 代謝符号 prior**（片側ヒンジ）で緩和；posterior は TMCMC。
3. ベスト：L1+L2+AGORA $W=1.0$ で **LOO-RMSE $0.0504$**、SA $100\%$。
   ただし prior-free gLV は $0.0455$ — prior は精度でなく**解釈可能性**を買う。
4. **LOO で符号は安定**（$\geq0.70$）；MacArthur 量 prior は失敗。
5. 誠実な検証：**cross-feeding 方向**（$p=4\times10^{-4}$）と
   **2 コホート強ペア**（$89\%$）のみが独立に支持。臨床検証は進行中。
