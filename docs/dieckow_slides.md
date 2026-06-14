---
title: "Dieckow 縦断16Sからの生態相互作用推定と検証"
subtitle: "組成時系列から符号付き相互作用行列 $A$ へ — フィット・LOO-CV・誠実な検証"
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

## 本デッキの位置づけ

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

- $P_{ij}=\sgn(F_{ij})$ は AGORA cross-feeding フラックス由来（別デッキで導出。AGORA 出力からの具体例は次スライド）。
- $\relu{-P_{ij}A_{ij}}$ は $\sgn(A_{ij})=P_{ij}$ のとき $0$ — **大きさは不問**。
- posterior は **TMCMC**（transitional MCMC）でサンプリング：$10^{4}$ 個の**粒子**（パラメータの引き）を、prior $\to$ posterior へと温度づけされた段階を通して進め、各段階で重み付け・再サンプリングする。最終的な粒子群が $(A,b)$ の posterior **そのもの**であり、$10^{4}$ がその分解能を決める。

---

## AGORA 出力から符号 prior $P_{ij}$ へ

**AGORA が出力するもの。** 各ギルドについて、AGORA2 の pFBA 解が
**分泌／取り込みフラックスのプロファイル**（mmol\,gDW$^{-1}$h$^{-1}$）を与える。例（実際の pFBA 出力）：

\begin{center}
\begin{tabular}{@{}ll@{}}
\toprule
ギルド & pFBA 分泌（mmol\,gDW$^{-1}$h$^{-1}$） \\
\midrule
Actinobacteria & acetate 49.8, formate 53.7, succinate 39.2 を分泌 \\
Bacilli & acetate 921, lactate 982, propionate 1000, succinate 18 を分泌 \\
\bottomrule
\end{tabular}
\end{center}

**正味の有向フロー。** ドナー $i$ が分泌した代謝物をアクセプター $j$ が消費する
$\Rightarrow$ 有向 cross-feeding フラックス $F_{ij}>0$；共有資源をめぐる競争
$\Rightarrow F_{ij}<0$。

**導出。** $P_{ij}=\sgn(F_{ij})$ — penalty に入るのは*符号*のみ
（FBA フラックスの単位 $\neq$ 生態的単位なので、大きさは設計上捨てる）。

**具体例。** Actinobacteria は acetate/formate/succinate を分泌し、Bacilli が
これらを消費する $\Rightarrow F>0 \Rightarrow P=+1$（促進）。これは推定された
$\hat A=+1.62$ と一致。主要な文献ペアは 2/3 で確認；阻害的リンク
（H$_2$O$_2$、競争）は群集レベルのモデリングを要する。

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

![](dieckow_paper/figures/fig2_loo_comparison.png){ height=40% }

![](dieckow_paper/figures/fig8_all_models_rmse_bc.png){ height=27% }

\small **左：** AGORA 層を加える（L1+L2+AGORA）と L1 のみより LOO 誤差が下がる。**右：** 全モデル変種を LOO-RMSE / LOO-BC で順位づけ — *低いほど良い*。要点：代謝 prior は効くが、量の prior は効かない。

---

## 主要な性能指標

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

**読み方。** 各セル = 1 つのギルドペア；その値 = 10 回の leave-one-patient-out
再推定のうち*同じ相互作用符号*を保った割合（1.0 = 10 回すべてで同符号）。
全ペアが $\geq 0.70$ $\Rightarrow$ 推定された符号構造はどの 1 患者にも依存しない。

---

## 批判的検証 — 符号構造は本物か？

\small

**問い。** ランダムなギルドラベルでも、フィット済みモデルと同じくらい高い
符号一致を出せるのか？（prior を *on* にすると全正になり、素朴な一致数は水増しされる。）

**検定。** **ラベル permutation null**：ギルドラベルを $10^{4}$ 回シャッフルし、
そのたびに一致を再計算し、実際のスコアをその null と比較する。

**結果。**

| モデル | cross-feeding 方向 | 競争方向 |
|---|---|---|
| **Hamilton $\alpha=0$** | $\mathbf{78.6\%}$ (11/14) vs null $37.7\%$, $p=4\times10^{-4}$, $z=+3.79$ | $\approx$ chance |
| gLV | $41\%$（null） | null |

![](results/dieckow_cr/loo_alpha_comparison_hamilton_noagora_glv_noagora.png){ height=18% }

**読み。** **cross-feeding 方向**だけが null を上回る。**競争方向は検証されない。**

---

## 2 コホート再現（設計をまたぐ確認）

Duran-Pinedo et al. 2021（**BMC Biol 19:240**；ENA **PRJNA725874**、15 患者
$\times$ 7 時点）を **prior 抜き**で独立にフィットし、そのうえで強ペアの符号を
Dieckow と比較する。

![](results/duranpinedo_validation/fig_duranpinedo_A_comparison_noprior.png){ height=40% }

強ペアの有向符号が **$89\%$**（$8/9$ 上三角）で一致（$p\approx0.02$）。
\textcolor{red}{\textbf{留意 — 厳密な同型再現ではない。}} 両コホートは
**時間スケールと臨床状態**が異なる：Dieckow = ペリインプラント生着初期（週単位；
健常 $\to$ 生着）、Duran-Pinedo = 長期的な**歯周炎**進行（疾患）。この 89\% は
設計をまたいだ粗い*強ペア符号*の一致であり、同型の再現ではない。
Actinobacteria 軸の対応は**示唆的**であり、確証には設計を揃えたコホートを要する。

---

## 誠実な解釈

- **AGORA prior 自体は 16S 力学で再現されない** $\Rightarrow$ それは
  **モデリング選択**であり、データ確認済みの事実ではない。
- データから独立に裏づくのは：
  1. **cross-feeding 方向**（Hamilton, $p=4\times10^{-4}$）、
  2. **2 コホート強ペア**符号（Dieckow $\times$ Duran-Pinedo, $89\%$；設計をまたぐ — 留意参照）。
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
