---
title: "相互作用行列のネットワーク解析"
subtitle: "推定した相互作用行列をグラフとして読む — keystone・bridge・栄養層・リワイヤリング"
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
- **Dieckow** — in-vivo 相互作用の推定と検証（生態モデル本体）
- **Network** — 相互作用行列 $A$ の構造解析  ◀ **本デッキ**
- **Spatial-PDE** — FISH 深さプロファイルの反応拡散
- **FISH pipeline** — .lif → 5 種深さ組成

---

## 動機：相互作用行列のグラフ構造解析

相互作用行列 $A$ の推定にとどまらず、その**グラフ構造**を定量的に解析する。

- **keystone**：低存在量でも群集を支配する菌種（高い中心性）。
- **bridge**：栄養層を結ぶ媒介種（高い betweenness）。
- **trophic layer**：cross-feeding の生産者 $\to$ 消費者の階層性。
- **rewiring**：健常 $\leftrightarrow$ dysbiosis 間でエッジ符号が反転する再配線。

\vspace{0.4em}
口腔バイオフィルムの古典的描像は、*P. gingivalis* を keystone、*F. nucleatum* を bridge とする。
本デッキは、in-vivo class レベルでこの描像が**支持されるか**をネットワーク統計で検証する。

---

## ゲージ不変性：$A$ ではなく $A_{\text{eff}}$ を読む

replicator 力学
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big],\qquad \sum_i\varphi_i=1$$
は $A$ の**列ごとの定数シフト** $A_{ij}\to A_{ij}+c_j$ に対して不変（$\sum_i\varphi_i=1$ で消える）。

$\Rightarrow$ 生の $A_{ij}$ の符号・大きさは**ゲージ依存**で意味を持たない。**列中心化**した
$$A_{\text{eff}}[i,j] = A_{ij} - \frac{1}{S}\sum_{k} A_{kj}$$
のみがゲージ不変。**以降の全ての符号・中心性解釈は $A_{\text{eff}}$ に対して行う。**

有向重み付きグラフ $G=(V,E)$ を $w_{ij}=A_{\text{eff}}[i,j]$ で定義する（$|V|=S=10$ ギルド）。

---

## 中心性の定義

$A_{\text{eff}}$ から複数の中心性を計算：

- **固有ベクトル中心性**：$A\,v = \lambda\,v$ の主固有ベクトル $v$。
  $v_i$ は「中心的な相手とつながる種ほど中心的」を再帰的に表す。
- **媒介中心性 (betweenness)**：最短経路が頂点 $i$ を通る割合
  $$g(i)=\sum_{s\neq i\neq t}\frac{\sigma_{st}(i)}{\sigma_{st}}.$$
- **PageRank**：定常分布 $\pi = (1-d)\,\mathbf{1}/S + d\,M\pi$（$M$ は列正規化推移行列）。
- **入次/出次強度**：$\;s^{\text{in}}_i=\sum_j |A_{\text{eff}}[i,j]|,\quad s^{\text{out}}_j=\sum_i |A_{\text{eff}}[i,j]|.$

---

## 中心性の結果（Dieckow 10 ギルド・no-prior consensus）

![](results/guild_network/guild_centrality_summary.png){ height=50% }

固有ベクトル中心性：Bacilli **0.61**、Actinobacteria **0.58**、Betaproteobacteria 0.44、
Negativicutes 0.22、Bacteroidia($\approx$ *P. gingivalis*) 0.19。
媒介：Bacilli **0.78**、Actinobacteria 0.22、他は 0。
平均存在量：Bacilli **0.547**（優占）、Actinobacteria 0.203、Bacteroidia 0.047。

---

## keystone / bridge 検定：古典像は成り立つか

中心性と存在量のランクを照合：

- *P. gingivalis*（Bacteroidia）は固有ベクトル中心性ランク $\sim$**4**、存在量ランク $\sim$**5**。
  $\Rightarrow$ **構造的 keystone ではない**（少量かつ低中心性）。
- bridge（最高 betweenness）は **Bacilli（*Streptococcus*）**であり、**Fusobacterium ではない**。

\vspace{0.3em}
$\Rightarrow$ 古典的「Pg-keystone / Fn-bridge」像は **in-vivo class レベルでは支持されない**。

\vspace{0.3em}
\textcolor{red}{注意（過大解釈を避ける）：class $\neq$ 種；密なグラフは betweenness を縮退させる；
希少ギルドの推定はノイズが大きい。}

---

## 影響力 vs 脆弱性（出次強度 vs 入次強度）

![](results/guild_network/influence_vulnerability.png){ height=55% }

出次強度 $s^{\text{out}}_j=\sum_i|A_{\text{eff}}[i,j]|$ は $j$ が他者に及ぼす**影響力**、
入次強度 $s^{\text{in}}_i=\sum_j|A_{\text{eff}}[i,j]|$ は $i$ の**脆弱性**を測る。
両者の非対称性が「駆動側／被駆動側」を分ける。

---

## 栄養的コヒーレンス（有向 AGORA cross-feeding）

有向 cross-feeding グラフに対し、MacKay の**栄養的非コヒーレンス** $F_0$ を測る
（栄養レベル $h_i$ を割り当て、エッジ両端のレベル差の分散）：
$$F_0 = \frac{1}{w}\sum_{ij} W_{ij}\,(h_i - h_j - 1)^2 .$$

- 実測 $F_0 = \mathbf{0.652}$ vs ランダム $0.646$、$p=0.50$。
- $\Rightarrow$ **栄養的にインコヒーレント** ＝ 階層を持たず、**相互 cross-feeding の循環**。

\vspace{0.3em}
生物学的に妥当：Fusobacterium は基底生産者、Veillonella は高位消費者だが、
ループ状に絡み合い、明確な一方向の食物連鎖を成さない。

---

## Concordant backbone：生態層 $\times$ 代謝層の合意

独立な 2 層を重ねる — 生態(Hamilton)層と AGORA 代謝層 — で両者が**符号一致**するエッジを抽出：

- **11 エッジ、すべて正（協調）**、$p=4\times10^{-4}$（AGORA デッキと同一の独立検証）。
- Negativicutes（*Veillonella*）が主要な**代謝シンク**：
$$\{\text{Bacilli, Bacteroidia, }\beta\text{Proteo, Fusobacteria}\}\ \longrightarrow\ \text{Negativicutes}.$$
- これは再構成された**乳酸 cross-feeding** に対応。

\vspace{0.3em}
**本ネットワーク解析で最も頑健な結果。**符号レベルの backbone は両層から独立に支持される。

---

## ネットワークの LOO 安定性

![](results/guild_network/loo_stability.png){ height=55% }

患者を 1 人ずつ抜く leave-one-out で $A_{\text{eff}}$ を再推定し、各ペアの符号一致率を測る。
**全ペアで符号一致率 $\geq 0.70$** $\Rightarrow$ ネットワークの符号構造は個々の患者に頑健。

---

## エッジ強度の permutation 検定（誠実に）

![](results/guild_network/permutation_test.png){ height=48% }

各 off-diagonal エッジの**大きさ**を permutation null と比較：
**90 ペア中わずか 2 ペア**が個別に $p<0.05$ を超えるのみ。

\vspace{0.3em}
$\Rightarrow$ ネットワークは密で、個々のエッジ**強度は弱くしか分解されない**。
頑健なのは前スライドの**符号レベル backbone**であって、個別 magnitude ではない。

---

## CS $\leftrightarrow$ DH リワイヤリング（Heine 5 種・posterior）

**① Heine 5 菌種 GPU-Bayesian TMCMC** の 4 アトラクター posterior（CS = commensal-static、
DH = dysbiotic-HOBIC, `ultimate_10000p`）を、列中心化 $A_{\text{eff}}$ で比較する。**促進確率**を
$$P_{\text{facilitation}}(i\!\leftarrow\! j) = \Pr\big[\,A_{\text{eff}}[i,j] > 0\,\big]$$
（posterior 上で $A_{\text{eff}}[i,j]>0$ となる確率）と定義。

$P_{\text{fac}}\to 1$ は相互作用が**促進**、$\to 0$ は**競争／抑制**を意味する。
CS と DH の間で $P_{\text{fac}}$ がどう動くかが「再配線」を定量化する。

---

## リワイヤリングの結果

CS $\to$ DH で促進確率 $P_{\text{fac}}$ が劇的に変化：

- **So $\leftrightarrow$ Vd 相互扶助の崩壊**：*Streptococcus*–*Veillonella* 共生が dysbiosis で競争へ。
  $$P_{\text{fac}}(\text{So}\!\to\!\text{Vd}):\ 1.00 \to 0.03,\qquad
    P_{\text{fac}}(\text{Vd}\!\to\!\text{So}):\ 0.9995 \to 0.00.$$
- **Fusobacterium が支えられる**：An$\to$Fn と Vd$\to$Fn が競争 $\to$ 促進へ反転。
  $$P_{\text{fac}}(\text{An}\!\to\!\text{Fn}):\ 0.23 \to 0.97,\qquad
    P_{\text{fac}}(\text{Vd}\!\to\!\text{Fn}):\ 0.09 \to 0.77.$$

---

## *P. gingivalis* の中心化

固有ベクトル中心性：CS **0.32** $\to$ DH **0.51**（DH で最高）。

- ただし DH で Pg は**正味抑制的**：出力エッジの $P_{\text{facilitation}} = 0.27$。
- $\Rightarrow$ Pg は**静的 keystone ではない**が、動的な dysbiotic 状態で
  **keystone 様**に「なる」。

\vspace{0.3em}
この結果は FISH の知見と整合する：dysbiosis において Pg は深部へ移行し、
周囲の菌種への依存を低下させながら中心化する。keystone 性は固定した属性ではなく、
群集の状態に依存する性質である。

---

## 結論

1. **代謝 backbone は頑健**：Veillonella を主シンクとする乳酸 cross-feeding が、
   生態層・代謝層から独立に符号支持される（11 エッジ・全正・$p=4\times10^{-4}$）。
2. **教科書的 keystone は支持されない**：in-vivo class レベルで Pg は構造的 keystone でなく、
   bridge は Fusobacterium でなく Bacilli。
3. **dysbiosis は abundance シフトでなく「リワイヤリング」**：
   So–Vd 相互扶助の喪失 ＋ Pg の中心化。
4. 個々のエッジ強度は弱く分解される（permutation で 2/90）— 頑健なのは符号構造。
   class $\neq$ 種・密グラフの縮退・希少ギルドのノイズは引き続き限界。
