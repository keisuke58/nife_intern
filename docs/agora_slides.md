---
title: "AGORA を軸とした代謝—生態の統合"
subtitle: "ゲノムスケール代謝モデルから相互作用符号へ — 形式的取り扱い"
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
- **AGORA** — 代謝 → 符号 prior（生態モデルの入力）  ◀ **本デッキ**
- **Dieckow** — in-vivo 相互作用の推定と検証（生態モデル本体）
- **Network** — 相互作用行列 $A$ の構造解析
- **Spatial-PDE** — FISH 深さプロファイルの反応拡散
- **FISH pipeline** — .lif → 5 種深さ組成

---

## 狙いと主張

**生態学的相互作用行列**を、**ゲノムから計算した代謝（AGORA2）**で制約し、
その制約がどこまで経験的に支持されるかを定量する。

- 生態モデル：一般化 Lotka–Volterra (gLV) / replicator
- 代謝エンジン：flux-balance analysis (FBA, pFBA, MICOM)
- 橋渡し：cross-feeding フラックスから導く **符号 prior** $P_{ij}=\sgn(F_{ij})$
- 検証：prior 抜きフィットの permutation test、2 コホート再現

\vspace{0.4em}
**主張：** 代謝シグナルが制約するのは協調的相互作用の**符号**（大きさでも競争でもない）。
$p=4\times10^{-4}$ で支持。

---

## 生態モデル

絶対量 $x_i$ の一般化 Lotka–Volterra：
$$\dot{x}_i = x_i\Big(b_i + \sum_{j=1}^{S} A_{ij}\,x_j\Big),\qquad i=1,\dots,S=10 .$$

単体上の組成 $\varphi_i = x_i/\sum_k x_k$（16S は組成データ）に対する
replicator / Hamilton 形：
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big],
\qquad \sum_i \varphi_i = 1 .$$

- $A_{ij}>0$：$j$ が $i$ を**促進**、$A_{ij}<0$：**抑制**。
- 推定は**劣決定**：$\mathcal{O}(S^2)=100$ パラメータ vs $10$ 患者 $\times\,3$ 週。
  存在量だけでは $A$ の符号は識別不能 $\Rightarrow$ **代謝符号 prior** を加える。

---

## FBA（Flux Balance Analysis）= 線形計画

ゲノムスケールモデルは化学量論行列 $S\in\mathbb{R}^{m\times n}$
（代謝物 $\times$ 反応）を与える。定常成長は LP：
$$\max_{v}\; c^{\!\top} v \quad\text{s.t.}\quad S v = 0,\;\; v^{-}\le v \le v^{+},$$
$c$ がバイオマス反応を選ぶ（$\mu = c^{\!\top}v$）。**pFBA** はフラックスループを
第2段で除去：
$$\min_{v}\; \sum_{j}\lvert v_j\rvert \quad\text{s.t.}\quad c^{\!\top}v=\mu^{\star},\; Sv=0,\; v^{-}\le v\le v^{+}.$$

交換フラックスから、ギルドごとの **分泌** $s_{j\alpha}=\relu{v^{\text{ex}}_{j\alpha}}$、
**取り込み** $u_{i\alpha}=\relu{-v^{\text{ex}}_{i\alpha}}$（代謝物 $\alpha$）を得る。

---

## AGORA2 が与えるもの

**AGORA2**（Heinken et al., *Nat. Biotechnol.* 2023, 41:1320）：**7,302** 株の
ゲノムスケール再構成。腸内から**口腔**菌へ拡張。ギルドごとに代表 SBML を 1 つ。

- 代表株：Bacilli = *S. gordonii*、Negativicutes = *V. parvula*、
  Bacteroidia = *P. melaninogenica*、Fusobacteriia = *F. nucleatum* …
- 口腔液培地（Dawes 2008）：糖・20アミノ酸・B群ビタミン・微量元素・細胞壁前駆体
  $\Rightarrow$ 全 10 ギルドで正の増殖（$\mu = 0.11\text{–}1.66\,\mathrm{h^{-1}}$）。

\vspace{0.3em}
培地設計は本質的な感度：貧弱だと $\mu\to0$ で cross-feeding 信号が消える。

---

## cross-feeding スコア $\to$ 符号 prior

毒素を $\mathcal{T}=\{\text{H}_2\text{O}_2,\text{H}_2\text{S}\}$ とし、$j\to i$ の
**正味代謝フロー**を定義：
$$F_{ij} \;=\; \underbrace{\sum_{\alpha\notin\mathcal{T}} w_\alpha\, s_{j\alpha}\,u_{i\alpha}}_{\text{cross-feeding }(+)}
\;-\; \underbrace{\sum_{\alpha\in\mathcal{T}} w_\alpha\, s_{j\alpha}\,u_{i\alpha}}_{\text{toxin }(-)} .$$

**符号 prior** は方向のみを残す：
$$P_{ij} = \sgn(F_{ij}) \in \{-1,\,0,\,+1\}.$$

大きさは設計上捨てる — FBA フラックス単位 $\neq$ 生態的単位。
$10\times 9=90$ 方向ペアから制約集合 $\mathcal{M}=\{(i,j): P_{ij}\neq 0\}$ を得る。

---

## パイプライン全体像

![](results/fig2_agora_pipeline.png){ height=64% }

(A) 手順；(B) 層追加で $|\mathcal{M}|$ が $10\to22\to58$（AGORA L3 で +36）；
(C) 出来上がる符号 prior 行列 $P=\sgn(F)$。

---

## ベイズ統合：符号 prior の罰則

フィットは軌道誤差に、**符号違反のみ**（大きさは不問）を罰する片側ヒンジを加える：
$$\mathcal{L}(A,b) = \frac{1}{2\sigma^{2}}\sum_{t}\big\lVert \varphi^{\text{obs}}_{t}-\varphi^{\text{pred}}_{t}(A,b)\big\rVert^{2}
\;+\; W\!\!\sum_{(i,j)\in\mathcal{M}}\!\! \relu{-\,P_{ij}\,A_{ij}} .$$

- $\relu{-P_{ij}A_{ij}}=\max(0,-P_{ij}A_{ij})$ は $\sgn(A_{ij})=P_{ij}$ のとき $0$。
- $W$ = prior の硬さ、$\sigma$ = 観測スケール。
- 5 種アトラクターの posterior は **TMCMC**（$10^4$ 粒子）、ギルドは L-BFGS-B / TMCMC。

---

## 3 つの証拠層

| 層 | ソース | $w_\alpha$ | 根拠 |
|---|---|---|---|
| **L1** | Szafrański Suppl.（実験 + KEGG/HMDB） | 2.0 | 直接観測 |
| **L1** | Szafrański Suppl.（実験・注釈なし） | 1.5 | 直接観測 |
| **L2** | Szafrański Suppl.（予測） | 1.0 | 予測 |
| **L3** | **AGORA2 pFBA cross-feeding** | 1.0 | ゲノムスケール |

$w_\alpha=$ 代謝物 $\alpha$ の行ごと重みの max。代表例 — **乳酸**：
$$\text{Bacilli (Strep)} \xrightarrow{\ \text{lactate}\ } \text{Negativicutes (Veillonella), Actinobacteria}.$$

---

## なぜ量 prior は失敗するか — MacArthur の視点

消費者–資源理論（MacArthur 1970；Marsland 2019）は次を導く：
$$A_{ij} = \underbrace{\sum_{\alpha} s_{j\alpha}\,c_{i\alpha}}_{\text{cross-feeding}\,(+)}
\;-\; \underbrace{\frac{c_i\!\cdot\! c_j}{\lVert c_i\rVert\,\lVert c_j\rVert}}_{\text{niche overlap}\,(-)} .$$

- **ニッチ重複（cosine）項が飽和**：口腔菌は generalist で $\cos(c_i,c_j)\approx1$
  → ほぼ全ペアが競争と誤判定（実測 正 $6$ / 負 $84$ ペア＝使えない）。
- Growth-rate-suppression も貧栄養培地で同様に破綻。

$\Rightarrow$ $\sgn(A_{ij})$ を残し、大きさは捨てる。

---

## 経験的 sign agreement（naive 推定）

![](results/fig3_agora_sign_validation.png){ height=55% }

$$\mathrm{SA}=\frac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}}\mathbb{1}\!\left[\sgn(\hat A_{ij})=P_{ij}\right]=\frac{66}{72}=92\%.$$
\textcolor{red}{これは過大評価——prior は符号縮退している（次スライド）。}

---

## MICOM — 群集 FBA

単種 pFBA は**実行可能性**のみ（$j$ が $X$ を出せる・$i$ が $X$ を食べられる）。
**MICOM**（Diener 2020）は群集を同時に解き、実際にフラックスが**流れるか**を、
**cooperative trade-off** で判定する：
$$\max\; \min_{i}\frac{\mu_i}{\mu_i^{\max}}\quad\text{s.t.}\quad
\sum_i \mu_i \ge \tau\sum_i \mu_i^{\max},\;\; S^{\text{com}}v=0,\;\tau=0.5 .$$

- 各菌に最大成長の $\tau$ 倍以上を保証した上で、共有フラックス配分を確定。
- generalist でも**実際に群集フラックスを運ぶ経路のみ**が活性化 → 「全員競合」の
  非特異アーティファクトが消える。
- 群集解の交換フラックス $s_{j\alpha},u_{i\alpha}$ をそのまま $F_{ij}$ に渡す。

---

## MICOM — 結果

| 手法 | Sign Agreement | $|\mathcal{M}|$ |
|---|:---:|:---:|
| 文献 L1+L2 | 45% (5/11) | 11/45 |
| 単種 pFBA v1 | 88% (29/33) | 33/45 |
| **MICOM（群集）** | **100% (36/36)** | **36/45** |

群集内で直接解けた乳酸交換：
$$\text{Bacilli}\xrightarrow[\;-97.9\;]{\;+97.7\;\text{mmol gDW}^{-1}\text{h}^{-1}}\text{Negativicutes}\quad(\mathrm{EX\_lac\_L}).$$
注意：$\hat A$ は v1 prior 下で推定 → $100\%$ は包含による可能性。

---

## prior の硬さ $W$ — 相転移

![](results/fig_agora_weight_sensitivity.png){ height=55% }

$W=1.0$ で $\mathrm{SA}\to100\%$、$\mathrm{LOO\text{-}RMSE}\approx0.050$。
prior-free gLV は $0.0455$ とさらに低い → prior の価値は予測精度でなく
**解釈可能性・符号整合性**にある（誠実な位置づけ）。

---

## 批判的検証：prior はデータから独立に裏づくか

$\alpha=0$ で prior は**全正**（cross-feeding のみ）→ $\mathrm{SA}$ は過大。
正しい対照は **prior 外**セルとのラベル permutation 比較、統計量：
$$z=\frac{\mathrm{SA}-\mathbb{E}_\pi[\mathrm{SA}]}{\sqrt{\operatorname{Var}_\pi[\mathrm{SA}]}},\qquad n_\pi=10^4 .$$

| モデル | cross-feeding 方向 | 競争方向 |
|---|---|---|
| **Hamilton（対称）, $\alpha=0$** | $\mathbf{78.6\%}$ (11/14) vs $\mathbb{E}_\pi=37.7\%$, $\;p=4\!\times\!10^{-4},\,z=+3.79$ | $\approx$ chance |
| gLV（非対称） | 41%（null） | null |

- **検証されるのは協調(cross-feeding)方向のみ**。競争方向は支持されない。
- AGORA prior は 16S 力学で**再現されない** $\Rightarrow$ モデリング選択（データ確認済みではない）。
- **2 コホート**（Dieckow $\times$ Botelho）を prior 抜きで比べ、強ペア符号が
  $\mathbf{89\%}$ 一致（$p\approx0.02$）。

---

## 機械論的クロスチェック（COMETS 動的 FBA）

同じ AGORA GEM が 5 種 **dFBA** 前向きシミュレーション（Monod 連成交換）を駆動、
prior とは独立：

![](comets/pipeline_results/sweep_crossfeeding.png){ height=50% }

健常：So/An 優占・乳酸 cross-feeding・$\mathrm{DI}=0.15$。
疾患：Pg/Fn 増殖・$\mathrm{DI}=0.70$。commensal$\leftrightarrow$dysbiotic の分岐が
前向きにも出現し、推定した相互作用を裏づける。

---

## 限界と結論

**限界：** ギルド $=$ class（代表株 $\neq$ ギルド全体）；阻害行の $20/22$ は酸素
（生産者不在）$\Rightarrow$ 毒素は H$_2$O$_2$ のみ発火；$w_\alpha$ は代謝物ごと max；
大きさは捨てている。

**結論：**
1. $\text{AGORA pFBA}\to F_{ij}\to P_{ij}=\sgn(F_{ij})$ という変換が方法論的新規性。
2. **符号は使え、量は使えない**（MacArthur cosine 飽和を回避）。
3. 単種（$92\%$）$\to$ **MICOM（$100\%$）** が群集文脈を捕捉。
4. 誠実な検証：**協調方向のみ** $p=4\times10^{-4}$；2 コホート $89\%$；prior はモデリング選択。
5. COMETS dFBA が dysbiotic 分岐を前向きにも再現。
