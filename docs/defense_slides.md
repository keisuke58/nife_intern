---
title: "代謝が制約する口腔バイオフィルムの群集動態"
subtitle: "生態 ODE 推定・代謝符号 prior・空間拡張 — 修士論文公聴会"
author: "西岡佳祐 — NIFE / SFB TRR-298"
date: "2026-11"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## North-star（目標）

ペリインプラント炎に関わる**口腔バイオフィルム群集の dysbiosis**を、
群集組成の時間発展としてモデル化する（SIIRI / SFB TRR-298 コンソーシアム）。

- 臨床問題：チタンインプラント周囲で健常（commensal）から疾患（dysbiotic）へ
  遷移する微生物群集。
- モデリングの問い：**何がこの遷移を駆動するのか** — そして
  ゲノムから計算した代謝は、生態的相互作用について何を教えてくれるか。
- 中心的アイデア：**代謝が生態的相互作用の符号を制約する**。

\vspace{0.4em}
本公聴会は、生態 ODE 推定・代謝符号 prior・空間拡張を 1 本の物語として束ねる。

---

## 概念図（全体像）

![](results/figures/concept_overview_pub.png){ height=78% }

代謝が相互作用の**符号**を決め、dysbiosis は**リワイヤリング＋空間再編**
（Pg の中心化・深部沈降）として現れる。

---

## 研究の弧（narrative）

本研究は 2 段で構成される：

1. **オリジナル（修士前半）** — 5 種 in-vitro 系の **GPU ベイズ ODE 推定**。
   TMCMC（$10^4$ 粒子）で 4 アトラクター **CS / CH / DS / DH**
   （commensal/dysbiotic $\times$ static/HOBIC）の posterior を得た。
2. **拡張（修士後半）** — その枠組みを実データへ：
   **代謝 AGORA 符号 prior** + **in-vivo 縦断 16S** + **空間 PDE** + **FISH**。

\vspace{0.4em}
前半が力学系の骨格（アトラクター・pH 予測）を確立し、後半が
代謝の制約と空間構造でそれを実データに接地する。

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

## データフロー（実データ）

![](results/figures/pipeline_overview_pub.png){ height=66% }

実 Dieckow データ：guild $\varphi$（10 患者 $\times$ 3 週）→ AGORA 符号 prior →
gLV/Hamilton フィット → LOO-CV 検証 → 4 アトラクター。
`fit_*.json` が交換フォーマット。

---

## AGORA 符号 prior パイプライン

![](results/fig2_agora_pipeline.png){ height=64% }

(A) 手順；(B) 層追加で制約集合 $|\mathcal{M}|$ が $10\to22\to58$
（AGORA L3 で +36）；(C) 出来上がる符号 prior 行列 $P=\sgn(F)$。

---

## 経験的 sign agreement（naive 推定）

![](results/fig3_agora_sign_validation.png){ height=55% }

$$\mathrm{SA}=\frac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}}\mathbb{1}\!\left[\sgn(\hat A_{ij})=P_{ij}\right]=92\%.$$
\textcolor{red}{これは過大評価 — prior は符号縮退している（次スライドで誠実に検証）。}

---

## 誠実な検証

naive 92% は prior の符号縮退による過大評価。正しい対照は permutation：

- **cross-feeding 方向のみ**が独立に検証される：
  $$p=4\times10^{-4}\quad(z=+3.79,\ n_\pi=10^4).$$
- **競争方向は検証されない**（chance レベル）。
- **2 コホート**（Dieckow $\times$ Botelho）を prior 抜きで比べ、
  強ペア符号が **89%** 一致（$p\approx0.02$）。

\vspace{0.3em}
\textcolor{red}{AGORA prior は 16S 力学で再現されない $\Rightarrow$ モデリング選択であり、
データで確認済みではない。}

---

## ネットワークの視点

推定 $A$ の構造解析（class レベル）：

- **Veillonella（Negativicutes）が代謝のシンク** — 乳酸などを集約する受け手。
- 古典的な **Pg-keystone / Fn-bridge は class レベルでは支持されない**。
- dysbiosis で **Pg が中心化**する（固有中心性 $0.32\to0.51$）。
- 同時に **S. oralis–Veillonella の相利共生が崩れる**。

\vspace{0.4em}
dysbiosis は単なる組成変化でなく、相互作用ネットワークの**リワイヤリング**。

---

## 空間拡張 — FISH 深さプロファイル（CH vs DH）

![](results/diffusion_fit/zprofiles_all_ti_overlay.png){ height=50% }

CLSM-FISH から得た**深さ分解 5 種プロファイル**。commensal-HOBIC（CH）と
dysbiotic-HOBIC（DH）で深さ方向の構造が異なる。

---

## dysbiosis = 空間的再編成

![](results/diffusion_fit/depth_niche.png){ height=52% }

dysbiosis は組成変化に留まらず**空間的再編成**：
*P. gingivalis* が深部へ沈む（6 日目から **+30 µm**、嫌気ニッチ）。

---

## 新規：空間的 cross-feeding テスト

![](results/diffusion_fit/spatial_crossfeeding.png){ height=48% }

乳酸ペア **S. oralis（生産者）$\to$ Veillonella（消費者）**が空間的に層状化：
Veillonella が**より浅い**（10/10 サンプル、Wilcoxon $p=0.002$）。
生産者が基底、消費者がその上 — **上向きの乳酸拡散**と整合。

---

## 反応拡散 PDE

0D gLV を深さ $z$ 方向の反応拡散へ拡張：
$$\partial_t\varphi_i = D_i\,\partial_z^2\varphi_i - u\,\partial_z\varphi_i + R_i(\varphi),$$
$R_i$ は replicator 反応項（$A,b$ は時間 fit から固定）、
$D_i$（拡散）と $u$（移流）のみを深さプロファイルから推定。

\vspace{0.4em}
\textcolor{red}{現状は preliminary — HPC でのパラメータ sweep が収束中。}
空間 fit は時間動態と独立な観測量（FISH 深さ）でモデルを接地する。

---

## pH 予測（オリジナル 5 種研究・独立検証）

![](results/figures/fig_ph_validation.png){ height=52% }

5 種アトラクター posterior から pH を前向きに予測：
独立 $R^2=0.78$、RMSE $0.13$、LOO $R^2=0.92$。
**pH はキャリブレーションに使っていない** — 真に独立な検証。

---

## 結論

1. **代謝が生態的相互作用の符号を制約する**
   （$P_{ij}=\sgn(F_{ij})$；大きさは使わない）。
2. dysbiosis は**リワイヤリング + 空間的再編成**であり、
   *P. gingivalis* の深部沈降（+30 µm）を伴う。
3. その主張は **2 コホート**（Dieckow $\times$ Botelho, 89%, $p\approx0.02$）と
   **独立な機械論経路**（COMETS dFBA）で再現される。
4. オリジナルの 5 種 ODE は pH を独立に予測（$R^2=0.78$, LOO $R^2=0.92$）。

---

## 今後（Outlook）

- **空間的 out-of-sample 検証** — 別サンプルで深さプロファイルを予測。
- **2D / 3D PDE** — $z$ 方向だけでなく面内・立体構造へ。
- **PINN 逆問題** — 物理情報ニューラルネットで $D_i,u$ を直接推定。
- **GDI / Joshi 臨床検証** — 臨床メタデータ統合（dysbiosis index）。

\vspace{0.4em}
代謝が符号を制約し、dysbiosis がリワイヤリング＋空間再編であるという
描像を、空間検証と臨床指標で固めていく。

---

## まとめ（Take-home）

\begin{center}
\large
代謝は生態的相互作用の\textbf{符号}を制約する。\\[0.3em]
dysbiosis は\textbf{リワイヤリング＋空間再編}である。\\[0.3em]
\textbf{2 コホート}＋\textbf{機械論的 dFBA}＋\textbf{独立な pH 予測}で再現。
\end{center}

\vspace{0.6em}
すべてを 1 つの **10 ギルド分類**と **`fit_*.json`** が束ねる。
