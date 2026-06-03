---
title: "バイオフィルムの空間反応拡散モデル（Heine HOBIC）"
subtitle: "FISH 深さプロファイルからの空間輸送パラメータ逆推定 — 形式的取り扱い"
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
- **Dieckow** — in-vivo 相互作用の推定と検証（生態モデル本体）
- **Network** — 相互作用行列 $A$ の構造解析
- **Spatial-PDE** — FISH 深さプロファイルの反応拡散  ◀ **本デッキ**
- **FISH pipeline** — .lif → 5 種深さ組成

---

## 問い：WHO に WHERE を足す

bulk な ODE フィットは **誰が誰と相互作用するか（WHO）** を与える。
FISH 深さプロファイルは **それがどこで起こるか（WHERE）** を加える。

- 既知：bulk gLV/Hamilton の相互作用行列 $A$、増殖ベクトル $b$（Dieckow フィット）。
- 未知：種ごとの基質深さ方向の**輸送**（拡散・移流）。
- 目標：反応項を bulk から固定したまま、**空間輸送パラメータのみ**を
  深さ分解 FISH データから逆推定する。

\vspace{0.4em}
**主張：** dysbiosis は bulk 組成の変化ではなく、**空間的再編成**として現れる。
特に *P. gingivalis* が深部へ沈み、*F. nucleatum* から自律化する。

---

## データ：HOBIC フローチャンバー FISH（Heine 2025）

**11 個の `.lif` ファイル**、2 条件 × 5 日 × チタン基板：

- 条件：**CH**（commensal, HOBIC22）と **DH**（dysbiotic, HOBIC24）。
- 時点：Day 1 / 6 / 10 / 15 / 21。
- 5 種：So（*S. oralis*）、An（*Actinomyces*）、Vd/Vp（*Veillonella*）、
  Fn（*F. nucleatum*）、Pg（*P. gingivalis*）。
- プールした FOV 数：CH $=15/5/15/16/16$、DH $=7/3/3/2/2$。

\vspace{0.3em}
4 チャンネル FISH を 5 種にデコード（*F. nucleatum* $=$ 青 $\cap$ 赤の二重標識；
FISH デッキ参照）。z 軸 voxel 強度 $\to$ 深さ方向の組成 $\varphi_i(z,t)$。

---

## 深さプロファイル：CH vs DH

![](results/diffusion_fit/zprofiles_all_ti_overlay.png){ height=68% }

種ごとの $\varphi_i(z)$ を CH（commensal）と DH（dysbiotic）で重ね描き。
深さ $z$ は基質（$z=0$）から bulk（$z=L$）方向。

---

## 深さごとの積層組成

![](results/diffusion_fit/zprofiles_all_ti_stacked.png){ height=68% }

各深さ $z$ における $\{\varphi_i(z)\}_i$ の積層表示。$\sum_i \varphi_i + \varphi_0 = 1$
（$\varphi_0$ は空隙／非占有）を各 $z$ で満たす。

---

## 支配方程式（反応拡散 PDE）

深さ $z\in[0,L]$、組成 $\varphi_i(z,t)$ に対する反応 — 移流 — 拡散方程式：
$$\frac{\partial \varphi_i}{\partial t}
 = D_i\,\frac{\partial^2 \varphi_i}{\partial z^2}
 \;-\; u\,\frac{\partial \varphi_i}{\partial z}
 \;+\; R_i(\varphi).$$

- $R_i(\varphi)$ は replicator / Hamilton 反応項。$A,b$ は bulk フィットから **固定**。
  $$R_i(\varphi)=\varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big]+\gamma\,\varphi_i.$$
- **自由パラメータは拡散率 $D_i$ と移流 $u$ のみ**。
- $\gamma$ は Lagrange 項：各 $z$ で $\sum_i \varphi_i + \varphi_0 = 1$ を強制する。

---

## 境界条件と数値解法

**境界条件：**
$$\left.\frac{\partial \varphi_i}{\partial z}\right|_{z=0}=0\quad(\text{基質，no-flux}),\qquad
\varphi_i\big|_{z=L}=\varphi_{\text{bulk},i}\quad(\text{bulk，Dirichlet}).$$
$\varphi_{\text{bulk},i}$ は Day 1 の中央値で固定。

**数値解法（Method of Lines）：** Lie 演算子分割で反応と輸送を交互に解く。
$$e^{\Delta t\,\mathcal{L}} \approx e^{\Delta t\,\mathcal{R}}\,e^{\Delta t\,\mathcal{T}} + \mathcal{O}(\Delta t).$$

- 反応 $\mathcal{R}$：各 $z$ ノードで陰的 Euler（Newton 反復、stiff 対応）。
- 輸送 $\mathcal{T}$：陽的有限差分（中心差分 $\partial_z^2$、風上 $\partial_z$）。

---

## 逆問題

観測 $\varphi^{\text{obs}}(z,t)$ に対する輸送パラメータの最小二乗推定：
$$\min_{D,\,u}\;\; \sum_t \sum_z
 \big\lVert \varphi^{\text{pred}}(z,t;D,u) - \varphi^{\text{obs}}(z,t)\big\rVert^2 .$$

- 最適化は **L-BFGS-B**。stiff PDE を通した数値勾配（JAX 前向きモデル）。
- 前向きモデルは上記 MoL ソルバ。$A,b$ 固定なので未知は $D\in\mathbb{R}^5$ と $u$。
- 無次元化：**Péclet 数** $\mathrm{Pe} = uL/D_i$ が移流 vs 拡散の優劣を表す。
  $\mathrm{Pe}\gg1$ で移流支配、$\mathrm{Pe}\ll1$ で拡散支配。

---

## 予測 vs 観測：深さプロファイル

::: columns
:::: column
![](results/diffusion_fit/fit_CH.png){ height=62% }

CH（commensal）
::::
:::: column
![](results/diffusion_fit/fit_DH.png){ height=62% }

DH（dysbiotic）
::::
:::

予測（実線）と観測（点）の深さプロファイル。反応項固定・輸送のみフィット。

---

## フィットされたパラメータ（暫定）

| 種 | $D$（CH） | $D$（DH） |
|---|:---:|:---:|
| So | $0.0123$ | $0.0596$ |
| An | $0.0121$ | $0.0021$ |
| Vd/Vp | $0.0042$ | $2.1\times10^{-5}$ |
| Fn | $0.0075$ | $0.0019$ |
| Pg | $0.0115$ | $0.0083$ |
| **$u$（移流）** | $0.0038$ | $0.0060$ |
| **loss** | $0.128$ | $0.102$ |

\vspace{0.2em}
\textcolor{red}{**暫定値**：両条件とも `success=False`（optimiser 未収束）。}
ハイパーパラメータ sweep と高速収束設定を HPC で実行中。

---

## 空間生態 知見 1：Pg は dysbiosis で深く沈む

![](results/diffusion_fit/depth_niche.png){ height=58% }

*P. gingivalis* の質量中心（center-of-mass）が、DH では Day 6 以降
**最大 $+30\,\mu\mathrm{m}$ 深く**シフトする。深部嫌気ニッチへの沈降。

---

## 空間生態 知見 2：Fn–Pg ブリッジは初期のみ

![](results/diffusion_fit/fn_pg_coloc.png){ height=50% }

Manders $M_1$（Pg のうち Fn と共局在する割合）：
DH Day 1 $=0.76$ → Day 6 以降 $0.15\text{–}0.20$。CH は $0.33\text{–}0.49$。
$\Rightarrow$ Pg は Fn から **脱共役**し、自律的に振る舞う。

---

## 空間生態 知見 3：cross-feeding は空間的に階層化

![](results/diffusion_fit/spatial_crossfeeding.png){ height=48% }

AGORA の乳酸バックボーン（**S.oralis 生産者 → Veillonella 消費者**）が深さに現れるか検定。
Veillonella の重心は **10/10 サンプルで一貫して S.oralis より浅い**（平均 $-2.8\,\mu$m, Wilcoxon $p=0.002$）。
$\Rightarrow$ 生産者が基底層・消費者がその上＝**上方への乳酸拡散**と整合。代謝バックボーンの**空間的検証**。
（A.naeslundii は非有意 $p=0.28$。）

---

## その他：bulk は CH ≈ DH

CH–DH の **Bray–Curtis divergence** $\approx 0.2$、Day 1 から**ほぼ一定**。

- 差は接種（inoculum）で決まり、時間発展でほとんど拡大しない。
- bulk 組成では CH $\approx$ DH $\Rightarrow$ dysbiosis は **空間的・時間的**であり、
  bulk 組成のシフトではない。

\vspace{0.4em}
これは知見 1・2 と整合する：違いは「誰がどれだけいるか」ではなく
「どこにいて、何と組むか」に現れる。

---

## 拡散率 vs 生態的中心性

![](results/diffusion_fit/d_vs_centrality.png){ height=52% }

フィットした拡散率 $D_i$ は gLV 相互作用強度（中心性）と **負相関**：
CH $r=-0.40$、DH $r=-0.47$（5 種、illustrative）。

\vspace{0.2em}
$\Rightarrow$ 「中心的な種ほど動かない」。生態的ハブが空間的に固定される傾向。

---

## in-vitro ↔ in-vivo マッピング

![](results/diffusion_fit/hobic_vs_dieckow.png){ height=50% }

HOBIC の 5 種を Dieckow の 5 ギルドに対応づけ、**順位が一致**：
Spearman $\rho = 0.70$。

- in-vivo は *S. oralis* 優占、定義済み inoculum は均等接種。
- 順位構造は保たれる $\Rightarrow$ in-vitro モデルは in-vivo の縮約として妥当。

---

## 結論

1. bulk ODE（WHO）に FISH 深さプロファイル（WHERE）を足し、反応項固定で
   **輸送パラメータ $D_i,u$ のみ**を逆推定する枠組みを構築。
2. **dysbiosis $=$ 空間的再編成**：*P. gingivalis* が深部へ沈み（$+30\,\mu\mathrm{m}$）、
   *F. nucleatum* から自律化（Manders $M_1$ が初期のみ高い）。
3. bulk は CH $\approx$ DH（Bray–Curtis $\approx0.2$ で一定）— 違いは bulk でなく空間。
4. 「中心的な種ほど動かない」（$D$ と中心性が負相関）；in-vitro↔in-vivo の
   順位は一致（$\rho=0.70$）。
5. \textcolor{red}{輸送パラメータは**暫定**（未収束）}。HPC で sweep・高速収束設定が収束中。
