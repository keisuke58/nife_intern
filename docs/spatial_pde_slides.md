---
title: "バイオフィルムの空間反応拡散モデル（Heine HOBIC）"
subtitle: "FISH 深さプロファイルからの空間輸送パラメータ逆推定 — 形式的取り扱い"
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
- **Network** — 相互作用行列 $A$ の構造解析
- **Spatial-PDE** — FISH 深さプロファイルの反応拡散  ◀ **本デッキ**
- **FISH pipeline** — .lif → 5 種深さ組成

---

## 動機：時間動態から空間構造へ

ODE フィットは**菌種間の相互作用関係**（誰が誰に作用するか）を与える。
FISH 深さプロファイルはその**空間的文脈**（バイオフィルム内での位置）を加える。

- 既知：gLV/Hamilton の相互作用行列 $A$ と増殖ベクトル $b$
  （**① Heine 5 菌種 TMCMC posterior**）。
- 未知：菌種ごとの深さ方向の**空間輸送**（拡散係数 $D_i$・移流速度 $u$）。
- 目標：反応項を①から固定し、**空間輸送パラメータのみ**を
  深さ分解 FISH データから逆推定する。

\vspace{0.4em}
**主張：** dysbiosis は bulk 組成のシフトではなく、**空間的再編成**として現れる。
とりわけ *P. gingivalis* が深部へ移行し、*F. nucleatum* との共局在が失われる。

---

## 土台：① Heine 5 菌種 Bayesian ODE フィット

本デッキは **Heine 2025 の 5 菌種 in-vitro 研究（①）の空間拡張**である。

![](results/heine2025/glv_heine_fit_thesis.pdf){ height=36% }

- 図：gLV MAP 軌道（非対称 $A$、RMSE 0.012–0.032）+ 実験 IQR 箱ひげ図。
- Hamilton NUTS（対称 $A$, GPU-TMCMC $N_p=10{,}000$）も実施；4 アトラクター CS/CH/DS/DH 再現。
- その **$A,b$（反応項）を固定**し、同じ Heine 系の **HOBIC FISH 深さデータ**へ拡張 →
  空間輸送 $D_i,u$ を加える。
- すなわち①は**時間動態**（相互作用）、本デッキは**空間構造**（輸送）を扱う。Heine 系が一貫した研究対象。

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

- $R_i(\varphi)$ は replicator / Hamilton 反応項。$A,b$ は **① Heine 5 菌種 TMCMC
  posterior**（$N_p=10{,}000$, `ultimate_10000p`）から **固定**（本デッキはその空間拡張）。
  $$R_i(\varphi)=\varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big]+\gamma\,\varphi_i.$$
- **自由パラメータは拡散率 $D_i$ と移流 $u$ のみ**（反応は①で確定済み）。
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

## フィットされたパラメータ（収束・限定同定）

| 種 | $D$（CH） | $D$（DH） |
|---|:---:|:---:|
| So | $10^{-5}\,^\dagger$ | $0.0100$ |
| An | $10^{-5}\,^\dagger$ | $0.0003$ |
| Vd/Vp | $0.0015$ | $10^{-5}\,^\dagger$ |
| Fn | $0.0106$ | $0.0020$ |
| Pg | $10^{-5}\,^\dagger$ | $0.0013$ |
| **$u$（移流）** | $0.0$ | $0.0070$ |
| **loss** | $\mathbf{0.030}$ | $\mathbf{0.091}$ |

\vspace{0.2em}
sweep の最良は**収束**（`success=True`）。しかし $^\dagger$ の **複数 $D_i$ が下限 $10^{-5}$ に張り付く**
（CH: So/An/Pg、DH: Vd/Vp）。
\textcolor{red}{$\Rightarrow$ 深さプロファイルは 5 つの拡散率を一意に決めない（**限定的同定性**）。}
$D_i$ は illustrative。3D PINN も同結論（同定性の限界自体が知見）。

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

## 空間生態 知見 4：Fn–Pg 脱共役は「深さ」の分離

![](results/fish_3d/fish_3d_fnpg_coloc.png){ height=50% }

84 FOV の**全 voxel 3-D デコード**で Manders $M_1$ を次元別に分解：

- **横（xy 投影）** $M_1 \approx 1.0$（DH, Day 6 から飽和）— Fn と Pg は同じ $x,y$ フットプリント。
- **3-D（voxel）** $M_1$ は Day 6 から **$0.22\text{–}0.33$ へ低下**（CH は $0.4\text{–}0.7$ を維持）。
- $\Rightarrow$ 横では重なるが**深さ $z$ で分離**：dysbiotic な Fn–Pg 脱共役は**垂直分離**で、Pg が Fn の**下**に位置。「Pg が深部へ沈む」の **3-D 確証**。

---

## 空間生態 知見 5：DH は横方向に均一化（confluent mat）

![](results/fish_3d/fish_3d_lateral_heterogeneity.png){ height=50% }

横方向（xy）の不均一性を時間で追跡（平均 xy-CV）：

- **DH** は Day 6 以降 **CV $\approx 0.33$** へ均一化 — 連続した**コンフルエントなマット**。
- **CH** は **CV $0.4\text{–}0.9$** のまま**パッチ状**（マイクロコロニー構造を保持）。
- $\Rightarrow$ dysbiosis は横方向では**平滑化**、commensal は**微小構造を保つ**。知見 4 の垂直分離と相補的。

---

## バルク組成：CH ≈ DH

CH–DH の **Bray–Curtis 非類似度** $\approx 0.2$、Day 1 から**ほぼ一定**。

- 群集組成の差は接種（inoculum）時点で決まり、その後ほとんど拡大しない。
- バルク組成において CH $\approx$ DH であることから、dysbiosis の差異は
  **空間的・時間的構造**にあり、バルク組成のシフトではない。

\vspace{0.4em}
この結果は知見 1・2 と整合する：差異は種の存在量ではなく、
その**空間的分布と相互作用の再配線**に現れる。

---

## 拡散率 vs 生態的中心性

![](results/diffusion_fit/d_vs_centrality.png){ height=52% }

フィットした拡散率 $D_i$ は gLV 相互作用強度（中心性）と **負相関**：
CH $r=-0.40$、DH $r=-0.47$（5 種、illustrative）。

\vspace{0.2em}
中心性の高い菌種ほど拡散係数が小さい傾向が示され、生態的ハブは空間的に固定される傾向がある。

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
