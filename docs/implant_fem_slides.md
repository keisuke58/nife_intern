---
title: "インプラント FEM：ペリインプラント・バイオフィルムの組成分解力学"
subtitle: "組成 $\\varphi_{Pg}$ → 成長固有ひずみ → インプラント界面残留応力 → 剥離"
author: "西岡佳祐 — NIFE / SFB TRR-298"
date: "2026-06-13"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## 本デッキの位置づけ

**Implant FEM** — ペリインプラント・バイオフィルムの力学を組成分解で扱う。

データの流れ：

raw 16S → guild $\varphi$ → gLV/Hamilton（＋符号 prior）→ 空間 PDE → **FEM 力学**

体制（supervisor context）：

- **Soleimani**（IKM, FEM）— 連続体力学・有限要素の枠組み
- **Junker**（IKM, Hamilton 原理）— 材料モデルの変分定式化
- **Szafrański**（MHH / NIFE）— 口腔バイオフィルム・臨床

\vspace{0.4em}
**一行命題：** dysbiosis を *力学的な*インプラント界面リスクとして読み出す。

---

## 臨床的問題：ペリインプラント炎

ペリインプラント炎はバイオフィルム駆動の炎症であり、インプラント周囲の
**支持骨喪失**を引き起こす。

- インプラント患者の **20–35%** に発症 — インプラント失敗の主因。
- **NIFE** = インプラント研究開発拠点。基材（substratum）は**チタン**。
- **Dieckow コホート** = アバットメント上のバイオフィルムを縦断観察。

\vspace{0.4em}
力学的に何が起きているか：バイオフィルムが Ti 界面で成長し、界面に
**応力**を生み、やがて**剥離**する。これを連続体力学で定量化する。

---

## アイデア：組成から力学への連鎖

中心となる因果連鎖：

$$\varphi_{Pg}\;\rightarrow\;\text{成長固有ひずみ}\;\rightarrow\;\text{界面残留応力}\;\rightarrow\;\text{剥離}\;\rightarrow\;\text{ペリインプラント炎}$$

![](masterarbeit_ansys_fem/figures/fem_clinical_schematic.pdf){ height=46% }

CH（常在）対 DH（dysbiotic）バイオフィルム下の Ti ねじ山：成長誘起の界面応力と剥離。

---

## 方法：Hamilton 原理の材料モデル

各 Gauss 点で**現象論則の代わりに** Python の Hamilton 原理材料モデルを呼ぶ。

- 乗法的な成長分解 $F = F_e\,F_g$ ＋ neo-Hookean 弾性。
- 成長 $F_g$ は組成 $\varphi_{Pg}$ の固有ひずみで駆動。
- 弾性応答 $F_e$ が応力を担い、界面で残留応力として蓄積。

\vspace{0.3em}
**検証（Abaqus）：**

- 微小ひずみ：$\sigma = E\,\varepsilon$ と一致。
- 有限ひずみ（NLGEOM）：neo-Hookean 解と一致。

\vspace{0.3em}
$\Rightarrow$ 材料則は外部 FEM ソルバへそのまま差し込める（カーネルとして検証済み）。

---

## 成長キャリブレーション

成長固有ひずみを **CLSM 深さプロファイル**からキャリブレーション。

![](masterarbeit_ansys_fem/figures/F2_growth_calibration.pdf){ height=52% }

- DH：深さ相関 $0.88$、$\beta_{\text{depth}} \approx 17.2$。
- 常在（commensal）コントロールは**逆符号** — 単なる体積膨張ではない。

---

## 結果：残留応力カラム $S_{11}(z)$

![](masterarbeit_ansys_fem/figures/F4_residual_stress_DHvsCH.pdf){ height=54% }

- 応力は**基材**（インプラント–アバットメント壁）に集中、自由な上端 $\approx 0$。
- $S_{11}(z)$ は $\varphi_{Pg}(z)$ を**追従** — 深部 Pg 負荷が界面応力を作る。

---

## 結果：剥離（cohesive 界面）

![](masterarbeit_ansys_fem/figures/F6_delamination_DHvsCH.pdf){ height=52% }

同一の界面強度で cohesive-interface FEM を回す：

- **dysbiotic 83%** 対 **commensal 25%** が剥離（**3.3×**）。
- 唯一の原因 = **Pg 深さプロファイル**（強度・荷重は同一）。

---

## 3D ヒーロー：ペリインプラント全アセンブリ

![](masterarbeit_ansys_fem/figures/fem_implant_screw3d.pdf){ height=50% }

インプラント＋隣接歯＋共有歯槽骨。咬合荷重は**歯頸部のペリインプラント骨**に
集中（marginal-bone-loss シグネチャ）。組成メカニズムは**現実の解剖学へ
そのまま転移**する。

---

## クラウン荷重伝達：モーメントアーム効果

![](masterarbeit_ansys_fem/figures/fem_implant_crown_fem.pdf){ height=50% }

実歯クラウンをアバットメントに装着し、荷重を**咬合面（z≈38, 隣在歯と同一平面）**
で受けると、首部まわりの**曲げモーメントが約3.4倍**に。生理的咬合（100 N, 30°,
ISO 14801）でも**ペリインプラント歯槽頂骨のピーク応力が ×1.6（55→88 MPa）**。
$\Rightarrow$ 補綴設計（クラウン高・荷重位置）が**辺縁骨リスクに直接効く**。

---

## 2D 連成セクション（補助）

![](masterarbeit_ansys_fem/figures/fem_tier2b_real.pdf){ height=52% }

連成 implant＋tooth＋bone の近遠心断面。(A) 多材料解剖、(B) 咬合 von Mises の
共有骨連成 — **歯頸部応力 = ペリインプラント辺縁骨喪失のシグネチャ**。

---

## 患者ブリッジ

![](masterarbeit_ansys_fem/figures/F1c_patient_bridge_validation.pdf){ height=48% }

Dieckow 10 患者を検証済みの剥離カーネル $J$ へ射影：

- 患者間ばらつき **$CV = 0.86$** は**測定された** Pg 負荷が決める。
- 借用した in-vitro 深さ形状は **34× 弱いレバー**（$\pm 8\%$）。

\vspace{0.2em}
$\Rightarrow$ **0-D の臨床 Pg 読み値**が順序的な剥離リスクの代理として十分。

---

## 検証・限界・展望

**正直な評価：**

- 成長の**形状**は CLSM から較正済み。
- 応力**絶対値**はパラメトリック（バイオフィルム弾性率は約 7 桁の幅 →
  $\sigma/\mu$ で報告）。
- 主張は**順序的・機構的**であり予測的ではない。差別化は*バイオフィルム連成*
  であって汎用インプラント力学ではない。

**検証実験：** 深さ分解 AFM／剪断流剥離 対 Pg 負荷。

\vspace{0.3em}
**展望：** 成長固有ひずみ→界面応力→剥離の能力は、慶應・村松研の計算固体力学
および半導体薄膜応力・ダイシングストリート剥離（DISCO）へ転移する。
