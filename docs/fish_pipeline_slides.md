---
title: "HOBIC FISH データ処理"
subtitle: "4チャンネル蛍光画像からの5菌種深さプロファイル抽出"
author: "Nishioka — NIFE"
date: "2026-06-04"
mainfont: "Noto Sans CJK JP"
monofont: "Noto Sans Mono CJK JP"
CJKmainfont: "Noto Sans CJK JP"
CJKmonofont: "Noto Sans Mono CJK JP"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
---

## 概要

Heine 2025（HOBIC フローチャンバー）の共焦点 FISH 画像を処理し、
反応拡散 PDE フィットの入力となる**深さ分解 5 菌種組成プロファイル**を構築した。

- 入力：Leica 共焦点 **FISH `.lif`**（CH / DH 条件，Ti 基質）
- 課題：検出チャンネルは **4** チャンネルのみ，対象菌種は **5 種**
- 対処：原著の標識設計を読み解き，colocalization デコードで分離
- 出力：**深さ × 5 菌種 の組成プロファイル**，PDE フィット結果（$D_i$, $u$）

\vspace{0.4em}
新規実装：`fish_decode.py`，`lif_quicklook.py`；更新：`lif_to_zprofiles.py`

---

## 入力データ：HOBIC FISH `.lif`（全 11 ファイル）

| 実験バッチ | 条件 | Day | 基質 |
|---|---|---|---|
| 220518/601/720, 220817, 240416 | CH | 1, 6, 10, 15, 21 | Ti |
| 241203 (Tag1) | DH | 1 | Ti |
| 241018 | DH | 6, 10, 15, 21 | Ti + Glass |

- 画素サイズ 0.18 µm，z ステップ 2 µm。Ti 基質のみを解析に使用
  （HOBIC はチタンインプラントモデル；Glass は 9 FOV を除外・別途保存）。
- ヘッドレス環境のため GUI ツール（Fiji / napari）は利用不可。
  `readlif` で読み込み，Times フォント＋µm スケールバー付き PNG を出力。

---

## 生 4 チャンネル画像

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__overview.png){ height=72% }

行 = 視野（FOV），列 = Blue / Yellow / Green / Red ＋合成

---

## チャンネル数と菌種数の不一致

4 チャンネル検出器に対して解析対象は 5 菌種であり，単純な色対菌種の対応は成立しない。

原著（Heine 2025, *Front. Oral Health*, Table S5 + Methods §2.6）には次の記述がある：

> *F. nucleatum was targeted by two probes ... labeled with different dyes
> – resulting in co-localized blue and red fluorescence.*

すなわち *F. nucleatum* のみが二重標識（Alexa405 = 青，Alexa647 = 赤）であり，
両チャンネルに蛍光シグナルが現れる。

---

## デコードルール（共局在解析）

| 検出チャンネル | 対応菌種 |
|---|---|
| **Blue** 405 nm | *S. oralis* ＋ *F. nucleatum* |
| **Green** 488 nm | *A. naeslundii* |
| **Yellow** 552 nm | *V. dispar/parvula* |
| **Red** 638 nm | *P. gingivalis* ＋ *F. nucleatum* |

$$
F_n = B \cap R,\quad S_o = B - F_n,\quad P_g = R - F_n
$$

共局在の計算はボクセル単位で行い，その後 xy 平均を取る
（$\mathrm{mean}(\min) \neq \min(\mathrm{mean})$）。

---

## デコード後の 5 菌種

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__species.png){ height=72% }

*F. nucleatum*（紫）が *S. oralis* / *P. gingivalis* と適切に分離されている。

---

## 旧コードのデコード誤りと修正

旧コードは「青 = *S. oralis* 純チャンネル / 赤 = *P. gingivalis* 純チャンネル /
紫 = *F. nucleatum*」を前提としていた。しかし実ファイルに紫チャンネルは存在しないため，
修正前のコードでは以下の誤りが生じる：

- *F. nucleatum* のシグナルを完全に取りこぼす
- *S. oralis* と *P. gingivalis* の計上値に *F. nucleatum* 由来の混入が生じる

この誤りは PDE の空間入力 5 菌種のうち 3 種に影響する。
正典デコーダ `fish_decode.py` を新規実装し，両ツールで共有する形で修正した。

---

## プロファイル抽出・統合・実験命名規則

**z プロファイル**：各 FOV の 3D 画像を深さごとの xy 平均強度に集約する。

**レプリケート統合**：同一 `(condition, day)` の複数 `.lif` をプール
（Day 1 は異なる実験日 3 本，計 15 FOV を一括平均）。

**実験命名規則**（Heine, 2026-05-26 メールより）をスクリプトに実装：

- HOBIC22 = commensal → **CH**，HOBIC24 = dysbiotic → **DH**（自動判定）
- Day 番号：ファイル名末尾の `TagN`（HOBIC24 のみ series 名先頭の整数で分割）
- **基質フィルタ** `--substrate ti`：Ti/Glass 混在データから Ti のみを選択

```bash
python scripts/pde/lif_to_zprofiles.py "HOBIC FISH"/*.lif --substrate ti
```

---

## 結果：CH / DH の深さ時系列（Ti）

![](results/diffusion_fit/zprofiles_all_ti.png){ height=62% }

CH / DH × Day 1/6/10/15/21 のプロファイルを `zprofiles_all_ti.csv` に出力（400 行）。
いずれの条件においても日数の経過とともにバイオフィルムが厚化し，深さ方向の組成シフトが観察される。

---

## CH / DH の比較（深部 *P. gingivalis*）

![](results/diffusion_fit/zprofiles_all_ti_overlay.png){ height=58% }

**実線 = CH，破線 = DH**；色は菌種，縦軸は µm（表層 → 深部）。
Day 6 以降，*P. gingivalis*（赤）の重心が DH 条件で CH より深部へ移行している。

---

## 各深さの組成（積み上げ面グラフ）

![](results/diffusion_fit/zprofiles_all_ti_stacked.png){ height=62% }

各深さにおける 5 菌種の比率を積み上げ（合計 = 1）。
DH 条件の深部層において *P. gingivalis*（赤）の割合が増大することが確認される。

---

## 反応拡散 PDE フィット

バイオフィルム内の組成変化を，生態的相互作用と空間的輸送に分離してモデル化する：

1. **反応項** $(A,\,b)$：菌種間相互作用行列。**既定値**（gLV / TMCMC 推定済み）
2. **拡散** $D_i$ **＋ 移流** $u$：深さ方向の空間的輸送。**フィット対象**

$$
\partial_t \varphi_i = D_i\,\partial_{zz}\varphi_i - u\,\partial_z\varphi_i
+ \varphi_i\!\left[(A\varphi+b)_i - \varphi^\top(A\varphi+b)\right]
$$

$D_i$ と $u$ を初期値から繰り返し更新し，MSE を最小化（L-BFGS）。
HPC（PBS）にて CH / DH 各条件で実行（Ti, $N_z=48$, 8 restarts）。

---

## フィット結果：拡散係数 $D_i$ と移流速度 $u$

| 菌種 | $D^\text{CH}$ | $D^\text{DH}$ |
|---|---|---|
| *S. oralis* | 0.029 | 0.015 |
| *A. naeslundii* | 0.006 | 0.006 |
| *Vd/Vp* | 0.005 | 0.009 |
| *F. nucleatum* | 0.006 | 0.015 |
| *P. gingivalis* | 0.006 | 0.006 |

移流速度：$u^\text{CH} = 0.0069$，$u^\text{DH} = 0.0059$（推定単位：µm/s 相当）

DH 条件：`success=True`（収束）；CH 条件：`success=False`（未収束，暫定値）。
`D_fit_*_nz48_eps3e-4.json` に保存。

---

## フィット結果：CH 条件

![](results/diffusion_fit/fit_CH_nz48_eps3e-4.png){ height=72% }

観測プロファイル（点）と PDE 予測（線）の比較（CH, $N_z=48$, loss = 0.125）。

---

## フィット結果：DH 条件

![](results/diffusion_fit/fit_DH_nz48_eps3e-4.png){ height=72% }

観測プロファイル（点）と PDE 予測（線）の比較（DH, $N_z=48$, loss = 0.154，収束済み）。

---

## 所見：dysbiosis における *P. gingivalis* の深部集積

![](results/diffusion_fit/depth_niche.png){ height=52% }

- *P. gingivalis* は DH 条件で Day 6 以降に深部へ移行（重心差 最大 +30 µm，嫌気域）
- *F. nucleatum*–*P. gingivalis* 共局在は DH 初期（Day 1，coloc = 0.76）に限定；
  Day 6 以降は CH > DH となり，*P. gingivalis* が *F. nucleatum* から独立して深部に集積する
- CH / DH の群集組成差は Day 1 からすでに ~0.2 で，以降ほぼ一定

---

## 成果物とデータの制約

| ファイル | 役割 |
|---|---|
| `fish_decode.py` | 新規・正典デコーダ（$F_n = B \cap R$） |
| `scripts/pde/lif_quicklook.py` | 新規・画像可視化（Times フォント，µm バー） |
| `scripts/pde/lif_to_zprofiles.py` | 更新・デコード＋統合＋命名規則＋基質フィルタ |
| `zprofiles_all_ti.csv` | PDE 入力（深さ × 菌種 × 条件） |
| `D_fit_*_nz48_eps3e-4.json` | フィット結果（$D_i$, $u$） |

**データ上の制約**：HOBIC 由来のため **CH / DH のみ**（static 条件 CS/DS は対象外）。
241018 (DH) の Glass FOV は除外済み。DH 後期（Day 15/21）は Ti FOV 数が 2 枚と少なく，推定精度に影響する。
