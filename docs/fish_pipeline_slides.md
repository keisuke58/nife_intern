---
title: "HOBIC FISH データ処理"
subtitle: "4チャンネル → 5菌種 の解読と深さプロファイル抽出"
author: "Nishioka — NIFE / SFB TRR-298"
date: "2026-06-03"
mainfont: "Noto Sans CJK JP"
monofont: "Noto Sans Mono CJK JP"
CJKmainfont: "Noto Sans CJK JP"
CJKmonofont: "Noto Sans Mono CJK JP"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
---

## このプロジェクトで何をしたか（1枚要約）

顕微鏡で撮ったバイオフィルムの3D画像を
**「深さごとに5菌種がどんな割合でいるか」** の数表に変換した。

- 入力：Leica 共焦点 **FISH `.lif`**（フローチャンバー HOBIC, Heine 2025）
- 難所：画像は **4色しかないのに菌は5種**
  → 論文の標識設計を読み解いて正しく分離
- 出力：**深さ × 5菌種 の組成プロファイル**（PDE 拡散フィットの入力）

\vspace{0.5em}
新規 `fish_decode.py` / `lif_quicklook.py`、修正 `lif_to_zprofiles.py`

---

## 入力データ：HOBIC FISH `.lif`

全11本。**commensal (HOBIC22→CH) と dysbiotic (HOBIC24→DH) が揃った**。

| 実験 | 条件 | Day | 基質 |
|---|---|---|---|
| 220518/601/720, 220817, 240416 | CH | 1,6,10,15,21 | Ti |
| 241203 (Tag1) | DH | 1 | Ti |
| 241018 | DH | 6,10,15,21 | **Ti＋Glass混在** |

- 1画素 0.18 µm、z 間隔 2 µm。**基質はTi採用**（HOBIC=チタンインプラント、CHも無印=Ti）。Glass 9 FOV は除外・別解析用に温存。
- **ヘッドレス環境** → Fiji/napari 不可。`readlif`→**PNG**（Times＋µmスケールバー）

---

## 生の4チャンネル（撮ったまま）

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__overview.png){ height=72% }

行＝視野、列＝Blue / Yellow / Green / Red ＋合成

---

## 問題：4チャンネル ↔ 5菌種（1対1ではない）

検出器は **4チャンネル**、群集は **5菌種**。単純な「色＝菌」ではない。

原著 **Heine 2025, Front. Oral Health, Table S5 + Methods §2.6**：

> *F. nucleatum was targeted by two probes ... labeled with different dyes
> – resulting in co-localized blue and red fluorescence.*

→ **F. nucleatum だけ二重標識**（Alexa405=青 ＆ Alexa647=赤）。
青と赤の **両方に光る**。

---

## 解読ルール（colocalization）

| LUT / レーザ | 寄与する菌種 |
|---|---|
| **Blue** 405 nm | S. oralis ＋ **F. nucleatum** |
| **Green** 488 nm | A. naeslundii |
| **Yellow** 552 nm | V. dispar/parvula |
| **Red** 638 nm | P. gingivalis ＋ **F. nucleatum** |

```
F. nucleatum  = Blue ∩ Red
S. oralis     = Blue − (Blue ∩ Red)
P. gingivalis = Red  − (Blue ∩ Red)
```

※ ボクセル単位で計算（xy平均の前に）。`min` の平均 ≠ 平均の `min`

---

## 解読後の5菌種

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__species.png){ height=72% }

F. nucleatum（紫）が S.oralis / P.gingivalis と分離できている

---

## 直したバグ（重要）

旧コードは「青=S.oralis純 / 赤=P.gingivalis純 / 紫=F.nucleatum」を仮定。
**実ファイルに紫チャンネルは無い** → そのままだと：

- **F. nucleatum を丸ごと取りこぼし**
- **S.oralis・P.gingivalis を過大計上**（Fn が混入）

→ PDE 入力が **3菌種ぶん間違う**ところだった。
`fish_decode.py`（両ツール共有の正典デコーダ）で修正。

---

## 抽出・統合・メール規約

**z プロファイル**：各視野の3Dを「深さ毎の xy 平均強度」に潰す。

**レプリケート統合**：複数 `.lif` を `(condition, day)` でプール
（Day1 は別実験日3本 → 15視野を1回だけ平均）。

**Heine メール規約（2026-05-26）をツールに実装**：

- HOBIC22 = commensal → **CH** / HOBIC24 = dysbiotic → **DH**（自動）
- 日付 = ファイル名末尾 `TagN`（HOBIC24 のみ series 名先頭で複数日分割）
- **基質フィルタ** `--substrate ti`（Ti/Glass混在からTi採用）

```bash
python scripts/pde/lif_to_zprofiles.py "HOBIC FISH"/*.lif --substrate ti
```

---

## 結果：CH / DH の深さ時系列（Ti）

![](results/diffusion_fit/zprofiles_all_ti.png){ height=62% }

CH/DH × Day 1/6/10/15/21 → `zprofiles_all_ti.csv`（400行）。
両条件とも厚化＋深さ方向の組成シフト。DH後期はTi FOV少（Glass除外）。

---

## CH vs DH を重ねて読む（深部 Pg）

![](results/diffusion_fit/zprofiles_all_ti_overlay.png){ height=58% }

**実線=CH / 破線=DH**、色=種、縦軸=µm「表層→深部」。
Day6以降、**Pg（赤）破線が実線より深部へ** = dysbiosis で Pg が沈む。

---

## 各深さの組成（積み上げ面）

![](results/diffusion_fit/zprofiles_all_ti_stacked.png){ height=62% }

各深さで5種比率を積み上げ（合計=1）。**DH下段の深部で赤(Pg)が張り出す**のが一目瞭然。

---

## この後：拡散フィットとは

組成を決める2つの力を分けて考える反応拡散 PDE：

1. **反応項 (A, b)** = 種間相互作用。**既知**（gLV / TMCMC）
2. **拡散 (D_i) ＋ 移流 (u)** = 空間的な動き。**未知＝フィット対象**

```
D_i, u を仮定 → PDE を解く → 予測 vs 実測(MSE)
            ↑___ ズレ最小の D_i, u を探索 (L-BFGS) ___|
```

**HPC(PBS)で本番フィット実行**（CH/DH, Ti）。暫定結果 u: CH 0.0038 / DH 0.0060。
※ 両者 `success=False`（未収束）＝暫定。DH 後期は Ti FOV 少で弱拘束 → 要 restart 増。

---

## 発見：dysbiosis = Pg の深部・自律化

![](results/diffusion_fit/depth_niche.png){ height=52% }

- **P.gingivalis は DH で Day6以降 深部へ沈降**（重心 +最大30µm、嫌気層）
- **Fn–Pg 橋渡しは初期限定**：DH Day1 で coloc 0.76 → Day6以降は CH>DH（Pgが Fn から離れ自律化）
- CH/DH 群集差は接種時から ~0.2 で一定。DH後期は n=2 で CI 広い

---

## 成果物＆データの限界

| ファイル | 役割 |
|---|---|
| `fish_decode.py` | 新規・正典デコーダ (Fn=blue∩red) |
| `scripts/pde/lif_quicklook.py` | 新規・可視化 (Times＋µmバー) |
| `scripts/pde/lif_to_zprofiles.py` | 修正・解読＋統合＋規約＋基質 |
| `zprofiles_all_ti.csv` / `D_fit_*.json` | PDE入力＋フィット結果 |

**限界**：HOBIC由来なので **CH/DH のみ**（static CS/DS 無し）。
241018(DH)の **Glass除外**。DH後期(Day15/21)は Ti FOV 2枚と少。
