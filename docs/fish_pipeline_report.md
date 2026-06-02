# HOBIC FISH（CLSM）データ処理レポート

**目的**: Heine 2025 のフローチャンバー（HOBIC）FISH 共焦点画像（`.lif`）を、空間 PDE 拡散フィットの入力（深さ方向の5種組成プロファイル）に変換する。

作成: 2026-06-03 / 対象データ: `HOBIC FISH/*.lif`（Szafrański lab → Nils Heine 提供）

---

## 0. 一言でいうと

> 顕微鏡で撮ったバイオフィルムの3D画像を「**深さごとに5菌種がどんな割合でいるか**」の数表に変換した。
> ただし画像は**4色しかないのに菌は5種**なので、論文の標識設計を読み解いて正しく5種に分離するのが肝だった。

---

## 1. どんなデータを処理したのか

Leica 共焦点（CLSM）の `.lif` ファイル。1ファイル＝ある実験日のバイオフィルムを複数視野(FOV)で撮った z スタック（輪切り画像の束）。

| ファイル | モデル | 日 (Day) | 視野数 | z枚数/視野 | 1画素 |
|---|---|---|---|---|---|
| 220518_HOBIC22 | commensal | 1 | 6 | 12–24 | 0.18 µm |
| 220601_HOBIC22 | commensal | 1 | 5 | 〃 | 〃 |
| 220720_HOBIC22 | commensal | 1 | 4 | 〃 | 〃 |
| 220817_HOBIC22_Tag15 | commensal | 15 | 7 | 〃 | 〃 |
| 220817_HOBIC22_Tag21 | commensal | 21 | 4 | 〃 | 〃 |

- **環境はヘッドレス**（`DISPLAY` 無し）→ Fiji / napari の GUI は使えない。
- そこで `readlif` で読んで **PNG に書き出す自作ツール**で確認・可視化した（`lif_quicklook.py`）。

### 生の4チャンネル（撮ったまま）

![raw 4 channels](../figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__overview.png)

各行が視野、列が検出チャンネル（Blue / Yellow / Green / Red）＋合成。青が密、黄/緑/赤がスパースな細胞として見える。

---

## 2. 核心：4チャンネル ↔ 5菌種（1対1ではない）

`.lif` の検出器は**4チャンネルしかない**のに、群集は**5菌種**。これは単純な「色＝菌」ではない。

原著 **Heine et al. 2025, *Front. Oral Health* 1649419, Supplementary Table S5 + Methods §2.6** を精読して判明：

> *F. nucleatum was targeted by two probes that shared the same nucleotide sequence, but were labeled with different dyes – resulting in co-localized blue and red fluorescence.*

**F. nucleatum だけ二重標識**（同じ配列のプローブを Alexa 405=青 と Alexa 647=赤 の両方で）。つまり Fn は青と赤の**両方に光る**。

| LUT / レーザ / 検出域 | 寄与する菌種 |
|---|---|
| **Blue** 405 nm / 413–477 nm | S. oralis ＋ **F. nucleatum** |
| **Green** 488 nm / 509–576 nm | A. naeslundii（単独） |
| **Yellow** 552 nm / 576–648 nm | V. dispar/parvula（単独） |
| **Red** 638 nm / 648–777 nm | P. gingivalis ＋ **F. nucleatum** |

この値は `.lif` のメタデータ（レーザ 405/488/552/638、検出窓）と完全一致して裏取り済み。

### 正しい解読（colocalization）

```
F. nucleatum  = Blue ∩ Red          （青と赤の両方に光るボクセル）
S. oralis     = Blue − (Blue ∩ Red)
P. gingivalis = Red  − (Blue ∩ Red)
A. naeslundii = Green
V. dispar/parv= Yellow
```

※ この colocalization は **xy 平均より前に、ボクセル単位**で計算する必要がある（`min` の平均 ≠ 平均の `min`）。

### 解読後の5菌種

![decoded 5 species](../figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__species.png)

F. nucleatum（紫）が S.oralis / P.gingivalis とは別の疎な分布として分離できている。

### 直していなければ起きていたバグ

旧 `lif_to_zprofiles.py` は「青＝S.oralis（純）/ 赤＝P.gingivalis（純）/ 紫＝F.nucleatum」を前提にしていた。実ファイルに紫チャンネルは無いので、そのまま走らせると：

- **F. nucleatum を丸ごと取りこぼす**
- **S.oralis と P.gingivalis を過大計上**（Fn の分が混入）

→ PDE 入力が**3菌種ぶん間違う**ところだった。`fish_decode.py`（両ツール共有の正典デコーダ）で修正済み。

---

## 3. z プロファイル抽出とレプリケート統合

各視野の3Dボクセルを「深さ z ごとの xy 平均強度」に潰し、**5菌種の深さプロファイル**にする。

- **複数 `.lif` を一括入力**し、`(condition, day)` でFOVを**プール（生プロファイルを1回だけ平均）**。
- **Day 1 は別実験日3本（220518/220601/220720）の biological replicate** → 15 視野としてまとめて平均。

### Heine のメール規約をツールに実装

Nils Heine からのメール（2026-05-26）の命名規約を自動適用：

- **HOBIC22 = commensal → CH**、**HOBIC24 = dysbiotic → DH**（ファイル名から自動判定）
- 日付は通常ファイル名末尾（`…Tag1`, `…Tag15`）。**HOBIC24 のみ1ファイルに複数日**が混在し、採取日は各 series 名の先頭整数 → 自動でグループ分け。

```bash
# 一発で全部処理（条件・日・レプリケート統合すべて自動）
python lif_to_zprofiles.py "HOBIC FISH"/*.lif
```

### 結果：commensal HOBIC の深さ時系列

![depth profiles](../results/diffusion_fit/zprofiles_CH_merged.png)

| condition | day | プールFOV | 深さ |
|---|---|---|---|
| CH | 1 | 15 | 43 µm |
| CH | 15 | 7 | 79 µm |
| CH | 21 | 4 | 69 µm |

→ `results/diffusion_fit/zprofiles_CH_merged.csv`（120行 = 3日 × 40深さ格子）。
バイオフィルムが時間とともに厚くなり、組成も深さ方向にシフトする様子が出ている。

---

## 4. この後：拡散フィット（`fit_diffusion_clsm.py`）

抽出した深さプロファイルを使って、**菌種ごとの空間的な動きやすさ**を推定する。

**モデル（反応拡散 PDE）** — 組成を決める2つの力：

1. **反応項 (A, b)** = 種間相互作用。**既知**（gLV / TMCMC でバルク時系列から推定済み）。
2. **拡散項 (D_i) ＋ 移流 (u)** = 空間的な動き。**これが未知＝フィット対象**。

```
D_i, u を仮定 → PDEを解いて深さプロファイルを予測 → 実測と比較(MSE)
            ↑______ ズレ最小の D_i, u を探索 (L-BFGS) ______|
```

6パラメータ **D = [D_So, D_An, D_V, D_Fn, D_Pg], u** を、実測の深さプロファイルに合わせて決める。
最適化の1ステップごとに PDE を時間積分するので計算は重い（JAX で高速化）。

**今回はパイプライン疎通のテスト**。本番は全データ（DH＝dysbiotic、欠けている Day 3/6/10）が揃ってから。

---

## 5. 成果物まとめ

| ファイル | 役割 |
|---|---|
| `fish_decode.py` | **新規**。4ch→5種の正典デコーダ（Fn=blue∩red）。両ツール共有 |
| `lif_quicklook.py` | **新規**。ヘッドレス可視化（readlif→PNG）。`--mode overview/species/montage` |
| `lif_to_zprofiles.py` | **修正**。colocalization 解読 ＋ 複数ファイル統合 ＋ メール規約自動適用 |
| `figures/lif_quicklook/*.png` | 生4ch・解読5種の確認画像 |
| `results/diffusion_fit/zprofiles_CH_merged.csv` | PDE フィット入力（CH 深さ時系列） |

**データの限界（要注意）**: 手元は全部 commensal（CH）のみ。dysbiotic（DH）未着。CH も Day 1/15/21 のみ（3/6/10 欠）。static（CS/DS）は本 FISH セットに含まれない。
