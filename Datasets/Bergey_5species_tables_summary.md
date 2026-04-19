# Bergey's PDF テーブル要約（Heine 2025 使用菌種に限定）

対象PDF（`nife/Datasets`）:
- `Bergey'sManualofSySBac_Dig_Actinomyces.pdf`
- `Bergey'sManualofSySBac_Dig_Fusobacterium.pdf`
- `Bergey'sManualofSySBac_Dig_Porphyromonas.pdf`
- `Bergey'sManualofSySBac_Dig_Streptococcus.pdf`
- `Bergey'sManualofSySBac_Dig_Veillonella.pdf`

このメモは、Heine 2025（in vitro dysbiosisモデル）で言及されている菌種に絞って、各Bergey PDF内テーブル（`TABLE n.`）の「どこを見ればよいか」を整理した要約です。

## Heine 2025 で使用されている菌種（本文記載）
- Streptococcus oralis
- Actinomyces naeslundii
- Veillonella dispar または Veillonella parvula（条件によって置き換え）
- Fusobacterium nucleatum
- Porphyromonas gingivalis（実験条件によってラボ株/臨床株が使い分け）

## Actinomyces naeslundii
- 主なテーブル:
  - `TABLE 1` Differential characteristics of the human species of the genus Actinomyces
  - `TABLE 3` Physiological characteristics of Actinomyces species from human sources
  - `TABLE 5` Signature nucleotides of 16S rRNA gene sequences (phylogenetic subclusters)
  - `TABLE 6` Actinomyces と近縁属（Actinobaculum, Arcanobacterium, Mobiluncus 等）との鑑別
  - `TABLE 7` Morphological characteristics (human sources)
- 使いどころ:
  - 種レベル同定（A. naeslundii を含むヒト由来種の鑑別）: `TABLE 1/3/7`
  - 系統クラスタ制約: `TABLE 5`（16Sシグネチャ）
  - 属間鑑別ルール: `TABLE 6`
- モデル化メモ:
  - Actinomycesは種間差が大きいので、Heineモデルを種レベル（A. naeslundii）で扱う根拠として `TABLE 1/3/7` が使える。

## Fusobacterium nucleatum
- 主なテーブル:
  - `TABLE 1` Characteristics differentiating the species and subspecies of Fusobacterium
  - `TABLE 2` Characteristics differentiating Fusobacterium from genetically related taxa
- 使いどころ:
  - `F. nucleatum` の亜種鑑別（必要なら）: `TABLE 1`
  - 近縁属との誤同定回避: `TABLE 2`
- モデル化メモ:
  - `F. nucleatum` は亜種差が出やすいので、メタゲノム/16S側で亜種が分かれる場合は `TABLE 1` を参照して整合を取る。

## Porphyromonas gingivalis
- 主なテーブル:
  - `TABLE 1` Differentiation of the genus Porphyromonas from other anaerobic Gram-negative rods
  - `TABLE 2` Differentiation of the species of the genus Porphyromonas
- 使いどころ:
  - 属レベル鑑別（Bacteroides/Prevotella/Tannerella 等との分離）: `TABLE 1`
  - 種レベル（P. gingivalisを含む）の生化学・感受性差: `TABLE 2`
- モデル化メモ:
  - 同じPorphyromonas属でも性質が違うため、HeineのP. gingivalisを「属まとめ」せずに保持する根拠として `TABLE 2` が使える。

## Streptococcus oralis
- 主なテーブル:
  - `TABLE 1` Characteristics of Streptococcus species（大規模）
  - `TABLE 3` Characteristics of Streptococcus species: oral streptococcal species groups
- 使いどころ:
  - `S. oralis` が属する口腔内群（Mitis group 等）のグループ分解: `TABLE 3`
  - 種候補を一気に俯瞰する辞書: `TABLE 1`
- モデル化メモ:
  - Streptococcusは種数が多いので、`S. oralis` の位置付け（oral group）を `TABLE 3` で固定してからモデル化すると混乱が減る。

## Veillonella dispar / Veillonella parvula
- 主なテーブル:
  - `TABLE 1` Relative content (%) of cellular fatty acids of Veillonella species
- 使いどころ:
  - `V. dispar` と `V. parvula` を含む種間の脂肪酸プロファイル比較（分類・同定補助）
- モデル化メモ:
  - Heineでは Veillonella を `dispar/parvula` で置換しているため、同じ「乳酸利用」でも種差があり得る点の根拠として `TABLE 1` が使える。

## 実装向けの最短利用方針（Heine 5菌種に限定）
- 最初に見るべき表:
  - `S. oralis`：Streptococcus `TABLE 3`（oral group）と `TABLE 1`（一覧）
  - `A. naeslundii`：Actinomyces `TABLE 1/3/7`
  - `F. nucleatum`：Fusobacterium `TABLE 1`（亜種）と `TABLE 2`（属鑑別）
  - `P. gingivalis`：Porphyromonas `TABLE 2`
  - `V. dispar/parvula`：Veillonella `TABLE 1`

## 注意
- 本要約は「PDF上のテーブル見出し + 周辺記述」の整理です。機械学習/数理モデルへ直結する数値行列（CSV）化が必要な場合は、該当テーブルを個別に再抽出（OCR/手修正）するのが確実です。
