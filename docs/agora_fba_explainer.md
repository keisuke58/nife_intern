# AGORA + FBA の仕組みと妥当性

## 1. AGORA とは

**AGORA2** = 7,000種以上の菌について「何を食べて何を出すか」をゲノム情報から作った代謝モデルのデータベース（Heinken et al. 2023, Nature Biotechnology）。

各菌の代謝モデルは **SBML（XML）ファイル** として提供される。

---

## 2. ファイル構造（このプロジェクト）

```
data/homd_db/agora_gems/
├── Actinobacteria_Actinomyces_naeslundii_...xml
├── Bacilli_Streptococcus_gordonii_...xml        ← Bacilli の代表種
├── Bacteroidia_Prevotella_melaninogenica_...xml
├── Fusobacteriia_Fusobacterium_nucleatum_...xml
└── ...（guild ごとに代表種 1 ファイル）
```

各 XML ファイルの中身：
- その菌が持っている代謝反応を全部リスト化したもの
- 「グルコースを取り込む反応」「乳酸を出す反応」「ATP を作る反応」… 数百〜数千本

---

## 3. FBA（Flux Balance Analysis）とは

**料理人の比喩：**

```
冷蔵庫（培地）の食材 ─── 唾液の栄養組成（Dawes 2008）
                              グルコース、アミノ酸、ビタミン…

料理人（菌）の目標 ─────── 最速で増殖する（成長率を最大化）

FBA がやること ─────────── 「この食材でこの目標を達成するには
                              どの代謝反応をどれだけ使うか」を
                              線形計画法で解く

副産物（排出物）がわかる ── Streptococcus → 乳酸を外に出す
                              Veillonella → 乳酸を取り込む
```

---

## 4. このプロジェクトでの使い方（3ステップ）

```
Step 1: 口腔液培地を設定
        ORAL_MEDIUM = {グルコース: 10, フルクトース: 5, アミノ酸類: ...}
        （Dawes 2008 の唾液組成に基づく）

Step 2: 各 guild の代表菌で pFBA を実行
        → 「この環境でこの菌は何を排出するか」がわかる

Step 3: guild ペアを比較
        A菌が「物質Xを排出」、B菌が「物質Xを取り込む」
        → cross-feeding 成立 → F_ij を正としてカウント
        → sign constraint: A_ij > 0 のはず
```

10 guild × 9 相手 = 90 方向ペアを全部調べる → 35 ペアに非ゼロの符号が付く。

---

## 5. 妥当性の問題点と対処

### 問題①：代表種が本当に「代表」か？

| Guild | FBA で使う代表種 | 実際に含まれる菌 |
|---|---|---|
| Bacilli | Streptococcus gordonii | Streptococcus + Gemella + Enterococcus + ... |
| Actinobacteria | Actinomyces naeslundii | Actinomyces + Rothia + ... |

**対処：** guild-level 粒度の限界として Discussion に明記。種レベルの MAG（患者固有ゲノム）を使えば改善可能（Future directions）。

### 問題②：FBA の仮定（最適増殖、定常状態）が現実と合うか？

- 「符号」は合うが「量（フラックス値）」は当てにならない
- だから magnitude prior（MacArthur）は失敗し、sign constraint だけが有効
- pFBA（parsimonious FBA）で最小フラックス解を使い現実寄りに近づけている

### 問題③：培地が唾液（saliva）で歯肉溝（subgingival）と異なる

- コードには `ORAL_MEDIUM_GCF`（歯肉溝液版）も実装済み
- メイン解析は v1（唾液）で 91.7% agreement → v1 で十分機能

---

## 6. 現在の validation（主要証拠）

**事後的 sign agreement: 66/72 = 91.7%**

FBA で予測した符号 vs. 患者データから推定した A 行列の符号が、
70 ペア中 66 ペアで一致（p < 10⁻¹⁰, 二項検定）。

これが「FBA の方向予測は使える」という経験的根拠。

---

## 7. Validation をさらに強くするアイデア

→ 別ドキュメント参照
