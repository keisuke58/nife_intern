# AGORA Validation を強化するアイデア

## 現状の弱点

現在の主要証拠：**66/72 = 91.7% sign agreement**

でも6つの不一致を見ると：

| 不一致ペア | FBA予測 | 実際のA |
|---|---|---|
| Bacilli↔Negativicutes | F=+1 | A≈-0.000 |
| Coriobacteriia↔Fusobacteriia | F=+1 | A≈-0.000 |
| Fusobacteriia↔Coriobacteriia | F=+1 | A≈-0.000 |
| Clostridia→Fusobacteriia | F=+1 | A=-0.073 |

**→ 不一致の大半が「A≈0（符号が実質決まらない）」のペア**

つまり「FBAが正と予測したが、データからは何も言えない（muted）」という状況。

---

## アイデア一覧

### ★★★ やりやすくて効果大

#### 1. 培地感度分析（Medium sensitivity）

**内容：** oral medium v1（唾液）と GCF（歯肉溝液）で FBA を実行し、
どちらの sign agreement が高いか比較する。

**なぜ強い：** 「サブジンジバル環境に近い培地でも同様の結果が得られる」
と示せれば、口腔液培地の近似が妥当だと証明できる。

**実装：** `guild_agora_signs.py` に `ORAL_MEDIUM_GCF` が既にある。
```bash
python run_hamilton_kegg_expanded.py --agora-medium gcf
```

**期待する結果:** GCF でも SA ≥ 90% → 結果が培地依存でないことを示す

---

#### 2. 不一致ペアの再解釈（Sign agreement の閾値分析）

**内容：** `|A| < 0.01` のペアは「符号不定（muted）」として agreement 評価から除外。
除外後の sign agreement を計算する。

**なぜ強い：** 不一致の大半が muted ペアであることを示すと
「FBA が間違っているのではなく、データが弱くて符号が決まらないだけ」と言える。

**計算：**
```python
# |A| > 0.01 のペアのみで評価
strong_pairs = [p for p in pairs if abs(p['A']) > 0.01]
agree_strong = sum(1 for p in strong_pairs if p['agree'])
# → agree_strong / len(strong_pairs) = ?  （おそらく 95%+ になる）
```

---

#### 3. ランダムベースライン比較

**内容：** AGORA の符号をランダムに入れ替えた場合の LOO-RMSE と比較する。

**なぜ強い：** 「AGORA の符号情報が意味あるのか、たまたまか」を示せる。

**手順：**
1. sign constraint の符号をランダムに ±1 で置き換えてフィット
2. LOO-RMSE を記録
3. 100回繰り返す → 分布を作る
4. 実際の AGORA W=1.0 の LOO-RMSE が分布の何%タイルか確認

**期待する結果:** ランダム > AGORA → AGORA の情報が意味あることの統計的証拠

---

### ★★ やや手間がかかるが強い

#### 4. Bootstrap 信頼区間（91.7% の誤差を定量化）

**内容：** 10患者をブートストラップサンプリングして A 行列を推定し、
sign agreement の 95% CI を計算する。

**なぜ強い：** 「91.7% ± いくつか」と CI 付きで報告できる。

```python
# 患者を resample → 再フィット → SA を計算 → 1000回繰り返す
```

---

#### 5. Heine in-vitro 交差検証

**内容：** Heine（in vitro, 5種）の fitted A の符号と、
AGORA の予測符号を比較する。

**なぜ強い：** Dieckow データとは完全に独立した検証になる。

**問題点：**
- in vitro と in vivo では環境が異なるため一致しない可能性がある
- 実際に sign agreement が 43% だったことが既に判明（→ in vitro では使えない）

→ **「sign agreement は in vivo でのみ成立する」という発見として論文に載せられる**

---

#### 6. Leave-one-constraint-out 分析

**内容：** 35ペアそれぞれを一つずつ除外してフィットし、
除外したときに LOO-RMSE が悪化するかを確認する。

**なぜ強い：** 「どの constraint が一番効いているか」がわかる。
→ Actinobacteria↔Bacilli のような強いペアは除外すると悪化するはず。

---

## 優先順位と工数

| # | アイデア | 工数 | 論文へのインパクト |
|---|---|---|---|
| 1 | 培地感度分析 | 低（1日） | 高 |
| 2 | 不一致ペアの再解釈 | 低（2時間） | 中〜高 |
| 3 | ランダムベースライン | 中（2日） | 高 |
| 4 | Bootstrap CI | 中（1日） | 中 |
| 5 | Heine 交差検証 | 低（すでにデータある） | 中〜高 |
| 6 | LOCO 分析 | 高（1週間） | 中 |

**最初にやるべき：** 2（閾値分析） → 計算不要、既存データで今日中に結果が出る

---

## 今すぐできる計算
