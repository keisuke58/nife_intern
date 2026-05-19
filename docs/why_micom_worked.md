# MICOM が Sign Prior として優れた理由

**作成日**: 2026-05-19  
**参考文献**: Diener et al. 2020, mSystems 5:e00606-19, DOI:10.1128/mSystems.00606-19  
**実装**: `guild_agora_signs.compute_micom_signals()`, `build_net_flow_expanded(agora_medium='micom')`

---

## 1. 結果サマリー

| 手法 | Sign Agreement | 制約ペア数 |
|------|:--------------:|:---------:|
| 文献 L1+L2 のみ | 45% (5/11) | 11/45 |
| 単種 pFBA v1 | 88% (29/33) | 33/45 |
| 単種 pFBA v2 | 81% (21/26) | 26/45 |
| **MICOM** | **100% (36/36)** | **36/45** |

MICOM は Hamilton モデルの fitted A 行列と **100% 一致**、制約ペア数も v1 より 3 ペア増加 (33→36/45)。

---

## 2. 失敗した手法とその理由

### MacArthur コサイン類似度

```
competition[i,j] = (c_i · c_j) / (|c_i| |c_j|)
```

- **失敗理由**: 口腔菌はすべて generalist。アミノ酸・糖を全員が使うため cosine ≈ 1 → 全ペアが競合と誤判定。
- MacArthur 1970 の前提「種ごとに異なるニッチ資源」が成立しない。

### Growth Rate Suppression (GRS)

```
competition[i,j] = (μ_i(full) - μ_i(depleted_by_j)) / μ_i(full)
```

- **失敗理由**: v2 medium（Dawes 2008 唾液相当）が極度に乏しく、菌は生存ギリギリで成長。j の uptake で何かが枯渇すると i の成長が完全に止まる (Δμ/μ ≈ 100%)。
- 81/90 ペアで >1% 競合 → 非特異的すぎて使えない。

---

## 3. MICOM が機能した理由

### 3a. Community context での flux 解決

単種 pFBA は「菌 j が代謝物 M を分泌できる」「菌 i が M を吸収できる」という **可能性**を見るだけ。  
MICOM は 10 菌を同時に入れた community FBA を解くため、**実際に M が j から i へ流れているか**を直接確認できる。

#### 具体例（実測）
```
Bacilli → Negativicutes  via  EX_lac_L(e)
  Bacilli secretion flux = +97.7 mmol/gDW/h
  Negativicutes uptake   = -97.9 mmol/gDW/h  ← community 内で実際に移動
```

これが有名な Streptococcus → Veillonella の乳酸 cross-feeding の FBA による直接実証。

### 3b. Cooperative Tradeoff による公平な資源分配

MICOM の cooperative tradeoff は：
```
maximize  min_i (μ_i / μ_i_max)    s.t. Σ μ_i ≥ τ × Σ μ_i_max
```
各菌が「最低でも自分の最大成長の τ × 100%」を達成できるよう資源を分配する。  
τ = 0.5（Diener 2020 デフォルト）により、generalist 同士でも特定の cross-feeding 経路が選択的に活性化される。

### 3c. Toxin signals が真に非ゼロ

`TOXINS = {'EX_h2o2(e)', 'EX_h2s(e)'}` の exchange が community FBA で活性化 → 37 新規競合ペア。  
単種 pFBA でも H₂O₂ 分泌は検出されるが、MICOM では community context で誰がそれを受けるかが明確。

### 3d. 過剰な非特異的競合シグナルが出ない

MacArthur や GRS は「全員が全員と競合」という結論になったが、  
MICOM は共有培地内での実際の flux を最適化するため、本当に cross-feeding が起きているペアのみが pos になり、genuine competition（代謝産物が重複している場合のみ）が neg になる。

---

## 4. なぜ v1 medium で動いたか

ORAL_MEDIUM v1（blood-plasma scale、糖・アミノ酸が v2 の 100 倍豊富）では：  
- 各菌が十分に成長 (μ = 120–1660/h)
- 乳酸・酢酸・プロピオン酸などの発酵産物を「余分に」分泌 → cross-feeding 量が大きい
- community FBA のフラックスに有意な信号が乗る

v2 では資源が乏しすぎて各菌が barely surviving → フラックスが微小 → cross-feeding 信号が弱い。  
**MICOM は v1 medium で使うのが現状最適**。

---

## 5. 理論的位置づけ (Diener et al. 2020 より)

MICOM は **Metagenome-Scale Community Metabolic Modeling** の実装：

1. 各微生物のゲノムスケール代謝モデル (GEM) を統合
2. `cooperative_tradeoff` アルゴリズム: トレードオフパラメータ τ ∈ [0,1] で個体適応度 vs 群集適応度のバランスを制御
3. 実腸内細菌叢データとの整合性 (Diener 2020, Fig. 4): 実験的に観測された cross-feeding fluxes と高い相関

口腔マイクロバイオームへの本論文の拡張適用は本研究が初（Diener 2020 は腸内細菌叢対象）。

---

## 6. "Perfect" 版の変更点 (2026-05-19)

初期実装（v0: fraction=0.3, binary count, toxin→consumer only）を以下の通り改善：

| 変更点 | v0 | perfect |
|--------|-----|---------|
| `fraction` τ | 0.3 | **0.5**（Diener 2020 推奨） |
| Cross-feeding 重み | バイナリ `+= 1` | **`+= min(src, \|tgt\|)`**（flux 量） |
| 毒素シグナル対象 | 代謝的 consumer のみ | **community 内の全 guild** |
| 戻り値型 | `int` | **`float`**（flux magnitude） |

乳酸 cross-feeding（97.7 mmol/gDW/h）とアミノ酸移動（0.5 mmol/gDW/h）が同等重みで扱われていた問題を解消。  
毒素は「消費」されるのではなく拡散して全菌に影響する → 生物学的に正確なモデル。

LOO-CV: jobs 40164–40173（`loo_micom_perf`）で評価中。

---

## 7. 残課題

- **LOO-RMSE 比較**: v0 (jobs 40153–40162) vs perfect (40164–40173) vs v1 baseline (mean=0.0562)
- **sign agreement 確認**: perfect 版の fraction=0.5 での比較 → job 40163 (`compare_signs`) 結果待ち
- **100% sign agreement の解釈**: fitted A は v1 prior で推定済み → MICOM が v1 を包含しているため高一致率が出る可能性。LOO-RMSE で実質的な改善を確認する必要あり。
- **v2 medium + MICOM**: 唾液相当の乏しい培地でも community FBA が meaningful な flux を出すか未検証

---

## 8. 参照文献

- Diener C, Gibbons SM, Resendis-Antonio O. **MICOM: Metagenome-Scale Modeling To Infer Metabolic Interactions in the Gut Microbiota.** mSystems. 2020;5(1):e00606-19. PMCID: PMC6977071.
- MacArthur R. **Species packing and competitive equilibrium for many species.** Theor Popul Biol. 1970;1:1–11.
- Marsland R et al. **The Community Simulator.** PLOS Comput Biol. 2019.
- Szafranski et al. **Elucidating diverse metabolic activities of the human microbiota.** (Supplementary File 1, 2026-04-16 version)
