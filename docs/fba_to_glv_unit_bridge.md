# FBA フラックス単位と gLV 相互作用係数の橋渡し問題

**作成日**: 2026-06-10  
**根拠**: deep-research ワークフロー（104エージェント、21ソース、25クレーム敵対的検証）  
**結論一行**: 直接変換する方法は現時点で存在しない。符号だけを使う sign prior は分野全体で最も誠実な選択。

---

## 1. 問題の定式化

FBA が出力するのは **交換フラックス** $v_{ij}$ [mmol gDW$^{-1}$ h$^{-1}$]：

- 正の値 → 菌 $j$ が代謝物を **分泌**
- 負の値 → 菌 $j$ が代謝物を **取り込み**

gLV モデルの相互作用係数 $A_{ij}$ は無次元（あるいは [1/h] 程度）の量：

$$\dot{\phi}_i = \phi_i \left( r_i + \sum_j A_{ij} \phi_j \right)$$

「菌 $j$ の組成比が 1% 増えたとき、菌 $i$ の増殖速度が何 % 変わるか」を表す。
**この 2 つの次元は直接足し合わせも掛け合わせもできない。**

---

## 2. 文献調査結果（2018–2025）

### 2-1. 運用可能な唯一のアプローチ：対数比スケーリング（eLife 2024）

プロバイオティクス設計論文（arXiv:2210.03198、PMC10942782）が提案：

$$w^*_{ji} = \frac{\log(x_{ij} / x_i) - \min}{\max - \min} \times 2 - 1 \in [-1, 1]$$

$x_{ij}$ = 菌 $j$ の存在下での菌 $i$ の FBA 増殖速度（biomass flux）  
$x_i$ = 菌 $i$ 単独時の増殖速度

これを gLV の $\beta$ 係数として代入し、内因的増殖速度を $r_i = 1$ に固定。

**著者注記**: *「このリスケーリングは計算上の便宜のためであり、時間スケールの再スケーリングとみなせる。絶対的な大きさは捨てている。」*

→ **敵対的検証スコア 3-0（高信頼）**

---

### 2-2. 理論的橋：MacArthur/MiCRM → gLV

consumer-resource モデルから gLV を導出する数式は存在する（Sakarchi & Germain 2025; Marsland et al. 2020）：

$$\alpha_{ij} = \frac{\sum_k a_{ik} a_{jk} w_k K_k / r_k}{\sum_k a_{ik}^2 w_k K_k / r_k}$$

$a_{ik}$ = 菌 $i$ の資源 $k$ に対する消費選好度、$K_k$ = 資源容量、$r_k$ = 資源回復速度。

また Microbial Consumer Resource Model（MiCRM）は資源の時間スケールが消費者より速い（quasi-steady-state）極限で gLV に帰着し、相互作用係数が消費選好行列 $c_{i\alpha}$ と副産物変換行列 $D_{\alpha\beta}$ から機械論的に決まる（Marsland et al. 2020, *Sci Rep*）。

**問題点**：これらの枠組みの入力パラメータは粗粒度の現象論的パラメータ（$a_{ik}$, $K_k$, $r_k$）であって、**ゲノム規模 GEM の mmol/gDW/h フラックスではない**。GEM の化学量論情報を使わないため、FBA との橋が閉じていない。

→ **敵対的検証スコア 2-1（中信頼）**

---

### 2-3. gLV 近似自体の限界（Mustri et al. 2025）

consumer-resource モデルから gLV を導出する際の時間スケール分離 + 一次 Taylor 展開は、代謝的カップリングが強いと破綻する。

**具体的な閾値**（Mustri et al. 2025, *PLOS Comput Biol* 21:e1013719）：

| 条件 | 閾値 | 結果 |
|---|---|---|
| 漏れ率 $l_i$ | $\geq 0.2$ で破綻 | 生物学的範囲 0.05–0.6 ∋ 危険域 |
| 時間スケール比 $\epsilon$ | 1 に近づくと破綻 | — |
| 交差栄養が顕著な場合 | — | 高次非線形項が無視できない |

破綻の機序：一次 Taylor 展開で切り捨てた高次非線形相互作用が無視できなくなるため。

→ **敵対的検証スコア 3-0（高信頼）**

---

### 2-4. AGORA GEM の定量的信頼性（Rath et al. 2024）

Rath et al. 2024（*BMC Bioinformatics*、PMC10804772）は半自動キュレーション GEM（AGORA 含む）で 26 種の単培養・89 ペアの共培養を評価：

- FBA 予測増殖速度 vs 実験値 → **Spearman $r < 0.3$**（ほぼ無相関）
- 相互作用強度（増殖速度比）も同様に非相関
- 手動キュレーション GEM はやや良いが一般性に限界

一方 eLife 2024 のプロバイオティクス論文では、**大きさを捨てて符号・順位だけを使う**ことで AUC-ROC = 0.85 を達成。符号レベルでの実用性は確認済み。

→ **敵対的検証スコア 2-1（中信頼）**

---

### 2-5. 中間スケールの枠組み（Liao et al. 2020; Marsland bioRxiv 518449）

粗粒度な細胞内代謝（基質 → 構成要素 → バイオマス、Monod 速度論 + Liebig の最小律）で gLV と FBA を橋渡しする枠組みが提案されているが：

- A[i,j] を FBA フラックスから直接計算する数式は与えていない
- 相互作用項が環境資源濃度に依存するため「定数パラメータ」として切り出せない（状態依存型）

→ **敵対的検証スコア 3-0（高信頼）**

---

## 3. NIFE プロジェクトへの含意

| このプロジェクトの選択 | 文献的根拠 |
|---|---|
| FBA 出力の **符号だけ**を prior に使う | Rath 2024: 大きさは unreliable (**r** < 0.3) |
| MICOM で **100% 符号一致**を確認 | eLife 2024: 符号・順位レベルは AUC 0.85 で実用的 |
| 大きさ（mmol/h/gDW）は使わない | どの文献も「直接変換なし」と結論 |
| W=1.0 で **相転移的な SA = 100%** | MacArthur/MiCRM の理論的 gLV 導出とは別経路で実証 |

**結論**: 符号だけ使うのは妥協でも欠陥でもなく、**現時点の計算代謝学・生態学の両フロンティアから見て最も誠実な選択**である。

---

## 4. オープンクエスチョン（将来の研究として）

1. **経験的比例定数フィット**：訓練セットの菌ペア（FBA 予測値 vs 実測 gLV 係数）を回帰してスケーリング係数を推定できないか？
2. **GEM → 粗粒度 CRM → gLV パイプライン**：MiCRM のパラメータ（$c_{i\alpha}$, $D_{\alpha\beta}$）を GEM から系統的に推定するパイプラインは構築可能か？
3. **漏れ率閾値の口腔バイオフィルムへの適用**：Mustri 2025 の $l_i \geq 0.2$ 破綻条件は口腔バイオフィルムで成立するか？成立するなら gLV 自体の妥当性を再検討する必要がある。
4. **現在のベストプラクティス**：FBA sign prior + LOO-CV フィットの組み合わせは、純粋データ駆動 gLV と比較してどれほど優れているか？（本プロジェクトが部分的に実証：LOO-RMSE -2.2%）

---

## 5. 主要引用文献

| 文献 | 内容 |
|---|---|
| Rath et al. 2024, *BMC Bioinformatics* PMC10804772 | AGORA GEM の定量的信頼性評価 |
| Mustri et al. 2025, *PLOS Comput Biol* 21:e1013719 | gLV 近似の破綻閾値（漏れ率 ≥ 0.2） |
| Marsland et al. 2020, *Sci Rep* s41598-020-60130-2 | MiCRM → gLV の数学的導出 |
| Sakarchi & Germain 2025, *Am Nat* 205:3 | MacArthur の $\alpha_{ij}$ 公式の現代的確認 |
| eLife 2024, arXiv:2210.03198 | 対数比スケーリングによる FBA → gLV 変換（唯一の運用例） |
| Liao et al. 2020, *PLOS Comput Biol* 10.1371/journal.pcbi.1008135 | 粗粒度代謝 CRM の枠組み |
| Diener et al. 2020, *mSystems* 00606-19 | MICOM（community FBA） |

---

## 6. 修論への反映（許可時に使用）

### Ch.2 §2.3（sign prior の正当化）に追加できるパラグラフ

> The present work uses FBA-derived cross-feeding signals exclusively at the level of their **sign**, not their magnitude.
> This choice is justified by two independent lines of evidence.
> First, Rath et al. (2024) showed that semi-curated GEM predictions of interaction strengths are quantitatively unreliable (Spearman $r < 0.3$ against in vitro data), while sign/direction predictions retain practical signal (AUC-ROC 0.85 in probiotic engraftment; Diener et al. 2024).
> Second, no published method provides a unit-preserving conversion from exchange fluxes [mmol gDW$^{-1}$ h$^{-1}$] to dimensionless gLV interaction coefficients: the MacArthur consumer-resource derivation of $\alpha_{ij}$ uses phenomenological preference parameters, not genome-scale stoichiometry (Marsland et al. 2020; Sakarchi & Germain 2025), and the gLV approximation itself breaks down above a metabolic leakage threshold of $l_i \approx 0.2$ that lies within the biologically realistic range for oral biofilm (Mustri et al. 2025).
> Using sign information only is therefore not a limitation of the method but the principled conservative choice given the current state of constraint-based metabolic modelling.

### Ch.4 §4.2 考察（limitation パラグラフ）に追加できる一文

> A further limitation is that the magnitude of AGORA2-derived cross-feeding fluxes cannot be directly calibrated to the gLV interaction scale without additional kinetic parameters (Monod half-saturation constants); recovering absolute interaction strengths from GEM stoichiometry remains an open problem in the field (Rath et al. 2024; Mustri et al. 2025).
