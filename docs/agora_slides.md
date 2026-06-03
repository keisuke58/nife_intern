---
title: "AGORA を軸とした代謝—生態の統合"
subtitle: "ゲノムスケール代謝モデルから相互作用符号へ — 基礎から考察まで"
author: "西岡佳祐 — NIFE / SFB TRR-298"
date: "2026-06-03"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
---

## このスライドの狙い

口腔バイオフィルムの **生態モデル（誰が誰を助ける／抑える）** を、
**ゲノムから計算した代謝（AGORA）** で裏づける——その仕組みと妥当性を
基礎から順に、批判的に整理する。

- **AGORA とは何か** / FBA の基礎
- 代謝（分泌・取り込み）→ **相互作用行列 A の符号** への変換
- 単種 pFBA → **MICOM（群集 FBA）** への発展
- **どこまで検証されたか**（ここが本題）
- 機械論的シミュレーション（COMETS dFBA）との突き合わせ

\vspace{0.5em}
結論を先に：**「符号」は使える・「量」は使えない。検証されるのは協調(cross-feeding)方向のみ。**

---

## なぜ代謝モデルが要るのか — 問題設定

16S の存在量時系列から **gLV 相互作用行列 A**（10ギルド × 10）を推定したい。

- パラメータ 100 個 vs データは 10 患者 × 3 週 → **劣決定（underdetermined）**
- 素朴なフィットは多数の等価解を持ち、符号すら定まらない

\vspace{0.5em}
**アイデア:** 代謝の知識で **A の符号に事前分布（sign prior）** を置く。

```
guild j が出す代謝物を guild i が食べる   →  j は i を助ける  →  A[i,j] > 0
guild j が毒素(H2O2,H2S)を出し i が受ける  →  j は i を害する  →  A[i,j] < 0
```

→ 大きさ(magnitude)ではなく **符号のみ** を制約する。これが本研究の中心思想。

---

## AGORA2 とは

**AGORA2** = ゲノム情報から作った **菌ごとの代謝モデル(GEM)** の大規模データベース。

- Heinken et al. **2023, Nature Biotechnology** 41:1320–1331
- **7,302 株**のゲノムスケール代謝再構成。腸内中心の v1 から **口腔菌を含む多部位**へ拡張
- 各菌 = **SBML(XML)** ファイル：その菌が持つ代謝反応（数百〜数千本）を全列挙
- 本研究では 10 ギルドの **代表株**を 1 ファイルずつ収録（`data/homd_db/agora_gems/`）

\vspace{0.5em}
「この菌は何を食べて何を出すか」を、培養せずゲノムから予測できる。

---

## FBA（Flux Balance Analysis）の基礎

**料理人の比喩:**

| 要素 | 対応 |
|---|---|
| 冷蔵庫の食材（培地） | 唾液の栄養組成（Dawes 2008：糖・アミノ酸・ビタミン…） |
| 料理人（菌）の目標 | 最速で増殖する（成長率 mu を最大化） |
| FBA が解くこと | その目標を満たす**反応フラックス配分**を線形計画で求める |
| 得られる副産物 | 分泌物（例：Streptococcus → 乳酸） |

- **pFBA**（parsimonious FBA）= 最小フラックス解 → 現実寄り
- 定常状態・最適増殖を仮定。**「符号」は信頼でき「量」は当てにならない**（後述）

---

## 代謝 → 相互作用 A の符号への変換

各ギルド代表株を口腔液培地で pFBA → **分泌プロファイル**と**取り込み能**を取得。

```
j の分泌フラックス > +0.05         →  j は代謝物 X を分泌
i の取り込みフラックス < −0.05      →  i は代謝物 X を消費

j が出す X を i が食べる            →  cross-feeding 成立
   X が H2O2 / H2S（毒素）か？
     はい → neg[i,j] += w   （j が i を害する → A[i,j] < 0）
     いいえ → pos[i,j] += w  （j が i を助ける → A[i,j] > 0）

net_flow = pos − sum(neg)  →  sign(net_flow) = +1 / −1 / 0
```

10 ギルド × 9 相手 = 90 方向ペアを総当たり → 符号付きペアに集約。

---

## パイプライン全体像

![](results/fig2_agora_pipeline.png){ height=66% }

(A) 5 ステップ手順 (B) 層を足すと制約ペアが **10 → 22 → 58**（AGORA で +36）
(C) 出来上がる符号 prior 行列 sgn(F[i,j])。

---

## 3 層の事前分布（証拠の重ね合わせ）

| 層 | ソース | 重み | 根拠 |
|---|---|---|---|
| **L1** | Szafrański Suppl.（実験 + KEGG/HMDB 注釈） | 2.0 | 直接観測 |
| **L1** | Szafrański Suppl.（実験・注釈なし） | 1.5 | 直接観測 |
| **L2** | Szafrański Suppl.（計算予測） | 1.0 | 予測 |
| **L3** | **AGORA2 FBA cross-feeding** | 1.0 | ゲノムスケール代謝 |

- 入力は Szafrański の微生物–代謝物関係 351 行（PRODUCES / USES / IS\_INHIBITED\_BY …）
- 各代謝物で「生産者 × 消費者」の全ペアに重みを加算
- 例：**乳酸** → Bacilli(Strep) が生産、Negativicutes(Veillonella)/Actinobacteria が消費

---

## ギルド代表株と培地

**代表株（AGORA2）:** Actino=*A. naeslundii* / Bacilli=*S. gordonii* /
Negat.=*V. parvula* / Bacteroidia=*P. melaninogenica* / Fusob.=*F. nucleatum* /
beta-Prot.=*E. corrodens* …（10 ギルド）

**口腔液培地（Dawes 2008 ベース）:** 糖・20アミノ酸・B 群ビタミン・微量元素
(Fe/Mg/Ca/Zn/**Cu**)・細胞壁前駆体(meso-DAP)・キノン類・グルタチオン

\vspace{0.4em}
→ **全 10 ギルドで正の増殖率**を確認（mu = 0.11–1.66 /h、Fusobacteriia 最高）。
培地が貧弱だと mu≈0 になるため、培地設計自体が結果を左右する重要点。

---

## なぜ「符号」だけなのか — 量(prior)の失敗

代謝フラックスの**大きさ**を A の prior に使う試みは、ことごとく破綻した。

| 手法 | 失敗理由 |
|---|---|
| **MacArthur コサイン**（ニッチ重複） | 口腔菌は全員 generalist。糖・アミノ酸を皆が使い cos≈1 → **全ペア競合と誤判定** |
| **Growth Rate Suppression** | 貧栄養培地で菌は生存ギリギリ → j の取り込みで何か枯渇すると mu が 0 に → **非特異的に全ペア競合** |

\vspace{0.4em}
**教訓:** FBA フラックスの単位 ≠ 生態的相互作用の単位。
**magnitude prior は失敗し、sign constraint だけが機能する。**（意図的な発見）

---

## 単種 pFBA の検証（naive な一致率）

![](results/fig3_agora_sign_validation.png){ height=58% }

FBA 予測符号 vs データ推定 A の符号は **66/72 = 92%** 一致（層別でも 91–92%）。
\textcolor{red}{ただし後述：これは「素朴な」数字で過大評価——次々スライドで批判的に再検証する。}

---

## MICOM — 群集 FBA への発展

単種 pFBA は「j が出せる・i が食べられる」という**可能性**を見るだけ。
**MICOM**（Diener 2020, mSystems）は 10 菌を**同時に**解き、実際に流れているかを見る。

![](results/fig_agora_v1_v2_micom_comparison.png){ height=52% }

- **cooperative tradeoff**（tau=0.5）：各菌が「最大成長の tau 倍」を確保するよう資源を公平分配
- generalist 同士でも**特定の cross-feeding 経路のみ**が選択的に活性化

---

## MICOM の結果

| 手法 | Sign Agreement | 制約ペア数 |
|---|:---:|:---:|
| 文献 L1+L2 のみ | 45% (5/11) | 11/45 |
| 単種 pFBA v1 | 88% (29/33) | 33/45 |
| **MICOM（群集）** | **100% (36/36)** | **36/45** |

**直接実証された乳酸 cross-feeding（群集内の実フラックス）:**
```
Bacilli(Strep) → Negativicutes(Veillonella)   via EX_lac_L(e)
   分泌 +97.7   /   取り込み −97.9  mmol/gDW/h
```
有名な Streptococcus → Veillonella を **FBA が再構成**。
（注意：fitted A は v1 prior 由来 → 100% は包含による可能性。RMSE で実質改善を要確認。）

---

## 事前分布の重み W の感度

![](results/fig_agora_weight_sensitivity.png){ height=56% }

**W=1.0 で相転移** — sign agreement 100%、LOO-RMSE 最小(≈0.050)。
ただし prior-free gLV(0.0455)はさらに低く、**prior の価値は予測精度でなく
「解釈可能性・符号整合性」**にある——と誠実に位置づける。

---

## 批判的検証：本当に独立に裏づくのか

「92% 一致」は **prior が全て正**（α=0 では cross-feeding のみ）なので過大評価。
正しい対照は **prior 外セルの正割合**との比較（permutation test）。

| モデル | cross-feeding 方向 | 競争方向 |
|---|---|---|
| **Hamilton（対称）α=0** | **78.6%(11/14) vs ランダム 37.7%, p=0.0004, z=+3.79** | 検証されず（≈chance） |
| gLV（非対称） | 41%（null） | null |

\vspace{0.3em}
- **検証されるのは協調(cross-feeding)方向のみ**。競争方向は支持されない。
- **AGORA prior 自体は 16S 力学から独立**（データは prior を再現しない＝prior はモデリング選択）。
- **2 独立コホート**(Dieckow×Botelho)を prior 抜きで比べると、**強い相互作用の符号が 89% 一致(p≈0.02)** → 生態シグナルは本物。

---

## 機械論的クロスバリデーション（COMETS dFBA）

AGORA GEM は符号 prior 以外に、**5 種群集の動的 FBA(dFBA)** にも使える。

![](comets/pipeline_results/sweep_crossfeeding.png){ height=52% }

健常：So/An 優占・乳酸 cross-feeding → **DI=0.15**。
疾患：Pg/Fn 増殖 → **DI=0.70**。同じ AGORA 代謝が**前向きシミュレーション**でも
commensal↔dysbiotic の分岐を再現（独立路線での整合）。

---

## 限界と結論

**限界**
- ギルド = class レベル（代表株 ≠ ギルド全体）。種レベル MAG で改善余地
- IS\_INHIBITED\_BY の 20/22 行は酸素（生産者不在）で**実質失効**、毒素信号は H2O2 の 2 ペアのみ
- 重みは代謝物ごとの max → 予測行が高 confidence に引き上げられる
- magnitude は捨てている（sign のみ）

**結論（体系）**
1. AGORA → cross-feeding → **符号 prior** という変換が本研究の新規性
2. **符号は使え、量は使えない**（MacArthur 型失敗の回避）
3. 単種 pFBA(92%) → **MICOM(100%)** で群集文脈を捕捉
4. 誠実な検証：**協調方向のみ p=0.0004 で独立裏づけ、2 コホート 89%**。prior 自体はモデリング選択
5. COMETS dFBA が前向きにも commensal↔dysbiotic を再現
