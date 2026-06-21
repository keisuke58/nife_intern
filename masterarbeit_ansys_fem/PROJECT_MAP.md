# プロジェクト地図 — ペリインプラント炎 FEM × 疾患ダイナミクス

> 自分が何を持っているかを取り戻すための1枚もの。新しい結果はここには無い。既存の図・スクリプトの「何を・なぜ・どうつながるか」と、本命/支持/可視化の区別だけ。
> タグ: **★=核心結果**, ○=支持・精緻化, ▢=可視化のみ, ◇=探索（修論本文外・Outlook/将来の種）

---

## 0. 30秒版 — 幹は2本だけ

このプロジェクトは結局2つの問いに答えているだけ。残り全部はこの2本のバリエーション。

1. **力学（FEM）**：「噛む力はどうやってインプラント周りの骨を傷めるか」
   → 核心: **クラウンの高さ（モーメントアーム）と偏心咬合が辺縁骨の応力を約1.5×上げる**。設計（径）は二次的、バイオフィルムが主因。
2. **生物（ODE）**：「細菌叢はどう時間をかけて骨を溶かすか」
   → 核心: **dysbiosis → 炎症 → RANKL/OPG スイッチ → 破骨細胞が骨吸収**。臨床値で較正済み。
3. **橋**：骨が減る → 応力↑（`A(L)`）→ RANKL↑ → さらに骨が減る = **悪循環（tipping point）**。

**まず1枚で全部見たいなら → [fem_graphical_abstract.pdf](figures/fem_graphical_abstract.pdf)**（5段階を矢印でつないだ要約図）。

---

## 1. 一番見るべき図（これだけで人に説明できる3枚）

| 図 | 何を示すか | 種別 |
|----|-----------|------|
| [fem_concept_overview.pdf](figures/fem_concept_overview.pdf) | **全体の論理構造（模式）**：2本柱(生物ODE/力学FEM)＋橋A(L)＋悪循環ループ。box&arrow、実データなし。人に幹を説明する1枚 | ★ |
| [fem_graphical_abstract.pdf](figures/fem_graphical_abstract.pdf) | ①生態→②biofilm解剖→③炎症→④力学悪循環→⑤患者別予測 の一気通貫（実データ5パネル） | ★ |
| [fem_implant_crown_fem.pdf](figures/fem_implant_crown_fem.pdf) | (A)実装解剖（FEM歯肉・エナメル）/ (B)咬合 von Mises、モーメントアームで辺縁骨 17→27 MPa | ★ |
| [fem_periimplantitis_rankl_opg.pdf](figures/fem_periimplantitis_rankl_opg.pdf) | RANKL/OPG 機構モデル：カスケード・tipping・患者別biomarker・治療 | ★ |

---

## 2. 柱1 — 力学（FEM インプラント）

**問い**: 噛む力 → どこの骨に応力が集中し、何がそれを悪化させるか。
**道具**: Abaqus（C3D4/C3D10 など）、ISO 14801（100N・30°斜め荷重）。

### 2a. モデルの積み上げ（単純→リアル）
| 図 | 中身 | 種別 |
|----|------|------|
| [fem_tier2_bone.pdf](figures/fem_tier2_bone.pdf) | パラメトリック骨ブロック＋インプラント＋隣在歯（理想形状） | ○ |
| [fem_tier2b_real.pdf](figures/fem_tier2b_real.pdf) | **実下顎STL＋実歯＋根型インプラント**の連成（方法論の到達点） | ★ |
| [fem_tier2b_generic.pdf](figures/fem_tier2b_generic.pdf) | 標準スクリュー型インプラント版（皮質/海綿2層・斜め荷重） | ○ |
| [fem_transmucosal.pdf](figures/fem_transmucosal.pdf) | 歯肉カフ・歯肉溝・Ti表面biofilm の解剖（軸対称） | ○ |

### 2b. 「何が応力を駆動するか」スタディ
| 図 | 結論 | 種別 |
|----|------|------|
| [fem_implant_crown_fem.pdf](figures/fem_implant_crown_fem.pdf) | **クラウン高＝モーメントアームで辺縁骨応力 ×1.5** | ★ |
| [fem_implant_bare_vs_restored.pdf](figures/fem_implant_bare_vs_restored.pdf) | クラウン有無の並置（同じ結論を一目で） | ○ |
| [fem_crown_design.pdf](figures/fem_crown_design.pdf) | クラウン**材料**は二次的（21倍剛性差で骨応力<2%）、高さが支配 | ★ |
| [fem_crown_eccentric.pdf](figures/fem_crown_eccentric.pdf) | **偏心咬合**で荷重側辺縁骨 ×1.37 → saucer状欠損は荷重側で開始 | ○ |
| [fem_crown_ci_cycle.pdf](figures/fem_crown_ci_cycle.pdf) | **C/I比**（クラウン/インプラント比）が骨吸収で悪化、PONR 38%早まる | ★ |
| [fem_implant_design.pdf](figures/fem_implant_design.pdf) | **径が支配**（細いほど高応力）、長さ・ピッチは二次的（ISO14801クーポン） | ★ |

### 2c. 力学の深掘り（手法バリエーション、本文orOutlook）
| 図 | 中身 | 種別 |
|----|------|------|
| [fem_implant_thread.pdf](figures/fem_implant_thread.pdf) | 実装ねじFEM：dysbiosis応力比≈3.3×、ねじが応力生成 | ○ |
| [fem_implant_axi.pdf](figures/fem_implant_axi.pdf) | 軸対称：円筒のフープ拘束 ∝1/R0（平面ひずみが捨てる3D効果） | ○ |
| [fem_implant_3d.pdf](figures/fem_implant_3d.pdf) | 真の3Dヘリカルねじ＋咬合疲労・流れせん断 | ○ |
| [fem_implant_extensions.pdf](figures/fem_implant_extensions.pdf) | microthread応力集中・剥離phase-field・粘弾性緩和・ポロ弾性 | ◇ |
| [fem_clinical_schematic.pdf](figures/fem_clinical_schematic.pdf) | 臨床向け模式（bonded vs delaminates、Pg深さ、応力帯） | ○ |
| [fem_implant_tooth_scene.pdf](figures/fem_implant_tooth_scene.pdf) | インプラント×天然歯の1解剖シーン（periimplantitis vs periodontitis） | ◇ |

### 2d. 可視化のみ（モデルの絵。新しい主張は無い）
fem_implant_mesh3d / voxel3d / screw3d / crown3d / crown_section、`figures/fem_implant_blender.png`（写実レンダー） … すべて▢。

---

## 3. 柱2 — 生物（疾患 ODE）

**問い**: 細菌叢の悪化 → 時間とともにどれだけ骨が減るか、誰が高リスクか。
**道具**: 常微分方程式（純Python、FEM再solveなし）。実16Sコホート（Duran-Pinedo/Dieckow）の GDI で駆動。

### 3a. 重症度の駆動（実データ）
| 図 | 中身 | 種別 |
|----|------|------|
| [fem_periimplantitis_duranpinedo.pdf](figures/fem_periimplantitis_duranpinedo.pdf) | 実歯周炎コホート（15患者×7時点）の GDI(t)→患者別軌道 | ○ |
| [fem_periimplantitis_invivo.pdf](figures/fem_periimplantitis_invivo.pdf) | Dieckow 健常コホートの GDI（相対層別） | ◇ |
| [fem_periimplantitis_timeseries.pdf](figures/fem_periimplantitis_timeseries.pdf) | in-vitro 5菌種版（φ_Pg driver、副） | ◇ |

### 3b. カスケード（生態→骨吸収）
| 図 | 中身 | 種別 |
|----|------|------|
| [fem_periimplantitis_progression.pdf](figures/fem_periimplantitis_progression.pdf) | 進行性骨吸収→応力フィードバック（2→8mmで14→35MPa、PONR） | ★ |
| [fem_periimplantitis_inflammation.pdf](figures/fem_periimplantitis_inflammation.pdf) | 炎症層（簡略 B→I→C→L、Hillスイッチ）※rankl_opgの前身 | ○ |
| **[fem_periimplantitis_rankl_opg.pdf](figures/fem_periimplantitis_rankl_opg.pdf)** | **機構RANKL/OPG（破骨/骨芽/TGF/炎症アンカップリング）＝現行の本命** | ★ |
| [fem_periimplantitis_rankl_opg_calib.pdf](figures/fem_periimplantitis_rankl_opg_calib.pdf) | 上を臨床PICF/GCF実測で較正（fold-change・相関・可逆性） | ★ |
| [fem_periimplantitis_remodel.pdf](figures/fem_periimplantitis_remodel.pdf) | mechanostat（SED集中→saucerization予測） | ○ |
| [fem_periimplantitis_micromotion.pdf](figures/fem_periimplantitis_micromotion.pdf) | 脱統合でも ~7–8µm ≪ 150µm閾値＝早中期は力学不安定でなく膜駆動 | ○ |

### 3c. 精緻化・介入（同じ幹の枝）
| 図 | 中身 | 種別 |
|----|------|------|
| [fem_periimplantitis_bistability.pdf](figures/fem_periimplantitis_bistability.pdf) | **双安定/ヒステリシス**：なぜ衛生だけで治らないか（修論本文に挿入済） | ★ |
| [fem_periimplantitis_reversibility.pdf](figures/fem_periimplantitis_reversibility.pdf) | 可逆性窓（介入時期×残存dysbiosis のヒートマップ） | ○ |
| [fem_periimplantitis_hostmod.pdf](figures/fem_periimplantitis_hostmod.pdf) | 宿主修飾（抗TNF/抗RANKL/biofilm制御の比較） | ○ |
| [fem_periimplantitis_brushing.pdf](figures/fem_periimplantitis_brushing.pdf) | 歯磨き/デブライドメント（縁下到達限界） | ◇ |
| [fem_periimplantitis_design_time.pdf](figures/fem_periimplantitis_design_time.pdf) | 設計は時間を稼ぐだけ（二次的） | ◇ |
| [fem_periimplantitis_uq.pdf](figures/fem_periimplantitis_uq.pdf) | 不確実性伝播＋感度（破骨動態が支配） | ○ |
| [fem_periimplantitis_calibration.pdf](figures/fem_periimplantitis_calibration.pdf) | 文献較正＆face validity（kL≈0.022/wk） | ○ |
| [fem_periimplantitis_clinical_calibration.pdf](figures/fem_periimplantitis_clinical_calibration.pdf) | PRJNA1215005治療コホートで severity↔PD/BOP 検証 | ★ |

---

## 4. データの流れ（誰が誰を食べさせているか）

```
実16S（Duran-Pinedo/Dieckow）  ──GDI──→  疾患ODE（柱2）
                                        │  loss(t)
                                        ▼
                            A(L)=FEMの crest応力比（柱1のpimp掃引）
                                        │  ← 橋（mechano-bio feedback）
                                        ▼
                            RANKL↑ → 破骨↑ → さらに loss ↑（悪循環）

FEM（柱1）: Abaqus solve は重い → 一度回して field.json / pimp_results.jsonl に保存。
ODE（柱2）: その保存値（A(L)）を読むだけ。だから疾患の図は全部 FEM再solve無し＝軽い。
```

---

## 5. 用語集（迷ったらここ）

- **GDI** (**Guild** Dysbiosis Index): dysbiotic guild/commensal guild の対数比 = log(Bact+Clos+Fuso)−log(Act+Bac+Neg)。16Sから計算する「悪化の指標」。疾患ODEの入力。形式は Gevers 2014 の Microbial Dysbiosis Index と同型（引用可）。振り分けの根拠＝Socransky 1998(red/orange complex)/Abusleme 2013/Griffen 2012/Pérez-Chaparro 2014。**注意：class階級は粗い。Negativicutes(=Veillonella)を commensal 側に置くのは係争点（健常寄りだが歯周炎で増える報告あり）、Bacteroidia は Prevotella(中間)を Porphyromonas/Tannerella と束ねる**→「種レベルの病原体量でなく粗い群集状態シフト」と記述すること。閾値 GDI₀ はコホート相対(40%タイル)＝絶対閾値 GDI=0 ではない点も明記。
- **dysbiosis**: 細菌叢のバランス崩壊（悪玉優勢）。健常=symbiosis の逆。
- **RANKL / OPG**: 破骨細胞を作る分子スイッチ。RANKL=アクセル、OPG=ブレーキ（おとり受容体）。骨吸収の**機構的ドライバーとしては確立**（osteoimmunology）。PICF（インプラント周浸出液）で測定可能だが、**バイオマーカーとしては議論あり**：Chaparro 2021 メタ解析では RANKL/OPG 比のプール解析で有意差なし → 「機構は確立、定量バイオマーカーは emerging」と控えめに記述すること。
- **破骨細胞 (osteoclast, OC)** / **骨芽細胞 (osteoblast, OB)**: 骨を壊す/作る細胞。OC>OB で正味骨吸収。
- **TGF-β カップリング**: 吸収された骨からTGF-βが出てOBを呼ぶ＝「吸収の後に形成」の正常な連動。これが**炎症で破綻（uncoupling）**すると正味骨減少。
- **モーメントアーム**: クラウンが高いほど、斜め荷重が首部を曲げるテコが長くなる→辺縁骨応力↑。
- **C/I比**: クラウン高/インプラント長。臨床のMBL（辺縁骨吸収）リスク因子。
- **A(L)**: 骨吸収量Lの関数としての crest応力増幅。柱1（FEM）が柱2（ODE）に渡す唯一の橋。
- **tipping point / 双安定**: 閾値を超えると後戻りしにくい（健常basinから疾患basinへロック）。「なぜ衛生だけで治らないか」の数理的説明。
- **mechanostat (Frost)**: 骨は力学刺激で吸収/形成を切り替える。過荷重→吸収、というSED（ひずみエネルギー密度）依存。
- **saucerization**: インプラント周の皿状の骨欠損（辺縁・頬側から始まる）。
- **BIC / micromotion**: 骨-インプラント接触率 / 微小動揺。許容閾値は **50–150µm**（Pilliar 1986; Szmukler-Moncler 1998）でこれを超えると線維性被包（失敗）。早中期は閾値下＝力学失敗でなく膜駆動、という結論の根拠。
- **p95**: 応力の95パーセンタイル。max は首部の幾何特異点なので、頑健な指標としてp95を使う。
- **ISO 14801**: インプラント疲労試験規格。100N・30°斜め荷重の根拠。
- **PONR** (point of no return): 加速領域に入って後戻りできない点。
- **PICF / GCF**: ペリインプラント / 歯肉 溝浸出液（RANKL/OPGを測る検体）。
- **TMCMC**: 5菌種ベイズ推論のサンプラ（柱2の別系統＝Heineモデル、`paper_data.py`）。

---

## 6. 正直なステータス（何が固い / 何が illustrative）

- **固い（実計算・検証あり）**: FEM連成solve（柱1の全モデル）、モーメントアーム/設計/偏心の応力数値、RANKL/OPG比↔骨吸収相関の臨床一致（r=0.99 vs 0.81）、severity↔PD/BOP の臨床検証。
- **半分固い（文献アンカー較正）**: 疾患ODEのrate定数。絶対mm/年でなく fold-change・方向性・相対ランキングが主張。
- **illustrative（模式・将来の種）**: ◇タグの図、3D解剖シーン、host修飾の絶対効果量。修論には1文 or Outlook で。
- **存在しないデータ（正直なギャップ）**: 縦断16S × 縦断RANKL/OPG の同一被験者ペア。これが埋まれば真のベイズ較正ができる。

---

## 7. 修論にどう載せるか（量より質）

- **本文に既に入っている**: 実装ねじFEM・RVE・schematic（§fem_delamination）、per-FOV heterogeneity（§fem_validation）、双安定（§fem_validation）、設計スタディ（Outlook）。
- **入れる候補（本命）**: crown_fem（モーメントアーム）、rankl_opg＋calib、graphical_abstract。
- **入れない**: ◇タグ・重複（粘弾性b4、13.4×の旧スキーマ等）。
- **ルール**: いい結果だけ本文へ、いじりすぎない、数値は実データ検証してから（[[feedback_thesis_inclusion_rules]] / [[feedback_thesis_freeze]]）。修論編集は許可制。

---

*この地図は理解の整理用。中身の正典は: 図=`figures/`、FEM条件=[FEM_CONDITIONS.md](FEM_CONDITIONS.md)、論文骨子=[PAPER_OUTLINE_periimplantitis.md](PAPER_OUTLINE_periimplantitis.md)、研究ログ=メモ `project_nife_fem_realism_track.md`。*
