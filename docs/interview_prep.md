# NIFE Interview Prep / 面接対策 (2026-04-10)

## 0. 今日のゴール / Goal for the interview

- 目的：インターンテーマを「B7（biofilm/COMETS）×B2（NGS/biomarkers）×自分のBayes/UQ」で具体化して、12週間の成果物（deliverables）まで落とす
- Goal: align on a concrete 12-week scope and deliverables (PoC + dataset/pipeline) that match SIIRI priorities

## 1. 基本情報 / Logistics

- **日時**: 2026年4月10日 10:00 AM (対面)
- **場所**: Stadtfelddamm 34, 30625 Hannover (NIFE)
  - 到着後: 正面玄関から内線 1415 (Dr. Szafranski) に電話
  - アクセス: 4番線 MHH駅 → バス「Neue-Land-Straße」or「Stadtfelddamm」停
- **面接官**:
  - Dr. Szymon Szafranski (B7: Biotechnology, Microbiology)
  - Dr. Rumjhum Mukherjee (Zentrumsmanagement)
  - Dr. Amruta Arun Joshi (B2: Clinic, NGS, Biomarker)
- **希望期間**: 2026年5月〜6月開始、12週以上、最長2027年3月まで可

---

## 2. 30秒自己紹介 / 30-sec pitch

- 日本語（短く）
  - IKMで、口腔バイオフィルムの「動態モデル＋ベイズ推定（TMCMC）＋不確実性評価」をGPUで高速に回す研究をしています。NIFE/SIIRIでは、COMETSのような機構的モデルにBayesian calibrationとUQを足して、感染の早期検知につながる指標（例：DI）や、NGS/バイオマーカーとの統合まで持っていくことに貢献できます。
- English (short)
  - I work on GPU-accelerated Bayesian inference for multi-species oral biofilm dynamics (TMCMC + uncertainty quantification). For SIIRI, I can add Bayesian calibration and UQ on top of COMETS-style mechanistic models, and connect model outputs to clinically relevant infection indicators and NGS/biomarker readouts.
- - “I can align model outputs with measurable readouts such as NGS composition and biomarkers, and use them for calibration/validation to produce interpretable infection indicators and uncertainty-aware predictions.”
- NGSは **Next-Generation Sequencing（次世代シーケンシング）**の略で、要は

- **サンプルのDNAやRNAの配列（文字列）を、大量に一気に読む技術**

です。口腔バイオフィルムだと「どの菌がどれくらいいるか」や「どの遺伝子が動いているか」を調べるのに使います。

**NGSで何がわかる？（よく使う3種類）**
- **16S rRNA sequencing（16S解析）**：  
  - 何の菌がいるか、だいたいの**構成比（相対存在量）**  
  - 例：Streptococcus 30%、Porphyromonas 5% みたいな
- **shotgun metagenomics（メタゲノム）**：  
  - 菌種だけじゃなく、持っている遺伝子から**機能（代謝能力など）**も推定
- **metatranscriptomics（メタトランスクリプトーム）**：  
  - RNAを見るので、実際に動いている遺伝子＝**発現（活動）**がわかる

**面接での超短い言い方（英語）**
- “NGS is next-generation sequencing. In our context, it gives community composition (who is there) and sometimes functional activity (what they are doing).”

## 3. SIIRI プロジェクト / SIIRI overview

**Sicherheitsintegrierte und infektionsreaktive Implantate**
(Safety-Integrated and Infection-Reactive Implants)

- **資金**: DFG 1000万ユーロ以上 (3.5年)
- **機関**: MHH, LUH, Helmholtz, TU Braunschweig, HMTMH (150名以上)
- **目標**: 航空安全工学の手法を医療インプラントに適用し、感染の早期検知・反応型インプラントを開発
- **対象**: 歯科インプラント、整形外科インプラント、補聴器など
- **関連サブプロジェクト**:
  - **B7** (Szafranski): Biotechnology, Microbiology — バイオフィルム形成・dysbiosis
  - **B2** (Joshi): NGS, バイオマーカー — 感染の遺伝子レベル診断

---

## 4. COMETS とは / What is COMETS?

**Computation Of Microbial Ecosystems in Time and Space**
— Daniel Segrè (Boston University) 開発

### コア技術
- **dFBA (dynamic Flux Balance Analysis)**: ゲノムスケール代謝モデル (GEM) を時間発展させる制約ベース手法
- **空間構造**: グリッド上でバイオマス・代謝物の拡散・消費を計算
- **cometspy**: Python ラッパー (COBRApy と統合)

```
GEM (genome-scale metabolic model)
    ↓ FBA (線形計画法) → 増殖速度, 代謝フラックス
    ↓ × 空間グリッド (拡散方程式)
空間的バイオマス分布 + 代謝物濃度場 (時系列)
```

### 面接での説明（English）
- COMETS is a dynamic flux balance analysis framework for simulating microbial communities in time and space.
- It enables mechanistic interpretation via metabolism (exchange fluxes / cross-feeding), which is complementary to phenomenological ODE models.

### Daniel Segrè 研究のキーワード
- Quantitative principles of microbial metabolism across scales (Nature Microbiology 2024)
- Metabolic complexity driving community divergence (Nature Ecology & Evolution 2024)
- 生物ネットワークの力学・進化、微生物生態系の理論モデル

---

## 5. Szafranski 研究の要点 / What I understand from B7

- インプラント周囲バイオフィルム (transmucosal abutment) の 12患者コホート研究
- **~371 taxa / species-level** の微生物同定（So, An, Aa, Fn, Pg などを含む）
- 口腔粘膜 + インプラント材料 + バイオフィルムの in vitro organotypic モデル構築
- commensal (S. oralis) vs. pathogenic (A. actinomycetemcomitans) の組織応答比較
- salivary flow 条件下での dysbiosis 組成変化モデリング

---

## 6. 自分の研究との接続点 / Where I can contribute

| 自分の手法 | COMETS / SIIRI | 補完性 |
|---|---|---|
| Hamilton ODE (phenomenological) | FBA (mechanistic stoichiometry) | 機構的根拠を補強 |
| TMCMC (Bayesian UQ) | deterministic FBA (不確実性なし) | UQ を COMETS に付加可能 |
| DI → E(DI) → FEM (力学) | 代謝・増殖のみ | 力学連成は COMETS に未実装 |
| 5菌種 oral biofilm | 371菌種 口腔バイオフィルム | 小規模PoC→拡張の足場 |
| GPU TMCMC (200× 高速化) | 計算コスト高 | 大規模 GEM への適用可 |
| DI (Dysbiosis Index) | SIIRI 感染モニタリング | リアルタイム指標として提案可能 |

### 接続ストーリー（面接用の一言）
- 日本語：COMETSで「機構的に説明できるモデル」を作り、そこにBayesian calibration/UQを足して「どこまで信じて良いか」を数値で出し、最終的に感染検知に使える指標（DIなど）へ落とします。
- English: Build a mechanistic COMETS-style model, calibrate it with Bayesian inference, quantify uncertainty, and translate outputs into an interpretable infection indicator (e.g., DI) aligned with SIIRI sensing/monitoring.

---

## 6b. Heine（in vitro）と 371 taxa（患者NGS）をどう扱い分けるか / How I would use Heine vs 371-taxa data

### なぜ “Heineのように” そのままA行列推定が難しいか
- Heine (2025): in vitroで条件が制御され、同じ系を反復でき、時系列のreadout（pH, gingipain, morphology等）を設計しやすい
- 371 taxa: 多くは患者コホートのNGSで、相対存在量（compositional）が中心、交絡が大きく、時系列や摂動が少ないことが多い

### 応用のしかた（現実的な3パターン）
1. 状態/指標（DI）に使う：371 taxa から dysbiosis state を定義し、モデル出力（φ, DI）と整合・予測を評価する
2. Prior/制約に使う：共起・機能群・代謝相互作用予測を、Aの符号/疎性/グルーピングの事前情報として使う
3. 実験設計に使う：371 taxa から代表種/機能群を選び、Heine/HOBICのような制御系で小規模に落として推定する

### 面接での短い言い方（English）
- "Heine provides controlled, perturbation-friendly data suitable for interaction inference. The 371-taxa cohort is often compositional and cross-sectional, so I would use it to define dysbiosis states/priors and then validate or design a smaller mechanistic experiment for inference."

---

## 6c. Szafrański × IKM（Soleimani / Klempt）連携の代表例 / Key SIIRI-style hybrid papers

### 1) Co-aggregation（物理的な細胞結合）を実験＋FEMで扱う
- What: 複数菌が“くっついて塊を作る”現象（co-aggregation / co-adhesion）を、phase-field＋栄養輸送＋増殖で連成し、FEMで実装して実験と比較
- Why it matters: “相互作用”が代謝だけでなく物理（凝集/付着）でも決まることをモデル化できる
- How I can contribute: 既存の決定論モデルに対して、ベイズ校正（TMCMC）や不確実性評価を載せて、実験設計（どの計測が効くか）まで提案できる
- Reference: Soleimani et al., Scientific Reports (2023) 13:11839, DOI 10.1038/s41598-023-38806-2

### 2) Pg biofilm on titanium（DHNA刺激）を “hybrid in vitro–in silico” で扱う
- What: P. gingivalis のチタン上バイオフィルム成長が DHNA（menaquinone前駆体）でどう変わるかを、confocal等の実験データでFEMモデルを校正して再現
- Why it matters: “in vitroのreadout→モデル校正→予測” の流れが明確で、あなたのTMCMC/UQを最も入れやすいタイプ
- How I can contribute: 校正を確率的にして予測区間まで出す、表面粗さやDHNA濃度の感度・同定可能性を整理する
- Reference: Mukherjee et al., Microbiology Spectrum (2025), DOI 10.1128/spectrum.00410-25

---

## 6d. VEM（Virtual Element Method）応用アイデア / Where VEM could help

### ねらい（面接での一言）
- 日本語：複雑形状（インプラントねじ山・粗面・多孔構造）や、動く界面（バイオフィルム境界）を、メッシュ品質の制約を減らして安定に解くためにVEMを検討できます。
- English: VEM can reduce meshing pain for complex implant geometries and moving biofilm interfaces, while keeping PDE solvers stable on general polygon/polyhedron meshes.

### 具体アイデア（B7/B3/B4/B7に接続）
1. 画像ベース形状（CT/μCT/表面スキャン）→ポリゴン/ポリヘドラル要素で即ソルブ
   - ねじ山・粗面・微細孔を「四面体で無理に切る」より、セグメンテーションからのセル分割（Voronoi等）を活かせる
2. 反応拡散（酸素・pH・薬剤）＋成長（バイオマス）を、複雑境界で頑健に解く
   - B03のpH/O2センシングや、B05/B04の薬剤放出プロファイルを“形状込み”で予測しやすい
3. インプラント–粘膜–バイオフィルムのマルチドメイン連成（接触/界面条件）に強い
   - 異なる材料・層（implant/coating/tissue/biofilm）を、要素形状の自由度で扱いやすい
4. 成長に伴う“動く境界”や局所剥離（delamination）を扱う拡張
   - フィルムの局所剥離・欠損ニッチの形成など、感染再発に効く形状変化をモデル化する足場になる
5. UQ（形状不確実性）との相性
   - 粗さ・孔径・コーティング厚みなどの形状パラメータをランダム化し、DIや局所濃度への感度を評価する（B07の“in silico前処理”に直結）

---

## 7. 提案できる12週間プラン（例）/ Proposed 12-week scope (example)

### Option A: COMETS/dFBA + Bayesian calibration (core)
- Week 1–2: define target system (species set, medium, outputs, available measurements) + reproduce baseline simulation
- Week 3–6: implement a robust dFBA pipeline + data I/O (biomass, metabolites, DI) + parameter interface for calibration
- Week 7–10: Bayesian calibration + uncertainty (posterior, predictive intervals) on a small but real dataset (or a curated subset)
- Week 11–12: final report + reproducible code + a short presentation/demo

### Option B: Link to biomarkers (B2 integration)
- Define a mapping: model state/fluxes → expected biomarker trends (e.g., anaerobiosis markers, pathogen-related transcripts)
- Validate correlations on existing NGS / metatranscriptomics datasets if accessible

### Deliverables（成果物を明確化）
- Reproducible pipeline (Python): simulation → summary metrics (DI etc.) → calibration/UQ outputs
- A compact “demo dataset” + notebook/scripts for reruns
- A short technical report (what worked, limits, what data would improve identifiability)

---

## 8. 面接で使えるフレーズ (英語 + 日本語の意図)

1. "I can add Bayesian calibration and uncertainty quantification to COMETS-style dFBA models, beyond purely deterministic simulations."
   - 意図：推定＋UQで「信頼区間まで出す」価値を強調

2. "A model-based dysbiosis indicator could align well with SIIRI’s goal of early infection detection, especially if we can link it to measurable biomarkers."
   - 意図：DIを単なる指標でなく、測定系（B2）に繋ぐ

3. "I’m used to building reproducible pipelines: from raw data to calibrated models, diagnostics, and reports."
   - 意図：研究だけでなく運用（再現性）に強い

4. "My GPU-accelerated TMCMC pipeline achieves ~200× speedup, which helps when calibration becomes computationally expensive."
   - 意図：計算が重い領域への実行力

5. "I can start with a small 5-species proof-of-concept and then discuss how to scale towards larger communities."
   - 意図：371種にいきなり飛ばず、現実的ロードマップを提示

---

## 9. 取得済みデータ・論文 / Papers & datasets already saved

### 論文 PDF (nife/data/ に保存済み)

| ファイル | 論文 | データ |
|---|---|---|
| `dieckow2024_npj_biofilms.pdf` | Dieckow, Szafrański et al. (2024) *npj Biofilms Microbiomes* 10, 155 | **ENA: PRJEB71108** (30サンプル, metagenomics) |
| `dieckow2024_supplementary.pdf` | 上記 Supplementary | Table S4: 菌種間代謝相互作用 DB (351 taxa) |
| `joshi2025_npj_peri-implantitis.pdf` | Joshi, Szafrański et al. (2025) *npj Biofilms Microbiomes* 11, 175 | **NCBI SRA: PRJNA1192962** (48サンプル, 16S + metatranscriptomics) |
| `frings2025_an_atrftir_strainlevel.pdf` | Frings, Mukherjee, Szafrański et al. (2025) *Analyst* | strain-level ID / ATR-FTIR |

### Joshi 2025 の詳細
- **対象**: 32患者 48バイオフィルム (健常 vs peri-implantitis)
- **方法**: 全長 16S rRNA + メタトランスクリプトーム (RNAseq)
- **結果**: AUC = 0.85 / 健常: *Streptococcus*, *Rothia* ↑ / 病態: 嫌気性グラム陰性菌 ↑
- **データ**: NCBI SRA PRJNA1192962 (公開済み)

### Joshi 2026 の詳細 (Journal of Dental Research)
- **論文**: "The Submucosal Microbiome Correlates with Peri-implantitis Severity"
  doi:10.1177/00220345251352809 / PMID: 40719760
- **対象**: 34患者 49インプラント
- **結果**: Pseudoramibacter ↑ → 重症化 / 中心炭素代謝経路が重症度と相関
- **データ公開**: 未確認 (JDR は有料誌、PMC12861548 で確認要)

---

## 10. 事前に読んでおく論文 / Pre-read list

- [ ] Segrè lab: COMETS 原著論文 (eLife 2021, doi:10.7554/eLife.63372)
- [x] Dieckow, Szafrański et al. (2024) doi:10.1038/s41522-024-00624-3 **→ PDF取得済み**
- [x] Joshi, Szafrański et al. (2025) doi:10.1038/s41522-025-00807-6 **→ PDF取得済み**
- [ ] Joshi, Szafrański et al. (2026) doi:10.1177/00220345251352809 (JDR, 有料)
- [ ] Szafrański/B7: mucosa–biofilm / implant–mucosa interface モデル（organotypic model の代表論文を特定して読む）

---

## 10b. Szafrańskiさん論文（Google Scholar）から “相性が良い”候補 / Best-fit shortlist (with your toolbox)

Google Scholar profile: https://scholar.google.com/citations?user=COsZ0Q4AAAAJ&hl=en

### Tier 1（面接で名前を出すと強い）

- Heine et al. (2025) *Frontiers in Oral Health*: peri-implant biofilm dysbiosis in vitro（HOBIC flow vs static, 21 days）
  - Fit: 5菌種・条件比較・readout豊富 → A行列推定/DI/検証の話に直結
- Dieckow et al. (2024) *npj Biofilms Microbiomes*: dental implant上の early biofilms が subject-specific で dynamic
  - Fit: 371 taxa / metagenomics を “状態定義・prior・実験設計” に使うストーリーが作れる
- Joshi et al. (2025) *npj Biofilms Microbiomes*: peri-implantitis の診断バイオマーカー（microbiome + metatranscriptome）
  - Fit: “モデル出力 ↔ NGS/biomarkers” の統合（校正・検証）の話に直結
- Ingendoh‑Tsakmakidis et al. (2019) *Cellular Microbiology*: organotypic mucosa model で commensal vs pathogenic の差
  - Fit: “モデルの検証ターゲット”を菌量以外（宿主応答）に拡張する入口になる
- Soleimani et al. (2023) *Scientific Reports*: multi-species co-aggregation（実験＋FEM）
  - Fit: “物理（凝集/付着）＋栄養”を含む連成。あなたのTMCMC/UQを載せやすい
- Mukherjee et al. (2025) *Microbiology Spectrum*: Pg on titanium + DHNA（hybrid in vitro–in silico）
  - Fit: “実験データでモデル校正→予測”の王道で、あなたのBayes校正に最適

### Tier 2（話題として便利）

- Doll et al. (2019) *ACS Applied Materials & Interfaces*: liquid-infused structured titanium surfaces の anti-adhesive 機構
  - Fit: “材料表面→バイオフィルム”の因子をモデルにどう入れるかの会話ができる
- Doll‑Nikutta et al. (2025) *Journal of Dental Research*: oral biofilm–implant interface の gradual acidification
  - Fit: pHなどの独立検証ターゲット（Heineとも整合）を作りやすい
- Szafrański et al. (2021) *Periodontology 2000*: human oral phageome
  - Fit: 直接モデリングしなくても、将来的な“介入（phage）”の話題に繋がる

## 11. 質問候補 (自分から聞く) / Questions to ask

1. SIIRI の B7 サブプロジェクトで COMETS を具体的にどのように使っているか？
2. NGS データ (B2) と力学モデルを統合する計画はあるか？
3. Fachpraktikum 期間中に論文共著の可能性はあるか？
4. どのプログラミング環境を主に使っているか (Python, MATLAB, R)?

### 英語版（そのまま読める）
1. "How exactly are you using COMETS in B7 right now (species scope, spatial settings, media, outputs)?"
2. "What measurements do you consider as ground truth for calibration or validation (NGS, metabolites, microscopy, cytokines)?"
3. "What would be a successful 12-week outcome for you—prototype code, a dataset, or a publication draft?"
4. "Which compute environment and tooling do you prefer (Python stack, HPC, Docker/conda)?"

---

## 11b. Heine et al. (2025, Frontiers in Oral Health) に基づく質問セット（EN + 日本語訳）

### Setup / model relevance

1. EN: "In Heine et al. (2025), you compared a titanium flow-chamber biofilm model (HOBIC) versus static 6-well plates over 21 days. Which setup is most relevant for your current SIIRI subproject, and why?"
   - JP: 「Heine et al. (2025)では、チタン上のフローチャンバー（HOBIC）と6-well静置を21日で比較していました。今回のSIIRIではどちらのセットアップが一番重要で、理由は何ですか？」

2. EN: "What is the key practical advantage of HOBIC for your research—shear/flow realism, sampling, or compatibility with downstream measurements?"
   - JP: 「HOBICの実務上の一番の利点は何ですか？（流れの再現、サンプリング、下流測定との相性など）」

3. EN: "If we had to start with a 12-week proof-of-concept, would you prefer beginning with the static system first (higher throughput) or directly with the flow chamber (higher realism)?"
   - JP: 「12週PoCで始めるなら、まず静置系（回しやすい）から入るのが良いですか？それとも最初からフロー系（現実性高い）に行きますか？」

### Measurements / readouts

4. EN: "Which readouts are the most reliable in your hands: fluorescence microscopy/FISH, qPCR/16S, metabolite measurements, pH, or gingipain ELISA?"
   - JP: 「一番信頼できるreadoutはどれですか？（蛍光顕微鏡/FISH、qPCR/16S、代謝物、pH、gingipain ELISAなど）」

5. EN: "At which time points do you usually measure, and which time points are most informative for detecting dysbiosis onset?"
   - JP: 「測定タイムポイントは通常いつで、dysbiosisの立ち上がり検出に一番効くのはどの時点ですか？」

6. EN: "Do you have absolute biomass measures, or mostly relative abundance? If mostly relative, are there any anchors (total biomass, CFU, DNA mass) to scale to absolute values?"
   - JP: 「絶対量（biomass）がありますか？それとも相対存在量が中心ですか？相対中心なら、絶対スケールに直すためのアンカー（総バイオマス、CFU、DNA量など）はありますか？」

### Species / strains / generalization

7. EN: "Heine et al. included Streptococcus oralis, Actinomyces naeslundii, Veillonella dispar/parvula, Fusobacterium nucleatum, and Porphyromonas gingivalis. Which strains are you currently using (and why those strains)?"
   - JP: 「Heine et al.ではSo/An/Vei/Fn/Pgを使っていました。現在使っている株はどれで、なぜその株ですか？」

8. EN: "How stable are the results across runs—what is the dominant source of variability: inoculum, medium, oxygen, or flow settings?"
   - JP: 「結果の再現性はどの程度で、ばらつきの主因は何ですか？（初期菌量、培地、酸素、フロー条件など）」

### Modeling / integration (your contribution)

9. EN: "If I build a computational model around your system, what would you consider a convincing validation target: species trajectories, pH trend, gingipain time-course, or morphology metrics?"
   - JP: 「もし計算モデルを組むなら、どれが説得力のある検証ターゲットになりますか？（菌比率の時系列、pH、gingipainの時系列、形態指標など）」

10. EN: "Would it be useful to use a model to propose the next most informative experiment (e.g., which condition/time point to measure) to reduce uncertainty?"
    - JP: 「不確実性を減らすために、次に一番情報量の高い実験条件や測定時点をモデルから提案するのは有用ですか？」

---

## 12. 想定される質問と答え（短文）/ Likely questions & short answers

1. Q: なぜBayesian（TMCMC）？ / Why Bayesian inference?
   - A (JP): パラメータが多くデータが限られると「点推定」だけでは危険なので、事後分布で不確実性込みの予測を出すため。
   - A (EN): With limited data and many parameters, point estimates can be misleading; a posterior gives uncertainty-aware predictions and diagnostics.

2. Q: 371種にどう拡張する？ / How would you scale from 5 to hundreds of taxa?
   - A (JP): まず小規模でパイプラインと評価指標（DI等）を固め、次に「機能群・代表種・モジュール化」で段階的に拡張する。全種を一度に同定するより、識別可能性とデータの有無に合わせて設計する。
   - A (EN): Start with a small validated pipeline, then scale via functional grouping / representative taxa / modular design, guided by identifiability and available measurements.

3. Q: 何が“成功”か？ / What does success look like?
   - A (JP): 再現可能なコード、入力（データ）→出力（予測＋CI）まで一気通貫、そして次の実験やデータ取得の優先順位が明確になること。
   - A (EN): A reproducible end-to-end pipeline (data → calibrated model → prediction intervals) plus clear guidance on which data would most improve the model.

4. Q: あなたは何をすぐ始められる？ / What can you start immediately?
   - A (JP): Pythonでのデータ整形、dFBAの実装・可視化、推定用のパラメータAPI設計、簡単な校正（calibration）と検証。
   - A (EN): Data pipelines, dFBA implementation + visualization, parameter interfaces for calibration, and initial calibration/validation runs.

---

## 13. 当日持っていくもの / What to bring

- ノートPC（電源/充電器）
- 関連PDF（nife/data）と、このメモ
- できれば：小さなデモ（1枚スライド or notebook）  
  - English: one-slide demo of “simulation → DI → uncertainty band”

---

## 14. 面接の進め方（台本）/ Suggested interview flow

1. Opening (30 sec)
   - "Thanks for the opportunity. I’d like to quickly align on a concrete 12-week scope and deliverables."
2. Your pitch (1–2 min)
   - 研究背景 → SIIRIへの接続（Bayes/UQ + mechanistic models + biomarkers）
3. Their context (5–10 min)
   - B7でCOMETSをどう使っているか / B2で何が測れているか を質問して把握
4. Propose 2–3 options (5 min)
   - Option A（core）＋ Option B（biomarkers）＋ Option C（spatial）
5. Negotiate deliverables (5–10 min)
   - dataset access / compute / NDA / report style / weekly check-in を確定
6. Close (30 sec)
   - "If you agree, I can send a one-page project plan and start with a reproducible baseline within the first two weeks."

---

## 15. 面接官別：刺さる話題 / What to emphasize for each interviewer

### Dr. Szymon Szafrański (B7: Biofilm/Microbiology)
- JP: 実験系に合わせて「モデルの入力/出力」を揃える姿勢（測れる量に合わせてモデル設計）
- EN: "I want to align model outputs with what you can measure (biomass, metabolites, imaging, cytokines), and use the model to propose the next most informative experiment."

### Dr. Rumjhum Mukherjee (Zentrumsmanagement)
- JP: 12週間の成果物、再現性、共有できる成果（コード・レポート・手順書）
- EN: "I can deliver a clean, reproducible pipeline and documentation-style reporting that is easy to hand over to the consortium."

#### 何をしている人か（把握できている範囲）/ What she seems to work on (from public info)
- NIFEのCenter management（運営側）としてSIIRIのような大型枠組みを回す役割が強い  
  - 連絡先ページに Center management として記載あり
- 研究面では、implant-associated biofilms の “metabolite exploitation” をテーマに学会発表している  
  - DGBM 2024: “Metabolite exploitation in implant-associated biofilms in health and disease”
- 論文では、口腔バイオフィルム関連の菌同定（ATR-FTIR spectroscopy + chemometrics）の共同研究に入っている  
  - SIIRI（SFB/TRR-298）資金の論文で、NIFE/MHHも関与している

#### Mukherjeeさんに刺さる話し方（英語テンプレ）
- "I’d like to agree on a clear 12-week deliverable and a handover-ready pipeline that fits the consortium workflow."
- "I can keep the scope realistic and produce reproducible outputs (code, data splits, diagnostics) that others can reuse."

#### Mukherjeeさんに聞くと良い質問（運営視点）
- "What would be the most valuable deliverable for the SIIRI consortium in 12 weeks?"
- "Are there preferred formats for reporting and handover (repo structure, data policy, internal documentation)?"
- "What are the constraints on data access, IP/NDA, and publication?"

### Dr. Amruta Arun Joshi (B2: Clinic/NGS/Biomarkers)
- JP: モデルとNGSを「相関」だけで終わらせず、検証可能な仮説と予測にする
- EN: "Rather than only correlating taxa with disease, I’d like to connect mechanistic model states/fluxes to biomarker readouts to enable testable predictions."

---

## 16. プロジェクト案メニュー（面接で提示）/ Project menu (pick 1–2)

### Idea 1: Bayesian calibration for COMETS-style dFBA (core deliverable)
- What: dFBAパラメータ（uptake, μmax, yields, inhibition）をデータで校正し、予測区間を出す
- Why: deterministicな1本線のシミュレーションを「意思決定に使える」形へ
- Deliverables: posterior + predictive intervals, calibration diagnostics, reproducible pipeline
- Risk: identifiability（測定が少ないとパラメータが潰れない）→ 事前分布設計・パラメータ削減で対処

### Idea 2: Dysbiosis indicator + early-warning rule (DI as a monitoring signal)
- What: DI（entropy等）を時系列で定義し、"early shift" を検知するルール（threshold / change-point）
- Why: SIIRIの“early detection”に直結するアウトプット
- Deliverables: indicator definition + robustness check + simple validation on existing cohorts
- Risk: DIが単純すぎる → 複数指標（DI + anaerobe ratio + pathogen module score）を候補にする

### Idea 3: Link model outputs to metatranscriptomics (B2 integration)
- What: model state/flux → “expected pathway activity” に変換し、RNAseq pathwayと整合を見る
- Why: mechanistic modelとNGSを接続して、説明と予測を強くする
- Deliverables: mapping spec + analysis notebook + validation plots
- Risk: データアクセス / メタデータ不足 → 公開データでPoC→内部データへ

### Idea 4: Spatial minimal model (1D gradient first, then COMETS grid)
- What: まず1DでO2/糖/代謝物の勾配を最小モデル化して、空間の“必要性”を定量化
- Why: いきなり大規模gridに行く前に、何が効くかを見極める
- Deliverables: 1D baseline + sensitivity + recommendation（gridをやるべき条件）
- Risk: 空間モデルが重い → coarse grid / surrogate で回す

### Idea 5: Speed layer (surrogate/acceleration for calibration)
- What: expensive simulationを surrogate（簡単な近似）で置き、校正を回せる速度にする
- Why: 校正が現実的に回ると、研究が進む
- Deliverables: runtime benchmark + accuracy trade-off + recommended default settings
- Risk: surrogate bias → holdout validation and conservative uncertainty

---

## 17. “わかりやすい”1枚スライド案 / One-slide demo outline

- Title: "From mechanistic simulation to uncertainty-aware infection signals"
- Left: COMETS/dFBA (mechanistic) → outputs (biomass, metabolites, fluxes)
- Middle: Bayesian calibration (posterior) → predictive intervals
- Right: infection signal (DI / biomarker-aligned score) → early-warning concept
- Bottom: 12-week deliverables (pipeline + report + demo dataset)

---

## 18. 注意点（言い方）/ Pitfalls to avoid

- "I can directly apply 5-species results to 371 species" は言い切らない（ロードマップとして言う）
- 相手のデータ/環境が不明な状態で「全部できます」は避ける（まず測定と目的を聞く）
- 研究トピックを増やしすぎない（面接では “選べる2案” に絞る）

---

## 19. 締めの一言（英語テンプレ）/ Closing lines

- "If you agree, I can send a one-page project plan after the interview and start by reproducing a baseline simulation in the first two weeks."
- "My goal is to deliver a reusable pipeline and a clear recommendation on what data would most improve model reliability."

---

## 20. 面接後フォローアップメール（雛形）/ Follow-up email template

Subject: Thank you — proposed 12-week internship scope (SIIRI / B7–B2)

Hi Dr. Szafrański / Dr. Mukherjee / Dr. Joshi,

Thank you for the interview today. As discussed, I propose the following 12-week scope:
- Goal:
- Option selected (A/B):
- Deliverables:
- Data access needed:
- Weekly check-in preference:

If this sounds good, I can send a one-page project plan and start with reproducing a baseline simulation in the first two weeks.

Best regards,  
Keisuke Nishioka

---

## 21. 1ページ計画（面接中に埋める）/ One-page plan (fill during interview)

- Project title:
- Scientific goal (1–2 lines):
- Inputs (data / measurements):
- Outputs (plots / metrics / code):
- Milestones:
  - Week 1–2:
  - Week 3–6:
  - Week 7–10:
  - Week 11–12:
- Risks & mitigations:
- Success criteria (what you will show at the end):

---

## 22. 最初に確認するチェックリスト / First things to clarify

- Data: what is available now (formats, metadata, access rules, anonymization)?
- Ground truth: which measurements are “must-fit” vs “validation-only”?
- Compute: can I use internal machines / HPC / GPU? any preferred environment?
- Collaboration: weekly meeting cadence, preferred reporting format
- Publication: authorship expectations, what can/cannot be shared

---

## 23. 厳しめ質問10+と模範回答（英語）/ Tough questions & short model answers (EN)

### A. General (almost always asked)

1. Q: "What exactly will you deliver in 12 weeks?"
   - A: "A reproducible pipeline: baseline simulation → clearly defined outputs (biomass/metabolites/DI) → validation plots, plus a short technical report. If data access allows, I’ll add calibration and uncertainty bands."

2. Q: "What data will you use, and what is your ground truth?"
   - A: "I’ll align outputs with what you already measure: NGS/biomarkers, microscopy, metabolites, or clinical labels. I’ll separate ‘must-fit’ data from ‘validation-only’ data to avoid overfitting."

3. Q: "Why COMETS/dFBA instead of a simpler ODE model?"
   - A: "COMETS provides a mechanistic handle through metabolism (exchange fluxes, cross-feeding) which improves interpretability. I’m also comfortable using a minimal ODE as a fast baseline or surrogate when needed."

4. Q: "How do you avoid an over-ambitious plan (371 taxa, spatial grids, etc.)?"
   - A: "I start with a small, validated proof-of-concept and define a clear stopping rule. Scaling is a roadmap, not a promise: expand only after baseline reproducibility and a working evaluation metric are in place."

5. Q: "What is the main risk, and what is your fallback?"
   - A: "The main risk is identifiability with limited measurements. The fallback is parameter reduction, stronger priors, and focusing on robust summary outputs (DI/ratios) with conservative uncertainty."

### B. Dr. Szafrański (B7: Microbiology/Biofilm)

6. Q: "How will you make sure the model matches biology and experiments?"
   - A: "I will explicitly map model variables to measured quantities and define validation checks early. If a measurement is not directly modelled, I’ll use proxy outputs with a clear justification."

7. Q: "How do you choose species/medium and avoid unrealistic growth?"
   - A: "I’ll start from your standard experimental conditions and a small species set that you consider meaningful. If the medium is uncertain, I’ll test a small set of plausible media and report sensitivity instead of committing to one guess."

8. Q: "What do you consider ‘good enough’ validation?"
   - A: "A baseline that reproduces key qualitative trends and quantitative ranges, plus out-of-sample checks on at least one validation-only measurement. I prefer honest uncertainty bounds over overly tight fits."

### C. Dr. Joshi (B2: Clinic/NGS/Biomarkers)

9. Q: "How will you connect model outputs to NGS or biomarkers?"
   - A: "I’ll define a mapping from model states/fluxes to expected biomarker trends or pathway-level activity, then validate correlations on available datasets. The goal is testable predictions, not only retrospective association."

10. Q: "How do you handle patient heterogeneity and small sample sizes?"
   - A: "I’ll use hierarchical thinking where possible (patient-level variability) and report uncertainty explicitly. If sample size is limiting, I’ll focus on robust stratification and careful cross-validation rather than complex models."

### D. Dr. Mukherjee (Management/Delivery)

11. Q: "How will you ensure the work is reusable for the consortium?"
   - A: "I’ll deliver a clean repository with one-command reproducibility, clear inputs/outputs, and a short handover note. I’ll also document assumptions and limitations so others can safely extend the work."

12. Q: "What about NDA/confidentiality vs university reporting?"
   - A: "I will write the report only with approved content and can replace sensitive results with representative examples. My deliverables can be structured into ‘public’ and ‘internal’ parts to comply with policy."

---

## 24. 実験系と会話するための短文テンプレ（EN + 日本語訳）

### Heine et al. (2025, Frontiers in Oral Health) の文脈に合わせた聞き方

0. EN: "I read Heine et al. (2025) comparing a titanium flow-chamber biofilm model (HOBIC) versus static 6-well plates over 21 days. Could you tell me which setup is most relevant for your current SIIRI subproject?"
   - JP: 「Heine et al. (2025)で、チタン上のフローチャンバー（HOBIC）と6-wellの静置を21日で比較しているのを読みました。今回のSIIRIではどちらのセットアップが一番重要ですか？」

1. EN: "Could you briefly describe the experimental setup—species, surface material, medium, oxygen conditions, and time scale?"
   - JP: 「実験セットアップ（菌種、材料表面、培地、酸素条件、観察期間）を簡単に教えてください。」

2. EN: "Which readouts do you routinely measure: biomass (CFU/qPCR), imaging (confocal), metabolites, pH, or cytokines?"
   - JP: 「普段どの指標を測っていますか？（菌量：CFU/qPCR、画像：共焦点、代謝物、pH、サイトカインなど）」

3. EN: "What do you consider ground truth for validation, and what is validation-only?"
   - JP: 「検証のground truth（必ず合わせるデータ）はどれで、validation-only（後から確認用）はどれですか？」

4. EN: "Biofilm experiments can have variability—what is your current reproducibility bottleneck?"
   - JP: 「バイオフィルム実験はばらつきが出やすいですが、今の再現性のボトルネックは何ですか？」

5. EN: "If the data is mainly relative abundance from NGS, I can define model outputs that are comparable (fractions) and still keep mechanistic interpretation."
   - JP: 「NGSの相対存在量が中心なら、モデル側も比較可能な出力（割合）に落として、機構的解釈も保てます。」

6. EN: "Do you have time-series data, or mostly cross-sectional samples?"
   - JP: 「時系列データがありますか？それとも基本は一時点のサンプル（横断データ）ですか？」

7. EN: "In Heine et al. (2025), biofilm morphology and species distribution were analyzed by fluorescence microscopy and molecular biology methods. Which methods are you using as your standard readouts (e.g., fluorescence microscopy, qPCR/16S, ELISA such as gingipain), and at which time points?"
   - JP: 「Heine et al. (2025)では、蛍光顕微鏡＋分子生物学的手法で形態と菌分布を見ていました。こちらでは標準のreadout（例：蛍光顕微鏡、qPCR/16S、gingipain等のELISA）は何で、どのタイムポイントで測っていますか？」

8. EN: "If confidentiality is an issue, I can structure results into an internal part and a reportable part with representative examples."
   - JP: 「機密がある場合は、内部用と提出可能用に分け、代表例で置き換えてまとめられます。」

9. EN: "For a 12-week internship, I’d like to start with a small proof-of-concept that reproduces a baseline and then extend only if validation looks good."
   - JP: 「12週間なので、まず小さなPoCでベースライン再現→検証が良ければ拡張、という順で進めたいです。」

10. EN: "My goal is a reproducible pipeline that your team can reuse—clear inputs/outputs, scripts, and validation plots."
   - JP: 「チームが再利用できる再現可能パイプライン（入出力、スクリプト、検証プロット）を残すのが目標です。」

### 単語だけ（出てきたらビビらない用）

- EN: biofilm, medium, anaerobic, abutment, implant surface, confocal, CFU, qPCR, cytokines, validation, reproducibility, time series, cross-sectional, protocol, controls
- JP: バイオフィルム、培地、嫌気、アバットメント、インプラント表面、共焦点、CFU、qPCR、サイトカイン、検証、再現性、時系列、横断、プロトコル、対照条件

---

## 25. 生物実験 用語ミニ辞典（JP + EN）

### 培養・系（setup）

- biofilm / バイオフィルム  
  - JP: 菌が表面に付着し、EPS（粘性マトリクス）に包まれて形成する集合体。性質や耐性が変わる。  
  - EN: Surface-attached community embedded in an EPS matrix; behaves differently from planktonic cells.

- planktonic / 浮遊菌  
  - JP: 液中に漂う状態の菌（表面に付着していない）。  
  - EN: Free-floating cells in liquid (not attached).

- flow chamber / フローチャンバー  
  - JP: 液を流して、唾液流やせん断（shear）を模擬する培養装置。  
  - EN: Culture device with controlled flow to mimic salivary flow and shear forces.

- static culture / 静置培養（6-wellなど）  
  - JP: 流れなしで培養する。簡単で回しやすいが、現実の流れは再現しにくい。  
  - EN: No-flow culture; higher throughput but less physiologically realistic.

- shear stress / せん断応力（shear）  
  - JP: 流れによる“こすられる力”。バイオフィルム形態や付着に影響。  
  - EN: Flow-induced frictional force affecting attachment and morphology.

- medium / 培地（メディウム）  
  - JP: 菌が増えるための栄養液。成分が結果に強く影響する。  
  - EN: Growth medium; composition strongly affects outcomes.

- anaerobic / 嫌気（低酸素）  
  - JP: 酸素が少ない/無い条件。口腔の深部や成熟バイオフィルムで重要。  
  - EN: Low/no oxygen conditions relevant for mature oral biofilms.

- implant surface / インプラント表面（titaniumなど）  
  - JP: 付着する材料表面。粗さや表面処理で挙動が変わる。  
  - EN: Material surface (often titanium) affecting adhesion and growth.

- abutment / アバットメント（healing abutment）  
  - JP: 歯肉を貫通して口腔に露出する接続部品。バイオフィルムが乗りやすい。  
  - EN: Transmucosal connector component exposed to the oral cavity.

- inoculum / 接種（初期菌量・初期組成）  
  - JP: 培養開始時に入れる菌。再現性の主要因。  
  - EN: Initial inoculated cells; a major source of variability.

- strain / 菌株  
  - JP: 同じ菌種でも性質の違う“株”（病原性や成長特性が違うことがある）。  
  - EN: Specific isolate/line; phenotype can differ within a species.

### 測定（readouts）

- fluorescence microscopy / 蛍光顕微鏡  
  - JP: 蛍光で標識・染色した菌や構造を観察。  
  - EN: Imaging of fluorescently labeled cells/structures.

- confocal microscopy / 共焦点顕微鏡  
  - JP: 厚みのあるバイオフィルムを断層撮影（z-stack）して3Dに見られる。  
  - EN: Optical sectioning (z-stacks) to reconstruct 3D biofilms.

- FISH (fluorescence in situ hybridization) / FISH（蛍光in situハイブリ）  
  - JP: 種特異的プローブで菌を染め分け、空間分布も見られる。  
  - EN: Probe-based labeling to identify taxa and spatial distribution.

- qPCR / 定量PCR  
  - JP: 特定菌のDNA量を定量し、菌量の指標にする（相対/準絶対）。  
  - EN: Quantifies target DNA; used as an abundance proxy.

- 16S rRNA sequencing / 16S解析  
  - JP: 菌叢組成（相対存在量）を推定する代表的手法。  
  - EN: Profiles community composition (relative abundances).

- metagenomics / メタゲノム  
  - JP: DNAを丸ごと読む。菌種＋遺伝子機能（潜在能力）を見られる。  
  - EN: Shotgun community DNA sequencing; functional potential.

- metatranscriptomics / メタトランスクリプトーム  
  - JP: RNAを読む。実際に動いている機能（発現）を見られる。  
  - EN: Community RNA sequencing; active gene expression.

- metabolites / 代謝物  
  - JP: 代謝で作る/消費する小分子（酸など）。環境変化の指標。  
  - EN: Small molecules produced/consumed; indicate metabolic state.

- supernatant / 上清  
  - JP: 細胞を除いた液体部分。ELISAなどでタンパク質を測るときに使う。  
  - EN: Cell-free liquid used for downstream assays (e.g., ELISA).

- ELISA / ELISA（酵素免疫測定）  
  - JP: 抗体でタンパク質量を測る方法。  
  - EN: Antibody-based protein quantification assay.

- gingipain / ジンジパイン  
  - JP: P. gingivalis の主要なプロテアーゼ（病原性因子）で、病態の指標として測る。  
  - EN: P. gingivalis protease virulence factor often measured as a pathogenicity marker.

- pH / pH  
  - JP: 酸性化（pH低下）は代謝変化や病態と関係しやすい。  
  - EN: Acidification is linked to metabolic shifts and disease.

- CFU (colony-forming units) / CFU（コロニー形成単位）  
  - JP: 平板培養のコロニー数で、生きて増える菌の数を見積もる。  
  - EN: Viable cell estimate based on colony counts.

### 実験デザイン・品質（design / quality）

- time point / タイムポイント  
  - JP: Day 7/14/21など測定する時点。  
  - EN: Measurement time points (e.g., Day 7/14/21).

- replicates / 反復（生物学的・技術的）  
  - JP: 同じ条件を複数回やってばらつきを見る。  
  - EN: Repeated runs to estimate variability (biological/technical).

- controls / 対照  
  - JP: 比較の基準条件（菌なし、薬なし、既知条件など）。  
  - EN: Reference/baseline conditions for comparison.

- reproducibility / 再現性  
  - JP: 同条件で同様の結果が出るか。biofilmは難しいことが多い。  
  - EN: Consistency across runs; often challenging for biofilms.

- validation / 検証  
  - JP: モデルや仮説がデータに合うか確認。must-fitとvalidation-onlyを分けると強い。  
  - EN: Checking whether a model/hypothesis matches data; separating must-fit vs validation-only is useful.
