# 審査対策 — Junker（主査）/ Soleimani（副査）に刺す修論・口頭試問

> **Canonical = `nife/docs/EXAMINER_STRATEGY.md`**、`30_Masterarbeit/notes/examiner_strategy.md` はミラー（`/thesis-sync` が同期）。
> [[THESIS_WRITING_GUIDE]] の姉妹文書。出典＝examiner-strategy ワークフロー（Junker 高確度／Soleimani 中確度、実コードと IKM 群論文を照合）。最終更新 2026-06-04。

審査は二つのレンズ。**Junker（主査）＝変分・拡張Hamilton原理・熱力学的整合性**、**Soleimani（副査）＝数値計算・FEM/SPH・収束/保存則**。
この修論は彼らの自陣（IKM の連続体バイオフィルムモデル: Junker–Balzani 2021 / Klempt 2024,2025 / Fritsch 2025）に乗っている。**彼らの言葉で、彼らの基準で**書けば一気に評価が上がる。逆にここを外すと「生物の当てはめ」に見えて減点。

---

## §1 Junker（主査）— 変分・熱力学のレンズ

**彼が必ず突く所**：①その「Hamilton replicator」は本当に Hamilton 汎関数（自由エネルギー＋散逸ポテンシャル）から導けるのか、名前を借りただけか／②内部変数 ψ（生存率）は？散逸 Δ≥0 か／③A の対称性は熱力学的必然（Onsager 相反）か統計的都合か／④replicator のゲージ不変性（A の各列に定数を足しても不変）を prior・推定は尊重しているか／⑤演算子分割の誤差と well-posedness／同定性を Bayesian で。

### 想定質問と即答（用意しておく）
1. **「Hamilton replicator の元になる汎関数を書け。自由エネルギーと散逸ポテンシャルは？散逸≥0 を示せ」**
   → 自由エネルギー Ψ(φ)=−½φ·Aφ−b·φ（ペイオフ）＋二次の散逸 Δ(φ̇)。δH=0 から replicator が出る。**replicator の Lyapunov 関数＝平均適応度／KL ダイバージェンス**が単調 → 「自由エネルギー減少＝散逸≥0」で2nd law を満たす、と言える所まで示す。対称 A の範囲では厳密、非対称・空間項では「形式的」と正直に線引き。
2. **「Klempt 2024 の 0D 極限か？拡散を切り ψ≡1 にすれば彼らの φ 発展方程式に戻るか？」**
   → **戻ることを実際に示す**（D_i→0, ψ_i≡1 で Klempt の反応項に一致）。彼自身のモデルを reduce して見せるのが最強のシグナル。
3. **「なぜ Hamilton 形では A 対称、gLV では非対称？」**
   → **対称＝変分（Onsager 相反）構造に必要**。二次形式の自由エネルギーから A_ij=A_ji が自動的に出る（imposed でなく emergent）。非対称 gLV はより一般だが非変分。「対称は変分構造＋同定性の両方のための原理的選択」と言い切る。
4. **「replicator は A の列ゲージ不変。sign prior はそれを固定しているか？報告した非同定性の一部はこのゲージでは？」**
   → ゲージを明示的に固定（列平均ゼロ or A_ii≤0 正規化）、sign prior を**ゲージ整合**に適用していると述べる。聞かれる前に言うと刺さる。
5. **「D_i が下限 1e-5 に張り付き“illustrative”と書いている。同定性を定量化せよ（Fisher 情報／感度／事後共分散）。どの D_i がデータで拘束されているか？」**
   → **弱同定を“発見”として提示**（[[THESIS_WRITING_GUIDE]] の方針）。Fisher 情報か感度解析で同定可能/不可能を分離。Fritsch 2025（彼の群）が hybrid uncertainty 下の同定性を主題にした—その土俵に乗る。
6. **「なぜ空間逆問題を L-BFGS の点推定で？群の Bayesian updating（Fritsch 2025）でやらない理由は？」**
   → 「空間 PDE 逆問題を Bayesian/credible interval でやるのが正攻法。今回は計算予算で点推定だが、Fritsch の TSM サロゲート＋TMCMC を将来研究として接続する」と認め、補完関係に位置づける。

### 加点ムーブ
- **冒頭で生態モデルを彼の拡張Hamilton原理に明示接続**（汎関数→δH=0→replicator）。これで「生物論文」が「連続体・変分の仕事」に変わる。
- 対称 A で **Lyapunov/自由エネルギー単調性**を証明（2nd law の彼が期待する論法）。
- 弱同定を **Fisher/感度で定量化**して先に開示。
- ゲージ不変を**自分から**指摘し固定。
- sign-only prior を**単位整合の議論**として正当化（FBA flux 単位 ≠ 生態相互作用単位；熱力学FBA/ΔG拘束を magnitude への原理的経路として検討済みと言う）。
- **将来研究＝連続体統合**を約束：1つの Hamilton 汎関数に 自由エネルギー（相互作用＋栄養＋抗菌）＋内部変数（φ, ψ, 死細胞）＋散逸、AGORA 符号が A に入る—彼が最も見たい統合像。
- **必ず正しく引用**：Junker–Balzani 2021、Klempt 2024（BMMB）、Klempt/Geisler/Soleimani/Junker 2025（arXiv:2509.01274）、Fritsch 2025（arXiv:2512.15145）、COMM-PINN（Comput. Mech. 2023）。

---

## §2 Soleimani（副査）— 数値計算のレンズ

Wriggers 門下（Dr.-Ing. 2017）、**SPH でバイオフィルム**（Soleimani & Wriggers 2016）、応力誘起異方成長（JMPS 2020）。**grid 法がバイオフィルムで何を苦手とするか熟知**。数値の甘さを最も突く人。

**実コードで彼が見つける所**（agent がコード照合済み）：Lie 分割（1次）、CFL を 0.9×に**黙ってクリップ**、対角拡散＋clip(0,1)＋phi0 patch は**非保存・simplex非保存**（コメントも認めている）、cross-diffusion(xdiff) 版は保存的（face flux の有限体積）、ghost-node BC、反応は固定 n_nsp=5 Newton ステップ、jax_enable_x64。

### 想定質問と即答
1. **「演算子分割の誤差次数は？なぜ Strang でない？分割解を非分割の参照解と照合したか？」**
   → Lie=O(Δt)。Strang(O(Δt²))を明示的アップグレードとして提示。非分割参照との誤差を示す。
2. **「陽的輸送の安定条件は？拡散 CFL は守っているが**移流 CFL（u·Δt/Δz≤1）**は別途チェックしていない」**
   → 両 CFL を明記し、production の Δt が**両方に余裕を持って**満たすことを示す。自動クリップ任せにせず固定 sub-CFL Δt で報告し、Δt 半分でも不変と示す。
3. **「格子収束スタディを見せろ。Nz→∞, Δt→0 で収束するか？2次中心拡散＋1次風上＝全体1次、観測次数を測ったか？」**
   → **格子細分化（Nz=20/40/80→2D/3D）と Δt 細分化の log-log 誤差図＋観測次数**。これ1枚が最大の加点。
4. **「保存的か？対角拡散して phi0=1−Σφ_i、clip(0,1) する—ステップ毎の質量誤差を定量化せよ。なぜ cross-diffusion 版が良い？両者は一致するか？」**
   → 対角＋clip は**意図的で定量化された近似**、volume-filling cross-diffusion 版が保存・simplex 保存の正解と正直に。両スキームの一致範囲を示す（Painter–Hillen 引用済み）。
5. **「Hamilton 原理（Klempt/Soleimani 2024）は熱力学的整合。そこに Fick/cross 拡散を足した反応拡散系はまだ整合か？散逸不等式を満たすか？」**
   → 反応項は群の NSP material-point モデルそのもの（引用）、空間項は method-of-lines＋分割の現象論的 closure と明言。散逸構造を保つ/壊す箇所を議論—**整合性問題を認識していると示すだけで刺さる**。
6. **「報告した深さ構造（Fn 表層・Pg 深部）のどれだけが物理で、どれだけが1次風上の数値拡散か？D,u が placeholder なら尚更」**
   → 数値拡散を見積もり・上限を与え、fitted D_i が数値でなく物理だと論じる。Strang/陰的輸送での頑健性も。
7. **「厳密解で solver を検証したか？（純拡散ガウシアン・単一 Fourier モード減衰・MMS）」**
   → **反応オフでガウシアンを拡散させ解析解と L2 比較、細分化で→0**。これが無いと「もっともらしいだけ」。

### 加点ムーブ
- **独立した「数値手法」章/節**：強形式→離散化（中心拡散・風上移流・ghost-node BC）→分割→時間積分、各々の打ち切り次数を明記。
- **収束スタディ図**＋**厳密解検証**（ガウシアン/Fourier/MMS）。
- **保存則チェックの時系列**（総質量 Σφ、simplex 残差 max|Σφ_i+φ0−1|）を対角版と cross-diffusion 版の両方で。
- **両 CFL** を明示。
- 彼の言葉：method of lines / operator splitting / monolithic vs staggered / **observed order of accuracy** / CFL / ghost-node Neumann / conservative finite-volume flux / degenerate (volume-filling) diffusion。SPH↔FEM↔FD のトレードオフに触れる（Soleimani & Wriggers 2016 引用）。
- **再現性**：float64、JIT 決定論カーネル、格子/Δt を meta JSON に記録、argparse CLI、図のend-to-end再現手順。
- placeholder 批判を先回り：D_i, u は CLSM z-profile 較正待ちの placeholder と明言し、**手法の寄与（スキームは検証済み）**と**定量的主張（データ待ち）**を分離。質的結論は D,u スケールに頑健と示す。

---

## §3 共通言語（home-turf terminology）— これで書くと「自陣の仕事」に見える
| 使う語 | 避ける/補足 |
|---|---|
| thermodynamic consistency（変分→エネルギー収支・散逸≥0） | 「biologically plausible」より強い |
| extended Hamilton principle, H=∫(G+C+D)dt | Junker–Balzani の基盤 |
| internal variables φ_i, ψ_i；A は**生存体積** φ̄=φψ に作用 | Klempt の特徴 |
| **replicator dynamics / compositional ODE** | IKM の前では「generalized Lotka–Volterra」を前面に出さない（replicator=変分形を強調） |
| symmetric A は二次形式の自由エネルギーから**自動** | 「なぜ対称?」に即答 |
| dissipation potential / Onsager–Biot reciprocity | |
| well-posedness（存在・一意・連続依存）、parabolic regularity | |
| identifiability / effective dimensionality（r=σ_post/Δ_prior） | 「confidence interval」でなく**credible interval / posterior** |
| balance/conservation law ∂φ/∂t+∇·flux=reaction | |
| constrained variational problem（holonomic Σφ=1、Lagrange γ） | γ を式から落とさない |
| cross-diffusion（volume-filling, Painter–Hillen） | |

---

## §4 先回りで潰す穴（gaps）
- **0D Hamilton ODE は熱力学整合か** → 自由エネルギー単調減少＋∂φ/∂t·1=0 を明示。
- **A 対称は仮定でなく emergent**（二次形式から）。非対称 gLV は対称エネルギーの replicator 射影。
- **FBA 単位 ≠ 生態単位は本物**（why_micom_worked: MacArthur magnitude prior は失敗 AUC0.87 は符号のみ）。sign-only を正当化。
- **PDE well-posedness**：no-flux+Dirichlet は標準 parabolic、反応は simplex 上で滑らか有界、barrier で箱拘束。Klempt 2024 から継承。
- **空間逆問題で A,b 固定は循環でない**：A は別の制御実験（in vitro 4条件）の事後 → 別データ（FISH深さ）への informative prior＝正当な階層 Bayes。
- **D_i 非同定は失敗でなく既知の限界**（rigor の証）。図・本文で “provisional” と明記。
- **sign prior は循環でない**：fit 中に適用＋**permutation で prior 無しでも符号一致 p=4e-4**＋gLV では効かず Hamilton で効く（model-selective）。
- **小標本**（Dieckow 10×3、Heine 5種）：パラメータ/観測比を正直に＋正則化＋LOO-CV で CI 提示。
- **A 普遍 / b 患者特異**の仮説は検証可能（A 固定で b のみ fit → RMSE 上昇なら臨床変動を正直に報告）。
- **「詰め込みすぎ」批判**：1つの連続体枠組み（Hamilton→0D→PDE）×3つの直交検証軸（AGORA符号 / in-vivo Dieckow LOO / 空間FISH）。conflation でなく convergent evidence。
- **生物学的新規性**：AGORA→口腔バイオフィルムへの符号 prior（gut MICOM と別）＋**dysbiosis は bulk 組成シフトでなく深さ空間の再編**（3D FISH で検証）。連続体形式は IKM が認める“器”。

---

## §5 修論に実際に足すもの（=「二人に対策した修論」の具体 TODO）
- **ch2 理論**：拡張Hamilton原理→自由エネルギー＋散逸→δH=0→replicator の導出を1節。対称 A の emergence、Lyapunov（KL/平均適応度）単調性＝散逸≥0、ゲージ不変と固定。Klempt 2024 の 0D 極限として自モデルを示す。
- **ch5 空間**：独立した**数値手法節**（離散化・分割・時間積分・各次数）＋**収束スタディ図**＋**厳密解検証**（ガウシアン/Fourier/MMS）＋**保存則時系列**（対角 vs cross-diffusion）＋両 CFL。0D→連続体の熱力学整合の議論。
- **methods/Bayesian**：**同定性（Fisher/感度 or 事後共分散）**を明示、credible interval、compositional 16S の尤度（simplex に Gaussian は妥当か）に言及。
- **Discussion**：sign-only prior の単位整合論、permutation 検証、A普遍/b特異の検証、将来＝Fritsch 流 Bayesian 空間逆問題＋連続体統合。
- **全体**：[[THESIS_WRITING_GUIDE]] の「弱同定は結果」「主張の自己点検（92%→permutation p=4e-4）」を貫く。

---

## §6 注意：Heine 2025 は2本ある（混同しない）
- **Heine2025PeriImplant**（bib）＝ *Influence of species composition and cultivation condition on peri-implant biofilm dysbiosis in vitro*, **Frontiers in Oral Health** 2025, doi 10.3389/froh.2025.1649419 — ch3 の5種アトラクター(CS/CH/DS/DH)データ。
- もう1本：Heine et al. 2025, *Biofilm development of P. gingivalis on titanium surfaces in response to DHNA — a hybrid in vitro–in silico approach*, **Microbiology Spectrum**, doi 10.1128/spectrum.00410-25（**Soleimani 共著**）。Pg/チタンの話。**副査 Soleimani 対策として引用すると効く**が、上の5種論文とは別物。
