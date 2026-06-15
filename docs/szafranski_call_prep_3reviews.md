# Szafranski 通話準備 — 3本の論文の関係と使い方

Szymon が 2026-06-14 に送ってきた2本のレビュー＋今回 citation を直したコホート2論文。
"inferring interactions" という reframe をどう自分の物語に組み込むかのメモ。

> PDFs: `docs/refs/nateco2022_ecological_modelling_emergent.pdf`,
> `docs/Ona2025_InteractionsMethodsTRENDSMICROBIOL.pdf`,
> `docs/refs/duran_pinedo2021_perio_bmcbiol.pdf`
> bib keys: `vandenBerg2022`, `Ona2025`, `DuranPinedo2021`

## 3本の関係（ざっくり）

| | 役割 | 自分の使い方 | どこで引く |
|---|---|---|---|
| **① van den Berg 2022, *Nat Ecol Evol*** (Patil) | **モデルの地図**（gLV / consumer-resource / trait・individual-based / GEM、complementarity） | 研究の正当化フレーム＋identifiability caveat | Intro / Discussion |
| **③ Oña 2025, *Trends Microbiol*** (Kost) | **推論手法の地図**（手法別の強み弱み・network用語） | 自手法の立ち位置＋honest caveat＋network用語 | Intro / Discussion |
| **② Duran-Pinedo 2021, *BMC Biol*** | **コホート2の実体**（疾患進行の生態学的遷移、PRJNA725874） | cross-cohort を限定主張にする根拠＋哲学の一致 | Methods / コホート記述 |

## 電話での一言（要旨）

> **①の complementarity フレームで自分の仕事を位置づけ、③の手法分類で「時系列 gLV × FBA 拘束」を *a combined method* と説明し、②の設計差を認めて cross-cohort は *limited claim* にする。**

これで3本とも自分の物語に組み込める。①と③は Intro/Discussion で実際に引く、②は Methods/コホート記述で正しく引く。

## 各論文の効きどころ（詳細）

### ① van den Berg et al. 2022 — Ecological modelling approaches for predicting emergent properties
24p レビュー。章立て＝ **Lotka-Volterra → MacArthur consumer-resource → trait-based → individual-based → genome-scale metabolic → Parameter inference and data integration** ＝研究の地図そのもの。
- メイン主張＝**「gLV と GEM の complementarity を活かせ」** → 「FBA の符号で gLV を拘束する」自分の設計の正当化に直接引ける。
- **co-occurrence ≠ interaction（非対称・間接を捉えない）** → なぜ相関でなく ODE 推論か。
- **gLV structural identifiability**（Remien 2021 引用）→ "inference is underdetermined" スライドに接続。
- MacArthur consumer-resource の節 → 落とした「MacArthur magnitude prior 失敗」を文脈づけられる。

### ③ Oña, Shreekar & Kost 2025 — Disentangling microbial interaction networks
16p、**推論手法カタログ＋長所短所**。手法を横並び評価：co-occurrence（誤りやすい・因果なし）／ causal inference（Granger＝因果区別できるが高次元時系列で計算重い）／ pairwise coculture（直接相互作用は明快だが組合せ爆発・高次相互作用に鈍感）／ FBA・community FBA・dynamic FBA（kinetic 不要だがゲノムアノテーション品質依存＋biomass 最大化前提が限界）。結論＝**「単一の最良手法はない、組合せが必要」**。
- **自手法の立ち位置：** 「時系列 gLV 推論（≒causal/dynamical）× AGORA-FBA 拘束」＝この論文の言う *combination of methods* の実践。
- **honest さの裏づけ：** FBA の限界（アノテーション・biomass 前提）明記 → 「AGORA prior は modelling choice、data-confirmed fact ではない」と一致。
- **用語の宝庫：** hub / motif / nestedness / centrality / sparsity → network 解析デッキ（中心性・backbone・Pg 中心化）の語彙統一に使える。

### ② Duran-Pinedo et al. 2021 — Long-term dynamics ... clinical disease progression
歯周炎の**疾患進行**縦断研究（PRJNA725874）。要点は組成変化でなく**生態学的ダイナミクス**：progressing sites は low asynchrony＋convergence、stable sites は directional change。
- **Szymon の caveat が正しい：** Dieckow（早期ペリインプラ定着・数週間・健康寄り）とは設計・時間スケール・疾患状態が別物 → 「Actino 軸がコホート間一致」を強く言えない（今回のトーンダウンは妥当）。
- 逆に Duran-Pinedo 自身が「組成でなく生態学的要素で health→disease を説明」と言うのは、自分の相互作用推論の哲学と同方向＝引きどころ。
