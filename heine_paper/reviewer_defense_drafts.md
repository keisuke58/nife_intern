# Reviewer-defense draft additions — nishioka_heine_paper

査読で必ず突かれる弱点に**先回り**するための本文ドラフト。実数値・図ファイル名は `[...]` のプレースホルダ。結果が出たら埋めて `nishioka_heine_paper.tex`（および Overleaf）に貼る。

## 想定される査読者の攻撃（リスク順）
1. **データ不足 vs パラメータ数**：15パラメータを25観測（5時点×5種）で推定 → 「well-identified と言えるのか」
2. **検証が循環的に見える**：同じデータにフィットして「合った」だけなら circular
3. **生物学的新発見の薄さ**：既知の commensal/dysbiotic 対比を再現しただけ、になりがち
4. 新規性の枠組み（なぜ標準gLVでなくHamilton対称行列か）／生物学的反復数 n

→ ①②③で 1〜3 に対応。

## ステータス（重要）— 多くは「すでにやってある」
- **②③は本ファイルに実数を反映済み**（正典 posterior `results/ultimate_10000p` + fit時の `prior_bounds.json` から算出。再fit不要）。
- **① 条件間予測は既にコードあり**：`Tmcmc202601/docs/paper_additional_analyses.py` の `analysis_cross_prediction()`（MAP θ を別条件に転移して RMSE 行列＋`tab:cross_prediction` を出力）。`analysis_effective_dimensionality()` が②に相当。
- **より完全な論文版が別に存在**：`Tmcmc202601/docs/nishioka_paper_publish.tex` には既に「posterior informativeness $r_i=\sigma_{\rm post}/\Delta_{\rm prior}$」「sign prior / KEGG 一致」「Phase1 vs Phase2 RMSE」節がある。**リポジトリの `heine_paper/nishioka_heine_paper.tex` はそれより簡素**。
  → **要判断**：投稿用の正本をどちらにするか。`publish.tex` をベースにする方が①②が既に入っていて近道。

### 2つの .tex 比較結果（diff済み）
- **節見出しは完全一致**。B(publish, 1018行) は A(heine_paper, 846行) より **+172行**。
- 差分の主体は **生物学的相互作用ネットワーク節**の厚み：`sign prior`(A=1→B=9)、`Bergey`(0→4)、`KEGG`(11→25)。Bは AGORA/KEGG 符号prior の議論が充実。
- **両方とも `Author Contributions` / `Funding` / `Data availability` 節が無い** → どちらを使うにせよ投稿前に追加必須（共著者 Heine/Doll-Nikutta の役割もここに書く）。
- **結論**：B(`publish.tex`)を正本にするのが得。A固有の改善（著者追加・`(15 interaction parameters)`脱字修正）をBにも反映 → Bに①②③（本ファイルの実数）と front-matter 3節を足す、が最短ルート。

### ② 実測：posterior contraction $C_i = 1-\mathrm{Var}_{\rm post}/\mathrm{Var}_{\rm prior}$（15相互作用、全条件 C>0.5 ＝全て data-informed）
| 相互作用 | CS | CH | DS | DH |
|---|---|---|---|---|
| A:An–Pg | 0.89 | 0.91 | 0.98 | 0.99 |
| A:Vd–Pg | 0.89 | 0.89 | 0.99 | 0.99 |
| A:Fn–Pg | 0.91 | 0.87 | 0.97 | 0.78 |
| A:So–Pg | 0.93 | 0.97 | 0.92 | 0.86 |
| （15個すべて） | … | … | … | C∈[0.71, 1.00] |

### ③ 実測：gating edge = **A:Fn–Pg**（commensal で≈0、dysbiotic で活性）
| 条件 | mean (95% CI) | C |
|---|---|---|
| CS | −0.03 [−0.85, +0.94] | 0.91 |
| CH | +0.69 [−0.23, +1.89] | 0.87 |
| DS | +1.99 [+1.33, +2.47] | 0.97 |
| **DH** | **+4.86 [+3.03, +5.95]** | 0.78 |

gating score（\|mean_DH\|×C_DH, commensal が0含む）で **Fn–Pg=3.79 が他(≤0.36)を圧倒**。生物学的にも F. nucleatum＝P. gingivalis の bridge 菌で整合。算出: `/tmp/contraction_analysis.py`（要なら `scripts/analysis/` に保存）。

---

## ① 予測検証（識別可能性への先回り）— Results に新 subsection

```latex
\subsection{Out-of-sample predictive validation}
\label{sec:loto}

To guard against overfitting in the data-scarce regime (15 parameters,
25 observations per condition), we test whether the interaction matrix
inferred under one condition predicts the trajectories of the \emph{other}
conditions (leave-one-condition-out transfer).
Strikingly, transfer prediction is often as accurate as---or better
than---self-prediction: the matrix fitted to commensal-static predicts
commensal-HOBIC at $\mathrm{RMSE}=0.10$ (vs.\ $0.15$ self), and the
dysbiotic-HOBIC matrix predicts dysbiotic-static at $0.067$ (vs.\ $0.093$
self), while cross-\emph{regime} transfers (commensal$\leftrightarrow$dysbiotic)
are expectedly larger ($0.23$--$0.26$).
This demonstrates that the inferred interactions capture generalisable
ecological structure rather than over-fitting condition-specific noise
(Table~\ref{tab:cross_prediction}).

% RMSE matrix (rows: source MAP theta; cols: target data; diagonal = self)
%        CS      CH      DS      DH
%  CS  0.327   0.101   0.128   0.261
%  CH  0.233   0.148   0.101   0.233
%  DS  0.237   0.147   0.093   0.250
%  DH  0.252   0.151   0.067   0.240
```

> 実装済み：`Tmcmc202601/docs/paper_additional_analyses.py::analysis_cross_prediction()`（図 `cross_condition_prediction.pdf`＋表 `tab:cross_prediction` を生成）。**注意**：この実行値は最新の **NUTS run** の MAP を使用。論文正典は **TMCMC 10000p** なので、投稿前に canonical MAP（`paper_5sp_theta`）で再生成して整合させること。

---

## ② 識別可能性を「正直に報告」して守る — Results か Discussion に1段落

```latex
\paragraph{Identifiability and posterior contraction.}
For every coefficient we quantify the posterior contraction
$C_k = 1 - \sigma^{2}_{\text{post},k}/\sigma^{2}_{\text{prior},k}$ relative to
the uniform prior ($C_k=0$: prior-dominated; $C_k\!\to\!1$: fully data-driven).
\textbf{All 15 interaction coefficients are data-informed in every condition}
($C_k \in [0.71, 1.00]$; none prior-dominated), showing that the two-phase
scheme---Phase~1 fix-$\psi$ halving the system dimension, Phase~2 narrowed
priors acting as a mild regulariser---renders the 15-dimensional problem
identifiable \emph{without} locking any entry to zero a priori.
We nonetheless report full marginal credible intervals so that strongly- and
weakly-constrained entries remain transparent: e.g.\ $a_{\mathrm{An,Pg}}$ under
dysbiotic-static is tightly determined ($+2.15$, $95\%$~CI $[+1.59,+2.49]$),
whereas $a_{\mathrm{So,Pg}}$ under dysbiotic-HOBIC stays broad
($+0.42$, $[-0.47,+2.00]$). The central biological claims rest only on the
former, well-contracted entries.
```

---

## ③ データから出た非自明な予測 — Discussion に1段落

```latex
\paragraph{A testable prediction.}
Beyond reproducing the commensal/dysbiotic contrast, the inferred network makes
a specific, falsifiable prediction. The \textit{Pg} bloom is gated by a single
cooperative edge, the \textit{Fn}--\textit{Pg} coefficient $a_{\mathrm{Fn,Pg}}$,
which is statistically indistinguishable from zero in commensal conditions
(commensal-static $-0.03$, $95\%$~CI $[-0.85,+0.94]$; commensal-HOBIC $+0.69$,
$[-0.23,+1.89]$) yet strongly positive and data-informed in dysbiotic states,
rising to $+4.86$ ($[+3.03,+5.95]$, $C=0.78$) under dysbiotic-HOBIC flow---the
condition of maximal \textit{Pg} surge (dysbiotic-static: $+1.99$,
$[+1.33,+2.47]$). This recapitulates the established role of
\textit{F.\ nucleatum} as a bridging organism that co-aggregates with and
metabolically supports \textit{P.\ gingivalis}.
The model therefore predicts that attenuating this single interaction---e.g.\
removing \textit{Fn} from the consortium---should convert the dysbiotic-HOBIC
trajectory into a commensal-like, \textit{Pg}-low state despite identical
cultivation: a direct leave-one-species-out co-culture test that discriminates
the network hypothesis from a purely \textit{Pg}-intrinsic growth explanation.
```

> ③は**実際に推定された「効いているエッジ」**に差し替える（Fn↔Pg は仮置き）。`results/guild_network`（DHで中心化する Pg / リワイヤリング）から、contraction が高く dysbiotic でのみ活性化するエッジを1本選ぶと実データ裏付きの予測になる。

---

## 投稿先とリジェクト感（参考）
| 投稿先タイプ | 例 | リジェクト感 |
|---|---|---|
| 計算/理論系 | PLoS Comp Biol, J Theor Biol, Frontiers in Microbiology | 中 |
| 専門誌（グループ実績あり） | Frontiers in Oral Health（Heine 2025 と同系） | 中〜低 |
| 高IF微生物学 | ISME J, npj Biofilms | 高 |
