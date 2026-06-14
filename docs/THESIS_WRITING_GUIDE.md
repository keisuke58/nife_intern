# 執筆ガイド — プロ研究者の論文の書き方を修論に適用

> **Canonical = `nife/docs/THESIS_WRITING_GUIDE.md`**（図・知識グラフ・PROVENANCE と同居）。
> `30_Masterarbeit/notes/writing_guide.md` はそのミラー（`/thesis-sync` が canonical→ミラーへ自動コピー）。
> **編集は canonical 側で**。出典は実在の定番：Mensh & Kording 2017（PLOS Comput Biol,
> "Ten simple rules for structuring papers"）, Whitesides 2004（Adv. Mater. "Writing a Paper"）,
> Schimel 2012（"Writing Science", book）, Gopen & Swan 1990（American Scientist
> "The Science of Scientific Writing"）, Booth/Colomb/Williams "The Craft of Research"。
> 最終更新 2026-06-04。

プロは「書いてから直す」のではなく **構造を先に決めてから書く**。順番は
**① 1文のメッセージ → ② 図の骨格 → ③ アウトライン（各段落の topic sentence）→
④ 本文 → ⑤ reverse outline で論理チェック → ⑥ フィードバックで削る**。
以下、各段階をこの修論の中身（①Heine 5菌種 → ②AGORA/in-vivo/空間）に当てはめる。

---

## 段階0：たった1つの中心主張（central contribution）を決める — 最優先

Mensh & Kording の Rule 1：**論文は1つの貢献に絞る**。修論全体が支える1文を先に固定し、
全章・全図・全段落を「これを支えているか？」で取捨選択する。叩き台（react してから直す）:

> **「口腔バイオフィルムの commensal→dysbiotic 転移は、再現可能なネットワークの
> *位相的再編成*（Pg の中心化・Veillonella の代謝シンク化）であり、それは
> in-vitro（Heine 5菌種, GPU-Bayesian）と in-vivo（Dieckow/Duran-Pinedo）で符号一致し、
> 代謝（AGORA）と整合し、深さ方向に空間構造化されている。」**

- ①は「**推定法と再現性**」（GPU-TMCMC で 10000p posterior、pH を独立予測 R²=0.78）。
- ②は「**因果の裏付けと一般性**」（代謝 prior・別コホート・空間）。
- この1文に乗らない解析は、面白くても本文から落として appendix に回す。「全部入れる」は素人。

**やること**：この1文を `main.tex` 冒頭にコメントで固定し、毎章これを見ながら書く。

---

## 段階1：図が骨格（figures-first）— Whitesides の核

Whitesides の主張は「**図（と1文の仮説）を先に作れ。論文＝図の連なり＋つなぎの文**」。
各図は**1つの主張**だけを担う。修論の図 backbone（既存の実資産にマップ）：

| 図 | 1つの主張 | ファイル（factory repo） | 章 |
|---|---|---|---|
| Graphical abstract | 研究全体像（①→②） | `nife/results/figures/concept_overview_pub.png` | ch1 |
| Pipeline | データ→推定→検証の流れ | `nife/results/figures/pipeline_overview_pub.png` | ch1/ch2 |
| 4 attractors | CS/CH/DS/DH が別状態 | `Tmcmc202601`（posterior 由来） | ch3 |
| Posterior A + 符号確率 | 相互作用が**確率的に同定**できる | `ultimate_10000p`, `sign_probability` | ch3 |
| **pH 予測** | モデルが**独立変数**を当てる（R²=0.78, LOO 0.92） | `fig_ph_validation.pdf` | ch3 |
| AGORA pipeline | 代謝→符号 prior の構築 | `fig2_agora_pipeline` | ch4 |
| no-prior 符号一致 | prior 抜きでも交差検証（perm p=4e-4） | `dieckow_cr` | ch4 |
| Duran-Pinedo 2コホート | 強ペア符号 89%（別コホート） | `duranpinedo_validation` | ch4 |
| Network 再編成 | Pg 中心化・Veillonella sink・ρ(DH−CS)=−0.49 | `guild_network` | ch4 |
| 深さプロファイル | Fn–Pg 深さ分離（3D M1 0.22–0.33） | `zprofiles_all_ti*`, `fish_3d` | ch5 |
| 横均一化 | DH は CV~0.33（均一）、CH はパッチ | `fish_3d_lateral_*` | ch5 |
| 空間 cross-feeding | Veillonella が So より浅い（Wilcoxon p=0.002） | `spatial_crossfeeding.png` | ch5 |
| 拡散フィット | 収束したが**弱同定**（D 下限張り付き）= それ自体が知見 | `diffusion_fit` | ch5 |

**やること**：本文を書く前に、この表を「最終図リスト」として確定し、各図の**キャプションを先に書く**
（キャプションは「主張＋読み取り方」の1〜2文。図だけ見て論旨が追えるのが理想）。図が決まれば本文は図の説明に収束する。

---

## 段階2：書く順番 — 難しい前付けは最後

プロは時系列に書かない。**易しい所＝Methods/Results から**書き、**Intro/Discussion を最後**に。
理由：結果が固まらないと「何の物語か」が決まらないから（Whitesides）。この修論の推奨順：

1. **ch2 Theory**（最初）— gLV/replicator・TMCMC・FBA・PDE。事実の記述で書きやすく、記号も確定する。
2. **ch3 Heine①** — 既存 `nife/heine_paper/nishioka_heine_paper.tex` を移植。原型なので最初の results 章に最適。
3. **ch4 Dieckow②** — AGORA prior＋in-vivo＋network（`dieckow_paper/dieckow_analysis.tex` 移植＋7月の結果）。
4. **ch5 Integration** — 空間 PDE/FISH/3D。**正直な弱同定ナラティブ**（下記）。
5. **ch1 Introduction**（results が固まってから）— 問い→ギャップ→本研究の貢献（段階0の1文）。
6. **ch6 Conclusion** — 主張の再述＋限界＋展望（GDI/Joshi, FNO surrogate, full-3D fit）。
7. **Abstract → Title**（最後）— 最も読まれる。Mensh Rule 9：ここに一番時間をかける。

---

## 段階3：段落の型 — C-C-C / Claim–Evidence–Interpretation

Mensh の Rule 3：**あらゆる階層で Context–Content–Conclusion**。段落も同じ。
1段落＝1メッセージ。**topic sentence を最初の文に置く**（その段落の主張）。Results はとくに
**CEI**（Claim → Evidence → Interpretation）で統一すると速く書ける。あなたのデータでの例：

> **(Claim)** The commensal→dysbiotic shift is a topological reorganisation, not a uniform
> intensification of interactions. **(Evidence)** Across the 10 000-particle posterior, the
> sign-resolved interaction graph rewires substantially between states (Spearman
> ρ(DH−CS) = −0.49), with *P. gingivalis* moving to high betweenness centrality and
> *Veillonella* acting as a net metabolic sink. **(Interpretation)** This means dysbiosis
> is better described by *who connects to whom* than by overall interaction strength,
> which is why a sign-constrained model transfers across cohorts where a magnitude-fit
> model would not.

各段落の最後に **"So what?" テスト**：その段落が段階0の1文に効いていなければ削る/移す。

---

## 段階4：文のメカニクス — Gopen & Swan（読者の期待）

英語で「伝わる」文の規則。3つだけ守れば質が上がる：

1. **Topic position（文頭）＝既知・文脈**、**Stress position（文末）＝新規・重要**。
   一番言いたい語を文末に置く。
   - ✗ *We observed a 0.49 anticorrelation in the rewiring when comparing DH and CS.*
   - ✓ *Comparing DH with CS, the interaction graph rewires strongly — a Spearman
     correlation of **−0.49**.*（重要な数字を stress position へ）
2. **主語と動詞を近づける**。長い修飾で離さない。
3. **動作は動詞に**（名詞化＝nominalization を減らす）。"performed an analysis of" → "analysed"。

Schimel の助言：**ギャップ（問い）を先に提示**してから結果を出す。読者は「物語」だから読み進む。
Intro は OCAR（Opening→Challenge→Action→Resolution）で：広い文脈→未解決の問い→本研究→結論の予告。

---

## 段階5：限界の見せ方 — 弱点を“発見”として書く（この修論の強み）

すでに実践している「**収束したが弱同定**（D が下限に張り付く＝深さプロファイルだけでは5拡散係数を
決められない）」は、隠すのではなく**それ自体を結果**として書くのがプロの作法
（"limited identifiability is itself a finding"）。Discussion で：
- 何が同定できて（符号・位相・pH 予測）何ができないか（D の大きさ・絶対 flux 単位）を**明確に分離**。
- 「FBA flux 単位 ≠ 生態相互作用単位だから magnitude でなく sign を拘束」も**設計上の正しい判断**として書く
  （`docs/why_micom_worked.md`）。負の結果（L4 prior が逆効果、cross-diffusion が Turing パターンを作らない）も、
  反応支配という**機構的結論**として提示する。

---

## 段階6：reverse outline（提出前の論理チェック）

書き上げたら、**各段落の最初の文だけ**を抜き出して1ページに並べる（= reverse outline）。
それだけ読んで論旨が一直線に通れば OK。通らなければ段落順を入れ替える/削る。Mensh の Rule 4
（zig-zag を消す・並列構造）はここで担保する。`pandoc` で topic sentence 抽出を半自動化してもよい。

---

## 段階7：フィードバックの回し方（Mensh Rule 10）

- **早く・粗く・頻繁に**。完璧な章を1回見せるより、**図の骨格＋アウトライン**を Junker に早期に見せる。
- 月次メール（`MONTHLY_EMAIL_STRATEGY.md`）に「今月固めた図1枚＋1文の主張」を必ず入れる。
- 受けた指摘は references.bib と本ファイルに反映。「reduce, reuse, recycle」：同じ説明を使い回す。

---

## チェックリスト（章を1つ書くたびに）

- [ ] この章の図リストが確定し、キャプションが先に書けている
- [ ] 各図は1主張だけ／図だけで論旨が追える
- [ ] 各 Results 段落が CEI で、topic sentence が冒頭にある
- [ ] 重要な数字が文末（stress position）にある
- [ ] 同定できる/できないを明示（弱同定は結果として記述）
- [ ] reverse outline で論理が一直線
- [ ] 段階0の1文を支えない段落を appendix に移したか確認
- [ ] 出典（references.bib）と図の出典（PROVENANCE）が付いている

---

## 1ページ要約（Ten rules, 凝縮）

1. 1つの中心主張に絞る（段階0）。 2. 図が骨格、キャプション先行（Whitesides）。
3. 易しい所＝Methods/Results から書く。 4. 段落＝C-C-C / CEI、topic sentence 冒頭（Mensh R3）。
5. 重要語は文末、主語と動詞は近く（Gopen & Swan）。 6. 問い（ギャップ）を先に出す物語構造（Schimel）。
7. 限界は隠さず結果として書く。 8. reverse outline で論理を一直線に（Mensh R4）。
9. Title/Abstract/図に一番時間をかける（Mensh R9）。 10. 早く粗く頻繁にフィードバック（Mensh R10）。
