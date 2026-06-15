# Job-fit mapping — requirements × portfolio assets (Tokyo AI-engineer, 2028 entry)

Working doc for application materials. Maps each target company's stated
requirements to concrete evidence in this repo (file links + real numbers).
Companion to [../PORTFOLIO.md](../PORTFOLIO.md). Final 5, fit-ordered.

Cross-cutting one-liner (use everywhere):
> 機械論モデル × ドメイン事前知識 × ベイズ的不確実性を組み合わせ、データが少なく
> 説明責任が要る領域で「機械論がブラックボックスMLに勝つ時/勝たない時」を定量判断
> できる。

---

## 1. Preferred Networks / PFCC — Matlantis  ◎ 最適合だが高競争 / 大手町

⚠️ **職種で難易度が割れる**: リサーチャー職=実質**博士/博士課程**前提(修士新卒には厳しめ)。
**エンジニア職(Matlantis材料探索含む)=修士でも応募可だがコーディングテスト＋複数面接で高競争。**
→ 修士新卒で狙うなら**エンジニア枠**。武器=①ポートフォリオ公開 ②AtCoder等のコーディング地力 ③投稿中論文。

求人の核: DL×原子シミュレータ(NNP)で材料探索、微分可能シミュレーション、科学計算。

| 要件 | レポの対応資産 | アピール |
|---|---|---|
| DL×シミュレーションの融合 | [hamilton_ode_jax_nsp_ift.py](hamilton_ode_jax_nsp_ift.py) — 陰関数定理でNewton解を通すcustom-VJP backprop | **微分可能シミュレータをゼロから実装**。NNP/可微分FFと同じ発想 |
| 材料探索の高速化 | [portfolio/mi_active_learning_demo.py](portfolio/mi_active_learning_demo.py) — GP代理+BO、ランダム比 ~65x低regret | サンプル効率の良い探索ループを実証 |
| 大規模科学計算/JAX | JAX `grad`/`vmap`/`jit`、GPU TMCMC([scripts/analysis/dieckow_hamilton_fit.py](scripts/analysis/dieckow_hamilton_fit.py)) | GPUバッチ化・autodiffの実務 |
| 機構モデル vs ML判断 | [results/benchmark_baselines/fig_benchmark_publication.pdf](results/benchmark_baselines/fig_benchmark_publication.pdf) | 生成モデルも自作した上で機構の勝ちを定量化 |
| 補強ポイント | NNP/分子動力学(MD)・原子スケールGNNの経験は薄い → Matlantisチュートリアル/簡易MD一本やると完璧 |

## 2. 富士フイルムAI（ソフトウエア）  ◎ 3軸直撃 / 新横浜・西麻布・修士可

求人の核: DL/画像アルゴリズム開発、統計・多変量＋**バイオインフォ＋数理最適化**、生成AI＋解釈性、MI=生成AI＋マルチスケールsim＋DL。

| 要件 | レポの対応資産 | アピール |
|---|---|---|
| DL/画像処理アルゴリズム | [portfolio/fish_unet_train.py](portfolio/fish_unet_train.py) — 3D U-Net実訓練(held-out Dice ~0.40)、[fish_decode.py](fish_decode.py) 4ch→5種アンミックス | 共焦点3D画像を前処理〜DLまで一気通貫 |
| バイオインフォ＋数理最適化 | [guild_agora_signs.py](guild_agora_signs.py)(AGORA dFBA)、gLV/replicator推論 | systems biology×最適化の実例 |
| 生成AI | [benchmark_diffusion.py](scripts/analysis/benchmark_diffusion.py)(DDPM)・[benchmark_flow_matching.py](scripts/analysis/benchmark_flow_matching.py) | 拡散/フローを自前実装 |
| 解釈性(interpretability) | 符号prior正則化＝physics-informedで符号を制約([loo_cv_kegg_prior.py](loo_cv_kegg_prior.py)) | 「なぜその相互作用か」を機構で説明 |
| MI=sim×DL | [portfolio/mi_active_learning_demo.py](portfolio/mi_active_learning_demo.py)＋FEM([masterarbeit_ansys_fem/extensions/](masterarbeit_ansys_fem/extensions/)) | マルチスケールsimの素地 |
| 補強ポイント | ほぼ穴なし。修士でOK＝新卒応募に最有利 |

## 3. TEL（半導体プロセス×AI / Physics AI）  ◎ 適合トップ級 / ⚠️AI拠点=札幌

求人の核: Physics AI＋デジタルツインで**少データ予測**、プロセス最適化(プラズマALD膜応力)、故障検知/予知、自律装置。

| 要件 | レポの対応資産 | アピール |
|---|---|---|
| Physics-informed ML | [scripts/pde/pinn_diffusion_inverse.py](scripts/pde/pinn_diffusion_inverse.py) — PINNで拡散係数を逆推定 | **物理組込MLそのもの** |
| 少データで高精度 | TMCMC＋LOO-CV(患者N=10) [loo_cv_kegg_prior.py](loo_cv_kegg_prior.py) | 小データでの正則化・汎化を実証 |
| プロセス最適化 | [portfolio/mi_active_learning_demo.py](portfolio/mi_active_learning_demo.py)(GP+BO)、SALib Sobol感度[comets/sweep_comets_0d.py](comets/sweep_comets_0d.py) | プラズマALD膜応力最適化の直接アナロジー |
| 膜応力=力学 | [b4_viscoelastic_growth_stress.py](masterarbeit_ansys_fem/extensions/b4_viscoelastic_growth_stress.py)、phase-field[b1](masterarbeit_ansys_fem/extensions/b1_phasefield_fracture.py) | 成長/粘弾性応力をFEMで扱える |
| デジタルツイン/UQ | ベイズposterior予測[scripts/analysis/dieckow_posterior_predictive.py](scripts/analysis/dieckow_posterior_predictive.py) | 予測に信頼区間を付ける |
| 補強/確認 | **勤務地(札幌 vs 東京)を最優先で確認**。SiC([sic-dicing-gp])で半導体プロセス文脈の即戦力訴求 |

## 4. キヤノン（AI・画像処理技術開発）  ○ 真の東京(下丸子) / 医用3D

求人の核: DL/強化学習/データマイニング、医用・**3D技術**、高画質化、C/C++/Python。

| 要件 | レポの対応資産 | アピール |
|---|---|---|
| DL画像アルゴリズム | [portfolio/fish_unet_train.py](portfolio/fish_unet_train.py)(3D U-Net)、[results/fish_3d/](results/fish_3d/) | 医用に近い3Dセグメンテーション実績 |
| 3D技術 | 共焦点zスタック処理[scripts/pde/lif_to_zprofiles.py](scripts/pde/lif_to_zprofiles.py)、3D可視化(ParaView) | ボリュームデータの実務 |
| 画質定量化/多変量 | colocalization/相関統計[results/fish_pair_correlation.pdf]、ベイズUQ | 定量評価の素地 |
| 補強ポイント | **強化学習・画質評価・C/C++**は薄い → C++小実装かRL入門を一本。ベイズ/simは前面に出ない求人なので画像実績で押す |

## 5. NEC 研究所（AI for Science）  ○ 科学ML軸 / 東京・川崎

求人の核: AI for Science、統計解析＋ML＋**数理最適化**、設計開発力。

| 要件 | レポの対応資産 | アピール |
|---|---|---|
| AI for Science | 機構ODE/PDE推論一式、[PAPER_OUTLINE.md] のサイエンス駆動 | 科学課題をMLで解く一連の実績 |
| 統計解析・ML | TMCMC/ベイズ、LOO-CV、ML baseline比較 | 統計的厳密さ(対па検定など) |
| 数理最適化 | GP+BO[portfolio/mi_active_learning_demo.py]、NNLS逆問題[b3_ml_surrogate.py](masterarbeit_ansys_fem/extensions/b3_ml_surrogate.py) | 最適化の実装力 |
| 補強ポイント | 研究職は博士相当を求める場合あり → 修士+論文(投稿中)+国際インターンで実績量を訴求 |

---

## 共通で効く3点（全社）
1. **再現可能な実証**: `portfolio/` のデモは数秒でreviewerが追試可能。
2. **誠実な評価設計**: FOV丸ごとhold-out、対патест、未達は未達と明記。
3. **機構×データのハイブリッド判断**: ブラックボックス一辺倒でない設計眼。

## 全社共通の補強候補（あれば更に強い）
- LLM/生成AIの実運用1本（どこでも加点、特に富士DX・NEC）
- C/C++ or プロダクション配備(MLOps)経験(キヤノン・DISCO系で重視)
- 公開: PORTFOLIO+demoをGitHub公開＆READMEに英語要約(全社の一次スクリーニングに効く)

## 優先順位(東京前提・修士新卒で補正)
- **本命(修士で十分戦える):** 富士フイルムAI(修士明示OK) ≧ 中外製薬 ≧ NEC研究所 ≧ キヤノン
- **挑戦(適合◎だが要対策):** PFNエンジニア枠(コーディングテスト＋公開実績必須)、TEL(適合最上位・**札幌確認**)
- **別枠(研究続けたいなら博士進学後):** PFNリサーチャー職
*中外製薬(深層生成＋ベイズ最適化で分子設計)も適合◎ — 本命級。*
