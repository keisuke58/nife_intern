# 専門用語単語帳

## 臨床診断

| 用語 | 読み | 意味 |
|------|------|------|
| **PIH** | ピーアイエイチ | Peri-Implant Health — インプラント周囲健康 |
| **PIM** | ピーアイエム | Peri-Implant Mucositis — インプラント周囲粘膜炎（炎症あり、骨吸収なし、可逆） |
| **PI** | ピーアイ | Peri-Implantitis — インプラント周囲炎（骨吸収あり、重症、難治） |
| **Probing depth** | プロービング デプス | 歯肉溝の深さ（mm）。深いほど病態が重い |
| **GCF** | ジーシーエフ | Gingival Crevicular Fluid — 歯肉溝浸出液。インプラント周囲の組織液 |
| **Peri-implant sulcus** | ペリインプラント サルカス | インプラントと歯肉の間の溝 |

---

## 微生物・生態学

| 用語 | 読み | 意味 |
|------|------|------|
| **So** | ソー | *Streptococcus oralis* — 健康関連の commensal 菌 |
| **An** | アン | *Actinomyces naeslundii* — commensal 菌 |
| **Vd / Vp** | ヴィーディー / ヴィーピー | *Veillonella dispar / parvula* — 乳酸消費菌 |
| **Fn** | エフエヌ | *Fusobacterium nucleatum* — commensal と pathogen をつなぐ橋渡し菌 |
| **Pg** | ピージー | *Porphyromonas gingivalis* — 主要病原菌 |
| **CT I-IV** | シーティー | Community Type — SIMPROF クラスタリングで定義された菌叢の型。CT I (健康) → CT IV (病的) |
| **SIMPROF** | シムプロフ | SIMilarity PROFile test — 菌叢データのクラスタリング手法 |
| **Coaggregation** | コアグリゲーション | 異種菌が物理的にくっつく現象。Fn が So と Pg を橋渡しする |
| **Commensal** | コメンサル | 共生菌。宿主に害を与えない常在菌 |
| **Dysbiotic** | ディスバイオティック | 菌叢異常。病原菌優占状態 |
| **16S rRNA** | 16Sリボソーム | 細菌の同定に使うマーカー遺伝子。どの菌がいるかを調べる |
| **Metatranscriptomics** | メタトランスクリプトミクス | 菌叢全体の RNA を解析。どの菌が「活動しているか」を調べる |
| **EC number** | イーシー ナンバー | Enzyme Commission 番号。酵素の種類を表す番号（例: 3.4.21.102） |
| **KEGG** | ケッグ | Kyoto Encyclopedia of Genes and Genomes — 代謝経路データベース |
| **Bray-Curtis dissimilarity** | ブレイカーティス | 菌叢間の違いを数値化する指標（0=同じ、1=全く違う） |

---

## 数理モデル

| 用語 | 読み | 意味 |
|------|------|------|
| **ODE** | オーディーイー | Ordinary Differential Equation — 常微分方程式。時間変化を表す式 |
| **Hamilton ODE** | ハミルトン | Hamilton 2020 が提案した5種バイオフィルムの replicator 方程式 |
| **Replicator ODE** | レプリケーター | 各菌種の割合（φ）の変化を相互作用行列 A で表す方程式 |
| **φ₀ (phi zero)** | ファイゼロ | ODE の初期値。125サンプルの菌種組成がこれに対応 |
| **φ_eq** | ファイイーケー | ODE を積分したときの定常状態（equilibrium） |
| **Attractor** | アトラクター | ODE が収束する安定状態。commensal attractor と dysbiotic attractor の2種類 |
| **Basin of attraction** | アトラクター盆地 | どの初期値がどの attractor に引き寄せられるかの領域 |
| **Separatrix** | セパラトリクス | 2つの basin の境界線。PIM はここに位置する |
| **Bifurcation** | バイファーケーション | パラメータ変化で attractor の数や位置が変わること |
| **A matrix** | エーマトリクス | 菌種間相互作用行列。A[i,j] = 種 j が種 i に与える影響 |
| **b vector** | ビーベクトル | 各菌種の intrinsic 増殖/減衰率 |
| **MAP** | マップ | Maximum A Posteriori — TMCMC で求めた最も確からしいパラメータ値 |
| **TMCMC** | ティーエムシーエムシー | Transitional Markov Chain Monte Carlo — ベイズパラメータ推定法 |
| **Hill gate** | ヒルゲート | 菌種が多すぎるとき成長を抑制する非線形関数（K=0.05, n=4） |
| **DI** | ディーアイ | Dysbiosis Index — 菌叢の病的度合いを0〜1で表す指標 |

---

## 統計・データ解析

| 用語 | 読み | 意味 |
|------|------|------|
| **PCA** | ピーシーエー | Principal Component Analysis — 多次元データを2次元に圧縮して可視化 |
| **MDI** | エムディーアイ | Microbiome Dysbiosis Index — log(病的菌/健康菌) で計算する指標 |
| **eMDI** | イーエムディーアイ | extended MDI — MDI に EC 活性 9 種を加えた指標（R²=0.51, AUC=0.87） |
| **Cross-sectional** | クロスセクショナル | 横断研究。各患者を一度だけ測定（時系列なし） |
| **Cohort** | コホート | 同じ条件で集めた患者群 |
| **BIC** | ビーアイシー | Biofilm Implant Cohort — Szafrański 2025 の 48人 125サンプルのコホート |
| **CAP** | キャップ | Constrained Analysis of Principal coordinates — 診断グループ間の差を可視化 |
| **Bray-Curtis** | ブレイカーティス | 菌叢間の非類似度指標 |
| **Nearest-neighbor** | ニアレストネイバー | 最近傍法。φ₀ から最も近い reference attractor に分類する方法 |

---

## 実験手法

| 用語 | 読み | 意味 |
|------|------|------|
| **HOBIC** | ホービック | Hydraulic Oral BIofilm Chamber — 流れのある in vitro バイオフィルム培養装置 |
| **Static** | スタティック | 流れなしの培養条件（6-well plate） |
| **BHI** | ビーエイチアイ | Brain Heart Infusion — 細菌培養用の培地 |
| **qRT-PCR** | キューアールティーピーシーアール | 定量的逆転写PCR。遺伝子発現量を測定する |
| **Ti disc** | チタンディスク | チタン製の円盤。インプラント表面の代替として in vitro で使用 |
| **Paper point** | ペーパーポイント | 歯肉溝からサンプルを採取する吸水性の細い棒 |
