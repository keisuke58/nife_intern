# Szafrański et al. 2025 — Peri-implant Ecosystem (bioRxiv)

**DOI**: https://doi.org/10.1101/2025.06.23.661096  
**PDF**: `szafranski2025_biorxiv_peri-implant-ecosystem.pdf`  
**ステータス**: bioRxiv preprint (2025-06-25), 査読前  
**著者**: Szafrański SP, Joshi AA et al. — MHH/NIFE, SFB/TRR-298-SIIRI

---

## データ公開状況 ⚠️

| データ種別 | 状態 |
|-----------|------|
| 生シーケンスデータ | ENA **PRJNA119296** — **出版後に公開予定**（現時点では非公開） |
| 生データ一般 | 対応著者へリクエスト（Szafranski.Szymon@mh-hannover.de） |
| STORMS チェックリスト | Zenodo: https://doi.org/10.5281/zenodo.15495741（現在アクセス可能） |
| Supplemental Tables (S1–S8) | 論文内記載、PDFには未掲載（別ファイル） |

> **結論**: 現時点でデータは公開されていない。出版後にENAから取得可能。Supplemental Tables は対応著者リクエストが必要。

---

## 研究概要

- **対象**: SIIRI Biofilm Implant Cohort (BIC) — 48人・125サンプル
- **診断群**: PIH (peri-implant health), PIM (peri-implant mucositis), PI (peri-implantitis)
- **手法**: full-length 16S rRNA (PacBio Sequel) + metatranscriptomics (Illumina HiSeq 2500)
- **データ規模**: 2.7億リード、うち細菌・ウイルス分 14億、EC注釈付き 3.5億

---

## 主要な数値・結果

### コホート構成
| 診断 | サンプル数 | 主な臨床特徴 |
|------|-----------|------------|
| PIH  | — | 健康 |
| PIM  | — | プラーク↑、粘膜炎 |
| PI   | 32 | 骨吸収、膿形成、疼痛、高プロービング深さ |

### バイオマス
| 診断 | 細菌RNA量（中央値） |
|------|-------------------|
| PIH  | 1 ng              |
| PIM  | 10 ng (P=0.0096 vs PIH) |
| PI   | 50 ng (P=2.1×10⁻⁶ vs PIH) |

> 4桁の差（0.05 ng〜1,500 ng）

### 菌叢組成
- **99 genus / 756 species-level phylotypes** を同定
- Shannon多様性 H': PIH 3.1±0.5 → PI 3.4±0.5 (P=0.02)
- 未命名種 (HMT番号付き): 325種中228種を検出、平均9.3%を占める
- 最優占未命名種: *Fusobacterium* sp. HMT-203

| 診断 | 優占クラス | 代表菌属 |
|------|-----------|---------|
| PIH  | Bacilli, Actinobacteria, Gammaproteobacteria | *Streptococcus* |
| PIM  | 同上 + Betaproteobacteria（CT II） | *Streptococcus*, *Shuttleworthia*↑, *Neisseria* |
| PI   | Bacteroidia, Fusobacteriia（嫌気性優占） | *Dialister*, *Prevotella*, *Porphyromonas*, *Tannerella*, *Parvimonas* |

---

## 4つのCommunity Types (CT)

SIMPROF法（置換検定）による有意クラスタリング

| CT | 診断関連 | 主な活性クラス | 代謝特徴 |
|----|---------|--------------|---------|
| CT I  | PIH  | Bacilli, Actinobacteria | 炭水化物代謝, ペントースリン酸 |
| CT II | PIM  | Betaproteobacteria (*Neisseria*) | ピルビン酸・リポ酸代謝 |
| CT III| PIM  | Bacteroidia (*Prevotella*), Fusobacteriia | リシン分解, ブタン酸代謝 |
| CT IV | PI   | Bacteroidia (*Tannerella*, *Porphyromonas*), Fusobacteriia | 異化亢進, 毒性因子 |

---

## 転写活性（機能）

- 同定EC数: 1,824種（うち36%が相対存在量 ≥0.1%）
- RNAseq 総リード: 2.6億 (98サンプル, 平均2,600万/サンプル)
- PI vs PIH/PIM: EC発現パターン有意差 (PERMANOVA P=0.0001)
- PIH vs PIM: 有意差なし (P=0.488) → PIHからPIMは「機能変化を伴わないバイオマス増大」

### PIに関連する代謝経路（EC level）
**亢進（PI ↑）**: アミノ酸分解（多種）, アシルグリセロール分解, ガラクトース分解,
イソプレノイド生合成, プリン生合成, セリン生合成, 炭素固定, 糖新生

---

## ファージ（phageome）
- 細菌組成の変化を反映したファージプロファイルの変化
- 診断特異的ファージ: *Streptococcus*, *Neisseria*, *Prevotella*, *Porphyromonas*, *Fusobacterium* 感染ファージが5大グループ

---

## 宿主応答（host transcriptome）
| 診断 | 主要経路 |
|------|---------|
| PIH  | ケラチン化 (keratinization) |
| PIM  | リボソーム構成要素↑ |
| PI   | 炎症シグナル, 低酸素応答 (NF-κB, TNF-α, hypoxia) |

---

## Figure 一覧

| Fig | 内容 | データ種別 |
|-----|------|----------|
| 1 | 研究デザイン + PI biofilm 電顕写真 | SEM |
| 2 | 種レベル組成ヒートマップ（CAP選択taxa） | full-16S |
| 3 | 4つのCommunity Types (CT I-IV) + 種共起ネットワーク | MTX-B |
| 4 | KEGG pathways × 診断群（EC数バープロット） | MTX-B |
| 5 | クラス別酵素活性（生態的役割ヒートマップ） | MTX-B |
| 6 | Phageome × 診断群 | MTX-P |
| 7 | 宿主遺伝子発現経路（GSEA） | MTX-H |
| 8 | 統合ネットワーク（菌種-EC, ファージ-宿主菌, 菌種-宿主） | all omics |

---

## 自分の研究との関連

### Hamilton ODEモデルへの対応
| 本研究 | Hamilton/TMCMCモデル |
|--------|---------------------|
| PIH (CT I) | commensal_static / commensal_hobic |
| PI (CT IV) | dysbiotic_static |
| *Porphyromonas* (Pg) + *Tannerella* (Tf) 優占 | Pg が dysbiotic 条件のキープレイヤー |
| バイオマス 50倍差 (PIH→PI) | DI 高値 (0.85) ← dysbiotic attractor |
| 嫌気性優占 + EPS・小胞構造 | DI = 物理的剛性指標として妥当 |

### 課題・今後の利用
- Supplemental Tables (S3-S5: 相対存在量, S8: クラス別EC) が入手できれば Hamilton ODE の初期値・パラメータ根拠として使用可能
- データ公開後に ENA PRJNA119296 からMTXリードを取得 → MetaPhlAn/KEGG解析との比較
- CTクラスタリングは「4条件 (CS/CH/DH/DS)」分類の独立した臨床的根拠になりうる

---

## 引用形式
```
Szafrański SP, Joshi AA, et al. High-resolution taxonomic profiling and metatranscriptomics 
identify microbial, biochemical, host and ecological factors in peri-implant disease. 
bioRxiv. 2025. https://doi.org/10.1101/2025.06.23.661096
```
