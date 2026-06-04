# Szafrański データ活用計画 (会話まとめ 2026-04-16)

## 1. MDI/eMDI 実装 (完了)

`nife/compute_di_szafranski.py` を完全書き直し。Hamilton DI (1−H/H_max) は廃止。

### なぜ Hamilton DI を却下したか
- `DI = 1 - H/H_max` はShannon エントロピーベース
- Streptococcus 優占 (健康) と Porphyromonas 優占 (病的) が同じ高DI値になる
- **誰が** 優占しているか分からない → 臨床意味なし

### MDI/eMDI (Joshi 2026 JDR)
```
MDI = log( Σ abundance(PD-positive genera) / Σ abundance(PD-negative genera) )

Positive (↑ with PI): Pseudoramibacter
Negative (↓ with PI): Capnocytophaga, Gemella

eMDI = MDI + 9 EC features (metatranscriptomics)
  Positive EC: 3.4.21.102, 3.4.23.36, 3.4.21.53 (endopeptidases)
  Negative EC: 5.4.99.9, 4.1.2.40, 5.3.1.26, 2.7.1.144, 3.2.1.20, 1.12.99.6
  R²=0.51 vs probing depth, AUC=0.87
```

### Demo 結果 (mock data)
| 診断 | MDI median |
|------|-----------|
| PIH  | −7.7 |
| PIM  | −1.9 |
| PI   | +2.8 |

---

## 2. Szafrański データの性質

### コホート構成
- SIIRI Biofilm Implant Cohort (BIC)
- 48人・125サンプル (1人あたり平均 2.6 サンプル → 複数インプラント)
- **横断研究 (cross-sectional)** — 各患者は一度のみ採取、時系列なし

### Szafrański は MDI を計算していない
- Szafrański 2025: 菌叢・ECの全解析を実施
- Joshi 2026 JDR: MDI/eMDI を別途開発・バリデーション
- → **自分の貢献**: Szafrański の 125 サンプルに MDI/eMDI を適用

---

## 3. 過去の作業との関連

### 強い関連

**CT I–IV ↔ 4 TMCMC 条件マッピング**
```
Szafrański CT     →  TMCMC 条件
CT I  (PIH)       →  commensal_static (CS)
CT II (PIM)       →  commensal_hobic (CH)
CT III(PIM)       →  dh_baseline (DH)
CT IV (PI)        →  dysbiotic_static (DS)
```

**S5 菌種存在量 → Hamilton ODE 初期値**
- 5種 (So, An, Vd, Fn, Pg) を抽出して φ₀ に使用
- ODE定常状態が PIH/PI と一致するか検証

**MDI per CT**
- CT IV → MDI 高値、CT I → MDI 低値 の確認

### 中程度の関連

**S8 EC活性 ↔ Hamilton 代謝予測**
- EC 実測 vs ODE の Monod フラックス 定性比較

**Probing depth ↔ MDI 相関**
- Joshi の R²=0.51 を独立再現できるか

### 弱い関連

**Shannon 多様性 H'**
- Szafrański: PIH 3.1 → PI 3.4 (差が小さい)
- → Hamilton DI 却下の根拠としても使える

---

## 4. ODEモデルで説明できるか

### 直接は不可能
- ODE 5種: So, An, Vd, Fn, Pg
- MDI に必要: Pseudoramibacter, Capnocytophaga, Gemella
- 種が違うので直接 MDI は計算できない

### 間接的に検証する方法
```
Szafrański 5種存在量 (φ₀)
    ↓ Hamilton ODE 定常状態まで積分
    ↓ dysbiotic/commensal attractor 判定
        ↕ 比較
    観測 MDI 高値/低値
```

### 疑似時系列 (pseudo-trajectory) として使う
```
PIH → PIM → PI (横断データ)
= disease progression の 3 ステージ近似
→ ODE の attractor 遷移 (CS → DS) と対応
```

---

## 5. Heine 2025 vs Szafrański 2025

| | Heine 2025 | Szafrański 2025 |
|--|------------|----------------|
| 環境 | in vitro (Ti disc, BHI) | in vivo (患者口腔内) |
| 種数 | 5種のみ | 756種 (99属) |
| 時系列 | あり (Day 1,3,6,10,15,21) | なし (横断) |
| サンプル数 | N=5〜15/条件 | N=125 (48人) |
| 条件制御 | 完全制御 (CS/CH/DS/DH) | 非制御 (自然な病態) |
| データ種別 | 種%・pH・gingipain・体積 | 16S + metatranscriptomics |
| 自分の研究での役割 | TMCMC キャリブレーション | 臨床バリデーション |

### パイプライン位置づけ
```
Heine 2025 (in vitro)
    → 5種ODE パラメータ推定 (TMCMC)
         ↓
Szafrański 2025 (in vivo)
    → ODEの予測が実臨床で成立するか検証
    → MDI計算で健康→病態スペクトラム確認
```

---

## 6. 今後のアクション

### STEP 1 — データをもらう (2026-04-17, メール)
Dr. Szafrański (Szafranski.Szymon@mh-hannover.de) にリクエスト:
- **S5**: 125サンプルの菌種相対存在量 (必須)
- **S8**: クラス別EC活性 (あれば)
- Dr. Joshi に: サンプルごとの eMDI 値 (あれば)

---

### STEP 2 — MDI 計算 (S5 入手直後、1日で完了)
```bash
python nife/compute_di_szafranski.py --s5 s5.csv --out nife/results/
```
**得られる結果**:
- PIH / PIM / PI ごとの MDI 分布図
- `mdi_szafranski_results.csv` (サンプルごとの MDI 値)

---

### STEP 3 — 臨床変数との相関 (S5 に臨床データがあれば)
- MDI vs probing depth → Joshi の R²=0.51 を独立再現できるか
- MDI per CT (CT I〜IV) の分布確認

---

### STEP 4 — ODE 検証 (発展、時間があれば)
S5 から 5種 (So, An, Vd, Fn, Pg) の存在量を抽出 → φ₀ として Hamilton ODE に入力
```
125サンプルの φ₀ → ODE定常状態 → dysbiotic/commensal attractor 判定
                                         ↕ 比較
                         診断 (PIH/PIM/PI) または MDI
```
一致率が高ければ「ODE が臨床データを再現できる」という検証になる。
