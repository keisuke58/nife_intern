# Soleimani et al. 2023 — 調査メモ
*作成: 2026-06-23 / 修論§5.2 validation data 調査*

**タイトル**: Numerical and experimental investigation of multi-species bacterial co-aggregation  
**著者**: Meisam Soleimani, Szymon P. Szafrański, Taoran Qu, Rumjhum Mukherjee, Meike Stiesch, Peter Wriggers, Philipp Junker  
**誌**: Scientific Reports (Nature Publishing Group) 2023, 13:11839  
**DOI**: 10.1038/s41598-023-38806-2  
**ライセンス**: CC BY 4.0  
**資金**: DFG SIIRI project TRR298 (No. 426335750), subproject B07  
**ローカルPDF**: `../../../../../IKM_Hiwi/Tmcmc202601/docs/soleimani_papers/soleimani2023_coaggregation_scirep.pdf`

> **著者ネットワーク**: Szafrański・Mukherjee・Stiesch・Junkerは西岡論文の共著者と完全一致。Soleimaniは修論の主査。同一グループの直系先行研究。

---

## 1. モデル概要

**4変数 Allen-Cahn 型 phase-field + ANSYS FEM**

| 変数 | 物理量 | DOF |
|---|---|---|
| φ1, φ2 | 菌種1・2の密度（体積分率） | 節点 |
| α | 共凝集度（co-aggregation） | 節点 |
| c | 栄養濃度 | 節点 |

→ **4 DOF/node**（Klempt 2024の5 DOF/nodeの前身）

### 支配方程式（Allen-Cahn型）

```
φ̇ = −f'(φ) + ε²∇²φ + S(φ, c)

S1(φ1, α, c) = Rs1 · H(φ1 − φcri) · (1 + τα) · φ1 · c
S2(φ2, α, c) = Rs2 · H(φ2 − φcri) · (1 + τα) · φ2 · c
Sα(φ1,φ2,α,c) = Rα · H(φ1−φcri) · H(φ2−φcri) · ∇α · ∇c/|∇c|
Sc(φ1,φ2,c) = −Rc(φ1 + φ2)c
```

**ポイント**: co-aggregation項 `(1 + τα)` が共存域で成長を加速。
Sαは栄養勾配方向へのco-aggregation境界の移流。

---

## 2. モデルパラメータ（Table 1、全値）

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| 栄養拡散率 | Dn | 10⁻⁶ | µm² Time⁻¹ |
| 菌密度閾値 | φcri | 0.1 | µg µm⁻³ |
| 消費速度 | Rc | 50 | µm³ µg⁻¹ Time⁻¹ |
| 共凝集係数 | Rα | 15 | µm Time⁻¹ |
| 共凝集増強 | τ | 1 | − |
| 種1成長係数 | Rs1 | 500 | µm³ µg⁻¹ Time⁻¹ |
| 種2成長係数 | Rs2 | 500 | µm³ µg⁻¹ Time⁻¹ |
| α用 phase-field係数 | M (α) | 10⁻⁴ | Time⁻¹ |
| φ/c用 | M (φ, c) | 0 | Time⁻¹ |
| φ拡散率 | ε²=Db | 10⁻⁸ | µm² Time⁻¹ |
| 計算ドメイン | − | 100×100 | µm (2D) |
| メッシュサイズ | − | 0.5 | µm |

→ 2D: ~2×10⁴ nodes / 8×10⁴ DOF。3D同メッシュ: >10⁷ DOF → 計算困難（論文内明記）

---

## 3. FEM実装

- **AceGen（Mathematica）** → 自動微分 → FORTRAN UEL → ANSYS（Klempt 2024と同一パイプライン）
- 完全陰解法（backward Euler）モノリシック
- コード: 対応著者への要求 or LUH公開リポジトリで入手可

---

## 4. 実験設定（CLSM・CLSMデータ）

### 使用菌株
- *Eikenella corrodens* SPS_010（ペリインプラント炎由来）
- *Streptococcus gordonii* SPS_017, SPS_886（同症例由来）
- *Candida albicans* SPS_888, SPS_889, SPS_887（ATCC標準株 + 臨床分離）

### プロトコル
- 培地: THBY（Todd-Hewitt Broth + 0.3% Yeast extract）
- 前培養: 24h, 37°C, 5% CO2
- Co-culture: 等量混合 → OD600=1.0（~10⁹ cells/ml）→ vortex 10秒 → 10倍希釈
- バイオフィルム形成: 6-wellプレート, 1h コーティング → 3ml THBY → **8h成長**
- 染色: LIVE/DEAD BacLight（SYTO9 + PI）

### CLSM設定
- 装置: Leica TCS SP2
- 488nm laser; SYTO9: 500-545nm（緑=生細胞）; PI: 590-680nm（赤=死細胞）
- **光学切片: 3 µm**
- 各wellから代表画像5枚取得
- 実験反復: 2回 × 2 wells × 5画像 = **計10画像/group**
- 解析: Imaris 6.2.1（3D再構成、緑/赤/黄フラクション算出）

### 定量データ（⚠️論文内は定性比較のみ・数値表なし）

論文中に数値化されたフラクション値は記載なし。図のみ。
- Fig. 6(d,e): モノカルチャー vs コカルチャーのCLSM像（スケールバー 15µm）
- 「コカルチャーでより大きいaggregateが観察された」（定性）

---

## 5. 主要知見

| 知見 | 定量値 |
|---|---|
| rD小（=Dn/Db小）→ 拡散制限 → 樹枝状（dendrite）形態 | rD = 10² |
| rD大 → 拡散非制限 → 球状コンパクト形態 | rD = 10⁶ |
| co-aggregation有無でバイオフィルム発達速度が**大幅に異なる** | Fig 10-12（定性） |
| 連続栄養供給あり: 平均栄養濃度が一旦上昇後→定常値（消費=供給）| Fig 10 |
| co-aggregation有: 定常到達が速い（消費能力の早期獲得） | Fig 10 |

---

## 6. 修論§5.2との接続

### Soleimani 2023 → Klempt 2024 → 本修論の系譜

```
Soleimani 2023: 2種, Allen-Cahn, 4DOF/node, ANSYS UEL, 定性検証のみ
      ↓ 単種化 + 力学結合追加
Klempt 2024: 1種, Hamilton原理, 5DOF/node (u追加), 定性検証のみ
      ↓ 多種化 + novel interaction scheme
Klempt 2025: N種, Hamilton原理, 定性検証のみ
      ↓ 本修論
本修論: TMCMC推定済みA行列 + Hamilton/gLV 2軸 + FISH定量検証（初）
```

### Validationデータとして使えるか？

| 観点 | 評価 |
|---|---|
| 菌種 | So/Gordonii ✅、Ec/Candida ≠ 5種系 ❌ |
| 空間分布データ | CLSMは形態画像のみ・φ_i(z)深さプロファイルなし ❌ |
| 定量値 | 論文に数値表なし・図のみ ❌ |
| 入手可能性 | 対応著者=Soleimani（修論主査）に直接依頼可 ✅ |

**結論**: Soleimani 2023の実験データ自体は直接の定量validationには使いにくい。ただしFelixへの確認事項に「このパイプラインで生成された未公開CLSMデータがあるか」を追加するのが有効。

---

## 7. ⭐ Nature強度について

Scientific Reports = Nature Publishing Group のオープンアクセス誌。
- Impact Factor ~4.6（2023年）
- 同グループ（IKM + MHH SIIRI TRR298）の旗艦ジャーナル論文
- DFG TRR298 B07資金 → 修論もこの資金枠内の継続研究

→ 修論の「先行研究」節でこの系譜（Soleimani 2023 → Klempt 2024 → Klempt 2025 → 本研究）を明示することで、**Nature誌系の先行研究に直接繋がるラインに本修論を位置づけられる**。
