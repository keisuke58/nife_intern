# Dieckow 2025 in-vivo 解析まとめ

**データ**: Dieckow et al. 2025, 10患者×3週間 peri-implant biofilm (PRJEB71108)  
**手法**: 共有 A 行列 (15パラメータ) + 患者別成長率 b^(p) (5×10=50パラメータ, 計 d=65) を joint 推定  
**ODE**: Hamilton 0D, 5菌種 (So, An, Vd, Fn, Pg)  
**観測**: 3時点 (week 1→2→3), 10患者

---

## 1. 推定精度 (RMSE: week2+3 予測)

| 手法 | N_p | Prior | RMSE |
|------|-----|-------|------|
| TMCMC joint | 500 | flat | 0.1068 |
| TMCMC joint | 1000 | flat | **0.1006** |
| TMCMC joint | 1000 | Bergey sign | **0.1008** |
| Adam per-patient (A+b) | — | L2 | 0.1116 |
| Baseline (MAP θ のみ) | — | — | 0.1134 |

- Sign prior は RMSE をほぼ変えないが、相互作用の符号を生物学的に整合させる
- Adam per-patient 最適化は TMCMC より若干劣るが、baseline は改善

---

## 2. 共有 A 行列 MAP (N_p=1000, flat prior)

```
         So      An      Vd      Fn      Pg
So    3.651   3.622   1.660  -1.074  -2.687
An    3.622   2.473  -2.988   4.179  -3.485
Vd    1.660  -2.988  -0.417   0.805   1.636
Fn   -1.074   4.179   0.805  -1.263   3.663
Pg   -2.687  -3.485   1.636   3.663  -1.993
```

### 注目パラメータ

| ペア | 値 | 解釈 |
|------|----|------|
| a_So,An | +3.62 | 強い相互促進（co-aggregation） |
| a_An,Vd | −2.99 | **Vd 競合** ← in-vitro では正 (V.parvula 株差) |
| a_So,Vd | +1.66 | 正 (Heine と一致) |
| a_Fn,Pg | +3.66 | 強い cooperative cross-feeding |
| a_So,Pg | −2.69 | 阻害 (W83 ではなく in-vivo 環境効果?) |

### Bergey sign prior による符号変化

| パラメータ | flat prior | sign prior | 変化理由 |
|-----------|-----------|------------|---------|
| a_Vd,Vd | −0.417 | +0.835 | self-growth 符号修正 |
| a_Fn,Fn | −1.263 | +0.244 | self-growth 符号修正 |
| a_An,Vd | −2.988 | − | **変わらず**（データ信号が強い） |

---

## 3. 患者別成長率 b̂ (N_p=1000, flat prior)

```
患者     So      An      Vd      Fn      Pg
A      7.36    6.55    4.62    8.39   11.90
B      3.84    9.34    2.10    2.34    9.62
C      6.06    0.48    6.98    7.21   13.76
D      4.53    8.32   10.46    4.92   11.15
E      8.54    2.31    7.85    4.65   13.39
F     10.99    0.58   13.52   13.31   11.39
G      0.84    0.81    7.13    7.42    7.54
H      6.72   14.21   10.64   11.17    5.21
K      9.45   12.16    4.73   12.90   10.78
L      3.59    4.74    5.00   12.44    8.32
平均    6.19    5.95    7.30    8.47   10.31
```

- Pg の b̂ は全患者で高め (平均 10.3) → in-vivo では Pg が増殖有利
- 患者 F, H は An/Fn が特異的に高い

---

## 4. In-vitro (Heine) との sign 比較

5条件 (CS, CH, DS, DH + Dieckow in-vivo) で一致した符号：

| ペア | 全5条件で一致 | 備考 |
|------|-------------|------|
| a_So,An | ✓ (+) 5/5 | robust mutualism |
| a_An,Vd | ✓ (+) 4/5 Heine, **−1/5 Dieckow** | V.dispar vs V.parvula |
| a_Vd,Pg | ✓ (+) 5/5 | pH facilitation |
| a_So,Pg | + in 4条件, **− in DH のみ** | W83 株特異的阻害 |

**主要な差異**: `a_An,Vd` が in-vivo で負 → V.dispar は V.parvula と代謝プロファイルが異なり An との cross-feeding が成立しない可能性

---

## 5. 出力ファイル

| ファイル | 内容 |
|---------|------|
| `dieckow_fits/fit_joint_5sp_1000p.json` | TMCMC MAP + samples (flat prior) |
| `dieckow_fits/fit_joint_5sp_1000p_meta.json` | TMCMC MAP + samples (sign prior) |
| `Ab_dieckow_timeseries/A_hat.npy` | Adam per-patient A行列 (10×15) |
| `Ab_dieckow_timeseries/b_hat.npy` | Adam per-patient b (10×5) |
| `Ab_dieckow_timeseries/summary.json` | Adam RMSE summary |
| `Ab_dieckow_timeseries/Ab_timeseries_analysis.png` | 可視化 |

---

## 6. 今後の方針

- **Cross-prediction**: 4 Heine MAP を Dieckow データに適用して RMSE 比較
- **n_stages=10** (TMCMC)は多い → 同定性確認のため posterior samples の有効サンプル数チェック
- **Patient F** の b̂ 異常 (So=11, Vd=13) → Adam 最適化で発散; TMCMC も確認要
- **PDE 拡張**: Dieckow の CLSM 空間データ (week1-3 volume/coverage) との接続
