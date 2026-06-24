# Klempt Multi-Species FEM — 残穴ロードマップ (2026-06-24, 更新 2026-06-25)

TMCMC 5種 → Klempt PDE (Eq.34-36) → conformal FEM パイプラインの残穴を優先順位付きで列挙。
修論 §5.2 に向けて徐々に潰す。

---

## ✅ 完了済み (参照用)

| 穴 | 内容 | コミット / 状態 |
|---|---|---|
| PDE 形式 | Klempt Eq.34 完全形 (Allen-Cahn+logistic+k_α·α+chemotaxis) | PDE v2 |
| ϕ²-gated E | Klempt Eq.20: E_eff = ϕ_local² × (E_Voigt/Σφ_i) | nife `954022c` (v3) |
| phi_gate バグ修正 | TMCMC Σφ_i < 1 でも phi_gate ≤ 1 を保証 (E_voigt_norm) | nife `954022c` |
| Eq.36 整合 | α̇ = k_α·φ, Monod なし (Felix exact) | PDE v2 |
| 多種 UMAT (mode C) | 5種 α_s 個別 (Klempt 2025 additive Mandel) | nife `2c782fe` |
| Voigt E | E = Σφ_i E_i (条件依存) | nife `2c782fe` |
| k_α,eff 重み | Σφ_i k_α_i (TMCMC 10000p MAP から) | PDE v2 |
| インプラント geometry | nife に 3D helical screw FEM 完成 (旧 UMAT) | nife existing |
| RANKL/OPG モデル | 骨リモデリングスタンドアロン実装 | extensions/ |
| **A1: 歯 FEM v3 PASS** | mode A 全4条件 PASS (CS=120s/CH=131s/DS=130s/DH=311s) | `tooth_klempt_results_A.csv` |
| **B1: インプラント FEM v3** | mode A 全4条件 PASS (CS=331s/CH=240s/DS=331s/DH=350s) | `implant_klempt_results_A.csv` |
| **E3: σ(depth) 比較図** | 3×2 figure (thesis_style, teal/orange) · σ^CH/σ^DH = 6.1×(tooth)/4.8×(implant) | `FEM/klempt_depth_comparison.pdf` |

**Felix 完全再現 (felix_complete_reproduction.py v4):** 全5基準 PASS ✅  
(Fig.2/Fig.5/p>0/Fig.7 layer/Fig.10 obstacle)

---

---

## 🟡 Tier C — 後験不確実性の伝播 (修論 §5.3 相当)

### C1: TMCMC posterior → 応力 CI

**問題:** 現在は 10000p posterior の MAP 点推定のみ。φ_i の不確実性が応力に与える影響を定量化していない。

**方針:**
1. `ultimate_10000p/{cond}/samples.npy` (shape 10000×20) から K=50 サンプルを等間隔抽出
2. 各サンプルで φ_i → k_α_eff → PDE → α → UMAT → .inp を生成 (スクリプト化)
3. frontale qsub で並列実行 (PBS job array)
4. 各条件の σ_max の分布 → 95% CI を図示

**難易度:** 中 (スクリプト繰り返し + frontale qsub job array)

**注意:** QSD/QXT ライセンスは厳密直列 → K=50 ジョブを job array で順次投入 (同時実行数=1)

---

## 🟡 Tier D — 空間 φ_i 分布 (発展)

### D1: 多種 PDE で φ_i(x,y) を空間場として計算

**問題:** 現在は TMCMC スカラー φ_i を全 Gauss 点に一様適用。実際には種ごとに空間分布が違う。

**方針 (Klempt 2025 準拠):**
- Klempt 2025 の A 行列 (5×5 相互作用) を使って multi-species PDE を解く
- 各条件の ultimate_10000p MAP の A 行列パラメータ (indices 0..14) を抽出
- `klempt_pde_multispecies.py` を 5 種それぞれの φ_i(x,y,t) に拡張
- PREDEF(2..6) を一様値ではなくノードごとの空間場として与える

**難易度:** 大 (PDE 拡張 + inp 生成の 5倍化)  
**優先度:** 低 (修論の新規貢献として Outlook に留めてもよい)

---

## 🟢 Tier E — 学術的仕上げ (修論完成直前)

### E1: 骨吸収との一方向連成

**現状:** `extensions/fig_periimplantitis_rankl_opg.py` — RANKL/OPG スタンドアロン実装済み  
**残り:** Abaqus 応力場 → 界面せん断 → 骨リモデリング入力への連成ループ

**Tier B1 が完了してから着手。**

### E2: Ti 表面パラメータ更新 (Heine2025)

**現状:** E_i, k_α_i などはエナメル質ベース (文献値)

**Heine 2025 確認済み (2026-06-25):**  
- *Front. Oral Health* 6:1649419 — peri-implant biofilm dysbiosis in vitro  
- **HOBIC 条件 = Ti grade 4 フローチャンバー上** → CH/DH の phi_i は Ti 表面の実験から推定済 ✅  
- **Static 条件 = ポリスチレン** → CS/DS は Ti 非特異 ⚠️  
- 5 菌種 (So/An/Vd/Fn/Pg) は Heine 2025 実験菌種と完全一致。追加種不要。  
- DS の TMCMC posterior (phi_So=0.944) は Heine Fig.3 の実験 (Vp=50-60% dominant) と乖離 → フィッティング問題として thesis に明示

**thesis での対応:** CH/DH は Ti 対応・CS/DS は参照のみと明記。E2 としての「パラメータ更新」は追加作業不要（HOBIC で既に Ti 上データ使用）。今後 Future Work として Ti 特異的 E_i 文献値への更新を記述。

### E3: σ(depth) 比較図 (修論用)

**やること:**
```python
# extract SDV4 (E_gated) and SDV1 (alpha) from odb → depth profile
# 4条件を1枚の図に: α(depth), E_gated(depth), σ11(depth)
python compare_tooth_klempt.py  # (未作成 → B1 後に作る)
```

---

## アクション順序 (2026-06-25 更新)

```
✅ A1 完了 → 歯 FEM v3 全4条件 PASS
✅ B1 完了 → インプラット FEM v3 全4条件 PASS
✅ E3 完了 → σ(depth) 図 klempt_depth_comparison.pdf (thesis_style)
✅ E2 確認済 → HOBIC = Ti grade 4 データ → CH/DH は Ti 対応 (Static は非Ti)
C1 → posterior UQ (K=50) → 修論 §5.3        ← 次のアクション
E1 → 骨吸収連成 → Outlook 節
D1 → (時間があれば) → Klempt 2025 完全実装
```

---

## ファイル参照

| ファイル | 内容 |
|---|---|
| `FEM/JAXFEM/klempt_pde_multispecies.py` | PDE v2 (Eq.34 完全形) |
| `FEM/gen_tooth_klempt_umat_inp.py` | 歯 FEM inp 生成 (mode A/C) |
| `FEM/run_tooth_klempt.sh` | 歯 FEM Abaqus 実行 |
| `coupling_prototype/abaqus/umat_klempt_voigt.f` | mode A UMAT v2 (ϕ²-gated) |
| `coupling_prototype/abaqus/umat_klempt2025.f` | mode C UMAT v2 (5-species ϕ²-gated) |
| `coupling_prototype/gen_implant_umat_3d_inp.py` | 3D インプラント inp 生成 (旧 socket UMAT) |
| `extensions/fig_periimplantitis_rankl_opg.py` | RANKL/OPG 骨吸収モデル |
| `Tmcmc202601/data_5species/_runs/ultimate_10000p/{cond}/samples.npy` | TMCMC 最終 posterior |
