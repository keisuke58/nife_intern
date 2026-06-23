# Klempt Multi-Species FEM — 残穴ロードマップ (2026-06-24)

TMCMC 5種 → Klempt PDE (Eq.34-36) → conformal FEM パイプラインの残穴を優先順位付きで列挙。
修論 §5.2 に向けて徐々に潰す。

---

## ✅ 完了済み (参照用)

| 穴 | 内容 | コミット |
|---|---|---|
| PDE 形式 | Klempt Eq.34 完全形 (Allen-Cahn+logistic+k_α·α+chemotaxis) | PDE v2 |
| ϕ²-gated E | Klempt Eq.20: E_eff = (ϕ_local/ϕ_total)² × E_Voigt | nife `2c782fe` |
| Eq.36 整合 | α̇ = k_α·φ, Monod なし (Felix exact) | PDE v2 |
| 多種 UMAT (mode C) | 5種 α_s 個別 (Klempt 2025 additive Mandel) | nife `2c782fe` |
| Voigt E | E = Σφ_i E_i (条件依存) | nife `2c782fe` |
| k_α,eff 重み | Σφ_i k_α_i (TMCMC 10000p MAP から) | PDE v2 |
| インプラント geometry | nife に 3D helical screw FEM 完成 (旧 UMAT) | nife existing |
| RANKL/OPG モデル | 骨リモデリングスタンドアロン実装 | extensions/ |

---

## 🔴 Tier A — Abaqus 結果の検証 (QXT 解放待ち)

### A1: 歯 FEM mode A PASS 確認

**状態:** QXT/50 が frontale02 の COPV-GW 明示 3 ジョブ (d077/d078/d079) に占有。自動スタート待ち。

**確認コマンド (次セッション冒頭):**
```bash
cat /home/nishioka/IKM_Hiwi/FEM/tooth_klempt_results_A.csv
cat /home/nishioka/IKM_Hiwi/FEM/tooth_klempt_results_C.csv
```

**確認ポイント:**
- 全 4 条件が `PASS` になっているか
- SDV4 (E_gated) が深さ=0 (inner face) で E_Voigt に近く、深さ=1 (outer) で 0 に近いか
- SDV5 (phi_gate) が deep → shallow で単調減少するか
- σ_max (CH) > σ_max (DH) になっているか (α 比 2.01× が応力に反映される)

**ERROR 時の診断:**
```bash
grep -A5 "\*\*\*ERROR" /home/nishioka/IKM_Hiwi/FEM/p23_klempt_A_commensal_static.msg | head -30
```
よくある原因: PREDEF 変数番号ミスマッチ (`DEPVAR` / `field variable` 宣言数と UMAT 内 PREDEF(N) のズレ)

---

## 🟠 Tier B — インプラント FEM への v2 UMAT 移植 (1–2 週)

### B1: gen_implant_umat_klempt_inp.py の作成

**問題:**
- 既存 `implant_DH_thread.sta`, `implantphi_DH_thread.sta` → `umat_growth_phi` (旧) / socket UMAT を使用
- ϕ²-gated v2 UMAT (`umat_klempt_voigt.f`, `umat_klempt2025.f`) はまだインプラント mesh に適用されていない

**やること:**
1. `gen_tooth_klempt_umat_inp.py` の `write_mode_A()` / `write_mode_C()` を参考に、  
   インプラント mesh (`gen_implant_umat_3d_inp.py` が生成する C3D8 screw+bone assembly) 向けの  
   `gen_implant_klempt_inp.py` を書く
2. PREDEF 割り当て:
   - mode A: PREDEF(1)=α_total(ramp), PREDEF(2..6)=φ_i, **PREDEF(7)=ϕ_local(depth)**
   - mode C: PREDEF(1..5)=α_s(ramp), PREDEF(6..10)=φ_i, **PREDEF(11)=ϕ_local(depth)**
3. `run_implant_klempt.sh` で 4条件直列実行

**phi_local の depth 計算 (implant):**
- tooth では `depth = z / H_tooth` (上から下)
- implant では `depth = r_biofilm / t_biofilm` (チタン表面からの厚さ方向、螺旋ねじ面から外側)
- `gen_implant_umat_3d_inp.py` の node 座標から距離を計算して PREDEF(7) / PREDEF(11) を割り当て

**期待される学術的価値:**
- 歯 (P1_Tooth_23) vs インプラント (Ti screw) で同じ UMAT v2 を使った応力比較
- Heine2025 の「チタン面上 5 種バイオフィルム」との整合

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
**Heine2025:** チタン面上 5 種バイオフィルムの成長・付着特性を報告  
**やること:** RAG で Heine2025 を再検索し、Ti 特有のパラメータを抽出して E_i または k_α_i を更新

### E3: σ(depth) 比較図 (修論用)

**やること:**
```python
# extract SDV4 (E_gated) and SDV1 (alpha) from odb → depth profile
# 4条件を1枚の図に: α(depth), E_gated(depth), σ11(depth)
python compare_tooth_klempt.py  # (未作成 → B1 後に作る)
```

---

## アクション順序

```
A1 → (QXT 解放後に自動実行中) → 結果確認
B1 → インプラント inp 生成 → Abaqus 実行
E3 → B1 の結果から図作成 → 修論 §5.2 図挿入
C1 → posterior UQ → 修論 §5.3
E1 → 骨吸収連成 → Outlook 節
D1 → (時間があれば) → Klempt 2025 完全実装
E2 → (最後) → Ti パラメータ更新
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
