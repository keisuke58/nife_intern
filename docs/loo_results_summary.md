# LOO-CV Results Summary
**Date**: 2026-05-07  
**Dataset**: Dieckow et al. (10 patients, 3 timepoints, 10 guilds)  
**Task**: Predict weeks 2 & 3 from week 1 (leave-one-patient-out cross-validation)

---

## Models tested

### Hamilton replicator model
- A matrix is **symmetric** (A[i,j] = A[j,i]) — upper triangle only
- Sign prior symmetrized: (net + net.T) / 2
- Constrained pairs: **14 undirected pairs** (α=0, no AGORA) → **35 pairs** (+AGORA)

### gLV replicator model
- A matrix is **asymmetric** (full N×N)
- Sign prior directed: net_flow_glv (not symmetrized)
- Constrained elements: **21** (α=0, no AGORA)

### Sign prior layers
| Layer | Source | Weight |
|-------|--------|--------|
| L1 | Szafranski Suppl. File 1, experimental + KEGG/HMDB | 1.5–2.0 |
| L2 | Szafranski Suppl. File 1, prediction evidence | 1.0 |
| L3 | AGORA2 FBA (oral-fluid medium, pFBA, 10 guild reps) | 0.5 |

### Competition weight α
Exploitative competition term: when guilds i, j both consume metabolite m,  
`neg[i,j] += w × α`  
Environmental metabolites (O₂, CO₂, H₂O₂) excluded.

---

## LOO-RMSE Results (10-fold, N=10 patients)

| Model | α | AGORA | N folds | mean RMSE | std | min | max |
|-------|---|-------|---------|-----------|-----|-----|-----|
| Hamilton | 0.00 | — | 10 | **0.0567** | 0.0267 | 0.0078 | 0.1153 |
| Hamilton | 0.25 | — | 10 | 0.0615 | 0.0247 | 0.0321 | 0.1172 |
| Hamilton | 0.50 | — | 10 | 0.0614 | 0.0247 | 0.0310 | 0.1151 |
| Hamilton | 0.00 | ✓  | 10 | **0.0562** | 0.0196 | 0.0314 | 0.1023 |
| gLV | 0.00 | — | 10 | 0.0501 | 0.0161 | 0.0201 | 0.0797 |
| gLV | 0.25 | — | 10 | **0.0490** | 0.0159 | 0.0195 | 0.0799 |
| gLV | 0.50 | — | 10 | 0.0494 | 0.0173 | 0.0189 | 0.0781 |

*(gLV + AGORA and no-prior baseline: jobs pending)*

---

## Key findings

### 1. gLV outperforms Hamilton
- Best gLV (α=0.25): mean RMSE = **0.0490**
- Best Hamilton (α=0): mean RMSE = **0.0567**
- Difference: **−13.5%** in mean LOO-RMSE

Likely reason: Hamilton constrains A to be symmetric (A[i,j]=A[j,i]), which is a strong structural assumption. gLV allows directed interactions (e.g., Fusobacterium benefits Porphyromonas unidirectionally), which is more realistic for oral biofilm ecology.

### 2. Competition weight α: opposite effects in the two models
- **Hamilton**: α=0 is best; adding competition (α>0) increases RMSE
- **gLV**: α=0.25 is best; a small competition term slightly improves prediction

Interpretation: In Hamilton's symmetric A, competition (which is inherently asymmetric in ecology) introduces conflicting constraints after symmetrization. In gLV, the directed competition signal is ecologically appropriate.

### 3. AGORA marginally improves Hamilton (α=0)
- Without AGORA: 0.0567
- With AGORA: 0.0562 (−0.9%, within noise)
- AGORA greatly increases constrained pairs: 14 → 35 undirected pairs
- Sign agreement remains high: 66–70/70 (94–100%)

The improvement is very small, consistent with the known limitations of single-strain FBA representatives and individual (non-community) simulations.

### 4. Sign agreement is high across all configurations
- Hamilton (no AGORA): 79–100% agreement per fold
- Hamilton + AGORA: 94–100% agreement (more pairs constrained)
- gLV: 90–100% agreement

The sign prior is consistent with the fitted A matrices — the prior does not conflict with the data.

---

## Sign prior: constrained pairs (Hamilton, α=0, no AGORA)

| Guild i | Guild j | net_flow (sym) | sign |
|---------|---------|----------------|------|
| Bacilli | Negativicutes | +4.0 | mutualism |
| Betaproteobacteria | Negativicutes | +6.0 | mutualism |
| Actinobacteria | Negativicutes | +5.5 | mutualism |
| Bacteroidia | Negativicutes | +3.5 | mutualism |
| Fusobacteriia | Negativicutes | +2.0 | mutualism |
| Bacilli | Actinobacteria | +4.0 | mutualism |
| Fusobacteriia | Bacilli | +2.0 | mutualism |
| Betaproteobacteria | Actinobacteria | +4.0 | mutualism |
| Betaproteobacteria | Bacilli | +4.0 | mutualism |
| Betaproteobacteria | Bacteroidia | +1.5 | mutualism |
| Actinobacteria | Bacilli | +2.0 | mutualism |
| Bacteroidia | Bacilli | +2.0 | mutualism |
| Fusobacteriia | Negativicutes | +2.0 | mutualism |
| Actinobacteria | Bacteroidia | — | (net cancels) |

*All constrained pairs are mutualistic (positive) — no antagonism survives symmetrization in the Hamilton model.*  
*This is because amensalism (+/−) cancels to 0 when symmetrized: (net + net.T)/2 = 0 for asymmetric pairs.*

---

## Pending results (2026-05-19 update)

| ジョブ | ID | 内容 | 状態 |
|--------|-----|------|------|
| loo_micom (v0) | 40153–40162 | MICOM fraction=0.3, binary | 実行中 |
| compare_signs | 40163 | v1/v2/MICOM-perfect sign agreement | 実行中 |
| loo_micom_perf | 40164–40173 | MICOM fraction=0.5, flux-weighted, toxin→all | 実行中 |

結果が揃い次第 `collect_loo_results.py` で集計。  
比較対象: v1 baseline (mean=0.0562), no-prior (mean=0.0595)

---

## Figure
`results/dieckow_cr/loo_summary_szafranski.png`

Left: per-patient RMSE for best configurations  
Right: mean ± std across all configurations
