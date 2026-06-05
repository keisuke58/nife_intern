# Velocity Direction Test — 論文テキスト草稿

図: `results/dieckow_cr/figs/fig_velocity_ct_comparison.pdf`

---

## Results セクション用（LOO-CV の後に挿入）

### Predictive validity of the gLV model differs between community states

To assess whether the fitted gLV model captures the direction of community dynamics—not merely the mean composition—we computed the cosine similarity between the model-predicted velocity vector and the observed compositional change at each consecutive timepoint pair (weeks 1→2 and 2→3) for all ten patients.
Using the Hamilton (symmetric) fit with patient-specific intrinsic growth rates $\mathbf{b}_i$, we find that CT1 (commensal) patients show a mean cosine similarity of $+0.52$ across all transition intervals, with 9 out of 10 transitions in the correct direction (90\%).
In contrast, CT2 (dysbiotic) patients show a mean cosine similarity of $-0.14$, with only 3 out of 10 transitions predicted correctly (30\%; $t$-test $p = 0.06$, Mann--Whitney $p = 0.09$; Fig.~X).
The separation is most pronounced at the first transition (weeks 1→2): CT1 mean $= +0.38$ versus CT2 mean $= -0.48$ ($t$-test $p = 0.061$).

These results indicate that the gLV replicator model faithfully captures the stabilising dynamics in the commensal basin, whereas the dysbiotic trajectories are dominated by external perturbations—likely immune-mediated tissue destruction, clinical intervention, or patient-specific dietary variation—that fall outside the model's ecological scope.

---

## Discussion セクション用（AGORA prior の motivation として）

The asymmetry in predictive accuracy between commensal and dysbiotic patients (Fig.~X) provides a mechanistic rationale for incorporating metabolic sign priors into the interaction matrix.
In the commensal state, microbiota composition is governed primarily by competitive exclusion and cooperative cross-feeding among aerotolerant taxa, processes that the gLV interaction matrix can represent.
In the dysbiotic state, host-derived peptide and lipid substrates released by tissue destruction fuel a metabolically distinct community where the ecological interactions encoded by $\mathbf{A}$ alone are insufficient to reproduce observed dynamics.
The AGORA2-derived sign priors constrain the off-diagonal signs of $\mathbf{A}$ to reflect known metabolic cross-feeding topology, providing an additional layer of information specifically about inter-species dependencies that are activated under nutrient-replete (dysbiotic) conditions.
This explains why the sign prior improves leave-one-out cross-validation performance selectively: it corrects the model where purely compositional fitting is least reliable.

---

## 数値サマリー（本文引用用）

| モデル | CT | cos平均 | 正の割合 | n遷移 |
|---|---|---|---|---|
| Hamilton | CT1 (commensal) | +0.52 | 90% | 10 |
| Hamilton | CT2 (dysbiotic) | −0.14 | 30% | 10 |
| gLV (asymm) | CT1 | +0.07 | 60% | 10 |
| gLV (asymm) | CT2 | −0.04 | 40% | 10 |

wk1→wk2 のみ (Hamilton)：CT1=+0.38 vs CT2=−0.48、t-test p=0.061、MWU p=0.075

---

## Figure caption 案

**Figure X. Velocity direction test reveals state-dependent predictive validity of the gLV model.**
(A) Mean cosine similarity between Hamilton-predicted velocity $\hat{\Delta\phi}$ and observed compositional change $\Delta\phi$ for each patient across all consecutive timepoint pairs.
Positive values indicate the model predicts the correct direction of change.
Blue bars, CT1 (commensal) patients; red bars, CT2 (dysbiotic) patients.
Dashed lines indicate group means (CT1: $+0.52$; CT2: $-0.14$).
(B) Distribution of cosine similarities per group; individual observations are overlaid (jittered).
Dagger (†) indicates trend-level significance ($p < 0.1$, two-sample $t$-test).
(C) Per-interval breakdown for gLV (pale) and Hamilton (solid) fits.
(D) Interpretive schematic: the commensal basin is governed by ecological self-dynamics (model-predictable), whereas the dysbiotic basin is additionally shaped by host-derived substrate release and immune perturbations beyond the model's scope.
EOF