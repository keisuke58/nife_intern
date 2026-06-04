# Research Plan: June – August 2026

**Goal**: Advance NIFE research to generate clean, thesis-worthy results.  
Thesis *writing* starts September. Between now and then: produce the results, not the prose.  
Criterion: **only clean results enter the thesis** — verify numbers against real data before any claim.

---

## Status snapshot (2026-06-04)

| Track | State | Thesis-ready? |
|---|---|---|
| 0D gLV LOO-RMSE (Dieckow) | 0.0490 (α=0.25) ✅ | Yes |
| Hamilton cross-feeding sign enrichment | 72.1%, p=0.0004, permtest ✅ | Yes |
| A-matrix prior asymmetry (health↑/dysbiosis↓) | kegg_sign_summary verified ✅ | Yes (corrected in §5.2) |
| Network analysis (centrality/trophic/rewiring) | slides done, figures exist | Need final pub-grade fig |
| DH diffusion fitting | floor ≈0.091, 8 jobs running ⏳ | Pending |
| CH diffusion fitting | loss 0.014–0.030, xdiff=D[An] pegged | Illustrative only |
| 3D FISH data findings | Fn-Pg z-separation, DH lateral homog, p=2e-4 ✅ | Yes (data pillar) |
| Botelho cross-cohort | 80–89% strong pairs, but fragile (n=9, p variable) | ⛔ exclude |
| Botelho trajectory prediction | fails (p=0.22) | ⛔ exclude |
| PINN 3D inverse (D_i GPU) | illustrative D=[.02,.01,.01,.01,.01] | Appendix/method only |

---

## Three pillars to push in June–August

---

### Pillar 1 — Strengthen the 0D ODE story (June)

The Hamilton symmetric model with AGORA cross-feeding signs independently validated is the **central result**. What is still missing to make it airtight:

#### 1-A — Neutral-init no-prior LOO (gold standard)
- Current no-prior LOO (`loo_expanded_noprior_a0p0`) started from the prior-fitted warm-start.  
  → Risk: warm-start imprints the answer.  
- **Action**: Submit jobs with random/zero init, `--no-prior --no-warmstart`, for all 10 folds (Hamilton expanded, α=0).  
- **Success criterion**: SA=72–78% and p<0.05 reproduce → warm-start contamination ruled out.  
- Script: extend `run_hamilton_expanded_loo.py --neutral-init` or new job `jobs/loo_noprior_neutralinit_submit.sh`.

#### 1-B — A-matrix stability across LOO folds
- Outline §2.5: 10 LOO estimates of A → per-pair std, CV, sign consistency.  
- Three regime classification: data+prior aligned / prior-constrained-muted / data-driven.  
- **Action**: `scripts/analysis/loo_stability_analysis.py` — read all 10 fold JSONs, compute across-fold std and sign consensus, produce heatmap.  
- Feeds directly into §5.1 (which interactions are robustly estimated).

#### 1-C — Prior asymmetry figure (health vs dysbiosis)
- kegg_sign_summary already verified: prior ↑ in CS (56→80%), CH (25→81%), ↓ in DS (88→44%), DH (53→44%).  
- **Action**: One clean bar chart (prior-on vs prior-off per attractor) — `scripts/figures/fig_prior_asymmetry.py`.  
- This is the thesis §5.2 figure replacing the corrected "100%/75%" claim.

#### 1-D — Network analysis final figure
- `scripts/figures/generate_fig7_Aij_Fij_scatter.py` exists (`results/fig7_Aij_Fij_scatter.pdf` modified in git status).  
- Need: concordant backbone, CS↔DH rewiring, trophic level bar — confirm these are in pub-grade format (usetex/lmodern, thesis_style.py).

---

### Pillar 2 — Spatial PDE / FISH story (June–July)

The **data findings** (3D FISH) are the primary contribution; PDE/PINN are methods.

#### 2-A — DH diffusion diagnostic (immediate)
- 8 diagnostic jobs running (`jobs/fit_diffusion_dhdiag_submit.sh`, job IDs 40371–40374 + variants).  
- **Collect**: `aggregate_diffusion_sweep.py` → if all DH variants converge to loss floor ≈0.091 regardless of `--d-max`, confirm **DH diffusion is intrinsically indeterminate** (reaction-dominated) → state as finding, not failure.  
- **Decision point**: if DH floor confirms → present CH D_i as illustrative, DH as "reaction-dominated regime where diffusion is not the limiting factor" — this is itself a scientifically meaningful asymmetry matching the data finding (DH lateral homogenization = well-mixed).

#### 2-B — Multi-day fitting for D_i (July)
- Day-1 alone is insufficient to constrain D_i (diffusion needs temporal evolution).  
- **Action**: Run `fit_diffusion_clsm.py` on Tag1+Tag3+Tag6 stacks (CH condition, where fitting converges), fitting all days jointly.  
- Check F.nucleatum channel: current .lif files are 4-channel (no Fn). Confirm with Szafrański lab whether Fn FISH channel is in a separate .lif or missing. If missing, 5-species PDE collapses to 4.

#### 2-C — 3D FISH figures (pub-grade)
- Fn-Pg depth separation: `results/fish_3d/fish_3d_fnpg_coloc.png` → reproduce with thesis_style.py (usetex/lmodern, 9pt).  
- DH lateral homogenization: `fish_3d_lateral_heterogeneity.png` → same.  
- These are already the strongest spatial results (Mann-Whitney p=2.17e-4, 84 FOVs) — just need style pass.

#### 2-D — PINN 3D result framing (if time)
- GPU result D=[.020,.012,.014,.012,.012] is illustrative (5 time points = weak constraint).  
- Frame explicitly as *proof-of-concept inverse problem* in appendix. Do not oversell as quantitative D_i.  
- Only worth extending if more FISH time points become available.

---

### Pillar 3 — Consolidation & figure pipeline (August)

By August, no new experiments. Integrate what exists into the thesis figure set.

#### 3-A — Freeze the canonical figure list
Target thesis figure set (8 main + 2 supplementary):

| # | Content | Script | Status |
|---|---|---|---|
| Fig 1 | Study design / pipeline | `fig1_study_design` | ✅ exists |
| Fig 2 | Dieckow data + guild trajectories | `generate_dieckow_paper_figures` | ✅ |
| Fig 3 | LOO-RMSE comparison (all models) | needs update | 🔧 |
| Fig 4 | W sweep + SA phase transition | slides → standalone | 🔧 |
| Fig 5 | Prior asymmetry (health/dysbiosis) | `fig_prior_asymmetry.py` (new) | ⬜ |
| Fig 6 | Network: concordant backbone + CS↔DH rewiring | `fig7_Aij_Fij_scatter` | 🔧 |
| Fig 7 | 3D FISH: Fn-Pg depth + lateral heterogeneity | `fish_3d_fnpg_coloc` | 🔧 style |
| Fig 8 | PDE: CH diffusion profile (illustrative D_i) | `fig_pde_ch_profile.py` | ⬜ |
| Supp A | A-matrix stability heatmap (10 LOO folds) | `loo_stability_analysis` | ⬜ |
| Supp B | AGORA FBA cross-feeding network | `agora_slides` figures | 🔧 |

#### 3-B — Botelho: archive, not thesis
- Keep all Botelho scripts and results in `results/botelho_validation/`.  
- Write a 3-paragraph `docs/botelho_validation_summary.md` (internal) so the finding is documented.  
- Do not include in thesis main text. Mention in Discussion as "preliminary cross-cohort evidence" if at all.

#### 3-C — Code/data handover prep (August)
- Ensure all thesis-figure scripts run cleanly from repo root with one command.  
- `scripts/figures/build_all_figs.sh` — collect all figure-generating calls in one script.  
- `paper_data.py` stays as single truth for posterior samples.

---

## Timeline

| Period | Focus | Milestone |
|---|---|---|
| **Jun 1–15** | 2-A DH diagnostic collect; 1-A neutral-init LOO submit | DH indeterminacy confirmed/denied |
| **Jun 15–30** | 1-B stability analysis; 1-C prior asymmetry fig; 1-D network fig | 0D story complete with figs |
| **Jul 1–20** | 2-B multi-day FISH fitting (CH); 2-C 3D FISH figures style pass | Spatial data pillar figures done |
| **Jul 20–Aug 10** | 3-A figure freeze; 3-B Botelho archive | Canonical fig set locked |
| **Aug 10–31** | Buffer: clean runs, reproducibility check, handover prep | All results reproducible from repo |
| **Sep 1 →** | Thesis writing begins | — |

---

## Decision rules ("clean results only")

1. **Number in thesis ⟺ number verified against real data** — no memory/draft values.  
2. **p-value required for any "significant" claim** — report n and test alongside.  
3. **Illustrative = labeled as illustrative** — D_i from single-day or weak-constraint fits get explicit "illustrative, not calibrated" caption language.  
4. **Negative result = result** — DH diffusion floor is a finding (reaction-dominated regime), not a failure to fix.  
5. **No Botelho in main text** unless the trajectory prediction story becomes clean (currently p=0.22, excluded).

---

## What is explicitly out of scope (Jun–Aug)

- Thesis writing (prose, LaTeX chapters) → September.  
- New ecological models beyond gLV/Hamilton.  
- FEM/continuum mechanics bridge → save for Keio/Muramatsu lab (2027).  
- Szafrański 2025 cross-sectional community-type comparison → separate project, not this thesis.  
- COMETS/dFBA additional runs unless they produce a clean testable prediction within 1 week.
