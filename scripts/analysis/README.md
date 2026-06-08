# scripts/analysis/

Analysis scripts for the NIFE oral biofilm project. All scripts are argparse CLIs;
run from the repo root so the path-shim resolves correctly.

---

## Core pipeline (results in paper / thesis)

| Script | Purpose | Thesis |
|---|---|---|
| `loo_stability_analysis.py` | LOO per-pair sign stability → `fig_loo_stability` | Ch4 |
| `spatial_crossfeeding.py` | Lactate/propionate depth stratification test | Ch5 §5.5 |
| `fish_cooccurrence_depth.py` | Fn–Pg co-occurrence vs depth | Ch5 |
| `fish_pair_correlation.py` | Pairwise FISH species correlation | Ch5 |
| `fish_spatial_cooccurrence.py` | Spatial co-occurrence over 3-D FOV | Ch5 |
| `fish_3d_profile.py` | Full 3-D FISH structure (no xy-averaging) | Ch5 |
| `fish_3d_batch.py` | Batch extraction over all .lif FOVs | Ch5 |
| `guild_network_analysis.py` | Centrality / trophic / CS↔DH rewiring | Ch4 §4.5 |
| `analyze_depth_niche.py` | Depth niche separation quantification | Ch5 |
| `analyze_d_vs_centrality.py` | Diffusivity vs network centrality | Ch5 |
| `analyze_fish_voxel.py` | Voxel-level ecology (Manders, biomass) | Ch5 |
| `analyze_hobic_vs_dieckow.py` | HOBIC↔Dieckow guild mapping check | Ch4 |
| `plot_depth_profiles.py` | Polished depth-profile figures | Ch5 |
| `dieckow_ode_continuous.py` | Continuous-time posterior trajectories (Dieckow) | Ch4 |
| `heine_ode_continuous.py` | Continuous-time posterior trajectories (Heine) | Ch3 |
| `ode_continuous_vs_comstat.py` | ODE prediction vs BiofilmQ structural metrics | Ch5 |
| `aggregate_diffusion_sweep.py` | Tabulate diffusion-parameter sweep, pick best | Ch5 |
| `compute_di_szafranski.py` | Guild Dysbiosis Index on Szafrański 127 samples | Ch4 |
| `joshi_gdi_improved.py` | Joshi GDI enrichment (R/G + Spearman) | Ch4 §4.4 |
| `joshi_gdi_expanded_check.py` | Expanded GDI check (mucositis, severity) | Ch4 |
| `joshi_attractor_analysis.py` | Attractor classification on Joshi cohort | Ch4 |
| `community_type_analysis.py` | GMM community-type labelling | Ch4 |
| `gmm_attractor_analysis.py` | GMM attractor characterisation | Ch4 |
| `compare_all_models_bc.py` | Bray-Curtis comparison across 13 model variants | Ch4 |
| `compute_bc_metrics.py` | Bray-Curtis metric computation | Ch4 |
| `compute_rmse_bc_noprior.py` | RMSE/BC for no-prior baseline | Ch4 |
| `dieckow_analysis.py` | Full Dieckow primary analysis | Ch4 |
| `dieckow_full_pipeline.py` | End-to-end Dieckow fit+LOO pipeline | Ch4 |
| `dieckow_hamilton_fit.py` | Hamilton MAP fit to Dieckow data | Ch4 |
| `dieckow_posterior_predictive.py` | Posterior predictive check (CPU) | Ch4 |
| `dieckow_posterior_predictive_gpu.py` | Posterior predictive check (GPU) | Ch4 |
| `dieckow_postpred_nweeks_scan.py` | N-week extrapolation scan | Ch4 |
| `dieckow_ccs_pipeline.py` | CCS community-composition pipeline | Ch4 |
| `dieckow_gmm_reassign.py` | GMM-based guild reassignment | Ch4 |
| `analysis_dieckow_extras.py` | Supplementary Dieckow diagnostics | Ch4 |
| `paper5sp_identifiability_gating.py` | Identifiability + gating edge (Heine 5sp) | Ch3 |
| `b_classify_stats.py` | Growth-rate classification statistics | Ch4 |
| `guild_importance_analysis.py` | Guild-level importance / sensitivity | Ch4 |
| `species_importance_analysis.py` | Species-level importance ranking | Ch3 |
| `prior_metatx_data.py` | Metatranscriptome prior data loading | Ch4 |
| `run_hamilton_kegg_expanded.py` | Hamilton + expanded KEGG prior fit | Ch4 |
| `run_hamilton_kegg_steadystate.py` | Hamilton KEGG steady-state scan | Ch4 |
| `run_hamilton_kegg_steadystate_v2.py` | Hamilton KEGG steady-state scan v2 | Ch4 |
| `compare_nsp_vs_replicator.py` | NSP vs replicator model comparison | Ch4 |
| `compare_guild_models.py` | Multi-model guild comparison | Ch4 |
| `compare_guild_vs_suppfile1.py` | Guild assignment vs Szafrański Suppl. File 1 | preprocessing |
| `verify_bergey_signs.py` | Bergey's-based sign sanity check | Ch4 |

---

## Exploratory (not in paper/thesis — kept for reference)

These scripts were run during development but their results did not enter the
final manuscript or thesis. They are retained in case of reviewer questions or
future extension.

| Script | Purpose | Connected to |
|---|---|---|
| `symbolic_regression_A.py` | gplearn symbolic regression of A_ij from metabolic features | `generate_fig_symbolic_regression_A.py`, `jobs/` |
| `validate_agora_prior_humann.py` | AGORA sign edges vs HUMAnN sample-level abundances | `jobs/joshi_mtx_humann_metaphlan_pipeline.sh`, `docs/joshi_humann_metaphlan_validation.md` |
| `validate_prior_metatranscriptome.py` | Sign prior vs metatranscriptome expression | standalone |
| `early_attractor_predictor.py` | Random Forest attractor predictor from t=0,1 | `generate_fig_attractor_predictor.py` |
| `guild_tipping_point.py` | Bifurcation / tipping-point analysis (b-vector shift) | `jobs/guild_tipping_pbs.sh` |
| `guild_relapse_prevention.py` | Relapse prevention threshold simulations | standalone |
| `guild_intervention_analysis.py` | Targeted guild-intervention simulations | standalone |
| `guild_phase_diagram_3d.py` | 3-D phase diagram of gLV attractor landscape | standalone |
| `compare_agora_v1_v2.py` | AGORA v1 vs v2 vs MICOM sign-prior comparison | standalone |
| `biofilmQ_comstat_metrics.py` | BiofilmQ/COMSTAT structural metrics from FISH | standalone |
| `run_mdsine2_dieckow.py` | MDSINE2 fit to Dieckow data | `jobs/mdsine2_loo_job.sh`, `scripts/loo_cv/run_mdsine2_loo.py` |
