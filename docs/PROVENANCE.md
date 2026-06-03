# Figure provenance

Maps each KEY thesis/paper figure to the script and the **exact command** that
regenerates it. All commands are run **from the repo root**
(`/home/nishioka/IKM_Hiwi/nife`); every script carries the `# [nife-pathshim]`
two-liner so bare imports resolve and `results/` paths are anchored at the root.

Two classes of figure:

- **LIGHT / local** — pure matplotlib (+ pandas/scipy), no cluster, no GPU, no
  external solver. These are the ones `/thesis-sync` and `make figures`
  regenerate. Safe to run on the shared login node.
- **HEAVY / extra-stack** — needs `.lif` re-decode (FISH) or the COBRApy/AGORA
  FBA stack. **Excluded** from the light sync; listed here only so the figure is
  traceable. Do **not** run these as part of a routine sync on the shared server.

The `D_fit_*.json` / `sweep_summary.csv` inputs to the depth/centrality figures
come from the diffusion-fit sweep (HPC); the plots themselves only *read* them and
are light. Promote a converged sweep via the `/hpc` aggregation, not here.

## Key figures

| Figure (output path) | Script | Exact command (from repo root) | Class |
|---|---|---|---|
| `results/figures/pipeline_overview_pub.png` (+`.pdf`) | `scripts/figures/make_pipeline_overview.py` | `python3 scripts/figures/make_pipeline_overview.py` | LIGHT |
| `results/figures/concept_overview_pub.png` (+`.pdf`) | `scripts/figures/make_concept_diagram.py` | `python3 scripts/figures/make_concept_diagram.py` | LIGHT |
| `results/diffusion_fit/zprofiles_all_ti_overlay.png` | `scripts/analysis/plot_depth_profiles.py` | `python3 scripts/analysis/plot_depth_profiles.py` | LIGHT |
| `results/diffusion_fit/zprofiles_all_ti_stacked.png` | `scripts/analysis/plot_depth_profiles.py` | `python3 scripts/analysis/plot_depth_profiles.py` | LIGHT |
| `results/diffusion_fit/zprofiles_all_ti_grid.png` | `scripts/analysis/plot_depth_profiles.py` | `python3 scripts/analysis/plot_depth_profiles.py` | LIGHT |
| `results/diffusion_fit/depth_niche.png` | `scripts/analysis/analyze_depth_niche.py` | `python3 scripts/analysis/analyze_depth_niche.py` | LIGHT |
| `results/diffusion_fit/ch_dh_divergence.png` | `scripts/analysis/analyze_depth_niche.py` | `python3 scripts/analysis/analyze_depth_niche.py` | LIGHT |
| `results/diffusion_fit/spatial_crossfeeding.png` | `scripts/analysis/spatial_crossfeeding.py` | `python3 scripts/analysis/spatial_crossfeeding.py` | LIGHT |
| `results/diffusion_fit/d_vs_centrality.png` | `scripts/analysis/analyze_d_vs_centrality.py` | `python3 scripts/analysis/analyze_d_vs_centrality.py` | LIGHT¹ |
| `results/diffusion_fit/hobic_vs_dieckow.png` | `scripts/analysis/analyze_hobic_vs_dieckow.py` | `python3 scripts/analysis/analyze_hobic_vs_dieckow.py` | LIGHT |
| `results/diffusion_fit/sweep_summary.csv` (sweep table) | `scripts/analysis/aggregate_diffusion_sweep.py` | `python3 scripts/analysis/aggregate_diffusion_sweep.py` | LIGHT (reads HPC sweep outputs) |
| `results/fish_3d/fish_3d_fnpg_coloc.png` | `scripts/analysis/fish_3d_batch.py` | `python3 scripts/analysis/fish_3d_batch.py --decode-ds 2` | **HEAVY** (re-decodes ~84 `.lif` FOVs; use `/fish3d`) |
| `results/fish_3d/fish_3d_lateral_heterogeneity.png` | `scripts/analysis/fish_3d_batch.py` | `python3 scripts/analysis/fish_3d_batch.py --decode-ds 2` | **HEAVY** |
| `results/fish_3d/fish3d_*_DH_d6_s0.png` (single-FOV ortho/proj/profiles) | `scripts/analysis/fish_3d_profile.py` | `python3 scripts/analysis/fish_3d_profile.py --file "HOBIC FISH/<glob>.lif" --series 0` | **HEAVY** |
| `results/fig2_agora_pipeline.png` (+`.pdf`) | `scripts/figures/generate_fig127.py` | `python3 scripts/figures/generate_fig127.py` | **HEAVY** (needs FBA/AGORA stack — not local-regenerable) |
| `results/fig3_agora_sign_validation.png` (+`.pdf`) | `scripts/figures/generate_fig3456.py` | `python3 scripts/figures/generate_fig3456.py` | **HEAVY** (needs FBA/AGORA stack — not local-regenerable) |

¹ `analyze_d_vs_centrality.py` reads the current `results/diffusion_fit/D_fit_<cond>.json`
(produced by the HPC diffusion-fit sweep). The plot is light; the inputs are not
regenerated locally. Values are illustrative (5 species → underpowered).

## Notes on the HEAVY rows

- **`fish_3d_batch.py` / `fish_3d_profile.py`** re-decode the raw HOBIC `.lif`
  confocal stacks per voxel (no xy-averaging). This is tens of minutes, heavy I/O
  and memory, and depends on the local `HOBIC FISH/` imaging tree (gitignored,
  ~9 GB). Run it deliberately in the background via the `/fish3d` command — never
  as part of a routine sync on the shared server.
- **`generate_fig127.py` (fig2) / `generate_fig3456.py` (fig3)** build the AGORA
  metabolic-pipeline and sign-validation figures. They depend on the FBA stack
  (COBRApy + AGORA GEMs / `build_net_flow_expanded.py` net-flow outputs) and the
  decks reference the already-committed `results/fig2_agora_pipeline.png` /
  `results/fig3_agora_sign_validation.png`. Treat those PNGs as committed
  artefacts; regenerate only when the underlying AGORA net-flow changes, with the
  FBA stack installed.

## See also

- `/thesis-sync` (`.claude/commands/thesis-sync.md`) — runs the LIGHT figures +
  rebuilds the decks + commits + uploads.
- `Makefile` (`make figures`, `make decks`, `make help`).
- `docs/pipeline_overview.md` — the data-flow these figures summarise.
