# scripts/

Runnable, never-imported scripts, grouped by purpose. Moved here out of the repo
root to keep the root tidy; **shared modules that other code imports stay at the
repo root** (see `CLAUDE.md`).

| dir | contents |
|-----|----------|
| `fitting/` | gLV / Hamilton / CR parameter fitting (`fit_*`, `estimate_*`) |
| `loo_cv/` | leave-one-out / cross-validation drivers, validation, permutation/robustness, related collectors and plots (`loo_*`, `run_*loo*`, `*_cv*`, `validate_*`, …) |
| `figures/` | publication figures and slides (`plot_*`, `generate_*`, `make_*`, `*slide*`) |
| `pde/` | spatial reaction–diffusion PDE and `.lif` imaging (`nsp_pde_*`, `glv_pde_*`, `lif_*`, diffusion) |
| `preprocessing/` | data prep / ETL / downloads (`build_*`, `download_*`, `get_*`, `aggregate_*`, `assign_*`, `export_*`, `collect_*`) |
| `analysis/` | everything else analytical (`compare_*`, `compute_*`, `analyze_*`, dataset-specific `dieckow_*`/`joshi_*`/`guild_*`, …) |

## Running

Run **from the repo root**, e.g.

```bash
python scripts/loo_cv/run_glv_loo.py --hold 0 ...
```

Each file has a two-line path-shim at the top (marked `# [nife-pathshim]`) that
adds the repo root to `sys.path`, so the bare sibling imports
(`from guild_replicator_dieckow import ...`) keep working from any location.
`Path(__file__)` anchors were rewritten to `parents[2]` so `results/`/`data/`
paths still resolve to the repo root. **Don't delete the shim or change the
`parents[2]`** — they are what makes a script in a subfolder behave exactly as it
did at the root.
