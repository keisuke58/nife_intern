# logs/ — local-only job logs

Dump location for run/job logs. **Gitignored** (`*.log`, `*.fifawc.log` in `.gitignore`)
— logs are regenerable byproducts and are never committed. Keep the repo root lean:
loose `*.log` should not sit at the repo root.

## Layout

| folder | contents |
|--------|----------|
| `logs/fifawc/` | PBS (fifawc cluster) job stdout logs — `fit_diff_sweep_*`, `fit_diffusion_*` `*.fifawc.log` etc. Swept here from the repo root on 2026-06-04 (19 files). |
| `logs/` (root) | other local run logs |

## Conventions

- New PBS logs landing at the repo root → move into `logs/fifawc/`:
  ```bash
  cd /home/nishioka/IKM_Hiwi/nife && mv *.fifawc.log logs/fifawc/ 2>/dev/null
  ```
- To make jobs write here directly, point the PBS output at this folder in `jobs/*.sh`:
  `#PBS -o logs/fifawc/` (and `#PBS -e logs/fifawc/`).
- **Results** (the actual fitted parameters / sweep outputs) are JSON/CSV under
  `results/diffusion_fit/`, *not* here — logs are only stdout/stderr.
