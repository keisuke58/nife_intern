---
description: Check PBS jobs (fifawc) + aggregate the diffusion-fit sweep, pick best
---
Report the HPC fit status and aggregate the diffusion sweep.

1. `qstat -u nishioka` — summarise: how many R / Q, and the job names (fit_diffusion_cl / fit_diff_sweep). Write the running count to `/tmp/hpc_jobcount` (just the number) so the statusline can show it.
2. List landed results: `ls -t results/diffusion_fit/D_fit_*.json | head`, count them.
3. Aggregate: `python3 scripts/analysis/aggregate_diffusion_sweep.py` — show the per-condition best (converged first, then lowest loss) and the full table tail.
4. Flag honestly: if the lowest-loss runs are still `success=False`, say the final D_fit promotion is still pending (don't promote a high-loss converged run).
5. If `$ARGUMENTS` is `promote`, AND a converged low-loss result exists per condition, copy it to the canonical `results/diffusion_fit/D_fit_<COND>.json` and note what was promoted; otherwise do not modify canonical files.