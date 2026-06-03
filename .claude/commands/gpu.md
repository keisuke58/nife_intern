---
description: Dispatch a script to the vancouver01 GPU (non-destructive scratch clone) and poll
---
Run a script on the vancouver01 GPU host. `$ARGUMENTS` = the python command to run, relative to the repo root
(e.g. `scripts/pde/pinn_3d_inverse.py --cond DH --crossdiff --epochs 4000`).

The GPU host has 4× CUDA (jax 0.6.2, conda env `klempt_fem`). The repo lives at the NON-DESTRUCTIVE scratch clone
`/home/nishioka/nife_gpurun` (a fresh `git clone` of nife_intern — NEVER touch the stale shared `/home/nishioka/nife`).

Steps:
1. `scp` any locally-changed scripts the command needs (and any required `results/fish_3d/ic/*.npy` data) to
   `vancouver01:/home/nishioka/nife_gpurun/<same relative path>`. (Do not `git pull` the scratch — it has local
   result files; pulling aborts. Just scp the specific files.)
2. Pick a free GPU (check `nvidia-smi` first), then:
   ```bash
   ssh vancouver01 'cd /home/nishioka/nife_gpurun && source /home/nishioka/miniconda3/etc/profile.d/conda.sh && conda activate klempt_fem && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 && CUDA_VISIBLE_DEVICES=<N> nohup python3 <ARGUMENTS> > /tmp/<logname>.log 2>&1 & echo PID $!'
   ```
3. Launch a robust background poller (run_in_background) that waits until the process is gone, tolerating SSH
   hiccups: `while true; do st=$(timeout 30 ssh -o BatchMode=yes vancouver01 'pgrep -f "<scriptname>" >/dev/null && echo RUN || echo DONE' 2>/dev/null); [ "$st" = DONE ] && break; sleep 60; done; echo DONE`.
4. When it finishes, `scp` the result JSON/PNG back into the local `results/` and report.
Note: nohup output is buffered → the log looks empty until the run ends; that is normal, not a hang.