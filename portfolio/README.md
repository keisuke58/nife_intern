# portfolio/ — runnable demos

Self-contained, CPU-friendly demos that back the claims in the top-level
[../PORTFOLIO.md](../PORTFOLIO.md). No project data, no GPU, fixed seeds.

| Demo | Run | What it shows |
|---|---|---|
| **MI active learning** | `python portfolio/mi_active_learning_demo.py` | GP surrogate + Bayesian optimization (Expected Improvement); ~65× lower regret than random search; figure → `figs/mi_active_learning.png` |
| **3-D U-Net (FISH)** | `python portfolio/fish_segmentation_unet3d.py --smoke-test` | Volumetric segmentation architecture + soft-Dice/CE loss + backprop; CPU smoke test. Untrained scaffold — see header. |

Both are deliberately small so a reviewer can run them in seconds. Real training /
inference for the FISH U-Net dispatches to the GPU host (see `../jobs/`).
