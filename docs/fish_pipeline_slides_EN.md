---
title: "HOBIC FISH Data Processing"
subtitle: "Decoding 4 channels -> 5 species and extracting depth profiles"
author: "Nishioka — NIFE / SFB TRR-298"
date: "2026-06-03"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
---

## What this project did (one slide)

We turned 3-D confocal images of the biofilm into a table of
**"which fraction of each of 5 species is present at each depth."**

- Input: Leica confocal **FISH `.lif`** (flow-chamber HOBIC, Heine 2025)
- Hard part: images have **only 4 colours but 5 species**
  -> decode the paper's labelling design to separate them
- Output: **depth × 5-species composition** (input to the PDE diffusion fit)

\vspace{0.5em}
New `fish_decode.py` / `lif_quicklook.py`, updated `lif_to_zprofiles.py`

---

## Input data: HOBIC FISH `.lif`

All 11 files. **Both models present: commensal (HOBIC22->CH) and dysbiotic (HOBIC24->DH).**

| Experiment | Cond. | Day | Substrate |
|---|---|---|---|
| 220518/601/720, 220817, 240416 | CH | 1,6,10,15,21 | Ti |
| 241203 (Tag1) | DH | 1 | Ti |
| 241018 | DH | 6,10,15,21 | **Ti + Glass mixed** |

- pixel 0.18 µm, z-step 2 µm. **Substrate = Ti** (HOBIC is a Ti implant; CH unlabelled = Ti). 9 Glass FOVs excluded, kept for separate analysis.
- **Headless** -> no Fiji/napari. `readlif` -> **PNG** (Times font + µm scale bar)

---

## Raw 4 channels (as acquired)

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__overview.png){ height=72% }

rows = FOVs, columns = Blue / Yellow / Green / Red + composite

---

## The problem: 4 channels <-> 5 species (not 1:1)

4 detector channels, 5 species. Not a simple "one colour = one species."

Source — **Heine 2025, Front. Oral Health, Table S5 + Methods §2.6**:

> *F. nucleatum was targeted by two probes ... labeled with different dyes
> – resulting in co-localized blue and red fluorescence.*

-> **F. nucleatum alone is dual-labelled** (Alexa405 = blue & Alexa647 = red).
It lights up in **both** blue and red.

---

## Decoding rule (colocalization)

| LUT / laser | contributing species |
|---|---|
| **Blue** 405 nm | S. oralis + **F. nucleatum** |
| **Green** 488 nm | A. naeslundii |
| **Yellow** 552 nm | V. dispar/parvula |
| **Red** 638 nm | P. gingivalis + **F. nucleatum** |

```
F. nucleatum  = Blue ∩ Red
S. oralis     = Blue − (Blue ∩ Red)
P. gingivalis = Red  − (Blue ∩ Red)
```

Note: Per voxel, before the xy-average. mean of `min` ≠ `min` of mean.

---

## Decoded 5 species

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__species.png){ height=72% }

F. nucleatum (purple) is separated from S. oralis / P. gingivalis

---

## The bug this avoided

The old code assumed "blue = pure S.oralis / red = pure P.gingivalis / purple = F.nucleatum."
**There is no purple channel** in the real files -> unchanged it would:

- **drop F. nucleatum entirely**
- **over-count S. oralis & P. gingivalis** (Fn leaks in)

-> the PDE input would be **wrong for 3 of 5 species**.
Fixed via `fish_decode.py` (canonical decoder shared by both tools).

---

## Extraction, merging, e-mail conventions

**z-profile**: collapse each FOV's 3-D to "xy-mean intensity per depth."

**Replicate merge**: pool multiple `.lif` per `(condition, day)`
(Day 1 = 3 separate experiment dates -> 15 FOVs, averaged once).

**Heine's e-mail conventions (2026-05-26), in the tool**:

- HOBIC22 = commensal -> **CH** / HOBIC24 = dysbiotic -> **DH** (auto)
- Day = filename suffix `TagN` (HOBIC24 only: leading int of series name)
- **Substrate filter** `--substrate ti` (Ti from the Ti/Glass mix)

```bash
python scripts/pde/lif_to_zprofiles.py "HOBIC FISH"/*.lif --substrate ti
```

---

## Result: CH / DH depth time-series (Ti)

![](results/diffusion_fit/zprofiles_all_ti.png){ height=62% }

CH/DH × Day 1/6/10/15/21 -> `zprofiles_all_ti.csv` (400 rows).
Both thicken with depth-wise composition shifts. DH late days: few Ti FOVs (Glass excluded).

---

## Next: what the diffusion fit does

A reaction–diffusion PDE separates two forces that set the composition:

1. **Reaction (A, b)** = species interactions. **Known** (gLV / TMCMC)
2. **Diffusion (D_i) + advection (u)** = spatial motion. **Unknown = fit target**

```
guess D_i, u -> solve PDE -> predicted vs observed (MSE)
            ^___ search D_i, u minimising the mismatch (L-BFGS) ___|
```

**Production fit run on the HPC** (CH/DH, Ti). Preliminary u: CH 0.0038 / DH 0.0060.
Note: both `success=False` (not converged) = preliminary; DH weakly constrained (few late Ti FOVs) -> needs more restarts.

---

## Deliverables & data limits

| File | Role |
|---|---|
| `fish_decode.py` | New · canonical decoder (Fn=blue∩red) |
| `scripts/pde/lif_quicklook.py` | New · visualiser (Times + µm bar) |
| `scripts/pde/lif_to_zprofiles.py` | Updated · decode+merge+conv.+substrate |
| `zprofiles_all_ti.csv` / `D_fit_*.json` | PDE input + fit results |

**Limits**: flow-chamber data -> **CH/DH only** (no static CS/DS). 241018 (DH) **Glass excluded**. DH late days (15/21): only 2 Ti FOVs.
