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

One file = the biofilm on a given experiment day, several FOVs as a z-stack.

| File | Model | Day | FOVs |
|---|---|---|---|
| 220518 / 220601 / 220720 | commensal | 1 | 6 / 5 / 4 |
| 220817_Tag15 | commensal | 15 | 7 |
| 220817_Tag21 | commensal | 21 | 4 |

- pixel 0.18 µm, z-step 2 µm
- **Headless environment** (no `DISPLAY`) -> no Fiji / napari
- -> read with `readlif`, write **PNG quick-looks** with a custom tool

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

```bash
python scripts/pde/lif_to_zprofiles.py "HOBIC FISH"/*.lif   # all automatic
```

---

## Result: commensal HOBIC depth time-series

![](results/diffusion_fit/zprofiles_CH_merged.png){ height=58% }

CH: Day 1 (15 FOV, 43µm) / Day 15 (7, 79µm) / Day 21 (4, 69µm)
-> `zprofiles_CH_merged.csv` (120 rows). Thickening + depth-wise composition shift.

---

## Next: what the diffusion fit does

A reaction–diffusion PDE separates two forces that set the composition:

1. **Reaction (A, b)** = species interactions. **Known** (gLV / TMCMC)
2. **Diffusion (D_i) + advection (u)** = spatial motion. **Unknown = fit target**

```
guess D_i, u -> solve PDE -> predicted vs observed (MSE)
            ^___ search D_i, u minimising the mismatch (L-BFGS) ___|
```

Estimate per-species **mobility** D=[So,An,V,Fn,Pg] and advection u.
This run is a pipeline smoke-test; the real fit awaits full data (DH, Days 3/6/10).

---

## Deliverables & data limits

| File | Role |
|---|---|
| `fish_decode.py` | New · canonical decoder (Fn=blue∩red) |
| `lif_quicklook.py` | New · headless visualiser (readlif->PNG) |
| `lif_to_zprofiles.py` | Updated · decode + merge + conventions |
| `zprofiles_CH_merged.csv` | PDE input (CH depth time-series) |

**Limits**: commensal (CH) only in hand. **DH not yet received**; CH only **Days 1/15/21** (3/6/10 missing); static (CS/DS) not in this FISH set.
