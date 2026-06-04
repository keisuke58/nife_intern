---
title: "HOBIC FISH Data Processing"
subtitle: "Extracting 5-species depth profiles from 4-channel fluorescence images"
author: "Nishioka — NIFE"
date: "2026-06-04"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
---

## Summary

We processed confocal FISH images from the Heine 2025 HOBIC flow-chamber experiment to
construct **depth-resolved 5-species composition profiles** as input to the reaction–diffusion
PDE fit.

- Input: Leica confocal **FISH `.lif`** (CH / DH conditions, Ti substrate)
- Challenge: only **4** detector channels for **5** target species
- Approach: decode the paper's labelling design via per-voxel colocalization
- Output: **depth × 5-species composition profiles** and PDE fit results ($D_i$, $u$)

\vspace{0.4em}
New: `fish_decode.py`, `lif_quicklook.py`; updated: `lif_to_zprofiles.py`

---

## Input data: HOBIC FISH `.lif` (11 files total)

| Experiment batch | Cond. | Day | Substrate |
|---|---|---|---|
| 220518/601/720, 220817, 240416 | CH | 1, 6, 10, 15, 21 | Ti |
| 241203 (Tag1) | DH | 1 | Ti |
| 241018 | DH | 6, 10, 15, 21 | Ti + Glass |

- Pixel size 0.18 µm, z-step 2 µm. Only Ti substrate used for analysis
  (HOBIC = titanium implant model; 9 Glass FOVs excluded, retained for separate analysis).
- Headless environment: no GUI tools (Fiji / napari). Files read with `readlif`;
  PNGs exported with Times font and µm scale bar.

---

## Raw 4-channel images

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__overview.png){ height=72% }

rows = FOVs, columns = Blue / Yellow / Green / Red + composite

---

## Mismatch between channels and species

Four detector channels cover five target species; a one-to-one colour–species mapping does not hold.

From Heine 2025 (*Front. Oral Health*, Table S5 + Methods §2.6):

> *F. nucleatum was targeted by two probes ... labeled with different dyes
> – resulting in co-localized blue and red fluorescence.*

*F. nucleatum* alone carries dual labels (Alexa405 = blue, Alexa647 = red) and therefore
produces signal in both channels.

---

## Decoding rule (colocalization)

| Channel | Contributing species |
|---|---|
| **Blue** 405 nm | *S. oralis* + *F. nucleatum* |
| **Green** 488 nm | *A. naeslundii* |
| **Yellow** 552 nm | *V. dispar/parvula* |
| **Red** 638 nm | *P. gingivalis* + *F. nucleatum* |

$$
F_n = B \cap R,\quad S_o = B - F_n,\quad P_g = R - F_n
$$

Colocalization is computed per voxel before taking the xy-average
($\mathrm{mean}(\min) \neq \min(\mathrm{mean})$).

---

## Decoded 5 species

![](figures/lif_quicklook/220518_HOBIC22_5Spezies_FISH_Tag1__species.png){ height=72% }

*F. nucleatum* (purple) is cleanly separated from *S. oralis* / *P. gingivalis*.

---

## Decoding error in the original code and the fix

The original code assumed "blue = pure *S. oralis* / red = pure *P. gingivalis* /
purple = *F. nucleatum*." No purple channel exists in the actual files; without correction:

- *F. nucleatum* signal is lost entirely
- *S. oralis* and *P. gingivalis* counts are inflated by *F. nucleatum* bleed-in

This affects 3 of the 5 species fed into the spatial PDE. A canonical decoder
`fish_decode.py` was implemented and is shared by both pipeline tools.

---

## Profile extraction, merging, and naming conventions

**z-profile**: each FOV's 3-D stack is reduced to an xy-mean intensity per depth slice.

**Replicate merging**: multiple `.lif` files sharing the same `(condition, day)` are pooled
(Day 1 = 3 separate experiment dates, 15 FOVs total, averaged once).

**Experimental naming conventions** (Heine, e-mail 2026-05-26) encoded in the tool:

- HOBIC22 = commensal → **CH**; HOBIC24 = dysbiotic → **DH** (resolved automatically)
- Day number: filename suffix `TagN`; for HOBIC24, leading integer of the series name
- **Substrate filter** `--substrate ti`: selects Ti from mixed Ti/Glass datasets

```bash
python scripts/pde/lif_to_zprofiles.py "HOBIC FISH"/*.lif --substrate ti
```

---

## Result: CH / DH depth time-series (Ti)

![](results/diffusion_fit/zprofiles_all_ti.png){ height=62% }

CH / DH × Day 1/6/10/15/21 profiles written to `zprofiles_all_ti.csv` (400 rows).
Both conditions show biofilm thickening and depth-wise compositional shifts over time.

---

## CH vs DH comparison (deep *P. gingivalis*)

![](results/diffusion_fit/zprofiles_all_ti_overlay.png){ height=58% }

**Solid = CH, dashed = DH**; colour = species, y-axis = µm (surface → deep).
From day 6 onward, the *P. gingivalis* (red) centre of mass is consistently deeper in DH than in CH.

---

## Per-depth composition (stacked area)

![](results/diffusion_fit/zprofiles_all_ti_stacked.png){ height=62% }

Each depth's 5-species fractions are stacked to unity. The expanded *P. gingivalis* (red) band
in the deep layers of the DH condition is apparent across late time points.

---

## Reaction–diffusion PDE fit

Compositional dynamics are decomposed into ecological interactions and spatial transport:

1. **Reaction** $(A,\,b)$: species interaction matrix. **Fixed** (gLV / TMCMC estimate)
2. **Diffusion** $D_i$ **+ advection** $u$: depth-wise transport. **Fit target**

$$
\partial_t \varphi_i = D_i\,\partial_{zz}\varphi_i - u\,\partial_z\varphi_i
+ \varphi_i\!\left[(A\varphi+b)_i - \varphi^\top(A\varphi+b)\right]
$$

$D_i$ and $u$ are optimised by L-BFGS to minimise MSE against the observed profiles.
Runs executed on the HPC (PBS), CH and DH separately (Ti, $N_z=48$, 8 restarts).

---

## Fit results: diffusion coefficients $D_i$ and advection $u$

| Species | $D^\text{CH}$ | $D^\text{DH}$ |
|---|---|---|
| *S. oralis* | 0.029 | 0.015 |
| *A. naeslundii* | 0.006 | 0.006 |
| *Vd/Vp* | 0.005 | 0.009 |
| *F. nucleatum* | 0.006 | 0.015 |
| *P. gingivalis* | 0.006 | 0.006 |

Advection: $u^\text{CH} = 0.0069$, $u^\text{DH} = 0.0059$ (units: µm/s-equivalent)

DH: `success=True` (converged); CH: `success=False` (not converged, preliminary).
Results stored in `D_fit_*_nz48_eps3e-4.json`.

---

## Fit result: CH condition

![](results/diffusion_fit/fit_CH_nz48_eps3e-4.png){ height=72% }

Observed profiles (dots) vs PDE prediction (lines). CH, $N_z=48$, loss = 0.125.

---

## Fit result: DH condition

![](results/diffusion_fit/fit_DH_nz48_eps3e-4.png){ height=72% }

Observed profiles (dots) vs PDE prediction (lines). DH, $N_z=48$, loss = 0.154 (converged).

---

## Finding: deep accumulation of *P. gingivalis* in dysbiosis

![](results/diffusion_fit/depth_niche.png){ height=52% }

- *P. gingivalis* migrates to deeper layers in DH from day 6
  (centre-of-mass shift up to +30 µm, consistent with the anaerobic niche)
- *F. nucleatum*–*P. gingivalis* co-localisation is restricted to the early stage
  (DH day 1: coloc = 0.76); from day 6 onward CH > DH, indicating that
  *P. gingivalis* separates from *F. nucleatum* and accumulates independently
- CH / DH community divergence is ~0.2 from day 1 and remains approximately constant

---

## Deliverables and data limitations

| File | Role |
|---|---|
| `fish_decode.py` | New · canonical decoder ($F_n = B \cap R$) |
| `scripts/pde/lif_quicklook.py` | New · visualiser (Times font, µm bar) |
| `scripts/pde/lif_to_zprofiles.py` | Updated · decode + merge + naming + substrate filter |
| `zprofiles_all_ti.csv` | PDE input (depth × species × condition) |
| `D_fit_*_nz48_eps3e-4.json` | Fit results ($D_i$, $u$) |

**Data limitations**: HOBIC origin limits analysis to **CH / DH only** (static conditions CS/DS unavailable).
Glass FOVs from batch 241018 (DH) are excluded. DH late time points (Day 15/21) have only 2 Ti FOVs,
which reduces the precision of the late-stage estimates.
