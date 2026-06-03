---
title: "Integrating Metabolism and Ecology through AGORA"
subtitle: "From genome-scale metabolic models to interaction signs — basics to critique"
author: "Keisuke Nishioka — NIFE / SFB TRR-298"
date: "2026-06-03"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
---

## What this deck is for

Back the **ecological model** of the oral biofilm (who helps / harms whom)
with **metabolism computed from genomes (AGORA)** — and examine, step by step
and critically, how that works and how far it is actually validated.

- **What AGORA is** / FBA basics
- Metabolism (secretion, uptake) -> **the sign of the interaction matrix A**
- Single-species pFBA -> **MICOM (community FBA)**
- **How far is it validated** (the real point)
- Cross-check against mechanistic simulation (COMETS dFBA)

\vspace{0.5em}
Bottom line up front: **the sign is usable, the magnitude is not. Only the
cross-feeding direction is independently validated.**

---

## Why a metabolic model at all — the problem

We want the **gLV interaction matrix A** (10 guilds x 10) from 16S abundance
time-series.

- ~100 parameters vs data = 10 patients x 3 weeks -> **underdetermined**
- A naive fit has many equivalent solutions; even the signs are not pinned down

\vspace{0.5em}
**Idea:** use metabolic knowledge as a **sign prior** on A.

```
guild j secretes a metabolite that guild i consumes  ->  j helps i  ->  A[i,j] > 0
guild j secretes a toxin (H2O2, H2S) that i takes up ->  j harms i  ->  A[i,j] < 0
```

-> constrain **only the sign**, never the magnitude. This is the central idea.

---

## What AGORA2 is

**AGORA2** = a large database of **per-microbe metabolic models (GEMs)** built
from genome sequence.

- Heinken et al. **2023, Nature Biotechnology** 41:1320-1331
- **7,302 strains** of genome-scale reconstructions; extended from the gut-only
  v1 to **multiple body sites including the oral cavity**
- each microbe = an **SBML (XML)** file listing all its reactions (hundreds to
  thousands)
- here, one **representative strain** per guild (10 files, `data/homd_db/agora_gems/`)

\vspace{0.5em}
"What does this microbe eat and secrete" — predicted from the genome, no culture.

---

## FBA (Flux Balance Analysis) — the basics

**A cook analogy:**

| Element | Maps to |
|---|---|
| Food in the fridge (medium) | saliva composition (Dawes 2008: sugars, amino acids, vitamins) |
| The cook's (microbe's) goal | grow as fast as possible (maximise growth rate mu) |
| What FBA solves | the **flux allocation** achieving that goal, by linear programming |
| Byproducts revealed | secretions (e.g. Streptococcus -> lactate) |

- **pFBA** (parsimonious FBA) = minimal-flux solution -> closer to reality
- assumes steady state and optimal growth. **The sign is trustworthy, the
  magnitude is not** (see later)

---

## From metabolism to the sign of A

Solve each guild's representative strain by pFBA in oral-fluid medium ->
get the **secretion profile** and **uptake capacity**.

```
j secretion flux > +0.05        ->  j secretes metabolite X
i uptake flux    < -0.05        ->  i consumes metabolite X

j secretes X and i eats X       ->  cross-feeding
   is X a toxin (H2O2 / H2S)?
     yes -> neg[i,j] += w   (j harms i -> A[i,j] < 0)
     no  -> pos[i,j] += w   (j helps i -> A[i,j] > 0)

net_flow = pos - sum(neg)  ->  sign(net_flow) = +1 / -1 / 0
```

10 guilds x 9 partners = 90 directed pairs, exhaustively -> signed pairs.

---

## The pipeline at a glance

![](results/fig2_agora_pipeline.png){ height=66% }

(A) the 5-step procedure (B) adding layers grows the constrained pairs
**10 -> 22 -> 58** (+36 from AGORA) (C) the resulting sign-prior matrix sgn(F[i,j]).

---

## The three-layer prior (stacking evidence)

| Layer | Source | Weight | Basis |
|---|---|---|---|
| **L1** | Szafrański Suppl. (experimental + KEGG/HMDB) | 2.0 | direct observation |
| **L1** | Szafrański Suppl. (experimental, unannotated) | 1.5 | direct observation |
| **L2** | Szafrański Suppl. (computational prediction) | 1.0 | prediction |
| **L3** | **AGORA2 FBA cross-feeding** | 1.0 | genome-scale metabolism |

- input: Szafrański's 351 microbe-metabolite rows (PRODUCES / USES / IS\_INHIBITED\_BY ...)
- for each metabolite, add weight to all (producer x consumer) pairs
- e.g. **lactate** -> produced by Bacilli (Strep), consumed by Negativicutes
  (Veillonella) / Actinobacteria

---

## Guild representatives and medium

**Representative strains (AGORA2):** Actino = *A. naeslundii* / Bacilli =
*S. gordonii* / Negat. = *V. parvula* / Bacteroidia = *P. melaninogenica* /
Fusob. = *F. nucleatum* / beta-Prot. = *E. corrodens* ... (10 guilds)

**Oral-fluid medium (Dawes 2008):** sugars, 20 amino acids, B-vitamins, trace
metals (Fe / Mg / Ca / Zn / **Cu**), cell-wall precursors (meso-DAP), quinones,
glutathione.

\vspace{0.4em}
-> **positive growth for all 10 guilds** (mu = 0.11-1.66 /h, Fusobacteriia
highest). A too-poor medium drives mu to ~0, so medium design itself drives the
result — a key sensitivity.

---

## Why only the sign — magnitude priors failed

Every attempt to use the flux **magnitude** as a prior on A broke down.

| Method | Why it failed |
|---|---|
| **MacArthur cosine** (niche overlap) | oral microbes are all generalists; everyone uses sugars/amino acids so cos ~ 1 -> **every pair flagged as competition** |
| **Growth-rate suppression** | in a poor medium microbes barely survive; any depletion by j zeroes i's mu -> **non-specific, all pairs competitive** |

\vspace{0.4em}
**Lesson:** FBA flux units != ecological interaction units.
**The magnitude prior fails; only the sign constraint works** (a deliberate finding).

---

## Single-species pFBA validation (the naive number)

![](results/fig3_agora_sign_validation.png){ height=58% }

FBA-predicted signs vs data-fitted A signs agree at **66/72 = 92%** (91-92% by
layer).
\textcolor{red}{But: this is the *naive* figure and overstates the case — re-examined critically next.}

---

## MICOM — moving to community FBA

Single-species pFBA only checks the **possibility** ("j can secrete, i can eat").
**MICOM** (Diener 2020, mSystems) solves all 10 microbes **together** and checks
whether the flux actually flows.

![](results/fig_agora_v1_v2_micom_comparison.png){ height=52% }

- **cooperative tradeoff** (tau = 0.5): allocate resources so each microbe reaches
  at least tau x its max growth
- even among generalists, **only specific cross-feeding routes** activate

---

## MICOM results

| Method | Sign agreement | Constrained pairs |
|---|:---:|:---:|
| Literature L1+L2 only | 45% (5/11) | 11/45 |
| Single-species pFBA v1 | 88% (29/33) | 33/45 |
| **MICOM (community)** | **100% (36/36)** | **36/45** |

**Lactate cross-feeding shown directly (actual community flux):**
```
Bacilli(Strep) -> Negativicutes(Veillonella)   via EX_lac_L(e)
   secretion +97.7  /  uptake -97.9  mmol/gDW/h
```
FBA reconstructs the classic Streptococcus -> Veillonella route.
(Caveat: the fitted A was estimated under the v1 prior, so 100% may reflect
containment — confirm a real gain via RMSE.)

---

## Sensitivity to the prior weight W

![](results/fig_agora_weight_sensitivity.png){ height=56% }

**A phase transition at W=1.0** — sign agreement 100%, minimal LOO-RMSE (~0.050).
But prior-free gLV (0.0455) is even lower, so **the prior's value is not predictive
accuracy but interpretability / sign-consistency** — stated honestly.

---

## Critical check: is it independently supported?

"92% agreement" overstates, because the **prior is all-positive** (at alpha=0 it is
cross-feeding only). The right control is the positive-rate of **off-prior** cells
(permutation test).

| Model | cross-feeding direction | competition direction |
|---|---|---|
| **Hamilton (symmetric) alpha=0** | **78.6% (11/14) vs random 37.7%, p=0.0004, z=+3.79** | not validated (~chance) |
| gLV (asymmetric) | 41% (null) | null |

\vspace{0.3em}
- **Only the cross-feeding direction is validated**; competition is not supported.
- **The AGORA prior itself is independent of the 16S dynamics** (data do not
  reproduce the prior -> the prior is a modelling choice).
- **Two independent cohorts** (Dieckow x Botelho), fit prior-free, agree on the
  **strong-interaction signs at 89% (p ~ 0.02)** -> the ecological signal is real.

---

## Mechanistic cross-check (COMETS dFBA)

Beyond sign priors, AGORA GEMs also drive a **5-species dynamic FBA (dFBA)**.

![](comets/pipeline_results/sweep_crossfeeding.png){ height=52% }

Healthy: So/An dominate, lactate cross-feeding -> **DI = 0.15**.
Diseased: Pg/Fn expand -> **DI = 0.70**. The same AGORA metabolism reproduces the
commensal<->dysbiotic split in a **forward** simulation — agreement from an
independent route.

---

## Limitations and conclusions

**Limitations**
- guild = class level (representative strain != whole guild); species-level MAGs would help
- 20/22 IS\_INHIBITED\_BY rows are oxygen (no producer) -> effectively dead; only the H2O2 (2 pairs) toxin signal fires
- per-metabolite max weight -> prediction rows get promoted to high confidence
- magnitude is discarded (sign only)

**Conclusions (systematic)**
1. AGORA -> cross-feeding -> **sign prior** is the novelty of this work
2. **Sign usable, magnitude not** (avoiding the MacArthur-type failure)
3. single-species pFBA (92%) -> **MICOM (100%)** captures community context
4. honest validation: **cross-feeding direction only, p=0.0004; two cohorts 89%**;
   the prior itself is a modelling choice
5. COMETS dFBA reproduces commensal<->dysbiotic in the forward direction too
