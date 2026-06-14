# Progress report — HOBIC FISH analysis & dysbiosis model

**To:** Szymon Szafrański
**From:** Keisuke Nishioka (IKM / SFB TRR-298)
**Date:** 2026-06-03

Dear Szymon,

A short update on what I have done with the HOBIC FISH dataset from your lab
(via Nils) and where the peri-implantitis model stands.

## 1. FISH (CLSM) pipeline — done

I built a headless pipeline that turns the Leica `.lif` confocal stacks into
**depth-resolved 5-species composition profiles**, the input for the spatial
(reaction–diffusion) model. All 11 files are processed: commensal (HOBIC22 → CH)
and dysbiotic (HOBIC24 → DH), Days 1/6/10/15/21, on titanium.

The key step was decoding the labelling design. The images have **4 detector
channels but the community has 5 species**, so it is not one colour = one
species. From your Methods (Heine et al. 2025, Table S5 §2.6), **F. nucleatum is
dual-labelled** (same probe sequence, Alexa 405 + Alexa 647), i.e. it appears in
both the blue and red channels. I decode it as the per-voxel colocalisation
`Fn = blue ∩ red`, then `S. oralis = blue − Fn` and `P. gingivalis = red − Fn`
(A. naeslundii = green, V. dispar/parvula = yellow). Doing the colocalisation
**before** the xy-average is essential. A naïve "blue = So / red = Pg" decode
would have dropped Fn entirely and over-counted So and Pg.

## 2. Spatial-ecology findings (independent of the fit)

The depth profiles and voxel decode already give a consistent picture —
**dysbiosis is a spatial reorganisation, not a bulk compositional shift**:

- **P. gingivalis sinks into the deep anaerobic layers in DH** from Day 6 on
  (centre-of-mass up to +30 µm deeper than CH).
- **The F. nucleatum–P. gingivalis "bridge" is an early-only effect**: Fn–Pg
  colocalisation (Manders M1) is high in DH at Day 1 (0.76, early co-colonisation)
  but then *lower* than CH from Day 6 (0.15–0.20 vs CH 0.33–0.49) — Pg decouples
  from Fn and goes autonomous.
- **Bulk composition CH ≈ DH** (CH/DH Bray–Curtis divergence ≈ 0.2 and flat from
  Day 1), so the dysbiosis signal is spatial/temporal, not bulk abundance.
- Mapping the 5 species to the in-vivo Dieckow guilds, the rank order agrees
  (**Spearman ρ = 0.70**); the in-vivo community is S. oralis–dominated whereas
  the defined inoculum is even, as expected.

## 3. Diffusion fit — running

Using the depth profiles I fit per-species diffusivities `D_i` and an advection
`u` through the reaction–diffusion PDE (the gLV reaction term is fixed from the
bulk time-series; only the spatial-transport parameters are free). The
production fits and a hyperparameter sweep are running on our HPC now; current
estimates are preliminary (the optimiser has not fully converged on the noisier
late-day DH profiles), and I will send the final numbers once the sweep settles.

## 4. In-vivo interaction model (Dieckow 16S) — progress & discussion

In parallel I fit the guild–guild interaction matrix on the Dieckow longitudinal
cohort (10 patients × 3 weeks, 10 class-level guilds) with the replicator gLV /
Hamilton ODE, regularised by the AGORA2 metabolic **sign priors** (your L1/L2
interactions + L3 FBA cross-feeding). Best model so far: LOO-RMSE ≈ 0.049 with
the expanded-AGORA prior. The more interesting part is the honesty checks I ran
on whether that prior is doing real work:

- **The cross-feeding direction is independently validated — but only that
  direction.** With the prior switched off (pure data fit), the *cooperative*
  (cross-feeding) cells of the fitted matrix agree with the AGORA signs at
  **78.6 % vs a 37.7 % random-cell baseline (permutation p = 0.0004)**. The
  *competition* direction, by contrast, is at chance — so I will state explicitly
  that the metabolism–ecology agreement holds for cross-feeding, not competition.
- **The AGORA prior is not, by itself, reproduced by the 16S dynamics.**
  Data-driven signs vs the prior sit around chance, so the metabolic prior is a
  *modelling choice* that is independent of the abundance dynamics, not something
  the 16S data confirms on its own. I think this is the correct, reviewer-proof
  framing.
- **Two independent cohorts agree prior-free.** Fitting Dieckow and Duran-Pinedo 2021
  separately with no prior, the signs of the **strong** directed interactions
  match at **≈ 89 % (p ≈ 0.02)**, with a consistent Actinobacteria axis. This —
  not the prior — is the credible cross-validation of the ecological signal.
- **Network view — the metabolic backbone is real, the textbook keystone is not.**
  The concordant ecology×metabolism backbone reconstructs the known lactate
  cross-feeding with **Veillonella (Negativicutes) as the main sink**, which is
  the result I am most confident in. At the same time, the classic
  *P. gingivalis keystone / F. nucleatum bridge* picture is **not** supported at
  the in-vivo class level (Pg is only mid-ranked in eigen-centrality; the
  structural bridge is Streptococcus, not Fusobacterium).
- **Link to the in-vitro HOBIC data.** Pg only becomes central in the *dynamic
  dysbiotic state* (eigen-centrality 0.32 → 0.51 from commensal to dysbiotic),
  and the Streptococcus–Veillonella mutualism flips to competition in dysbiosis.
  Both dovetail with the FISH finding above (Pg goes deep and decouples from Fn
  as dysbiosis develops) — the static network and the spatial data tell the same
  story. Note the converse caution: the raw interaction *signs* do differ between
  the in-vitro and in-vivo fits, so I treat the in-vitro matrix as a guide to
  *which pairs* matter, not as a prior on their signs.

## 5. GDI / Joshi 2025 validation — in progress

In parallel I am validating the Guild Dysbiosis Index against the Joshi et al.
2025 cohort (PRJNA1192962), processing the full-length 16S to add diagnostic
genera (Treponema, Tannerella, Prevotella). This is **waiting on the clinical
metadata** (sample ID → Health / Mucositis / Peri-implantitis), which I asked
about separately — a processed genus-level table from your QIIME2 pipeline would
be ideal if it exists.

## 6. What would help next (FISH side)

- The **original φ time-series** behind the four ODE attractors (CS / CH / DS / DH),
  if available, to tie the in-vitro and in-vivo fits together.
- A few more **DH Day 15 / 21 titanium FOVs** — those late days currently rest on
  only 2 Ti FOVs each (the 241018 file mixes Ti and glass; I kept Ti for a
  same-substrate CH-vs-DH comparison), so their statistics are weak.
- Any **Day 3** acquisitions, to fill the gap between Day 1 and Day 6.

## Deliverables

A full report and slide deck (EN + JA), with all figures, are available — happy
to share the PDFs or walk through them whenever convenient.

Thank you again for the FISH data; please let me know if any of the
interpretation above looks off relative to how the experiments were run.

Best regards,
Keisuke
