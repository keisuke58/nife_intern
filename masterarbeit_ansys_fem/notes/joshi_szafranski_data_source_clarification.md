# Data-source clarification — "Joshi" 127-sample external validation = Szafrański (2026-06-19)

The historical "Joshi" external-validation analysis (`scripts/analysis/joshi_attractor_analysis.py`,
`scripts/loo_cv/joshi_Amatrix_validation.py`, `scripts/analysis/joshi_gdi_improved.py`,
`results/dieckow_cr/joshi_attractor_results.csv`, paper section §3 "External validation: Joshi
attractor analysis") loads `Datasets/20260416_mSystems_16S_5genera_all_profiles.xlsx`. That file is
the **Szafrański et al. 2025 mSystems** dataset — 127 cross-sectional peri-implant samples × 5
genera × Health/Mucositis/Peri-implantitis. It is NOT a Joshi dataset.

The two actual Joshi peri-implantitis datasets are:

| paper | cohort | sequencing | groups |
|---|---|---|---|
| Joshi et al. 2025, npj Biofilms & Microbiomes 11:175 | 48 samples × 32 patients | full-length 16S + metatranscriptome | Health vs. Peri-implantitis (**no Mucositis**) |
| Joshi et al. 2026, J Dent Res (PMC12861548) | 49 PI implants × 34 patients | submucosal biofilm | severity scores (PD, MBL, BoP) (**no Health**) |

Neither matches the 127 H/M/PI × 5 genera ingestion logic of the historical scripts; both can be
correctly cross-referenced once their supplementary tables are obtained from PMC.

## Actions taken (this commit)

1. **Manuscript citations.** The four `\citep{joshi2025}` occurrences in
   `dieckow_paper/dieckow_analysis.tex` that referred to the 127-sample dataset are corrected to
   `\citep{szafranski2025}`:
   - line 83 (Abstract): "External validation on 127 independent cross-sectional peri-implant samples"
   - line 144 (Introduction summary): "validate the inferred interaction matrix on an independent
     peri-implant dataset"
   - line 425 (Methods §2 subsection title): "External validation: Joshi attractor analysis" →
     "External validation: Szafrański peri-implant attractor analysis" (label also renamed
     `ssec:joshi` → `ssec:szafranski_attractor`)
   - line 1360 (Discussion: Outlook): "Long-time R/G distribution vs. independent peri-implant
     dataset of Joshi et al." → "Szafrański et al."
   - The single occurrence at line 1402 (a general reference to "implant dysbiosis" being different
     from periodontitis) is legitimately a Joshi 2025 npj reference and is left unchanged.

2. **Script header notices.** A data-provenance NOTE has been added to the docstrings of
   `joshi_attractor_analysis.py` and `joshi_Amatrix_validation.py` clarifying that the loaded data
   is Szafrański. The filenames themselves are **not** renamed to avoid breaking the multiple
   downstream `fig_joshi_*` figure pipelines (see `grep -rn "joshi"` in scripts/figures/); a future
   refactor can rename when the call graph is stable.

3. **Real Joshi validation (queued).** A placeholder script
   `scripts/loo_cv/joshi_jdr2026_severity_validation.py` is committed, ready to consume the JDR 2026
   per-sample composition + severity tables once they are extracted from PMC12861548
   supplementary materials. The script computes R/G ratio per sample and reports Spearman
   correlations against three severity axes (PD, MBL, BoP) with a 3-panel scatter figure. Its
   docstring contains a full DATA ACQUISITION CHECKLIST.

## Action items still required

- [ ] Download Joshi JDR 2026 supplementary tables from PMC12861548 (open access).
- [ ] Map the supplementary genus columns onto the 10-guild ontology.
- [ ] Run `joshi_jdr2026_severity_validation.py` and verify R/G ↔ severity Spearman ρ ≳ 0.4.
- [ ] If ρ is in that range, add a one-paragraph subsection to §3 of the manuscript ("True Joshi
      external validation: severity correlation"). The existing Szafrański validation becomes the
      categorical-discrimination check; Joshi 2026 becomes the continuous-severity check.
- [ ] Optional / lower priority: rename `joshi_*.py` → `szafranski_*.py` once
      `joshi_jdr2026_severity_validation.py` has produced its own outputs and no more
      cross-references are pointing at the old name.

## Open question (escalate if needed)

If the supplementary materials of Joshi JDR 2026 do NOT contain per-sample taxonomic compositions,
the corresponding author (Szafrański / Stiesch — both already on the project's contact list) should
be asked directly. The thread is already open via the FISH-data request email; the JDR 2026 ask is
a single-line follow-up.

The Joshi 2025 npj (n=48, Health vs. PI) is the secondary fallback target: it gives a categorical
discrimination check on a different cohort than Szafrański. Less informative than the JDR 2026
severity correlation but still a true Joshi validation.
