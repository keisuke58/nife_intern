# DRAFT — data follow-up to Szymon Szafrański (NOT SENT)

> ⚠️ LOCAL DRAFT ONLY. Do not auto-send / do not push to Gmail (per project policy).
> Verify the email address and the exact supplementary-table names before sending.
> To: Szafranski.Szymon@mh-hannover.de  (confirm)
> Cc: (supervisor, if appropriate)
> Subject: Follow-up: data for the biofilm community-dynamics modelling (S5/S8 + metatranscriptome + in-vitro)

---

Dear Szymon,

I hope you are well. As I have about two months left in the NIFE internship, I would like to make the most of the remaining time on the community-dynamics modelling, and a few datasets from your side would unlock the most valuable next steps. Sorry to come back with a list — I have grouped it by priority and tried to be specific so it is easy for you to judge what is feasible.

**1. Following up on my April request (peri-implant cross-sectional cohort)**
- **Supplementary Table S5** — the full per-sample species abundance matrix (≈756 species × 125 samples). This is the key input for the out-of-sample test of the ODE attractors against the observed community types. *(Most important.)*
- **Supplementary Table S8** — the class-level EC / enzyme-activity table, if it exists in a shareable form.
- **Per-sample metadata** — sample/patient ID, diagnosis (PIH/PIM/PI), and, if available, probing depth / BoP and the community-type (CT I–IV) assignment per sample.

**2. For an independent functional check of our metabolic interaction priors**
Our sign priors come from the Dieckow Supplementary File 1. To validate them *independently*, I would like to use the Joshi et al. 2025 metatranscriptome (PRJNA1192962). The public supplement only gives group means, so if possible:
- the **per-sample EC × taxon (class-level) count matrix** behind Supplements F/H (rows = (class, 4-digit EC), columns = the 116 samples, raw or CPM, with health/peri labels), and
- the **eggNOG gene → EC → taxon assignment table**, which would also give us an annotation-independent metabolite→EC bridge.

**3. In-vitro interaction data (highest scientific leverage)**
Any raw pairwise interaction measurements — growth / inhibition / co-culture outcomes from the organotypic or co-culture system — would let us validate the *direction* of the inferred interactions directly, rather than only against AGORA/literature.

**4. The 2025 cross-sectional preprint**
Could you confirm the current accession/DOI and publication status of the 2025 peri-implant ecosystem manuscript? I want to cite it correctly and make sure I am using the right version of the data.

I am happy to receive any of these in whatever format is easiest (raw exports are fine — I will do the parsing), and to send a precise column specification if that helps. Thank you very much for your help, and please let me know if any of these are not feasible.

Best regards,
Keisuke Nishioka

---

### Internal notes (not part of the email)
- Public, download-without-asking: **PRJEB71108** (Dieckow longitudinal 16S), **PRJNA1192962** (Joshi 16S+metatranscriptome). Pull these first; only the *processed/per-sample* tables and in-vitro data are request-only.
- Item 2 mirrors `results/prior_metatx_validation/DATA_REQUEST.md` (from the metatranscriptome-validation workflow) — keep the two specs consistent.
- Item 1 supersedes the 2026-04-17 request in `szafranski_data_plan.md` (this is the polite follow-up).
- See `docs/szafranski_papers_data_investigation_2026-06-04.md` for the full provenance.
