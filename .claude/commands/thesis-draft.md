---
description: Draft (or restructure) a thesis chapter/section using the knowledge graph + the pro writing guide. WRITING ONLY — no compute, no new results.
---
# /thesis-draft — chapter drafting from the knowledge graph + writing guide

Draft or restructure a thesis section for **`$ARGUMENTS`** (a chapter id like
`ch3` / `ch4` / `ch5`, or a freer target like `ch4 network section` or
`abstract`). This is a **writing task only**: it reads existing results,
the knowledge graph, and the source `.tex`, and produces prose + a figure plan.
It MUST NOT run any compute, submit jobs, dispatch GPU work, or invent new
numbers — every quantitative claim must trace to an existing result file or the
source manuscript. If a number can't be sourced, write `[TODO: source ...]`.

Thesis repo: `/home/nishioka/LUHsummer26/30_Masterarbeit` (chapters in
`chapters/`, bib `references.bib`). Factory repo: `/home/nishioka/IKM_Hiwi/nife`.

## Inputs to load first (in order)

1. **The writing method** — `nife/docs/THESIS_WRITING_GUIDE.md` (canonical). Apply
   its rules: one central-contribution sentence, figure-backbone, CEI paragraphs,
   topic sentence first, important number in the stress position, limitations as a
   finding, reverse-outline at the end.
2. **The knowledge graph** — query the memory MCP for the entities/relations
   relevant to `$ARGUMENTS` (e.g. for `ch4`: `AGORA_sign_prior`, `F_duranpinedo_2cohort`,
   `F_pg_centralises`, `F_veillonella_sink`, the Dieckow/Duran-Pinedo datasets). Use
   `mcp__memory__search_nodes` / `open_nodes` / `read_graph`. The graph is the map
   of which findings/methods/figures belong to which chapter.
3. **The source manuscript** for that chapter (README `Quellen` table):
   - ch3 → `nife/heine_paper/nishioka_heine_paper.tex`
   - ch4 → `nife/dieckow_paper/dieckow_analysis.tex`
   - ch5 → `nife/results/diffusion_fit/*` + FISH outputs + `docs/spatial_pde_slides.md`
   - ch1/ch2/ch6 → `PAPER_OUTLINE.md`, `ANALYSIS_NOTES.md`, `docs/*_slides.md`
4. **The current chapter stub** — `30_Masterarbeit/chapters/<chapter>.tex` (don't
   discard existing structure; build on it).
5. **The bibliography** — `30_Masterarbeit/references.bib` for the right citation
   keys. If a needed work is missing or a key is a `[TODO]` placeholder, FLAG it
   (do not fabricate a citation).

## Procedure

1. **State the chapter's one message** (one sentence) and how it supports the
   thesis central contribution. Show it back before drafting.
2. **Figure plan** — list the figures this section will use (from the writing
   guide's backbone table), one claim each, with the real asset path and a
   one-sentence caption. Mark any figure not yet frozen as `[draft figure]`.
3. **Topic-sentence outline** — one bullet per intended paragraph (the topic
   sentence only). This is the skeleton; get it right before prose.
4. **Draft the prose** in LaTeX, Results paragraphs in **Claim–Evidence–
   Interpretation** form, citing `references.bib` keys. Numbers sourced to result
   files / the source `.tex`; anything unsourced → `[TODO: source ...]`.
5. **Reverse-outline check** — re-print just the topic sentences and confirm the
   logic is a straight line; note any reorder.
6. **Write** the draft into `30_Masterarbeit/chapters/<chapter>.tex` (or a clearly
   named `*_draft.tex` if the user wants to preserve the stub). Do **not** auto-commit
   unless asked — leave it for the user to review, or for `/thesis-sync`.

## Output report

- the one-sentence chapter message + central-contribution link,
- the figure plan (table: figure → claim → asset → caption),
- the topic-sentence outline,
- the path written, and
- an explicit list of every `[TODO: source ...]` / missing-citation flag so the
  user knows exactly what still needs a real number or reference.

Keep German/English/Japanese consistent with the surrounding thesis text
(`main.tex` is English; comments may be JA). No compute, no new results, no
fabricated numbers or citations — ever.
