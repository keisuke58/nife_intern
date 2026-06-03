---
description: arXiv literature-watch — search the project's themes, append a few NEW papers to docs/reading_list.md
---
On-demand literature scan for this project's modelling themes. This is a **gentle, one-shot** command: a
handful of results per theme, summarise only the few most relevant NEW ones, append them to the reading list,
then stop. **Do NOT loop, schedule, or poll** — run once when invoked. This is a shared research-lab server, so
keep it light (read-only + one file append; no cluster jobs, no heavy compute, no network hammering).

## Tool to use

Search via the arXiv MCP. The search tool is **deferred** — fetch its schema first:

```
ToolSearch  query="select:mcp__arxiv__search_papers"
```

Then call `mcp__arxiv__search_papers` for each theme below. Ask for only a small number of results per query
(≈5, newest first if the tool supports sorting). Optionally use `mcp__arxiv__read_paper` to confirm a single
borderline abstract — but do not bulk-download.

## Query themes

`$ARGUMENTS` overrides the themes: if non-empty, treat it as ONE custom query (or a `;`-separated list of queries)
and search only those. If empty, search these six seed themes:

1. `cross-diffusion biofilm pattern formation`
2. `generalized Lotka-Volterra interaction inference microbiome`
3. `physics-informed neural network inverse PDE reaction-diffusion`
4. `oral microbiome dysbiosis peri-implantitis modelling`
5. `genome-scale metabolic community MICOM cross-feeding`
6. `replicator dynamics ecology`

## Steps

1. Read `docs/reading_list.md` and collect every arXiv id already listed (so you can skip duplicates — across
   BOTH the curated references and the auto-appended section).
2. For each theme, call `mcp__arxiv__search_papers` (small result count). Keep only papers that are (a) genuinely
   relevant to one of the modelling pillars (ecological gLV/replicator inference, spatial reaction-diffusion /
   cross-diffusion PDE + PINN inverse problems, genome-scale metabolic / cross-feeding communities, oral biofilm
   / peri-implantitis), and (b) **NEW** — arXiv id not already in the file. Drop the rest. Prefer recent papers;
   if nothing new and relevant turns up for a theme, that's fine — skip it.
3. De-duplicate the survivors across themes. Aim for roughly the 3–8 most relevant overall; do not pad.
4. **Append** (never rewrite the existing content) to `docs/reading_list.md`, under the
   `## New (auto-appended by /litwatch)` section, a new dated subheading:

   ```markdown
   ### YYYY-MM-DD
   - **<Title>** — <First Author> et al. (<year>). <one line: why it matters to this project>. arXiv:<id> · theme: <which seed theme>
   ```

   Use today's date for `YYYY-MM-DD`. One bullet per paper, the why-it-matters tied to a concrete project pillar
   (e.g. "supports the L3 AGORA sign-prior", "alternative to solve_ivp gLV inference", "PINN inverse-PDE for the
   Heine HOBIC fit"). Keep arXiv ids in the `arXiv:NNNN.NNNNN` form so the dedup in step 1 keeps working.
5. If a theme returned nothing new, do not write an empty heading for it. If the whole run found nothing new,
   append a single line under a dated subheading noting "no new relevant papers" rather than leaving the run silent.
6. Report a short summary: per-theme count of new papers kept, and the total appended.

Do not commit — leave the edited file in the working tree for the user to review.
