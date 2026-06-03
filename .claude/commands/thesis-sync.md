---
description: One-command thesis sync — regenerate LIGHT data figures, rebuild all JA+EN decks, lint logs, commit/push, upload PDFs
---
# /thesis-sync — LOCAL LIGHT BUILD ONLY. NO CLUSTER. NO GPU.

This command runs entirely on the **login node** with **light local tools only**
(matplotlib for figures, pandoc→xelatex for slides). It MUST NOT submit PBS/`qsub`
jobs, dispatch GPU work to `vancouver01`, run `fish_3d_batch.py` or any FBA/COMETS
job, `git stash`/`git pull` the shared checkout, or do anything heavy on this
SHARED LAB SERVER. If a step looks like it would do heavy compute, STOP and report
instead of running it.

Scope: `$ARGUMENTS` — optional space-separated list of deck stems to restrict the
deck rebuild to (e.g. `overview spatial_pde`). Accepted bare names map to the
`docs/<name>_slides{,_EN}.md` files: `overview agora dieckow network spatial_pde
defense`. If `$ARGUMENTS` is empty, rebuild ALL six. The figure regen in step (a)
always runs regardless of `$ARGUMENTS` (it is cheap), unless `$ARGUMENTS` is
exactly `decks-only`, in which case skip step (a).

Run every command **from the repo root** (`/home/nishioka/IKM_Hiwi/nife`).

---

## (a) Regenerate the LIGHT data figures (matplotlib only — NO cluster, NO GPU)

These are quick local matplotlib scripts. Run them in order; report any traceback
but keep going:

```bash
python3 scripts/figures/make_pipeline_overview.py
python3 scripts/figures/make_concept_diagram.py
python3 scripts/analysis/plot_depth_profiles.py
python3 scripts/analysis/analyze_depth_niche.py
python3 scripts/analysis/spatial_crossfeeding.py
```

DO NOT run `scripts/analysis/fish_3d_batch.py` here — it re-decodes ~84 .lif FOVs
(tens of minutes, heavy I/O + memory). It is intentionally excluded from the sync;
use the `/fish3d` command in the background if you actually need it. Likewise DO
NOT run the AGORA/FBA figure generators (`generate_fig127.py`,
`generate_fig3456.py`) here — they need the COBRApy/AGORA stack and are not part of
the light local path; see `docs/PROVENANCE.md`.

## (b) Rebuild ALL decks, JA + EN (pandoc → xelatex, twice each)

Use this exact texlive incantation (a clean PATH + an empty TEXMFHOME so no stray
user texmf leaks in). EN decks use the STIX-Two fonts and the shared symbol header
`docs/_en_symbols.tex` (which maps circled digits etc. to plain `(1)` so STIX has
the glyphs):

```bash
export PATH="$HOME/texlive/2025/bin/x86_64-linux:/usr/bin:/bin"
export TEXMFHOME=/tmp/empty_texmf; mkdir -p "$TEXMFHOME"
STIX=/home/nishioka/texlive/2025/texmf-dist/fonts/opentype/public/stix2-otf
```

For each stem in the resolved set (default: `overview agora dieckow network
spatial_pde defense`), the file stems are `<stem>_slides` and `<stem>_slides_EN`.
Build BOTH languages, running `xelatex` **twice** per language (TOC / nav refs):

```bash
S=overview_slides   # = <stem>_slides

# --- JA ---
pandoc docs/$S.md -t beamer -s \
  -V mainfont="Noto Serif CJK JP" -V monofont="Noto Sans Mono CJK JP" \
  -V CJKmainfont="Noto Serif CJK JP" -V CJKmonofont="Noto Sans Mono CJK JP" \
  -o docs/$S.tex
xelatex -interaction=nonstopmode -output-directory=docs docs/$S.tex
xelatex -interaction=nonstopmode -output-directory=docs docs/$S.tex

# --- EN ---
pandoc docs/${S}_EN.md -t beamer -s \
  -V mainfont="STIXTwoText-Regular.otf" \
  -V mainfontoptions="Path=$STIX/, BoldFont=STIXTwoText-Bold.otf, ItalicFont=STIXTwoText-Italic.otf, BoldItalicFont=STIXTwoText-BoldItalic.otf" \
  -V monofont="DejaVu Sans Mono" \
  -H docs/_en_symbols.tex \
  -o docs/${S}_EN.tex
xelatex -interaction=nonstopmode -output-directory=docs docs/${S}_EN.tex
xelatex -interaction=nonstopmode -output-directory=docs docs/${S}_EN.tex
```

## (c) Lint the build logs

For each pass-2 `docs/<stem>_slides{,_EN}.log`, grep and count the three failure
classes, and report a small table (stem → JA pages / EN pages → issue counts):

```bash
grep -c "Missing character"  docs/<log>
grep -c "! LaTeX Error"      docs/<log>
grep -c "Overfull \\\\vbox"  docs/<log>
```

`Missing character` usually means the chosen font lacks a glyph (most often an EN
deck hitting a CJK/symbol char that should be handled via `docs/_en_symbols.tex`);
`! LaTeX Error` is a hard build problem; `Overfull \vbox` is a slide that runs off
the bottom. Report counts; only flag as needing attention if `! LaTeX Error > 0`
or the PDF page count is 0 / clearly wrong.

## (d) Commit & push (only the regenerated artefacts)

Stage just what this run regenerated, commit, and push the current branch. Do NOT
`git stash` or `git pull` (shared checkout). Stage the changed deck sources and the
regenerated figure / sweep artefacts:

```bash
git add docs/*_slides*.md docs/*_slides*.tex \
        results/figures/*.png \
        results/diffusion_fit/{depth_niche.png,ch_dh_divergence.png,spatial_crossfeeding.png,zprofiles_all_ti_*.png,sweep_summary.csv}
git status        # eyeball before committing
git commit -m "docs: thesis-sync — regen light figures + rebuild decks (JA+EN)"
git push
```

Note: compiled deck **PDFs** and LaTeX junk (`*.aux/.log/.nav/.snm/.toc/.vrb/.out`)
are gitignored — do not force-add them. If `git status` shows unexpected large or
unrelated files, STOP and report rather than committing them.

## (e) Upload the deck PDFs to Drive

For each stem actually built, copy the PDF(s) to the Drive folder:

```bash
rclone copy docs/<stem>_slides.pdf    gdrive:NIFE_slides/
rclone copy docs/<stem>_slides_EN.pdf gdrive:NIFE_slides/
```

---

## Final report

Print a short summary table:

| deck | JA pages | EN pages | Missing char | LaTeX Error | Overfull vbox | uploaded |
|------|---------:|---------:|-------------:|------------:|--------------:|:--------:|

plus: which figures were regenerated, the commit hash that was pushed, and an
explicit reminder line that `fish_3d_batch` and the AGORA/FBA figures were
**skipped on purpose** (heavy / need extra stacks — see `docs/PROVENANCE.md`).
