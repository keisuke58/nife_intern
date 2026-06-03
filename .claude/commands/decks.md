---
description: Rebuild slide decks (JA+EN beamer) and re-upload PDFs to Drive
---
Rebuild the project slide decks and re-upload the PDFs to `gdrive:NIFE_slides/`.

Scope: `$ARGUMENTS` — a space-separated list of deck stems to rebuild (e.g. `overview agora`).
If empty, rebuild ALL of: overview_slides agora_slides dieckow_slides network_slides spatial_pde_slides defense_slides.

For each stem, build BOTH languages from the repo root, then upload:

```bash
export PATH="$HOME/texlive/2025/bin/x86_64-linux:/usr/bin:/bin"
export TEXMFHOME=/tmp/empty_texmf; mkdir -p "$TEXMFHOME"
STIX=/home/nishioka/texlive/2025/texmf-dist/fonts/opentype/public/stix2-otf
# JA
pandoc docs/<stem>.md -t beamer -s -V mainfont="Noto Serif CJK JP" -V monofont="Noto Sans Mono CJK JP" -V CJKmainfont="Noto Serif CJK JP" -V CJKmonofont="Noto Sans Mono CJK JP" -o docs/<stem>.tex
xelatex -interaction=nonstopmode -output-directory=docs docs/<stem>.tex   # run twice
# EN
pandoc docs/<stem>_EN.md -t beamer -s -V mainfont="STIXTwoText-Regular.otf" -V mainfontoptions="Path=$STIX/, BoldFont=STIXTwoText-Bold.otf, ItalicFont=STIXTwoText-Italic.otf, BoldItalicFont=STIXTwoText-BoldItalic.otf" -V monofont="DejaVu Sans Mono" -H docs/_en_symbols.tex -o docs/<stem>_EN.tex
xelatex -interaction=nonstopmode -output-directory=docs docs/<stem>_EN.tex   # run twice
```

Then: grep each pass-2 log for `Missing character` / `! LaTeX Error` / `Overfull \vbox` and report counts + page counts. If clean, `rclone copy docs/<stem>.pdf gdrive:NIFE_slides/` (and `_EN.pdf`). Use `(1)` not the circled digit in EN. Report a short table of stem → pages → (JA/EN clean?).