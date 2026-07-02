# Project structure diagrams

Vector (`*.svg`) and raster (`*.png`) versions of the four structure diagrams,
the same content as `docs/pipeline_diagrams.tex` (TikZ).

| file | diagram |
|------|---------|
| `fig1_pillars`     | three modelling pillars sharing the 10-guild taxonomy contract |
| `fig2_pipeline`    | 16S → guild φ → gLV/Hamilton LOO-CV data pipeline |
| `fig3_sign_prior`  | sign-prior layering (L1/L2/L3) constraining interaction matrix A |
| `fig4_attractors`  | the four ODE attractors (CS / CH / DS / DH) |

## Regenerate

```bash
python scripts/figures/gen_pipeline_diagrams_svg.py   # writes the .svg sources
scripts/figures/render_diagrams_png.sh                # rasterises .svg -> .png (2x)
```

The PNGs are produced with headless Chromium (no LaTeX/ImageMagick needed);
override the browser with `CHROME=/path/to/chrome`. Edit the `.svg` sources
(or the generator) and re-run the render step. `docs/pipeline_diagrams.tex`
is the LaTeX/TikZ twin for `\input` into the beamer decks or the manuscript.
