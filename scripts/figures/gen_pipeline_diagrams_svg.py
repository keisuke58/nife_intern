#!/usr/bin/env python3
"""Generate SVG versions of the four project structure diagrams.

Vector twins of ``docs/pipeline_diagrams.tex`` (same content/palette), emitted
as standalone SVG so they can be rasterised to PNG without a LaTeX toolchain
(see ``scripts/figures/render_diagrams_png.sh``) and edited by hand.

    python scripts/figures/gen_pipeline_diagrams_svg.py [--outdir docs/diagrams]

Writes: fig1_pillars.svg, fig2_pipeline.svg, fig3_sign_prior.svg,
fig4_attractors.svg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

# palette mirrors the beamer decks (blue!15 / orange!20 / teal / green / violet)
PAL = {
    "data": "#dce7f7", "proc": "#fce3c8", "model": "#d3ecea",
    "out": "#daf0d6", "prior": "#e7dcf2", "gray": "#eaeaea", "red": "#f7d6d6",
}
STROKE = "#4a4a4a"
FONT = "Helvetica, Arial, sans-serif"


def _wrap(lines):
    return lines if isinstance(lines, (list, tuple)) else [lines]


def box(x, y, w, h, fill, lines, *, bold_first=True, fs=15, rx=8):
    """Rounded rect centred text block. (x,y) is top-left."""
    els = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
           f'fill="{fill}" stroke="{STROKE}" stroke-width="1.5"/>']
    lines = _wrap(lines)
    lh = fs + 5
    total = lh * len(lines)
    cy = y + h / 2 - total / 2 + fs
    cx = x + w / 2
    for i, ln in enumerate(lines):
        weight = "bold" if (i == 0 and bold_first) else "normal"
        size = fs if i == 0 else fs - 3
        els.append(
            f'<text x="{cx}" y="{cy + i * lh}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="#1a1a1a" '
            f'text-anchor="middle">{escape(ln)}</text>')
    return "\n".join(els)


def arrow(x1, y1, x2, y2, *, dashed=False, curve=None, label=None,
          label_dx=0, label_dy=-6):
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    if curve is None:
        path = f'M {x1} {y1} L {x2} {y2}'
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    else:
        cx, cy = curve
        path = f'M {x1} {y1} Q {cx} {cy} {x2} {y2}'
        mx, my = 0.25 * x1 + 0.5 * cx + 0.25 * x2, 0.25 * y1 + 0.5 * cy + 0.25 * y2
    out = [f'<path d="{path}" fill="none" stroke="{STROKE}" '
           f'stroke-width="1.8"{dash} marker-end="url(#arw)"/>']
    if label:
        out.append(
            f'<text x="{mx + label_dx}" y="{my + label_dy}" font-family="{FONT}" '
            f'font-size="11" font-style="italic" fill="#333" '
            f'text-anchor="middle">{escape(label)}</text>')
    return "\n".join(out)


def svg(w, h, body, title=None):
    t = (f'<text x="{w/2}" y="26" font-family="{FONT}" font-size="17" '
         f'font-weight="bold" fill="#111" text-anchor="middle">{escape(title)}</text>'
         if title else "")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>
  <marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{STROKE}"/>
  </marker>
</defs>
<rect width="{w}" height="{h}" fill="white"/>
{t}
{body}
</svg>
'''


# --------------------------------------------------------------- figure 1 ----
def fig_pillars():
    W, H = 900, 560
    els = []
    els.append(f'<rect x="30" y="40" width="{W-60}" height="{H-70}" rx="12" '
               f'fill="none" stroke="{STROKE}" stroke-width="1.4" '
               f'stroke-dasharray="7,5"/>')
    els.append(box(280, 70, 340, 88, PAL["gray"], [
        "Shared contract: 10-guild taxonomy",
        "GUILD_ORDER (guild_replicator_dieckow.py)",
        "positional index for every .npy / .json"]))
    els.append(box(60, 250, 250, 100, PAL["data"], [
        "3. 16S / metadata preprocessing",
        "vsearch+SILVA, QIIME2, MetaPhlAn",
        "-> guild phi array (N x T x 10)"]))
    els.append(box(325, 320, 250, 100, PAL["model"], [
        "1. Ecological ODE inference (paper)",
        "gLV / Hamilton replicator fit",
        "sign-prior regularised, LOO-CV"]))
    els.append(box(590, 250, 250, 100, PAL["out"], [
        "2. COMETS / dFBA",
        "5-species dynamic FBA",
        "cross-validates interaction signs"]))
    els.append(arrow(450, 158, 185, 250))
    els.append(arrow(450, 158, 450, 320))
    els.append(arrow(450, 158, 715, 250))
    els.append(arrow(230, 350, 325, 372, curve=(255, 400), label="guild phi",
                     label_dy=22))
    els.append(arrow(670, 350, 575, 372, curve=(645, 400), dashed=True,
                     label="AGORA2 signs", label_dy=22))
    els.append(f'<text x="{W/2}" y="60" font-family="{FONT}" font-size="16" '
               f'font-weight="bold" fill="#111" text-anchor="middle">'
               f'Oral biofilm community-dynamics modelling</text>')
    return svg(W, H, "\n".join(els))


# --------------------------------------------------------------- figure 2 ----
def fig_pipeline():
    W, H = 820, 600
    els = []
    # top chain
    els.append(box(60, 60, 190, 74, PAL["data"], ["Raw 16S reads", "ENA / SRA"]))
    els.append(box(300, 60, 210, 74, PAL["proc"], [
        "vsearch", "merge->QC->chimera", "-> SILVA classify"]))
    els.append(box(560, 60, 200, 74, PAL["data"], [
        "Guild phi array", "N_pat x T x 10"]))
    # inference + inputs (inputs stacked in one column, left of LOO)
    els.append(box(560, 250, 200, 100, PAL["model"], [
        "gLV / Hamilton LOO-CV", "loo_cv_kegg_prior.py"]))
    els.append(box(280, 235, 210, 80, PAL["prior"], [
        "Metabolic sign prior", "L1/L2 Szafranski + L3 AGORA2",
        "constrains sign of A_ij"]))
    els.append(box(280, 345, 210, 74, PAL["model"], ["fit_*.json", "A matrix + b_k"]))
    # outputs
    els.append(box(280, 445, 200, 74, PAL["out"], ["LOO-RMSE, BC", "per-patient"]))
    els.append(box(560, 445, 200, 74, PAL["out"], ["Paper figures", "vector PDF"]))
    # edges
    els.append(arrow(250, 97, 300, 97))
    els.append(arrow(510, 97, 560, 97))
    els.append(arrow(660, 134, 660, 250))
    els.append(arrow(490, 272, 560, 285))
    els.append(arrow(490, 380, 560, 330))
    els.append(f'<text x="512" y="398" font-family="{FONT}" font-size="11" '
               f'font-style="italic" fill="#333" text-anchor="middle">--fit-file</text>')
    els.append(arrow(660, 350, 660, 445))
    els.append(arrow(570, 345, 400, 445, curve=(470, 420)))
    els.append(arrow(480, 482, 560, 482))
    return svg(W, H, "\n".join(els),
               title="16S -> guild phi -> LOO-CV data pipeline")


# --------------------------------------------------------------- figure 3 ----
def fig_sign_prior():
    W, H = 960, 400
    els = []
    els.append(box(40, 70, 300, 64, PAL["prior"], [
        "L1  experimental interactions",
        "Szafranski measured metabolic flows"]))
    els.append(box(40, 162, 300, 64, PAL["prior"], [
        "L2  predicted interactions",
        "KEGG / HMDB inferred cross-feeding"]))
    els.append(box(40, 254, 300, 64, PAL["prior"], [
        "L3  AGORA2 FBA cross-feeding (opt.)",
        "build_net_flow_expanded.py net flux"]))
    els.append(box(410, 154, 230, 80, PAL["proc"], [
        "Signed net-flow matrix",
        "w_ij = |net_ij|  (confidence)",
        "s_ij = sign(net)_ij"]))
    els.append(box(700, 148, 230, 92, PAL["model"], [
        "Sign penalty on A",
        "w_ij/(2s^2) * max(0, -s_ij A_ij)^2",
        "constrains sign, never magnitude"]))
    els.append(arrow(340, 102, 410, 178))
    els.append(arrow(340, 194, 410, 194))
    els.append(arrow(340, 286, 410, 212))
    els.append(arrow(640, 194, 700, 194))
    els.append(f'<text x="815" y="262" font-family="{FONT}" font-size="11" '
               f'font-style="italic" fill="#333" text-anchor="middle">'
               f'sigma = 0.15</text>')
    return svg(W, H, "\n".join(els),
               title="Sign-prior layering constrains interaction matrix A")


# --------------------------------------------------------------- figure 4 ----
def fig_attractors():
    W, H = 720, 600
    els = []
    ox, oy = 120, 470          # axis origin
    els.append(arrow(ox, oy, ox, 90))
    els.append(arrow(ox, oy, 660, oy))
    els.append(f'<text x="{ox+6}" y="86" font-family="{FONT}" font-size="12" '
               f'fill="#333" text-anchor="start">dysbiosis</text>')
    els.append(f'<text x="400" y="{oy+30}" font-family="{FONT}" font-size="12" '
               f'fill="#333" text-anchor="middle">HOBIC (flow / oxygen)</text>')
    qw, qh = 230, 150
    els.append(box(150, 120, qw, qh, PAL["out"], [
        "CS", "Commensal - Static", "healthy, low-flow"]))
    els.append(box(410, 120, qw, qh, PAL["model"], [
        "CH", "Commensal - HOBIC", "healthy, high-flow"]))
    els.append(box(150, 290, qw, qh, PAL["proc"], [
        "DS", "Dysbiotic - Static", "disease, low-flow"]))
    els.append(box(410, 290, qw, qh, PAL["red"], [
        "DH", "Dysbiotic - HOBIC", "disease, high-flow"]))
    return svg(W, H, "\n".join(els),
               title="The four ODE attractors (Heine 2025)")


FIGS = {
    "fig1_pillars": fig_pillars,
    "fig2_pipeline": fig_pipeline,
    "fig3_sign_prior": fig_sign_prior,
    "fig4_attractors": fig_attractors,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/diagrams")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        p = out / f"{name}.svg"
        p.write_text(fn(), encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
