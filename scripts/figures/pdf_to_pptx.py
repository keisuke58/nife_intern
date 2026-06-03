#!/usr/bin/env python3
"""
pdf_to_pptx.py — convert slide PDFs to PowerPoint (.pptx), one full-bleed image
per slide (Ghostscript raster + python-pptx). For beamer/dzslides decks this
preserves the exact visual layout (images are not re-flowed / re-typeset).

  python scripts/figures/pdf_to_pptx.py FILE.pdf [FILE2.pdf ...] [--dpi 170] [--outdir docs/pptx]
  python scripts/figures/pdf_to_pptx.py --all     # all docs/*_slides*.pdf + the Heine deck

Note: slides become images (not editable text). Aspect ratio is detected from
the first page (16:9 decks -> 13.333x7.5in).
"""
from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches

HERE = Path(__file__).resolve().parents[2]


def render_pages(pdf: Path, dpi: int, tmp: Path) -> list[Path]:
    out = tmp / (pdf.stem + "_%03d.png")
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
                    f"-r{dpi}", f"-o{out}", str(pdf)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(tmp.glob(pdf.stem + "_*.png"))


def build_pptx(pdf: Path, dpi: int, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pages = render_pages(pdf, dpi, tmp)
        if not pages:
            raise RuntimeError(f"no pages rendered from {pdf}")
        w, h = Image.open(pages[0]).size
        ar = w / h
        prs = Presentation()
        prs.slide_height = Inches(7.5)
        prs.slide_width = Inches(round(7.5 * ar, 4))
        blank = prs.slide_layouts[6]
        for pg in pages:
            slide = prs.slides.add_slide(blank)
            slide.shapes.add_picture(str(pg), 0, 0,
                                     width=prs.slide_width, height=prs.slide_height)
        out = outdir / (pdf.stem + ".pptx")
        prs.save(out)
        return out, len(pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", type=Path)
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--outdir", type=Path, default=HERE / "docs" / "pptx")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    pdfs = list(args.pdfs)
    if args.all:
        pdfs += sorted((HERE / "docs").glob("*slides*.pdf"))
        heine = HERE / "heine_paper" / "slides_nishioka_heine2025.pdf"
        if heine.exists():
            pdfs.append(heine)
    # de-dup, keep order
    seen, uniq = set(), []
    for p in pdfs:
        p = p if p.is_absolute() else (HERE / p)
        if p not in seen and p.exists():
            seen.add(p); uniq.append(p)

    for pdf in uniq:
        out, n = build_pptx(pdf, args.dpi, args.outdir)
        print(f"  {pdf.name:38s} -> {out.relative_to(HERE)}  ({n} slides)")


if __name__ == "__main__":
    main()
