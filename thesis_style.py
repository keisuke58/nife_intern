"""thesis_style.py — shared matplotlib style for the master-thesis figures.

Goal: every figure in 30_Masterarbeit looks identical in font and size. We render
text with LaTeX (usetex) using the thesis body font (lmodern), at a fixed 9 pt, and
generate each figure at its on-page display width so `\\includegraphics[width=...]`
needs NO rescaling (→ uniform physical font size across all figures).

Requirements: run the figure script with TeX Live 2025 on PATH so matplotlib finds
`latex`/`dvipng`/`dvips`/`gs`, e.g.

    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python scripts/.../foo.py

Usage in a figure script:
    from thesis_style import use
    fig, ax = plt.subplots(figsize=use(width_frac=1.0, aspect=0.45))
    ...
    fig.savefig(path)            # emit .pdf (vector); .png only for imaging
"""
import matplotlib as mpl

# Thesis \textwidth, read from main.log ("\textwidth=455.24411pt"); 72.27 pt/inch.
TEXTWIDTH_PT = 455.24411
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27          # = 6.299 in


def use(width_frac=1.0, aspect=0.62):
    """Apply the thesis rcParams and return a figsize (w, h) in inches.

    width_frac : fraction of \\textwidth the figure will be \\includegraphics'd at
                 (use the SAME fraction in LaTeX → 1:1, no font rescaling).
    aspect     : height / width ratio of the figure.
    """
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],                                  # LaTeX (lmodern) chooses
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)
