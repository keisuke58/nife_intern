#!/usr/bin/env python3
"""
export_slides_pdf.py
Renders the 6 new guild-analysis slides as a standalone PDF (A4 landscape).
Uses matplotlib for layout — no LibreOffice needed.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np

HERE    = Path(__file__).resolve().parent
R       = HERE / 'results'
OUT_PDF = HERE / 'dieckow_paper' / 'guild_analysis_slides.pdf'

# Widescreen: 13.33 × 7.5 inches
W_IN, H_IN = 13.33, 7.50

# Colors
NAVY  = '#1A3A5C'
MINT  = '#EDFAF3'
WHITE = '#FFFFFF'
LBLUE = '#ADD8E6'
RED   = '#C0392B'
BLUE  = '#1A6BBF'


def make_fig():
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor=WHITE)
    return fig


def draw_header(fig, title, subtitle=''):
    h_frac = 0.72 / H_IN
    ax_h = fig.add_axes([0, 1 - h_frac, 1, h_frac])
    ax_h.set_facecolor(NAVY)
    ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)
    ax_h.axis('off')
    ax_h.text(0.03, 0.65, title, color=WHITE,
              fontsize=14, fontweight='bold', va='center', ha='left',
              transform=ax_h.transAxes)
    if subtitle:
        ax_h.text(0.03, 0.18, subtitle, color=LBLUE,
                  fontsize=8.5, va='center', ha='left',
                  transform=ax_h.transAxes)
    return ax_h


def draw_footer(fig, text):
    f_y    = 0.10
    f_h    = f_y
    ax_f   = fig.add_axes([0, 0, 1, f_h])
    ax_f.set_facecolor(MINT)
    ax_f.set_xlim(0, 1); ax_f.set_ylim(0, 1)
    ax_f.axis('off')
    ax_f.text(0.025, 0.5, text, color=NAVY,
              fontsize=8.5, va='center', ha='left',
              transform=ax_f.transAxes, wrap=True,
              multialignment='left')
    return ax_f


def paste_image(fig, img_path, rect):
    """rect = [left, bottom, width, height] in figure fraction."""
    img = Image.open(img_path)
    ax  = fig.add_axes(rect)
    ax.imshow(np.array(img), aspect='auto')
    ax.axis('off')
    return ax


def paste_two_images(fig, left_path, right_path,
                     cap_left='', cap_right=''):
    """Two side-by-side images with captions."""
    HEADER_FRAC = 0.72 / H_IN
    FOOTER_FRAC = 0.10
    TOP  = 1 - HEADER_FRAC - 0.01
    BOT  = FOOTER_FRAC + (0.045 if cap_left else 0.01)
    H    = TOP - BOT

    def _paste(path, xl, xr, caption):
        img   = Image.open(path)
        iw, ih = img.size
        ratio  = iw / ih
        avail_w = xr - xl - 0.02
        pw = avail_w
        ph = pw / ratio * (W_IN / H_IN)
        if ph > H:
            ph = H
            pw = ph * ratio * (H_IN / W_IN)
        px = xl + (avail_w - pw) / 2 + 0.01
        py = BOT + (H - ph) / 2
        ax = fig.add_axes([px, py, pw, ph])
        ax.imshow(np.array(img), aspect='auto')
        ax.axis('off')
        if caption:
            fig.text(xl + avail_w / 2 + 0.01, BOT - 0.015, caption,
                     ha='center', va='top', fontsize=8,
                     color=NAVY, style='italic')

    _paste(left_path,  0.01, 0.50, cap_left)
    _paste(right_path, 0.50, 0.99, cap_right)


def paste_one_image(fig, img_path):
    HEADER_FRAC = 0.72 / H_IN
    FOOTER_FRAC = 0.10
    TOP  = 1 - HEADER_FRAC - 0.01
    BOT  = FOOTER_FRAC + 0.01
    H    = TOP - BOT

    img   = Image.open(img_path)
    iw, ih = img.size
    ratio  = iw / ih
    pw = 0.88
    ph = pw / ratio * (W_IN / H_IN)
    if ph > H:
        ph = H
        pw = ph * ratio * (H_IN / W_IN)
    px = (1 - pw) / 2
    py = BOT + (H - ph) / 2
    ax = fig.add_axes([px, py, pw, ph])
    ax.imshow(np.array(img), aspect='auto')
    ax.axis('off')


# ── Build pages ───────────────────────────────────────────────────────────────

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(str(OUT_PDF)) as pdf:

    # ── Page 1: Section divider ──────────────────────────────────────────────
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor=NAVY)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(NAVY); ax.axis('off')
    ax.text(0.5, 0.55, 'Guild-Level Dynamical Analysis',
            color=WHITE, fontsize=26, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.40,
            'Keystone  ·  Network  ·  Tipping  ·  Relapse  ·  Prevention  ·  Phase Diagram',
            color=LBLUE, fontsize=13, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.12,
            'Dieckow 2024 · 10-guild gLV · N=10 patients · Nishioka (2026)',
            color='#8ab0cc', fontsize=10, ha='center', va='center',
            transform=ax.transAxes, style='italic')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 2: Keystone Analysis ────────────────────────────────────────────
    fig = make_fig()
    draw_header(fig,
        'Guild-Level Keystone Analysis: Knockout BC Divergence',
        'Which guild, when removed, shifts the community most?')
    paste_one_image(fig, R / 'guild_importance' / 'knockout_bars_base.png')
    draw_footer(fig,
        ' Bacilli: highest BC divergence (0.839) — compositional keystone.  '
        'Removal increases volume → competitive suppressor.  '
        ' | '
        'Actinobacteria: vol_drop = +0.846 → growth engine.  '
        'Strongest single interaction: Acti → Baci  A = +3.43.')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 3: Network Centrality ───────────────────────────────────────────
    fig = make_fig()
    draw_header(fig,
        'Guild Network Centrality: Influence, Vulnerability, Net Role',
        'Column-sum |A| = influence  |  Row-sum |A| = vulnerability  |  Σ A_ij = net mutualist (+) / competitor (−)')
    paste_one_image(fig, R / 'guild_network' / 'guild_centrality_summary.png')
    draw_footer(fig,
        'Actinobacteria: influence = 5.58 (highest), net = +2.02 (mutualist hub).  '
        'Bacilli: vulnerability = 8.24 (most regulated).  '
        'Betaproteobacteria: net = −0.14 → net competitor.  '
        'A matrix encodes stable dysbiotic attractor via Acti–Baci–Beta triad.')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 4: Relapse Dynamics ─────────────────────────────────────────────
    fig = make_fig()
    draw_header(fig,
        'Predicted Relapse Dynamics: Three Patient Archetypes',
        'Long-time simulation (t = 150 d) from patient-specific b vectors  |  week-3 GDI as prognostic marker')
    paste_two_images(fig,
        R / 'guild_tipping'  / 'relapse_dynamics.png',
        R / 'guild_relapse'  / 'gdi_w3_vs_relapse.png',
        cap_left  = 'GDI trajectories to 90 days (per patient)',
        cap_right = 'Week-3 GDI → relapse day')
    draw_footer(fig,
        '① Persistent dysbiotic (A,B,E,F,H,K)  '
        '② Transient responder: C→24d, G→38d, L→48d (commensal at wk-3, dysbiotic by t<50d)  '
        '③ Permanent commensal (D only).  '
        '→ Lower week-3 GDI = longer remission.  3-week data extrapolated to 150d.')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 5: Prevention + Joshi ───────────────────────────────────────────
    fig = make_fig()
    draw_header(fig,
        'Prevention Threshold & External Validation (Joshi 2025)',
        'Min. Δb_i to prevent relapse  |  Model equilibrium GDI vs peri-implantitis reference')
    paste_two_images(fig,
        R / 'guild_relapse' / 'prevention_threshold.png',
        R / 'guild_relapse' / 'joshi_comparison.png',
        cap_left  = 'Required Δb per guild (patient C only)',
        cap_right = 'Predicted eq. GDI vs Joshi PI distribution')
    draw_footer(fig,
        'Prevention: only Patient C convertible by single-guild intervention '
        '(Δb_Bact = 1.33). G & L require multi-guild change → deep attractor.  '
        '|  Joshi validation: predicted eq. GDI = +0.65 ≈ Joshi PI mean +0.53  '
        '(9/10 patients in PI range, difference < 0.12 < Joshi SD=1.30).')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 6: Phase Diagram ────────────────────────────────────────────────
    fig = make_fig()
    draw_header(fig,
        '3D Attractor Landscape: GDI = 0 Boundary in Growth-Rate Space',
        'Equilibrium GDI on 18³ grid (Acti × Baci × Bact),  t = 80 d,  12-core parallel.  '
        'Blue = commensal  |  Red = dysbiotic  |  Black contour = GDI = 0')
    paste_one_image(fig, R / 'guild_phase' / 'phase3d_static.png')
    draw_footer(fig,
        '98% of parameter space is dysbiotic.  Commensal region confined to b_Bact < ~1.5 '
        '(Bacteroidia growth rate is the dominant control variable).  '
        'CT1 patients with low b_Bact cluster near the boundary; '
        'CT2 patient A (b_Bact=4.5) is deep in the dysbiotic zone.  '
        '→ Structural attractor robustness — not initial-condition sensitive.')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # PDF metadata
    d = pdf.infodict()
    d['Title']   = 'Guild-Level Dynamical Analysis — Dieckow 2024 gLV'
    d['Author']  = 'Keisuke Nishioka'
    d['Subject'] = 'Keystone, Network, Relapse, Prevention, Phase Diagram'

print(f'Saved: {OUT_PDF}')
