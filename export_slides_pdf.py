#!/usr/bin/env python3
"""
export_slides_pdf.py  v2
Renders 6 guild-analysis slides as a widescreen PDF.
- Header/footer use fig.add_patch(Rectangle) for reliable background in PDF
- Each figure slide has callout annotations explaining key findings
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

HERE    = Path(__file__).resolve().parent
R       = HERE / 'results'
OUT_PDF = HERE / 'dieckow_paper' / 'guild_analysis_slides.pdf'

W_IN, H_IN = 13.33, 7.50
HEADER_FRAC = 0.72 / H_IN   # ~0.096
FOOTER_FRAC = 0.115

NAVY  = '#1A3A5C'
MINT  = '#EDFAF3'
WHITE = '#FFFFFF'
LBLUE = '#ADD8E6'
RED   = '#C0392B'
BLUE  = '#1A6BBF'
GOLD  = '#E8A020'


# ── Layout helpers ─────────────────────────────────────────────────────────────

def base_fig():
    return plt.figure(figsize=(W_IN, H_IN), facecolor=WHITE)


def draw_chrome(fig, title, subtitle, footer):
    """Draw header rect + title + subtitle + footer rect + footer text."""
    tf = fig.transFigure

    # One full-slide axes for background drawing
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax_bg.set_xlim(0, 1); ax_bg.set_ylim(0, 1)
    ax_bg.axis('off')

    # Header rectangle (filled polygon — renders reliably in PDF)
    hy = 1 - HEADER_FRAC
    ax_bg.fill([0, 1, 1, 0], [hy, hy, 1, 1], color=NAVY, zorder=1,
               transform=ax_bg.transAxes)

    # Footer rectangle
    ax_bg.fill([0, 1, 1, 0], [0, 0, FOOTER_FRAC, FOOTER_FRAC],
               color=MINT, zorder=1, transform=ax_bg.transAxes)

    # Title text
    ax_bg.text(0.03, 1 - HEADER_FRAC * 0.38, title,
               color=WHITE, fontsize=13.5, fontweight='bold',
               va='center', ha='left', zorder=3, transform=ax_bg.transAxes)
    # Subtitle
    if subtitle:
        ax_bg.text(0.03, 1 - HEADER_FRAC * 0.82, subtitle,
                   color=LBLUE, fontsize=8.5,
                   va='center', ha='left', zorder=3, transform=ax_bg.transAxes)
    # Footer text
    ax_bg.text(0.025, FOOTER_FRAC * 0.50, footer,
               color=NAVY, fontsize=8.5, va='center', ha='left',
               zorder=3, transform=ax_bg.transAxes)


def fig_area():
    """Return (left, bottom, width, height) of the usable figure area."""
    bot = FOOTER_FRAC + 0.01
    top = 1 - HEADER_FRAC - 0.01
    return 0.01, bot, 0.98, top - bot


def paste_img(fig, path, rect, zorder=1):
    """rect = (l, b, w, h) in figure fractions, aspect-ratio preserved."""
    l, b, w, h = rect
    img  = Image.open(path)
    iw, ih = img.size
    ratio = (iw / ih) * (H_IN / W_IN)   # pixel ratio → figure fraction ratio
    if w / ratio > h:
        w2 = h * ratio
        l2 = l + (w - w2) / 2
        h2 = h
    else:
        h2 = w / ratio
        l2 = l
        b2 = b + (h - h2) / 2
        b  = b2
        w2 = w
    ax = fig.add_axes([l2, b, w2, h2], zorder=zorder)
    ax.imshow(np.array(img), aspect='auto')
    ax.axis('off')
    return ax, (l2, b, w2, h2)


def callout(fig, x, y, text, arrow_to=None,
            fc='#FFFDE7', ec=NAVY, fontsize=8.5, color=NAVY, zorder=5):
    """Add a text callout box at figure coordinates (x, y)."""
    bbox = dict(boxstyle='round,pad=0.3', fc=fc, ec=ec, lw=1.2, alpha=0.93)
    ann = fig.text(x, y, text, color=color, fontsize=fontsize,
                   va='center', ha='center', zorder=zorder,
                   bbox=bbox, transform=fig.transFigure)
    if arrow_to:
        ax_main = fig.add_axes([0, 0, 1, 1], zorder=zorder - 1)
        ax_main.axis('off')
        ax_main.annotate('', xy=arrow_to, xycoords='figure fraction',
                         xytext=(x, y), textcoords='figure fraction',
                         arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.5),
                         zorder=zorder)
    return ann


def callout_red(fig, x, y, text, arrow_to=None):
    return callout(fig, x, y, text, arrow_to, fc='#FDECEA', ec=RED, color=RED)


def callout_blue(fig, x, y, text, arrow_to=None):
    return callout(fig, x, y, text, arrow_to, fc='#E8F4FD', ec=BLUE, color=BLUE)


# ── Slides ─────────────────────────────────────────────────────────────────────

with PdfPages(str(OUT_PDF)) as pdf:

    # ── Page 1: Title divider ────────────────────────────────────────────────
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor=NAVY)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(NAVY); ax.axis('off')
    ax.text(0.5, 0.57, 'Guild-Level Dynamical Analysis',
            color=WHITE, fontsize=28, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.42,
            'Keystone  ·  Network Centrality  ·  Relapse Dynamics  ·  Prevention  ·  Phase Diagram',
            color=LBLUE, fontsize=13, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.14,
            'Dieckow 2024  ·  10-guild gLV  ·  N = 10 patients  ·  Nishioka (2026)',
            color='#8ab0cc', fontsize=10, ha='center', va='center',
            transform=ax.transAxes, style='italic')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 2: Keystone ─────────────────────────────────────────────────────
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Guild-Level Keystone Analysis: Knockout BC Divergence',
        subtitle = 'Remove one guild → simulate gLV 21 d → measure BC shift, productivity drop, volume change',
        footer   = (
            'Bacilli: highest BC divergence (0.839) — compositional keystone. '
            'Removal increases total volume → competitive suppressor.  '
            '|  Actinobacteria: vol_drop = +0.846 → growth engine.  '
            'Strongest single interaction: Acti → Baci  A = +3.43.'
        ))

    l, b, w, h = fig_area()
    ax_img, pos = paste_img(fig, R / 'guild_importance' / 'knockout_bars_base.png',
                            (l + 0.02, b, w * 0.96, h))

    # Callouts
    callout_red(fig, 0.215, 0.75,
                'Bacilli\nBC = 0.839\n← compositional keystone',
                arrow_to=(0.168, 0.62))
    callout(fig, 0.115, 0.56,
            'Actinobacteria\nvol_drop = +0.846\n← growth engine',
            arrow_to=(0.09, 0.68))
    callout_red(fig, 0.46, 0.65,
                'Bacilli\nproductivity ↓ 7.79\n← dominant driver',
                arrow_to=(0.39, 0.37))
    callout(fig, 0.70, 0.75,
            'Bacilli\nvol_drop = −0.73\n(removal → volume ↑)',
            arrow_to=(0.685, 0.60))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 3: Network Centrality ───────────────────────────────────────────
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Guild Network Centrality: Influence, Vulnerability, Net Role',
        subtitle = 'Influence = Σ|A_ji| (how much guild i drives others)  |  Vulnerability = Σ|A_ij| (how much others drive guild i)',
        footer   = (
            'Actinobacteria: influence = 5.58 (highest), net = +2.02 → mutualist hub.  '
            'Bacilli: vulnerability = 8.24 → most regulated by others.  '
            'Betaproteobacteria: net = −0.14 → net competitor.  '
            'A matrix encodes dysbiotic attractor via Acti–Baci–Beta triad.'
        ))

    paste_img(fig, R / 'guild_network' / 'guild_centrality_summary.png',
              (0.03, FOOTER_FRAC + 0.01, 0.94, 1 - HEADER_FRAC - FOOTER_FRAC - 0.02))

    callout_blue(fig, 0.16, 0.82,
                 'Acti: influence 5.58\nNet = +2.02 (mutualist hub)\n→ drives all other guilds',
                 arrow_to=(0.138, 0.70))
    callout_red(fig, 0.32, 0.82,
                'Baci: vulnerability 8.24\n→ most controlled by others\nCompositional keystone',
                arrow_to=(0.298, 0.70))
    callout(fig, 0.48, 0.82,
            'Beta: net = −0.14\n→ net competitor\n(unique role)',
            arrow_to=(0.458, 0.70))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 4: Relapse Dynamics ─────────────────────────────────────────────
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Predicted Relapse Dynamics: Three Patient Archetypes',
        subtitle = 'gLV simulation t = 150 d with patient-specific b vectors  |  GDI = log(φ_dys) − log(φ_com)',
        footer   = (
            '① Persistent dysbiotic (A,B,E,F,H,K): GDI > 0 throughout.  '
            '② Transient responders (C→24d, G→38d, L→48d): commensal at wk-3, dysbiotic by day 50.  '
            '③ Permanent commensal (D only): stable GDI < 0.  '
            '→ Week-3 GDI as prognostic marker: lower = longer remission.'
        ))

    fw = 0.485; gap = 0.02
    paste_img(fig, R / 'guild_tipping' / 'relapse_dynamics.png',
              (0.01, FOOTER_FRAC + 0.05, fw, 1 - HEADER_FRAC - FOOTER_FRAC - 0.06))
    paste_img(fig, R / 'guild_relapse' / 'gdi_w3_vs_relapse.png',
              (0.01 + fw + gap, FOOTER_FRAC + 0.05, fw, 1 - HEADER_FRAC - FOOTER_FRAC - 0.06))

    # Left panel annotations
    callout_blue(fig, 0.15, 0.82,
                 'Patient D\nOnly permanent commensal\nGDI stays < 0',
                 arrow_to=(0.17, 0.68))
    callout(fig, 0.30, 0.87,
            'C, G, L: Transient recovery\nCommensal at wk-3\nbut relapse by day 24–48',
            arrow_to=(0.28, 0.72))
    callout_red(fig, 0.10, 0.56,
                'A, B, E, F, H, K\nPersistently dysbiotic\n(GDI > 0 always)',
                arrow_to=(0.12, 0.45))

    # Right panel
    callout(fig, 0.79, 0.82,
            'Lower wk-3 GDI\n→ later relapse\n(prognostic marker)',
            arrow_to=(0.74, 0.68))
    callout_blue(fig, 0.61, 0.60,
                 'L: GDI = −5.66\nrelapse day 48\n(deepest recovery)',
                 arrow_to=(0.615, 0.52))

    # Panel labels
    fig.text(0.255, FOOTER_FRAC + 0.005, '(a) GDI trajectories to 90 d',
             ha='center', fontsize=8.5, color=NAVY, style='italic',
             transform=fig.transFigure)
    fig.text(0.755, FOOTER_FRAC + 0.005, '(b) Week-3 GDI as prognostic marker',
             ha='center', fontsize=8.5, color=NAVY, style='italic',
             transform=fig.transFigure)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 5: Prevention + Joshi ───────────────────────────────────────────
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Prevention Threshold & External Validation (Joshi 2025)',
        subtitle = 'Min. Δb_i to prevent relapse (single-guild sweep)  |  Model eq. GDI vs peri-implantitis reference distribution',
        footer   = (
            'Prevention: only Patient C convertible by single-guild intervention '
            '(Δb_Bact = 1.33). G & L require multi-guild change → deep attractor.  '
            '|  Joshi validation: predicted eq. GDI = +0.65 ≈ Joshi PI mean +0.53  '
            '(9/10 patients; difference < 0.12 < Joshi SD = 1.30).'
        ))

    fw = 0.485; gap = 0.02
    paste_img(fig, R / 'guild_relapse' / 'prevention_threshold.png',
              (0.01, FOOTER_FRAC + 0.05, fw, 1 - HEADER_FRAC - FOOTER_FRAC - 0.06))
    paste_img(fig, R / 'guild_relapse' / 'joshi_comparison.png',
              (0.01 + fw + gap, FOOTER_FRAC + 0.05, fw, 1 - HEADER_FRAC - FOOTER_FRAC - 0.06))

    # Left panel
    callout_blue(fig, 0.19, 0.84,
                 'Patient C\nΔb_Bact = 1.33 sufficient\n→ permanent commensal',
                 arrow_to=(0.16, 0.72))
    callout_red(fig, 0.37, 0.84,
                'G & L: no threshold found\neven Δb = 2.0 insufficient\n→ deep dysbiotic attractor',
                arrow_to=(0.32, 0.72))

    # Right panel
    callout_blue(fig, 0.72, 0.84,
                 'Week-3 GDI mean = −0.50\n≈ Joshi Mucositis (−1.74)',
                 arrow_to=(0.695, 0.72))
    callout_red(fig, 0.84, 0.55,
                'Eq. GDI mean = +0.65\n≈ Joshi PI (+0.53)\n9/10 patients match',
                arrow_to=(0.84, 0.45))

    fig.text(0.255, FOOTER_FRAC + 0.005,
             '(a) Required Δb per guild for prevention',
             ha='center', fontsize=8.5, color=NAVY, style='italic',
             transform=fig.transFigure)
    fig.text(0.755, FOOTER_FRAC + 0.005,
             '(b) External validation: Joshi 2025 peri-implant cohort',
             ha='center', fontsize=8.5, color=NAVY, style='italic',
             transform=fig.transFigure)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 6: Phase Diagram ────────────────────────────────────────────────
    fig = base_fig()
    draw_chrome(fig,
        title    = '3D Attractor Landscape: GDI = 0 Boundary in Growth-Rate Space',
        subtitle = '18³ grid, Acti × Baci × Bact, t = 80 d  |  Blue = commensal  |  Red = dysbiotic  |  Black = GDI = 0 boundary',
        footer   = (
            '98% of parameter space is dysbiotic (GDI > 0).  '
            'Commensal region confined to b_Bact < ~1.5 — Bacteroidia growth rate is the dominant control variable.  '
            'CT1 patients with low b_Bact cluster near boundary; CT2 patient A (b_Bact = 4.5) is deep in dysbiotic zone.  '
            '→ Structural attractor robustness — not initial-condition sensitive.'
        ))

    # Figure takes left 74% — right 26% for key findings text
    paste_img(fig, R / 'guild_phase' / 'phase3d_static.png',
              (0.01, FOOTER_FRAC + 0.01, 0.72, 1 - HEADER_FRAC - FOOTER_FRAC - 0.02))

    # Right-side key findings panel
    ax_txt = fig.add_axes([0.75, FOOTER_FRAC + 0.02, 0.23, 1 - HEADER_FRAC - FOOTER_FRAC - 0.04], zorder=3)
    ax_txt.set_facecolor('#F4F8FF')
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
    for sp in ax_txt.spines.values():
        sp.set_color('#BBCCDD'); sp.set_linewidth(1)
    ax_txt.set_xticks([]); ax_txt.set_yticks([])

    findings = [
        ('Key findings', 11, True,  NAVY,  0.95),
        ('', 8, False, NAVY, 0.88),
        ('98% of b-space', 9, True,  RED,   0.85),
        ('is dysbiotic', 9, False, RED,   0.80),
        ('(GDI > 0)', 9, False, RED,   0.75),
        ('', 8, False, NAVY, 0.68),
        ('Commensal zone:', 9, True,  BLUE,  0.65),
        ('b_Bact < ~1.5', 9, False, BLUE,  0.60),
        ('only 2% of space', 9, False, BLUE,  0.55),
        ('', 8, False, NAVY, 0.48),
        ('b_Bact (Bacteroidia)', 9, True,  NAVY,  0.45),
        ('= dominant axis', 9, False, NAVY,  0.40),
        ('boundary at ~1.5', 9, False, NAVY,  0.35),
        ('', 8, False, NAVY, 0.28),
        ('CT1 (○): near boundary', 8.5, False, BLUE, 0.25),
        ('CT2 patient A:', 8.5, False, RED,  0.19),
        ('b_Bact=4.5 → deep dys.', 8.5, False, RED,  0.14),
        ('', 8, False, NAVY, 0.08),
        ('→ structural attractor', 8, False, NAVY, 0.06),
    ]
    for (txt, fs, bold, col, yy) in findings:
        ax_txt.text(0.08, yy, txt, fontsize=fs, fontweight='bold' if bold else 'normal',
                    color=col, va='center', ha='left', transform=ax_txt.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    d = pdf.infodict()
    d['Title']  = 'Guild-Level Dynamical Analysis — Dieckow 2024 gLV'
    d['Author'] = 'Keisuke Nishioka'

print(f'Saved: {OUT_PDF}')
