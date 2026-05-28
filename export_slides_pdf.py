#!/usr/bin/env python3
"""
export_slides_pdf.py  v3
6-page PDF of guild-level dynamical analysis results.

Layout system:
  - Single-figure slides: fig 72% left + key-findings panel 26% right
  - Two-figure slides:    fig L 47% + fig R 47%, findings in footer area
  - Consistent navy header + mint footer + clean callout system
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.family': 'sans-serif'})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

HERE    = Path(__file__).resolve().parent
R       = HERE / 'results'
OUT_PDF = HERE / 'dieckow_paper' / 'guild_analysis_slides.pdf'

W_IN, H_IN   = 13.33, 7.50
HDR_H        = 0.80 / H_IN   # header fraction
FTR_H        = 0.72 / H_IN   # footer fraction
CONTENT_BOT  = FTR_H + 0.01
CONTENT_TOP  = 1 - HDR_H - 0.01
CONTENT_H    = CONTENT_TOP - CONTENT_BOT

NAVY   = '#1A3A5C'
NAVY_L = '#2A5080'
MINT   = '#E8F8F0'
WHITE  = '#FFFFFF'
LBLUE  = '#A8D4F0'
RED    = '#C0392B'
BLUE   = '#1A6BBF'
GOLD   = '#D4860A'
LGRAY  = '#F5F5F8'


# ── Core layout helpers ────────────────────────────────────────────────────────

def base_fig():
    return plt.figure(figsize=(W_IN, H_IN), facecolor=WHITE)


def draw_chrome(fig, title, subtitle, footer):
    ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # Header
    ax.fill([0,1,1,0], [1-HDR_H, 1-HDR_H, 1, 1], color=NAVY)
    ax.text(0.025, 1 - HDR_H*0.35, title,
            color=WHITE, fontsize=14, fontweight='bold',
            va='center', ha='left', transform=ax.transAxes)
    if subtitle:
        ax.text(0.025, 1 - HDR_H*0.78, subtitle,
                color=LBLUE, fontsize=8.5, va='center', ha='left',
                transform=ax.transAxes)

    # Footer
    ax.fill([0,1,1,0], [0,0,FTR_H,FTR_H], color=MINT)
    ax.text(0.025, FTR_H*0.5, footer,
            color=NAVY, fontsize=8.8, va='center', ha='left',
            transform=ax.transAxes)


def paste_img(fig, path, l, b, w, h):
    """Paste image with aspect-ratio preservation within the given rect."""
    img = Image.open(path)
    iw, ih = img.size
    ratio = (iw / ih) * (H_IN / W_IN)
    pw = w; ph = pw / ratio
    if ph > h:
        ph = h; pw = ph * ratio
    pl = l + (w - pw) / 2
    pb = b + (h - ph) / 2
    ax = fig.add_axes([pl, pb, pw, ph])
    ax.imshow(np.array(img), aspect='auto')
    ax.axis('off')
    return ax


def right_panel(fig, headline, items):
    """
    Right-side key findings panel (x=0.745, full content height).
    headline: large bold text at top
    items: list of (text, color, bold, fontsize)
    """
    ax = fig.add_axes([0.745, CONTENT_BOT, 0.248, CONTENT_H])
    ax.set_facecolor(LGRAY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    for sp in ax.spines.values():
        sp.set_color('#BBCCDD'); sp.set_linewidth(0.8)

    # Headline
    ax.text(0.5, 0.955, headline,
            color=NAVY, fontsize=10.5, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.axhline(0.935, color='#BBCCDD', lw=0.8)

    # Items: evenly spaced
    n = len(items)
    ys = np.linspace(0.89, 0.06, n)
    for (txt, col, bold, fs), y in zip(items, ys):
        ax.text(0.06, y, txt,
                color=col, fontsize=fs,
                fontweight='bold' if bold else 'normal',
                va='center', ha='left', transform=ax.transAxes)
    return ax


def callout(ax, x, y, text, color=NAVY, fc='#FFFDE7', ec=None, fs=8.5):
    ec = ec or color
    ax.text(x, y, text, color=color, fontsize=fs,
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.35', fc=fc, ec=ec, lw=1.2, alpha=0.95))


def callout_arrow(fig, cx, cy, text, ax_tip, tx, ty,
                  color=NAVY, fc='#FFFDE7', fs=8.5):
    """Callout on fig with arrow to data point."""
    ann_ax = fig.add_axes([0, 0, 1, 1], zorder=5)
    ann_ax.set_xlim(0,1); ann_ax.set_ylim(0,1); ann_ax.axis('off')
    ann_ax.annotate(
        text, xy=(tx, ty), xytext=(cx, cy),
        xycoords='axes fraction' if ax_tip else 'figure fraction',
        textcoords='figure fraction',
        ha='center', va='center', fontsize=fs, color=color,
        bbox=dict(boxstyle='round,pad=0.35', fc=fc, ec=color, lw=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.4,
                        connectionstyle='arc3,rad=0.1'),
        zorder=6
    )


# ── Page builders ──────────────────────────────────────────────────────────────

with PdfPages(str(OUT_PDF)) as pdf:

    # ════════════════════════════════════════════════════════════════════════════
    # Page 1 — Title
    # ════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor=NAVY)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(NAVY); ax.axis('off')

    # Decorative top bar
    ax.fill([0,1,1,0],[0.92,0.92,1,1], color=NAVY_L)
    ax.text(0.5, 0.96, 'Dieckow 2024  ·  10-guild gLV  ·  N = 10 patients  ·  Nishioka (2026)',
            color='#7aabcc', fontsize=9.5, ha='center', va='center',
            style='italic', transform=ax.transAxes)

    ax.text(0.5, 0.63,
            'Guild-Level Dynamical Analysis',
            color=WHITE, fontsize=30, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)

    topics = ['Keystone Analysis', 'Network Centrality', 'Relapse Dynamics',
              'Prevention Threshold', '3D Phase Diagram']
    colors = [LBLUE, '#A8E6CF', '#FFD3B6', '#FFB3BA', '#D4B8E0']
    xs = np.linspace(0.12, 0.88, len(topics))
    for xi, (top, col) in zip(xs, zip(topics, colors)):
        ax.text(xi, 0.42, top, color=col, fontsize=11, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', fc=NAVY_L, ec=col, lw=1.5))

    ax.fill([0.25,0.75,0.75,0.25],[0.33,0.33,0.30,0.30], color='#8ab0cc', alpha=0.3)

    ax.text(0.5, 0.20,
            'Key question: What does the fitted gLV A matrix tell us about\n'
            'the long-term fate of the oral microbiome after dental treatment?',
            color='#c8d8e8', fontsize=11.5, ha='center', va='center',
            transform=ax.transAxes, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 2 — Keystone Analysis
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Guild-Level Keystone Analysis',
        subtitle = 'Knockout experiment: remove one guild → simulate 21 d → measure BC shift / productivity / volume change',
        footer   = (
            'Bacilli: highest BC divergence (0.839) + vol_drop = −0.73 → compositional keystone & competitive suppressor.  '
            '|  Actinobacteria: vol_drop = +0.846 → growth engine.  '
            '|  Strongest interaction: Acti → Baci  A = +3.43.'
        ))

    paste_img(fig, R/'guild_importance'/'knockout_bars_base.png',
              0.01, CONTENT_BOT, 0.72, CONTENT_H)

    right_panel(fig, 'Key Findings', [
        ('Compositional keystone:', NAVY,  True,  9.0),
        ('Bacilli  BC = 0.839', RED,   True,  9.5),
        ('→ removal shifts community most', RED, False, 8.5),
        ('', NAVY, False, 4),
        ('Competitive suppressor:', NAVY,  True,  9.0),
        ('Baci vol_drop = −0.73', RED,   False, 8.5),
        ('removing Baci → volume ↑', RED, False, 8.5),
        ('(other guilds fill the space)', '#888', False, 8.0),
        ('', NAVY, False, 4),
        ('Growth engine:', NAVY,  True,  9.0),
        ('Actinobacteria  +0.846', BLUE,  False, 8.5),
        ('→ drives total biomass', BLUE, False, 8.5),
        ('', NAVY, False, 4),
        ('Strongest interaction:', NAVY,  True,  9.0),
        ('Acti → Baci  A = +3.43', GOLD,  False, 8.5),
        ('→ Acti stimulates Baci', GOLD, False, 8.5),
    ])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 3 — Network Centrality
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Guild Network Centrality: Influence, Vulnerability, Net Role',
        subtitle = 'Influence = Σ|A_ji|  (column sum = how much guild i drives others)   ·   Vulnerability = Σ|A_ij|  (row sum = how much others drive guild i)',
        footer   = (
            'Actinobacteria: influence 5.58, net +2.02 → mutualist hub.  '
            'Bacilli: vulnerability 8.24 → most regulated by others (compositional keystone confirmed).  '
            'Betaproteobacteria: net −0.14 → only net competitor in the network.'
        ))

    paste_img(fig, R/'guild_network'/'guild_centrality_summary.png',
              0.01, CONTENT_BOT, 0.72, CONTENT_H)

    right_panel(fig, 'Network Roles', [
        ('Mutualist hub:', NAVY,  True,  9.0),
        ('Actinobacteria', BLUE,  True,  9.5),
        ('influence = 5.58  (highest)', BLUE, False, 8.5),
        ('net effect = +2.02', BLUE, False, 8.5),
        ('→ drives all other guilds', BLUE, False, 8.5),
        ('', NAVY, False, 4),
        ('Most regulated:', NAVY,  True,  9.0),
        ('Bacilli', RED,   True,  9.5),
        ('vulnerability = 8.24  (highest)', RED, False, 8.5),
        ('→ controlled by others', RED, False, 8.5),
        ('', NAVY, False, 4),
        ('Net competitor:', NAVY,  True,  9.0),
        ('Betaproteobacteria', GOLD, True, 9.5),
        ('net effect = −0.14', GOLD, False, 8.5),
        ('→ suppresses other guilds', GOLD, False, 8.5),
        ('', NAVY, False, 4),
        ('→ Acti–Baci–Beta triad', NAVY, True, 8.5),
        ('stabilises dysbiotic attractor', NAVY, False, 8.5),
    ])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 4 — Relapse Dynamics
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Predicted Relapse Dynamics: Three Patient Archetypes',
        subtitle = 'gLV simulation to t = 150 d with patient-specific b vectors  ·  GDI = log(φ_dys) − log(φ_com)  ·  GDI < 0 = commensal',
        footer   = (
            '① Persistent dysbiotic (A,B,E,F,H,K): GDI > 0 throughout — gLV attractor dominates from day 1.  '
            '② Transient responders (C→24d, G→38d, L→48d): commensal at wk-3, relapse by day 50.  '
            '③ Permanent commensal (D only): uniquely stable b vector keeps GDI < 0 at t = 150d.  '
            '→ Week-3 GDI is a prognostic marker for relapse timing.'
        ))

    GAP   = 0.015
    FW    = (0.74 - GAP) / 2
    FBOT  = CONTENT_BOT + 0.03
    FHH   = CONTENT_H - 0.05

    paste_img(fig, R/'guild_tipping'/'relapse_dynamics.png',
              0.01, FBOT, FW, FHH)
    paste_img(fig, R/'guild_relapse'/'gdi_w3_vs_relapse.png',
              0.01 + FW + GAP, FBOT, FW, FHH)

    # Panel labels
    fig.text(0.01 + FW/2, CONTENT_BOT + 0.005,
             '(a)  GDI trajectories to 90 d (per patient)',
             ha='center', fontsize=9, color=NAVY, style='italic',
             transform=fig.transFigure)
    fig.text(0.01 + FW + GAP + FW/2, CONTENT_BOT + 0.005,
             '(b)  Week-3 GDI as a prognostic marker for relapse',
             ha='center', fontsize=9, color=NAVY, style='italic',
             transform=fig.transFigure)

    # Right panel
    right_panel(fig, 'Three archetypes', [
        ('① Persistent dysbiotic', RED,  True,  9.0),
        ('A, B, E, F, H, K', RED, False, 8.5),
        ('GDI > 0 from day 1', RED, False, 8.5),
        ('', NAVY, False, 4),
        ('② Transient responder', GOLD, True,  9.0),
        ('C → relapse day 24', GOLD, False, 8.5),
        ('G → relapse day 38', GOLD, False, 8.5),
        ('L → relapse day 48', GOLD, False, 8.5),
        ('commensal at wk-3,', GOLD, False, 8.5),
        ('then attractor pulls back', GOLD, False, 8.5),
        ('', NAVY, False, 4),
        ('③ Permanent commensal', BLUE, True,  9.0),
        ('Patient D only', BLUE, False, 8.5),
        ('GDI stable < 0', BLUE, False, 8.5),
        ('', NAVY, False, 4),
        ('Prognostic rule:', NAVY, True,  8.5),
        ('GDI(wk3) < −5  →  >6wk', NAVY, False, 8.0),
        ('GDI(wk3) > −2  →  <4wk', NAVY, False, 8.0),
    ])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 5 — Prevention Threshold + Joshi Validation
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Prevention Threshold & External Validation',
        subtitle = 'Left: min. Δb_i to prevent relapse (single-guild b sweep)   ·   Right: model equilibrium GDI vs Joshi 2025 peri-implant cohort (N=127)',
        footer   = (
            'Prevention: Patient C convertible with Δb_Bact = 1.33; G & L require multi-guild change → deep attractor.  '
            '|  Joshi validation: predicted eq. GDI = +0.65 ≈ Joshi PI mean +0.53 '
            '(9/10 patients in PI range; Δ = 0.12 < Joshi SD = 1.30).  '
            '→ 3-week calibration extrapolates correctly to long-term peri-implant dysbiosis.'
        ))

    FW2  = (0.74 - GAP) / 2
    FBOT2 = CONTENT_BOT + 0.03
    FHH2  = CONTENT_H - 0.05

    paste_img(fig, R/'guild_relapse'/'prevention_threshold.png',
              0.01, FBOT2, FW2, FHH2)
    paste_img(fig, R/'guild_relapse'/'joshi_comparison.png',
              0.01 + FW2 + GAP, FBOT2, FW2, FHH2)

    fig.text(0.01 + FW2/2, CONTENT_BOT + 0.005,
             '(a)  Required Δb per guild for relapse prevention',
             ha='center', fontsize=9, color=NAVY, style='italic',
             transform=fig.transFigure)
    fig.text(0.01 + FW2 + GAP + FW2/2, CONTENT_BOT + 0.005,
             '(b)  External validation: Joshi 2025 peri-implant cohort',
             ha='center', fontsize=9, color=NAVY, style='italic',
             transform=fig.transFigure)

    right_panel(fig, 'Key Results', [
        ('Prevention:', NAVY, True, 9.0),
        ('Patient C:', BLUE, True, 9.0),
        ('Δb_Bact = 1.33 sufficient', BLUE, False, 8.5),
        ('→ permanent commensal', BLUE, False, 8.5),
        ('Patients G & L:', RED, True, 9.0),
        ('no threshold found', RED, False, 8.5),
        ('→ deep dysbiotic attractor', RED, False, 8.5),
        ('', NAVY, False, 4),
        ('Joshi validation:', NAVY, True, 9.0),
        ('Predicted eq. GDI:', NAVY, False, 8.5),
        ('+0.65  (model)', NAVY, True, 9.0),
        ('Joshi PI mean:', NAVY, False, 8.5),
        ('+0.53  (data)', GOLD, True, 9.0),
        ('9/10 patients match', GOLD, False, 8.5),
        ('', NAVY, False, 4),
        ('→ 3-wk data predicts', NAVY, True, 8.5),
        ('   long-term dysbiosis', NAVY, False, 8.5),
    ])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 6 — 3D Phase Diagram
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = '3D Attractor Landscape: GDI = 0 Boundary in Growth-Rate Space',
        subtitle = '18³ grid · Acti × Baci × Bact · t = 80 d · 12-core parallel   ·   Blue = commensal  |  Red = dysbiotic  |  Black contour = GDI = 0',
        footer   = (
            '98% of parameter space is dysbiotic.  '
            'Commensal region confined to b_Bact < ~1.5 — Bacteroidia growth rate is the dominant control variable.  '
            'CT1 patients with low b_Bact cluster near the boundary; CT2 patient A (b_Bact = 4.5) is deep in the dysbiotic zone.  '
            '→ Structural attractor robustness: initial-condition manipulation alone cannot achieve permanent commensalism.'
        ))

    paste_img(fig, R/'guild_phase'/'phase3d_static.png',
              0.01, CONTENT_BOT, 0.72, CONTENT_H)

    right_panel(fig, 'Key Findings', [
        ('Dysbiotic dominance:', NAVY, True,  9.0),
        ('98% of grid: GDI > 0', RED,  True,  9.5),
        ('Only 2% commensal', RED,  False, 8.5),
        ('', NAVY, False, 4),
        ('Critical threshold:', NAVY, True,  9.0),
        ('b_Bact < ~1.5', BLUE, True,  9.5),
        ('→ commensal equilibrium', BLUE, False, 8.5),
        ('Bacteroidia = dominant', BLUE, False, 8.5),
        ('control variable', BLUE, False, 8.5),
        ('', NAVY, False, 4),
        ('Patient positions:', NAVY, True,  9.0),
        ('CT1 (○): near boundary', BLUE, False, 8.5),
        ('CT2 patient A (b=4.5):', RED,  False, 8.5),
        ('deep in dysbiotic zone', RED,  False, 8.5),
        ('', NAVY, False, 4),
        ('Clinical implication:', NAVY, True,  8.5),
        ('Must lower b_Bact, not', NAVY, False, 8.0),
        ('just change φ(0)', NAVY, False, 8.0),
    ])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    d = pdf.infodict()
    d['Title']  = 'Guild-Level Dynamical Analysis — Dieckow 2024 gLV'
    d['Author'] = 'Keisuke Nishioka'

print(f'Saved: {OUT_PDF}')
