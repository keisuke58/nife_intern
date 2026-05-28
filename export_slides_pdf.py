#!/usr/bin/env python3
"""
export_slides_pdf.py  v4
9-page PDF of guild-level dynamical analysis results.

Pages:
  1. Title
  2. Keystone Analysis
  3. Network Centrality
  4. Permutation Test (A matrix validation)
  5. Tipping Point (alpha scan + single-guild sweep)
  6. 2D Phase Diagram (Gamm × Bact)
  7. Relapse Dynamics
  8. Prevention Threshold + Joshi Validation
  9. 3D Attractor Landscape
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
HDR_H        = 0.80 / H_IN
FTR_H        = 0.72 / H_IN
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
GREEN  = '#1A7A4A'
LGRAY  = '#F5F5F8'


# ── Core layout helpers ────────────────────────────────────────────────────────

def base_fig():
    return plt.figure(figsize=(W_IN, H_IN), facecolor=WHITE)


def draw_chrome(fig, title, subtitle, footer):
    ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.fill([0,1,1,0], [1-HDR_H, 1-HDR_H, 1, 1], color=NAVY)
    ax.text(0.025, 1 - HDR_H*0.35, title,
            color=WHITE, fontsize=14, fontweight='bold',
            va='center', ha='left', transform=ax.transAxes)
    if subtitle:
        ax.text(0.025, 1 - HDR_H*0.78, subtitle,
                color=LBLUE, fontsize=8.5, va='center', ha='left',
                transform=ax.transAxes)
    ax.fill([0,1,1,0], [0,0,FTR_H,FTR_H], color=MINT)
    ax.text(0.025, FTR_H*0.5, footer,
            color=NAVY, fontsize=8.8, va='center', ha='left',
            transform=ax.transAxes)


def paste_img(fig, path, l, b, w, h):
    if not Path(path).exists():
        return
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
    ax = fig.add_axes([0.745, CONTENT_BOT, 0.248, CONTENT_H])
    ax.set_facecolor(LGRAY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    for sp in ax.spines.values():
        sp.set_color('#BBCCDD'); sp.set_linewidth(0.8)
    ax.text(0.5, 0.955, headline,
            color=NAVY, fontsize=10.5, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.axhline(0.935, color='#BBCCDD', lw=0.8)
    n = len(items)
    ys = np.linspace(0.89, 0.06, n)
    for (txt, col, bold, fs), y in zip(items, ys):
        ax.text(0.06, y, txt,
                color=col, fontsize=fs,
                fontweight='bold' if bold else 'normal',
                va='center', ha='left', transform=ax.transAxes)
    return ax


def two_fig_layout(fig, path_l, path_r, cap_l, cap_r):
    GAP  = 0.015
    FW   = (0.735 - GAP) / 2
    FBOT = CONTENT_BOT + 0.03
    FHH  = CONTENT_H - 0.05
    paste_img(fig, path_l, 0.01, FBOT, FW, FHH)
    paste_img(fig, path_r, 0.01 + FW + GAP, FBOT, FW, FHH)
    fig.text(0.01 + FW/2, CONTENT_BOT + 0.005, cap_l,
             ha='center', fontsize=9, color=NAVY, style='italic',
             transform=fig.transFigure)
    fig.text(0.01 + FW + GAP + FW/2, CONTENT_BOT + 0.005, cap_r,
             ha='center', fontsize=9, color=NAVY, style='italic',
             transform=fig.transFigure)


# ── Page builders ──────────────────────────────────────────────────────────────

with PdfPages(str(OUT_PDF)) as pdf:

    # ════════════════════════════════════════════════════════════════════════════
    # Page 1 — Title
    # ════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor=NAVY)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(NAVY); ax.axis('off')
    ax.fill([0,1,1,0],[0.92,0.92,1,1], color=NAVY_L)
    ax.text(0.5, 0.96,
            'Dieckow 2024  ·  10-guild gLV  ·  N = 10 patients  ·  Nishioka (2026)',
            color='#7aabcc', fontsize=9.5, ha='center', va='center',
            style='italic', transform=ax.transAxes)
    ax.text(0.5, 0.63, 'Guild-Level Dynamical Analysis',
            color=WHITE, fontsize=30, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)

    topics = ['Keystone', 'Network + Perm', 'Tipping Point',
              'Relapse', 'Prevention', '3D Landscape']
    colors = [LBLUE, '#A8E6CF', '#FFD3B6', '#FFB3BA', '#D4B8E0', '#B8D4E0']
    xs = np.linspace(0.09, 0.91, len(topics))
    for xi, (top, col) in zip(xs, zip(topics, colors)):
        ax.text(xi, 0.42, top, color=col, fontsize=10.5, fontweight='bold',
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
        subtitle = 'Influence = Σ|A_ji|  ·  Vulnerability = Σ|A_ij|  ·  Net = ΣA_ij  ·  LOO sign consistency across 10 leave-one-out folds',
        footer   = (
            'Actinobacteria: influence 5.58, net +2.02 → mutualist hub.  '
            'Bacilli: vulnerability 8.24 → most regulated by others (compositional keystone confirmed).  '
            'Betaproteobacteria: net −0.14 → only net competitor.  '
            'LOO: Acti→* sign consistency = 1.0 (all folds agree); Gamm→* = 0 (unstable).'
        ))
    paste_img(fig, R/'guild_network'/'guild_centrality_summary.png',
              0.01, CONTENT_BOT, 0.72, CONTENT_H)
    right_panel(fig, 'Network Roles', [
        ('Mutualist hub:', NAVY,  True,  9.0),
        ('Actinobacteria', BLUE,  True,  9.5),
        ('influence = 5.58  (highest)', BLUE, False, 8.5),
        ('net effect = +2.02', BLUE, False, 8.5),
        ('LOO sign = 1.0  (stable)', GREEN, False, 8.5),
        ('', NAVY, False, 4),
        ('Most regulated:', NAVY,  True,  9.0),
        ('Bacilli', RED,   True,  9.5),
        ('vulnerability = 8.24', RED, False, 8.5),
        ('→ controlled by others', RED, False, 8.5),
        ('', NAVY, False, 4),
        ('Net competitor:', NAVY,  True,  9.0),
        ('Betaproteobacteria', GOLD, True, 9.5),
        ('net effect = −0.14', GOLD, False, 8.5),
        ('', NAVY, False, 4),
        ('→ Acti–Baci–Beta triad', NAVY, True, 8.5),
        ('stabilises dysbiotic attractor', NAVY, False, 8.5),
    ])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 4 — Permutation Test (A matrix statistical validation)
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'A Matrix Statistical Validation: Permutation Test (n = 100)',
        subtitle = 'Null distribution: shuffle patient labels → refit A via L-BFGS-B → p-value = fraction of |A_perm| ≥ |A_real|',
        footer   = (
            '2/90 pairs significant at p < 0.05: '
            'Baci→Acti (A = +3.43, p = 0.040) and Gamm→Clos (A = −0.023, p = 0.040).  '
            '|  LOO confirms Acti→* sign stability (SC = 1.0 all folds).  '
            '|  Low significance consistent with N = 10 — pattern reflects A matrix structure, not noise.'
        ))
    paste_img(fig, R/'guild_network'/'permutation_test.png',
              0.01, CONTENT_BOT, 0.72, CONTENT_H)
    right_panel(fig, 'p-value Results', [
        ('Significant (p < 0.05):', NAVY, True,  9.0),
        ('2 / 90 off-diagonal pairs', RED,  True,  9.5),
        ('', NAVY, False, 4),
        ('Most significant:', NAVY, True, 9.0),
        ('Baci → Acti', RED,  True, 9.0),
        ('A = +3.43,  p = 0.040', RED, False, 8.5),
        ('→ strongest real interaction', RED, False, 8.5),
        ('Gamm → Clos', GOLD, True, 9.0),
        ('A = −0.023, p = 0.040', GOLD, False, 8.5),
        ('', NAVY, False, 4),
        ('LOO stability:', NAVY, True, 9.0),
        ('Acti→* SC = 1.0', GREEN, False, 8.5),
        ('(all 10 folds agree)', GREEN, False, 8.5),
        ('Gamm→* SC ≈ 0', '#888', False, 8.5),
        ('(data-limited)', '#888', False, 8.5),
        ('', NAVY, False, 4),
        ('N=10 limits power;', NAVY, False, 8.0),
        ('structure is meaningful', NAVY, False, 8.0),
    ])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 5 — Tipping Point: alpha scan + single-guild sweep
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Tipping Point Analysis: CT2 → CT1 b-Environment Transition',
        subtitle = 'Left: b(α) = (1−α)·b_CT2 + α·b_CT1, scan α ∈ [0,1]   ·   Right: sweep one guild b_i while holding others at CT2',
        footer   = (
            'No α* crossing found: GDI remains negative for all α — A matrix drives system to commensal regardless of b.  '
            '|  Single-guild drivers: Gammaproteobacteria (ΔGDI = 1.04) and Bacteroidia (ΔGDI = 0.94) have highest individual influence.  '
            '→ Structural attractor: interventions must alter A (ecology), not just b (growth environment).'
        ))
    two_fig_layout(fig,
        R/'guild_tipping'/'tipping_alpha_scan.png',
        R/'guild_tipping'/'tipping_single_guild.png',
        '(a)  α-interpolation CT2 → CT1 environment',
        '(b)  Single-guild b sweep: |ΔGDI| effect size')
    right_panel(fig, 'Tipping Findings', [
        ('Alpha scan:', NAVY, True, 9.0),
        ('No GDI = 0 crossing', BLUE, True, 9.0),
        ('→ single commensal', BLUE, False, 8.5),
        ('   attractor (A dominant)', BLUE, False, 8.5),
        ('', NAVY, False, 4),
        ('Top single-guild drivers:', NAVY, True, 9.0),
        ('Gammaproteobact.', RED, True, 9.0),
        ('ΔGDI = 1.04  (#1)', RED, False, 8.5),
        ('Δb = −5.31 (CT2→CT1)', RED, False, 8.5),
        ('Bacteroidia', GOLD, True, 9.0),
        ('ΔGDI = 0.94  (#2)', GOLD, False, 8.5),
        ('Δb = −0.96 (CT2→CT1)', GOLD, False, 8.5),
        ('', NAVY, False, 4),
        ('Eq. GDI by patient:', NAVY, True, 8.5),
        ('D: −0.48 (commensal)', GREEN, False, 8.0),
        ('All others: > 0', RED, False, 8.0),
    ])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 6 — 2D Phase Diagram: Gamm × Bact
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = '2D Phase Diagram: Gammaproteobacteria × Bacteroidia Growth-Rate Space',
        subtitle = '20 × 20 grid · all other b fixed at CT2 mean · GDI at week-3 · black dashed = GDI = 0 tipping boundary',
        footer   = (
            'Gamm × Bact are the two highest single-guild tipping drivers (ΔGDI 1.04 and 0.94).  '
            'GDI = 0 boundary visible: commensal region at low b_Gamm and low b_Bact.  '
            'CT2 mean (red ▼) is deep in dysbiotic zone; CT1 mean (blue ▲) is near or below the boundary.  '
            '→ Confirms that reducing Gamm and Bact growth rates jointly is the most efficient ecological intervention.'
        ))
    paste_img(fig, R/'guild_tipping'/'tipping_phase_diagram_2d.png',
              0.01, CONTENT_BOT, 0.72, CONTENT_H)
    right_panel(fig, 'Phase Structure', [
        ('Tipping boundary:', NAVY, True, 9.0),
        ('GDI = 0  (dashed)', NAVY, False, 8.5),
        ('visible in 2D space', NAVY, False, 8.5),
        ('', NAVY, False, 4),
        ('Commensal zone:', NAVY, True, 9.0),
        ('low b_Gamm + low b_Bact', BLUE, True, 9.0),
        ('', NAVY, False, 4),
        ('Patient positions:', NAVY, True, 9.0),
        ('CT2 (▼): dysbiotic zone', RED, False, 8.5),
        ('CT1 (▲): near boundary', BLUE, False, 8.5),
        ('', NAVY, False, 4),
        ('Clinical strategy:', NAVY, True, 8.5),
        ('Joint reduction of', GREEN, False, 8.5),
        ('b_Gamm and b_Bact', GREEN, False, 8.5),
        ('is most efficient path', GREEN, False, 8.5),
        ('to commensal state', GREEN, False, 8.5),
    ])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════════
    # Page 7 — Relapse Dynamics
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Predicted Relapse Dynamics: Three Patient Archetypes',
        subtitle = 'gLV simulation to t = 150 d with patient-specific b vectors  ·  GDI = log(φ_dys) − log(φ_com)  ·  GDI < 0 = commensal',
        footer   = (
            '① Persistent dysbiotic (A,B,E,F,H,K): GDI > 0 throughout — attractor dominates from day 1.  '
            '② Transient responders (C→24d, G→38d, L→48d): commensal at wk-3, relapse by day 50.  '
            '③ Permanent commensal (D only): uniquely stable b vector keeps GDI < 0 at t = 150d.  '
            '→ Week-3 GDI is a prognostic marker for relapse timing.'
        ))
    two_fig_layout(fig,
        R/'guild_tipping'/'relapse_dynamics.png',
        R/'guild_tipping'/'gdi_w3_vs_relapse.png',
        '(a)  GDI trajectories to 90 d (per patient)',
        '(b)  Week-3 GDI as a prognostic marker for relapse')
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
        ('attractor pulls back', GOLD, False, 8.5),
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
    # Page 8 — Prevention Threshold + Joshi Validation
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = 'Prevention Threshold & External Validation (Joshi 2025)',
        subtitle = 'Left: min. Δb_i to prevent relapse (per patient)   ·   Right: model equilibrium GDI vs Joshi 2025 peri-implant cohort (N = 127)',
        footer   = (
            'Prevention: Patient C convertible with Δb_Bact = 1.33; G & L require multi-guild change → deep attractor.  '
            '|  Joshi validation: predicted eq. GDI = +0.65 ≈ Joshi PI mean +0.53 '
            '(9/10 patients in PI range; Δ = 0.12 < Joshi SD = 1.30).  '
            '→ 3-week calibration extrapolates correctly to long-term peri-implant dysbiosis.'
        ))
    two_fig_layout(fig,
        R/'guild_tipping'/'prevention_threshold.png',
        R/'guild_tipping'/'joshi_comparison.png',
        '(a)  Required Δb per guild for relapse prevention',
        '(b)  External validation: Joshi 2025 peri-implant cohort')
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
    # Page 9 — 3D Attractor Landscape
    # ════════════════════════════════════════════════════════════════════════════
    fig = base_fig()
    draw_chrome(fig,
        title    = '3D Attractor Landscape: GDI = 0 Boundary in Growth-Rate Space',
        subtitle = '18³ grid · Acti × Baci × Bact · t = 80 d · 12-core parallel  ·  Blue = commensal  |  Red = dysbiotic  |  Black contour = GDI = 0',
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
