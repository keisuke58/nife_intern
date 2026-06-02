#!/usr/bin/env python3
"""
plot_outlier_analysis.py
Diagnostic figure: worst-predicted patients (H, C) in SS v2 LOO.
Shows observed vs SS-v2-predicted trajectories + oscillation metrics.
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_here   = Path(__file__).resolve().parents[2]
OUT_DIR = _here / 'results' / 'dieckow_cr'

phi_all  = np.load(_here / 'results' / 'dieckow_otu' / 'phi_guild.npy')  # (10,3,10)
d        = json.load(open(OUT_DIR / 'fit_hamilton_kegg_steadystate_v2.json'))
A_np     = np.array(d['A'])
b_all    = np.array(d['b_all'])
PATIENTS = list('ABCDEFGHKL')
GUILDS   = ['Actinobacteria','Bacilli','Bacteroidia','Betaproteobacteria','Clostridia',
            'Coriobacteriia','Fusobacteriia','Gammaproteobacteria','Negativicutes','Other']
n_sp     = 10
SS_ITER, SS_DT = 2000, 0.03

LOO = {'A':0.084,'B':0.091,'C':0.118,'D':0.065,'E':0.052,
       'F':0.082,'G':0.054,'H':0.176,'K':0.088,'L':0.066}

COLORS = ['#4C8DC4','#5BB85B','#E0873A','#9B59B6','#D62728',
          '#17BECF','#BCBD22','#F7B6D2','#8C6D31','#AAAAAA']
SHORT  = ['Actin.','Bacil.','Bact.','β-Prot.','Clost.',
          'Coriob.','Fusob.','γ-Prot.','Negat.','Other']

def damped_iter(A, b, phi0, n_iter=SS_ITER, dt=SS_DT):
    phi = phi0.copy().astype(np.float64)
    for _ in range(n_iter):
        f = b + A @ phi
        phi = phi * np.exp(dt * f)
        phi = np.maximum(phi, 1e-12)
        phi /= phi.sum()
    return phi

# ── Compute SS v2 predictions for all patients ────────────────────────────────
preds = {}
for pi, pat in enumerate(PATIENTS):
    phi2 = damped_iter(A_np, b_all[pi], phi_all[pi, 0])
    phi3 = damped_iter(A_np, b_all[pi], phi2)
    preds[pat] = (phi2, phi3)

# ── Oscillation score = Σ |Δwk1→2| + |Δwk2→3| per guild, normalised ─────────
osc_scores = {}
for pi, pat in enumerate(PATIENTS):
    d12 = np.abs(phi_all[pi, 1] - phi_all[pi, 0])
    d23 = np.abs(phi_all[pi, 2] - phi_all[pi, 1])
    # non-monotonicity: how much wk3 reverses the wk1→2 trend
    sign_reverse = np.sum((phi_all[pi,1]-phi_all[pi,0]) * (phi_all[pi,2]-phi_all[pi,1]) < 0)
    osc_scores[pat] = {'d12': d12.sum(), 'd23': d23.sum(), 'reversals': int(sign_reverse)}

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32,
                        left=0.07, right=0.97, top=0.90, bottom=0.12)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False})

# ── Top-left: LOO-RMSE bar coloured by CT ────────────────────────────────────
ax_bar = fig.add_subplot(gs[0, 0])
ct2 = {'A','B','C','F','H'}   # Actinobacteria-moderate
pats_sorted = sorted(LOO, key=lambda p: LOO[p], reverse=True)
bar_colors = ['#E0873A' if p in ct2 else '#4C8DC4' for p in pats_sorted]
bars = ax_bar.barh(pats_sorted, [LOO[p] for p in pats_sorted],
                   color=bar_colors, edgecolor='k', linewidth=0.5)
ax_bar.axvline(0.0875, color='k', lw=1.0, ls='--', label='mean 0.0875')
ax_bar.set_xlabel('LOO-RMSE (SS v2)', fontsize=9)
ax_bar.set_title('LOO-RMSE by patient', fontweight='bold', fontsize=10)
for i, (p, b_) in enumerate(zip(pats_sorted, bars)):
    ax_bar.text(LOO[p]+0.003, b_.get_y()+b_.get_height()/2,
                f'{LOO[p]:.3f}', va='center', fontsize=8)
patch_ct1 = mpatches.Patch(color='#4C8DC4', label='CT1 Bacilli-dom.')
patch_ct2 = mpatches.Patch(color='#E0873A', label='CT2 Actin-mod.')
ax_bar.legend(handles=[patch_ct1, patch_ct2], fontsize=8, loc='lower right')

# ── Top-middle: oscillation score vs LOO-RMSE scatter ────────────────────────
ax_osc = fig.add_subplot(gs[0, 1])
for pat in PATIENTS:
    rev = osc_scores[pat]['reversals']
    ax_osc.scatter(osc_scores[pat]['d12'] + osc_scores[pat]['d23'],
                   LOO[pat], s=60,
                   c='#E0873A' if pat in ct2 else '#4C8DC4',
                   edgecolors='k', linewidths=0.5, zorder=3)
    ax_osc.annotate(pat, (osc_scores[pat]['d12']+osc_scores[pat]['d23'],
                          LOO[pat]),
                    textcoords='offset points', xytext=(4, 2), fontsize=8)
ax_osc.set_xlabel('Total guild-fraction oscillation\n(Σ|Δwk1→2| + Σ|Δwk2→3|)', fontsize=9)
ax_osc.set_ylabel('LOO-RMSE', fontsize=9)
ax_osc.set_title('LOO-RMSE vs dynamics oscillation', fontweight='bold', fontsize=10)

# ── Top-right: non-monotone reversals bar ────────────────────────────────────
ax_rev = fig.add_subplot(gs[0, 2])
rev_vals = [osc_scores[p]['reversals'] for p in pats_sorted]
ax_rev.barh(pats_sorted, rev_vals,
            color=['#E0873A' if p in ct2 else '#4C8DC4' for p in pats_sorted],
            edgecolor='k', linewidth=0.5)
ax_rev.set_xlabel('# guilds with wk2→wk3 direction reversal', fontsize=9)
ax_rev.set_title('Non-monotone guild reversals', fontweight='bold', fontsize=10)
ax_rev.axvline(5, color='gray', lw=0.8, ls=':')

# ── Bottom: Patient H and C detailed trajectories ─────────────────────────────
for col, pat in enumerate(['H', 'C']):
    pi = PATIENTS.index(pat)
    ax = fig.add_subplot(gs[1, col])
    phi_obs = phi_all[pi]       # (3, 10)
    phi2p, phi3p = preds[pat]
    weeks = [1, 2, 3]

    for gi in range(n_sp):
        obs_line = [phi_obs[t, gi] for t in range(3)]
        ax.plot(weeks, obs_line, '-o', color=COLORS[gi], lw=1.8,
                ms=5, label=SHORT[gi] if col == 1 else None)
        pred_line = [phi_obs[0, gi], phi2p[gi], phi3p[gi]]
        ax.plot(weeks, pred_line, '--s', color=COLORS[gi], lw=1.2,
                ms=4, alpha=0.6)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Week 1', 'Week 2', 'Week 3'])
    ax.set_ylabel('Guild fraction φ', fontsize=9)
    loo_v = LOO[pat]
    rev_n = osc_scores[pat]['reversals']
    ax.set_title(f'Patient {pat}  —  LOO-RMSE={loo_v:.3f},  {rev_n}/10 guild reversals',
                 fontweight='bold', fontsize=10)
    ax.text(0.5, -0.16,
            '— solid: observed  · · · dashed: SS v2 predicted',
            transform=ax.transAxes, ha='center', fontsize=8, color='#555')
    ax.set_ylim(0, 0.7)

# Patient H annotation
pi_H = PATIENTS.index('H')
ax_H = fig.get_axes()[3]
ax_H.annotate('Bacilli: 57%→30%→50%\n(U-shaped, unpredictable)',
              xy=(2, phi_all[pi_H, 1, 1]),
              xytext=(2.15, 0.45),
              arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
              fontsize=8, color='#c0392b')

# Bottom-right: summary text box
ax_txt = fig.add_subplot(gs[1, 2])
ax_txt.axis('off')
summary = (
    "Root causes of high LOO-RMSE\n"
    "─────────────────────────────\n\n"
    "▶  Patient H  (LOO-RMSE = 0.176)\n"
    "   Bacilli: 57% → 30% → 50% (U-shape)\n"
    "   8/10 guilds reverse direction wk2→wk3\n"
    "   → Non-monotone transient dynamics\n"
    "   → Possibly: external perturbation,\n"
    "     measurement noise in wk2 sample,\n"
    "     or multi-attractor basin crossing\n\n"
    "▶  Patient C  (LOO-RMSE = 0.118)\n"
    "   Negativicutes: 5% → 16% → 7%\n"
    "   Gammaproteobacteria: 15% → 6% → 18%\n"
    "   → Transient peak/valley pattern\n"
    "   → ODE predicts monotone convergence,\n"
    "     misses transient overshoot\n\n"
    "▶  Model limitation:\n"
    "   SS v2 predicts convergence to a\n"
    "   single steady state; cannot capture\n"
    "   reversals or transient overshoots\n"
    "   within a 3-week window."
)
ax_txt.text(0.02, 0.97, summary, transform=ax_txt.transAxes,
            fontsize=8.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdf0ef',
                      edgecolor='#c0392b', linewidth=1.2))

# Legend for line plots
handles = [mpatches.Patch(color=COLORS[i], label=SHORT[i]) for i in range(n_sp)]
fig.legend(handles=handles, loc='lower center', ncol=10,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02),
           title='Guild class', title_fontsize=8)

fig.suptitle('LOO-CV Failure Analysis — Patients H & C (SS v2 model)',
             fontsize=12, fontweight='bold', y=0.97)

for ext in ('png', 'pdf'):
    fig.savefig(OUT_DIR / f'fig_outlier_analysis.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved fig_outlier_analysis.*')
