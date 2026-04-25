#!/usr/bin/env python3
"""
Reproduce Dieckow 2024 Supplementary Fig 1 panels c, d, e.

c: Biofilm volume (µm³ × 10⁶)
d: % volume non-permeable "live" cells
e: Area covered by biofilm (µm² × 10⁶)

Data: CLSM_digitized_boxplots_week1-3_patientA-L.csv
  columns: parameter, patient, week, q1, median, q3
  (n=5 CLSM images per patient × week, summarised as Q1/median/Q3)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})

DATA_CSV = Path(__file__).parent / 'Szafranski_Published_Work' / 'Szafranski_Published_Work' \
           / 'public_data' / 'Dieckow' / 'CLSM_digitized_boxplots_week1-3_patientA-L.csv'
OUT_DIR  = Path(__file__).parent / 'results' / 'dieckow_otu'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = list('ABCDEFGHIJKL')   # 12 patients in CLSM study
WEEKS    = [1, 2, 3]

# Colors matching original figure
WEEK_COLORS = {1: '#2166ac', 2: '#d62728', 3: '#2ca02c'}
WEEK_LABELS = {1: '1', 2: '2', 3: '3'}

# Patient symbols (matching Dieckow Supp Fig 1 x-axis symbols)
PATIENT_MARKERS = {
    'A': ('^', 'k',  True,  6),    # filled triangle up
    'B': ('v', 'k',  True,  6),    # filled inverted triangle
    'C': ('o', 'k',  True,  5),    # filled circle
    'D': ('+', 'k',  False, 7),    # plus
    'E': ('x', 'k',  False, 7),    # x
    'F': ('o', 'k',  False, 5),    # open circle
    'G': ('s', 'k',  False, 5),    # open square
    'H': ('^', 'k',  False, 6),    # open triangle
    'I': ('s', 'k',  True,  5),    # filled square
    'J': ('D', 'k',  False, 5),    # open diamond
    'K': ('D', 'k',  True,  5),    # filled diamond
    'L': ('*', 'k',  True,  7),    # star
}

BOX_W    = 0.22   # box width
OFFSETS  = {1: -0.28, 2: 0.0, 3: 0.28}   # x-offset per week within patient group


def sanitize_stats(q1, med, q3, clip_min=None, clip_max=None, fill_missing=None):
    q1, med, q3 = float(q1), float(med), float(q3)
    if np.isnan(q1) or np.isnan(med) or np.isnan(q3):
        if fill_missing is None:
            return np.nan, np.nan, np.nan
        q1 = med = q3 = float(fill_missing)
    if clip_min is not None:
        q1 = max(q1, clip_min)
        med = max(med, clip_min)
        q3 = max(q3, clip_min)
    if clip_max is not None:
        q1 = min(q1, clip_max)
        med = min(med, clip_max)
        q3 = min(q3, clip_max)
    q1, med, q3 = np.sort([q1, med, q3])
    return float(q1), float(med), float(q3)


def draw_boxplot(ax, x_center, q1, med, q3, color, box_w=BOX_W):
    if np.isnan(q1) or np.isnan(med) or np.isnan(q3):
        return

    h = q3 - q1
    if h <= 1e-12:
        ax.hlines(med, x_center - box_w/2, x_center + box_w/2,
                  colors=color, linewidths=3.0, alpha=0.55, zorder=5)
        return

    rect = mpatches.Rectangle(
        (x_center - box_w/2, q1), box_w, h,
        linewidth=0.8, edgecolor=color, facecolor=color, alpha=0.45, zorder=3,
    )
    ax.add_patch(rect)
    rect2 = mpatches.Rectangle(
        (x_center - box_w/2, q1), box_w, h,
        linewidth=0.8, edgecolor=color, facecolor='none', zorder=4,
    )
    ax.add_patch(rect2)
    ax.hlines(med, x_center - box_w/2, x_center + box_w/2,
              colors=color, linewidths=1.5, zorder=5)


def make_panel(ax, df_param, ylabel, title_label, ylim=None):
    """Draw one panel (c, d, or e) for a given parameter."""
    clip_min = None
    clip_max = None
    fill_missing = None
    if not df_param.empty:
        p = str(df_param['parameter'].iloc[0])
        if p in {'volume', 'area'}:
            clip_min = 0.0
            fill_missing = 0.0
        elif p == 'viability':
            clip_min, clip_max = 0.0, 100.0
            fill_missing = 0.0

    for pi, patient in enumerate(PATIENTS):
        for week in WEEKS:
            sub = df_param[(df_param['patient'] == patient) & (df_param['week'] == week)]
            if sub.empty:
                continue
            q1, med, q3 = sanitize_stats(
                sub['q1'].values[0], sub['median'].values[0], sub['q3'].values[0],
                clip_min=clip_min, clip_max=clip_max, fill_missing=fill_missing,
            )
            x   = pi + OFFSETS[week]
            draw_boxplot(ax, x, q1, med, q3, WEEK_COLORS[week])

    # X-axis: patient labels with symbols
    ax.set_xticks(range(len(PATIENTS)))
    # Labels: letter above, symbol below — approximate with text + marker overlay
    ax.set_xticklabels(PATIENTS, fontsize=8)

    ax.set_xlabel('Patient', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(-0.6, len(PATIENTS) - 0.4)
    if ylim:
        ax.set_ylim(ylim)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3, zorder=1)

    # Add patient marker symbols below axis
    for pi, patient in enumerate(PATIENTS):
        mk, col, filled, ms = PATIENT_MARKERS[patient]
        mfc = col if filled else 'none'
        ax.plot(pi, -0.18,
                marker=mk, color=col, markerfacecolor=mfc, markersize=ms,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)

    # Legend for weeks
    legend_patches = [
        mpatches.Patch(facecolor=WEEK_COLORS[w], edgecolor=WEEK_COLORS[w],
                       alpha=0.7, label=f'{w}')
        for w in WEEKS
    ]
    leg = ax.legend(handles=legend_patches, title='Weeks', title_fontsize=8,
                    fontsize=7.5, loc='upper right', frameon=True,
                    framealpha=0.9, edgecolor='#cccccc',
                    handlelength=1.2, handleheight=1.0,
                    borderpad=0.5, labelspacing=0.3)
    leg._legend_box.align = 'left'

    ax.text(-0.08, 1.04, title_label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')


def main():
    df = pd.read_csv(DATA_CSV)
    print(f'Loaded {len(df)} rows')
    print(f'Parameters: {df["parameter"].unique()}')
    print(f'Patients: {sorted(df["patient"].unique())}')
    print(f'Weeks: {sorted(df["week"].unique())}')
    missing = df[['q1', 'median', 'q3']].isna().any(axis=1)
    if bool(missing.any()):
        miss = df.loc[missing, ['parameter', 'patient', 'week', 'q1', 'median', 'q3']]
        print('WARNING: Missing box stats detected (these are drawn as 0 by default):')
        print(miss.to_string(index=False))

    df_vol  = df[df['parameter'] == 'volume'].copy()
    df_via  = df[df['parameter'] == 'viability'].copy()
    df_area = df[df['parameter'] == 'area'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    make_panel(axes[0], df_vol,
               ylabel='Biofilm volume (µm³ × 10⁶)',
               title_label='c', ylim=(-0.8, 11.5))

    make_panel(axes[1], df_via,
               ylabel='% volume non-permeable cells',
               title_label='d', ylim=(-5, 115))

    make_panel(axes[2], df_area,
               ylabel='Area (µm² × 10⁶)',
               title_label='e', ylim=(-0.5, 10))

    fig.tight_layout(pad=1.2)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'suppfig1_cde.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=300)
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
