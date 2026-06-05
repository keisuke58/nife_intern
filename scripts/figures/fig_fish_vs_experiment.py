# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_fish_vs_experiment.py — FISH voxel composition vs Heine experiment CSV (C).

Compares the z-integrated species composition from FISH 3D imaging (day 21,
averaged over x-y-z and all FOVs per condition) with the replicate median from
the Heine 2025 culture experiment (day 21 HOBIC).

Species order in IC files: [So, An, Vd/Vp, Fn, Pg]

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_fish_vs_experiment.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from thesis_style import use, clean_ax, SP_COLORS

ROOT   = pathlib.Path(__file__).parents[2]
IC     = ROOT / "results" / "fish_3d" / "ic"
CSV    = pathlib.Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species"
                      "/experiment_data/fig3_species_distribution_replicates.csv")
OUT    = ROOT / "results" / "fig_fish_vs_experiment"

SHORT  = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
COLORS = SP_COLORS

# CSV species names per condition → maps to SHORT index
SP_MAP = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',
                  'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula',
                  'F. nucleatum', 'P. gingivalis_W83'],
}
CONDS = [('CH', 'Commensal', 'HOBIC'), ('DH', 'Dysbiotic', 'HOBIC')]


def fish_composition(cond, day=21):
    """Mean species fraction across x-y-z voxels from IC file."""
    f = IC / f"phi_xyz_{cond}_d{day}.npy"
    if not f.exists():
        return None
    arr = np.load(f)               # (128, 128, Nz, 5)
    mean = arr.mean(axis=(0, 1, 2))  # (5,) — mean over all voxels
    total = mean.sum()
    return mean / total if total > 0 else mean


def csv_composition(condition, cultivation, day=21):
    """Median species fraction from Heine CSV."""
    df = pd.read_csv(CSV)
    sub = df[(df['condition'] == condition) &
             (df['cultivation'] == cultivation) &
             (df['day'] == day)]
    sp_names = SP_MAP[condition]
    vals = []
    for sp in sp_names:
        rows = sub[sub['species'] == sp]['distribution_pct'].values
        vals.append(np.median(rows) if len(rows) > 0 else 0.0)
    vals = np.array(vals)
    total = vals.sum()
    return vals / total if total > 0 else vals


def main():
    fig, axes = plt.subplots(1, 2, figsize=use(width_frac=0.9, aspect=0.55),
                             sharey=False)

    for ax, (fish_cond, csv_cond, csv_cult) in zip(axes, CONDS):
        fish = fish_composition(fish_cond, day=21)
        csv  = csv_composition(csv_cond, csv_cult, day=21)

        if fish is None:
            ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        x = np.arange(5)
        w = 0.35
        ax.bar(x - w/2, fish * 100, w, color=COLORS, alpha=0.9,
               edgecolor='white', label='FISH (imaging)')
        ax.bar(x + w/2, csv  * 100, w, color=COLORS, alpha=0.45,
               edgecolor='white', hatch='//', label='Culture (CSV median)')

        ax.set_xticks(x)
        ax.set_xticklabels(SHORT, fontsize=7.5)
        ax.set_ylabel(r'Relative abundance (\%)', fontsize=8)
        ax.set_title(fish_cond, fontsize=9)
        ax.set_ylim(0, 100)
        clean_ax(ax)

        # Pearson correlation
        r = np.corrcoef(fish, csv)[0, 1]
        ax.text(0.97, 0.97, f'$r={r:.2f}$', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5)

    # shared legend
    handles = [
        mpatches.Patch(color='0.5', alpha=0.9,            label='FISH 3-D imaging (day 21)'),
        mpatches.Patch(color='0.5', alpha=0.45, hatch='//', label='Culture experiment (day 21 median)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=7.5, framealpha=0.9, edgecolor='0.8',
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(r'FISH imaging vs.\ culture experiment: day-21 composition',
                 fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}')
        print(f'saved → {OUT}.{ext}')


if __name__ == '__main__':
    main()
