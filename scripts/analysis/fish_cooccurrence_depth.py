# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fish_cooccurrence_depth.py — Depth-stratified Pg-pair co-occurrence (B deep-dive).

Three panels:
  A. Pg-pair Pearson r vs normalised depth [0=substrate,1=surface], per condition
  B. Pg-involved vs non-Pg sign agreement (binomial test)
  C. Day-by-day sign agreement evolution

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/analysis/fish_cooccurrence_depth.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import pearsonr
from scipy.stats import binom

def binom_test(k, n, p, alternative='greater'):
    if alternative == 'greater':
        return binom.sf(k-1, n, p)
    return binom.cdf(k, n, p)
from scipy.interpolate import interp1d

from thesis_style import use, clean_ax, PALETTE
from paper_data import paper_5sp_theta
from attractor_analysis import theta_to_matrices

ROOT     = pathlib.Path(__file__).parents[2]
IC       = ROOT / "results" / "fish_3d" / "ic"
OUT      = ROOT / "results" / "fish_cooccurrence_depth"
OUT_JSON = ROOT / "results" / "fish_cooccurrence_depth_stats.json"

SHORT  = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
SP_COL = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
DAYS   = [1, 6, 10, 15, 21]
CONDS  = ['CH', 'DH']
N_SP, PG = 5, 4
PAIRS    = [(i, j) for i in range(N_SP) for j in range(i+1, N_SP)]
PG_PAIRS = [(i, j) for (i, j) in PAIRS if j == PG]
N_ZGRID  = 50


def pearson_z(arr):
    """(128,128,Nz,5) → (Nz, 10) Pearson r per pair."""
    nz = arr.shape[2]
    r_z = np.full((nz, len(PAIRS)), np.nan)
    for k, (i, j) in enumerate(PAIRS):
        xi = arr[:, :, :, i].reshape(-1, nz)
        xj = arr[:, :, :, j].reshape(-1, nz)
        for z in range(nz):
            x, y = xi[:, z], xj[:, z]
            m = (x > 0) | (y > 0)
            if m.sum() > 15:
                r_z[z, k], _ = pearsonr(x[m], y[m])
    return r_z


def load_A(cond):
    return theta_to_matrices(paper_5sp_theta(cond))[0]


# ── collect profiles ──────────────────────────────────────────────────────────
zgrid = np.linspace(0, 1, N_ZGRID)
r_profiles = {c: {k: [] for k in range(len(PAIRS))} for c in CONDS}
r_by_day   = {c: {d: {} for d in DAYS} for c in CONDS}

for cond in CONDS:
    for day in DAYS:
        f = IC / f"phi_xyz_{cond}_d{day}.npy"
        if not f.exists():
            continue
        arr  = np.load(f)
        nz   = arr.shape[2]
        r_z  = pearson_z(arr)
        znorm = np.linspace(0, 1, nz)
        for k in range(len(PAIRS)):
            col   = r_z[:, k]
            valid = ~np.isnan(col)
            if valid.sum() > 2:
                interp = interp1d(znorm[valid], col[valid], kind='linear',
                                  bounds_error=False, fill_value=np.nan)
                r_profiles[cond][k].append(interp(zgrid))
            r_by_day[cond][day][k] = float(np.nanmean(col[valid])) if valid.sum() else np.nan


def mean_profile(cond, k):
    rows = [r for r in r_profiles[cond][k] if not np.all(np.isnan(r))]
    return np.nanmean(rows, axis=0) if rows else np.full(N_ZGRID, np.nan)


# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.68))
gs  = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.38)
ax_ch  = fig.add_subplot(gs[0, 0])
ax_dh  = fig.add_subplot(gs[1, 0])
ax_bar = fig.add_subplot(gs[:, 1])
ax_day = fig.add_subplot(gs[:, 2])

# ── A/B: Pg-pair depth profiles ───────────────────────────────────────────────
for ax, cond in zip([ax_ch, ax_dh], CONDS):
    A = load_A(cond)
    for i, j in PG_PAIRS:
        k   = PAIRS.index((i, j))
        rp  = mean_profile(cond, k)
        sym = A[i, j] + A[j, i]
        ax.plot(rp, zgrid, color=SP_COL[i], lw=1.2,
                ls='-' if sym < 0 else '--',
                label=f'{SHORT[i]}--Pg')
    ax.axvline(0, color='black', lw=0.6)
    ax.axhspan(0.7, 1.0, color='#d62728', alpha=0.07)
    ax.text(0.98, 0.83, 'Pg zone', fontsize=6, color='#d62728',
            transform=ax.transAxes, ha='right')
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0, 1); ax.invert_yaxis()
    ax.set_ylabel("Norm. depth\n(0=substrate)", fontsize=7)
    ax.set_xlabel("Pearson $r$", fontsize=7.5)
    ax.set_title(cond, fontsize=8)
    clean_ax(ax)

leg_h = [mlines.Line2D([], [], color=SP_COL[i], lw=1.2,
                        label=f'{SHORT[i]}--Pg') for i in range(PG)]
leg_h += [mlines.Line2D([], [], color='grey', lw=1.2, ls='-',
                         label='$A_{sym}<0$ (competitive)'),
          mlines.Line2D([], [], color='grey', lw=1.2, ls='--',
                         label='$A_{sym}>0$ (facilitative)')]
ax_ch.legend(handles=leg_h, fontsize=5.5, loc='upper left',
             framealpha=0.85, edgecolor='0.8')

# ── C: Pg vs non-Pg bar ───────────────────────────────────────────────────────
agree_pg, total_pg   = {c: 0 for c in CONDS}, {c: 0 for c in CONDS}
agree_np, total_np   = {c: 0 for c in CONDS}, {c: 0 for c in CONDS}

for cond in CONDS:
    A = load_A(cond)
    for i, j in PAIRS:
        k   = PAIRS.index((i, j))
        sym = A[i, j] + A[j, i]
        if sym == 0:
            continue
        rs  = [r_by_day[cond][d][k] for d in DAYS
               if k in r_by_day[cond][d] and not np.isnan(r_by_day[cond][d][k])]
        if not rs:
            continue
        mr = np.nanmean(rs)
        ag = int(np.sign(sym) == np.sign(mr))
        if j == PG:
            total_pg[cond] += 1; agree_pg[cond] += ag
        else:
            total_np[cond] += 1; agree_np[cond] += ag

x = np.arange(2)
for ci, cond in enumerate(CONDS):
    c  = PALETTE['ch'] if cond == 'CH' else PALETTE['dh']
    off = ci * 0.32 - 0.16
    tp, ap = total_pg[cond], agree_pg[cond]
    tn, an = total_np[cond], agree_np[cond]
    ax_bar.bar(0 + off, ap/max(tp,1)*100, 0.3, color=c, alpha=0.85, edgecolor='white')
    ax_bar.bar(1 + off, an/max(tn,1)*100, 0.3, color=c, alpha=0.40, edgecolor='white')
    ax_bar.text(0+off, ap/max(tp,1)*100+2, f'{ap}/{tp}', ha='center',
                va='bottom', fontsize=6.5, color=c)
    ax_bar.text(1+off, an/max(tn,1)*100+2, f'{an}/{tn}', ha='center',
                va='bottom', fontsize=6.5, color=c)
    pv = binom_test(ap, tp, 0.5, alternative='greater') if tp else 1.0
    if pv < 0.15:
        ax_bar.text(0+off, ap/max(tp,1)*100+9, f'p={pv:.2f}',
                    ha='center', va='bottom', fontsize=5.5, color='0.4')

ax_bar.axhline(50, color='black', lw=0.6, ls='--', alpha=0.5)
ax_bar.text(1.55, 51, 'chance', fontsize=6, color='0.4', va='bottom')
ax_bar.set_xticks([0,1])
ax_bar.set_xticklabels(['Pg-involved\npairs', 'Non-Pg\npairs'], fontsize=7.5)
ax_bar.set_ylabel(r'Sign agreement (\%)', fontsize=8)
ax_bar.set_ylim(0, 110)
clean_ax(ax_bar)
ax_bar.legend(handles=[mpatches.Patch(color='#2196A6', label='CH'),
                        mpatches.Patch(color='#D95F02', label='DH')],
              fontsize=7, framealpha=0.85, edgecolor='0.8')

# ── D: day-by-day agreement ────────────────────────────────────────────────────
for cond in CONDS:
    c = PALETTE['ch'] if cond == 'CH' else PALETTE['dh']
    A = load_A(cond)
    day_ag = []
    for day in DAYS:
        ag, tot = 0, 0
        for i, j in PAIRS:
            k   = PAIRS.index((i, j))
            sym = A[i, j] + A[j, i]
            if sym == 0 or k not in r_by_day[cond][day]:
                continue
            mr = r_by_day[cond][day][k]
            if np.isnan(mr):
                continue
            tot += 1; ag += int(np.sign(sym) == np.sign(mr))
        day_ag.append(ag/tot*100 if tot else np.nan)
    ax_day.plot(DAYS, day_ag, 'o-', color=c, lw=1.2, ms=4, label=cond)

ax_day.axhline(50, color='black', lw=0.6, ls='--', alpha=0.5)
ax_day.set_xlabel('Day', fontsize=8)
ax_day.set_ylabel(r'Sign agreement (\%)', fontsize=8)
ax_day.set_ylim(0, 100)
ax_day.set_xticks(DAYS)
ax_day.legend(fontsize=7, framealpha=0.85, edgecolor='0.8')
clean_ax(ax_day)

fig.suptitle(r'Depth-stratified spatial co-occurrence: Pg-involved pairs',
             fontsize=9)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', bbox_inches='tight')
    print(f'saved → {OUT}.{ext}')

# ── JSON ──────────────────────────────────────────────────────────────────────
OUT_JSON.write_text(json.dumps({
    c: {
        'pg_agree': int(agree_pg[c]), 'pg_total': int(total_pg[c]),
        'nonpg_agree': int(agree_np[c]), 'nonpg_total': int(total_np[c]),
        'pg_binom_p': float(binom_test(agree_pg[c], total_pg[c], 0.5,
                                       alternative='greater') if total_pg[c] else 1.0),
    } for c in CONDS
}, indent=2))
print(f'stats → {OUT_JSON}')
for cond in CONDS:
    print(f'{cond}  Pg: {agree_pg[cond]}/{total_pg[cond]}  non-Pg: {agree_np[cond]}/{total_np[cond]}')
