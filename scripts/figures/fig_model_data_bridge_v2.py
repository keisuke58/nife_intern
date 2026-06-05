# [nife-pathshim]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

"""
がっつり版：gLV数理モデル × 実験データの3本柱

① 固定点解析 + 安定性 (eigenvalue) + 全患者軌道の PCA重ね
② 速度ベクトル方向テスト: gLV予測Δφ vs 実測Δφ (患者別b使用)
③ MDI軌道: 各患者の実測 vs gLV積分予測
"""

import csv, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.integrate    import solve_ivp
from scipy.optimize     import fsolve
from scipy.stats        import pearsonr, spearmanr
from sklearn.decomposition import PCA

import pub_style
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS, GUILD_SHORT

ROOT    = Path(__file__).parents[2]
JDIR    = ROOT / 'Datasets' / 'joshi2025_SI'
RESDIR  = ROOT / 'results' / 'dieckow_cr'
OUT     = RESDIR / 'figs'
OUT.mkdir(exist_ok=True)

GUILDS  = GUILD_ORDER
GUILDS_9 = [g for g in GUILDS if g != 'Other']
N9      = len(GUILDS_9)

# ── load gLV fit ─────────────────────────────────────────────────────────────

fit      = json.load(open(RESDIR / 'fit_glv_hamilton_kegg_expanded.json'))
fit_glds = fit['guilds']
pats_ord = fit['patients']   # ['A','B','C','D','E','F','G','H','K','L']
A_fit    = np.array(fit['A'])
b_all    = np.array(fit['b_all'])   # (10, 10)

idx = [fit_glds.index(g) for g in GUILDS]
A   = A_fit[np.ix_(idx, idx)]
b_mat = b_all[:, idx]               # (10 patients, 10 guilds)

A9_idx = [GUILDS.index(g) for g in GUILDS_9]
A9     = A[np.ix_(A9_idx, A9_idx)]
b9_mat = b_mat[:, A9_idx]           # (10, 9) — per-patient b for GUILDS_9
b9_mean = b9_mat.mean(axis=0)

# per-patient dict: {'A': b_9, ...}
b9_pat = {p: b9_mat[i] for i, p in enumerate(pats_ord)}

# ── load Dieckow observations ─────────────────────────────────────────────────

df = pd.read_csv(RESDIR / 'dieckow_guild_abundance.csv')
ct_pred  = json.load(open(RESDIR / 'cross_ct_prediction.json'))
ct1_pats = ct_pred['ct1_patients']

df['pat'] = df['sample'].str[0]
df['tp']  = df['sample'].str[2].astype(int)   # 1,2,3

# normalise to sum=1 over GUILDS_9
for g in GUILDS_9:
    if g not in df.columns:
        df[g] = 0.0
phi_raw = df[GUILDS_9].values.astype(float)
phi_raw[phi_raw < 0] = 0
row_sums = phi_raw.sum(axis=1, keepdims=True)
phi_norm = phi_raw / (row_sums + 1e-12)
df[GUILDS_9] = phi_norm

ct1_arr = df.loc[df['pat'].isin(ct1_pats),  GUILDS_9].mean().values
ct2_arr = df.loc[~df['pat'].isin(ct1_pats), GUILDS_9].mean().values
ct1_arr /= ct1_arr.sum(); ct2_arr /= ct2_arr.sum()

# ── gLV ODE (replicator form) ─────────────────────────────────────────────────

def glv_rhs(phi, b, A):
    phi  = np.clip(phi, 0, None)
    phi /= phi.sum() + 1e-12
    r    = b + A @ phi
    return phi * (r - phi @ r)

def integrate_glv(phi0, b, A, T=60, n=600):
    phi0 = np.clip(phi0, 1e-4, None); phi0 /= phi0.sum()
    sol  = solve_ivp(lambda t, x: glv_rhs(x, b, A), [0, T], phi0,
                     t_eval=np.linspace(0, T, n), method='RK45',
                     rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T

# ── load Joshi ────────────────────────────────────────────────────────────────

def load_joshi(path):
    rows = list(csv.reader(open(path))); hd = rows[0]
    hi, pi = hd.index('mean_health'), hd.index('mean_periimplantitis')
    return {r[0]: (float(r[hi]), float(r[pi])) for r in rows[1:] if r[0]}

j16s = load_joshi(JDIR / 'joshi2025_suppE_class.csv')
jrna = load_joshi(JDIR / 'joshi2025_suppG_class_rnaseq.csv')

def guild_vec(d):
    h = np.array([d.get(g,(0,0))[0] for g in GUILDS_9])
    p = np.array([d.get(g,(0,0))[1] for g in GUILDS_9])
    return h/h.sum(), p/p.sum()

j16s_h, j16s_p = guild_vec(j16s)
jrna_h, jrna_p = guild_vec(jrna)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

pub_style.apply()
fig = plt.figure(figsize=(17, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.50, wspace=0.38,
                        top=0.94, bottom=0.06, left=0.07, right=0.97)

ax_pca  = fig.add_subplot(gs[0, :2])   # PCA + trajectories + fixed points
ax_eig  = fig.add_subplot(gs[0, 2])    # eigenvalue stability
ax_vel  = fig.add_subplot(gs[1, :2])   # velocity direction test
ax_cs   = fig.add_subplot(gs[1, 2])    # cosine similarity distribution
ax_mdi  = fig.add_subplot(gs[2, :2])   # MDI actual vs predicted
ax_mdi2 = fig.add_subplot(gs[2, 2])    # MDI scatter observed vs predicted

# ═══════════════════════════════════════════════════════════════════════════
# ① FIXED POINT ANALYSIS + PCA PHASE PORTRAIT
# ═══════════════════════════════════════════════════════════════════════════

# find fixed points from many random starts
fps = []
for _ in range(800):
    phi0 = np.random.dirichlet(np.ones(N9))
    try:
        def eqs(phi):
            phi = np.clip(phi, 0, None)
            s   = phi.sum()
            if s < 1e-10: return np.ones(N9)
            phi = phi / s
            r   = b9_mean + A9 @ phi
            return phi * (r - phi @ r)
        sol = fsolve(eqs, phi0, full_output=True)
        fp  = sol[0]
        fp  = np.clip(fp, 0, None)
        if fp.sum() < 1e-8: continue
        fp /= fp.sum()
        # check stationarity
        if np.linalg.norm(glv_rhs(fp, b9_mean, A9)) < 1e-6:
            # deduplicate
            if not any(np.linalg.norm(fp - x) < 0.02 for x in fps):
                fps.append(fp)
    except Exception:
        pass

print(f"Found {len(fps)} fixed points")

# Jacobian eigenvalues at each fixed point
def jacobian(fp, b, A):
    n  = len(fp)
    J  = np.zeros((n, n))
    r  = b + A @ fp
    avg_r = fp @ r
    for i in range(n):
        for j in range(n):
            J[i, j] = (fp[i] * A[i, j]
                       - fp[i] * (r[j] + (A[:, j] * fp).sum())
                       + (i == j) * (r[i] - avg_r))
    return J

fp_stability = []
for fp in fps:
    J     = jacobian(fp, b9_mean, A9)
    eigs  = np.linalg.eigvals(J)
    max_re = eigs.real.max()
    fp_stability.append({'fp': fp, 'eigs': eigs, 'max_re': max_re,
                         'stable': max_re < 0})

# PCA on all patient observations + FPs
all_pts  = df[GUILDS_9].values
pca      = PCA(n_components=2)
pca.fit(all_pts)
pts_pc   = pca.transform(all_pts)
ct1_pc   = pca.transform(ct1_arr.reshape(1, -1))[0]
ct2_pc   = pca.transform(ct2_arr.reshape(1, -1))[0]
j16sh_pc = pca.transform(j16s_h.reshape(1, -1))[0]
j16sp_pc = pca.transform(j16s_p.reshape(1, -1))[0]
fp_pc    = pca.transform(np.array([s['fp'] for s in fp_stability]))

# Patient trajectories in PCA
pat_col  = {}
cmap_ct1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(ct1_pats)))
cmap_ct2 = plt.cm.Reds(np.linspace(0.4, 0.9, 10 - len(ct1_pats)))
for i, p in enumerate([p for p in pats_ord if p in ct1_pats]):
    pat_col[p] = cmap_ct1[i]
for i, p in enumerate([p for p in pats_ord if p not in ct1_pats]):
    pat_col[p] = cmap_ct2[i]

for pat in pats_ord:
    rows = df[df['pat'] == pat].sort_values('tp')
    traj = pca.transform(rows[GUILDS_9].values)
    ax_pca.plot(traj[:, 0], traj[:, 1], '-o', color=pat_col[pat],
                markersize=4, lw=1.2, alpha=0.85)
    ax_pca.text(traj[0, 0], traj[0, 1], pat, fontsize=6.5,
                color=pat_col[pat], ha='center', va='bottom')

# Fixed points
for s in fp_stability:
    pc = pca.transform(s['fp'].reshape(1, -1))[0]
    mk = 'o' if s['stable'] else 'x'
    col = '#222' if s['stable'] else '#aaa'
    ax_pca.scatter(pc[0], pc[1], marker=mk, s=120, color=col,
                   edgecolors='k', lw=0.8, zorder=6)
    if s['stable']:
        ax_pca.annotate(f"FP\n(stable)", pc, fontsize=6.5, xytext=(5, 5),
                        textcoords='offset points', color='#222')

# Joshi + Dieckow CT markers
for lbl, vec, col, mk in [
    ('Dieckow CT1',   ct1_arr,  '#2166ac', 's'),
    ('Dieckow CT2',   ct2_arr,  '#d6604d', 'D'),
    ('Joshi health',  j16s_h,   '#92c5de', '^'),
    ('Joshi peri',    j16s_p,   '#f4a582', 'v'),
]:
    pc = pca.transform(vec.reshape(1, -1))[0]
    ax_pca.scatter(pc[0], pc[1], color=col, marker=mk, s=90,
                   edgecolors='k', lw=0.8, zorder=7)
    ax_pca.annotate(lbl, pc, fontsize=7, xytext=(4, 3),
                    textcoords='offset points')

from matplotlib.lines import Line2D
leg_handles = [
    Line2D([0], [0], color='none', marker='o', mfc='#2166ac', mec='k', ms=7, label='CT1 patients (trajectory)'),
    Line2D([0], [0], color='none', marker='o', mfc='#d6604d', mec='k', ms=7, label='CT2 patients (trajectory)'),
    Line2D([0], [0], color='none', marker='o', mfc='#222',    mec='k', ms=9, label='Stable fixed point'),
    Line2D([0], [0], color='none', marker='x', mfc='#aaa',    mec='#aaa', ms=9, label='Unstable FP'),
]
ax_pca.legend(handles=leg_handles, fontsize=7, loc='upper right')
ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
ax_pca.set_title('①  gLV fixed points + patient trajectories (PCA space)\n'
                 f'Stable FPs = •, Unstable = ×  [found {len(fps)} total, '
                 f'{sum(s["stable"] for s in fp_stability)} stable]',
                 fontsize=9, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue panel
# ─────────────────────────────────────────────────────────────────────────────

stable_fps = [s for s in fp_stability if s['stable']]
for si, s in enumerate(stable_fps[:3]):
    eigs = s['eigs']
    ax_eig.scatter(eigs.real, eigs.imag, s=40, alpha=0.8, label=f'FP{si+1}')

ax_eig.axvline(0, color='k', lw=1, ls='--')
ax_eig.set_xlabel('Re(λ)', fontsize=9)
ax_eig.set_ylabel('Im(λ)', fontsize=9)
ax_eig.set_title('① Jacobian eigenvalues\nat stable fixed points',
                  fontsize=9, fontweight='bold')
ax_eig.legend(fontsize=7)
ax_eig.text(0.02, 0.95, 'Re(λ)<0 → stable', transform=ax_eig.transAxes,
            fontsize=7, va='top', color='#333')

# ═══════════════════════════════════════════════════════════════════════════
# ② VELOCITY DIRECTION TEST
# gLV predicted Δφ direction vs observed Δφ direction (per-patient b)
# ═══════════════════════════════════════════════════════════════════════════

cos_sims   = []
r_vals     = []
records    = []

for pat in pats_ord:
    rows = df[df['pat'] == pat].sort_values('tp')
    if len(rows) < 2:
        continue
    b_p  = b9_pat[pat]
    phis = rows[GUILDS_9].values  # (3, 9)

    for ti in range(len(rows) - 1):
        phi_t  = phis[ti]
        phi_t1 = phis[ti + 1]

        # observed change
        obs    = phi_t1 - phi_t

        # gLV predicted velocity at phi_t (Δt=1 week, approximate)
        pred   = glv_rhs(phi_t, b_p, A9)

        # cosine similarity
        norm_o = np.linalg.norm(obs);  norm_p = np.linalg.norm(pred)
        if norm_o < 1e-8 or norm_p < 1e-8:
            continue
        cos    = (obs @ pred) / (norm_o * norm_p)
        r, _   = pearsonr(obs, pred)

        cos_sims.append(cos)
        r_vals.append(r)
        records.append({'pat': pat, 'tp': ti+1, 'cos': cos, 'r': r,
                        'ct': 'CT1' if pat in ct1_pats else 'CT2'})

cos_sims = np.array(cos_sims)
r_vals   = np.array(r_vals)

# bar plot per patient-timepoint pair
rec_df = pd.DataFrame(records)
colors_vel = [GUILD_COLORS.get('Bacilli', '#2166ac') if r['ct'] == 'CT1' else '#d6604d'
              for _, r in rec_df.iterrows()]
labels_vel = [f"{r['pat']}_t{r['tp']}→t{r['tp']+1}" for _, r in rec_df.iterrows()]

x_v = np.arange(len(rec_df))
bars = ax_vel.bar(x_v, rec_df['cos'], color=colors_vel, edgecolor='k', lw=0.4, alpha=0.85)
ax_vel.axhline(0, color='k', lw=0.8)
ax_vel.axhline(np.mean(cos_sims), color='purple', lw=1.5, ls='--',
               label=f'mean cosine = {np.mean(cos_sims):.3f}')
ax_vel.set_xticks(x_v)
ax_vel.set_xticklabels(labels_vel, rotation=45, ha='right', fontsize=7)
ax_vel.set_ylabel('Cosine similarity\n(gLV Δφ̂ vs observed Δφ)', fontsize=8)
ax_vel.set_title('②  Velocity direction test: does gLV predict the right direction of community change?\n'
                 '(cos > 0 = correct direction; per-patient b_i used; blue=CT1 patients, red=CT2)',
                 fontsize=8.5, fontweight='bold')
ax_vel.legend(fontsize=8)

# significance test (sign test: H0 = mean cos = 0)
from scipy.stats import ttest_1samp, wilcoxon
t_stat, t_p = ttest_1samp(cos_sims, 0)
try:
    w_stat, w_p = wilcoxon(cos_sims)
except Exception:
    w_p = float('nan')
ax_vel.text(0.01, 0.97,
            f'n={len(cos_sims)} transitions\nmean cos={np.mean(cos_sims):.3f}  '
            f't-test p={t_p:.3f}\nWilcoxon p={w_p:.3f}',
            transform=ax_vel.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

# ── Cosine distribution ───────────────────────────────────────────────────────
ax_cs.hist(cos_sims, bins=8, color='steelblue', edgecolor='k', alpha=0.8)
ax_cs.axvline(0, color='k', lw=1, ls='--')
ax_cs.axvline(np.mean(cos_sims), color='purple', lw=1.5, ls='-',
              label=f'mean={np.mean(cos_sims):.2f}')
ax_cs.set_xlabel('Cosine similarity', fontsize=9)
ax_cs.set_ylabel('Count', fontsize=9)
ax_cs.set_title('② Distribution\n(positive = model\npredicts right direction)',
                fontsize=8.5, fontweight='bold')
ax_cs.legend(fontsize=7)
pos_frac = np.mean(cos_sims > 0)
ax_cs.text(0.05, 0.95, f'{pos_frac*100:.0f}% positive',
           transform=ax_cs.transAxes, fontsize=8, va='top', color='purple')

# ═══════════════════════════════════════════════════════════════════════════
# ③ MDI: actual vs gLV predicted trajectories
# guild-level proxy: Clostridia / (Bacilli + Actinobacteria)
# (Pseudoramibacter=Clostridia, Gemella=Bacilli, Capnocytophaga=Bacteroidia
#  → simplified as Clostridia / (Bacilli + Actinobacteria))
# ═══════════════════════════════════════════════════════════════════════════

CLO_I = GUILDS_9.index('Clostridia')
BAC_I = GUILDS_9.index('Bacilli')
ACT_I = GUILDS_9.index('Actinobacteria')
BCT_I = GUILDS_9.index('Bacteroidia')

def mdi(phi):
    num = phi[CLO_I]
    den = phi[BAC_I] + phi[ACT_I] + 1e-9
    return np.log10(num / den + 1e-9)

TIMEPOINTS = [0, 7, 21]   # days: week 1, 2, 3

obs_mdi_all  = []
pred_mdi_all = []

for pat in pats_ord:
    rows = df[df['pat'] == pat].sort_values('tp')
    phis = rows[GUILDS_9].values
    b_p  = b9_pat[pat]
    col  = '#2166ac' if pat in ct1_pats else '#d6604d'

    # observed MDI at 3 timepoints
    obs_mdi  = [mdi(phis[ti]) for ti in range(len(phis))]

    # predicted: integrate from tp1, evaluate at day 7 and 21
    t_sol, traj = integrate_glv(phis[0], b_p, A9, T=25, n=250)
    pred_mdi = [mdi(traj[0])]
    for day in TIMEPOINTS[1:]:
        idx_t = np.argmin(np.abs(t_sol - day))
        pred_mdi.append(mdi(traj[idx_t]))

    ax_mdi.plot(TIMEPOINTS[:len(obs_mdi)],  obs_mdi,  'o-', color=col, alpha=0.85,
                lw=1.2, markersize=4)
    ax_mdi.plot(TIMEPOINTS[:len(pred_mdi)], pred_mdi, 's--', color=col, alpha=0.50,
                lw=1.0, markersize=3)

    obs_mdi_all.extend(obs_mdi)
    pred_mdi_all.extend(pred_mdi[:len(obs_mdi)])

# Legend
from matplotlib.lines import Line2D
lh = [Line2D([0],[0],color='#2166ac',lw=1.5,marker='o',label='CT1 observed'),
      Line2D([0],[0],color='#2166ac',lw=1.5,marker='s',ls='--',alpha=0.6,label='CT1 gLV predicted'),
      Line2D([0],[0],color='#d6604d',lw=1.5,marker='o',label='CT2 observed'),
      Line2D([0],[0],color='#d6604d',lw=1.5,marker='s',ls='--',alpha=0.6,label='CT2 gLV predicted')]
ax_mdi.legend(handles=lh, fontsize=7, loc='lower right', ncol=2)
ax_mdi.set_xticks(TIMEPOINTS)
ax_mdi.set_xticklabels(['Day 0\n(wk1)', 'Day 7\n(wk2)', 'Day 21\n(wk3)'], fontsize=8)
ax_mdi.set_xlabel('Timepoint', fontsize=9)
ax_mdi.set_ylabel('MDI proxy  log₁₀(Clostridia / (Bacilli + Actinobacteria))', fontsize=8)
ax_mdi.set_title('③  MDI proxy: actual patient trajectories (solid) vs gLV prediction from t=0 (dashed)\n'
                 '(each line = 1 patient; integrated with patient-specific b_i)',
                 fontsize=8.5, fontweight='bold')

# ── MDI scatter: obs vs pred ──────────────────────────────────────────────────
obs_arr  = np.array(obs_mdi_all)
pred_arr = np.array(pred_mdi_all)
r_mdi, p_mdi = pearsonr(obs_arr, pred_arr)
lim = [min(obs_arr.min(), pred_arr.min()) - 0.3,
       max(obs_arr.max(), pred_arr.max()) + 0.3]
ax_mdi2.plot(lim, lim, '--', color='gray', lw=0.8)
ax_mdi2.scatter(obs_arr, pred_arr, color='steelblue', s=35, alpha=0.7,
                edgecolors='k', lw=0.4)
ax_mdi2.set_xlabel('MDI observed', fontsize=9)
ax_mdi2.set_ylabel('MDI gLV predicted', fontsize=9)
ax_mdi2.set_title(f'③  MDI obs vs pred\nr={r_mdi:.2f}, p={p_mdi:.3f}',
                   fontsize=9, fontweight='bold')
ax_mdi2.set_xlim(lim); ax_mdi2.set_ylim(lim)
ax_mdi2.text(0.05, 0.95, f'n={len(obs_arr)} obs\nr={r_mdi:.2f}  p={p_mdi:.3f}',
             transform=ax_mdi2.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

fig.suptitle('gLV model × experimental data — 3-axis validation\n'
             '① Fixed-point stability  ·  ② Velocity direction test  ·  ③ MDI trajectory prediction',
             fontsize=10.5, fontweight='bold')

for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'fig_model_data_bridge_v2.{ext}', bbox_inches='tight', dpi=150)

plt.close()
print("Done →", OUT / 'fig_model_data_bridge_v2.pdf')
