#!/usr/bin/env python3
"""
guild_phase_diagram_3d.py — 3D phase diagram of the gLV attractor landscape.

Sweeps b_Acti × b_Baci × b_Bact on a 3D grid, simulates to t=150d,
computes equilibrium R/G ratio at each point.

Outputs:
  results/guild_phase/phase3d_plotly.html  — interactive 3D isosurface (R/G=0)
  results/guild_phase/phase3d_static.png   — 3-panel 2D slices for paper
  results/guild_phase/phase3d_data.npz     — raw R/G ratio grid
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
})
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.integrate import solve_ivp

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / 'results' / 'guild_phase'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_JSON = HERE / 'results' / 'dieckow_cr' / 'fit_guild.json'
PHI_NPY  = HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy'

T_END_EQ = 80.0
N_GRID   = 18   # per axis  → 18^3 = 5832 points, parallel on 12 cores → ~30s

GUILD_SHORT = {
    'Actinobacteria':      'Acti',
    'Bacilli':             'Baci',
    'Bacteroidia':         'Bact',
    'Betaproteobacteria':  'Beta',
    'Clostridia':          'Clos',
    'Coriobacteriia':      'Cori',
    'Fusobacteriia':       'Fuso',
    'Gammaproteobacteria': 'Gamm',
    'Negativicutes':       'Nega',
    'Other':               'Othr',
}

CT_LABELS = {
    'A': 2, 'B': 2, 'C': 2, 'D': 1, 'E': 1,
    'F': 2, 'G': 1, 'H': 2, 'K': 1, 'L': 1,
}

# ── Load ──────────────────────────────────────────────────────────────────────

def load_data():
    d        = json.loads(FIT_JSON.read_text())
    A        = np.array(d['A'])
    b_all    = np.array(d['b_all'])
    guilds   = d['guilds']
    patients = d['patients']
    short    = [GUILD_SHORT.get(g, g[:4]) for g in guilds]

    phi_all = np.load(PHI_NPY)
    phi0_pp = phi_all[:, 0, :]
    sums    = phi0_pp.sum(axis=1, keepdims=True)
    phi0_pp = np.where(sums > 1e-12, phi0_pp / sums, phi0_pp)

    return A, b_all, phi0_pp, guilds, short, patients


# ── gLV ───────────────────────────────────────────────────────────────────────

def glv_rhs(t, phi, A, b):
    phi = np.clip(phi, 0, 1e3)   # prevent overflow at extreme b values
    return phi * (b + A @ phi)


def equilibrium_gdi(A, b, phi0, short):
    """Simulate to T_END_EQ and return R/G ratio of final state."""
    sol = solve_ivp(glv_rhs, [0, T_END_EQ], phi0,
                    args=(A, b), method='LSODA',
                    rtol=1e-4, atol=1e-6, dense_output=False)
    phi_eq = np.maximum(sol.y[:, -1], 0)
    phi_eq /= max(phi_eq.sum(), 1e-12)

    dys_idx = [short.index(s) for s in ['Bact', 'Fuso', 'Clos'] if s in short]
    com_idx = [short.index(s) for s in ['Baci', 'Acti', 'Nega'] if s in short]
    phi_dys = phi_eq[dys_idx].sum() + 1e-9
    phi_com = phi_eq[com_idx].sum() + 1e-9
    val = float(np.log(phi_dys) - np.log(phi_com))
    # NaN / Inf from overflow → assign extreme dysbiotic value
    if not np.isfinite(val):
        val = 5.0
    return val


# ── 3D grid computation ───────────────────────────────────────────────────────

def _eval_point(args):
    A, b_mean, phi0_mean, short, idx_x, idx_y, idx_z, xv, yv, zv = args
    b = b_mean.copy()
    b[idx_x] = xv
    b[idx_y] = yv
    b[idx_z] = zv
    return equilibrium_gdi(A, b, phi0_mean, short)


def compute_grid(A, b_mean, phi0_mean, short, axes, n_grid=N_GRID):
    """
    axes: list of 3 (axis_name, b_range_lo, b_range_hi)
    Returns: (xs, ys, zs, R/G ratio_grid)  shapes: (n,)^3 + (n,n,n)
    Uses joblib parallel for speed.
    """
    from joblib import Parallel, delayed

    idx_x = short.index(axes[0][0])
    idx_y = short.index(axes[1][0])
    idx_z = short.index(axes[2][0])

    xs = np.linspace(axes[0][1], axes[0][2], n_grid)
    ys = np.linspace(axes[1][1], axes[1][2], n_grid)
    zs = np.linspace(axes[2][1], axes[2][2], n_grid)

    # Build flat list of jobs
    jobs = [(A, b_mean, phi0_mean, short, idx_x, idx_y, idx_z, xv, yv, zv)
            for ix, xv in enumerate(xs)
            for iy, yv in enumerate(ys)
            for iz, zv in enumerate(zs)]

    total = len(jobs)
    print(f"  Parallel jobs: {total} on 12 cores", flush=True)

    results = Parallel(n_jobs=12, verbose=5)(
        delayed(_eval_point)(j) for j in jobs
    )

    R/G ratio = np.array(results).reshape(n_grid, n_grid, n_grid)
    return xs, ys, zs, R/G ratio


# ── Plotly 3D isosurface ──────────────────────────────────────────────────────

def plot_plotly(xs, ys, zs, R/G ratio, axes, b_all, short, patients):
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("plotly not installed, skipping HTML output")
        return

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    # Patient scatter
    idx_x = short.index(axes[0][0])
    idx_y = short.index(axes[1][0])
    idx_z = short.index(axes[2][0])

    ct1_mask = np.array([CT_LABELS.get(p, 2) == 1 for p in patients])
    ct2_mask = ~ct1_mask

    px = b_all[:, idx_x]
    py = b_all[:, idx_y]
    pz = b_all[:, idx_z]

    fig = go.Figure()

    # R/G=0 isosurface
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=R/G ratio.flatten(),
        isomin=-0.05, isomax=0.05,
        surface_count=1,
        colorscale=[[0, 'rgba(150,150,150,0.5)'], [1, 'rgba(150,150,150,0.5)']],
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='R/G ratio = 0 boundary',
    ))

    # Volume coloring: dysbiotic (R/G>0) = red, commensal (R/G<0) = blue
    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=R/G ratio.flatten(),
        isomin=R/G ratio.min(), isomax=R/G ratio.max(),
        opacity=0.08,
        surface_count=12,
        colorscale=[
            [0.0, 'rgba(30,100,200,0.6)'],
            [0.5, 'rgba(240,240,240,0.0)'],
            [1.0, 'rgba(200,40,40,0.6)'],
        ],
        cmin=R/G ratio.min(), cmax=R/G ratio.max(),
        showscale=True,
        colorbar=dict(title='R/G ratio', len=0.5),
        name='R/G ratio volume',
    ))

    # CT1 patients (commensal)
    fig.add_trace(go.Scatter3d(
        x=px[ct1_mask], y=py[ct1_mask], z=pz[ct1_mask],
        mode='markers+text',
        marker=dict(size=8, color='royalblue', symbol='circle',
                    line=dict(color='white', width=1)),
        text=[p for p, m in zip(patients, ct1_mask) if m],
        textposition='top center',
        name='CT1 (commensal)',
    ))

    # CT2 patients (dysbiotic)
    fig.add_trace(go.Scatter3d(
        x=px[ct2_mask], y=py[ct2_mask], z=pz[ct2_mask],
        mode='markers+text',
        marker=dict(size=8, color='firebrick', symbol='square',
                    line=dict(color='white', width=1)),
        text=[p for p, m in zip(patients, ct2_mask) if m],
        textposition='top center',
        name='CT2 (dysbiotic)',
    ))

    axis_labels = [axes[i][0] for i in range(3)]
    fig.update_layout(
        title=dict(text='gLV attractor landscape: R/G ratio = 0 boundary in growth-rate space',
                   font=dict(size=16)),
        scene=dict(
            xaxis_title=f'b<sub>{axis_labels[0]}</sub>',
            yaxis_title=f'b<sub>{axis_labels[1]}</sub>',
            zaxis_title=f'b<sub>{axis_labels[2]}</sub>',
            bgcolor='rgba(240,240,250,1)',
        ),
        legend=dict(x=0.0, y=1.0),
        margin=dict(l=0, r=0, b=0, t=40),
        width=900, height=700,
    )

    out = OUT_DIR / 'phase3d_plotly.html'
    pio.write_html(fig, str(out))
    print(f"Saved: {out}")


# ── Matplotlib 2D slices ──────────────────────────────────────────────────────

def plot_slices(xs, ys, zs, R/G ratio, axes, b_all, short, patients):
    """
    Three 2D slice panels, each with b_Bact (dominant axis) on y-axis.
    Clip R/G ratio to [-2, 4] so the commensal boundary is colour-visible.
    """
    labels = [a[0] for a in axes]   # Acti, Baci, Bact
    cmap   = plt.get_cmap('RdBu_r')
    VMIN, VMAX = -2.0, 4.0          # saturate to emphasise boundary

    # Slice positions: mid Acti, mid Baci, and low-Bact (where commensal exists)
    mid_x  = len(xs) // 2           # mid b_Acti
    mid_y  = len(ys) // 2           # mid b_Baci
    low_z  = max(0, np.searchsorted(zs, 0.5) - 1)   # near b_Bact=0 (commensal zone)

    def scatter(ax, idx_h, idx_v):
        for pi, pat in enumerate(patients):
            ct  = CT_LABELS.get(pat, 2)
            col = '#1a6bbf' if ct == 1 else '#c0392b'   # blue CT1, red CT2
            mk  = 'o'       if ct == 1 else 's'
            ax.scatter(b_all[pi, idx_h], b_all[pi, idx_v],
                       c=col, marker=mk, s=70, zorder=5,
                       edgecolors='white', linewidths=0.9)
            ax.annotate(pat, (b_all[pi, idx_h], b_all[pi, idx_v]),
                        fontsize=7.5, fontweight='bold',
                        ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points',
                        color=col, zorder=6)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: b_Acti (x) × b_Bact (y) ─ slice at mid b_Baci ─────────────
    ax = axs[0]
    sl = R/G ratio[:, mid_y, :]          # shape (n_Acti, n_Bact)
    sl_clip = np.clip(sl, VMIN, VMAX)
    im = ax.contourf(xs, zs, sl_clip.T, levels=40, cmap=cmap,
                     vmin=VMIN, vmax=VMAX)
    try:
        ax.contour(xs, zs, sl.T, levels=[0.0], colors='black', linewidths=2.5)
    except Exception:
        pass
    plt.colorbar(im, ax=ax, label='R/G ratio (eq.)', shrink=0.85)
    scatter(ax, short.index(labels[0]), short.index(labels[2]))
    ax.set_xlabel(f'$b_{{\\rm {labels[0]}}}$ (Actinobacteria)', fontsize=10)
    ax.set_ylabel(f'$b_{{\\rm {labels[2]}}}$ (Bacteroidia)', fontsize=10)
    ax.set_title(f'Slice: $b_{{\\rm {labels[1]}}}$ = {ys[mid_y]:.2f}', fontsize=10)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    # Zone labels
    ax.text(0.03, 0.97, '← Commensal\n   (R/G ratio < 0)', transform=ax.transAxes,
            va='top', fontsize=8, color='#1a6bbf', fontstyle='italic')
    ax.text(0.97, 0.03, 'Dysbiotic →\n(R/G ratio > 0)', transform=ax.transAxes,
            va='bottom', ha='right', fontsize=8, color='#c0392b', fontstyle='italic')

    # ── Panel 2: b_Baci (x) × b_Bact (y) ─ slice at mid b_Acti ─────────────
    ax = axs[1]
    sl = R/G ratio[mid_x, :, :]          # shape (n_Baci, n_Bact)
    sl_clip = np.clip(sl, VMIN, VMAX)
    im = ax.contourf(ys, zs, sl_clip.T, levels=40, cmap=cmap,
                     vmin=VMIN, vmax=VMAX)
    try:
        ax.contour(ys, zs, sl.T, levels=[0.0], colors='black', linewidths=2.5)
    except Exception:
        pass
    plt.colorbar(im, ax=ax, label='R/G ratio (eq.)', shrink=0.85)
    scatter(ax, short.index(labels[1]), short.index(labels[2]))
    ax.set_xlabel(f'$b_{{\\rm {labels[1]}}}$ (Bacilli)', fontsize=10)
    ax.set_ylabel(f'$b_{{\\rm {labels[2]}}}$ (Bacteroidia)', fontsize=10)
    ax.set_title(f'Slice: $b_{{\\rm {labels[0]}}}$ = {xs[mid_x]:.2f}', fontsize=10)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)

    # ── Panel 3: b_Acti (x) × b_Baci (y) ─ slice at low b_Bact ─────────────
    ax = axs[2]
    sl = R/G ratio[:, :, low_z]          # shape (n_Acti, n_Baci) at low Bact
    sl_clip = np.clip(sl, VMIN, VMAX)
    im = ax.contourf(xs, ys, sl_clip.T, levels=40, cmap=cmap,
                     vmin=VMIN, vmax=VMAX)
    try:
        ax.contour(xs, ys, sl.T, levels=[0.0], colors='black', linewidths=2.5)
    except Exception:
        pass
    plt.colorbar(im, ax=ax, label='R/G ratio (eq.)', shrink=0.85)
    scatter(ax, short.index(labels[0]), short.index(labels[1]))
    ax.set_xlabel(f'$b_{{\\rm {labels[0]}}}$ (Actinobacteria)', fontsize=10)
    ax.set_ylabel(f'$b_{{\\rm {labels[1]}}}$ (Bacilli)', fontsize=10)
    ax.set_title(f'Slice: $b_{{\\rm {labels[2]}}}$ = {zs[low_z]:.2f} (low Bact zone)',
                 fontsize=10)

    # ── Shared legend ──────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a6bbf',
               markeredgecolor='white', markersize=9, label='CT1 patients (commensal)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#c0392b',
               markeredgecolor='white', markersize=9, label='CT2 patients (dysbiotic)'),
        Line2D([0], [0], color='k', linewidth=2.5, label='R/G ratio = 0 boundary'),
        Patch(facecolor='#3a7fd5', alpha=0.7, label='Commensal attractor (R/G ratio < 0)'),
        Patch(facecolor='#d53a3a', alpha=0.7, label='Dysbiotic attractor (R/G ratio > 0)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.07), fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        'gLV attractor landscape: equilibrium R/G ratio as a function of guild-specific growth rates\n'
        r'(other $b_i$ fixed at population mean;  colour saturated at R/G ratio $\in [-2,+4]$)',
        fontsize=11, y=1.03)

    plt.tight_layout()
    out = OUT_DIR / 'phase3d_static.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── 3D isosurface (matplotlib + marching cubes) ───────────────────────────────

def plot_3d_isosurface(xs, ys, zs, R/G ratio, axes, b_all, short, patients):
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("skimage not available, skipping 3D isosurface matplotlib plot")
        return

    labels = [a[0] for a in axes]
    verts, faces, normals, values = marching_cubes(R/G ratio, level=0.0,
                                                    spacing=(xs[1]-xs[0],
                                                             ys[1]-ys[0],
                                                             zs[1]-zs[0]))
    # verts are in array index coords — convert to actual b values
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Plot isosurface mesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = Poly3DCollection(verts[faces], alpha=0.35, linewidth=0,
                            facecolor='#888888', edgecolor='none')
    ax.add_collection3d(mesh)

    # Commensal zone hint: scatter a thin cloud of R/G ratio<-0.5 points
    mask_com = R/G ratio < -0.5
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing='ij')
    ax.scatter(X3[mask_com][::4], Y3[mask_com][::4], Z3[mask_com][::4],
               c='royalblue', alpha=0.04, s=4)
    mask_dys = R/G ratio > 0.5
    ax.scatter(X3[mask_dys][::4], Y3[mask_dys][::4], Z3[mask_dys][::4],
               c='firebrick', alpha=0.04, s=4)

    # Patient markers
    idx_x = short.index(labels[0])
    idx_y = short.index(labels[1])
    idx_z = short.index(labels[2])
    for p_idx, pat in enumerate(patients):
        ct  = CT_LABELS.get(pat, 2)
        col = 'royalblue' if ct == 1 else 'firebrick'
        mk  = 'o' if ct == 1 else '^'
        ax.scatter(b_all[p_idx, idx_x], b_all[p_idx, idx_y], b_all[p_idx, idx_z],
                   c=col, marker=mk, s=80, edgecolors='white', linewidths=0.8, zorder=10)
        ax.text(b_all[p_idx, idx_x], b_all[p_idx, idx_y], b_all[p_idx, idx_z] + 0.03,
                pat, fontsize=7, ha='center', color=col)

    ax.set_xlabel(f'$b_{{\\rm {labels[0]}}}$', labelpad=6)
    ax.set_ylabel(f'$b_{{\\rm {labels[1]}}}$', labelpad=6)
    ax.set_zlabel(f'$b_{{\\rm {labels[2]}}}$', labelpad=6)
    ax.set_title('gLV attractor boundary (R/G ratio = 0 isosurface)\n'
                 'Blue dots = commensal zone, Red = dysbiotic zone',
                 fontsize=11)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
               markersize=9, label='CT1 patients (commensal)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='firebrick',
               markersize=9, label='CT2 patients (dysbiotic)'),
        plt.Rectangle((0,0), 1, 1, fc='#888888', alpha=0.4, label='R/G ratio = 0 surface'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper left')

    plt.tight_layout()
    out = OUT_DIR / 'phase3d_isosurface.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading gLV data...")
    A, b_all, phi0_pp, guilds, short, patients = load_data()

    b_mean   = b_all.mean(axis=0)
    phi0_mean = phi0_pp.mean(axis=0)
    phi0_mean /= max(phi0_mean.sum(), 1e-12)

    # Axis ranges: cover actual patient range with 0.3 margin
    def rng(s):
        idx = short.index(s)
        lo  = b_all[:, idx].min() - 0.3
        hi  = b_all[:, idx].max() + 0.3
        return (s, lo, hi)

    axes = [rng('Acti'), rng('Baci'), rng('Bact')]

    print(f"\nComputing 3D R/G ratio grid ({N_GRID}^3 = {N_GRID**3} points)...")
    print(f"  Axes: {[a[0] for a in axes]}")
    print(f"  Ranges: {[(round(a[1],2), round(a[2],2)) for a in axes]}")

    # Check cache
    cache = OUT_DIR / 'phase3d_data.npz'
    if cache.exists():
        print("  Loading cached grid...")
        data = np.load(cache)
        xs, ys, zs, R/G ratio = data['xs'], data['ys'], data['zs'], data['R/G ratio']
        # Check grid size matches
        if R/G ratio.shape[0] != N_GRID:
            print("  Grid size mismatch, recomputing...")
            xs, ys, zs, R/G ratio = compute_grid(A, b_mean, phi0_mean, short, axes)
            np.savez(cache, xs=xs, ys=ys, zs=zs, R/G ratio=R/G ratio)
    else:
        xs, ys, zs, R/G ratio = compute_grid(A, b_mean, phi0_mean, short, axes)
        np.savez(cache, xs=xs, ys=ys, zs=zs, R/G ratio=R/G ratio)
        print(f"  Saved cache: {cache}")

    print(f"\nR/G ratio range: [{R/G ratio.min():.2f}, {R/G ratio.max():.2f}]")
    dys_frac = (R/G ratio > 0).sum() / R/G ratio.size
    print(f"Dysbiotic fraction of grid: {dys_frac:.1%}")

    print("\nGenerating 2D slice figure (paper-ready)...")
    plot_slices(xs, ys, zs, R/G ratio, axes, b_all, short, patients)

    try:
        from skimage.measure import marching_cubes  # noqa: F401
        print("Generating 3D isosurface (matplotlib)...")
        plot_3d_isosurface(xs, ys, zs, R/G ratio, axes, b_all, short, patients)
    except ImportError:
        print("skimage not found — skipping matplotlib 3D isosurface")

    print("Generating interactive 3D plot (plotly HTML)...")
    plot_plotly(xs, ys, zs, R/G ratio, axes, b_all, short, patients)

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == '__main__':
    main()
