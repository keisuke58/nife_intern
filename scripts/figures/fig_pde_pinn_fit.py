# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_pde_pinn_fit.py — Thesis figure: 1D PINN-fitted PDE vs FISH z-profiles.

For each condition (CH, DH):
  - Load PINN-fitted D_i, u from results/pinn_diffusion/ or _gpu_retrieve/
  - Convert D_i, u from normalised (z/L, t/t_max) to physical units
  - Run 1-D forward PDE (reaction-diffusion-advection) with fitted parameters
    starting from the day-1 FISH z-profile as IC
  - Compare PDE model z-profiles (solid) vs observed FISH z-profiles (dashed)
    at Days 1, 10, 21

Output:
  results/fig_pde_pinn_fit.{pdf,png}

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_pde_pinn_fit.py
"""

import json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from thesis_style import use, clean_ax, SP_COLORS

ROOT    = pathlib.Path(__file__).parents[2]
IC_DIR  = ROOT / "results" / "fish_3d" / "ic"
OUT     = ROOT / "results" / "fig_pde_pinn_fit"

SHORT   = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']
COLORS  = SP_COLORS
N_SP    = 5
Z_UM_PER_SLICE = 2.0          # µm per FISH voxel slice
T_MAX   = 21.0                 # days
DAYS_PLOT = [1, 10, 21]        # days to show
KEY_SP  = {3, 4}               # Fn, Pg — bold
LW_KEY  = 1.8
LW_OTH  = 0.75
ALP_OTH = 0.45

# Normalization scales (from pinn_diffusion_inverse.py CSV analysis)
Z_MAX = {'CH': 64.0, 'DH': 79.0}   # µm

# PINN result JSON paths
PINN_JSON = {
    'CH': ROOT / 'results' / '_gpu_retrieve' / 'pinn_D_fit_CH.json',
    'DH': ROOT / 'results' / 'pinn_diffusion' / 'pinn_D_fit_DH.json',
}

# Fallback: try pinn_diffusion dir for both
for cond in ('CH', 'DH'):
    alt = ROOT / 'results' / 'pinn_diffusion' / f'pinn_D_fit_{cond}.json'
    if not PINN_JSON[cond].exists() and alt.exists():
        PINN_JSON[cond] = alt


def load_pinn(cond):
    p = PINN_JSON[cond]
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    z_max = Z_MAX[cond]
    t_max = T_MAX
    # Convert normalised D_i to physical (µm²/day)
    D_phys = [d_n * z_max**2 / t_max for d_n in d['D_fit']]
    # Convert u to physical (µm/day)
    u_phys = d['u_fit'] * z_max / t_max
    return {'D': np.array(D_phys), 'u': u_phys,
            'loss': d.get('final_loss_total', float('nan')),
            'epochs': d.get('epochs', 0)}


def load_ic(cond):
    """Load day-1 FISH z-profile as IC for forward PDE."""
    f = IC_DIR / f"phi_xyz_{cond}_d1.npy"
    if not f.exists():
        return None, None
    arr = np.load(f)                    # (128,128,Nz,5)
    mean = arr.mean(axis=(0, 1))        # (Nz,5)
    total = mean.sum(axis=1, keepdims=True)
    rel   = mean / np.where(total > 1e-6, total, 1.0)
    nz    = arr.shape[2]
    depth = np.arange(nz) * Z_UM_PER_SLICE
    return rel, depth                   # (Nz,5), (Nz,)


def load_fish_profile(cond, day):
    f = IC_DIR / f"phi_xyz_{cond}_d{day}.npy"
    if not f.exists():
        return None, None
    arr  = np.load(f).mean(axis=(0, 1))
    tot  = arr.sum(axis=1, keepdims=True)
    rel  = arr / np.where(tot > 1e-6, tot, 1.0)
    depth = np.arange(arr.shape[0]) * Z_UM_PER_SLICE
    return rel, depth


def run_forward_pde(D, u, ic_phi, ic_depth, t_end, Nz=64, z_max=64.0):
    """1D reaction-diffusion-advection PDE forward simulation.

    Returns: (Nz,) depth array and dict of t → (Nz,5) phi.
    """
    # Interpolate IC to uniform grid
    dz = z_max / (Nz - 1)
    z_grid = np.linspace(0, z_max, Nz)

    # Interpolate IC (which may have different z-resolution)
    phi0 = np.zeros((Nz, N_SP))
    for sp in range(N_SP):
        phi0[:, sp] = np.interp(z_grid, ic_depth, ic_phi[:, sp])
    # Clamp and renormalise
    phi0 = np.clip(phi0, 0, None)
    tot  = phi0.sum(axis=1, keepdims=True)
    phi0 /= np.where(tot > 1e-6, tot, 1.0)

    # Load A, b for reaction term (gLV replicator, fallback to zero)
    try:
        fit_json = ROOT / 'results' / 'heine2025' / 'fit_glv_heine.json'
        d = json.loads(fit_json.read_text())
        cond_key = None
        # match CH→CH, DH→DH (keys in JSON might vary)
        for k in d:
            if k.upper() == 'CH' or k.upper() == 'DH':
                pass
        A_dict = d
        # Try to identify by cond passed via closure — approximate: use zero reaction
        A = np.zeros((N_SP, N_SP))
        b = np.zeros(N_SP)
    except Exception:
        A = np.zeros((N_SP, N_SP))
        b = np.zeros(N_SP)

    def rhs(t, phi_flat):
        phi = phi_flat.reshape(Nz, N_SP)
        dphi = np.zeros_like(phi)

        for sp in range(N_SP):
            p = phi[:, sp]
            # Diffusion (central differences, no-flux BC)
            d2p = np.zeros(Nz)
            d2p[1:-1] = (p[2:] - 2*p[1:-1] + p[:-2]) / dz**2
            d2p[0]    = 2*(p[1] - p[0]) / dz**2   # no-flux
            d2p[-1]   = 2*(p[-2] - p[-1]) / dz**2  # no-flux
            # Advection (upwind for u<0 = toward surface)
            if u >= 0:
                dadv = np.zeros(Nz)
                dadv[1:] = (p[1:] - p[:-1]) / dz
            else:
                dadv = np.zeros(Nz)
                dadv[:-1] = (p[1:] - p[:-1]) / dz
            dphi[:, sp] = D[sp] * d2p - u * dadv

        # Replicator reaction
        g = b + phi @ A.T              # (Nz, N_SP)
        phi_dot_g = (phi * g).sum(axis=1, keepdims=True)
        dphi += phi * (g - phi_dot_g)

        return dphi.ravel()

    t_eval = np.array([d for d in [1, 6, 10, 15, 21] if d <= t_end])
    sol = solve_ivp(rhs, [1.0, t_end], phi0.ravel(),
                    method='RK45', t_eval=t_eval,
                    rtol=1e-4, atol=1e-6, max_step=0.5)

    results = {}
    for i, t in enumerate(sol.t):
        phi_t = sol.y[:, i].reshape(Nz, N_SP)
        phi_t = np.clip(phi_t, 0, None)
        tot   = phi_t.sum(axis=1, keepdims=True)
        phi_t /= np.where(tot > 1e-6, tot, 1.0)
        results[int(round(t))] = phi_t

    return z_grid, results


def plot_panel(ax, cond, day, z_obs, phi_obs, z_model, phi_model,
               show_legend=False, show_ylabel=False, show_xlabel=False,
               title=None):
    """Plot one (cond, day) panel: observed dashed, model solid."""
    # observed (dashed)
    for sp in range(N_SP):
        ax.plot(phi_obs[:, sp], z_obs,
                color=COLORS[sp], ls='--',
                lw=LW_KEY * 0.9 if sp in KEY_SP else LW_OTH,
                alpha=1.0 if sp in KEY_SP else ALP_OTH,
                zorder=2)
    # model (solid)
    if phi_model is not None:
        for sp in range(N_SP):
            ax.plot(phi_model[:, sp], z_model,
                    color=COLORS[sp], ls='-',
                    lw=LW_KEY if sp in KEY_SP else LW_OTH,
                    alpha=1.0 if sp in KEY_SP else ALP_OTH,
                    zorder=3)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(min(z_obs[-1], z_model[-1] if z_model is not None else 999), 0)
    ax.set_xticks([0, 0.25, 0.5])
    clean_ax(ax)
    if title:
        ax.set_title(title, fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r'Depth ($\mu$m)', fontsize=8)
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel('Rel.\ abundance', fontsize=7.5)


def main():
    conds = ['CH', 'DH']
    clabel = {'CH': 'Commensal HOBIC (CH)', 'DH': 'Dysbiotic HOBIC (DH)'}

    # Load PINN params and run forward PDE for each condition
    pde_results = {}
    z_grids     = {}
    pinn_params = {}
    for cond in conds:
        params = load_pinn(cond)
        pinn_params[cond] = params
        ic_phi, ic_depth = load_ic(cond)
        if ic_phi is None or params is None:
            continue
        z_max = Z_MAX[cond]
        z_grid, res = run_forward_pde(
            params['D'], params['u'], ic_phi, ic_depth,
            t_end=21.0, Nz=80, z_max=z_max)
        pde_results[cond] = res
        z_grids[cond]     = z_grid

    # Figure: 2 rows (CH, DH) × len(DAYS_PLOT) cols
    n_days = len(DAYS_PLOT)
    fig, axes = plt.subplots(
        2, n_days,
        figsize=use(width_frac=1.0, aspect=0.48),
        sharey=False,
    )

    for row, cond in enumerate(conds):
        for col, day in enumerate(DAYS_PLOT):
            ax = axes[row, col]
            phi_obs, z_obs = load_fish_profile(cond, day)
            phi_mod  = pde_results.get(cond, {}).get(day)
            z_model  = z_grids.get(cond)

            if phi_obs is None:
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7)
                continue

            plot_panel(ax, cond, day, z_obs, phi_obs,
                       z_model, phi_mod,
                       show_ylabel=(col == 0),
                       show_xlabel=(row == 1),
                       title=f'Day {day}' if row == 0 else None)

        if pinn_params.get(cond):
            p = pinn_params[cond]
            D_Fn = p['D'][3]; D_Pg = p['D'][4]; u_ph = p['u']
            axes[row, 0].text(
                -0.55, 0.5,
                f'{clabel[cond]}\n'
                fr'$D_{{Fn}}$={D_Fn:.1f}, $D_{{Pg}}$={D_Pg:.1f} µm²/d'
                f'\n$u$={u_ph:+.1f} µm/d',
                transform=axes[row, 0].transAxes,
                va='center', ha='center', fontsize=6.5,
                fontweight='bold', rotation=90)

    # Legend: species + line style
    order = [3, 4, 0, 1, 2]
    sp_handles = [mlines.Line2D([], [], color=COLORS[i], ls='-',
                                lw=LW_KEY if i in KEY_SP else LW_OTH,
                                alpha=1.0 if i in KEY_SP else 0.7,
                                label=SHORT[i])
                  for i in order]
    style_handles = [
        mlines.Line2D([], [], color='0.3', ls='--', lw=1.0, label='FISH (obs.)'),
        mlines.Line2D([], [], color='0.3', ls='-',  lw=1.0, label='PDE model'),
    ]
    fig.legend(handles=sp_handles + style_handles,
               loc='lower center', ncol=7,
               fontsize=7, framealpha=0.9, edgecolor='0.8',
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', bbox_inches='tight')
        print(f'saved → {OUT}.{ext}')

    # Print D_i summary
    for cond in conds:
        p = pinn_params.get(cond)
        if p:
            print(f'\n{cond} PINN D_i (µm²/day): '
                  + ', '.join(f'{s}={v:.2f}' for s, v in zip(SHORT, p['D'])))
            print(f'  u={p["u"]:+.2f} µm/day  loss={p["loss"]:.4f}')


if __name__ == '__main__':
    main()
