# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_o2_thiele.py — 1D steady-state oxygen profile in HOBIC biofilm.

Solves the coupled diffusion-Monod PDE for O2 at steady state:

    D_O2 d²C/dz² = μ_max · ρ · C / (K_s + C)

with:
  z=0   substrate (no-flux, dC/dz=0)
  z=δ   bulk: C = C_bulk

This is solved numerically via scipy.solve_bvp.  The dimensionless Thiele
modulus Φ² = μ_max ρ δ² / (D_O2 K_s) governs stratification depth.

Parameters are taken from published values at 37°C for oral biofilm / P. gingivalis
(Phalak et al. 2016, BMC Syst Biol):
  D_O2  = 2.68e-5  cm²/s  (aqueous at 37°C)
  C_bulk = 0.28    mg/L   (venous pO2 ≈ 40 mmHg, ~8% sat → in tissue ~0.28 mg/L)
  K_s_Pg = 0.01   mg/L   (high-affinity anaerobe)
  K_s_So = 0.5    mg/L   (aerobe, lower affinity)
  μ_max  = 0.5    h⁻¹    = 12 day⁻¹ (typical oral bacteria)
  ρ      = 10     g DW / L  (biofilm dry weight density)
  δ      = 25     µm      (HOBIC flow cell biofilm thickness)

Output: results/fig_o2_thiele.{pdf,png}

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_o2_thiele.py
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from thesis_style import use, clean_ax

ROOT = pathlib.Path(__file__).parents[2]
OUT  = ROOT / "results" / "fig_o2_thiele"

# --- Parameters (SI: m, s, kg) converted to (µm, day, mg/L) ---
D_O2   = 2.68e-5   # cm²/s  → µm²/day
D_O2  *= (1e4)**2 * 86400          # 1 cm = 1e4 µm; 1 day = 86400 s
# D_O2 ≈ 2.68e-5 * 1e8 * 86400 / 1e6 ... let me redo:
# 1 cm² = (1e4 µm)² = 1e8 µm²
# D_O2 [µm²/day] = 2.68e-5 [cm²/s] × 1e8 [µm²/cm²] × 86400 [s/day] / 1e6
# Actually: D_O2 [µm²/s] = 2.68e-5 cm²/s × (10000 µm/cm)² = 2.68e-5 × 1e8 = 2680 µm²/s
# D_O2 [µm²/day] = 2680 × 86400 = 2.32e8 µm²/day
D_O2 = 2.68e-5 * 1e8  # µm²/s
D_O2_day = D_O2 * 86400  # µm²/day ≈ 2.32e8

C_BULK = 0.28    # mg/L  (tissue O2 at venous pO2)
MU_MAX = 12.0    # day⁻¹ (0.5 h⁻¹ × 24 h/day)
RHO    = 10.0    # g DW/L  (typical biofilm density; unitless factor below)
DELTA  = 25.0    # µm  (biofilm thickness)

# K_s values (mg/L)
KS = {
    r'$P.\!g$ (anaerobe, low $K_s$)':  0.01,
    r'$S.\!o$ (aerobe, high $K_s$)':   0.50,
    r'$F.\!n$ (intermediate)':          0.10,
}
COLORS_SP = {'$P.\\!g$ (anaerobe, low $K_s$)': '#d62728',
             '$S.\\!o$ (aerobe, high $K_s$)': '#1f77b4',
             '$F.\\!n$ (intermediate)': '#9467bd'}
# For matplotlib without raw-string clash, use simpler keys below:
KS_KEYS  = list(KS.keys())
KS_VALS  = list(KS.values())
SP_COLS  = ['#d62728', '#1f77b4', '#9467bd']


def monod_rhs(z, y, Ks):
    """BVP: y[0]=C, y[1]=dC/dz. Returns [y[1], d²C/dz²]."""
    C  = np.maximum(y[0], 0.0)
    # consumption rate (mg/L/day per µm² ... scaled):
    # D d²C/dz² = μ_max ρ C/(Ks+C) * scale
    # Here scale absorbs RHO so units balance: RHO in g/L ≈ mg/g → dimensionless
    rate = MU_MAX * RHO * C / (Ks + C + 1e-12) / D_O2_day * DELTA**2
    return np.vstack([y[1], rate])


def bc(ya, yb, Ks):
    """BC: dC/dz=0 at z=0 (no-flux), C=C_bulk at z=1."""
    return np.array([ya[1], yb[0] - C_BULK])


def solve_o2(Ks, n=200):
    """Solve BVP on z ∈ [0,1] (normalised), return (z_um, C_mg_L)."""
    z  = np.linspace(0, 1, n)
    y0 = np.ones((2, n)) * C_BULK
    y0[1] = 0.0
    sol = solve_bvp(lambda z, y: monod_rhs(z, y, Ks),
                    lambda ya, yb: bc(ya, yb, Ks),
                    z, y0, tol=1e-6, max_nodes=2000)
    z_um = sol.x * DELTA        # µm from substrate
    C    = np.maximum(sol.y[0], 0.0)
    return z_um, C


def main():
    fig, axes = plt.subplots(1, 2, figsize=use(width_frac=0.85, aspect=0.55))

    # --- left: O2 profiles for each species' Ks ---
    ax = axes[0]

    for i, (label, Ks) in enumerate(zip(KS_KEYS, KS_VALS)):
        z_um, C = solve_o2(Ks)
        # Relative depletion: (C_bulk - C) / C_bulk
        depletion = (C_BULK - C) / C_BULK * 100
        ax.plot(depletion, z_um, color=SP_COLS[i], lw=1.6, label=label)

    ax.axvline(5, color='0.4', ls='--', lw=0.8)
    ax.text(5.5, DELTA * 0.05, r'5\% depletion', color='0.4', fontsize=6.5)
    ax.set_xlabel(r'O$_2$ depletion (\%)', fontsize=8)
    ax.set_ylabel(r'Depth from surface $z$ (µm)', fontsize=8)
    ax.set_title(r'\textbf{(a)} O$_2$ depletion ($\delta=25\,\mu$m)', fontsize=8, pad=3)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, DELTA)
    ax.invert_yaxis()   # surface at top
    clean_ax(ax)
    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9)

    # Annotation: all species show <5% depletion → O2 not limiting
    ax.text(0.97, 0.5,
            r'$\Phi^2 \ll 1$: O$_2$ uniform' + '\n' +
            r'$\Rightarrow$ transport, not O$_2$,' + '\n' + r'drives stratification',
            transform=ax.transAxes, ha='right', va='center',
            fontsize=6.5, color='0.3',
            bbox=dict(boxstyle='round,pad=0.3', fc='0.97', ec='0.8', lw=0.5))

    # --- right: Φ² vs biofilm thickness (which δ would cause stratification?) ---
    ax2 = axes[1]
    delta_range = np.logspace(1, 4, 300)   # µm: 10 → 10000

    for i, (label, Ks) in enumerate(zip(KS_KEYS, KS_VALS)):
        Phi2 = MU_MAX * RHO * delta_range**2 / (D_O2_day * Ks)
        ax2.loglog(delta_range, Phi2, color=SP_COLS[i], lw=1.5,
                   label=label.split('(')[0].strip())

    ax2.axhline(1.0, color='0.4', ls='--', lw=0.8)
    ax2.text(12, 1.5, r'$\Phi^2=1$ (onset of stratification)',
             fontsize=6.5, color='0.4')
    ax2.axvspan(10, 80, alpha=0.1, color='0.3', label='HOBIC range')
    ax2.text(25, ax2.get_ylim()[0] * 10 if ax2.get_ylim()[0] > 0 else 1e-5,
             'HOBIC\n25–80µm', fontsize=6.5, color='0.4', ha='center')

    ax2.set_xlabel(r'Biofilm thickness $\delta$ (µm)', fontsize=8)
    ax2.set_ylabel(r'Thiele modulus $\Phi^2$', fontsize=8)
    ax2.set_title(r'\textbf{(b)} $\Phi^2$ vs.\ thickness: O$_2$ becomes limiting at $\delta \gg 25\,\mu$m',
                  fontsize=7.5, pad=3)
    clean_ax(ax2)
    ax2.legend(fontsize=6.5, framealpha=0.9)
    ax2.set_xlim(delta_range[0], delta_range[-1])

    fig.text(0.5, -0.04,
             r'Steady-state O$_2$ diffusion--Monod model. '
             r'$D_{\mathrm{O_2}}=2.68\times10^{-5}\,\mathrm{cm^2\,s^{-1}}$ '
             r'(Phalak et al.\ 2016), '
             r'$\delta=25\,\mu\mathrm{m}$, $C_{\mathrm{bulk}}=0.28\,\mathrm{mg\,L^{-1}}$, '
             r'$\mu_{\max}=0.5\,\mathrm{h^{-1}}$, $\rho=10\,\mathrm{g\,L^{-1}}$.',
             ha='center', fontsize=6, color='0.4')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}', bbox_inches='tight')
        print(f'saved → {OUT}.{ext}')

    # Print Thiele moduli
    print('\nThiele moduli:')
    for label, Ks in zip(KS_KEYS, KS_VALS):
        phi2 = MU_MAX * RHO * DELTA**2 / (D_O2_day * Ks)
        print(f'  {label.split("(")[0].strip()}: Ks={Ks}, Φ²={phi2:.2e}')


if __name__ == '__main__':
    main()
