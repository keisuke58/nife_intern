"""A2 — Poroelastic drainage of the biofilm residual stress (Terzaghi consolidation).

The elastic residual stress (thesis) is the SHORT-time limit. The biofilm is a fluid-saturated porous
EPS, so on longer times the growth-induced pressure DRAINS and the effective residual stress relaxes —
this is the long-time, pressure-restricted limit of Bolea-Albero/Böl. We model the growth-induced
pressurisation as a uniform initial excess pore pressure p0 in a column drained at the biofilm-bulk
surface (z=H) and sealed at the substratum (z=0), governed by 1-D consolidation

    dp/dt = c_v d^2p/dz^2 ,   c_v = k * E_oed / gamma_w  (consolidation coefficient),

drained BC p(H,t)=0, sealed BC dp/dz(0,t)=0, initial p(z,0)=p0. We solve it by finite differences AND
compare to the closed-form Terzaghi series (verification), then report the drainage timescale tau and
the residual-stress relaxation curve. The effective interface stress that drives delamination relaxes
as the average excess pressure dissipates: sigma_eff(t)/sigma_eff(0) = 1 - U(t), U = degree of
consolidation. (This is the Abaqus *SOILS C3D8P deck `abaqus/gen_poro_inp.py` done analytically;
the FE deck is a runnable scaffold that needs poroelastic material-card tuning.)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use
RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

# biofilm poroelastic parameters (order of magnitude)
H = 1.0                      # film thickness (normalised)
k = 1.0e-4                   # permeability  [length/time]
E_oed = 5000.0               # oedometric (drained) modulus [Pa]
gamma_w = 10.0               # specific weight of pore fluid
cv = k * E_oed / gamma_w     # consolidation coefficient
tau = H**2 / cv              # Terzaghi drainage timescale
print("consolidation coefficient c_v = %.4g,  drainage timescale tau = H^2/c_v = %.2f" % (cv, tau))

# --- finite-difference solver (single-drainage: drained top, sealed base) ---
Nz = 80; dz = H / Nz; z = np.linspace(0, H, Nz + 1)
p = np.ones(Nz + 1); p[-1] = 0.0                  # p0 = 1, drained top
dt = 0.4 * dz**2 / cv
T = 2.0 * tau; nt = int(T / dt)
times = np.linspace(0, T, 60); snap = {}; Uavg = []
ti = 0
p0_mass = (dz*(np.sum(np.ones(Nz + 1))-0.5*(np.ones(Nz + 1)[0]+np.ones(Nz + 1)[-1])))
for n in range(nt + 1):
    t = n * dt
    while ti < len(times) and t >= times[ti]:
        snap[times[ti]] = p.copy()
        Uavg.append(1.0 - (dz*(np.sum(p)-0.5*(p[0]+p[-1]))) / p0_mass)   # degree of consolidation
        ti += 1
    pn = p.copy()
    pn[1:-1] = p[1:-1] + cv * dt / dz**2 * (p[2:] - 2 * p[1:-1] + p[:-2])
    pn[0] = pn[1]            # sealed base (no-flux): mirror
    pn[-1] = 0.0            # drained top
    p = pn
Uavg = np.array(Uavg); tt = times[:len(Uavg)]

# --- analytic Terzaghi series (verification) ---
def U_analytic(t):
    Tv = cv * t / H**2
    s = 0.0
    for m in range(0, 200):
        M = np.pi * (2 * m + 1) / 2
        s += 2.0 / M**2 * np.exp(-M**2 * Tv)
    return 1.0 - s
Ua = np.array([U_analytic(t) for t in tt])
rms = np.sqrt(np.nanmean((Uavg[1:] - Ua[1:])**2))
print("FD vs analytic Terzaghi degree-of-consolidation:  RMS = %.2e  (verification)" % rms)
t50 = tau * 0.197           # Terzaghi t for U=50%
print("residual stress half-relaxed (U=50%%) at t50 = 0.197*tau = %.2f; ~95%% relaxed by t~tau." % t50)

fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.5))
for tf, col in [(0.05 * tau, "0.7"), (0.2 * tau, "0.5"), (0.5 * tau, BLUE), (tau, RED)]:
    key = min(snap, key=lambda s: abs(s - tf))
    a1.plot(snap[key], z, "-", color=col, lw=1.1, label=r"$t=%.2f\,\tau$" % (key / tau))
a1.set_xlabel(r"excess pore pressure $p/p_0$"); a1.set_ylabel(r"depth $z$ (0 = substratum)")
a1.set_title(r"Drainage of growth pressure", fontsize=8); a1.legend(fontsize=6.5, loc="lower right")
a2.plot(tt / tau, 1 - Uavg, "-", color=RED, lw=1.3, label="FD solver")
a2.plot(tt / tau, 1 - Ua, "o", ms=2.5, mfc="none", color="0.3", label="Terzaghi analytic")
a2.axvline(0.197, ls=":", color="0.5", lw=0.8); a2.text(0.21, 0.6, r"$t_{50}$", fontsize=7)
a2.set_xlabel(r"time $t/\tau$"); a2.set_ylabel(r"residual stress fraction $1-U(t)$")
a2.set_title(r"Stress relaxes by drainage (RMS %.0e)" % rms, fontsize=8); a2.legend(fontsize=7)
fig.suptitle(r"Poroelastic A2: elastic (thesis, $t\!\to\!0$) $\to$ drained (B\"ol, $t\!\to\!\infty$)", fontsize=8, y=1.0)
fig.savefig(OUT / "F15_poroelastic_terzaghi.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "F15_poroelastic_terzaghi.pdf")
