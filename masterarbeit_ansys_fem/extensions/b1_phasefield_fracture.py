"""B1 — Phase-field interface fracture (mesh-independent replacement for the cohesive zone).

The B2 cohesive-zone delamination needs a predefined interface and is stabilisation-sensitive (its
high-phi risk branch was non-monotone, see A3). A phase-field (AT2) damage field d(x) in [0,1] regular-
ises the crack over a length scale l and is mesh-independent. Along the film-substrate interface x in
[0,L] the growth-induced interfacial energy H(x) (peaking at the free edge, where B0 showed the shear
concentrates) drives damage. Minimising the AT2 functional

  E[d] = integral [ (1-d)^2 H(x) + (Gc/2l)( d^2 + l^2 (d')^2 ) ] dx

gives, at stationarity, the linear interface equation

  -Gc l d'' + (Gc/l + 2H(x)) d = 2H(x),   d'(0)=d'(L)=0,

solved here by finite differences on TWO mesh resolutions (mesh-independence check). The dysbiotic
interface (higher growth -> higher H) develops a longer crack (d>0.5) than the commensal one — the same
dysbiosis->detachment contrast as B2, now from a mesh-independent phase-field. This is the Keio/Muramatsu
phase-field route; the production version is an Abaqus UEL/UMAT-HETVAL coupling.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use
RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

L = 4.0; Gc = 1.0; ell = 0.25

def Hdrive(x, amp):
    # growth-induced interfacial energy: concentrated at the free edge (x=L), as B0 found
    return amp * np.exp(-(L - x) / 0.6)

def solve(amp, Nx):
    x = np.linspace(0, L, Nx + 1); dx = L / Nx
    H = Hdrive(x, amp)
    # assemble  (-Gc l d'' + (Gc/l + 2H) d = 2H), Neumann ends
    A = np.zeros((Nx + 1, Nx + 1)); b = np.zeros(Nx + 1)
    for i in range(Nx + 1):
        A[i, i] += Gc / ell + 2 * H[i]; b[i] = 2 * H[i]
        if 0 < i < Nx:
            A[i, i] += 2 * Gc * ell / dx**2
            A[i, i - 1] += -Gc * ell / dx**2; A[i, i + 1] += -Gc * ell / dx**2
        elif i == 0:
            A[i, i] += 2 * Gc * ell / dx**2; A[i, i + 1] += -2 * Gc * ell / dx**2
        else:
            A[i, i] += 2 * Gc * ell / dx**2; A[i, i - 1] += -2 * Gc * ell / dx**2
    d = np.linalg.solve(A, b); return x, np.clip(d, 0, 1)

def cracklen(x, d):
    return (d > 0.5).sum() / len(d) * L

# DH (high growth -> high H) vs CH (low), each on two meshes (mesh-independence)
ampDH, ampCH = 3.0, 1.2
xDH1, dDH1 = solve(ampDH, 80); xDH2, dDH2 = solve(ampDH, 240)
xCH1, dCH1 = solve(ampCH, 80); xCH2, dCH2 = solve(ampCH, 240)
print("phase-field crack length (d>0.5):")
print("  DH:  Nx=80 -> %.3f   Nx=240 -> %.3f   (mesh-independent: diff %.1e)"
      % (cracklen(xDH1, dDH1), cracklen(xDH2, dDH2), abs(cracklen(xDH1, dDH1) - cracklen(xDH2, dDH2))))
print("  CH:  Nx=80 -> %.3f   Nx=240 -> %.3f" % (cracklen(xCH1, dCH1), cracklen(xCH2, dCH2)))
print("  DH/CH crack-length ratio = %.2f  (dysbiosis detaches further; cf. B2 cohesive 3.3x)"
      % (cracklen(xDH2, dDH2) / max(cracklen(xCH2, dCH2), 1e-6)))

fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))
a1.plot(xDH2, dDH2, "-", color=RED, lw=1.3, label="DH (dysbiotic)")
a1.plot(xCH2, dCH2, "-", color=BLUE, lw=1.3, label="CH (commensal)")
a1.axhline(0.5, ls=":", color="0.5", lw=0.8); a1.text(0.1, 0.53, "crack threshold", fontsize=6)
a1.set_xlabel(r"interface position $x$ (free edge at $L$)"); a1.set_ylabel(r"phase-field damage $d$")
a1.set_title(r"Mesh-independent interface crack", fontsize=8); a1.legend(fontsize=7, loc="upper left")
a2.plot(xDH1, dDH1, "o", ms=2, mfc="none", color="0.5", label=r"$N_x=80$")
a2.plot(xDH2, dDH2, "-", color=RED, lw=1.1, label=r"$N_x=240$")
a2.set_xlabel(r"interface position $x$"); a2.set_ylabel(r"damage $d$ (DH)")
a2.set_title(r"Mesh independence (two resolutions overlap)", fontsize=8); a2.legend(fontsize=7, loc="upper left")
fig.savefig(OUT / "F16_phasefield_fracture.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", OUT / "F16_phasefield_fracture.pdf")
