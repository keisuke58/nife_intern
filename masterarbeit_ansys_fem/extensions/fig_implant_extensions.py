"""Implant-thread FEM extensions (Outlook): two-scale micro-thread, interface delamination, relaxation.

Three further implant-thread results built on the eigenstrain/VE Abaqus decks:
  (A) two-scale roughness -- a micro-thread superimposed on the thread concentrates the interface stress
      further (interior peak vM ~1030 vs ~610 for the plain thread);
  (B) interface delamination -- an AT2 phase-field damage field driven by the REAL thread interface
      stress profile (H(x) ~ sigma_vM(x)^2 from the FEM) cracks the dysbiotic interface further than the
      commensal (mesh-independent; the Keio/Muramatsu phase-field route, cf. b1);
  (C) viscoelastic relaxation -- with the biofilm Prony series the thread interface von Mises stress
      relaxes to ~10% of its instantaneous value over t >> tau (the elastic result is the short-time limit).

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_extensions.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

AB = ROOT / "masterarbeit_ansys_fem" / "coupling_prototype" / "abaqus"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
RED, BLUE = "#c0392b", "#1f6fb4"
PERIODS = 3

# ---- AT2 phase-field interface crack driven by data H(x) (cf. b1) ----
Gc, ELL = 1.0, 0.25


def iface_profile(job):
    rows = sorted(json.load(open(AB / ("%s_iface.json" % job)))["rows"], key=lambda r: r["x"])
    rows = [r for r in rows if 0.5 <= r["x"] <= PERIODS - 0.5]   # interior (exclude edge singularity)
    x = np.array([r["x"] for r in rows]); x = x - x.min()
    return x, np.array([r["vm"] for r in rows])


def solve_pf(xg, H, Nx):
    L = xg.max()
    x = np.linspace(0, L, Nx + 1); dx = L / Nx
    Hi = np.interp(x, xg, H)
    A = np.zeros((Nx + 1, Nx + 1)); b = np.zeros(Nx + 1)
    for i in range(Nx + 1):
        A[i, i] += Gc / ELL + 2 * Hi[i]; b[i] = 2 * Hi[i]
        if 0 < i < Nx:
            A[i, i] += 2 * Gc * ELL / dx**2; A[i, i-1] += -Gc*ELL/dx**2; A[i, i+1] += -Gc*ELL/dx**2
        elif i == 0:
            A[i, i] += 2*Gc*ELL/dx**2; A[i, i+1] += -2*Gc*ELL/dx**2
        else:
            A[i, i] += 2*Gc*ELL/dx**2; A[i, i-1] += -2*Gc*ELL/dx**2
    return x, np.clip(np.linalg.solve(A, b), 0, 1)


def main():
    use()
    fig, (aA, aB, aC) = plt.subplots(1, 3, figsize=use(width_frac=0.96, aspect=0.34),
                                     gridspec_kw={"wspace": 0.55})

    # (A) micro-thread field
    e = json.load(open(AB / "implant_DH_micro_field.json"))["els"]
    x = np.array([r["x"] for r in e]); y = np.array([r["y"] for r in e]); vm = np.array([r["vm"] for r in e])
    m = (x >= 0.5) & (x <= PERIODS - 0.5)
    aA.tricontourf(mtri.Triangulation(x[m], y[m]), vm[m],
                   levels=np.linspace(0, np.quantile(vm[m], 0.99), 16), cmap="inferno", extend="max")
    aA.set_aspect("equal"); aA.set_xlim(0.5, PERIODS - 0.5); aA.set_ylim(-0.05, 1.2)
    aA.set_title(r"(A) Two-scale (micro-)thread", fontsize=7.5)
    aA.set_xlabel(r"lateral"); aA.set_ylabel(r"height")

    # (B) phase-field delamination driven by real thread interface stress
    xgD, vmD = iface_profile("implant_DH_thread"); xgC, vmC = iface_profile("implant_CH_thread")
    alpha = 8.0 / (vmD.max() ** 2)                  # scale so DH interior peak energy H ~ 8 (threshold 2)
    HD, HC = alpha * vmD**2, alpha * vmC**2
    xD, dD = solve_pf(xgD, HD, 240); xC, dC = solve_pf(xgC, HC, 240)
    dext = lambda d: float(np.mean(d))              # integrated damage (delamination extent), finite
    aB.plot(xD, dD, "-", color=RED, lw=1.4, label="DH (dysbiotic)")
    aB.plot(xC, dC, "-", color=BLUE, lw=1.4, label="CH (commensal)")
    aB.axhline(0.5, ls=":", color="0.5", lw=0.8)
    aB.set_xlabel(r"thread interface position"); aB.set_ylabel(r"phase-field damage $d$")
    aB.set_title(r"(B) Interface delamination", fontsize=7.5); aB.legend(fontsize=6, loc="upper right")
    aB.set_ylim(0, 1.05)

    # (C) viscoelastic relaxation (instant vs relaxed)
    ve = json.load(open(AB / "ve_DH_ve.json"))
    aC.bar([0, 1], [1.0, ve["ratio"]], width=0.55, color=[RED, "0.6"])
    aC.set_xticks([0, 1]); aC.set_xticklabels(["instant\n($t\\!\\ll\\!\\tau$)", "relaxed\n($t\\!\\gg\\!\\tau$)"], fontsize=6.5)
    aC.set_ylabel(r"thread interface $\sigma_\mathrm{vM}$ (norm.)")
    aC.text(1, ve["ratio"] + 0.04, r"%.0f\%%" % (100 * ve["ratio"]), ha="center", fontsize=7)
    aC.set_title(r"(C) Viscoelastic relaxation", fontsize=7.5); aC.set_ylim(0, 1.15)

    fig.suptitle(r"Implant-thread extensions: micro-thread concentrates stress; the real interface "
                 r"stress damages the dysbiotic interface $%.1f\times$ more than the commensal; VE relaxes "
                 r"the stress to $\sim$%.0f\%%." % (dext(dD) / max(dext(dC), 1e-6), 100 * ve["ratio"]),
                 fontsize=6.6, y=1.02)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_extensions.pdf", bbox_inches="tight")
    plt.close(fig)
    print("(B) integrated damage DH=%.3f CH=%.3f ratio=%.2f" % (dext(dD), dext(dC), dext(dD)/max(dext(dC), 1e-6)))
    print("(C) VE relaxed/instant=%.2f" % ve["ratio"])
    print("wrote", OUT / "fem_implant_extensions.pdf")


if __name__ == "__main__":
    main()
