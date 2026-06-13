"""Rigorous implant-thread FEM: stress field + mesh convergence + thread-depth (curvature) sweep.

Reads the eigenstrain-deck odb extracts (coupling_prototype/abaqus/*_iface.json, *_field.json) and shows,
for the base-bonded biofilm patch on a smooth threaded titanium section (isotropic volume-matched growth,
CPE4 plane strain):
  (A) the von Mises stress field over interior thread periods -- stress concentrates at the bonded thread;
  (B) mesh convergence of the interior peak interface stress (edge corner singularity excluded);
  (C) thread-depth (curvature) sweep -- a flat substrate relaxes nearly stress-free, the thread GENERATES
      the interface stress, rising monotonically with thread depth. Implant geometry is the primary route
      by which laterally-relaxed biofilm growth loads the interface; the dysbiosis contrast (DH/CH ~2.5x)
      is preserved on top of it.

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_fem.py
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
AMP_MAIN, PERIODS = 0.18, 3


def iface(job):
    return json.load(open(AB / ("%s_iface.json" % job)))


def field(job):
    e = json.load(open(AB / ("%s_field.json" % job)))["els"]
    return (np.array([r["x"] for r in e]), np.array([r["y"] for r in e]),
            np.array([r["vm"] for r in e]))


def thread_y(x, amp):
    return amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * x))


def main():
    use()
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.68)
    axF = fig.add_subplot(gs[0]); axM = fig.add_subplot(gs[1]); axS = fig.add_subplot(gs[2])

    # (A) DH field, cropped to interior periods (exclude edge singularity), thread visible
    xD, yD, vD = field("implant_DH_thread")
    m = (xD >= 0.5) & (xD <= PERIODS - 0.5)
    vmax = np.quantile(vD[m], 0.99)
    tri = mtri.Triangulation(xD[m], yD[m])
    tcf = axF.tricontourf(tri, vD[m], levels=np.linspace(0, vmax, 18), cmap="inferno", extend="max")
    xx = np.linspace(0.5, PERIODS - 0.5, 300)
    axF.fill_between(xx, thread_y(xx, AMP_MAIN), -0.2, color="#9aa3ab", zorder=3)
    axF.plot(xx, thread_y(xx, AMP_MAIN), color="#5d646b", lw=0.8, zorder=4)
    axF.set_xlim(0.5, PERIODS - 0.5); axF.set_ylim(-0.12, 1.2); axF.set_aspect("equal")
    axF.set_title(r"DH interface stress field", fontsize=8)
    axF.set_xlabel(r"lateral (thread periods)"); axF.set_ylabel(r"height")
    axF.text(0.6, -0.07, "Ti", fontsize=6, color="#3d444b", zorder=5)
    cb = fig.colorbar(tcf, ax=axF, fraction=0.04, pad=0.02)
    cb.set_label(r"$\sigma_\mathrm{vM}$ (arb.)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)

    # (B) mesh convergence: interior peak vM vs elements per thread period
    conv = [(36, "imp_conv_n36"), (72, "imp_conv_n72"), (108, "imp_conv_n108"), (144, "imp_conv_n144")]
    epp = [nx / PERIODS for nx, _ in conv]
    vmc = [iface(j)["interior_peak_vm"] for _, j in conv]
    axM.plot(epp, vmc, "o-", color=RED, lw=1.3, ms=4)
    axM.set_xlabel(r"elements / thread period"); axM.set_ylabel(r"interior peak $\sigma_\mathrm{vM}$")
    axM.set_title(r"Mesh convergence", fontsize=8)
    axM.set_ylim(0, max(vmc) * 1.3)
    axM.axhspan(min(vmc[1:]), max(vmc[1:]), color=RED, alpha=0.08)
    axM.text(0.5, 0.06, r"bounded $\pm$%.0f\%%" % (100 * (max(vmc[1:]) - min(vmc[1:])) / (2 * np.mean(vmc[1:]))),
             transform=axM.transAxes, fontsize=6, color="0.4")

    # (C) thread-depth (curvature) sweep: interior peak vM vs thread amplitude
    sweep = [(0.0, "imp_amp_000"), (0.06, "imp_amp_060"), (0.12, "imp_amp_120"),
             (0.18, "imp_conv_n108"), (0.30, "imp_amp_300"), (0.45, "imp_amp_450")]
    amps = [a for a, _ in sweep]
    vms = [iface(j)["interior_peak_vm"] for _, j in sweep]
    axS.plot(amps, vms, "o-", color=RED, lw=1.4, ms=4)
    axS.scatter([0.0], [vms[0]], s=40, facecolor="white", edgecolor=BLUE, zorder=5)
    axS.annotate("flat:\nnear stress-free", xy=(0.0, vms[0]), xytext=(0.13, vms[0] + 0.30 * max(vms)),
                 fontsize=6, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    axS.set_xlabel(r"thread depth $A/h$"); axS.set_ylabel(r"interior peak $\sigma_\mathrm{vM}$")
    axS.set_title(r"Thread generates the stress", fontsize=8)

    fig.suptitle(r"Implant-thread FEM (base-bonded patch): a flat surface relaxes nearly stress-free; the "
                 r"thread \emph{generates} the interface stress, rising with thread depth "
                 r"(mesh-convergent). Dysbiosis adds $\mathrm{DH/CH}\!\approx\!2.5\times$.",
                 fontsize=6.8, y=1.04)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_thread.pdf", bbox_inches="tight")
    plt.close(fig)
    dh, ch = iface("implant_DH_thread")["interior_peak_vm"], iface("implant_CH_thread")["interior_peak_vm"]
    print("interior peak vM: DH=%.0f CH=%.0f (ratio %.2f); flat=%.0f -> thread@0.18=%.0f (%.1fx)"
          % (dh, ch, dh / ch, vms[0], vms[3], vms[3] / vms[0]))
    print("wrote", OUT / "fem_implant_thread.pdf")


if __name__ == "__main__":
    main()
