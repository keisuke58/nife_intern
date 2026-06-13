"""Rigorous implant-thread FEM: field + mesh convergence + thread-depth sweep + finite-strain check.

Base-bonded biofilm patch on a smooth threaded titanium section (CPE4 plane strain). Four panels:
  (A) von Mises stress field over interior thread periods -- stress concentrates at the bonded thread;
  (B) mesh convergence of the interior peak interface stress (edge corner singularity excluded);
  (C) thread-depth (curvature) sweep -- a flat substrate relaxes nearly stress-free, the thread GENERATES
      the interface stress, rising monotonically with thread depth;
  (D) small-strain (thermal eigenstrain) vs finite-strain (F=Fe.Fg neo-Hookean, umat_growth_phi.f, NLGEOM)
      -- the scale-free DH/CH and thread/flat ratios agree, so the linearisation is adequate.
Dysbiosis adds DH/CH ~2.5-2.7x on top of the geometric stress.

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
    return json.load(open(AB / ("%s_iface.json" % job)))["interior_peak_vm"]


def field(job):
    e = json.load(open(AB / ("%s_field.json" % job)))["els"]
    return (np.array([r["x"] for r in e]), np.array([r["y"] for r in e]),
            np.array([r["vm"] for r in e]))


def thread_y(x, amp):
    return amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * x))


def main():
    use()
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.55, wspace=0.62)
    axF = fig.add_subplot(gs[0, :])
    axM = fig.add_subplot(gs[1, 0]); axS = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[1, 2])

    # (A) DH field, cropped to interior periods, thread visible
    xD, yD, vD = field("implant_DH_thread")
    m = (xD >= 0.5) & (xD <= PERIODS - 0.5)
    vmax = np.quantile(vD[m], 0.99)
    tcf = axF.tricontourf(mtri.Triangulation(xD[m], yD[m]), vD[m],
                          levels=np.linspace(0, vmax, 18), cmap="inferno", extend="max")
    xx = np.linspace(0.5, PERIODS - 0.5, 300)
    axF.fill_between(xx, thread_y(xx, AMP_MAIN), -0.2, color="#9aa3ab", zorder=3)
    axF.plot(xx, thread_y(xx, AMP_MAIN), color="#5d646b", lw=0.8, zorder=4)
    axF.set_xlim(0.5, PERIODS - 0.5); axF.set_ylim(-0.12, 1.2); axF.set_aspect("equal")
    axF.set_title(r"(A) DH interface stress field (stress concentrates at the bonded thread)", fontsize=7.5)
    axF.set_xlabel(r"lateral (thread periods)"); axF.set_ylabel(r"height")
    axF.text(0.6, -0.07, "Ti", fontsize=6, color="#3d444b", zorder=5)
    cb = fig.colorbar(tcf, ax=axF, fraction=0.018, pad=0.015)
    cb.set_label(r"$\sigma_\mathrm{vM}$ (arb.)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)

    # (B) mesh convergence
    conv = [(36, "imp_conv_n36"), (72, "imp_conv_n72"), (108, "imp_conv_n108"), (144, "imp_conv_n144")]
    epp = [nx / PERIODS for nx, _ in conv]; vmc = [iface(j) for _, j in conv]
    axM.plot(epp, vmc, "o-", color=RED, lw=1.3, ms=4)
    axM.set_xlabel(r"elements / thread period"); axM.set_ylabel(r"interior peak $\sigma_\mathrm{vM}$")
    axM.set_title(r"(B) Mesh convergence", fontsize=7.5); axM.set_ylim(0, max(vmc) * 1.35)
    axM.axhspan(min(vmc[1:]), max(vmc[1:]), color=RED, alpha=0.08)
    axM.text(0.5, 0.06, r"bounded $\pm$%.0f\%%" % (100 * (max(vmc[1:]) - min(vmc[1:])) / (2 * np.mean(vmc[1:]))),
             transform=axM.transAxes, fontsize=6, color="0.4")

    # (C) thread-depth sweep
    sweep = [(0.0, "imp_amp_000"), (0.06, "imp_amp_060"), (0.12, "imp_amp_120"),
             (0.18, "imp_conv_n108"), (0.30, "imp_amp_300"), (0.45, "imp_amp_450")]
    amps = [a for a, _ in sweep]; vms = [iface(j) for _, j in sweep]
    axS.plot(amps, vms, "o-", color=RED, lw=1.4, ms=4)
    axS.scatter([0.0], [vms[0]], s=40, facecolor="white", edgecolor=BLUE, zorder=5)
    axS.annotate("flat:\nnear stress-free", xy=(0.0, vms[0]), xytext=(0.12, vms[0] + 0.30 * max(vms)),
                 fontsize=6, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    axS.set_xlabel(r"thread depth $A/h$"); axS.set_ylabel(r"interior peak $\sigma_\mathrm{vM}$")
    axS.set_title(r"(C) Thread generates the stress", fontsize=7.5)

    # (D) small-strain vs finite-strain: scale-free ratios
    lin = {"DH": iface("implant_DH_thread"), "CH": iface("implant_CH_thread"), "flat": iface("imp_amp_000")}
    fs = {"DH": iface("implantphi_DH_thread"), "CH": iface("implantphi_CH_thread"), "flat": iface("implantphi_DH_flat")}
    ratios = [("DH/CH", lin["DH"] / lin["CH"], fs["DH"] / fs["CH"]),
              ("thread/flat", lin["DH"] / lin["flat"], fs["DH"] / fs["flat"])]
    x = np.arange(len(ratios)); w = 0.36
    axD.bar(x - w / 2, [r[1] for r in ratios], w, color="0.6", label="small-strain")
    axD.bar(x + w / 2, [r[2] for r in ratios], w, color=RED, alpha=0.85, label=r"finite-strain $F{=}F_eF_g$")
    for xi, r in zip(x, ratios):
        axD.text(xi - w / 2, r[1] + 0.1, "%.1f" % r[1], ha="center", fontsize=5.6, color="0.3")
        axD.text(xi + w / 2, r[2] + 0.1, "%.1f" % r[2], ha="center", fontsize=5.6, color=RED)
    axD.set_xticks(x); axD.set_xticklabels([r[0] for r in ratios], fontsize=7)
    axD.set_ylabel(r"interior peak ratio ($\times$)")
    axD.set_title(r"(D) Linearisation check", fontsize=7.5)
    axD.legend(fontsize=5.6, loc="upper right"); axD.set_ylim(0, max(r[2] for r in ratios) * 1.35)

    fig.suptitle(r"Implant-thread FEM: the thread \emph{generates} the interface stress (flat $\approx$ "
                 r"stress-free), mesh-convergent and confirmed at finite strain; dysbiosis adds "
                 r"$\mathrm{DH/CH}\!\approx\!2.5$--$2.7\times$.", fontsize=7, y=0.99)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_thread.pdf", bbox_inches="tight")
    plt.close(fig)
    print("ratios DH/CH: lin=%.2f fs=%.2f ; thread/flat: lin=%.2f fs=%.2f"
          % (lin["DH"] / lin["CH"], fs["DH"] / fs["CH"], lin["DH"] / lin["flat"], fs["DH"] / fs["flat"]))
    print("wrote", OUT / "fem_implant_thread.pdf")


if __name__ == "__main__":
    main()
