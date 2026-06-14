"""Graphical abstract: the end-to-end peri-implantitis chain in one clean figure.
  1 microbial ecology (real GDI(t))  ->  2 biofilm anatomy (sulcus, on Ti)  ->  3 host inflammation
  (tipping point)  ->  4 biomechanics (resorption->stress feedback)  ->  5 per-patient prediction.
All panels reconstructed from the project data/results (Duran-Pinedo GDI, pimp_results FEM feedback, the
inflammation ODE). Run: python masterarbeit_ansys_fem/extensions/fig_graphical_abstract.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
DATA = ROOT / "data" / "prjna725874"; FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
RED, BLUE, TAN, GREY, PINK, GREEN = "#c0392b", "#1f5fa6", "#d9c7a0", "#636363", "#f2a0ad", "#1B7837"
DYS, COM = [2, 4, 6], [0, 1, 8]


def feedback():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R]); crest = np.array([r["crest_p95"] for r in R])
    return loss, crest


def cascade(B, A, NT=520, WK=156.0):
    dt = WK / NT; I = C = 0.0; L = 0.3; out = []
    Ithr, n, kC, gC, kL, lam, gI = 1.0, 8, 0.3, 0.35, 0.02, 1.6, 1.0
    for _ in range(NT):
        Cdr = I**n / (I**n + Ithr**n + 1e-12)
        I += (B - gI * I) * dt; C += (kC * Cdr * (1 + lam * (A(L) - 1)) - gC * C) * dt
        L = min(8.0, L + kL * C * dt); out.append(L)
    return np.array(out)


def main():
    use()
    fig = plt.figure(figsize=(11.0, 3.8))
    gs = fig.add_gridspec(1, 5, wspace=0.62, left=0.045, right=0.99, top=0.78, bottom=0.20)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    titles = ["1  Microbial ecology", "2  Biofilm anatomy", "3  Host inflammation",
              "4  Biomechanics", "5  Patient prediction"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=10.5, pad=5)
        a.tick_params(labelsize=8, length=2.5)
    LB = 9       # axis-label fontsize

    # 1 ecology: Duran-Pinedo GDI(t)
    phi = np.load(DATA / "phi_guild.npy"); meta = json.load(open(DATA / "metadata.json"))
    wk = np.array([int(x) for x in meta["timepoints"]])
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)
    gm = gdi.mean(1); odr = np.argsort(gm)
    for i in list(odr[:3]) + list(odr[-3:]):
        c = RED if gm[i] > np.median(gm) else GREEN
        ax[0].plot(wk, gdi[i], "-", color=c, lw=1.7, alpha=0.9)
    ax[0].axhline(0, color="0.5", lw=0.6, ls=":")
    ax[0].set_xlabel("weeks", fontsize=LB); ax[0].set_ylabel("dysbiosis GDI", fontsize=LB)

    # 2 anatomy schematic (patches)
    a = ax[1]
    a.add_patch(Rectangle((-0.6, -1), 1.2, 4.2, fc=GREY, ec="none"))          # implant/abutment
    a.add_patch(Rectangle((-3, -1), 2.4, 2.0, fc=TAN, ec="none"))             # bone L
    a.add_patch(Rectangle((0.6, -1), 2.4, 2.0, fc=TAN, ec="none"))            # bone R
    a.add_patch(Rectangle((-3, 1.0), 2.0, 2.2, fc=PINK, ec="none"))          # gingiva L
    a.add_patch(Rectangle((1.0, 1.0), 2.0, 2.2, fc=PINK, ec="none"))         # gingiva R
    for sx in (-0.85, 0.6):                                                   # biofilm strips on Ti
        a.add_patch(Rectangle((sx, -0.6), 0.25, 3.6, fc=RED, ec="none"))
    a.annotate("biofilm", xy=(0.75, 0.2), xytext=(1.6, -0.8), fontsize=8, color=RED,
               arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
    a.axhline(1.0, color="0.45", lw=0.5, ls="--"); a.text(-2.9, 1.2, "crest", fontsize=7, color="0.4")
    a.set_xlim(-3, 3); a.set_ylim(-1, 3.4); a.set_aspect("auto"); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("Ti | sulcus | gingiva", fontsize=8.5)

    # 3 inflammation tipping point
    B = np.linspace(0, 2.2, 60); thr = 1.1
    y = 7.5 * B**8 / (B**8 + thr**8)
    ax[2].plot(B, y, "-", color=RED, lw=2.4)
    ax[2].axvline(thr, color="0.5", lw=0.7, ls=":"); ax[2].axhspan(0, 0.7, color="0.88")
    ax[2].text(0.1, 0.9, "stable", fontsize=7.5, color="0.4")
    ax[2].set_xlabel("dysbiosis drive", fontsize=LB); ax[2].set_ylabel("bone loss (mm)", fontsize=LB)
    ax[2].set_title("3  Host inflammation", fontsize=10.5, pad=5)

    # 4 biomechanics feedback
    loss, crest = feedback()
    ax[3].plot(loss, crest, "o-", color=BLUE, lw=2.2, ms=5)
    ax[3].set_xlabel("bone loss (mm)", fontsize=LB); ax[3].set_ylabel(r"crestal $\sigma$ (MPa)", fontsize=LB)
    ax[3].annotate("vicious\ncycle", xy=(loss[-1], crest[-1]), xytext=(loss[0] + 0.5, crest[-1] * 0.8),
                   fontsize=8, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.6))

    # 5 per-patient prediction
    A = np.interp; lv, cv = loss, crest / crest[0]
    Af = lambda L: np.interp(L, lv, cv)
    t = np.linspace(0, 36, 520)
    Bvals = np.maximum(0.0, gm - gm.min())
    for i in list(odr[-3:]) + [odr[0]]:
        c = RED if gm[i] > np.median(gm) else GREEN
        ax[4].plot(t, cascade(Bvals[i], Af), "-", color=c, lw=1.9)
    ax[4].axhline(6.0, color="0.4", lw=0.7, ls="--"); ax[4].text(1, 6.2, "critical", fontsize=7, color="0.4")
    ax[4].set_xlabel("months", fontsize=LB); ax[4].set_ylabel("predicted loss (mm)", fontsize=LB)
    ax[4].set_ylim(0, 7)

    # arrows between panels (figure coords)
    fig.canvas.draw()
    for i in range(4):
        x0 = ax[i].get_position().x1; x1 = ax[i + 1].get_position().x0; yc = 0.45
        fig.patches.append(FancyArrowPatch((x0 + 0.004, yc), (x1 - 0.004, yc),
                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=15,
                           lw=1.6, color="0.35"))
    fig.suptitle("From the patient's microbiome to peri-implant bone-loss risk: one mechanistic chain",
                 fontsize=11.5, y=0.97, fontweight="bold")
    fig.text(0.5, 0.025, "real longitudinal ecology  $\\to$  biofilm in the sulcus  $\\to$  inflammation "
             "(tipping point)  $\\to$  mechanical feedback  $\\to$  per-patient prediction (Bayesian UQ)",
             ha="center", fontsize=8.5, color="0.3")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_graphical_abstract.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_graphical_abstract.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fem_graphical_abstract.pdf", "+ .png (300 dpi)")


if __name__ == "__main__":
    main()
