"""Polished publication-grade graphical abstract of the end-to-end peri-implantitis chain.
Same five stages as fig_graphical_abstract, refined: numbered stage badges, soft per-stage background
cards, de-spined minimal axes, a threaded-implant anatomy schematic, unified palette and typography.
Run: python masterarbeit_ansys_fem/extensions/fig_graphical_abstract_polished.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Polygon

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
DATA = ROOT / "data" / "prjna725874"; FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
RED, BLUE, TAN, GREY, PINK, GREEN = "#c0392b", "#1f5fa6", "#d9c7a0", "#5d6066", "#f1a7b4", "#2e8b57"
INK = "#222428"
STAGE_COL = ["#2e6fb0", "#c0556a", "#c0392b", "#1f5fa6", "#3a6b4f"]
DYS, COM = [2, 4, 6], [0, 1, 8]


def feedback():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    return (np.array([10.0 - r["z_bone_top"] for r in R]),
            np.array([r["crest_p95"] for r in R]))


def cascade(B, A, NT=520, WK=156.0):
    dt = WK / NT; I = C = 0.0; L = 0.3; out = []
    Ithr, n, kC, gC, kL, lam, gI = 1.0, 8, 0.3, 0.35, 0.02, 1.6, 1.0
    for _ in range(NT):
        Cdr = I**n / (I**n + Ithr**n + 1e-12)
        I += (B - gI * I) * dt; C += (kC * Cdr * (1 + lam * (A(L) - 1)) - gC * C) * dt
        L = min(8.0, L + kL * C * dt); out.append(L)
    return np.array(out)


def style(a):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color("0.6"); a.spines[sp].set_linewidth(0.8)
    a.tick_params(labelsize=8, length=2.5, color="0.6")


def badge(fig, axx, num, col, label):
    p = axx.get_position()
    fig.text(p.x0 - 0.004, p.y1 + 0.055, str(num), fontsize=11, color="white", fontweight="bold",
             ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.34", fc=col, ec="none"))
    fig.text(p.x0 + 0.028, p.y1 + 0.055, label, fontsize=10.5, color=INK, fontweight="bold",
             ha="left", va="center")


def main():
    use()
    fig = plt.figure(figsize=(12.0, 4.3))
    gs = fig.add_gridspec(1, 5, wspace=0.6, left=0.05, right=0.99, top=0.70, bottom=0.21)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    for a in ax:
        style(a)
    LB = 9.5

    # soft background card behind each panel
    fig.canvas.draw()
    for a, col in zip(ax, STAGE_COL):
        p = a.get_position()
        fig.patches.append(FancyBboxPatch((p.x0 - 0.018, p.y0 - 0.11), p.width + 0.036, p.height + 0.20,
                           boxstyle="round,pad=0.006,rounding_size=0.02", transform=fig.transFigure,
                           fc=col + "12", ec=col + "55", lw=0.8, zorder=-5, mutation_aspect=0.4))

    # 1 ecology
    phi = np.load(DATA / "phi_guild.npy"); meta = json.load(open(DATA / "metadata.json"))
    wk = np.array([int(x) for x in meta["timepoints"]])
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)
    gm = gdi.mean(1); odr = np.argsort(gm)
    for i in list(odr[:3]) + list(odr[-3:]):
        c = RED if gm[i] > np.median(gm) else GREEN
        ax[0].plot(wk, gdi[i], "-", color=c, lw=1.8, alpha=0.9, solid_capstyle="round")
    ax[0].axhline(0, color="0.55", lw=0.7, ls=":")
    ax[0].set_xlabel("time (weeks)", fontsize=LB); ax[0].set_ylabel("dysbiosis index", fontsize=LB)
    ax[0].set_xticks([0, 6, 12])

    # 2 anatomy: threaded implant + gingiva + biofilm
    a = ax[1]
    a.add_patch(Rectangle((-3.2, -1), 2.55, 2.0, fc=TAN, ec="none"))
    a.add_patch(Rectangle((0.65, -1), 2.55, 2.0, fc=TAN, ec="none"))
    a.add_patch(FancyBboxPatch((-3.0, 1.02), 2.2, 2.1, boxstyle="round,pad=0,rounding_size=0.25",
                fc=PINK, ec="none"))
    a.add_patch(FancyBboxPatch((0.8, 1.02), 2.2, 2.1, boxstyle="round,pad=0,rounding_size=0.25",
                fc=PINK, ec="none"))
    # implant body + thread notches
    a.add_patch(FancyBboxPatch((-0.5, -1), 1.0, 4.15, boxstyle="round,pad=0,rounding_size=0.12",
                fc=GREY, ec="none", zorder=3))
    for zt in np.arange(-0.7, 1.0, 0.42):                       # threads in the bone region
        a.add_patch(Polygon([(-0.5, zt), (-0.85, zt + 0.16), (-0.5, zt + 0.32)], fc=GREY, ec="none", zorder=3))
        a.add_patch(Polygon([(0.5, zt), (0.85, zt + 0.16), (0.5, zt + 0.32)], fc=GREY, ec="none", zorder=3))
    for sx in (-0.92, 0.66):                                    # biofilm on the Ti flanks, descending
        a.add_patch(Rectangle((sx, -0.55), 0.26, 3.55, fc=RED, ec="none", zorder=4))
    a.axhline(1.0, color="0.5", lw=0.7, ls="--")
    a.text(-2.95, 1.18, "crest", fontsize=7, color="0.4", style="italic")
    a.text(2.0, 2.0, "gingiva", fontsize=7.5, color="#a23a52", ha="center", style="italic")
    a.text(0.0, -0.4, "Ti", fontsize=7.5, color="white", ha="center", va="center", fontweight="bold", zorder=5)
    a.annotate("biofilm", xy=(0.78, -0.1), xytext=(1.7, -0.85), fontsize=8, color=RED,
               arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    a.set_xlim(-3.2, 3.2); a.set_ylim(-1, 3.3)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_xticks([]); a.set_yticks([])

    # 3 inflammation tipping point
    B = np.linspace(0, 2.2, 80); thr = 1.1
    y = 7.5 * B**8 / (B**8 + thr**8)
    ax[2].fill_between([0, 2.2], 0, 0.7, color="0.9", zorder=0)
    ax[2].plot(B, y, "-", color=RED, lw=2.6, solid_capstyle="round")
    ax[2].axvline(thr, color="0.55", lw=0.8, ls=":")
    ax[2].text(0.12, 0.95, "stable", fontsize=8, color="0.45", style="italic")
    ax[2].text(thr + 0.05, 6.6, "threshold", fontsize=7.5, color="0.4", rotation=90, va="top")
    ax[2].set_xlabel("dysbiosis drive", fontsize=LB); ax[2].set_ylabel("bone loss (mm)", fontsize=LB)
    ax[2].set_xticks([0, 1, 2])

    # 4 biomechanics feedback
    loss, crest = feedback()
    ax[3].plot(loss, crest, "o-", color=BLUE, lw=2.4, ms=6, solid_capstyle="round")
    ax[3].set_xlabel("bone loss (mm)", fontsize=LB); ax[3].set_ylabel(r"crestal stress (MPa)", fontsize=LB)
    ax[3].annotate("vicious\ncycle", xy=(loss[-1], crest[-1]), xytext=(loss[0] + 0.4, crest[-1] * 0.78),
                   fontsize=8.5, color=BLUE, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))

    # 5 prediction
    lv, cv = loss, crest / crest[0]; Af = lambda L: np.interp(L, lv, cv)
    t = np.linspace(0, 36, 520); Bv = np.maximum(0.0, gm - gm.min())
    for i in list(odr[-3:]) + [odr[0]]:
        c = RED if gm[i] > np.median(gm) else GREEN
        ax[4].plot(t, cascade(Bv[i], Af), "-", color=c, lw=2.0, solid_capstyle="round")
    ax[4].axhline(6.0, color="0.45", lw=0.8, ls="--")
    ax[4].text(1, 6.25, "critical", fontsize=7.5, color="0.4", style="italic")
    ax[4].set_xlabel("time (months)", fontsize=LB); ax[4].set_ylabel("predicted loss (mm)", fontsize=LB)
    ax[4].set_ylim(0, 7); ax[4].set_xticks([0, 12, 24, 36])

    # badges + arrows
    labels = ["Microbial ecology", "Biofilm anatomy", "Host inflammation", "Biomechanics", "Patient prediction"]
    for i, (a, col, lab) in enumerate(zip(ax, STAGE_COL, labels)):
        badge(fig, a, i + 1, col, lab)
    for i in range(4):
        x0 = ax[i].get_position().x1; x1 = ax[i + 1].get_position().x0
        fig.patches.append(FancyArrowPatch((x0 + 0.006, 0.45), (x1 - 0.006, 0.45),
                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18,
                           lw=2.0, color="0.45"))
    fig.suptitle("From the patient's microbiome to peri-implant bone-loss risk",
                 fontsize=13, y=0.965, fontweight="bold", color=INK)
    fig.text(0.5, 0.04, "real longitudinal ecology  →  biofilm in the sulcus  →  inflammatory tipping point"
             "  →  mechanical feedback  →  per-patient prediction (Bayesian UQ)",
             ha="center", fontsize=9, color="0.35")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_graphical_abstract_polished.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_graphical_abstract_polished.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote fem_graphical_abstract_polished.pdf + .png")


if __name__ == "__main__":
    main()
