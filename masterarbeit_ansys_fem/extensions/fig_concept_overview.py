"""Whole-project CONCEPT diagram (schematic, not data) -- the logical structure of the peri-implantitis
mechano-biology model in one clean figure:

  Real 16S  --GDI(relative)-->  [Pillar 2: Biology ODE]  dysbiosis -> inflammation -> RANKL/OPG ->
  osteoclast -> marginal bone loss L  --A(L) bridge-->  [Pillar 1: Mechanics FEM]  crestal stress up
  --> feeds RANKL back  ==> VICIOUS CYCLE (tipping point / bistability)  -->  per-patient risk.

Box-and-arrow schematic; no plots, no solves.  Companion to the data-panel graphical abstract.
Bilingual: EN uses the thesis usetex style; JA switches to a Noto CJK font (usetex off).
Run: python masterarbeit_ansys_fem/extensions/fig_concept_overview.py [--lang en|ja]
"""
import sys, argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_CJK_TTC = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

# palette
BIO = "#2a7f62"; BIO_BG = "#e7f3ee"      # biology pillar (green)
MEC = "#2166ac"; MEC_BG = "#e6eef6"      # mechanics pillar (blue)
FB  = "#b2182b"                           # feedback / vicious cycle (red)
OUT_C = "#762a83"                         # output (purple)
INK = "#222222"

# Per-language strings. EN uses LaTeX (usetex); JA uses literal unicode (usetex off).
STR = {
    "en": {
        "p2": r"Pillar 2 — Biology (ODE)", "p1": r"Pillar 1 — Mechanics (FEM)",
        "inp": "Real longitudinal 16S\n(Dieckow, Duran-Pinedo)",
        "gdi": r"GDI" + "\n" + r"\small (relative dysbiosis index)",
        "rankl": "Inflammation\n" + r"RANKL/OPG $\uparrow$",
        "loss": "Osteoclast resorption\n" + r"marginal bone loss $L$",
        "fem": r"Occlusal load 100 N / $30^\circ$" + "\n" + r"\small crown moment arm, diameter",
        "stress": r"Crestal stress $\sigma \uparrow$" + "\n" + r"\small bridge $A(L)$",
        "out": r"Per-patient bone-loss risk \& ranking $+$ bistability (why hygiene alone fails)",
        "cycle": r"\textbf{VICIOUS CYCLE} $\to$ tipping point",
        "bridge": r"$A(L)$ bridge",
        "title": r"Peri-implantitis as a mechano-biological feedback loop: "
                 r"ecology $\to$ host $\to$ mechanics $\to$ ecology",
        "note": (r"GDI = guild dysbiosis index (log-ratio of dysbiotic vs.\ commensal guilds; "
                 r"relative / ordinal). \quad RANKL/OPG = osteoclast accelerator / brake" "\n"
                 r"(RANKL: receptor activator of NF-$\kappa$B ligand; OPG: osteoprotegerin, a decoy "
                 r"receptor). \quad $A(L)$ = crestal-stress amplification vs.\ bone loss $L$" "\n"
                 r"-- the single bridge FEM $\to$ ODE. \quad ODE / FEM = disease ODE model / "
                 r"finite-element model."),
    },
    "ja": {
        "p2": "柱2 — 生物 (ODE)", "p1": "柱1 — 力学 (FEM)",
        "inp": "実縦断 16S\n(Dieckow, Duran-Pinedo)",
        "gdi": "GDI\n(相対 dysbiosis 指標)",
        "rankl": "炎症\nRANKL/OPG ↑",
        "loss": "破骨細胞による吸収\n辺縁骨吸収 L",
        "fem": "咬合荷重 100 N / 30°\nクラウンモーメントアーム・径",
        "stress": "辺縁(crest)応力 σ↑\n橋 A(L)",
        "out": "患者別 骨吸収リスク・ランキング ＋ 双安定（なぜ衛生だけでは治らないか）",
        "cycle": "悪循環 → ティッピングポイント",
        "bridge": "A(L) 橋",
        "title": "ペリインプラント炎＝力学-生物フィードバックループ：生態→宿主→力学→生態",
        "note": ("GDI = ギルド dysbiosis 指標（dysbiotic/commensal ギルドの対数比；相対・順序指標）。  "
                 "RANKL/OPG = 破骨細胞のアクセル/ブレーキ\n"
                 "（RANKL: receptor activator of NF-κB ligand、OPG: おとり受容体 osteoprotegerin）。  "
                 "A(L) = 骨吸収量 L に対する辺縁応力の増幅\n"
                 "＝力学(FEM)から生物(ODE)への唯一の橋。  ODE / FEM = 疾患ODEモデル / 有限要素モデル。"),
    },
}


def text2(ax, x, y, s, fs, color, fp, weight="normal", **kw):
    """Draw text that works in both modes. `fp` is either None (EN/usetex) or the CJK .ttc PATH (JA).
    Build a FRESH FontProperties(fname=..., size=...) each call -- FontProperties.copy() drops the
    fname (falls back to serif -> garbled CJK), and passing fontweight/fontsize kwargs alongside
    fontproperties also makes matplotlib re-resolve by family name and lose the fname."""
    if fp is not None:
        ax.text(x, y, s, color=color, fontproperties=fm.FontProperties(fname=fp, size=fs), **kw)
    else:
        ax.text(x, y, s, color=color, fontsize=fs, fontweight=weight, **kw)


def box(ax, xy, w, h, text, fc, ec, tc="white", fs=7.0, bold=True, pad=0.02, fp=None):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=1.1,
                       boxstyle="round,pad=%.3f,rounding_size=0.10" % pad, zorder=3)
    ax.add_patch(p)
    text2(ax, x, y, text, fs, tc, fp, weight="bold" if bold else "normal",
          ha="center", va="center", zorder=4, linespacing=1.25)
    return (x, y)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.6, ls="-", shrink=14):
    a = FancyArrowPatch(p0, p1, connectionstyle="arc3,rad=%.3f" % rad, color=color,
                        lw=lw, ls=ls, arrowstyle="-|>", mutation_scale=14,
                        shrinkA=shrink, shrinkB=shrink, zorder=2)
    ax.add_patch(a)


def main(lang="en"):
    figsize = use(width_frac=1.0, aspect=0.62)   # applies the thesis style + returns the size
    FP = None
    if lang == "ja":                             # AFTER use(): switch off usetex, draw via a CJK font
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "dejavusans",
                             "axes.unicode_minus": False})
        FP = _CJK_TTC                            # the .ttc PATH; text2() builds a fresh FontProperties
        #                                          per call (rcParams family is unreliable for .ttc)
    T = STR[lang]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10); ax.set_ylim(-1.9, 10); ax.axis("off")

    # ---- pillar background bands ----
    ax.add_patch(plt.Rectangle((0.2, 0.4), 5.3, 8.4, fc=BIO_BG, ec="none", zorder=0))
    ax.add_patch(plt.Rectangle((5.7, 0.4), 4.1, 8.4, fc=MEC_BG, ec="none", zorder=0))
    text2(ax, 2.85, 8.55, T["p2"], 7.2, BIO, FP, weight="bold", ha="center")
    text2(ax, 7.75, 8.55, T["p1"], 7.2, MEC, FP, weight="bold", ha="center")

    # ---- input + biology cascade (left/centre) ----
    inp = box(ax, (2.85, 7.7), 4.3, 0.95, T["inp"], "white", BIO, tc=INK, fs=6.8, bold=False, fp=FP)
    gdi = box(ax, (2.85, 6.05), 3.5, 0.95, T["gdi"], BIO, BIO, fs=7.2, fp=FP)
    rankl = box(ax, (2.85, 4.25), 3.4, 1.0, T["rankl"], BIO, BIO, fs=7.0, fp=FP)
    loss = box(ax, (2.85, 2.15), 3.4, 1.0, T["loss"], FB, FB, fs=7.0, fp=FP)

    # ---- mechanics node (right) ----
    fem_in = box(ax, (7.75, 6.05), 3.6, 1.05, T["fem"], "white", MEC, tc=INK, fs=6.6, bold=False, fp=FP)
    stress = box(ax, (7.75, 4.25), 3.4, 1.0, T["stress"], MEC, MEC, fs=7.0, fp=FP)

    # ---- output ----
    out = box(ax, (5.3, 0.95), 6.6, 0.9, T["out"], OUT_C, OUT_C, fs=6.8, fp=FP)

    # ---- arrows: linear cascade ----
    arrow(ax, inp, gdi, color=BIO, lw=1.5)
    arrow(ax, gdi, rankl, color=BIO, lw=1.5)
    arrow(ax, rankl, loss, color=BIO, lw=1.7)
    arrow(ax, fem_in, stress, color=MEC, lw=1.5)

    # ---- the vicious cycle (red): loss -> stress -> RANKL (green RANKL->loss closes the triangle) ----
    arrow(ax, loss, stress, color=FB, lw=2.0, rad=-0.16)      # bone loss raises stress (A(L) bridge)
    arrow(ax, stress, rankl, color=FB, lw=2.0, rad=0.30)      # overload raises RANKL (closes loop)
    text2(ax, 5.3, 3.18, T["cycle"], 6.6, FB, FP, weight="bold", ha="center", va="center")
    text2(ax, 5.55, 2.42, T["bridge"], 6.8, FB, FP, weight="bold", ha="center", va="center")

    # ---- output arrow ----
    arrow(ax, loss, (5.3, 1.45), color=OUT_C, lw=1.6, rad=0.0)

    # ---- abbreviation note (legend) below the figure ----
    ax.plot([0.2, 9.8], [-0.45, -0.45], color="0.8", lw=0.6, zorder=1)
    text2(ax, 0.2, -0.7, T["note"], 5.0, "#555555", FP, weight="normal",
          ha="left", va="top", linespacing=1.5)

    if FP is not None:
        ax.set_title(T["title"], pad=6, fontproperties=fm.FontProperties(fname=FP, size=8.0))
    else:
        ax.set_title(T["title"], fontsize=8.0, pad=6)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    suf = "" if lang == "en" else "_ja"
    fig.savefig(OUT / ("fem_concept_overview%s.pdf" % suf), bbox_inches="tight")
    fig.savefig(OUT / ("fem_concept_overview%s.png" % suf), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", OUT / ("fem_concept_overview%s.pdf" % suf))
    print("wrote", OUT / ("fem_concept_overview%s.png" % suf))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ja"], default="en")
    main(ap.parse_args().lang)
