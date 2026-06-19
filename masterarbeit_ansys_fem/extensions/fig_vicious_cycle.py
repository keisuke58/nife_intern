"""Schematic of the peri-implantitis mechano-biological VICIOUS CYCLE (EN + JA).

Visualises the closed feedback loop implemented in fig_periimplantitis_rankl_opg.py: dysbiosis ignites
inflammation, which (via RANKL/OPG) drives osteoclastic bone loss; the loss raises the crestal stress
amplification A(L) (FEM), pushing the crest into Frost's pathological-overload window, which feeds back
into RANKL — a self-reinforcing loop with an emergent tipping point. A homeostatic TGF-beta OB-coupling
counter-loop normally balances it and fails above the tipping point.

Writes both languages:  fig_vicious_cycle_en.{pdf,png}  and  fig_vicious_cycle_ja.{pdf,png}
Run:  python fig_vicious_cycle.py            # both
      python fig_vicious_cycle.py en|ja      # one
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

RED = "#a93226"; RED_SOFT = "#f9ebea"; GREEN = "#1e8449"; GREEN_SOFT = "#eafaf1"
INK = "#1c2833"; TRIG = "#6c3483"; TRIG_SOFT = "#f4ecf7"; TIP_SOFT = "#fef9e7"

TEXT = {
    "en": {
        "font": "Nimbus Roman",
        "title": "Peri-implantitis mechano-biological vicious cycle",
        "trigger": "Dysbiosis\n(biofilm, GDI ↑)",
        "nodes": [
            "Inflammation\nRANKL ↑ / OPG ↓",
            "Osteoclastic\nbone resorption  (L ↑)",
            "Crestal stress\namplification  A(L) ↑   [FEM]",
            "Mechanical overload\n(crest > 3000 µε, Frost)",
        ],
        "center": "VICIOUS\nCYCLE",
        "tipping": "tipping point\n$b_{\\mathrm{crit}}\\approx0.44$,  RANKL:OPG $\\approx2.1$\n12/15 patients $\\rightarrow$ > 2 mm loss",
        "counter": "homeostasis (TGF-β: OC→OB)\nfails above the tipping point",
        "edge": ["RANKL/OPG ↑", "bone loss  $L\\uparrow$", "crest stress ↑", "pathological\nmechanostat"],
    },
    "ja": {
        "font": "Noto Serif CJK JP",
        "title": "ペリインプラント炎の力学–生物悪循環",
        "trigger": "ディスバイオシス\n(バイオフィルム, GDI ↑)",
        "nodes": [
            "炎症\nRANKL ↑ / OPG ↓",
            "破骨細胞による\n骨吸収  (L ↑)",
            "歯頸部応力の\n増幅  A(L) ↑  ［FEM］",
            "力学的過負荷\n(歯頸部 > 3000 µε, Frost)",
        ],
        "center": "悪循環",
        "tipping": "ティッピングポイント\n$b_{\\mathrm{crit}}\\approx0.44$,  RANKL:OPG $\\approx2.1$\n15例中12例 → >2 mm 吸収",
        "counter": "恒常性 (TGF-β: OC→OB)\n閾値を超えると破綻",
        "edge": ["RANKL/OPG ↑", "骨損失 $L\\uparrow$", "歯頸部応力 ↑", "病理的\nメカノスタット"],
    },
}


def make(lang: str) -> None:
    import matplotlib.patheffects as pe
    t = TEXT[lang]
    plt.rcParams["font.family"] = t["font"]
    plt.rcParams["mathtext.fontset"] = "stix"          # serif math to match Times / Mincho
    plt.rcParams["axes.unicode_minus"] = False
    shadow = [pe.withSimplePatchShadow(offset=(1.4, -1.4), alpha=0.18)]

    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    ax.set_xlim(-1.95, 1.95); ax.set_ylim(-2.0, 1.95); ax.axis("off")
    ax.set_aspect("equal")

    # faint circular guide suggesting the cycle
    ring = plt.Circle((0, 0), 1.10, fill=False, ec=RED, lw=1.0, alpha=0.18, ls=(0, (6, 5)), zorder=0)
    ax.add_patch(ring)

    # four ring nodes at top, right, bottom, left
    ang = np.deg2rad([90, 0, -90, 180])
    R = 1.10
    pos = np.c_[R * np.cos(ang), R * np.sin(ang)]
    BW, BH = 1.02, 0.46
    for (x, y), label in zip(pos, t["nodes"]):
        box = FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                             boxstyle="round,pad=0.02,rounding_size=0.11",
                             fc="white", ec=RED, lw=1.6, zorder=3)
        box.set_path_effects(shadow)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color=INK, zorder=4,
                linespacing=1.35)

    # clockwise reinforcing arrows between consecutive ring nodes
    for i in range(4):
        p0, p1 = pos[i], pos[(i + 1) % 4]
        a = FancyArrowPatch(_edge(p0), _edge(p1, incoming=True),
                            connectionstyle="arc3,rad=0.30", arrowstyle="-|>",
                            mutation_scale=26, lw=3.0, color=RED, zorder=2,
                            capstyle="round", joinstyle="round")
        ax.add_patch(a)
        mid = 0.5 * (p0 + p1) * 2.02
        ax.text(mid[0], mid[1], t["edge"][i], ha="center", va="center", fontsize=10,
                color=RED, style="italic", zorder=4, linespacing=1.25)

    # external trigger feeding the inflammation node (top)
    tx, ty = -1.55, 1.62
    trig = FancyBboxPatch((tx - 0.46, ty - 0.21), 0.92, 0.42,
                          boxstyle="round,pad=0.02,rounding_size=0.11", fc=TRIG_SOFT, ec=TRIG,
                          lw=1.5, zorder=3)
    trig.set_path_effects(shadow)
    ax.add_patch(trig)
    ax.text(tx, ty, t["trigger"], ha="center", va="center", fontsize=10.5, color=TRIG, zorder=4,
            linespacing=1.35)
    ax.add_patch(FancyArrowPatch((tx + 0.42, ty - 0.16), (pos[0][0] - 0.46, pos[0][1] + 0.20),
                 connectionstyle="arc3,rad=-0.18", arrowstyle="-|>", mutation_scale=20,
                 lw=2.2, color=TRIG, zorder=2, capstyle="round"))

    # centre label + tipping annotation
    ax.text(0, 0.30, t["center"], ha="center", va="center", fontsize=27, fontweight="bold",
            color=RED, alpha=0.92, zorder=4, linespacing=1.0)
    ax.text(0, -0.34, t["tipping"], ha="center", va="center", fontsize=9.5, color=INK, zorder=4,
            linespacing=1.4, bbox=dict(boxstyle="round,pad=0.45", fc=TIP_SOFT, ec="#caa94a", lw=0.8))

    # homeostatic counter-loop note (green)
    ax.text(0, -1.78, t["counter"], ha="center", va="center", fontsize=9.5, color=GREEN, zorder=4,
            linespacing=1.35, bbox=dict(boxstyle="round,pad=0.35", fc=GREEN_SOFT, ec=GREEN, lw=1.0))

    ax.set_title(t["title"], fontsize=14.5, color=INK, pad=14)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_vicious_cycle_{lang}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_vicious_cycle_{lang}.pdf / .png")


def _edge(p, incoming=False):
    """Offset from a node centre so the curved arrow leaves/enters at the box edge cleanly."""
    v = np.array(p, float)
    n = v / (np.linalg.norm(v) + 1e-9)
    tang = np.array([-n[1], n[0]]) * (-0.30 if not incoming else 0.30)
    return tuple(v + n * 0.10 + tang)


if __name__ == "__main__":
    langs = [sys.argv[1]] if len(sys.argv) > 1 else ["en", "ja"]
    for lg in langs:
        make(lg)
