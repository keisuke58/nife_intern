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

RED = "#c0392b"; GREEN = "#27ae60"; INK = "#2c3e50"; TRIG = "#8e44ad"

TEXT = {
    "en": {
        "font": "DejaVu Sans",
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
        "font": "Noto Sans CJK JP",
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
    t = TEXT[lang]
    plt.rcParams["font.family"] = t["font"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.set_xlim(-1.65, 1.65); ax.set_ylim(-1.7, 1.65); ax.axis("off")
    ax.set_aspect("equal")

    # four ring nodes at top, right, bottom, left
    ang = np.deg2rad([90, 0, -90, 180])
    R = 1.05
    pos = np.c_[R * np.cos(ang), R * np.sin(ang)]
    for (x, y), label in zip(pos, t["nodes"]):
        ax.add_patch(FancyBboxPatch((x - 0.46, y - 0.20), 0.92, 0.40,
                     boxstyle="round,pad=0.03,rounding_size=0.08",
                     fc="white", ec=RED, lw=1.8, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=INK, zorder=4)

    # clockwise reinforcing arrows between consecutive ring nodes (top->right->bottom->left->top)
    for i in range(4):
        p0 = pos[i]; p1 = pos[(i + 1) % 4]
        a = FancyArrowPatch(_edge(p0, 0.50), _edge(p1, 0.50, incoming=True),
                            connectionstyle="arc3,rad=-0.32", arrowstyle="-|>",
                            mutation_scale=22, lw=2.6, color=RED, zorder=2)
        ax.add_patch(a)
        mid = 0.5 * (p0 + p1) * 1.42
        ax.text(mid[0], mid[1], t["edge"][i], ha="center", va="center", fontsize=8.2,
                color=RED, style="italic", zorder=4)

    # external trigger feeding the inflammation node (top)
    tx, ty = -1.35, 1.42
    ax.add_patch(FancyBboxPatch((tx - 0.40, ty - 0.18), 0.80, 0.36,
                 boxstyle="round,pad=0.03,rounding_size=0.08", fc="#f5eef8", ec=TRIG, lw=1.6, zorder=3))
    ax.text(tx, ty, t["trigger"], ha="center", va="center", fontsize=9.0, color=TRIG, zorder=4)
    ax.add_patch(FancyArrowPatch((tx + 0.36, ty - 0.10), (pos[0][0] - 0.40, pos[0][1] + 0.16),
                 connectionstyle="arc3,rad=-0.2", arrowstyle="-|>", mutation_scale=18,
                 lw=2.0, color=TRIG, zorder=2))

    # centre label + tipping annotation
    ax.text(0, 0.16, t["center"], ha="center", va="center", fontsize=20, fontweight="bold",
            color=RED, alpha=0.85, zorder=4)
    ax.text(0, -0.30, t["tipping"], ha="center", va="center", fontsize=8.0, color=INK, zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec="0.7", lw=0.6))

    # homeostatic counter-loop note (green, faint)
    ax.text(0, -1.55, t["counter"], ha="center", va="center", fontsize=8.2, color=GREEN, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#eafaf1", ec=GREEN, lw=0.8))

    ax.set_title(t["title"], fontsize=12, color=INK, pad=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_vicious_cycle_{lang}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_vicious_cycle_{lang}.pdf / .png")


def _edge(p, frac, incoming=False):
    """Offset a point from a node centre toward the ring so arrows start/end at box edges."""
    v = np.array(p, float)
    n = v / (np.linalg.norm(v) + 1e-9)
    # tangential offset so the curved arrow leaves/enters cleanly
    tang = np.array([-n[1], n[0]]) * (0.30 if not incoming else -0.30)
    return tuple(v - n * 0.30 + tang)


if __name__ == "__main__":
    langs = [sys.argv[1]] if len(sys.argv) > 1 else ["en", "ja"]
    for lg in langs:
        make(lg)
