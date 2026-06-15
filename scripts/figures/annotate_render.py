#!/usr/bin/env python3
"""
annotate_render.py — overlay a title, a colour-key legend and an optional scale bar
onto a rendered PNG (from ParaView). Runs under the normal python (matplotlib + PIL),
NOT pvbatch.

  python scripts/figures/annotate_render.py --png fig.png \
      --title "Peri-implant hemisection" \
      --legend "Ti implant:#d1d4e0,Bone:#dbca9e,Dentin:#edddb8,Biofilm:#d42123,Gingiva:#e88f99,Crown:#f7f5ee" \
      --scalebar-mm 2 --mm-per-px 0.0123 --out fig_annotated.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def parse_legend(spec):
    out = []
    for item in spec.split(","):
        if ":" not in item:
            continue
        name, col = item.rsplit(":", 1)
        out.append((name.strip(), col.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--legend", default=None, help="name:#hex,name:#hex,...")
    ap.add_argument("--legend-loc", default="upper right")
    ap.add_argument("--scalebar-mm", type=float, default=None)
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--unit", default="mm")
    args = ap.parse_args()

    img = Image.open(args.png).convert("RGBA")
    W, H = img.size
    dpi = 200
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    if args.title:
        ax.text(W * 0.5, H * 0.045, args.title, ha="center", va="center",
                fontsize=15, fontweight="bold", color="#222222",
                family="serif")

    if args.legend:
        items = parse_legend(args.legend)
        n = len(items)
        sw = W * 0.022          # swatch size
        pad = H * 0.012
        line = sw + pad * 0.6
        x0 = W * 0.985 if "right" in args.legend_loc else W * 0.015
        y0 = H * (0.085 if "upper" in args.legend_loc else 0.6)
        right = "right" in args.legend_loc
        for i, (name, col) in enumerate(items):
            y = y0 + i * line
            xsw = x0 - sw if right else x0
            ax.add_patch(Rectangle((xsw, y), sw, sw, facecolor=col,
                                   edgecolor="#333333", lw=0.6))
            tx = xsw - pad * 0.6 if right else xsw + sw + pad * 0.6
            ax.text(tx, y + sw / 2, name, ha="right" if right else "left",
                    va="center", fontsize=11, color="#222222", family="serif")

    if args.scalebar_mm and args.mm_per_px:
        bar_px = args.scalebar_mm / args.mm_per_px
        x1 = W * 0.06
        y1 = H * 0.93
        ax.add_patch(Rectangle((x1, y1), bar_px, H * 0.008,
                               facecolor="#111111", edgecolor="none"))
        ax.text(x1 + bar_px / 2, y1 - H * 0.018,
                f"{args.scalebar_mm:g} {args.unit}", ha="center", va="bottom",
                fontsize=11, color="#111111", family="serif")

    out = args.out or str(Path(args.png).with_name(Path(args.png).stem + "_annot.png"))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[annotate] wrote {out}")


if __name__ == "__main__":
    main()
