"""T0 — re-score the existing implant-design sweep by the *mechanical arm of the vicious cycle*.

The descriptive model (fig_periimplantitis_rankl_opg.py / bone_remodeling_fem_field.py) identifies the
crestal pathological-overload driver as the mechanical trigger of marginal bone loss. The existing
parametric FEM design sweep (run_coupons.sh -> coupon_results.jsonl; run_crown_design.sh ->
design_results.jsonl) reports the crestal stress p95 for each geometry. Here we simply *re-read* that
sweep through the disease lens: rank designs by their crestal overload driver and decompose the effect
by design knob, to identify which geometry choices weaken the mechanical arm of the loop.

HONEST SCOPE. This T0 ranks the mechanical driver at a single (tested) condition; it is real FEM data
re-framed, not a fabricated threshold. The full prescriptive quantity — the dysbiosis tipping threshold
b_crit(theta) at which each design lets the loop run away — requires a per-design bone-loss sweep
(crest stress vs L for each geometry), i.e. a further FEM sweep (T1). The model's A(L) term is
normalised, so b_crit responds to the *shape* of stress-vs-loss, which a single-condition sweep cannot
supply. T0 therefore answers "which knobs lower the overload driver" (actionable now); T1 will answer
"by how much does each design raise b_crit".

Run:  python rank_designs_overload.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
OUT.mkdir(exist_ok=True)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> int:
    coup = {r["tag"]: r for r in load_jsonl(FEM / "coupon_results.jsonl")}
    par = {r["tag"]: r for r in load_jsonl(FEM / "coupon_params.jsonl")}
    crown = load_jsonl(FEM / "design_results.jsonl")

    base = coup["base"]["crest_p95"]
    print(f"baseline crestal overload driver (crest p95) = {base:.1f} MPa")

    # knob groups within the coupon sweep (one knob varied vs base)
    knobs = {
        "diameter D [mm]":  [("D35", 3.5), ("base", 4.1), ("D48", 4.8)],
        "length L [mm]":    [("L8", 8), ("base", 10), ("L12", 12)],
        "thread pitch [mm]": [("P06", 0.6), ("base", 0.8), ("P10", 1.0)],
        "taper":            [("base", 0.0), ("TP3", 0.3)],
        "load angle [deg]": [("A0", 0), ("base", 30)],
    }

    print("\n=== crestal overload driver by design knob (lower = weaker mechanical arm) ===")
    rows = []
    for knob, items in knobs.items():
        for tag, val in items:
            if tag in coup:
                cp = coup[tag]["crest_p95"]
                rows.append((knob, tag, val, cp, (cp / base - 1) * 100))
                print(f"  {knob:18s} {tag:5s} {val:>5}  crest_p95={cp:6.1f} MPa  ({(cp/base-1)*100:+5.0f}% vs base)")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # (A) coupon designs ranked by crest stress
    order = sorted(coup.values(), key=lambda r: r["crest_p95"])
    tags = [r["tag"] for r in order]; vals = [r["crest_p95"] for r in order]
    cols = ["#27ae60" if v < base else ("#7f8c8d" if v == base else "#c0392b") for v in vals]
    axes[0].barh(range(len(tags)), vals, color=cols, alpha=0.85)
    axes[0].set_yticks(range(len(tags))); axes[0].set_yticklabels(tags, fontsize=9)
    axes[0].axvline(base, color="k", ls="--", lw=1.0, label=f"base = {base:.1f} MPa")
    axes[0].set_xlabel("crestal overload driver, crest $p_{95}$ [MPa]")
    axes[0].set_title("(A) designs ranked by overload driver", fontsize=11, loc="left")
    axes[0].legend(fontsize=8); axes[0].grid(True, axis="x", lw=0.4, alpha=0.4)

    # (B) crown height (moment arm) — the clearest lever
    ch = sorted(crown, key=lambda r: r["crown_h"])
    axes[1].plot([r["crown_h"] for r in ch], [r["crest_p95"] for r in ch], "o-", color="#c0392b", lw=2)
    axes[1].axhline(base, color="k", ls="--", lw=1.0, label=f"base crest = {base:.1f} MPa")
    axes[1].set_xlabel("crown height (occlusal moment arm) [mm]")
    axes[1].set_ylabel("crest $p_{95}$ [MPa]")
    axes[1].set_title("(B) moment arm raises the driver", fontsize=11, loc="left")
    axes[1].legend(fontsize=8); axes[1].grid(True, lw=0.4, alpha=0.4)

    # (C) knob sensitivity: % change in driver from the lowest to highest setting of each knob
    knob_span = []
    for knob, items in knobs.items():
        cps = [coup[t]["crest_p95"] for t, _ in items if t in coup]
        if len(cps) >= 2:
            knob_span.append((knob, (max(cps) / min(cps) - 1) * 100))
    knob_span.sort(key=lambda x: x[1])
    axes[2].barh([k for k, _ in knob_span], [v for _, v in knob_span], color="#34495e", alpha=0.85)
    axes[2].set_xlabel("driver swing across the knob range [%]")
    axes[2].set_title("(C) which knob moves the driver most", fontsize=11, loc="left")
    axes[2].grid(True, axis="x", lw=0.4, alpha=0.4)

    fig.suptitle("Re-scoring the implant design sweep by the mechanical arm of the vicious cycle (T0)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_design_overload_ranking.{ext}", dpi=200)
    print(f"\nwrote {(OUT / 'fig_design_overload_ranking.pdf').name}")

    best = min(coup.values(), key=lambda r: r["crest_p95"])
    worst = max(coup.values(), key=lambda r: r["crest_p95"])
    print(f"lowest driver: {best['tag']} ({best['crest_p95']:.1f} MPa); "
          f"highest: {worst['tag']} ({worst['crest_p95']:.1f} MPa)")
    print("T1 next: per-design bone-loss sweep -> crest(L) per geometry -> b_crit(design).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
