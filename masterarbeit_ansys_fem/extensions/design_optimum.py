"""T2-entry — parametric design optimisation toward the lowest crestal overload driver.

Bridges T0 (re-scoring) toward T2 (free-form topology optimisation, a project-scale effort). The
existing FEM design sweep is *one-knob-at-a-time*, which supports an ADDITIVE main-effects model
(no interaction data). We fit per-knob main effects of the crestal overload driver (crest p95),
predict the additive optimum across knobs, and recommend the single highest-value FEM run next
(active-learning step): validate the combined optimum (interactions untested) and push the dominant
knob past its tested range.

HONEST SCOPE. This optimises within the *parametric* design box from existing data; it does not yet
do free-form shape/topology optimisation (true T2), which needs an FEM-in-the-loop optimiser
(adjoint / level-set / surrogate-BO over a richer DOE). The additive assumption is explicit and is
exactly what the recommended validation run tests.

Run:  python design_optimum.py
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


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> int:
    coup = {r["tag"]: r for r in load_jsonl(FEM / "coupon_results.jsonl")}
    crown = load_jsonl(FEM / "design_results.jsonl")
    base = coup["base"]["crest_p95"]

    # geometric knobs (exclude load angle A0 = an occlusal/placement condition, not implant geometry)
    knobs = {
        "diameter D [mm]":   ([("D35", 3.5), ("base", 4.1), ("D48", 4.8)], "min"),
        "length L [mm]":     ([("L8", 8), ("base", 10), ("L12", 12)], "min"),
        "thread pitch [mm]": ([("P06", 0.6), ("base", 0.8), ("P10", 1.0)], "min"),
        "taper":             ([("base", 0.0), ("TP3", 0.3)], "min"),
    }

    # additive main effect of each knob = best (lowest-driver) level minus base
    best_levels = {}
    deltas = {}
    print("=== per-knob best setting (lowest crestal overload driver) ===")
    for knob, (items, _) in knobs.items():
        avail = [(t, v, coup[t]["crest_p95"]) for t, v in items if t in coup]
        bt, bv, bc = min(avail, key=lambda a: a[2])
        best_levels[knob] = (bt, bv, bc)
        deltas[knob] = bc - base
        print(f"  {knob:18s} best = {bv} ({bt}) -> {bc:.1f} MPa  (Δ vs base {bc-base:+.1f})")

    # crown height (moment arm) — pick the lowest-driver height
    ch_best = min(crown, key=lambda r: r["crest_p95"])
    print(f"  {'crown height [mm]':18s} best = {ch_best['crown_h']} -> {ch_best['crest_p95']:.1f} MPa")

    # additive prediction at the combined optimum (geometry knobs; crown handled separately)
    pred_geom = base + sum(deltas.values())
    print(f"\nadditive-predicted combined geometric optimum: {pred_geom:.1f} MPa "
          f"({(pred_geom/base-1)*100:+.0f}% vs base)")
    dom = min(deltas, key=deltas.get)
    print(f"dominant knob = {dom} (Δ {deltas[dom]:+.1f} MPa)")
    print("\nNEXT FEM RUN (active learning):")
    print(f"  1) validate the COMBINED optimum (D={best_levels['diameter D [mm]'][1]}, "
          f"L={best_levels['length L [mm]'][1]}, pitch={best_levels['thread pitch [mm]'][1]}, "
          f"taper={best_levels['taper'][1]}, crown_h={ch_best['crown_h']}) — interactions untested")
    print(f"  2) push the dominant knob past range: diameter > 4.8 mm to locate the optimum")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # (A) main effects
    for knob, (items, _) in knobs.items():
        xs = [v for t, v in items if t in coup]
        ys = [coup[t]["crest_p95"] for t, v in items if t in coup]
        xs01 = (np.array(xs) - min(xs)) / (max(xs) - min(xs) + 1e-9)
        axes[0].plot(xs01, ys, "o-", lw=1.8, label=knob)
        bt, bv, bc = best_levels[knob]
        axes[0].plot((bv - min(xs)) / (max(xs) - min(xs) + 1e-9), bc, "*", ms=14, color="#27ae60", zorder=5)
    axes[0].axhline(base, color="k", ls="--", lw=1.0, label=f"base {base:.1f} MPa")
    axes[0].set_xlabel("knob setting (normalised low→high)")
    axes[0].set_ylabel("crestal overload driver, crest $p_{95}$ [MPa]")
    axes[0].set_title("(A) main effects (★ = best level)", fontsize=11, loc="left")
    axes[0].legend(fontsize=7.5); axes[0].grid(True, lw=0.4, alpha=0.4)

    # (B) base vs additive optimum vs dominant-knob-only
    labels = ["base", f"{dom.split()[0]} only", "combined\n(additive pred.)"]
    vals = [base, base + deltas[dom], pred_geom]
    axes[1].bar(range(3), vals, color=["#7f8c8d", "#e67e22", "#27ae60"], alpha=0.88)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("predicted crest $p_{95}$ [MPa]")
    axes[1].set_title("(B) toward the optimum", fontsize=11, loc="left")
    axes[1].grid(True, axis="y", lw=0.4, alpha=0.4)

    fig.suptitle("Parametric implant-design optimisation toward the lowest overload driver (T2-entry)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_design_optimum.{ext}", dpi=200)
    print(f"\nwrote {(OUT / 'fig_design_optimum.pdf').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
