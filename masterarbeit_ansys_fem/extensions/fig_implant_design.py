"""Generic-implant DESIGN study (ISO 14801-style coupon): four panels from the parametric sweep
(coupon_results.jsonl + coupon_params.jsonl produced by FEM/tier2b_real/run_coupons.sh):
  (A) diameter sweep -- the dominant parameter: thread-root peak vM and coupon stiffness vs Ø;
  (B) parameter sensitivity -- thread-root vM change vs baseline for length / pitch / taper;
  (C) load angle  -- axial vs 30deg oblique (ISO 14801): thread vM, crestal bone vM, displacement;
  (D) mesh order  -- C3D4 vs C3D10 thread-root stress (quadratic captures the concentration).

Run:  python masterarbeit_ansys_fem/extensions/fig_implant_design.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"


def load():
    res = {json.loads(l)["tag"]: json.loads(l) for l in open(FEM / "coupon_results.jsonl")}
    par = {json.loads(l)["tag"]: json.loads(l) for l in open(FEM / "coupon_params.jsonl")}
    for t in res:
        res[t].update(par.get(t, {}))
    return res


def main():
    use()
    R = load()
    C1, C2 = "#b22222", "#1f5fa6"
    fig, ax = plt.subplots(2, 2, figsize=use(width_frac=1.0, aspect=0.85))
    (axA, axB), (axC, axD) = ax

    # (A) diameter sweep: thread vM (bars) + stiffness (line, twin axis)
    dtags = [("D35", 3.5), ("base", 4.1), ("D48", 4.8)]
    D = [d for _, d in dtags]
    thr = [R[t]["thread_vM"] for t, _ in dtags]
    sti = [R[t]["stiffness_N_per_um"] for t, _ in dtags]
    axA.bar([f"{d}" for d in D], thr, color=C1, width=0.55, label="thread-root $\\sigma_\\mathrm{vM}$")
    axA.set_ylabel(r"thread-root $\sigma_\mathrm{vM}$ (MPa)", color=C1, fontsize=7)
    axA.tick_params(axis="y", labelcolor=C1, labelsize=6); axA.tick_params(axis="x", labelsize=6)
    axA.set_xlabel(r"implant diameter $\varnothing$ (mm)", fontsize=7)
    a2 = axA.twinx()
    a2.plot(range(len(D)), sti, "o-", color=C2, lw=1.3, ms=4)
    a2.set_ylabel(r"stiffness (N/$\mu$m)", color=C2, fontsize=7)
    a2.tick_params(axis="y", labelcolor=C2, labelsize=6)
    axA.set_title(r"(A) diameter dominates", fontsize=7.5)

    # (B) parameter sensitivity vs baseline (thread vM): length, pitch, taper
    base = R["base"]["thread_vM"]
    groups = [("length", [("L8", "8"), ("base", "10"), ("L12", "12")]),
              ("pitch", [("P06", "0.6"), ("base", "0.8"), ("P10", "1.0")]),
              ("taper", [("base", "0"), ("TP3", "0.3")])]
    xs = []; lab = []; val = []; col = []
    palette = ["#6a51a3", "#2b8cbe", "#41ab5d"]
    pos = 0
    for gi, (gname, items) in enumerate(groups):
        for tag, lv in items:
            xs.append(pos); lab.append(lv); val.append(R[tag]["thread_vM"]); col.append(palette[gi])
            pos += 1
        pos += 0.8
    axB.bar(xs, val, color=col, width=0.8)
    axB.axhline(base, color="0.5", lw=0.7, ls="--")
    axB.set_xticks(xs); axB.set_xticklabels(lab, fontsize=5.5)
    axB.set_ylabel(r"thread-root $\sigma_\mathrm{vM}$ (MPa)", fontsize=7)
    axB.tick_params(axis="y", labelsize=6)
    # group labels
    centers = [1, 4.8, 7.3]
    for c, (gname, _) in zip(centers, groups):
        axB.text(c, -0.16 * max(val), {"length": "length (mm)", "pitch": "pitch (mm)",
                                       "taper": "taper"}[gname], ha="center", fontsize=6)
    axB.set_title(r"(B) length / pitch / taper: secondary", fontsize=7.5)

    # (C) load angle: axial vs 30deg oblique
    metrics = [("thread_vM", "thread\nvM"), ("crest_p95", "crestal\nbone vM"), ("disp_um", "disp.\n($\\mu$m)")]
    ax0 = [R["A0"][k] for k, _ in metrics]
    ob = [R["base"][k] for k, _ in metrics]
    xi = np.arange(len(metrics)); w = 0.36
    axC.bar(xi - w / 2, ax0, w, color="#7fb3d5", label=r"axial (0$^\circ$)")
    axC.bar(xi + w / 2, ob, w, color="#c0392b", label=r"30$^\circ$ oblique")
    axC.set_yscale("log")
    axC.set_xticks(xi); axC.set_xticklabels([l for _, l in metrics], fontsize=5.8)
    axC.set_ylabel("magnitude (MPa or $\\mu$m, log)", fontsize=6.5)
    axC.tick_params(axis="y", labelsize=6)
    axC.legend(fontsize=5.8, loc="upper right")
    for i, (a, b) in enumerate(zip(ax0, ob)):
        axC.text(i, b * 1.15, r"$\times$%.0f" % (b / a), ha="center", fontsize=6, color="#c0392b")
    axC.set_title(r"(C) oblique load dominates (ISO 14801)", fontsize=7.5)

    # (D) mesh order: C3D4 vs C3D10
    mt = [("thread_vM", "thread-root\nvM"), ("ti_max", "max Ti\nvM")]
    c4 = [R["C3D4"][k] for k, _ in mt]; c10 = [R["base"][k] for k, _ in mt]
    xi = np.arange(len(mt)); w = 0.36
    axD.bar(xi - w / 2, c4, w, color="#aab7b8", label="C3D4 (linear)")
    axD.bar(xi + w / 2, c10, w, color="#34495e", label="C3D10 (quadratic)")
    axD.set_xticks(xi); axD.set_xticklabels([l for _, l in mt], fontsize=6)
    axD.set_ylabel(r"$\sigma_\mathrm{vM}$ (MPa)", fontsize=7); axD.tick_params(axis="y", labelsize=6)
    axD.legend(fontsize=5.8, loc="upper left")
    for i, (a, b) in enumerate(zip(c4, c10)):
        axD.text(i + w / 2, b * 1.02, "+%.0f%%" % (100 * (b - a) / a), ha="center", fontsize=6, color="#34495e")
    axD.set_title(r"(D) C3D10 captures thread-root stress", fontsize=7.5)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_implant_design.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fem_implant_design.pdf")


if __name__ == "__main__":
    main()
