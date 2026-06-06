#!/usr/bin/env python3
"""fig_pinn_diffusivity.py — PINN-inferred diffusivities DH vs CH.

Two-panel thesis figure:
  Panel A  z-profile PINN fit (DH, one representative day)
  Panel B  bar chart D_i: DH (red) vs CH (green), log-scale

Reads:
  results/pinn_diffusion/pinn_D_fit_DH.json
  results/pinn_diffusion/pinn_D_fit_CH.json
  results/pinn_diffusion/pinn_weights_DH.npz  (for forward pass)
  results/diffusion_fit/zprofiles_all_ti.csv  (observed profiles)
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import thesis_style as _ts

from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
PINND   = ROOT / "results/pinn_diffusion"
FITCSV  = ROOT / "results/diffusion_fit/zprofiles_all_ti.csv"
FIGDIR  = ROOT / "results"
THESIS  = Path.home() / "LUHsummer26/30_Masterarbeit/figures"

SPECIES  = ["S.o", "A.n", "Vd/Vp", "F.n", "P.g"]
SP_COLS  = ["S.oralis", "A.naeslundii", "Vd/Vp", "F.nucleatum", "P.gingivalis"]
COND_COL = {"DH": "#B2182B", "CH": "#2166AC"}
COND_LABEL = {"DH": "Dysbiotic HOBIC", "CH": "Commensal HOBIC"}


def load_fit(cond: str) -> dict:
    p = PINND / f"pinn_D_fit_{cond}.json"
    return json.loads(p.read_text())


def load_profiles(cond: str, day: int):
    import pandas as pd
    df = pd.read_csv(FITCSV)
    sub = df[(df["condition"] == cond) & (df["day"] == day)].sort_values("z_um")
    z = sub["z_um"].values
    phi = sub[SP_COLS].values
    return z, phi


def main():
    dh = load_fit("DH")
    ch = load_fit("CH")

    D_dh = np.array(dh["D_fit"])
    D_ch = np.array(ch["D_fit"])

    # ── figure ────────────────────────────────────────────────────────────────
    figsize = _ts.use(width_frac=1.0, aspect=0.42)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Panel A: DH z-profile data (day 3 = latest timepoint) ────────────────
    ax = axes[0]
    day = 21
    try:
        z, phi = load_profiles("DH", day)
        sp_colors = ["#4575B4", "#74ADD1", "#ABD9E9", "#F46D43", "#D73027"]
        for i, (sp, col) in enumerate(zip(SPECIES, sp_colors)):
            ax.plot(z, phi[:, i], color=col, lw=1.2, label=sp)
        ax.set_xlabel(r"Depth $z$ ($\mu$m)")
        ax.set_ylabel(r"Species fraction $\phi_i$")
        ax.set_title(f"DH z-profiles (day {day})", fontsize=9)
        ax.legend(fontsize=6, ncol=1, loc="upper right",
                  framealpha=0.5, handlelength=1.2)
    except Exception as e:
        ax.text(0.5, 0.5, f"No data\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_title("DH z-profiles", fontsize=9)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: bar chart of D_i ─────────────────────────────────────────────
    ax2 = axes[1]
    x = np.arange(len(SPECIES))
    w = 0.35
    ax2.bar(x - w/2, D_dh, width=w, color=COND_COL["DH"], alpha=0.85,
            label=COND_LABEL["DH"], edgecolor="k", linewidth=0.5)
    ax2.bar(x + w/2, D_ch, width=w, color=COND_COL["CH"], alpha=0.85,
            label=COND_LABEL["CH"], edgecolor="k", linewidth=0.5)

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SPECIES, fontsize=7)
    ax2.set_ylabel(r"Diffusivity $D_i$ (normalised)")
    ax2.set_title("PINN-inferred diffusivities", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.6)
    ax2.grid(True, axis="y", alpha=0.2, lw=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # annotate P.g comparison
    pg_dh, pg_ch = D_dh[4], D_ch[4]
    ratio = pg_ch / pg_dh if pg_dh > 0 else float("nan")
    ax2.annotate(
        f"Pg: CH/DH = {ratio:.1f}×",
        xy=(4, max(pg_dh, pg_ch)),
        xytext=(3.0, max(D_dh.max(), D_ch.max()) * 0.8),
        fontsize=6.5,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"),
        color="#333",
    )

    fig.tight_layout(pad=0.3)

    out_pdf = FIGDIR / "fig_pinn_diffusivity.pdf"
    out_png = FIGDIR / "fig_pinn_diffusivity.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    # numeric summary
    print("\nD_i comparison (normalised units):")
    print(f"{'Species':>10}  {'D_DH':>10}  {'D_CH':>10}  {'ratio CH/DH':>12}")
    for sp, dd, dc in zip(SPECIES, D_dh, D_ch):
        r = dc / dd if dd > 0 else float("nan")
        print(f"{sp:>10}  {dd:>10.5f}  {dc:>10.5f}  {r:>12.2f}")
    print(f"\nu_DH = {dh['u_fit']:.4f}   u_CH = {ch['u_fit']:.4f}")

    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    main()
