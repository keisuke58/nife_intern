#!/usr/bin/env python3
"""
analyze_hobic_vs_dieckow.py — does the in-vitro HOBIC FISH community match the
in-vivo Dieckow 16S guilds?

The 5 HOBIC species map 1:1 to 5 class-level Dieckow guilds:
  S.oralis→Bacilli, A.naeslundii→Actinobacteria, V.dispar/parvula→Negativicutes,
  F.nucleatum→Fusobacteriia, P.gingivalis→Bacteroidia.
Compares relative abundance (renormalised among the 5) of Dieckow (in-vivo, mean
over samples) vs HOBIC CH and DH (FISH voxel biomass), with Spearman rank corr.

Run from the repo root.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})
HERE = Path(__file__).resolve().parents[2]
FITDIR = HERE / "results" / "diffusion_fit"
SP2GUILD = {"S.oralis": "Bacilli", "A.naeslundii": "Actinobacteria",
            "Vd/Vp": "Negativicutes", "F.nucleatum": "Fusobacteriia",
            "P.gingivalis": "Bacteroidia"}
SP = list(SP2GUILD)
GUILDS = [SP2GUILD[s] for s in SP]


def norm(v):
    v = np.asarray(v, float); s = v.sum(); return v / s if s > 0 else v


def main():
    die = pd.read_csv(HERE / "results" / "dieckow_cr" / "dieckow_guild_abundance.csv")
    dieckow = norm([die[g].mean() for g in GUILDS])

    bm = pd.read_csv(FITDIR / "voxel_biomass.csv")
    hob = {}
    for cond in ["CH", "DH"]:
        sub = bm[bm.cond == cond]
        hob[cond] = norm([sub[s].sum() for s in SP])

    x = np.arange(len(SP)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - w, dieckow, w, label="Dieckow in-vivo", color="#555")
    ax.bar(x, hob["CH"], w, label="HOBIC CH (commensal)", color="#1f77b4")
    ax.bar(x + w, hob["DH"], w, label="HOBIC DH (dysbiotic)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({g})" for s, g in zip(SP, GUILDS)], fontsize=9)
    ax.set_ylabel("relative abundance (renorm. among 5)", fontsize=11)
    rc = spearmanr(dieckow, hob["CH"]).correlation
    rd = spearmanr(dieckow, hob["DH"]).correlation
    ax.set_title(f"HOBIC vs Dieckow guilds — Spearman: CH ρ={rc:+.2f}, DH ρ={rd:+.2f}",
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FITDIR / "hobic_vs_dieckow.png", dpi=140, bbox_inches="tight")

    print("species (guild)        Dieckow   HOBIC-CH  HOBIC-DH")
    for k, s in enumerate(SP):
        print(f"  {s:13s}({GUILDS[k]:14s}) {dieckow[k]:6.3f}   {hob['CH'][k]:6.3f}   {hob['DH'][k]:6.3f}")
    print(f"\nSpearman rank corr vs Dieckow:  CH ρ={rc:+.3f}   DH ρ={rd:+.3f}")
    print(f"wrote {FITDIR / 'hobic_vs_dieckow.png'}")


if __name__ == "__main__":
    main()
