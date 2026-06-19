#!/usr/bin/env python3
"""
joshi_jdr2026_severity_validation.py — TRUE Joshi external validation against the JDR 2026
submucosal-microbiome severity dataset (Joshi et al. 2026 J Dent Res, PMC12861548).

PURPOSE
-------
Resolve the data-source confusion documented in
notes/joshi_szafranski_data_source_clarification.md: the historical "joshi_*" scripts in this repo
actually load the Szafrański et al. 2025 mSystems data (127 samples, Health/Mucositis/PI). The TRUE
Joshi peri-implantitis data are:

  (a) Joshi et al. 2025 npj Biofilms Microbiomes — 48 biofilm samples × 32 patients, Health vs. PI
      (no Mucositis), full-length 16S + metatranscriptome.
  (b) Joshi et al. 2026 JDR (PMC12861548) — 49 PI implants × 34 patients, submucosal biofilm,
      peri-implantitis severity scores (no Health group).

This script targets (b) — the severity correlation route — which complements the Szafrański
3-group discrimination and turns the headline R/G-ratio metric from a "category separator" into a
"continuous severity predictor".

INPUTS (expected once supplementary materials are obtained)
-----------------------------------------------------------
data/joshi_jdr2026/per_sample_compositions.tsv
    columns: sample_id, genus, relative_abundance, total_reads, patient_id, severity_score
    (Severity score from JDR supp: probing depth, marginal bone loss, BoP composite — exact
     definition pending Joshi supp materials S1 / S2)

data/joshi_jdr2026/sample_metadata.tsv
    columns: sample_id, patient_id, implant_position, severity_pd_mm, severity_mbl_mm, severity_bop

OUTPUTS
-------
results/joshi_jdr2026_validation/
    rg_ratio_per_sample.csv             R/G ratio per sample (Eq. \\ref{eq:gdi})
    severity_correlation.json           Spearman ρ, p, n, and per-axis breakdown
    fig_severity_correlation.{pdf,png}  R/G vs severity scatter + regression line per axis

EXPECTED HEADLINE
-----------------
If the model is genuinely capturing peri-implantitis biology, R/G ratio should monotone-increase with
each severity axis (PD, MBL) and correlate at ρ ≳ 0.4 over n=49 samples. This would constitute a true
independent external validation (Stiesch group but different cohort, different sequencing protocol,
different reporting standard than Szafrański).

USAGE
-----
    python scripts/loo_cv/joshi_jdr2026_severity_validation.py \\
        --comp data/joshi_jdr2026/per_sample_compositions.tsv \\
        --meta data/joshi_jdr2026/sample_metadata.tsv \\
        --out  results/joshi_jdr2026_validation/

DATA ACQUISITION CHECKLIST (to remove the "placeholder" status of this script)
-----------------------------------------------------------------------------
[ ] Download Joshi JDR 2026 supplementary tables from PMC12861548 (open access).
[ ] Confirm the supp contains a per-sample taxonomic composition table; if not, contact corresponding
    author (Stiesch / Szafrański) — they are already known collaborators on this project.
[ ] Resolve genus naming inconsistencies with our 10-guild map (Streptococcus, Actinomyces,
    Porphyromonas, Fusobacterium, Veillonella all present; remaining 5 guilds zero-imputed as in
    Szafrański route).
[ ] Once data arrive, run this script; expected runtime <10 s.
"""
from __future__ import annotations

import sys as _sys
import pathlib as _pathlib  # noqa  [nife-pathshim]

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 10-guild dysbiotic / commensal definition (matches joshi_attractor_analysis.py)
DYS_GUILDS = ["Fusobacteriia", "Bacteroidia", "Clostridia"]
COM_GUILDS = ["Bacilli", "Actinobacteria", "Negativicutes"]

GENUS_TO_GUILD = {
    "Actinomyces":   "Actinobacteria",
    "Streptococcus": "Bacilli",
    "Porphyromonas": "Bacteroidia",
    "Fusobacterium": "Fusobacteriia",
    "Veillonella":   "Negativicutes",
}

EPS = 1e-5


def rg_ratio(phi_per_guild: dict[str, float]) -> float:
    num = sum(max(phi_per_guild.get(g, 0.0), EPS) for g in DYS_GUILDS)
    den = sum(max(phi_per_guild.get(g, 0.0), EPS) for g in COM_GUILDS)
    return float(np.log(num) - np.log(den))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument("--comp", type=Path, required=True,
                   help="per_sample_compositions.tsv (sample_id, genus, relative_abundance, ...)")
    p.add_argument("--meta", type=Path, required=True,
                   help="sample_metadata.tsv (sample_id, severity_pd_mm, severity_mbl_mm, severity_bop)")
    p.add_argument("--out", type=Path,
                   default=Path("results/joshi_jdr2026_validation"),
                   help="output directory")
    args = p.parse_args()

    if not args.comp.exists() or not args.meta.exists():
        print("ERROR: required input files not present yet — this script is a queued placeholder.")
        print("       see the DATA ACQUISITION CHECKLIST in the docstring for next steps.")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(args.comp, sep="\t")
    meta = pd.read_csv(args.meta, sep="\t")

    # Aggregate composition → per-guild relative abundance per sample.
    comp["guild"] = comp["genus"].map(GENUS_TO_GUILD)
    comp = comp.dropna(subset=["guild"])
    sample_guild = (comp.groupby(["sample_id", "guild"])["relative_abundance"]
                        .sum().reset_index())
    sample_phi = sample_guild.pivot(index="sample_id", columns="guild",
                                    values="relative_abundance").fillna(0.0)

    # R/G ratio per sample
    rg = sample_phi.apply(lambda row: rg_ratio(row.to_dict()), axis=1).rename("rg_ratio")
    df = meta.set_index("sample_id").join(rg, how="inner").reset_index()
    df.to_csv(args.out / "rg_ratio_per_sample.csv", index=False)

    # Severity correlations
    from scipy import stats
    summary = {"n": int(len(df))}
    for axis in ("severity_pd_mm", "severity_mbl_mm", "severity_bop"):
        if axis not in df.columns:
            continue
        sub = df.dropna(subset=[axis, "rg_ratio"])
        if len(sub) < 5:
            summary[axis] = {"n": len(sub), "rho": None, "p": None}
            continue
        rho, pval = stats.spearmanr(sub["rg_ratio"], sub[axis])
        summary[axis] = {"n": int(len(sub)), "rho": float(rho), "p": float(pval)}
    (args.out / "severity_correlation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Scatter figure: R/G vs each severity axis
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes_present = [a for a in ("severity_pd_mm", "severity_mbl_mm", "severity_bop") if a in df.columns]
    if axes_present:
        fig, axs = plt.subplots(1, len(axes_present), figsize=(4.5 * len(axes_present), 4.0))
        if len(axes_present) == 1:
            axs = [axs]
        for ax, axis in zip(axs, axes_present):
            sub = df.dropna(subset=[axis, "rg_ratio"])
            ax.scatter(sub[axis], sub["rg_ratio"], s=22, alpha=0.75, color="#d62728",
                       edgecolor="black", lw=0.4)
            # OLS regression line
            if len(sub) >= 2:
                slope, intercept, r, _, _ = stats.linregress(sub[axis], sub["rg_ratio"])
                xs = np.linspace(sub[axis].min(), sub[axis].max(), 50)
                ax.plot(xs, slope * xs + intercept, "k--", lw=1.0)
                rho = summary[axis]["rho"]; pval = summary[axis]["p"]
                ax.text(0.05, 0.95, f"ρ = {rho:.2f}  p = {pval:.2g}\nn = {len(sub)}",
                        transform=ax.transAxes, va="top", fontsize=9)
            ax.set_xlabel(axis.replace("_", " "))
            ax.set_ylabel("R/G ratio")
        fig.suptitle("Joshi JDR 2026 — R/G ratio vs peri-implantitis severity")
        fig.tight_layout()
        fig.savefig(args.out / "fig_severity_correlation.pdf")
        fig.savefig(args.out / "fig_severity_correlation.png", dpi=200)
        print(f"wrote {(args.out / 'fig_severity_correlation.pdf').name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
