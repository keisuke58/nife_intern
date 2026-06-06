# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_szafranski_dindex.py — Guild Dysbiosis Index on Szafrański 127 samples.

Method:
  1. CS attractor = mean composition (So/An/Vp/Fn/Pg) from Heine Commensal_Static data.
     DH attractor = mean composition from Heine Dysbiotic_HOBIC data (last 2 timepoints).
  2. Dysbiosis axis v = DH_mean - CS_mean  (5-dim, normalised).
  3. Each Szafrański sample is projected: DI = (phi_s - CS_mean) · v / ||v||²
     DI = 0 → at CS attractor; DI = 1 → at DH attractor.
  4. The 5 genera map to 5 guilds:
       Streptococcus → Bacilli (So proxy)
       Actinomyces   → Actinobacteria (An proxy)
       Veillonella   → Negativicutes (Vp proxy)
       Fusobacterium → Fusobacteriia (Fn proxy)
       Porphyromonas → Bacteroidia   (Pg proxy)

Run from repo root:
    python scripts/figures/fig_szafranski_dindex.py
"""

import numpy as np
import pandas as pd
import matplotlib as _mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu

import thesis_style

_ROOT  = Path(__file__).resolve().parents[2]
_PROC  = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/processed_data")
_SZAF  = _ROOT / "Datasets" / "20260416_mSystems_16S_5genera_all_profiles.xlsx"
OUT_PDF = _ROOT / "results" / "fig_szafranski_dindex.pdf"
OUT_PNG = _ROOT / "results" / "fig_szafranski_dindex.png"

# species order in Heine data
SP_COLS = ["Frac_S.oralis", "Frac_A.naeslundii", "Frac_Veillonella",
           "Frac_F.nucleatum", "Frac_P.gingivalis"]

# corresponding Szafrański genus → position index
GENERA = ["Streptococcus", "Actinomyces", "Veillonella", "Fusobacterium", "Porphyromonas"]
GENUS_LABEL = ["Streptococcus\n(Bacilli)",
               "Actinomyces\n(Actinobact.)",
               "Veillonella\n(Negativicutes)",
               "Fusobacterium\n(Fusobacteriia)",
               "Porphyromonas\n(Bacteroidia)"]

DIAG_ORDER  = ["Health", "Mucositis", "Peri-implantitis"]
DIAG_COLORS = {"Health": "#2ecc71", "Mucositis": "#f39c12", "Peri-implantitis": "#e74c3c"}
DIAG_SHORT  = {"Health": "PIH", "Mucositis": "PIM", "Peri-implantitis": "PI"}


def load_attractors():
    cs_df = pd.read_csv(_PROC / "target_data_Commensal_Static_normalized.csv")
    dh_df = pd.read_csv(_PROC / "target_data_Dysbiotic_HOBIC_normalized.csv")
    # use last two timepoints as attractor estimate
    cs_mean = cs_df.tail(2)[SP_COLS].mean().values
    dh_mean = dh_df.tail(2)[SP_COLS].mean().values
    # re-normalise to simplex
    cs_mean /= cs_mean.sum()
    dh_mean /= dh_mean.sum()
    return cs_mean, dh_mean


def load_szafranski():
    df = pd.read_excel(_SZAF)
    # drop Total row
    df = df[df["Genus"] != "Total"].copy()
    genera_present = df["Genus"].values
    sample_cols = [c for c in df.columns if c != "Genus"]
    diag = [c.split(".")[0] for c in sample_cols]

    # build (n_samples, 5) matrix
    phi = np.zeros((len(sample_cols), 5))
    for gi, genus in enumerate(GENERA):
        rows = df[df["Genus"] == genus]
        if len(rows) == 0:
            continue
        vals = rows[sample_cols].values.sum(axis=0)  # sum in case multiple rows
        phi[:, gi] = vals

    # normalise each sample to sum=1 (5-genera only)
    row_sums = phi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    phi = phi / row_sums
    return phi, diag, sample_cols


def main():
    thesis_style.use()
    _mpl.rcParams["text.usetex"] = False

    cs_mean, dh_mean = load_attractors()
    phi_s, diag, sample_cols = load_szafranski()

    # dysbiosis axis
    v = dh_mean - cs_mean                          # (5,)
    v_norm2 = np.dot(v, v)

    # project each sample: DI = (phi - cs) · v / ||v||²
    di = np.array([np.dot(phi_s[i] - cs_mean, v) / v_norm2
                   for i in range(len(diag))])

    df_res = pd.DataFrame({"sample": sample_cols, "diagnosis": diag, "DI": di})

    # ── statistics ────────────────────────────────────────────────────────────
    groups = [df_res[df_res["diagnosis"] == d]["DI"].values for d in DIAG_ORDER]
    H, p_kw = kruskal(*groups)
    # pairwise PIH vs PI
    _, p_pih_pi = mannwhitneyu(groups[0], groups[2], alternative="less")

    print(f"Kruskal-Wallis H={H:.2f}, p={p_kw:.4f}")
    print(f"Mann-Whitney PIH < PI: p={p_pih_pi:.4f}")
    for d, g in zip(DIAG_ORDER, groups):
        print(f"  {d}: n={len(g)}, median DI={np.median(g):.3f}, "
              f"IQR=[{np.percentile(g,25):.3f},{np.percentile(g,75):.3f}]")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.16, wspace=0.32)

    # Panel A: strip + box
    ax = axes[0]
    for xi, d in enumerate(DIAG_ORDER):
        vals = df_res[df_res["diagnosis"] == d]["DI"].values
        jitter = np.random.default_rng(42 + xi).uniform(-0.15, 0.15, len(vals))
        ax.scatter(xi + jitter, vals, color=DIAG_COLORS[d], alpha=0.55, s=18, zorder=3)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax.plot([xi - 0.2, xi + 0.2], [med, med], color="k", lw=2.0, zorder=4)
        ax.fill_between([xi - 0.18, xi + 0.18], q25, q75,
                        color=DIAG_COLORS[d], alpha=0.20, zorder=2)

    # attractor reference lines
    ax.axhline(0.0, color="#2166ac", ls="--", lw=0.9, alpha=0.7, label="CS attractor (DI=0)")
    ax.axhline(1.0, color="#b2182b", ls="--", lw=0.9, alpha=0.7, label="DH attractor (DI=1)")

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{DIAG_SHORT[d]}\n(n={len(groups[i])})"
                        for i, d in enumerate(DIAG_ORDER)], fontsize=8)
    ax.set_ylabel("Dysbiosis Index (DI)", fontsize=9)
    ax.set_title("(A) DI by diagnosis", fontsize=9, pad=4)
    ax.legend(fontsize=6.5, loc="upper left")

    # significance annotation
    y_max = di.max() + 0.1
    ax.plot([0, 2], [y_max, y_max], color="k", lw=0.8)
    stars = "***" if p_pih_pi < 0.001 else ("**" if p_pih_pi < 0.01 else
            ("*" if p_pih_pi < 0.05 else "ns"))
    ax.text(1, y_max + 0.03, f"{stars} (p={p_pih_pi:.3f})", ha="center", fontsize=7.5)

    # Panel B: axis composition (CS vs DH)
    ax2 = axes[1]
    x = np.arange(5)
    w = 0.35
    ax2.bar(x - w/2, cs_mean, w, color="#2166ac", alpha=0.75, label="CS attractor")
    ax2.bar(x + w/2, dh_mean, w, color="#b2182b", alpha=0.75, label="DH attractor")
    ax2.set_xticks(x)
    ax2.set_xticklabels(GENUS_LABEL, fontsize=6.5, rotation=25, ha="right")
    ax2.set_ylabel("Mean composition (fraction)", fontsize=9)
    ax2.set_title("(B) CS vs DH attractor profiles\n(Heine et al., last 2 timepoints)",
                  fontsize=8.5, pad=4)
    ax2.legend(fontsize=7)

    fig.suptitle(
        f"Guild Dysbiosis Index: Szafrański 127 samples  "
        f"(Kruskal–Wallis p={p_kw:.3f})",
        fontsize=9, y=0.97)

    fig.savefig(OUT_PDF, dpi=150)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PDF}")


if __name__ == "__main__":
    main()
