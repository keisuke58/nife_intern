# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_sign_enrichment_alpha.py — Sign enrichment permutation test results.

Two-panel figure:
  Left  — Hamilton (symmetric A): SA by alpha, cross-feeding vs competition
  Right — gLV (asymmetric A): SA by alpha, cross-feeding vs competition

Chance level shown as dashed line.

Run from repo root:
    python scripts/figures/fig_sign_enrichment_alpha.py
"""

import json
import numpy as np
import matplotlib as _mpl
import matplotlib.pyplot as plt
from pathlib import Path

import thesis_style

_here = Path(__file__).resolve().parents[2]
RES = _here / "results" / "dieckow_cr"
OUT_PDF = _here / "results" / "fig_sign_enrichment_alpha.pdf"
OUT_PNG = _here / "results" / "fig_sign_enrichment_alpha.png"


def main():
    thesis_style.use()
    _mpl.rcParams["text.usetex"] = False

    data = json.loads((RES / "noprior_permutation_test.json").read_text())

    MODEL_KEYS = [
        ("hamilton_expanded", "Hamilton (symmetric $A$)"),
        ("glv",               "gLV (asymmetric $A$)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    fig.subplots_adjust(wspace=0.12, left=0.10, right=0.97, top=0.88, bottom=0.18)

    for ax, (mkey, mlabel) in zip(axes, MODEL_KEYS):
        ar_list = data["models"][mkey]["alpha_results"]

        alphas   = [r["alpha"] for r in ar_list]
        sa_pos   = [r["agreement_on_predicted_pos"] if r["agreement_on_predicted_pos"] is not None else np.nan
                    for r in ar_list]
        sa_neg   = [r["agreement_on_predicted_neg"] if r["agreement_on_predicted_neg"] is not None else np.nan
                    for r in ar_list]
        sa_all   = [r["observed_agreement"] for r in ar_list]
        pvals    = [r["p_value"] for r in ar_list]
        perm_mu  = [r["perm_mean"] for r in ar_list]

        alphas  = np.array(alphas)
        sa_pos  = np.array(sa_pos, dtype=float)
        sa_neg  = np.array(sa_neg, dtype=float)
        sa_all  = np.array(sa_all, dtype=float)
        perm_mu = np.array(perm_mu, dtype=float)

        # chance band (permutation mean ± 1 std not stored, just draw chance=0.5)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="Chance (0.5)")

        # permutation null mean
        ax.plot(alphas, perm_mu, color="gray", lw=1.0, ls=":", label="Perm. null mean")

        # cross-feeding direction (predicted +)
        ax.plot(alphas, sa_pos, "o-", color="#2166ac", lw=1.5, ms=5,
                label="Cross-feeding (+)")

        # competition direction (predicted −)
        mask = ~np.isnan(sa_neg)
        if mask.any():
            ax.plot(alphas[mask], sa_neg[mask], "s--", color="#d6604d", lw=1.5, ms=5,
                    label="Competition (−)")

        # significance stars at alpha=0 for cross-feeding
        for i, (a, p, sp) in enumerate(zip(alphas, pvals, sa_pos)):
            if not np.isnan(sp) and p < 0.05:
                ax.text(a, sp + 0.03, "*" if p < 0.05 else "",
                        ha="center", va="bottom", fontsize=8, color="#2166ac")
            if not np.isnan(sp) and p < 0.001:
                ax.text(a, sp + 0.03, "***", ha="center", va="bottom",
                        fontsize=8, color="#2166ac")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Competition weight $\\alpha$", fontsize=9)
        ax.set_title(mlabel, fontsize=9, pad=4)
        ax.set_xticks([0, 0.25, 0.5, 1.0])
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Sign agreement (SA)", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.85)

    fig.suptitle("Sign enrichment: Hamilton vs gLV across prior stiffness",
                 fontsize=9, y=0.97)

    fig.savefig(OUT_PDF, dpi=150)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
