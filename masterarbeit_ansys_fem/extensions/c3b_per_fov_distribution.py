"""C3b — Per-FOV detachment-risk DISTRIBUTION from the depth-resolved profiles of all 84 real FOVs.

C3 (c3_real_fov_stress.py) reports the detachment driver J for only the 10 POOLED (condition x day) depth
profiles. Here we use the per-FOV depth profiles phi_Pg^f(z) extracted for every one of the 84 imaged
titanium fields of view (results/fish_3d/zprofiles_per_fov.csv, from fish_3d_batch.py), so each FOV is an
independent realisation of the substratum-boundary-layer detachment driver:
        J_f = sum_z w(z) phi_Pg^f(z),   w(z) = exp(-z/z0)   (only near-substratum growth builds interface
                                                             stress; verified reduced model, c3 / b4).

KEY (honest) FINDING: the per-FOV driver does NOT separate CH (n=67) from DH (n=17) — neither by total Pg
load (control) NOR by the depth-resolved profile (Mann-Whitney p ~ 0.8). The CH/DH divergence seen in C3
is a POOLED, absolute-magnitude + time-trajectory effect; at the single-FOV level the compositional
phi_Pg(z) is too heterogeneous to classify a field of view as commensal or dysbiotic by its mechanical
driver (DH especially: days 6-21 have only 2-3 FOVs each). => per-FOV mechanical risk maps are not
supported; the C3 claim should be stated at the population/trajectory level. What the per-FOV data DOES
quantify is the spatial HETEROGENEITY of the detachment driver (its spread), which is large in both arms.

Run:  python masterarbeit_ansys_fem/extensions/c3b_per_fov_distribution.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife")
sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402

RED, BLUE = "#d62728", "#1f77b4"
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
PERFOV = ROOT / "results" / "fish_3d" / "zprofiles_per_fov.csv"
Z0 = 0.25
RNG = np.random.default_rng(0)


def per_fov_J():
    """J per FOV from the depth-resolved phi_Pg(z); also return each FOV's (z_frac, phi_Pg) profile."""
    df = pd.read_csv(PERFOV)
    J = {"CH": [], "DH": []}
    profiles = {"CH": [], "DH": []}
    for (c, d, f, fov), g in df.groupby(["cond", "day", "file", "fov"]):
        if c not in J:
            continue
        g = g.sort_values("z_idx")
        zf = g["z_frac"].values
        pg = g["P.g"].values
        J[c].append(float(np.sum(np.exp(-zf / Z0) * pg) / len(zf)))
        profiles[c].append((zf, pg))
    return {c: np.array(v) for c, v in J.items()}, profiles


def boot_ci(x, n=10000, alpha=0.05):
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def main():
    J, profiles = per_fov_J()
    print("Depth-resolved per-FOV detachment driver J (n_CH=%d, n_DH=%d):" % (len(J["CH"]), len(J["DH"])))
    for c in ("CH", "DH"):
        lo, hi = boot_ci(J[c])
        print("  %s: median=%.4f [%.4f, %.4f]  mean=%.4f  IQR=[%.4f, %.4f]  CV=%.2f"
              % (c, np.median(J[c]), lo, hi, J[c].mean(),
                 np.quantile(J[c], .25), np.quantile(J[c], .75), J[c].std() / J[c].mean()))
    U, p = mannwhitneyu(J["DH"], J["CH"], alternative="greater")
    rbc = 2 * U / (len(J["DH"]) * len(J["CH"])) - 1
    print("Mann-Whitney (DH>CH): U=%.0f  p=%.3f  rank-biserial=%.2f  -> NO per-FOV separation." % (U, p, rbc))
    print("(C3's pooled endpoint DH>CH is an absolute-magnitude + trajectory effect, not per-FOV.)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=use(width_frac=0.92, aspect=0.46))
    # left: every FOV's depth profile (faint) + per-condition mean
    for c, col in (("DH", RED), ("CH", BLUE)):
        for zf, pg in profiles[c]:
            a1.plot(pg, zf, "-", color=col, lw=0.3, alpha=0.18)
        # mean on a common grid
        grid = np.linspace(0, 1, 40)
        M = np.mean([np.interp(grid, zf, pg) for zf, pg in profiles[c]], axis=0)
        a1.plot(M, grid, "-", color=col, lw=2.0, label="%s mean (n=%d)" % (c, len(profiles[c])))
    a1.axhspan(0, Z0, color=RED, alpha=0.07)
    a1.text(0.02, 0.02, "interface zone", fontsize=6, color="0.35", transform=a1.get_yaxis_transform())
    a1.set_xlabel(r"per-FOV $\phi_{Pg}(z)$ (composition)")
    a1.set_ylabel(r"depth $z$ (0 = substratum)")
    a1.set_title(r"All 84 per-FOV depth profiles", fontsize=8)
    a1.legend(fontsize=6.5, loc="upper right")

    # right: per-FOV J distribution
    for i, (c, col) in enumerate((("CH", BLUE), ("DH", RED))):
        x = J[c]
        jit = RNG.normal(0, 0.05, size=len(x))
        a2.scatter(np.full(len(x), i) + jit, x, s=10, color=col, alpha=0.55, edgecolor="none", zorder=3)
        a2.hlines(np.median(x), i - 0.25, i + 0.25, color="0.15", lw=1.6, zorder=4)
        lo, hi = np.quantile(x, .25), np.quantile(x, .75)
        a2.add_patch(plt.Rectangle((i - 0.18, lo), 0.36, hi - lo, fill=False, edgecolor=col, lw=1.1, zorder=2))
    a2.set_xticks([0, 1]); a2.set_xticklabels(["CH (n=%d)" % len(J["CH"]), "DH (n=%d)" % len(J["DH"])])
    a2.set_ylabel(r"per-FOV detachment driver $J$")
    a2.set_title(r"Depth-resolved $J$ does NOT separate per-FOV ($p=%.2f$):" "\n"
                 r"large heterogeneity; risk is a pooled/trajectory effect" % p, fontsize=7)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "F18_per_fov_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", OUT / "F18_per_fov_distribution.pdf")


if __name__ == "__main__":
    main()
