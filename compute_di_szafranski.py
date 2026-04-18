"""
Compute MDI / eMDI from Szafrański 2025 / Joshi 2026 species abundance data.

Usage:
    python compute_di_szafranski.py --s5 path/to/supplemental_table_s5.csv
    python compute_di_szafranski.py --demo   # run with mock data

MDI formula (Kröger 2018 JCP, Gevers 2014):
    MDI = log( Σ abundance(PD-positive taxa) / Σ abundance(PD-negative taxa) )

eMDI formula (Joshi & Szafrański 2026 JDR):
    Taxa (genus-level):
      Positive (↑ with PD severity): Pseudoramibacter
      Negative (↓ with PD severity): Capnocytophaga, Gemella
    + 9 EC features (metatranscriptomics, optional):
      Positive: 3.4.21.102, 3.4.23.36, 3.4.21.53 (endopeptidases)
      Negative: 5.4.99.9, 4.1.2.40, 5.3.1.26, 2.7.1.144, 3.2.1.20 (galactose), 1.12.99.6 (hydrogenase)
    Performance: R²=0.51 vs probing depth, AUC=0.87
    Range: −5.9 (least dysbiotic) to +5.6 (most dysbiotic)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DIAGNOSIS_ORDER = ["PIH", "PIM", "PI"]
DIAGNOSIS_COLORS = {"PIH": "#2ecc71", "PIM": "#f39c12", "PI": "#e74c3c"}

# MDI taxa (Joshi & Szafrański 2026 JDR, SIIRI BIC cohort)
MDI_POS_GENERA = ["Pseudoramibacter"]
MDI_NEG_GENERA = ["Capnocytophaga", "Gemella"]

# eMDI EC features (metatranscriptomics only)
EMDI_POS_EC = ["3.4.21.102", "3.4.23.36", "3.4.21.53"]
EMDI_NEG_EC = ["5.4.99.9", "4.1.2.40", "5.3.1.26", "2.7.1.144", "3.2.1.20", "1.12.99.6"]


# ---------------------------------------------------------------------------
# MDI / eMDI
# ---------------------------------------------------------------------------
def _sum_genus(row: pd.Series, genera: list) -> float:
    total = 0.0
    for g in genera:
        for col in row.index:
            if g.lower() in col.lower():
                total += float(row[col])
    return total


def compute_mdi(row: pd.Series,
                pos_genera: list = MDI_POS_GENERA,
                neg_genera: list = MDI_NEG_GENERA,
                eps: float = 1e-8) -> float:
    """
    MDI = log( Σ pos_genera / Σ neg_genera )
    Returns np.nan if either group is absent.
    """
    pos = _sum_genus(row, pos_genera)
    neg = _sum_genus(row, neg_genera)
    if neg < eps or pos < eps:
        return np.nan
    return np.log(pos / neg)


def compute_emdi(row: pd.Series,
                 pos_genera: list = MDI_POS_GENERA,
                 neg_genera: list = MDI_NEG_GENERA,
                 pos_ec: list = EMDI_POS_EC,
                 neg_ec: list = EMDI_NEG_EC,
                 eps: float = 1e-8) -> float:
    """
    eMDI = MDI + EC contributions (standardized).
    If no EC columns found, returns MDI (genus-level proxy).
    EC columns expected as '3.4.21.102' or 'EC:3.4.21.102'.
    """
    mdi = compute_mdi(row, pos_genera, neg_genera, eps)
    if np.isnan(mdi):
        return np.nan

    ec_score = 0.0
    n_ec = 0
    for ec in pos_ec:
        for col in row.index:
            if ec in col:
                ec_score += float(row[col])
                n_ec += 1
    for ec in neg_ec:
        for col in row.index:
            if ec in col:
                ec_score -= float(row[col])
                n_ec += 1

    return mdi if n_ec == 0 else mdi + ec_score


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
def make_mock_s5(n_pih=30, n_pim=63, n_pi=32, seed=42) -> pd.DataFrame:
    """
    Mock species abundance table (S5 format).
    Includes MDI-relevant genera (Joshi 2026):
      Capnocytophaga, Gemella  (health-associated, ↓ with PD)
      Pseudoramibacter          (dysbiotic, ↑ with PD)
    Plus representative oral taxa for each diagnosis group.
    """
    rng = np.random.default_rng(seed)
    rows = []

    species_cols = [
        # health-associated commensals
        "Streptococcus oralis", "Streptococcus sanguinis", "Streptococcus gordonii",
        "Actinomyces naeslundii", "Actinomyces oris",
        "Rothia dentocariosa",
        "Veillonella dispar", "Veillonella parvula",
        # MDI health markers
        "Capnocytophaga gingivalis", "Gemella morbillorum",
        # bridge / intermediate
        "Fusobacterium nucleatum", "Fusobacterium periodonticum",
        # dysbiotic pathobionts
        "Porphyromonas gingivalis", "Tannerella forsythia",
        "Treponema denticola", "Prevotella intermedia",
        # MDI dysbiosis marker
        "Pseudoramibacter alactolyticus",
    ]
    # alpha layout: [So×3, An×2, Ro, Vd×2, Capno, Gemella, Fn×2, Pg, Tf, Td, Pi, Pseudo]
    # PIH: health commensals + MDI-neg high; pathogens + Pseudo low
    alpha_pih = np.array([8, 6, 5, 4, 3, 3, 2, 2,  5.0, 3.0,  1, 1,  0.3, 0.2, 0.2, 0.2,  0.1])
    # PIM: mixed
    alpha_pim = np.array([4, 3, 3, 3, 3, 2, 2, 2,  2.0, 1.5,  2, 2,  1.0, 0.8, 0.5, 0.5,  1.0])
    # PI: pathogens dominant; Pseudo ↑, Capno/Gemella ↓
    alpha_pi  = np.array([1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1,  0.3, 0.2,  2, 2,  5, 4, 3, 3,  4.0])

    for diag, alpha, n in [("PIH", alpha_pih, n_pih), ("PIM", alpha_pim, n_pim), ("PI", alpha_pi, n_pi)]:
        for j in range(n):
            ab = rng.dirichlet(alpha)
            r = {"sample_id": f"{diag}_{j:03d}", "diagnosis": diag}
            for k, sp in enumerate(species_cols):
                r[sp] = ab[k]
            rows.append(r)

    return pd.DataFrame(rows).set_index("sample_id")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, emdi_col: str = None, out_dir: Path = Path(".")):
    """
    df: rows=samples, columns include 'diagnosis' + species abundances (+ optional EC columns).
    emdi_col: column name of externally provided eMDI (Joshi 2026 values, if available).
    """
    results = []
    for sample_id, row in df.iterrows():
        mdi = compute_mdi(row)
        emdi_calc = compute_emdi(row)
        results.append({
            "sample_id": sample_id,
            "diagnosis": row["diagnosis"],
            "MDI": mdi,
            "eMDI_calc": emdi_calc,
            "eMDI_ext": row[emdi_col] if emdi_col and emdi_col in row.index else np.nan,
        })

    res = pd.DataFrame(results)
    has_emdi_ext = res["eMDI_ext"].notna().any()

    # --- Figure layout ---
    n_cols = 1 + int(has_emdi_ext)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # --- Panel 1: MDI by diagnosis ---
    ax = axes[0]
    for diag in DIAGNOSIS_ORDER:
        sub = res[res["diagnosis"] == diag]["MDI"].dropna()
        x = DIAGNOSIS_ORDER.index(diag)
        ax.scatter([x] * len(sub), sub, alpha=0.5, color=DIAGNOSIS_COLORS[diag], s=30)
        if len(sub):
            ax.plot([x - 0.2, x + 0.2], [sub.median(), sub.median()],
                    color=DIAGNOSIS_COLORS[diag], lw=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(DIAGNOSIS_ORDER)
    ax.set_ylabel("MDI = log(Pseudoramibacter / (Capno + Gemella))")
    ax.set_title("MDI by diagnosis (Joshi & Szafrański 2026)\nMDI>0: dysbiotic, MDI<0: healthy")
    ax.axhline(0, ls="--", color="gray", alpha=0.5, lw=1)
    patches = [mpatches.Patch(color=DIAGNOSIS_COLORS[d], label=d) for d in DIAGNOSIS_ORDER]
    ax.legend(handles=patches)

    # --- Panel 2: computed eMDI vs externally provided eMDI ---
    if has_emdi_ext:
        ax2 = axes[1]
        valid = res.dropna(subset=["eMDI_ext", "eMDI_calc"])
        for diag in DIAGNOSIS_ORDER:
            sub = valid[valid["diagnosis"] == diag]
            ax2.scatter(sub["eMDI_ext"], sub["eMDI_calc"],
                        alpha=0.6, color=DIAGNOSIS_COLORS[diag], label=diag, s=40)
        ax2.set_xlabel("eMDI (Joshi 2026, provided)")
        ax2.set_ylabel("eMDI (computed here)")
        ax2.set_title("eMDI: provided vs computed\n(should be identical if taxa match)")
        ax2.legend()
        if len(valid) > 5:
            r = np.corrcoef(valid["eMDI_ext"], valid["eMDI_calc"])[0, 1]
            ax2.text(0.05, 0.95, f"r = {r:.3f}", transform=ax2.transAxes,
                     va="top", fontsize=11)

    plt.tight_layout()
    out_path = out_dir / "mdi_szafranski.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Summary stats ---
    print("\n=== MDI by diagnosis ===")
    print(res.groupby("diagnosis")["MDI"].agg(["mean", "std", "median"]).round(3))
    if res["eMDI_calc"].notna().any():
        print("\n=== eMDI (calc) by diagnosis ===")
        print(res.groupby("diagnosis")["eMDI_calc"].agg(["mean", "std", "median"]).round(3))

    out_csv = out_dir / "mdi_szafranski_results.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s5", type=str, default=None,
                        help="Path to Supplemental Table S5 CSV (species relative abundances)")
    parser.add_argument("--emdi", type=str, default=None,
                        help="Column name of externally provided eMDI values in CSV")
    parser.add_argument("--demo", action="store_true",
                        help="Run with mock data (no real data needed)")
    parser.add_argument("--out", type=str, default=".",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo or args.s5 is None:
        print("Running with mock data (--demo mode)")
        df = make_mock_s5()
    else:
        df = pd.read_csv(args.s5, index_col=0)
        if "diagnosis" not in df.columns:
            raise ValueError("CSV must have a 'diagnosis' column (PIH / PIM / PI)")

    run(df, emdi_col=args.emdi, out_dir=out_dir)
