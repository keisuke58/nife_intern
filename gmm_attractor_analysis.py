#!/usr/bin/env python3
"""
gmm_attractor_analysis.py — GMM/EM attractor basin assignment
                             for Szafranski 2025 BIC cohort (127 samples, 5 genera)

手法:
  1. 127サンプルの5次元組成データにGMM(k=4)をCLR空間でフィット
  2. GMMクラスター重心 ← → Hamilton ODE 参照アトラクター(CS/CH/DH/DS) を
     ハンガリアン法でマッチング
  3. 結果を最近傍アトラクター割り当て (attractor_analysis.csv) と比較

出力:
  results/gmm_attractor_analysis.png  — 3パネル図
  results/gmm_attractor_analysis.csv  — サンプルごとのGMMラベル付き
  results/gmm_summary.txt             — テキストサマリー

Usage:
  cd /home/nishioka/IKM_Hiwi/nife
  python gmm_attractor_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent          # IKM_Hiwi/
RUNS       = ROOT / "Tmcmc202601" / "data_5species" / "_runs"
HERE       = Path(__file__).parent
DATA_XLS   = HERE / "Datasets" / "20260416_mSystems_16S_5genera_all_profiles.xlsx"
PREV_CSV   = HERE / "results" / "attractor_analysis.csv"
OUT_DIR    = HERE / "results"

CONDITION_RUNS = {
    "CS": "commensal_static_posterior",
    "CH": "commensal_hobic_posterior",
    "DH": "dh_baseline",
    "DS": "dysbiotic_static_posterior",
}
CONDITION_COLORS = {"CS": "#1a9850", "CH": "#91cf60", "DH": "#fc8d59", "DS": "#d73027"}
DIAG_COLORS      = {"PIH": "#2ecc71", "PIM": "#f39c12", "PI": "#e74c3c"}
DIAGNOSIS_ORDER  = ["PIH", "PIM", "PI"]

SPECIES_ORDER = ["So", "An", "Vd", "Fn", "Pg"]
GENUS_MAP     = {"So": "Streptococcus", "An": "Actinomyces",
                 "Vd": "Veillonella",  "Fn": "Fusobacterium", "Pg": "Porphyromonas"}
DIAG_MAP      = {"Health": "PIH", "Mucositis": "PIM", "Peri-implantitis": "PI"}

C_CONST = 25.0;  K_HILL = 0.05;  N_HILL = 4.0;  T_STEADY = 50.0


# ---------------------------------------------------------------------------
# θ → A, b  (20-param layout, same as attractor_analysis.py)
# ---------------------------------------------------------------------------
def theta_to_matrices(theta):
    A = np.zeros((5, 5));  b = np.zeros(5)
    A[0,0]=theta[0]; A[0,1]=A[1,0]=theta[1]; A[1,1]=theta[2]
    b[0]=theta[3];   b[1]=theta[4]
    A[2,2]=theta[5]; A[2,3]=A[3,2]=theta[6]; A[3,3]=theta[7]
    b[2]=theta[8];   b[3]=theta[9]
    A[0,2]=A[2,0]=theta[10]; A[0,3]=A[3,0]=theta[11]
    A[1,2]=A[2,1]=theta[12]; A[1,3]=A[3,1]=theta[13]
    A[4,4]=theta[14]; b[4]=theta[15]
    A[0,4]=A[4,0]=theta[16]; A[1,4]=A[4,1]=theta[17]
    A[2,4]=A[4,2]=theta[18]; A[3,4]=A[4,3]=theta[19]
    return A, b


def load_map_theta(run_name):
    path = RUNS / run_name / "theta_MAP.json"
    with open(path) as f:
        d = json.load(f)
    return np.array(d.get("theta_full", d.get("theta_sub")))


def run_to_steady(phi0, A, b):
    phi0 = np.clip(phi0, 1e-6, None);  phi0 /= phi0.sum()
    def rhs(t, p):
        p = np.clip(p, 1e-10, 1-1e-10)
        gate = K_HILL**N_HILL / (K_HILL**N_HILL + p**N_HILL)
        return p * (C_CONST*(A@p) - b) * gate
    sol = solve_ivp(rhs, [0, T_STEADY], phi0, method="RK45", rtol=1e-6, atol=1e-8)
    phi_eq = np.clip(sol.y[:,-1], 0, None)
    if phi_eq.sum() > 0:  phi_eq /= phi_eq.sum()
    return phi_eq


def compute_reference_attractors():
    refs = {}
    phi0_uniform = np.ones(5)/5
    for cond, run in CONDITION_RUNS.items():
        path = RUNS / run / "theta_MAP.json"
        if not path.exists():
            print(f"  [skip] {run}")
            continue
        theta = load_map_theta(run)
        A, b = theta_to_matrices(theta)
        refs[cond] = run_to_steady(phi0_uniform, A, b)
    return refs


# ---------------------------------------------------------------------------
# CLR transform  (Aitchison / compositional data analysis)
# ---------------------------------------------------------------------------
def clr(X, eps=1e-6):
    """Centered log-ratio transform.  X: (N,5) relative abundances → (N,5) CLR."""
    Xc = X + eps
    log_X = np.log(Xc)
    return log_X - log_X.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Data loading  (same wide-format as attractor_analysis.py)
# ---------------------------------------------------------------------------
def load_szafranski():
    raw = pd.read_excel(DATA_XLS)
    raw = raw[raw["Genus"] != "Total"].copy()
    genus_df = raw.groupby("Genus").sum(numeric_only=True)   # (5, N)
    samples_df = genus_df.T.copy()
    samples_df.index.name = "sample_id"
    samples_df["diagnosis"] = [DIAG_MAP.get(c.split(".")[0], c) for c in samples_df.index]

    phi_mat = np.zeros((len(samples_df), 5))
    for k, key in enumerate(SPECIES_ORDER):
        genus = GENUS_MAP[key]
        cols = [c for c in samples_df.columns
                if genus.lower() in c.lower() and c != "diagnosis"]
        if cols:
            phi_mat[:, k] = samples_df[cols].values.sum(axis=1)
    row_sums = phi_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-10] = 1.0
    phi_mat /= row_sums

    diag = samples_df["diagnosis"].values
    return phi_mat, diag


# ---------------------------------------------------------------------------
# GMM fitting
# ---------------------------------------------------------------------------
def fit_gmm(phi_mat, n_components=4, seed=42):
    X = clr(phi_mat)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          n_init=20, random_state=seed, max_iter=500)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs  = gmm.predict_proba(X)
    return gmm, labels, probs, X


def match_gmm_to_attractors(gmm_means_clr, ref_attractors, pca):
    """
    Hungarian matching: GMM cluster centroid (in original φ-space via PCA)
    ← → Hamilton attractor positions.
    Returns: cluster_id → attractor_name dict
    """
    ref_keys = list(ref_attractors.keys())
    ref_phi  = np.array([ref_attractors[k] for k in ref_keys])          # (4,5)

    # GMM means are in CLR space.  Back-transform via softmax approximation.
    def clr_inv(clr_vec, eps=1e-6):
        """Approximate CLR inverse (softmax)."""
        e = np.exp(clr_vec)
        return e / e.sum()

    gmm_phi = np.array([clr_inv(m) for m in gmm_means_clr])             # (4,5)

    # L2 distance matrix
    dist = np.linalg.norm(gmm_phi[:,None,:] - ref_phi[None,:,:], axis=2) # (4,4)
    row_ind, col_ind = linear_sum_assignment(dist)

    mapping = {}  # cluster_id → attractor name
    for r, c in zip(row_ind, col_ind):
        mapping[r] = ref_keys[c]
    return mapping, gmm_phi


# ---------------------------------------------------------------------------
# Drawing helper: GMM confidence ellipse on PCA projection
# ---------------------------------------------------------------------------
def draw_gmm_ellipse(ax, mean, cov, pca, color, alpha=0.25, n_std=1.5):
    """Project GMM covariance (in CLR-5D) onto PCA 2D and draw ellipse."""
    # Project mean
    m2 = pca.transform(mean.reshape(1,-1))[0]
    # Project covariance via Jacobian (linear approximation: J = pca.components_)
    J = pca.components_   # (2, 5)
    cov2 = J @ cov @ J.T  # (2, 2)

    vals, vecs = eigh(cov2)
    angle = np.degrees(np.arctan2(vecs[1,1], vecs[0,1]))
    w, h  = 2 * n_std * np.sqrt(np.abs(vals))
    ell   = Ellipse(xy=m2, width=w, height=h, angle=angle,
                    facecolor=color, edgecolor=color,
                    alpha=alpha, linewidth=1.5, linestyle="--")
    ax.add_patch(ell)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Szafranski 2025 data ===")
    phi_mat, diag = load_szafranski()
    print(f"  {len(phi_mat)} samples, {phi_mat.shape[1]} genera")

    print("\n=== Computing Hamilton ODE reference attractors ===")
    refs = compute_reference_attractors()
    ref_keys = list(refs.keys())
    ref_phi  = np.array([refs[k] for k in ref_keys])
    print("  " + ", ".join(f"{k}: So={refs[k][0]:.2f} Fn={refs[k][3]:.2f} Pg={refs[k][4]:.2f}"
                            for k in ref_keys))

    print("\n=== Fitting GMM (k=4) in CLR space ===")
    gmm, gmm_labels, gmm_probs, X_clr = fit_gmm(phi_mat)
    print(f"  BIC = {gmm.bic(X_clr):.1f},  converged = {gmm.converged_}")

    print("\n=== Matching GMM clusters → Hamilton attractors (Hungarian) ===")
    cluster_map, gmm_phi = match_gmm_to_attractors(gmm.means_, refs, None)
    gmm_attractor = np.array([cluster_map[l] for l in gmm_labels])

    for c_id, att in cluster_map.items():
        n = (gmm_labels == c_id).sum()
        m = gmm_phi[c_id]
        print(f"  Cluster {c_id} → {att}  (n={n})  "
              f"So={m[0]:.2f} An={m[1]:.2f} Vd={m[2]:.2f} Fn={m[3]:.2f} Pg={m[4]:.2f}")

    # PCA on CLR space for visualization
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X_clr)
    ref_proj = pca.transform(clr(ref_phi + 1e-6))

    # Nearest attractor (L2 in φ-space) for comparison
    dists_phi = np.linalg.norm(phi_mat[:,None,:] - ref_phi[None,:,:], axis=2)
    nearest_attractor = np.array([ref_keys[i] for i in dists_phi.argmin(axis=1)])

    # Agreement between GMM and nearest-neighbor
    agree = (gmm_attractor == nearest_attractor).mean()
    print(f"\n  GMM vs nearest-attractor agreement: {agree*100:.1f}%")

    # -------------------------------------------------------------------------
    # Figure: 3 panels
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ---- Panel 1: PCA colored by GMM-assigned attractor + confidence ellipses ----
    ax = axes[0]
    for key in ref_keys:
        mask = gmm_attractor == key
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=CONDITION_COLORS[key], s=45, alpha=0.75, zorder=3, label=key)

    # GMM ellipses
    for c_id, att in cluster_map.items():
        m_clr = gmm.means_[c_id]
        cov_clr = gmm.covariances_[c_id]
        draw_gmm_ellipse(ax, m_clr, cov_clr, pca,
                         color=CONDITION_COLORS[att], alpha=0.18, n_std=1.5)

    # Reference attractor stars
    for ki, key in enumerate(ref_keys):
        ax.scatter(*ref_proj[ki], marker="*", s=320,
                   color=CONDITION_COLORS[key], zorder=6,
                   edgecolors="k", linewidth=0.8)

    ax.set_xlabel(f"PC1-CLR ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2-CLR ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("GMM clusters (CLR-PCA)\nStars = Hamilton ODE attractors")
    handles = [mpatches.Patch(color=CONDITION_COLORS[k], label=k) for k in ref_keys]
    ax.legend(handles=handles, fontsize=8, loc="best")

    # ---- Panel 2: Stacked bar — GMM attractor label × diagnosis ----
    ax = axes[1]
    bar_w = 0.5
    diag_labels = [d for d in DIAGNOSIS_ORDER if (diag == d).sum() > 0]
    bottom = np.zeros(len(diag_labels))
    for key in ref_keys:
        counts = np.array([(gmm_attractor[diag == d] == key).sum() for d in diag_labels])
        totals = np.array([(diag == d).sum() for d in diag_labels])
        ax.bar(diag_labels, counts, bar_w, bottom=bottom,
               color=CONDITION_COLORS[key], label=key)
        for xi, (c, t, b) in enumerate(zip(counts, totals, bottom)):
            if c > 0:
                ax.text(xi, b + c/2, f"{c/t*100:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottom += counts
    ax.set_ylabel("Number of samples")
    ax.set_title(f"GMM attractor assignment × diagnosis\n(agree w/ NN: {agree*100:.1f}%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, bottom.max() * 1.15)

    # ---- Panel 3: GMM cluster composition (mean φ per cluster) ----
    ax = axes[2]
    cluster_order = [c for c in ref_keys if any(v == c for v in cluster_map.values())]
    x = np.arange(len(cluster_order))
    bar_w3 = 0.15
    colors5 = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
    labels5 = [GENUS_MAP[k] for k in SPECIES_ORDER]
    for j, sp_idx in enumerate(range(5)):
        vals = [gmm_phi[{v:k for k,v in cluster_map.items()}[att]][sp_idx]
                for att in cluster_order]
        ax.bar(x + j*bar_w3, vals, bar_w3, color=colors5[j], label=labels5[j])
    ax.set_xticks(x + 2*bar_w3)
    ax.set_xticklabels(cluster_order)
    ax.set_ylabel("Mean relative abundance (CLR inverse)")
    ax.set_title("GMM cluster centroid composition\n(in φ-space, softmax of CLR mean)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 0.8)

    plt.suptitle("GMM/EM Attractor Basin Analysis — Szafrański 2025 BIC cohort",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_png = OUT_DIR / "gmm_attractor_analysis.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.close()

    # -------------------------------------------------------------------------
    # Output CSV
    # -------------------------------------------------------------------------
    out_df = pd.DataFrame({
        "sample_id":       np.arange(len(phi_mat)),
        "diagnosis":       diag,
        "phi0_So":         phi_mat[:,0], "phi0_An": phi_mat[:,1],
        "phi0_Vd":         phi_mat[:,2], "phi0_Fn": phi_mat[:,3],
        "phi0_Pg":         phi_mat[:,4],
        "gmm_cluster":     gmm_labels,
        "gmm_attractor":   gmm_attractor,
        "nn_attractor":    nearest_attractor,
        "gmm_nn_agree":    (gmm_attractor == nearest_attractor),
        "prob_CS":         gmm_probs[:, [k for k,v in cluster_map.items() if v=="CS"][0]] if "CS" in cluster_map.values() else 0,
        "prob_CH":         gmm_probs[:, [k for k,v in cluster_map.items() if v=="CH"][0]] if "CH" in cluster_map.values() else 0,
        "prob_DH":         gmm_probs[:, [k for k,v in cluster_map.items() if v=="DH"][0]] if "DH" in cluster_map.values() else 0,
        "prob_DS":         gmm_probs[:, [k for k,v in cluster_map.items() if v=="DS"][0]] if "DS" in cluster_map.values() else 0,
        "PC1_clr":         proj[:,0], "PC2_clr": proj[:,1],
    })
    csv_path = OUT_DIR / "gmm_attractor_analysis.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # -------------------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------------------
    lines = []
    lines.append("=== GMM/EM Attractor Basin Summary ===\n")
    lines.append(f"Data: {len(phi_mat)} samples, 5 genera (CLR-transformed)")
    lines.append(f"GMM: k=4, full covariance, n_init=20")
    lines.append(f"BIC: {gmm.bic(X_clr):.1f},  converged: {gmm.converged_}")
    lines.append(f"\nCluster → Attractor mapping (Hungarian):")
    for c_id, att in sorted(cluster_map.items()):
        n = (gmm_labels == c_id).sum()
        m = gmm_phi[c_id]
        lines.append(f"  Cluster {c_id} → {att}  (n={n:3d})  "
                     f"So={m[0]:.3f} An={m[1]:.3f} Vd={m[2]:.3f} Fn={m[3]:.3f} Pg={m[4]:.3f}")
    lines.append(f"\nGMM vs nearest-neighbor agreement: {agree*100:.1f}%")
    lines.append(f"\nGMM attractor × diagnosis:")
    for d in DIAGNOSIS_ORDER:
        mask = diag == d
        if not mask.any(): continue
        total = mask.sum()
        line = f"  {d} (n={total}): "
        line += " ".join(f"{k}={int((gmm_attractor[mask]==k).sum())}"
                         f"({(gmm_attractor[mask]==k).mean()*100:.0f}%)"
                         for k in ref_keys)
        lines.append(line)
    lines.append(f"\nHamilton ODE reference attractor φ_eq:")
    for k in ref_keys:
        m = refs[k]
        lines.append(f"  {k}: So={m[0]:.3f} An={m[1]:.3f} Vd={m[2]:.3f} Fn={m[3]:.3f} Pg={m[4]:.3f}")

    summary_txt = "\n".join(lines) + "\n"
    summary_path = OUT_DIR / "gmm_summary.txt"
    summary_path.write_text(summary_txt)
    print(f"Saved: {summary_path}")
    print(f"\n{summary_txt}")


if __name__ == "__main__":
    main()
