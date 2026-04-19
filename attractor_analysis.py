"""
attractor_analysis.py — CT I-IV = Hamilton ODE Attractor Basin 検証

Szafrański 2025 の 125 サンプル (S5 相対存在量) を Hamilton 5種 ODE の初期値 φ₀ として使い、
定常状態 φ_eq を計算する。PIH/PIM/PI (または CT I-IV) が ODE の commensal/dysbiotic
attractor の周辺にクラスタリングするかを検証する。

Usage:
    python attractor_analysis.py --demo                      # mock data
    python attractor_analysis.py --s5 s5.csv                 # real S5
    python attractor_analysis.py --s5 s5.csv --ct ct_col     # with CT column

Species mapping (Szafrański 756種 → ODE 5種):
    So (0): Streptococcus 属
    An (1): Actinomyces 属
    Vd (2): Veillonella 属
    Fn (3): Fusobacterium 属
    Pg (4): Porphyromonas 属 (gingivalis が主)

ODE (simplified replicator, c_const=25.0, K=0.05, n=4):
    dφᵢ/dt = φᵢ · (c · Σⱼ Aᵢⱼ·φⱼ - bᵢ) · K^n/(K^n + φᵢ^n)

Reference:
    Hamilton 5-species model: improved_5species_jit.py
    MAP θ:                    data_5species/_runs/*/theta_MAP.json
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent          # IKM_Hiwi/
RUNS = ROOT / "data_5species" / "_runs"

DIAGNOSIS_ORDER  = ["PIH", "PIM", "PI"]
DIAGNOSIS_COLORS = {"PIH": "#2ecc71", "PIM": "#f39c12", "PI": "#e74c3c"}
CT_COLORS        = {"CT I": "#2ecc71", "CT II": "#f39c12", "CT III": "#e67e22", "CT IV": "#e74c3c"}

# 4 TMCMC 条件と S5 診断の対応
CONDITION_RUNS = {
    "CS": "commensal_static_posterior",
    "CH": "commensal_hobic_posterior",
    "DH": "dh_baseline",
    "DS": "dysbiotic_static_posterior",
}
CONDITION_COLORS = {"CS": "#1a9850", "CH": "#91cf60", "DH": "#fc8d59", "DS": "#d73027"}

# Szafrański 5種マッピング (substring match, genus-level)
GENUS_MAP = {
    "So": "Streptococcus",
    "An": "Actinomyces",
    "Vd": "Veillonella",
    "Fn": "Fusobacterium",
    "Pg": "Porphyromonas",
}
SPECIES_ORDER = ["So", "An", "Vd", "Fn", "Pg"]

# ODE parameters (match improved_5species_jit defaults)
C_CONST = 25.0
K_HILL  = 0.05
N_HILL  = 4.0
T_STEADY = 50.0   # [days] — 21日で定常、余裕をもって50日


# ---------------------------------------------------------------------------
# θ → A, b
# ---------------------------------------------------------------------------
def theta_to_matrices(theta: np.ndarray):
    """20パラメータ → A(5,5), b(5)。improved_5species_jit と同一レイアウト。"""
    A = np.zeros((5, 5))
    b = np.zeros(5)
    A[0, 0] = theta[0];  A[0, 1] = A[1, 0] = theta[1];  A[1, 1] = theta[2]
    b[0] = theta[3];     b[1] = theta[4]
    A[2, 2] = theta[5];  A[2, 3] = A[3, 2] = theta[6];  A[3, 3] = theta[7]
    b[2] = theta[8];     b[3] = theta[9]
    A[0, 2] = A[2, 0] = theta[10];  A[0, 3] = A[3, 0] = theta[11]
    A[1, 2] = A[2, 1] = theta[12];  A[1, 3] = A[3, 1] = theta[13]
    A[4, 4] = theta[14]; b[4] = theta[15]
    A[0, 4] = A[4, 0] = theta[16];  A[1, 4] = A[4, 1] = theta[17]
    A[2, 4] = A[4, 2] = theta[18];  A[3, 4] = A[4, 3] = theta[19]
    return A, b


def load_map_theta(run_name: str) -> np.ndarray:
    path = RUNS / run_name / "theta_MAP.json"
    with open(path) as f:
        d = json.load(f)
    return np.array(d.get("theta_full", d.get("theta_sub")))


# ---------------------------------------------------------------------------
# ODE integration (simplified φ-only replicator)
# ---------------------------------------------------------------------------
def make_ode(A: np.ndarray, b: np.ndarray):
    """A, b を閉包にした dφ/dt 関数を返す。"""
    def rhs(t, phi):
        phi = np.clip(phi, 1e-10, 1.0 - 1e-10)
        gate = K_HILL**N_HILL / (K_HILL**N_HILL + phi**N_HILL)
        growth = C_CONST * (A @ phi) - b
        dphi = phi * growth * gate
        return dphi
    return rhs


def run_to_steady(phi0: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    φ₀ から ODE を T_STEADY 日まで積分し、定常状態 φ_eq を返す。
    φ は [0,1] クリップ後に L1 正規化 (合計 = Σφ₀)。
    """
    phi0 = np.clip(phi0, 1e-6, None)
    phi0 = phi0 / phi0.sum()   # 正規化

    rhs = make_ode(A, b)
    sol = solve_ivp(rhs, [0, T_STEADY], phi0, method="RK45",
                    rtol=1e-6, atol=1e-8, dense_output=False)
    phi_eq = np.clip(sol.y[:, -1], 0, None)
    if phi_eq.sum() > 0:
        phi_eq /= phi_eq.sum()
    return phi_eq


# ---------------------------------------------------------------------------
# Reference attractors (4条件の MAP θ から定常状態を計算)
# ---------------------------------------------------------------------------
def compute_reference_attractors() -> dict:
    """4 TMCMC 条件それぞれの MAP θ → φ_eq (reference attractor)。"""
    refs = {}
    phi0_uniform = np.ones(5) / 5   # 均一初期値
    for cond, run in CONDITION_RUNS.items():
        path = RUNS / run / "theta_MAP.json"
        if not path.exists():
            print(f"  [skip] {run} not found")
            continue
        theta = load_map_theta(run)
        A, b = theta_to_matrices(theta)
        phi_eq = run_to_steady(phi0_uniform, A, b)
        refs[cond] = phi_eq
        print(f"  {cond}: φ_eq = So={phi_eq[0]:.3f} An={phi_eq[1]:.3f} "
              f"Vd={phi_eq[2]:.3f} Fn={phi_eq[3]:.3f} Pg={phi_eq[4]:.3f}")
    return refs


# ---------------------------------------------------------------------------
# S5 データ → 5種 φ₀ 抽出
# ---------------------------------------------------------------------------
def extract_5species(df: pd.DataFrame) -> np.ndarray:
    """
    df の各行 (サンプル) について Szafrański の 756種 → 5種 φ を抽出。
    属名 substring match で集計し、L1 正規化。
    Returns: (N, 5) array
    """
    N = len(df)
    phi_mat = np.zeros((N, 5))
    species_cols = [c for c in df.columns if c not in ("diagnosis", "ct", "sample_id")]

    for k, key in enumerate(SPECIES_ORDER):
        genus = GENUS_MAP[key]
        matched = [c for c in species_cols if genus.lower() in c.lower()]
        if matched:
            phi_mat[:, k] = df[matched].values.sum(axis=1)
        else:
            print(f"  [warn] {genus} not found in columns")

    # 行ごとに正規化
    row_sums = phi_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-10] = 1.0
    phi_mat /= row_sums
    return phi_mat


# ---------------------------------------------------------------------------
# Mock data (demo モード)
# ---------------------------------------------------------------------------
def make_mock_s5(n_pih=30, n_pim=63, n_pi=32, seed=42) -> pd.DataFrame:
    """
    5種に絞ったモック S5。診断別に attractor 付近の Dirichlet サンプル。
    PIH: So 優占, Pg 微量
    PI:  Pg/Fn 優占, So 微量
    PIM: 中間
    """
    rng = np.random.default_rng(seed)
    rows = []
    # [So, An, Vd, Fn, Pg]
    alpha_pih = np.array([8.0, 4.0, 3.0, 1.0, 0.1])
    alpha_pim = np.array([3.0, 2.0, 2.0, 2.0, 1.5])
    alpha_pi  = np.array([0.5, 0.5, 1.0, 3.0, 5.0])

    for diag, alpha, n in [("PIH", alpha_pih, n_pih),
                            ("PIM", alpha_pim, n_pim),
                            ("PI",  alpha_pi,  n_pi)]:
        for j in range(n):
            ab = rng.dirichlet(alpha)
            r = {"sample_id": f"{diag}_{j:03d}", "diagnosis": diag}
            for k, sp in enumerate(SPECIES_ORDER):
                r[f"{GENUS_MAP[sp]} sp."] = ab[k]
            rows.append(r)

    return pd.DataFrame(rows).set_index("sample_id")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, ct_col: str = None, out_dir: Path = Path(".")):
    """
    df: rows=samples, 'diagnosis' 列 (PIH/PIM/PI) + 菌種列
    ct_col: Community Type 列名 (CT I〜IV、あれば)
    """
    print("=== Reference attractors (MAP θ → φ_eq) ===")
    refs = compute_reference_attractors()

    print("\n=== Extracting 5-species φ₀ from S5 ===")
    phi0_mat = extract_5species(df)
    print(f"  {len(df)} samples, shape: {phi0_mat.shape}")

    # --- ODE integration: φ₀ → φ_eq ---
    # --- 各 reference attractor への距離で分類 ---
    # phi0 が 4 attractor のどれに最も近いか → nearest-neighbor in 5D φ-space
    print("\n=== Assigning attractor labels (nearest reference in φ-space) ===")
    ref_keys = [k for k in ["CS", "CH", "DH", "DS"] if k in refs]
    ref_mat = np.array([refs[k] for k in ref_keys])   # (4, 5)

    # L2 距離: (N, 4)
    dists = np.linalg.norm(phi0_mat[:, np.newaxis, :] - ref_mat[np.newaxis, :, :], axis=2)
    nearest_idx = dists.argmin(axis=1)
    attractor_label = np.array([ref_keys[i] for i in nearest_idx])

    # commensal/dysbiotic の2値
    commensal_keys = {"CS", "CH"}
    binary_label = np.array(["commensal" if k in commensal_keys else "dysbiotic"
                              for k in attractor_label])

    # ODE 定常状態は参考用: DH theta で全サンプル積分
    print("  Running ODE (DH theta) for reference...")
    theta_dh = load_map_theta("dh_baseline")
    A_dh, b_dh = theta_to_matrices(theta_dh)
    phi_eq_mat = np.zeros_like(phi0_mat)
    for i, phi0 in enumerate(phi0_mat):
        phi_eq_mat[i] = run_to_steady(phi0, A_dh, b_dh)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} done")

    attractor_label_ode = np.where(phi_eq_mat[:, 4] > phi_eq_mat[:, 0],
                                    "dysbiotic", "commensal")

    # 診断とのクロス集計
    diag = df["diagnosis"].values
    print("\n=== Nearest attractor vs Diagnosis ===")
    for d in DIAGNOSIS_ORDER:
        mask = diag == d
        for k in ref_keys:
            n = (attractor_label[mask] == k).sum()
            if n > 0:
                print(f"  {d} → {k}: {n}")
    print("\n=== Binary (commensal/dysbiotic) vs Diagnosis ===")
    for d in DIAGNOSIS_ORDER:
        mask = diag == d
        n_dys = (binary_label[mask] == "dysbiotic").sum()
        n_com = (binary_label[mask] == "commensal").sum()
        print(f"  {d}: commensal={n_com}, dysbiotic={n_dys}")

    # --- PCA (5D φ₀ のみで fit) ---
    pca = PCA(n_components=2)
    proj0 = pca.fit_transform(phi0_mat)

    # Reference attractor projections
    ref_keys = list(refs.keys())
    ref_proj = pca.transform(np.array([refs[k] for k in ref_keys]))

    # --- Figure: 左=PCA散布図, 右=attractor分類棒グラフ ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左: PCA scatter colored by diagnosis ---
    ax = axes[0]
    for ki, key in enumerate(ref_keys):
        ax.scatter(*ref_proj[ki], marker="*", s=300,
                   color=CONDITION_COLORS[key], zorder=5,
                   edgecolors="k", linewidth=0.8, label=f"{key} attractor")

    color_col = ct_col if ct_col and ct_col in df.columns else "diagnosis"
    color_dict = CT_COLORS if color_col == ct_col else DIAGNOSIS_COLORS
    groups = (["CT I", "CT II", "CT III", "CT IV"]
              if color_col == ct_col else DIAGNOSIS_ORDER)
    for group in groups:
        mask = df[color_col].values == group
        if mask.sum() == 0:
            continue
        ax.scatter(proj0[mask, 0], proj0[mask, 1],
                   alpha=0.7, s=40,
                   color=color_dict.get(group, "gray"),
                   label=group, zorder=3)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Initial state φ₀ (S5 observed)")
    ax.legend(fontsize=8, loc="best", ncol=2)

    # --- 右: 積み上げ棒グラフ (attractor 分類 × 診断) ---
    ax = axes[1]
    diag_labels = [d for d in DIAGNOSIS_ORDER if (diag == d).sum() > 0]
    bar_width = 0.5
    bottom = np.zeros(len(diag_labels))
    for key in ref_keys:
        counts = np.array([(attractor_label[diag == d] == key).sum()
                           for d in diag_labels])
        ax.bar(diag_labels, counts, bar_width,
               bottom=bottom, color=CONDITION_COLORS[key], label=key)
        # パーセント表示
        totals = np.array([(diag == d).sum() for d in diag_labels])
        for xi, (c, t, b) in enumerate(zip(counts, totals, bottom)):
            if c > 0:
                ax.text(xi, b + c / 2, f"{c/t*100:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += counts

    ax.set_ylabel("Number of samples")
    ax.set_title("Nearest attractor assignment by diagnosis")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, bottom.max() * 1.1)

    plt.suptitle("Hamilton ODE Attractor Analysis — Szafrański 2025 BIC cohort",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "attractor_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # --- CSV output ---
    out_df = pd.DataFrame({
        "sample_id": df.index,
        "diagnosis": diag,
        "phi0_So": phi0_mat[:, 0], "phi0_An": phi0_mat[:, 1],
        "phi0_Vd": phi0_mat[:, 2], "phi0_Fn": phi0_mat[:, 3],
        "phi0_Pg": phi0_mat[:, 4],
        "phieq_So": phi_eq_mat[:, 0], "phieq_An": phi_eq_mat[:, 1],
        "phieq_Vd": phi_eq_mat[:, 2], "phieq_Fn": phi_eq_mat[:, 3],
        "phieq_Pg": phi_eq_mat[:, 4],
        "attractor_nearest": attractor_label,
        "attractor_binary": binary_label,
        "attractor_ode": attractor_label_ode,
        "PC1_init": proj0[:, 0], "PC2_init": proj0[:, 1],
    })
    csv_path = out_dir / "attractor_analysis.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return out_df


# ---------------------------------------------------------------------------
# Szafrański 実データ ローダー
# (行=属, 列=サンプル診断名 形式を rows=samples, cols=genus に変換)
# ---------------------------------------------------------------------------
DIAG_MAP = {
    "Health": "PIH",
    "Mucositis": "PIM",
    "Peri-implantitis": "PI",
}

def load_szafranski_wide(path: str) -> pd.DataFrame:
    """
    20260416_mSystems_16S_5genera_all_profiles.xlsx 形式を読み込む。
    行=Genus, 列=サンプル(診断名.N) → 転置して rows=samples, cols=genus。
    Returns DataFrame with 'diagnosis' column + genus columns.
    """
    raw = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path)

    # Total 行を除外
    raw = raw[raw["Genus"] != "Total"].copy()

    # Genus ごとに合計（同属の複数行を統合）
    genus_df = raw.groupby("Genus").sum(numeric_only=True)  # shape: (5, N_samples)

    # 転置: rows=samples, cols=genus
    samples_df = genus_df.T.copy()
    samples_df.index.name = "sample_id"

    # 診断ラベル抽出: "Peri-implantitis.3" → "PI"
    def parse_diag(col_name):
        base = col_name.split(".")[0]
        return DIAG_MAP.get(base, base)

    samples_df["diagnosis"] = [parse_diag(c) for c in samples_df.index]

    return samples_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s5",  type=str, default=None,
                        help="Szafrański S5 CSV/XLSX (species relative abundances)")
    parser.add_argument("--ct",  type=str, default=None,
                        help="Community Type 列名 (CT I-IV)")
    parser.add_argument("--demo", action="store_true",
                        help="Mock data で動作確認")
    parser.add_argument("--out", type=str, default=".",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo or args.s5 is None:
        print("Running with mock data (--demo mode)")
        df = make_mock_s5()
    else:
        # wide format (行=Genus) か通常の CSV かを自動判定
        raw = pd.read_excel(args.s5) if args.s5.endswith(".xlsx") else pd.read_csv(args.s5)
        if "Genus" in raw.columns:
            print("Detected wide format (Genus × samples) — converting...")
            df = load_szafranski_wide(args.s5)
        else:
            df = raw.set_index(raw.columns[0])
            if "diagnosis" not in df.columns:
                raise ValueError("CSV must have a 'diagnosis' column (PIH / PIM / PI)")

    run(df, ct_col=args.ct, out_dir=out_dir)
