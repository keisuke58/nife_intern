"""Intervention: what tooth-brushing / professional debridement does to the biofilm and the disease.
Adds mechanical biofilm removal to the inflammatory tipping-point cascade. Brushing knocks down the
SUPRAGINGIVAL biofilm but a toothbrush only reaches ~1-2 mm subgingivally, so its efficacy FADES as the
pocket deepens (bone loss grows). Professional debridement reaches the pocket. For a progressing
patient (top Duran-Pinedo dysbiosis) we compare: no care / regular brushing / brushing+debridement, and
brushing started early vs late.
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_brushing.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
DATA = ROOT / "data" / "prjna725874"; FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
DYS, COM = [2, 4, 6], [0, 1, 8]
REACH = 2.0          # toothbrush subgingival reach (mm)
H_BRUSH = 0.80       # brushing removal efficacy on the reachable biofilm
DEB_RES = 0.15       # residual dysbiosis after professional debridement (reaches the pocket)
WK, NT = 156.0, 1560
PONR = 6.0


def Afun():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    lv = np.array([10.0 - r["z_bone_top"] for r in R]); cv = np.array([r["crest_p95"] for r in R])
    cv = cv / cv[0]
    return lambda L: float(np.interp(L, lv, cv))


def run(B0, A, mode, t_start=0.0):
    """mode: none | brush | brush_debride. t_start = weeks before intervention begins."""
    dt = WK / NT; I = C = 0.0; L = 0.3
    Ithr, n, kC, gC, kL, lam, gI = 1.0, 8, 0.3, 0.35, 0.02, 1.6, 1.0
    Ls = []
    for k in range(NT):
        t = k * dt
        reachable = np.clip(1.0 - L / REACH, 0.0, 1.0)        # fraction of biofilm a brush can reach
        if mode == "none" or t < t_start:
            Beff = B0
        elif mode == "brush":
            Beff = B0 * (1.0 - H_BRUSH * reachable)
        else:  # brush + professional debridement (reaches the pocket)
            Beff = min(B0 * (1.0 - H_BRUSH * reachable), B0 * DEB_RES)
        Cdr = I**n / (I**n + Ithr**n + 1e-12)
        I += (Beff - gI * I) * dt
        C += (kC * Cdr * (1 + lam * (A(L) - 1)) - gC * C) * dt
        L = min(8.0, L + kL * C * dt)
        Ls.append(L)
    return np.array(Ls)


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy")
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)
    gm = gdi.mean(1)
    B0 = float(np.max(gm) - np.min(gm))           # the most dysbiotic (progressing) patient
    A = Afun(); t = np.linspace(0, 36, NT)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.5))

    scen = [("none", 0, "#c0392b", "no oral hygiene"),
            ("brush", 0, "#e08a1e", "brushing from the start"),
            ("brush", 78 / 2, "#7b6010", "brushing started late (18 mo)"),
            ("brush_debride", 0, "#2e8b57", "brushing + professional debridement")]
    for mode, ts, col, lab in scen:
        L = run(B0, A, mode, t_start=ts)
        axA.plot(t, L, "-", color=col, lw=2.0, label=lab)
    axA.axhline(PONR, color="0.4", lw=0.8, ls="--")
    axA.text(0.5, PONR + 0.15, "point of no return", fontsize=6, color="0.4")
    axA.set_xlabel("time (months)", fontsize=7.5); axA.set_ylabel("marginal bone loss (mm)", fontsize=7.5)
    axA.set_title("(A) what tooth-brushing does to the disease", fontsize=7.3)
    axA.legend(fontsize=5.8, loc="upper left"); axA.tick_params(labelsize=6); axA.set_ylim(0, 8.3)

    # (B) why: brushing efficacy fades with pocket depth (reach-limited)
    Lp = np.linspace(0, 5, 100)
    reach = np.clip(1.0 - Lp / REACH, 0, 1)
    axB.fill_between(Lp, 0, reach, color="#2e8b57", alpha=0.25)
    axB.plot(Lp, reach, "-", color="#2e8b57", lw=2.2, label="brush reaches (supragingival)")
    axB.plot(Lp, np.full_like(Lp, 1 - DEB_RES), "--", color="#1f5fa6", lw=1.8,
             label="professional debridement")
    axB.axvline(REACH, color="0.5", lw=0.8, ls=":")
    axB.text(REACH + 0.05, 0.5, "brush reach limit\n(~2 mm)", fontsize=6, color="0.4")
    axB.set_xlabel("pocket depth / bone loss (mm)", fontsize=7.5)
    axB.set_ylabel("biofilm fraction removed", fontsize=7.5)
    axB.set_title("(B) why brushing fails once the pocket deepens", fontsize=7.3)
    axB.legend(fontsize=5.8, loc="upper right"); axB.tick_params(labelsize=6); axB.set_ylim(0, 1.05)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_brushing.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_brushing.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("B0=%.2f  final loss: none=%.2f brush0=%.2f brushLate=%.2f debride=%.2f"
          % (B0, run(B0, A, "none")[-1], run(B0, A, "brush")[-1],
             run(B0, A, "brush", 78 / 2)[-1], run(B0, A, "brush_debride")[-1]))
    print("wrote", OUT / "fem_periimplantitis_brushing.pdf")


if __name__ == "__main__":
    main()
