"""The window of reversibility: how SOON and how AGGRESSIVELY must you intervene to still rescue a
progressing peri-implant? We sweep (intervention time t_int) x (therapy efficacy = residual dysbiosis
fraction) for the worst-case (most dysbiotic) Duran-Pinedo patient and map the final marginal bone loss.
A clinical "point of no return" contour (>=2 mm beyond the healthy remodelling envelope, here 6 mm
absolute critical) separates RESCUED from LOST. Reach-limited home care (toothbrush, residual ~0.4)
vs professional debridement (residual ~0.15) are marked, showing why deep pockets need the clinician.

Anchors: healthy remodelling <=0.2 mm/yr, pathological progression >1.5 mm yr-1 (Albrektsson; Schwarz
2018 JCP S246); subgingival cleaning efficacy collapses at PD>=5 mm (PMC8327450).
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_reversibility.py
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
WK, NT = 156.0, 1560
CRIT = 2.0     # peri-implantitis diagnostic threshold: ~2 mm bone loss beyond initial remodelling


def Afun():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    lv = np.array([10.0 - r["z_bone_top"] for r in R]); cv = np.array([r["crest_p95"] for r in R])
    return lambda L: float(np.interp(L, lv, cv / cv[0]))


def final_loss(B0, A, t_int_wk, resid):
    dt = WK / NT; I = C = 0.0; L = 0.3
    Ithr, n, kC, gC, kL, lam, gI = 1.0, 8, 0.3, 0.35, 0.02, 1.6, 1.0
    for k in range(NT):
        t = k * dt
        Beff = B0 if t < t_int_wk else B0 * resid
        Cdr = I**n / (I**n + Ithr**n + 1e-12)
        I += (Beff - gI * I) * dt
        C += (kC * Cdr * (1 + lam * (A(L) - 1)) - gC * C) * dt
        L = min(8.0, L + kL * C * dt)
    return L


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy")
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)
    gm = gdi.mean(1); B0 = float(np.max(gm) - np.min(gm))
    A = Afun()

    tmo = np.linspace(0, 30, 60)                 # intervention time (months)
    resid = np.linspace(0.05, 0.85, 60)          # residual dysbiosis after therapy (1 = no effect)
    Z = np.empty((resid.size, tmo.size))
    for i, r in enumerate(resid):
        for j, tm in enumerate(tmo):
            Z[i, j] = final_loss(B0, A, tm * WK / 36.0, r)

    fig, ax = plt.subplots(figsize=use(width_frac=0.62, aspect=0.82))
    pc = ax.pcolormesh(tmo, resid, Z, cmap="RdYlGn_r", shading="auto", vmin=0.3, vmax=3.6)
    cs = ax.contour(tmo, resid, Z, levels=[CRIT], colors="k", linewidths=1.8)
    ax.clabel(cs, fmt={CRIT: "2 mm diagnostic threshold"}, fontsize=6.2)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("final marginal bone loss (mm)", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
    bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.78)
    ax.axhline(0.40, color="#15406e", lw=1.4, ls="--")
    ax.text(0.6, 0.42, "home brushing (reach-limited)", fontsize=6, color="#15406e", va="bottom", bbox=bb)
    ax.axhline(0.15, color="#13502f", lw=1.4, ls="--")
    ax.text(0.6, 0.165, "professional debridement", fontsize=6, color="#13502f", va="bottom", bbox=bb)
    ax.set_xlabel("when therapy starts (months)", fontsize=8)
    ax.set_ylabel("residual dysbiosis after therapy", fontsize=8)
    ax.set_title("Window of reversibility (worst-case patient)", fontsize=8.5)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_reversibility.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_reversibility.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("B0=%.2f  loss(brush@6mo)=%.2f  loss(debride@6mo)=%.2f  loss(debride@24mo)=%.2f"
          % (B0, final_loss(B0, A, 6 * WK / 36, 0.40), final_loss(B0, A, 6 * WK / 36, 0.15),
             final_loss(B0, A, 24 * WK / 36, 0.15)))
    print("wrote", OUT / "fem_periimplantitis_reversibility.pdf")


if __name__ == "__main__":
    main()
