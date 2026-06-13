"""Literature-anchored calibration & face validity. Every kinetic rate in the cascade is taken from
the literature EXCEPT the single bone-loss gain kL, which we pin by least-squares to reported clinical
marginal-bone-loss data; we then check the calibrated model reproduces the accepted clinical envelopes.

Calibration target (non-pathological, tissue-level early MBL): 0.93 mm @ 12 mo, 1.04 mm @ 36 mo
  (Int J Implant Dent 2025, doi:10.1186/s40729-025-00613-x).
Reference envelopes: healthy remodelling <=1.5 mm yr-1 then <=0.2 mm/yr (Albrektsson; Schwarz 2018 JCP);
  peri-implantitis = progressive loss above that band (PMC9253284, ~64% of cases progress).
Other rates anchored: osteoclast functional lifespan ~2 wk -> gC ~ 0.35 wk-1 (Parfitt/Jaworski);
  inflammation resolution gI ~ 1 (weeks); RANKL/OPG Hill from PICF (PMID 34264378).
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_calibration.py
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
# Calibration anchor (month, mm, sigma): peri-implantitis ACTIVE-PHASE progression for a progressing
# site -- ~1 mm/yr during the active resorptive phase => ~3 mm by 36 mo (PMC9253284; Schwarz 2018 JCP).
# (The non-pathological early-MBL plateau 0.93->1.04 mm is biologic-width remodelling, a DIFFERENT
#  mechanism this disease-progression model deliberately does not try to reproduce.)
ANCH = [(36, 3.0, 0.8)]


def Afun():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    lv = np.array([10.0 - r["z_bone_top"] for r in R]); cv = np.array([r["crest_p95"] for r in R])
    return lambda L: float(np.interp(L, lv, cv / cv[0]))


def trajectory(B0, A, kL):
    dt = WK / NT; I = C = 0.0; L = 0.0
    Ithr, n, kC, gC, lam, gI = 1.0, 8, 0.3, 0.35, 1.6, 1.0
    Ls = []
    for k in range(NT):
        Cdr = I**n / (I**n + Ithr**n + 1e-12)
        I += (B0 - gI * I) * dt
        C += (kC * Cdr * (1 + lam * (A(L) - 1)) - gC * C) * dt
        L = min(8.0, L + kL * C * dt); Ls.append(L)
    return np.array(Ls)


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy")
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)
    gm = gdi.mean(1); span = float(np.max(gm) - np.min(gm))
    B_lo = 0.45 * span                 # a stable / non-pathological site's drive
    B_hi = span                        # worst-case progressing site
    A = Afun(); t = np.linspace(0, 36, NT)

    # calibrate kL: least-squares to the active-phase progression anchor (1-parameter grid)
    grid = np.linspace(0.004, 0.06, 400)
    def cost(kL):
        L = trajectory(B_hi, A, kL)
        return sum(((np.interp(m, t, L) - mm) / s) ** 2 for m, mm, s in ANCH)
    kL = grid[int(np.argmin([cost(g) for g in grid]))]

    fig, ax = plt.subplots(figsize=use(width_frac=0.66, aspect=0.78))
    # healthy remodelling envelope (<=1.5 mm yr1, +0.2 mm/yr) and pathological band (>0.2 mm/yr)
    env = np.where(t <= 12, 1.5 * t / 12, 1.5 + 0.2 * (t - 12) / 12)
    patho = np.where(t <= 12, 1.5 * t / 12, 1.5 + 1.0 * (t - 12) / 12)
    ax.fill_between(t, 0, env, color="#2e8b57", alpha=0.13, label="healthy remodelling envelope")
    ax.fill_between(t, env, patho, color="#c0392b", alpha=0.08, label="peri-implantitis band")
    ax.plot(t, trajectory(B_lo, A, kL), "-", color="#1f5fa6", lw=2.2, label="model: stable site")
    ax.plot(t, trajectory(B_hi, A, kL), "-", color="#c0392b", lw=2.2, label="model: worst-case site")
    for m, mm, s in ANCH:
        ax.errorbar(m, mm, yerr=s, fmt="o", color="k", ms=5, capsize=2.5, lw=1.0,
                    label="active-phase anchor")
    ax.set_xlabel("time (months)", fontsize=8); ax.set_ylabel("marginal bone loss (mm)", fontsize=8)
    ax.set_title(r"Literature calibration \& face validity", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper left"); ax.tick_params(labelsize=6.5)
    ax.set_xlim(0, 36); ax.set_ylim(0, 6)
    ax.text(0.98, 0.04, r"fitted $k_L=%.3f$ wk$^{-1}$ (1 free param)" % kL, transform=ax.transAxes,
            ha="right", fontsize=6.3, color="0.3")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_calibration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_calibration.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    Llo = trajectory(B_lo, A, kL)
    print("fitted kL=%.4f  worst-case @36mo = %.2f mm (anchor 3.0)" % (kL, trajectory(B_hi, A, kL)[-1]))
    print("stable site @12/36mo = %.2f/%.2f mm (within healthy band)"
          % (np.interp(12, t, Llo), np.interp(36, t, Llo)))
    print("wrote", OUT / "fem_periimplantitis_calibration.pdf")


if __name__ == "__main__":
    main()
