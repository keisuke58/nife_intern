"""Host-modulation therapy: where in the cascade you intervene matters. The same disease can be
attacked at three different nodes -- (1) the BIOFILM (antimicrobial / mechanical, reduces the dysbiotic
drive B), (2) INFLAMMATION (anti-cytokine, e.g. anti-TNF, raises resolution / lowers the cytokine gain),
(3) the OSTEOCLAST (anti-RANKL, e.g. denosumab / OPG-Fc, blocks resorption coupling downstream).

Key teaching point, consistent with the literature: anti-RANKL ARRESTS bone loss even while the biofilm
and inflammation persist (it acts below them) -- a rescue, not a cure; only biofilm control removes the
cause. Combination (biofilm + anti-RANKL) is best.

Anchors: RANKL/OPG elevated in peri-implantitis PICF (Clin Oral Investig 2021, PMID 34264378); anti-RANKL
inhibits alveolar bone destruction in ligature models (PMID 29607937; Valverde 2025 JPR); denosumab
FREEDOM +5% hip BMD / -68% vertebral fracture (Cummings 2009 NEJM); anti-TNF reduces alveolar bone loss
but effect is time-dependent (Kobayashi & Yoshie 2020 Front Immunol).
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_hostmod.py
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


def Afun():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    lv = np.array([10.0 - r["z_bone_top"] for r in R]); cv = np.array([r["crest_p95"] for r in R])
    return lambda L: float(np.interp(L, lv, cv / cv[0]))


def run(B0, A, therapy, t_start=52.0, eff=0.70, track=("I", "C", "L")):
    """therapy in {none, biofilm, antiTNF, antiRANKL, combo}. t_start in weeks. eff = efficacy 0..1."""
    dt = WK / NT; I = C = 0.0; L = 0.3
    Ithr, n, kC, gC, kL, lam, gI = 1.0, 8, 0.3, 0.35, 0.02, 1.6, 1.0
    out = {k: [] for k in track}
    for k in range(NT):
        t = k * dt; on = t >= t_start
        Beff, kCx, gIx = B0, kC, gI
        if on and therapy in ("biofilm", "combo"):
            Beff = B0 * (1 - eff)                       # antimicrobial / mechanical: cut the drive
        if on and therapy == "antiTNF":
            gIx = gI * (1 + 1.8 * eff)                  # anti-cytokine: faster inflammation resolution
        if on and therapy in ("antiRANKL", "combo"):
            kCx = kC * (1 - eff)                         # block RANKL->osteoclast coupling (downstream)
        Cdr = I**n / (I**n + Ithr**n + 1e-12)
        I += (Beff - gIx * I) * dt
        C += (kCx * Cdr * (1 + lam * (A(L) - 1)) - gC * C) * dt
        L = min(8.0, L + kL * C * dt)
        for key, val in (("I", I), ("C", C), ("L", L)):
            if key in out:
                out[key].append(val)
    return {k: np.array(v) for k, v in out.items()}


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy")
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)
    gm = gdi.mean(1); B0 = float(np.max(gm) - np.min(gm))
    A = Afun(); t = np.linspace(0, 36, NT)

    fig, (axL, axI) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.46))
    scen = [("none", "#c0392b", "no therapy"),
            ("antiTNF", "#e08a1e", "anti-cytokine (anti-TNF)"),
            ("antiRANKL", "#7b3f9e", "anti-RANKL (denosumab)"),
            ("biofilm", "#1f5fa6", "biofilm control"),
            ("combo", "#2e8b57", "biofilm + anti-RANKL")]
    for th, col, lab in scen:
        r = run(B0, A, th)
        axL.plot(t, r["L"], "-", color=col, lw=2.0, label=lab)
        axI.plot(t, r["I"], "-", color=col, lw=2.0, label=lab)
    t_start_mo = 52.0 / WK * 36.0          # therapy start (weeks) -> months
    axL.axvline(t_start_mo, color="0.5", lw=0.8, ls=":")
    axL.text(t_start_mo + 0.3, 5.0, "therapy\nstart", fontsize=6, color="0.4")
    axL.axhline(6.0, color="0.4", lw=0.8, ls="--"); axL.text(0.5, 6.2, "critical", fontsize=6, color="0.4")
    axL.set_xlabel("time (months)", fontsize=8); axL.set_ylabel("marginal bone loss (mm)", fontsize=8)
    axL.set_title("(A) bone loss by therapy node", fontsize=8)
    axL.legend(fontsize=5.6, loc="upper left"); axL.tick_params(labelsize=6.5); axL.set_ylim(0, 8.3)

    axI.axvline(t_start_mo, color="0.5", lw=0.8, ls=":")
    axI.set_xlabel("time (months)", fontsize=8); axI.set_ylabel("inflammation $I$ (a.u.)", fontsize=8)
    axI.set_title("(B) anti-RANKL leaves inflammation untouched", fontsize=8)
    axI.tick_params(labelsize=6.5)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_hostmod.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_hostmod.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    fin = {th: run(B0, A, th)["L"][-1] for th, _, _ in scen}
    print("B0=%.2f  final loss:" % B0, {k: round(v, 2) for k, v in fin.items()})
    print("wrote", OUT / "fem_periimplantitis_hostmod.pdf")


if __name__ == "__main__":
    main()
