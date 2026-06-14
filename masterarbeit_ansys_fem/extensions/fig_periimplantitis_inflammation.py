"""Peri-implantitis FEM + HOST INFLAMMATORY RESPONSE: the biologically complete cascade.
Inserts the immune layer between dysbiosis and bone loss --
   dysbiosis B -> inflammation I (cytokines) -> osteoclast C (RANKL/OPG switch) -> bone loss L,
with a mechanical-overload synergy A(loss) from study (1). Coupled ODEs (per-patient, driven by the
real Duran-Pinedo GDI):
   dI/dt = kI*B - gI*I
   Cdrive = I^n/(I^n + Ithr^n)                         # sigmoidal RANKL/OPG threshold switch
   dC/dt = kC*Cdrive*(1 + lam*(A(L)-1)) - gC*C         # osteoclast, amplified by mechanical overload
   dL/dt = kL*C                                        # resorption
The cytokine threshold makes the disease a TIPPING-POINT process (episodic, bistable) -- the clinical
hallmark the direct GDI->loss model lacked. No new FEM solves.
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_inflammation.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
DATA = ROOT / "data" / "prjna725874"
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
DYS, COM = [2, 4, 6], [0, 1, 8]
# biological rate constants (time unit = weeks); illustrative but physiologically ordered
kI, gI = 1.0, 1.0          # inflammation rises with dysbiosis, resolves ~1 wk
n_hill = 8                 # sharp RANKL/OPG switch
kC, gC = 0.20, 0.20        # osteoclast activation / turnover ~5 wk
kL = 0.022                 # mm per (osteoclast.week)
lam = 1.6                  # mechanical-overload synergy
WEEKS, NT = 156.0, 1560    # 36 months


def amplification():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R]); crest = np.array([r["crest_p95"] for r in R])
    return lambda L: float(np.interp(L, loss, crest / crest[0]))


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy"); meta = json.load(open(DATA / "metadata.json"))
    pats = meta["patients"]
    gdi = (np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)).mean(1)
    B = np.maximum(0.0, gdi - gdi.min())          # bacterial dysbiosis drive (>=0)
    Ithr = float(np.percentile(kI * B / gI, 50))  # cytokine threshold at the cohort median
    A = amplification(); t = np.linspace(0, WEEKS, NT); dt = t[1] - t[0]

    def cascade(Bp):
        I = C = 0.0; L = 0.3; Is, Cs, Ls = [], [], []
        for _ in range(NT):
            Cdr = I ** n_hill / (I ** n_hill + Ithr ** n_hill + 1e-12)
            dI = kI * Bp - gI * I
            dC = kC * Cdr * (1 + lam * (A(L) - 1)) - gC * C
            dL = kL * C
            I += dI * dt; C += dC * dt; L = min(8.0, L + dL * dt)
            Is.append(I); Cs.append(C); Ls.append(L)
        return np.array(Is), np.array(Cs), np.array(Ls)

    res = {p: cascade(B[i]) for i, p in enumerate(pats)}
    finalL = {p: res[p][2][-1] for p in pats}
    months = t / (52 / 12)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.5))

    # (A) the cascade for a high- vs low-dysbiosis patient
    rank = sorted(pats, key=lambda p: finalL[p], reverse=True)
    hi, lo = rank[0], rank[-1]
    for p, ls, lab in ((hi, "-", "progressor P%s" % hi), (lo, "--", "stable P%s" % lo)):
        I, C, L = res[p]
        axA.plot(months, I / max(I.max(), 1e-9), ls, color="#c0392b", lw=1.2)
        axA.plot(months, C / max(res[hi][1].max(), 1e-9), ls, color="#e08020", lw=1.2)
        axA.plot(months, L / 8.0, ls, color="#34495e", lw=1.4)
    axA.plot([], [], "-", color="#c0392b", label="inflammation $I$")
    axA.plot([], [], "-", color="#e08020", label="osteoclast $C$")
    axA.plot([], [], "-", color="#34495e", label="bone loss $L$")
    axA.plot([], [], "-", color="0.4", label="progressor (—) vs stable (- -)")
    axA.set_xlabel("clinical time (months)", fontsize=7); axA.set_ylabel("normalised level", fontsize=7)
    axA.set_title(r"(A) dysbiosis $\to$ inflammation $\to$ osteoclast $\to$ bone loss", fontsize=6.7)
    axA.legend(fontsize=5.3, loc="center right"); axA.tick_params(labelsize=6)

    # (B) tipping point: final bone loss vs dysbiosis drive, with the inflammatory threshold
    order = np.argsort(B)
    axB.plot(B[order], [finalL[pats[i]] for i in order], "o-", color="#8B0000", lw=1.0, ms=4)
    Bthr = Ithr * gI / kI
    axB.axvline(Bthr, color="0.4", lw=0.9, ls=":")
    axB.text(Bthr + 0.03, 6.6, "inflammatory\nthreshold", fontsize=5.8, color="0.35")
    axB.axhspan(0, 0.6, color="0.85", alpha=0.5)
    axB.text(B.max() * 0.5, 0.75, "stable (sub-threshold)", fontsize=5.6, color="0.4")
    axB.set_xlabel("dysbiosis drive $B$ (from GDI)", fontsize=7)
    axB.set_ylabel("predicted bone loss at 36 mo (mm)", fontsize=7)
    axB.set_title(r"(B) tipping point: episodic, threshold-gated disease", fontsize=6.7)
    axB.tick_params(labelsize=6); axB.set_ylim(0, 8.3)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_inflammation.pdf", bbox_inches="tight")
    plt.close(fig)
    n_prog = sum(1 for p in pats if finalL[p] > 1.0)
    print("Ithr=%.2f Bthr=%.2f  progressors(>1mm): %d/15" % (Ithr, Bthr, n_prog))
    print("final loss:", {p: round(finalL[p], 2) for p in rank})
    print("wrote", OUT / "fem_periimplantitis_inflammation.pdf")


if __name__ == "__main__":
    main()
