"""Accuracy upgrade of the end-to-end peri-implantitis cascade: literature-anchored parameter PRIORS +
Monte-Carlo UNCERTAINTY propagation + clinical absolute-scale calibration + sensitivity.
Pipeline: real Duran-Pinedo GDI(t) -> dysbiosis B -> inflammation I -> osteoclast C (RANKL/OPG switch)
-> bone loss L, with mechanical-overload synergy A(loss) from study (1).

Upgrades over the deterministic version:
  * priors ordered to physiology: cytokine resolution gI ~ 0.5-2 /wk; osteoclast turnover gC ~ 1/(2-4 wk);
    RANKL/OPG Hill n 4-10; mechanical synergy lam 0.5-3; resorption kL calibrated so an active
    progressor loses ~0.3 mm/yr early (Derks/Fransson peri-implantitis progression).
  * Monte-Carlo (N samples) over priors AND the per-patient GDI timepoint scatter -> median + 90%
    credible band per patient, and P(crossing the mechanical point-of-no-return by 36 mo).
  * global sensitivity (Spearman of each parameter vs 36-mo loss) -> what to measure next.
Pure ODE, no FEM solves.  Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_uq.py
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
N_MC = 600
WEEKS, NT = 156.0, 520
PONR = 6.0
RNG = np.random.default_rng(0)


def amplification():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R]); crest = np.array([r["crest_p95"] for r in R])
    lv, cv = loss, crest / crest[0]
    return lambda L: np.interp(L, lv, cv)


def sample_priors(n):
    """literature-anchored priors (lognormal/uniform); returns dict of (n,) arrays."""
    def ln(center, cv):           # lognormal around center with coeff. of variation
        s = np.sqrt(np.log(1 + cv**2)); return center * np.exp(RNG.normal(0, s, n) - s*s/2)
    return {
        "gI": ln(1.0, 0.4),       # cytokine/inflammation resolution (1/wk)
        "gC": ln(0.35, 0.4),      # osteoclast turnover (1/wk), ~2-3 wk lifespan
        "kC": ln(0.30, 0.4),      # osteoclast activation gain
        "n":  RNG.uniform(4, 10, n),          # RANKL/OPG Hill exponent
        "lam": RNG.uniform(0.5, 3.0, n),      # mechanical-overload synergy
        "kL": ln(0.020, 0.4),     # resorption gain (mm per osteoclast.wk) -- clinical anchor
        "thr_mult": ln(1.0, 0.25),            # threshold multiplier on cohort-median I*
    }


def integrate_ensemble(Bp, P, Ithr0, A, dt):
    """vectorised Euler over the MC ensemble for one patient drive Bp (scalar or (n,))."""
    n = len(P["gI"])
    I = np.zeros(n); C = np.zeros(n); L = np.full(n, 0.3)
    Ithr = Ithr0 * P["thr_mult"]
    for _ in range(NT):
        Cdr = I**P["n"] / (I**P["n"] + Ithr**P["n"] + 1e-12)
        I += (Bp - P["gI"] * I) * dt                          # dI/dt = B - gI*I  (kI=1)
        amp = 1.0 + P["lam"] * (A(L) - 1.0)
        C += (P["kC"] * Cdr * amp - P["gC"] * C) * dt         # osteoclast, mech-amplified
        L = np.minimum(8.0, L + P["kL"] * C * dt)             # resorption
    return L


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy"); meta = json.load(open(DATA / "metadata.json"))
    pats = meta["patients"]
    gdi_t = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)   # (15,7)
    gmean = gdi_t.mean(1); gsd = gdi_t.std(1)
    B = np.maximum(0.0, gmean - gmean.min())
    A = amplification(); dt = WEEKS / NT
    Ithr0 = float(np.percentile(B / 1.0, 50))             # cohort-median steady I* (kI=gI)

    # Monte-Carlo per patient (resample params once; add per-patient GDI scatter)
    P = sample_priors(N_MC)
    finalL = {}; bands = {}
    for i, p in enumerate(pats):
        Bsamp = np.maximum(0.0, RNG.normal(gmean[i], gsd[i] / np.sqrt(7), N_MC) - gmean.min())
        L = integrate_ensemble(Bsamp, P, Ithr0, A, dt)
        finalL[p] = L
        bands[p] = np.percentile(L, [5, 50, 95])
    p_progress = {p: float(np.mean(finalL[p] > PONR)) for p in pats}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=use(width_frac=1.0, aspect=0.36))

    # (A) per-patient 36-mo bone loss: median + 90% credible interval, ranked
    order = sorted(pats, key=lambda p: bands[p][1])
    y = np.arange(len(order))
    med = [bands[p][1] for p in order]
    lo = [bands[p][1] - bands[p][0] for p in order]; hi = [bands[p][2] - bands[p][1] for p in order]
    axA.errorbar(med, y, xerr=[lo, hi], fmt="o", ms=3, color="#8B0000", ecolor="#c08080", capsize=2, lw=0.8)
    axA.axvline(PONR, color="0.4", lw=0.8, ls="--")
    axA.set_yticks(y); axA.set_yticklabels(["P%s" % p for p in order], fontsize=5)
    axA.set_xlabel("36-mo bone loss (mm), median + 90% CI", fontsize=6.5)
    axA.set_title("(A) per-patient prediction with UQ", fontsize=7.0); axA.tick_params(labelsize=6)

    # (B) probability of crossing the point of no return
    pp = [p_progress[p] for p in order]
    axB.barh(y, pp, color=["#c0392b" if v > 0.5 else "#e0a080" if v > 0.1 else "#bbb" for v in pp])
    axB.set_yticks(y); axB.set_yticklabels(["P%s" % p for p in order], fontsize=5)
    axB.set_xlabel(r"P(cross point-of-no-return $\le$36 mo)", fontsize=6.3)
    axB.set_xlim(0, 1); axB.set_title("(B) risk as a probability", fontsize=7.0); axB.tick_params(labelsize=6)

    # (C) global sensitivity: Spearman |rho| of each parameter vs 36-mo loss (pooled high-drive patient)
    hi_p = order[-1]
    Lh = finalL[hi_p]
    from scipy.stats import spearmanr
    keys = ["kL", "kC", "gC", "lam", "n", "gI", "thr_mult"]
    rho = []
    for k in keys:
        r, _ = spearmanr(P[k], Lh); rho.append(abs(r) if np.isfinite(r) else 0.0)
    si = np.argsort(rho)
    axC.barh(np.arange(len(keys)), [rho[j] for j in si], color="#34495e")
    axC.set_yticks(np.arange(len(keys))); axC.set_yticklabels([keys[j] for j in si], fontsize=5.6)
    axC.set_xlabel(r"|Spearman $\rho$| vs 36-mo loss", fontsize=6.3)
    axC.set_title("(C) what drives the uncertainty", fontsize=7.0); axC.tick_params(labelsize=6)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_uq.pdf", bbox_inches="tight")
    plt.close(fig)
    print("median 36-mo loss:", {p: round(bands[p][1], 2) for p in order})
    print("P(progress):", {p: round(p_progress[p], 2) for p in order})
    print("sensitivity |rho|:", {keys[j]: round(rho[j], 2) for j in si[::-1]})
    print("wrote", OUT / "fem_periimplantitis_uq.pdf")


if __name__ == "__main__":
    main()
