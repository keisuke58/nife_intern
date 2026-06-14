"""Peri-implantitis FEM x REAL IN-VIVO longitudinal DISEASE data (Duran-Pinedo 2021, PRJNA725874:
15 patients x 7 timepoints over 12 weeks, periodontitis subgingival plaque -- a dysbiotic bone-loss
disease analogous to peri-implantitis).  Per-patient guild dysbiosis index
   GDI = log(phi_Bact + phi_Clos + phi_Fuso) - log(phi_Act + phi_Bac + phi_Neg)
is computed from the real 10-guild abundances and drives the resorption rate amplified by the study-(1)
FEM stress feedback A(loss).  Yields per-patient predicted marginal-bone-loss trajectories + ranking.
Longer window + a genuine disease cohort make this the strongest in-vivo driver (vs the Dieckow
development cohort).  No new FEM solves.
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_duranpinedo.py
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
DYS, COM = [2, 4, 6], [0, 1, 8]          # GUILD_ORDER: Bact,Clos,Fuso / Act,Bac,Neg
K, MONTHS, NT, PONR = 0.55, 36.0, 720, 6.0


def amplification():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R]); crest = np.array([r["crest_p95"] for r in R])
    return lambda L: np.interp(L, loss, crest / crest[0])


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy")            # (15,7,10)
    meta = json.load(open(DATA / "metadata.json"))
    weeks = np.array([int(t) for t in meta["timepoints"]])     # 0,2,...,12
    pats = meta["patients"]
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)   # (15,7)
    gmean = gdi.mean(1)
    gdi0 = float(np.percentile(gmean, 40))
    cmap = plt.get_cmap("RdYlGn_r")
    norm = lambda g: (g - gmean.min()) / (gmean.max() - gmean.min() + 1e-9)
    A = amplification(); t = np.linspace(0, MONTHS, NT); dt = t[1] - t[0]

    def integrate(g):
        L = np.zeros(NT); L[0] = 0.3
        for i in range(1, NT):
            L[i] = min(8.0, L[i - 1] + K * max(0.0, g - gdi0) * A(L[i - 1]) / 12.0 * dt)
        return L

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.5))
    for i, p in enumerate(pats):
        axA.plot(weeks, gdi[i], "-", color=cmap(norm(gmean[i])), lw=1.0, alpha=0.9)
    axA.axhline(0.0, color="0.4", lw=0.8, ls="-"); axA.text(0.3, 0.08, "dysbiotic ($>0$)", fontsize=5.6, color="0.4")
    axA.axhline(gdi0, color="0.5", lw=0.7, ls=":")
    axA.set_xlabel("time (weeks, in-vivo)", fontsize=7); axA.set_ylabel("guild dysbiosis index GDI", fontsize=7)
    axA.set_title(r"(A) in-vivo per-patient dysbiosis (Duran-Pinedo, $N{=}15\times12$wk)", fontsize=6.8)
    axA.tick_params(labelsize=6)

    finals = {}
    for i, p in enumerate(pats):
        L = integrate(gmean[i]); finals[p] = L[-1]
        axB.plot(t, L, color=cmap(norm(gmean[i])), lw=1.2)
    axB.axhline(PONR, color="0.3", lw=0.9, ls="--")
    axB.text(0.5, PONR + 0.15, "mechanical point of no return (study 1)", fontsize=5.6, color="0.3")
    rank = sorted(pats, key=lambda p: finals[p], reverse=True)
    for p in rank[:3] + rank[-1:]:
        i = pats.index(p)
        axB.annotate("P%s" % p, xy=(MONTHS, finals[p]), fontsize=5.8, color=cmap(norm(gmean[i])),
                     ha="left", va="center", xytext=(MONTHS + 0.6, finals[p]))
    axB.set_xlabel("clinical time (months, extrapolated)", fontsize=7)
    axB.set_ylabel("predicted marginal bone loss (mm)", fontsize=7)
    axB.set_title(r"(B) predicted per-patient bone-loss trajectory \& ranking", fontsize=6.8)
    axB.set_ylim(0, 8.3); axB.set_xlim(0, MONTHS + 4); axB.tick_params(labelsize=6)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_duranpinedo.pdf", bbox_inches="tight")
    plt.close(fig)
    print("GDI mean:", {p: round(float(gmean[i]), 2) for i, p in enumerate(pats)})
    print("final loss mm:", {p: round(finals[p], 2) for p in rank})
    print("gdi0=%.2f  frac dysbiotic GDI>0: %.2f" % (gdi0, float((gdi > 0).mean())))
    print("wrote", OUT / "fem_periimplantitis_duranpinedo.pdf")


if __name__ == "__main__":
    main()
