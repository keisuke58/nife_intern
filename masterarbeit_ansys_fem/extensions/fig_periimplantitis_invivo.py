"""Peri-implantitis FEM x REAL IN-VIVO longitudinal data (Dieckow 2024, 10 patients x 3 weeks).
The clinically-validated guild dysbiosis index
   GDI = log(phi_Fuso + phi_Bact + phi_Clos) - log(phi_Bac + phi_Act + phi_Neg)
(Joshi-validated, rho=0.46 vs clinical severity) is computed PER PATIENT over time from the real guild
abundances, then drives the peri-implant resorption rate amplified by the FEM stress-feedback A(loss)
of study (1):
   d(loss)/dt = k * relu(GDI - GDI0) * A(loss).
This yields a PER-PATIENT predicted marginal-bone-loss trajectory and risk ranking -- the in-vivo
counterpart of the in-vitro attractor model, and the mechanical extension of the thesis's per-patient
J detachment-risk ranking. No new FEM solves (reuses pimp_results.jsonl response surface).
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_invivo.py
"""
import sys, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
GUILD_CSV = ROOT / "results" / "dieckow_cr" / "dieckow_guild_abundance.csv"
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
DYS = ["Bacteroidia", "Fusobacteriia", "Clostridia"]
COM = ["Bacilli", "Actinobacteria", "Negativicutes"]
K, MONTHS, NT, PONR = 0.55, 36.0, 720, 6.0


def amplification():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R]); crest = np.array([r["crest_p95"] for r in R])
    return lambda L: np.interp(L, loss, crest / crest[0])


def main():
    use()
    rows = list(csv.DictReader(open(GUILD_CSV)))
    pat = {}
    for r in rows:
        p, tp = r["sample"].split("_")
        dys = sum(float(r[g]) for g in DYS); com = sum(float(r[g]) for g in COM)
        gdi = np.log(dys + 1e-6) - np.log(com + 1e-6)
        pat.setdefault(p, []).append((int(tp), gdi))
    for p in pat:
        pat[p].sort()
    pats = sorted(pat)
    gdi_mean = {p: np.mean([g for _, g in pat[p]]) for p in pats}
    # resorption baseline = healthy-tier reference (40th percentile of patient GDI); patients below it
    # are commensal-dominant and predicted not to progress. NB: this Dieckow cohort is a longitudinal
    # development study, so all GDI<0 (sub-clinical) -- the result is RELATIVE per-patient risk
    # stratification; the absolute disease range comes from the Joshi-validated DI-severity link.
    gdi0 = float(np.percentile(list(gdi_mean.values()), 40))
    order = sorted(pats, key=lambda p: gdi_mean[p])
    cmap = plt.get_cmap("RdYlGn_r")
    col = {p: cmap((gdi_mean[p] - gdi0) / (max(gdi_mean.values()) - gdi0 + 1e-9)) for p in pats}

    A = amplification(); t = np.linspace(0, MONTHS, NT); dt = t[1] - t[0]

    def integrate(gdi):
        L = np.zeros(NT); L[0] = 0.3
        for i in range(1, NT):
            rate = K * max(0.0, gdi - gdi0) * A(L[i - 1])
            L[i] = min(8.0, L[i - 1] + rate / 12.0 * dt)
        return L

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.5))
    # (A) per-patient in-vivo GDI(t)
    wk = {1: 1, 2: 2, 3: 3}
    for p in pats:
        ts = [wk[tp] for tp, _ in pat[p]]; gs = [g for _, g in pat[p]]
        axA.plot(ts, gs, "o-", color=col[p], lw=1.1, ms=3)
    axA.axhline(gdi0, color="0.5", lw=0.7, ls=":")
    axA.text(1.05, gdi0 + 0.1, "healthiest baseline", fontsize=5.6, color="0.4")
    axA.set_xlabel("time (weeks, in-vivo)", fontsize=7); axA.set_xticks([1, 2, 3])
    axA.set_ylabel("guild dysbiosis index GDI", fontsize=7)
    axA.set_title(r"(A) in-vivo per-patient dysbiosis (Dieckow, $N{=}10\times3$wk)", fontsize=7.0)
    axA.set_ylim(-6, 1)   # clip: 3 patients dip to GDI~-18 at wk2 (very commensal transient)
    axA.tick_params(labelsize=6)

    # (B) predicted per-patient bone-loss trajectory
    finals = {}
    for p in pats:
        L = integrate(gdi_mean[p]); finals[p] = L[-1]
        axB.plot(t, L, color=col[p], lw=1.3)
    axB.axhline(PONR, color="0.3", lw=0.9, ls="--")
    axB.text(0.5, PONR + 0.15, "mechanical point of no return (study 1)", fontsize=5.6, color="0.3")
    # label the top-3 and bottom patient
    rank = sorted(pats, key=lambda p: finals[p], reverse=True)
    for p in rank[:3] + rank[-1:]:
        axB.annotate("P-%s" % p, xy=(MONTHS, finals[p]), fontsize=5.8, color=col[p],
                     ha="left", va="center", xytext=(MONTHS + 0.6, finals[p]))
    axB.set_xlabel("clinical time (months, extrapolated)", fontsize=7)
    axB.set_ylabel("predicted marginal bone loss (mm)", fontsize=7)
    axB.set_title(r"(B) predicted per-patient bone-loss trajectory \& ranking", fontsize=7.0)
    axB.set_ylim(0, 8.3); axB.set_xlim(0, MONTHS + 4); axB.tick_params(labelsize=6)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_invivo.pdf", bbox_inches="tight")
    plt.close(fig)
    print("GDI mean per patient:", {p: round(gdi_mean[p], 2) for p in order})
    print("predicted final bone loss (mm):", {p: round(finals[p], 2) for p in rank})
    print("wrote", OUT / "fem_periimplantitis_invivo.pdf")


if __name__ == "__main__":
    main()
