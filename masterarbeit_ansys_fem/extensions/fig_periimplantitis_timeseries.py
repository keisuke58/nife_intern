"""Peri-implantitis FEM x REAL longitudinal ecology: the full chain closed with data.
The Bayesian-inferred 5-species trajectories phi_Pg(t) (TMCMC posterior, days 1-21, four community
states CS/CH/DS/DH; Tmcmc202601 fig2 cache) drive a biofilm-severity resorption rate that is amplified
by the FEM stress-feedback A(loss) measured in study (1) (pimp_results.jsonl). Integrating
   d(loss)/dt = k * max(0, phi_Pg - phi0) * A(loss)
gives a PREDICTED per-state marginal-bone-loss trajectory: the dysbiotic-HOBIC state runs away to the
mechanical point-of-no-return, the commensal states stay stable -- the ecology -> mechanics -> disease
loop, entirely from existing data and the existing FEM response (no new solves).

Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_timeseries.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
CACHE = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/docs/paper_figures/_cache")
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
PG = 4                       # species order So/An/Vp/Fn/Pg
STATES = [("CS", "#2166AC", "commensal-static"), ("CH", "#1B7837", "commensal-HOBIC"),
          ("DS", "#DAA520", "dysbiotic-static"), ("DH", "#8B0000", "dysbiotic-HOBIC")]
PHI0 = 0.10                  # commensal Pg baseline below which resorption is negligible
K = 1.9                      # rate constant (mm/yr per unit excess phi_Pg) -- illustrative calibration
MONTHS, NT = 36.0, 720
POINT_OF_NO_RETURN = 6.0     # mm (study 1: stiffness ~halved, crest stress accelerates)


def amplification():
    """A(loss) = crest stress(loss)/crest stress(2mm) from study (1) -- the mechanical vicious cycle."""
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    loss = np.array([10.0 - r["z_bone_top"] for r in R])
    crest = np.array([r["crest_p95"] for r in R])
    A0 = crest[0]
    return lambda L: np.interp(L, loss, crest / A0)


def main():
    use()
    A = amplification()
    t = np.linspace(0, MONTHS, NT); dt = t[1] - t[0]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.5))

    # (A) real phi_Pg(t) over the 21-day observation, with posterior band
    days = np.linspace(1, 21, 2501)
    plateau = {}
    for st, col, lab in STATES:
        d = np.load(CACHE / ("fig2_phi_%s.npz" % st))
        pg = d["phi_map"][:, PG]
        band = np.percentile(d["trajs"][:, :, PG], [16, 84], axis=0)
        axA.plot(days, pg, color=col, lw=1.4, label=st)
        axA.fill_between(days, band[0], band[1], color=col, alpha=0.15)
        plateau[st] = float(np.mean(pg[days >= 15]))          # steady attractor Pg fraction
        plateau[st + "_band"] = np.mean(band[:, days >= 15], axis=1)
    axA.axhline(PHI0, color="0.5", lw=0.7, ls=":")
    axA.text(1.3, PHI0 + 0.02, r"resorption threshold $\phi_0$", fontsize=5.6, color="0.4")
    axA.set_xlabel("time (days, observed)", fontsize=7); axA.set_ylabel(r"$\phi_{\mathrm{Pg}}(t)$", fontsize=7.5)
    axA.set_title(r"(A) inferred ecology: $\phi_\mathrm{Pg}(t)$ (TMCMC posterior)", fontsize=7.3)
    axA.legend(fontsize=6, ncol=2); axA.tick_params(labelsize=6); axA.set_ylim(0, 1)

    # (B) predicted marginal bone loss vs clinical time, driven by each state's phi_Pg plateau
    def integrate(phi):
        L = np.zeros(NT); L[0] = 0.3
        for i in range(1, NT):
            rate = K * max(0.0, phi - PHI0) * A(L[i - 1])      # mm/yr ; /12 -> per month
            L[i] = min(8.0, L[i - 1] + rate / 12.0 * dt)
        return L
    for st, col, lab in STATES:
        L = integrate(plateau[st])
        lo = integrate(plateau[st + "_band"][0]); hi = integrate(plateau[st + "_band"][1])
        axB.plot(t, L, color=col, lw=1.6, label="%s (%s)" % (st, lab))
        axB.fill_between(t, lo, hi, color=col, alpha=0.13)
    axB.axhline(POINT_OF_NO_RETURN, color="0.3", lw=0.9, ls="--")
    axB.text(0.5, POINT_OF_NO_RETURN + 0.15, "mechanical point of no return (study 1)",
             fontsize=5.6, color="0.3")
    axB.set_xlabel("clinical time (months, extrapolated)", fontsize=7)
    axB.set_ylabel("predicted marginal bone loss (mm)", fontsize=7)
    axB.set_title(r"(B) predicted peri-implant bone loss per community state", fontsize=7.3)
    axB.legend(fontsize=5.6, loc="center right"); axB.tick_params(labelsize=6); axB.set_ylim(0, 8.3)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_timeseries.pdf", bbox_inches="tight")
    plt.close(fig)
    print("phi_Pg plateau:", {k: round(v, 3) for k, v in plateau.items() if "band" not in k})
    print("wrote", OUT / "fem_periimplantitis_timeseries.pdf")


if __name__ == "__main__":
    main()
