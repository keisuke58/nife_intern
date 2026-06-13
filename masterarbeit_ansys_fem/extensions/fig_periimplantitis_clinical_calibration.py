"""Clinical calibration / validation of the dysbiosis-driven bone-loss model against an INDEPENDENT
longitudinal peri-implantitis cohort (Anuntakarun et al. 2025, Int Dent J 75(5):100951; PRJNA1215005;
7 peri-implantitis patients, 16S + clinical at baseline / 3 mo / 6 mo post-treatment).

Upgrades the resorption ODE from illustrative to clinically anchored, honestly:
(A) The guild dysbiosis index GDI = log(dysbiotic) - log(commensal), computed from the cohort's
    measured genus abundances, TRACKS the measured clinical inflammation (probing depth PD, bleeding
    BOP): all fall together as non-surgical + regenerative therapy resolves the disease. -> the model's
    severity proxy is clinically valid (not just an in-vitro construct).
(B) Driving d(loss)/dt = K * relu(GDI(t)-gdi0) * A(loss) with the MEASURED GDI(t): once treatment pushes
    GDI below the dysbiotic threshold the predicted bone loss ARRESTS, reproducing the observed clinical
    resolution; an untreated counterfactual (GDI held at baseline) keeps progressing to the point of no
    return. The clinical pocket scale (diseased PD 6.14 vs healthy 2.77 mm) anchors the loss axis.

Honest limit: radiographic bone-loss in mm is not tabulated per timepoint in this cohort, so this
validates the severity proxy (GDI<->PD/BOP), the direction (treatment -> arrest), and the clinical
scale -- NOT an absolute mm/yr rate (which needs a progression cohort with radiographic timepoints;
flagged as the remaining gap). A(loss) is the bare-implant coupon feedback (ci_results.jsonl).

Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_clinical_calibration.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

# --- measured cohort data (Anuntakarun 2025, peri-implantitis sites; %, mm, BOP sites) ---
MONTHS = np.array([0.0, 3.0, 6.0])
FUSO = np.array([20.99, 8.99, 8.84])                 # dysbiotic anchor (reported at all timepoints)
STREP = np.array([15.61, 11.30, 13.10])              # commensal
HAEMO = np.array([2.0, 19.94, 17.70])                # commensal (low at baseline, rises with health)
PD = np.array([6.14, np.nan, 3.66])                  # probing depth (mm): baseline, (3mo n/a), 6mo
BOP = np.array([5.66, np.nan, 1.71])                 # bleeding on probing (sites)
PD_HEALTHY = 2.77                                    # healthy-implant baseline PD (clinical floor)
PONR = 4.0


def gdi_traj():
    dys = FUSO
    com = STREP + HAEMO
    return np.log(dys) - np.log(com)                 # +ve dysbiotic, -ve commensal


def Afeedback():
    """bare-implant crestal-stress feedback A(loss)=crest(loss)/crest(0) from the coupon sweep."""
    rows = [json.loads(l) for l in open(FEM / "ci_results.jsonl")]
    r = sorted([x for x in rows if x["crown_h"] == 0], key=lambda x: x["expose"])
    L = np.array([x["expose"] for x in r]); c = np.array([x["crest_p95"] for x in r])
    p = np.polyfit(L, c, 1); c0 = c[0]
    return lambda x: max(np.polyval(p, np.clip(x, 0, PONR)), 1e-3) / c0


def integrate(gdi_of_t, gdi0, A, K=3.5, months=36, dt=0.25):
    t = np.arange(0, months + dt, dt); L = np.zeros_like(t)
    for i in range(1, len(t)):
        sev = max(0.0, gdi_of_t(t[i]) - gdi0)
        L[i] = min(PONR + 0.5, L[i - 1] + K * sev * A(L[i - 1]) / 12.0 * dt)
    return t, L


def main():
    use()
    gdi = gdi_traj()
    gdi0 = -0.2                                       # dysbiotic threshold (just below baseline crossing)
    A = Afeedback()
    # treated: measured GDI(t), linearly interpolated then held at the 6-mo (resolved) value
    g_treat = lambda tt: float(np.interp(tt, MONTHS, gdi))
    # untreated counterfactual: GDI frozen at the measured baseline (sustained dysbiosis)
    g_unt = lambda tt: float(gdi[0])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.46))

    # ---------- (A) GDI vs clinical inflammation ----------
    axA.plot(MONTHS, gdi, "o-", color="#d73027", lw=1.6, ms=5, label="GDI (from 16S)")
    axA.axhline(0, color="0.5", lw=0.7, ls=":")
    axA.text(4.2, 0.06, "dysbiotic $>0$", fontsize=5.6, color="0.45")
    axA.set_xlabel("months post-treatment", fontsize=7.5)
    axA.set_ylabel("guild dysbiosis index GDI", fontsize=7.5, color="#d73027")
    axA.tick_params(labelsize=6.5); axA.tick_params(axis="y", colors="#d73027")
    axA.set_title("(A) the dysbiosis index tracks clinical inflammation", fontsize=7.3)
    axC = axA.twinx()
    mp = ~np.isnan(PD)
    axC.plot(MONTHS[mp], PD[mp], "s--", color="#4575b4", lw=1.2, ms=4, label="probing depth (mm)")
    axC.plot(MONTHS[mp], BOP[mp], "^--", color="#1a9850", lw=1.0, ms=4, label="BOP (sites)")
    axC.axhline(PD_HEALTHY, color="#4575b4", lw=0.6, ls=":", alpha=0.6)
    axC.text(0.1, PD_HEALTHY + 0.1, "healthy PD", fontsize=5.2, color="#4575b4")
    axC.set_ylabel("PD (mm) / BOP", fontsize=7.0, color="0.3"); axC.tick_params(labelsize=6.0)
    h1, l1 = axA.get_legend_handles_labels(); h2, l2 = axC.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, fontsize=5.6, loc="center right", framealpha=0.9)

    # ---------- (B) ODE driven by measured GDI(t) ----------
    tt, Lt = integrate(g_treat, gdi0, A)
    tu, Lu = integrate(g_unt, gdi0, A)
    axB.plot(tu, Lu, "-", color="#7a0177", lw=1.6, label="untreated (GDI held at baseline)")
    axB.plot(tt, Lt, "-", color="#1a9850", lw=1.8, label="treated (measured GDI$(t)$)")
    axB.axhline(PONR, color="0.5", lw=0.7, ls=":"); axB.text(0.5, PONR + 0.06, "point of no return", fontsize=5.6, color="0.4")
    axB.axhspan(0, PD[0] - PD_HEALTHY, color="#4575b4", alpha=0.06)
    axB.text(24, (PD[0] - PD_HEALTHY) - 0.25, "clinical pocket-excess scale\n(diseased $-$ healthy PD)",
             fontsize=5.0, color="#4575b4", ha="center")
    axB.annotate("treatment arrests loss\n(matches PD/BOP resolution)", xy=(8, Lt[np.argmin(abs(tt - 8))]),
                 xytext=(13, 1.1), fontsize=5.4, color="#1a9850",
                 arrowprops=dict(arrowstyle="->", color="#1a9850", lw=0.7))
    axB.set_xlabel("months", fontsize=7.5); axB.set_ylabel("predicted marginal bone loss (mm)", fontsize=7.5)
    axB.set_title("(B) measured GDI$(t)$ drives the ODE: treatment arrests loss", fontsize=7.2)
    axB.legend(fontsize=6.0, loc="center left", framealpha=0.9); axB.tick_params(labelsize=6.5)
    axB.set_ylim(0, PONR + 0.6)

    fig.suptitle(r"Clinical validation (Anuntakarun 2025, $N{=}7$ longitudinal): the dysbiosis index "
                 r"tracks measured inflammation and predicts treatment-driven arrest", fontsize=7.4, y=1.0)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_clinical_calibration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_clinical_calibration.png", dpi=200, bbox_inches="tight")
    print("wrote clinical_calibration  GDI %.2f->%.2f->%.2f  PD 6.14->3.66  BOP 5.66->1.71  "
          "treated loss=%.2f untreated=%.2f mm @36mo" % (gdi[0], gdi[1], gdi[2], Lt[-1], Lu[-1]))


if __name__ == "__main__":
    main()
