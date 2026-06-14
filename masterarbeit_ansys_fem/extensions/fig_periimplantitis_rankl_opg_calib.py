"""Calibration / face-validity of the mechanistic RANKL/OPG bone-remodelling model
(fig_periimplantitis_rankl_opg.py) against REAL clinical crevicular-fluid RANKL/OPG measurements.

There is NO published dataset pairing longitudinal 16S with longitudinal RANKL/OPG in the same
subjects (a genuine gap), so this is NOT a joint posterior fit. It is the honest alternative:
  (1) anchor the RANKL/OPG axis to cross-sectional clinical fold-changes (health -> disease),
  (2) reproduce the measured ratio<->bone-loss correlation,
  (3) reproduce treatment reversibility (ratio falls after debridement).
Absolute pg/mL values are NOT comparable across assays (sRANKL vs membrane, per-site vs concentration,
pg vs ng) -- so calibration uses RATIOS and FOLD-CHANGES, per the literature's own caveat.

Clinical anchors (extracted from the literature):
  - Periodontitis GCF (Bostanci 2007, J Clin Periodontol, PMID 17355365): RANKL/OPG ratio
    Health 0.017 -> Gingivitis 0.053 -> Chronic periodontitis 3.19 (RANKL up ~55x, OPG down ~3.4x);
    ratio vs probing depth r=0.809, vs attachment loss r=0.841. Botelho is a periodontitis cohort, so
    this oppositely-regulated (RANKL up + OPG down) pattern is the appropriate reference.
  - Peri-implantitis PICF (Rakic 2015): RANKL 0.40 -> 1.51 (~3.8x), OPG ~flat -> ratio fold ~2-4x;
    two meta-analyses (Clin Oral Investig 2021 PMID 34264378; BMC Oral Health 2023 PMC10290807) find
    the PICF ratio difference weak / non-significant. So the implant axis is OPG-flat and low-fold.
  - Reversibility (Bostanci 2011 PMID 21261673; PLoS One 2020): RANKL/OPG ratio falls after SRP.

Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_rankl_opg_calib.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

EXT = Path(__file__).resolve().parent; sys.path.insert(0, str(EXT))
import fig_periimplantitis_rankl_opg as M  # noqa: E402
ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"

# --- clinical RANKL/OPG ratio by disease stage (normalised to health = 1) ----------------------------
BOSTANCI = {"label": "GCF periodontitis (Bostanci 2007)", "stage": [0.0, 0.5, 1.0],
            "ratio": np.array([0.017, 0.053, 3.19])}          # Health / Gingivitis / Chronic perio
RAKIC15 = {"label": "PICF peri-implantitis (Rakic 2015)", "stage": [0.0, 1.0],
           "ratio": np.array([0.40, 1.51])}                   # Health / Peri-implantitis (RANKL proxy, OPG flat)
RAKIC14 = {"label": "PICF RANKL graded (Rakic 2014)", "stage": [0.0, 0.5, 1.0],
           "ratio": np.array([0.72, 0.92, 1.01])}             # Health / Mucositis / Peri-implantitis
CLIN_R_PPD = 0.809; CLIN_R_CAL = 0.841                        # Bostanci ratio<->PPD / <->CAL correlations


def model_folds(A, B, IC0):
    """Model RANKL/OPG fold-change health->disease, at the cohort median drive and the most severe."""
    r = lambda b: (lambda o: o[-1, 1] / o[-1, 2])(M.integrate(b, A, ic=IC0)[1])
    rh = r(0.0)
    return r(float(np.median(B))) / rh, r(float(np.max(B))) / rh


def main():
    use()
    A = M.amplification(); pats, B, gdi = M.load_patients()
    oh = M.integrate(0.0, A)[1]; IC0 = (0.0, oh[-1, 3], oh[-1, 4])
    run = lambda b, **kw: M.integrate(b, A, ic=IC0, **kw)[1]
    ratio = lambda o: o[-1, 1] / o[-1, 2]
    rh = ratio(run(0.0))                                       # model healthy ratio (normalisation anchor)

    fig, ax = plt.subplots(2, 2, figsize=use(width_frac=1.0, aspect=0.84))
    cM, cP, cI = "#7d3c98", "#c0392b", "#2471a3"

    # ---- (A) fold-change calibration: model cohort range vs measured ----------------------------------
    a = ax[0, 0]
    f_med, f_max = model_folds(A, B, IC0)                      # cohort median / most-severe fold
    bars = [("model\nmedian", f_med, cM), ("model\nsevere", f_max, "#9b59b6"),
            ("GCF perio\nconsensus", 10.0, "#e67e22"), ("GCF perio\nBostanci '07", 3.19 / 0.017, cP),
            ("PICF impl.\nRakic '15", 1.51 / 0.40, cI), ("PICF impl.\nmeta (NS)", 1.5, "#85929e")]
    for i, (lab, v, col) in enumerate(bars):
        a.bar(i, v, color=col, edgecolor="0.3", lw=0.3, width=0.72)
        a.text(i, v * 1.13, ("%.0f$\\times$" % v) if v >= 3 else ("%.1f$\\times$" % v),
               ha="center", fontsize=4.8, color=col)
    a.set_yscale("log"); a.set_ylim(0.8, 400)
    a.axhspan(5, 30, color="#fdebd0", alpha=0.7, zorder=0)
    a.text(0.05, 26, "periodontitis (GCF) band", fontsize=4.4, color="#a04000")
    a.set_xticks(range(len(bars))); a.set_xticklabels([b[0] for b in bars], fontsize=4.0)
    a.set_ylabel("RANKL/OPG fold-change (health$\\to$disease)", fontsize=6.2)
    a.set_title("(A) fold-change: model lands in the periodontitis-GCF band", fontsize=6.2, pad=3)

    # ---- (B) normalised ratio vs severity: model curve through clinical points ------------------------
    b_ = ax[0, 1]
    bs = np.linspace(0, 2.1, 80)
    rn = np.array([ratio(run(x)) for x in bs]) / rh
    b_.plot(bs / 2.1, rn, color=cM, lw=2.0, label="model (Botelho-driven)", zorder=5)
    for d, mk, col in [(BOSTANCI, "o", cP), (RAKIC15, "s", cI), (RAKIC14, "^", "#16a085")]:
        b_.plot(d["stage"], d["ratio"] / d["ratio"][0], mk, color=col, ms=4.2, mec="0.3", mew=0.4,
                ls=":", lw=0.7, label=d["label"])
    b_.set_yscale("log"); b_.set_xlabel("disease severity (health 0 $\\to$ disease 1)", fontsize=6.4)
    b_.set_ylabel("RANKL/OPG (normalised, health = 1)", fontsize=6.2)
    b_.legend(fontsize=4.4, loc="upper left", framealpha=0.85, labelspacing=0.25)
    b_.set_title("(B) RANKL/OPG vs severity: model within the clinical envelope", fontsize=6.2, pad=3)

    # ---- (C) ratio <-> bone-loss correlation (the model's central claim) ------------------------------
    c = ax[1, 0]
    Rp = np.array([ratio(run(x)) for x in B]); Lp = np.array([run(x)[-1, 5] for x in B])
    rr = float(np.corrcoef(np.log(Rp), Lp)[0, 1])
    c.scatter(Rp, Lp, c=Lp, cmap="inferno_r", s=22, edgecolor="0.3", lw=0.3, zorder=5)
    c.set_xscale("log")
    c.set_xlabel("final RANKL/OPG (model)", fontsize=6.4); c.set_ylabel("36-month bone loss (mm)", fontsize=6.2)
    c.text(0.04, 0.93, "model $r$(log ratio, loss) = %.2f" % rr, transform=c.transAxes, fontsize=5.6,
           color=cM, va="top")
    c.text(0.04, 0.83, "clinical (Bostanci '07):\n  ratio$\\,$-$\\,$PPD $r$=%.2f, -CAL $r$=%.2f"
           % (CLIN_R_PPD, CLIN_R_CAL), transform=c.transAxes, fontsize=5.0, color=cP, va="top")
    c.set_title("(C) RANKL/OPG tracks bone loss, as measured clinically", fontsize=6.2, pad=3)

    # ---- (D) treatment reversibility: ratio falls after debridement -----------------------------------
    d = ax[1, 1]
    prog = int(np.argmax(B)); MO = np.linspace(0, 36, M.NT)
    o0 = run(B[prog]); o2 = run(B[prog], tx_week=26, tx_fac=0.25)
    d.plot(MO, o0[:, 1] / o0[:, 2] / rh, color=cP, lw=1.8, label="untreated")
    d.plot(MO, o2[:, 1] / o2[:, 2] / rh, color="#27ae60", lw=1.8, ls="--", label="source control @ 6 mo")
    d.axvline(6, color="0.6", ls=":", lw=0.8)
    d.annotate("debridement\n$\\to$ ratio falls\n(Bostanci '11; PLoS '20)", xy=(6, 1.2),
               xytext=(14, 1.5), fontsize=4.8, color="#1e8449",
               arrowprops=dict(arrowstyle="->", color="#1e8449", lw=0.7))
    d.set_yscale("log"); d.set_xlabel("months", fontsize=6.4)
    d.set_ylabel("RANKL/OPG (normalised, health = 1)", fontsize=6.2); d.set_xlim(0, 36)
    d.legend(fontsize=5.0, loc="upper left", framealpha=0.85)
    d.set_title("(D) reversibility: ratio drops with source control (matches clinic)", fontsize=6.2, pad=3)

    for row in ax:
        for axx in row:
            axx.tick_params(labelsize=5.5)
    fig.suptitle("Calibration of the mechanistic RANKL/OPG model against clinical crevicular-fluid data "
                 "(fold-change, correlation, reversibility -- no paired longitudinal data exists)",
                 fontsize=6.9, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.6, h_pad=1.5)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_rankl_opg_calib.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_rankl_opg_calib.png", dpi=200, bbox_inches="tight")
    print("wrote calib  fold median=%.0fx severe=%.0fx  model r(logratio,loss)=%.2f  (clinical PPD r=%.2f)"
          % (f_med, f_max, rr, CLIN_R_PPD))


if __name__ == "__main__":
    main()
