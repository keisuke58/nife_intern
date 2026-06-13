"""Crown-to-implant (C/I) ratio x marginal-bone-loss vicious cycle.

(A) ISO-14801 coupon sweep (FEM/tier2b_real/ci_results.jsonl): crestal peri-implant bone stress vs
marginal bone loss, for a bare abutment vs a crowned implant (+7 mm occlusal lever arm).  The crown
raises the crestal stress ~1.6x at every bone level (the C/I ratio, twin axis, worsens as bone resorbs).

(B) Feeding that mechanical amplification A(loss)=crest(loss)/crest(healthy-bare) into the resorption
ODE  d(loss)/dt = k * severity * A(loss)  (same biofilm severity for both): the crowned implant's
higher crestal stress accelerates resorption and reaches the point-of-no-return sooner -> the crown is
a MECHANICAL accelerator of the biofilm-driven disease.  Pure post-processing; no extra FEM solves.

Run: python masterarbeit_ansys_fem/extensions/fig_crown_ci_cycle.py
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
L_IMPL, HAB = 10.0, 8.0          # coupon implant length / abutment height (mm)
PONR = 4.0                       # point-of-no-return marginal bone loss (mm) — runaway threshold


def ci_ratio(loss, crown_h):
    """crown-to-implant ratio = (load height above crest) / (anchored length in bone)."""
    return (HAB + crown_h + loss) / (L_IMPL - loss)


def load_sweep():
    rows = [json.loads(l) for l in open(FEM / "ci_results.jsonl")]
    out = {}
    for ch in (0, 7):
        r = sorted([x for x in rows if x["crown_h"] == ch], key=lambda x: x["expose"])
        out[ch] = (np.array([x["expose"] for x in r]), np.array([x["crest_p95"] for x in r]))
    return out


def main():
    use()
    sw = load_sweep()
    ref = sw[0][1][0]                                    # healthy bare-abutment crestal stress (loss=0)
    # linear fits crest(loss) -> robust A(loss)=crest/ref for the ODE
    fits = {ch: np.polyfit(sw[ch][0], sw[ch][1], 1) for ch in (0, 7)}
    A = {ch: (lambda L, p=fits[ch]: max(np.polyval(p, np.clip(L, 0, PONR)), 1e-3) / ref) for ch in (0, 7)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.46))

    # ---------- Panel A: crestal stress + C/I ratio vs bone loss ----------
    cols = {0: "#4575b4", 7: "#d73027"}
    lab = {0: "bare abutment", 7: "crowned (+7 mm)"}
    Lg = np.linspace(0, 4, 50)
    for ch in (0, 7):
        ex, cr = sw[ch]
        axA.plot(ex, cr, "o", color=cols[ch], ms=4, zorder=5)
        axA.plot(Lg, np.polyval(fits[ch], Lg), "-", color=cols[ch], lw=1.4, label=lab[ch])
    axA.set_xlabel("marginal bone loss (mm)", fontsize=7.5)
    axA.set_ylabel(r"crestal peri-implant bone $\sigma_\mathrm{vM}$ (MPa)", fontsize=7.5)
    axA.set_title(r"(A) crown raises crestal stress $\sim$1.6$\times$ at every level", fontsize=7.6)
    axA.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    axA.tick_params(labelsize=6.5); axA.set_xlim(-0.2, 4.2)
    axA.annotate(r"$\times$%.2f" % (sw[7][1][-1] / sw[0][1][-1]), xy=(4, sw[7][1][-1]),
                 xytext=(3.0, sw[7][1][-1] + 3), fontsize=7, color="#d73027", fontweight="bold")
    axT = axA.twinx()
    axT.plot(Lg, ci_ratio(Lg, 7), "--", color="#d73027", lw=0.9, alpha=0.6)
    axT.plot(Lg, ci_ratio(Lg, 0), "--", color="#4575b4", lw=0.9, alpha=0.6)
    axT.set_ylabel("C/I ratio (dashed)", fontsize=7.0, color="0.35")
    axT.tick_params(labelsize=6.0, colors="0.35")

    # ---------- Panel B: resorption trajectories, same severity ----------
    def trajectory(ch, sev, k=0.060, months=72, dt=0.1):
        t, L = [0.0], [0.0]
        while t[-1] < months and L[-1] < PONR:
            L.append(L[-1] + dt * k * sev * A[ch](L[-1])); t.append(t[-1] + dt)
        return np.array(t), np.array(L)
    sev = 1.0                                            # one fixed (moderate) biofilm severity, both cases
    for ch in (0, 7):
        t, Lt = trajectory(ch, sev)
        axB.plot(t, Lt, "-", color=cols[ch], lw=1.6, label=lab[ch])
        if Lt[-1] >= PONR:
            axB.plot(t[-1], Lt[-1], "v", color=cols[ch], ms=6)
            axB.annotate("PONR\n%.0f mo" % t[-1], xy=(t[-1], PONR), xytext=(t[-1] - 12, PONR - 0.9),
                         fontsize=6, color=cols[ch], ha="center")
    axB.axhline(PONR, color="0.5", lw=0.7, ls=":")
    axB.text(1, PONR + 0.08, "point of no return", fontsize=6, color="0.4")
    axB.set_xlabel("time (months)", fontsize=7.5)
    axB.set_ylabel("marginal bone loss (mm)", fontsize=7.5)
    axB.set_title("(B) same biofilm severity: the crown accelerates loss", fontsize=7.6)
    axB.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    axB.tick_params(labelsize=6.5); axB.set_ylim(0, PONR + 0.5)

    fig.suptitle(r"Crown-to-implant ratio feedback: the prosthetic moment arm mechanically accelerates "
                 r"biofilm-driven marginal bone loss", fontsize=7.8, y=1.0)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_crown_ci_cycle.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_crown_ci_cycle.png", dpi=200, bbox_inches="tight")
    tb, _ = trajectory(0, sev); tc, _ = trajectory(7, sev)
    print("wrote fem_crown_ci_cycle  PONR: bare=%.0f mo crowned=%.0f mo (%.0f%% sooner)"
          % (tb[-1], tc[-1], 100 * (1 - tc[-1] / tb[-1])))


if __name__ == "__main__":
    main()
