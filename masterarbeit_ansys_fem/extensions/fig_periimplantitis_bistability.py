"""Why peri-implantitis is hard to reverse: the ecology<->host positive-feedback loop makes the
disease a BISTABLE system (a healthy attractor and a diseased attractor), so it shows HYSTERESIS --
once tipped, lowering the dysbiotic drive back to where you started does NOT recover health.

Mechanism (closes the loop that the one-way cascade left open): dysbiosis B drives host inflammation I
(cytokines, RANKL up / OPG down); but inflammation FEEDS the dysbiotic anaerobes back -- bleeding/GCF
delivers heme-iron and protein that P. gingivalis & co. thrive on ("inflammophilic"/nutritional-immunity
dysbiosis, Hajishengallis keystone-pathogen synergy). Mutual activation -> two stable states.

  B-nullcline:  dB/dt = b0 + a*hill(I) - dB*B = 0
  I-nullcline:  dI/dt = c0 + k*hill(B) - gI*I = 0     hill(x)=x^n/(x^n+K^n)
  readout    :  resorption rate ~ RANKL/OPG(I) -> marginal bone loss

control parameter b0 = baseline dysbiotic colonisation pressure (plaque/hygiene burden).
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_bistability.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
RED, BLUE, GREEN, GREY = "#c0392b", "#1f5fa6", "#2e8b57", "#5d6066"

# mutual-activation parameters (dimensionless; tuned to sit in the bistable regime)
A, KI, N = 0.90, 0.62, 4          # inflammation -> dysbiotic overgrowth gain / half-sat / Hill
DB = 1.0                          # dysbiosis clearance
C0, K, KB, M = 0.02, 1.25, 0.62, 4  # baseline + dysbiosis -> inflammation gain / half-sat / Hill
GI = 1.0                          # inflammation resolution


def hill(x, k, n):
    return x**n / (x**n + k**n + 1e-12)


def deriv(B, I, b0):
    dB = b0 + A * hill(I, KI, N) - DB * B
    dI = C0 + K * hill(B, KB, M) - GI * I
    return dB, dI


def settle(b0, B0, I0, steps=4000, dt=0.02):
    B, I = B0, I0
    for _ in range(steps):
        dB, dI = deriv(B, I, b0)
        B += dB * dt; I += dI * dt
    return B, I


def rankl_opg(I):
    """RANKL/OPG ratio as a saturating function of inflammation; ~1 at health, up to ~5x diseased
    (Duarte 2009 / Rakic 2014 PICF). Resorption rate is proportional to it above the OPG-buffered floor."""
    return 1.0 + 4.0 * hill(I, 0.8, 3)


def main():
    use()
    fig, ax = plt.subplots(1, 3, figsize=use(width_frac=1.0, aspect=0.46))

    # ---- (A) nullclines in the bistable regime: three fixed points ----
    b0_mid = 0.30
    Ig = np.linspace(0, 2.2, 400)
    B_of_I = (b0_mid + A * hill(Ig, KI, N)) / DB           # B-nullcline: B as fn of I
    Bg = np.linspace(0, 2.2, 400)
    I_of_B = (C0 + K * hill(Bg, KB, M)) / GI               # I-nullcline: I as fn of B
    a = ax[0]
    a.plot(B_of_I, Ig, "-", color=BLUE, lw=2.0, label=r"$\dot B=0$ (dysbiosis)")
    a.plot(Bg, I_of_B, "-", color=RED, lw=2.0, label=r"$\dot I=0$ (inflammation)")
    # locate fixed points by settling from a grid of starts
    fps = set()
    for B0 in np.linspace(0, 2, 6):
        for I0 in np.linspace(0, 2, 6):
            Bs, Is = settle(b0_mid, B0, I0)
            fps.add((round(Bs, 2), round(Is, 2)))
    fps = sorted(fps)
    for (Bs, Is) in fps:
        a.plot(Bs, Is, "o", ms=8, mfc=GREEN if Is < 0.5 else RED, mec="k", mew=0.6, zorder=5)
    a.text(0.05, 1.95, "two stable states\n(healthy / diseased)", fontsize=6.5, color="0.3")
    a.set_xlabel("dysbiotic load $B$", fontsize=8); a.set_ylabel("inflammation $I$", fontsize=8)
    a.set_title("(A) bistable phase plane", fontsize=8); a.legend(fontsize=6, loc="lower right")
    a.set_xlim(0, 2.2); a.set_ylim(0, 2.2); a.tick_params(labelsize=6.5)

    # ---- (B) hysteresis: ramp the colonisation pressure b0 up then down ----
    b_up = np.linspace(0.0, 0.55, 220)
    b_dn = b_up[::-1]
    B, I = 0.02, 0.02; up = []
    for b0 in b_up:
        B, I = settle(b0, B, I, steps=600)
        up.append(rankl_opg(I))
    dn = []
    for b0 in b_dn:
        B, I = settle(b0, B, I, steps=600)
        dn.append(rankl_opg(I))
    a = ax[1]
    a.plot(b_up, up, "-", color=GREEN, lw=2.2, label="increasing burden")
    a.plot(b_dn, dn, "-", color=RED, lw=2.2, label="decreasing burden")
    # bistable window = where up and down branches differ
    diff = np.abs(np.array(up) - np.array(dn)[::-1])
    on = b_up[np.argmax(np.array(up) > 3.0)]
    off = b_dn[::-1][np.argmax(np.array(dn)[::-1] > 3.0)]
    a.axvspan(off, on, color="0.85", zorder=0)
    a.text((off + on) / 2, 1.3, "bistable\n(hysteresis)", fontsize=6.3, color="0.35", ha="center")
    a.annotate("", xy=(on, 4.0), xytext=(on, 1.3),
               arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
    a.annotate("", xy=(off, 1.3), xytext=(off, 4.0),
               arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    a.set_xlabel("colonisation / plaque burden $b_0$", fontsize=8)
    a.set_ylabel("RANKL/OPG ratio", fontsize=8)
    a.set_title("(B) prevention $\\ll$ cure (hysteresis)", fontsize=8)
    a.set_ylim(0.8, 5.3); a.legend(fontsize=6, loc="upper left"); a.tick_params(labelsize=6.5)

    # ---- (C) tipping: a transient insult locks the patient into disease ----
    dt, T = 0.02, 120.0; nt = int(T / dt); tt = np.linspace(0, T, nt)
    def trajectory(b0_base, insult, t0, t1):
        B, I = 0.05, 0.05; Ls = []; L = 0.0; kL = 0.06
        for k in range(nt):
            t = k * dt
            b0 = b0_base + (insult if t0 <= t < t1 else 0.0)
            dB, dI = deriv(B, I, b0); B += dB * dt; I += dI * dt
            L += kL * max(0.0, rankl_opg(I) - 1.0) * dt   # bone loss accrues with RANKL/OPG excess
            Ls.append(L)
        return np.array(Ls)
    a = ax[2]
    a.plot(tt, trajectory(0.10, 0.0, 0, 0), "-", color=GREEN, lw=2.0, label="stays healthy")
    a.plot(tt, trajectory(0.10, 0.40, 20, 45), "-", color=RED, lw=2.0,
           label="transient insult\n$\\to$ locked diseased")
    a.axvspan(20, 45, color="#c0392b", alpha=0.12)
    a.text(32, 7.6, "insult", fontsize=6.3, color=RED, ha="center")
    a.set_xlabel("time (months)", fontsize=8); a.set_ylabel("marginal bone loss (mm)", fontsize=8)
    a.set_title("(C) a transient insult is permanent", fontsize=8)
    a.set_ylim(0, 8.5); a.legend(fontsize=6, loc="center left"); a.tick_params(labelsize=6.5)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_bistability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_bistability.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("fixed points (B,I) at b0=%.2f: %s" % (b0_mid, fps))
    print("hysteresis window b0 in [%.3f, %.3f]" % (off, on))
    print("wrote", OUT / "fem_periimplantitis_bistability.pdf")


if __name__ == "__main__":
    main()
