"""Design buys time: coupling the FEM implant-design sweep to the disease-progression model. A wider
implant carries the same occlusal load at lower crestal stress (FEM coupons: crestal sigma_p95 Phi3.5 =
22.2, Phi4.1 = 14.4, Phi4.8 = 10.1 MPa). Because mechanical overload only AGGRAVATES bone loss once
inflammation is present (overload-needs-inflammation synergy: Chambrone 2010 JP; Naert 2012), a lower
crestal stress weakens the mechanical-feedback term and slows the vicious cycle -- so a better mechanical
design BUYS TIME to the clinical bone-loss threshold, though it never substitutes for biofilm control.

Mechanical feedback is scaled by each design's crestal stress relative to the Phi4.1 baseline.
Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_design_time.py
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
WK, NT = 520.0, 5200       # 10-year horizon to capture slower designs
CRIT = 6.0


def Afun():
    R = sorted([json.loads(l) for l in open(FEM / "pimp_results.jsonl")],
               key=lambda r: r["z_bone_top"], reverse=True)
    lv = np.array([10.0 - r["z_bone_top"] for r in R]); cv = np.array([r["crest_p95"] for r in R])
    return lambda L: float(np.interp(L, lv, cv / cv[0]))


def trajectory(B0, A, mech_scale):
    dt = WK / NT; I = C = 0.0; L = 0.3
    Ithr, n, kC, gC, kL, lam, gI = 1.0, 8, 0.3, 0.35, 0.02, 1.6, 1.0
    Ls = []
    for k in range(NT):
        Cdr = I**n / (I**n + Ithr**n + 1e-12)
        I += (B0 - gI * I) * dt
        C += (kC * Cdr * (1 + lam * mech_scale * (A(L) - 1)) - gC * C) * dt
        L = min(8.0, L + kL * C * dt); Ls.append(L)
    return np.array(Ls)


def t_to_crit(L, t):
    idx = np.argmax(L >= CRIT)
    return t[idx] if L[idx] >= CRIT else np.nan


def main():
    use()
    phi = np.load(DATA / "phi_guild.npy")
    gdi = np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)
    gm = gdi.mean(1); B0 = float(np.max(gm) - np.min(gm))
    A = Afun(); t = np.linspace(0, 120, NT)             # months

    cp = {json.loads(l)["tag"]: json.loads(l) for l in open(FEM / "coupon_results.jsonl")}
    base = cp["base"]["crest_p95"]
    designs = [("D35", r"$\varnothing$3.5 mm narrow", "#c0392b"),
               ("base", r"$\varnothing$4.1 mm standard", "#e08a1e"),
               ("D48", r"$\varnothing$4.8 mm wide", "#2e8b57")]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.46))
    rows = []
    for tag, lab, col in designs:
        scale = cp[tag]["crest_p95"] / base
        L = trajectory(B0, A, scale)
        tc = t_to_crit(L, t)
        rows.append((lab, cp[tag]["crest_p95"], tc, col))
        axA.plot(t, L, "-", color=col, lw=2.1, label=lab)
    axA.axhline(CRIT, color="0.4", lw=0.8, ls="--"); axA.text(2, 6.2, "critical", fontsize=6, color="0.4")
    axA.set_xlabel("time (months)", fontsize=8); axA.set_ylabel("marginal bone loss (mm)", fontsize=8)
    axA.set_title("(A) progression by implant design", fontsize=8)
    axA.legend(fontsize=6, loc="lower right"); axA.tick_params(labelsize=6.5)
    axA.set_xlim(0, 120); axA.set_ylim(0, 8.3)

    labs = [r[0].split(" ")[0].replace("$", "").replace(r"\varnothing", "Ø") for r in rows]
    tcs = [r[2] for r in rows]; cols = [r[3] for r in rows]
    bars = axB.bar(range(len(rows)), tcs, color=cols, width=0.6)
    for i, (b, r) in enumerate(zip(bars, rows)):
        h = r[2] if np.isfinite(r[2]) else 0
        axB.text(i, h + 1.5, "%.0f mo" % r[2] if np.isfinite(r[2]) else ">120", ha="center", fontsize=6.5)
    axB.set_xticks(range(len(rows))); axB.set_xticklabels(labs, fontsize=6.5)
    axB.set_ylabel("time to critical bone loss (months)", fontsize=8)
    axB.set_title("(B) wider design buys time", fontsize=8)
    axB.tick_params(labelsize=6.5)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_design_time.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_design_time.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("B0=%.2f  time-to-critical (months):" % B0,
          {r[0].split(" ")[0]: (round(r[2], 1) if np.isfinite(r[2]) else ">120") for r in rows})
    print("wrote", OUT / "fem_periimplantitis_design_time.pdf")


if __name__ == "__main__":
    main()
