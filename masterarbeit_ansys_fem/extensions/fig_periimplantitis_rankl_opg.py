"""Peri-implantitis disease dynamics, MECHANISTIC host-response upgrade: explicit RANKL/OPG bone-
remodelling (reduced Lemaire 2004 / Pivonka 2008) replacing the lumped osteoclast + ad-hoc Hill switch
of fig_periimplantitis_inflammation.py.

The bone unit is now a coupled osteoclast / osteoblast / bone system whose master switch is the
RANKL : OPG balance -- the actual molecular control of osteoclastogenesis (Boyce & Xing 2007;
Kostenuik 2005). Dysbiosis-driven inflammation up-regulates RANKL and suppresses OPG (TNF-alpha /
IL-1 / IL-17 on the osteoblast lineage); mechanical overload (FEM crest stress amplification A(loss))
adds to RANKL (pathological mechanostat); TGF-beta released by resorbed bone couples OC -> OB
(the "resorption precedes formation" coupling that keeps homeostasis -- and whose failure under a high
RANKL/OPG ratio yields net loss).

State (3 dynamic variables + 1 upstream cytokine driver):
   dI /dt = kI*B - gI*I                                    # inflammation from dysbiosis B (GDI)
   RANKL  = r0*(1 + aI*I + aM*max(0,A(L)-1))               # osteoblastic + inflammatory + mechanical
   OPG    = o0*OB/(1 + bI*I)                               # decoy from mature OB, suppressed by inflammation
   psi    = RANKL/(RANKL + kOPG*OPG + kap)                 # effective RANKL-RANK activation (0..1)
   TGF    = eta*OC                                         # TGF-beta from active resorption
   dOC/dt = Doc*psi/(1 + gT*TGF) - aoc*OC                  # osteoclast: RANKL-driven, TGF-restrained
   dOB/dt = (Dob*TGF/(TGF+Ts)+ob0)/(1+cI*I) - aob*OB       # osteoblast: TGF-coupled, INFLAMMATION-suppressed
   dL /dt = kres*OC - kform*OB                             # bone loss (mm): net resorption

Inflammation BOTH drives resorption (RANKL up, OPG down) AND suppresses formation (TNF/IL-1 inhibit
osteoblast differentiation) -- the bidirectional uncoupling that makes inflammatory bone loss net-
negative, which a resorption-only term cannot capture.

The RANKL:OPG ratio is now an explicit, measurable read-out (PICF biomarker) and the tipping point is
mechanistic (RANKL/OPG competition + mechanical feedback), not an imposed Hill threshold. No new FEM
solves -- A(loss) is reused from the peri-implantitis coupon sweep.

Run: python masterarbeit_ansys_fem/extensions/fig_periimplantitis_rankl_opg.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/nishioka/IKM_Hiwi/nife"); sys.path.insert(0, str(ROOT))
from thesis_style import use  # noqa: E402
DATA = ROOT / "data" / "prjna725874"
FEM = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = ROOT / "masterarbeit_ansys_fem" / "figures"
DYS, COM = [2, 4, 6], [0, 1, 8]                # Duran-Pinedo GUILD_ORDER dysbiotic / commensal indices

# --- rate constants (time unit = weeks); literature-ORDERED, illustrative magnitudes -----------------
# Osteoclast lifespan ~2 wk, osteoblast active life longer; RANKL/OPG is the master regulator.
kI, gI = 1.0, 1.0          # inflammation rise / resolution (~1 wk)
r0, o0 = 1.0, 1.0          # basal RANKL / OPG-per-osteoblast scales
aI, aM = 1.3, 1.6          # inflammation, mechanical overload -> RANKL up (aI/bI clinically calibrated:
bI = 0.9                   # inflammation -> OPG down -- RANKL/OPG fold-change health->disease now ~17x
#                            (median) / ~38x (severe), matching periodontitis GCF (Bostanci 2007 consensus
#                            ~10x, up to ~190x); ratio<->bone-loss r=0.99 vs clinical r=0.81. See *_calib.py
kOPG, kap = 1.0, 0.12      # OPG decoy strength / basal RANK availability
eta = 1.0                  # TGF-beta released per active osteoclast
Doc, aoc, gT = 0.85, 0.5, 1.0   # OC differentiation / clearance (~2 wk) / TGF restraint
Dob, aob, Ts, ob0 = 0.55, 0.2, 0.5, 0.06   # OB coupling / apoptosis / TGF half-sat / basal recruitment
cI = 1.4                   # inflammation suppression of osteoblast formation (uncoupling)
kres, kform = 0.050, 0.027   # bone resorption / formation gains (mm per cell.week), clinical ~1.5 mm/yr active
WEEKS, NT = 156.0, 3120    # 36 months


def amplification():
    """A(loss) = crest peri-implant stress(loss)/crest(0): the FEM mechanical-overload feedback, reused
    from the peri-implantitis coupon bone-level sweep (no new solve). The sweep holds several configs at
    each loss level, so aggregate to the MEAN crest per loss level (a well-defined monotone-ish feedback;
    the raw per-row interp has duplicate-x kinks) and normalise to the intact (loss=0) crest -> A(0)=1."""
    R = [json.loads(l) for l in open(FEM / "pimp_results.jsonl")]
    loss = np.array([10.0 - r["z_bone_top"] for r in R]); crest = np.array([r["crest_p95"] for r in R])
    ul = np.unique(loss); uc = np.array([crest[loss == l].mean() for l in ul]); ratio = uc / uc[0]
    return lambda L: float(np.interp(np.clip(L, ul[0], ul[-1]), ul, ratio))


def integrate(B, A, L0=0.3, tx_week=None, tx_fac=0.0, block_rankl=0.0, ic=None):
    """Integrate the mechanistic remodelling ODE for a constant dysbiosis drive B.
    ic=(I,OC,OB): start state (default = the unstressed seed; pass the B=0 homeostatic steady state so
    progressor curves show OB suppression from baseline, not a transient climb).
    tx_week/tx_fac: optional debridement (B -> B*tx_fac after tx_week).  block_rankl in [0,1]:
    anti-RANKL (denosumab-like) fractional reduction of effective RANKL."""
    t = np.linspace(0, WEEKS, NT); dt = t[1] - t[0]
    I, OC, OB = (0.0, 0.0, ob0 / aob) if ic is None else ic
    L = L0
    out = np.empty((NT, 6))                     # I, RANKL, OPG, OC, OB, L
    for k in range(NT):
        Bk = B * (tx_fac if (tx_week is not None and t[k] >= tx_week) else 1.0)
        RANKL = r0 * (1 + aI * I + aM * max(0.0, A(L) - 1.0)) * (1 - block_rankl)
        OPG = o0 * OB / (1 + bI * I)
        psi = RANKL / (RANKL + kOPG * OPG + kap)
        TGF = eta * OC
        dI = kI * Bk - gI * I
        dOC = Doc * psi / (1 + gT * TGF) - aoc * OC
        dOB = (Dob * TGF / (TGF + Ts) + ob0) / (1 + cI * I) - aob * OB
        dL = kres * OC - kform * OB
        I = max(0.0, I + dI * dt); OC = max(0.0, OC + dOC * dt)
        OB = max(0.0, OB + dOB * dt); L = min(9.0, max(0.0, L + dL * dt))
        out[k] = (I, RANKL, OPG, OC, OB, L)
    return t, out


def load_patients():
    phi = np.load(DATA / "phi_guild.npy"); meta = json.load(open(DATA / "metadata.json"))
    gdi = (np.log(phi[:, :, DYS].sum(2) + 1e-6) - np.log(phi[:, :, COM].sum(2) + 1e-6)).mean(1)
    B = np.maximum(0.0, gdi - gdi.min())
    return meta.get("patients", [f"P{i+1}" for i in range(len(B))]), B, gdi


def main():
    use()
    A = amplification(); pats, B, gdi = load_patients()
    MO = np.linspace(0, 36, NT)                     # months axis (WEEKS = 156 = 36 months)

    # homeostatic seed: the B=0 steady state, so loss starts balanced and progressor OB drops from it
    oh = integrate(0.0, A)[1]; IC0 = (0.0, oh[-1, 3], oh[-1, 4])
    run = lambda b, **kw: integrate(b, A, ic=IC0, **kw)[1]
    ratio = lambda o: o[-1, 1] / o[-1, 2]

    # critical RANKL/OPG: drive at which 36-mo loss first exceeds 0.5 mm, and the ratio there
    bsweep = np.linspace(0, 2.2, 90)
    Lsw = np.array([run(b)[-1, 5] for b in bsweep])
    Rsw = np.array([ratio(run(b)) for b in bsweep])
    bcrit = float(np.interp(0.5, Lsw, bsweep)); rcrit = float(np.interp(bcrit, bsweep, Rsw))

    prog = int(np.argmax(B)); stab = int(np.argmin(B))      # extreme progressor / stable patients
    op = run(B[prog]); os_ = run(B[stab])

    fig, ax = plt.subplots(2, 2, figsize=use(width_frac=1.0, aspect=0.82))
    cR, cC, cB, cL = "#c0392b", "#e67e22", "#2471a3", "#7d3c98"

    # ---- (A) mechanistic cascade: temporal ordering in a progressor --------------------------------
    a = ax[0, 0]
    a.plot(MO, op[:, 0] / op[:, 0].max(), color="#7f8c8d", lw=1.1, label="inflammation $I$")
    a.plot(MO, (op[:, 1] / op[:, 2]) / (op[:, 1] / op[:, 2]).max(), color=cR, lw=1.4, label=r"RANKL/OPG")
    a.plot(MO, op[:, 3] / oh[-1, 3], color=cC, lw=1.2, label="osteoclasts $OC$ (rel. homeo.)")
    a.plot(MO, op[:, 4] / oh[-1, 4], color=cB, lw=1.2, label="osteoblasts $OB$ (rel. homeo.)")
    a.axhline(1.0, color="0.6", ls=":", lw=0.6)
    a.set_ylabel("normalised signal", fontsize=7)
    a.set_xlabel("months", fontsize=7); a.set_xlim(0, 36)
    a2 = a.twinx()
    a2.plot(MO, op[:, 5], color=cL, lw=1.8, label="bone loss $L$")
    a2.fill_between(MO, 0, op[:, 5], color=cL, alpha=0.10)
    a2.set_ylabel("bone loss $L$ (mm)", color=cL, fontsize=7); a2.tick_params(axis="y", labelcolor=cL)
    a.set_title("(A) cascade (progressor): inflammation $\\to$ RANKL/OPG $\\to$ OC$\\uparrow$/OB$\\downarrow$ $\\to$ loss",
                fontsize=6.2, pad=3)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=5.0, loc="center right", framealpha=0.85, labelspacing=0.25)

    # ---- (B) RANKL/OPG tipping point: 36-mo loss & ratio vs dysbiosis drive -------------------------
    b_ = ax[0, 1]
    b_.plot(bsweep, Lsw, color=cL, lw=1.8); b_.fill_between(bsweep, 0, Lsw, color=cL, alpha=0.10)
    b_.axhline(2.0, color="0.4", ls=":", lw=0.8); b_.text(0.02, 2.07, "2 mm diagnostic", fontsize=5, color="0.3")
    b_.axvline(bcrit, color=cR, ls="--", lw=1.0)
    b_.text(bcrit + 0.03, 0.3, "tipping:\n"+r"RANKL/OPG $\approx$ "+"%.1f" % rcrit, fontsize=5.2, color=cR)
    b_.set_xlabel("dysbiosis drive $B$ (relative GDI)", fontsize=7)
    b_.set_ylabel("36-month bone loss (mm)", color=cL, fontsize=7); b_.tick_params(axis="y", labelcolor=cL)
    b2 = b_.twinx()
    b2.plot(bsweep, Rsw, color=cR, lw=1.2, ls="-.")
    b2.set_yscale("log"); b2.set_ylabel(r"final RANKL/OPG", color=cR, fontsize=7)
    b2.tick_params(axis="y", labelcolor=cR)
    b_.set_title("(B) emergent tipping: the RANKL/OPG switch", fontsize=6.2, pad=3)
    b_.set_xlim(0, 2.2)

    # ---- (C) per-patient risk, coloured by the RANKL/OPG biomarker ----------------------------------
    c = ax[1, 0]
    Lp = np.array([run(b)[-1, 5] for b in B])
    Rp = np.array([ratio(run(b)) for b in B])
    order = np.argsort(Lp)[::-1]
    cols = plt.get_cmap("inferno_r")(np.clip(np.log10(Rp[order]) / np.log10(Rp.max() + 1), 0.12, 0.95))
    c.bar(range(len(B)), Lp[order], color=cols, edgecolor="0.3", lw=0.3)
    c.axhline(2.0, color="0.3", ls=":", lw=0.9); c.text(len(B) - 4.5, 2.1, "2 mm", fontsize=5.2, color="0.3")
    c.set_xticks(range(len(B))); c.set_xticklabels([pats[i] for i in order], fontsize=4.2, rotation=90)
    c.set_ylabel("36-month bone loss (mm)", fontsize=7); c.set_xlabel("Duran-Pinedo patient", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap="inferno_r",
                               norm=plt.Normalize(np.log10(Rp.min() + 1), np.log10(Rp.max() + 1)))
    cb = fig.colorbar(sm, ax=c, fraction=0.045, pad=0.02)
    cb.set_label("$\\log_{10}$(RANKL/OPG)", fontsize=5.6); cb.ax.tick_params(labelsize=5)
    c.set_title("(C) per-patient risk; colour = RANKL/OPG biomarker", fontsize=6.2, pad=3)

    # ---- (D) anti-RANKL (denosumab-like): explicit RANKL blockade arrests loss ----------------------
    d = ax[1, 1]
    od0 = run(B[prog])
    od1 = run(B[prog], block_rankl=0.85)                   # near-complete RANKL neutralisation (denosumab-like)
    od2 = run(B[prog], tx_week=26, tx_fac=0.25)            # early source control (debridement) at 6 mo
    d.plot(MO, od0[:, 5], color=cR, lw=1.8, label="untreated (%.1f mm)" % od0[-1, 5])
    d.plot(MO, od1[:, 5], color=cB, lw=1.6, label="anti-RANKL block 85%% (%.1f mm)" % od1[-1, 5])
    d.plot(MO, od2[:, 5], color="#27ae60", lw=1.6, ls="--",
           label="source control @ 6 mo (%.1f mm)" % od2[-1, 5])
    d.axhline(2.0, color="0.4", ls=":", lw=0.8); d.text(0.5, 2.1, "2 mm", fontsize=5, color="0.3")
    d.set_xlabel("months", fontsize=7); d.set_ylabel("bone loss $L$ (mm)", fontsize=7); d.set_xlim(0, 36)
    d.legend(fontsize=5.0, loc="upper left", framealpha=0.85)
    d.set_title("(D) anti-RANKL slows; source control arrests", fontsize=6.2, pad=3)

    for row in ax:
        for axx in row:
            axx.tick_params(labelsize=5.5)
    fig.suptitle("Mechanistic host response: explicit RANKL/OPG bone remodelling (reduced Lemaire/Pivonka)", fontsize=7.2, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.6, h_pad=1.4)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fem_periimplantitis_rankl_opg.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fem_periimplantitis_rankl_opg.png", dpi=200, bbox_inches="tight")
    print("wrote fem_periimplantitis_rankl_opg  bcrit=%.2f RANKL/OPG_crit=%.1f  Lrange %.2f-%.2f  prog>2mm %d/%d"
          % (bcrit, rcrit, Lp.min(), Lp.max(), int(np.sum(Lp > 2)), len(B)))


if __name__ == "__main__":
    main()
