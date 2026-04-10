#!/usr/bin/env python3
"""
sweep_comets_0d.py
==================
COMETS 0D パラメータスイープ (純 Python / PBS 不要)

COMETS が内部でやっていること:
  各タイムステップで species ごとに LP (FBA) を解き biomass と栄養を更新する。
  ここでは FBA の代わりに Monod 式で同じ動作を近似:

    uptake flux  [mmol/gDW/h]:  r_i = q_max * S/(Km + S)
    growth rate  [gDW/h]:       dX/dt = r_i * Y * X
    nutrient:    [mmol/L/h]:    dS/dt -= r_i * X
    secretion:   [mmol/L/h]:    dP/dt += stoich * r_primary * X

スイープ 1: 初期グルコース濃度  0.02 → 2.0 mM
スイープ 2: 初期 Pg 比率        0.01 → 0.50

Output: nife/comets/pipeline_results/sweep_glucose.png
        nife/comets/pipeline_results/sweep_pg_init.png
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from nife.comets.oral_biofilm import (
    SPECIES, MONOD_PARAMS, MEDIA_HEALTHY, INIT_FRACTIONS,
    TOTAL_INIT_BIOMASS, compute_di,
)

OUTDIR = Path(__file__).parent / "pipeline_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "So": "#2196F3", "An": "#4CAF50", "Vp": "#FF9800",
    "Fn": "#9C27B0", "Pg": "#F44336",
}

# ─────────────────────────────────────────────────────────────────────────────
# Monod ODE solver (COMETS 内部動作の近似)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_monod_0d(
    media_init: dict,
    init_fracs: dict,
    T: float = 80.0,          # simulation hours
    dt: float = 0.02,         # time step [h]
    o2_inhibit_factor: float = 0.05,  # anaerobic growth penalty when O2 present
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Monod-based 0D dFBA simulation.

    COMETS の LP (FBA) は1ステップ=1 Monod 式の評価に相当する。
    cross-feeding (So/An → lactate → Vp/Fn/Pg) が自然に出てくる。

    Returns
    -------
    biomass_df : pd.DataFrame  columns=[cycle, So, An, Vp, Fn, Pg]
    media_df   : pd.DataFrame  columns=[cycle, metabolite, conc_mmol]
    """
    n_steps = int(T / dt)
    sp_list = list(SPECIES.keys())  # ["So", "An", "Vp", "Fn", "Pg"]

    # Initial state
    X = {sp: TOTAL_INIT_BIOMASS * init_fracs.get(sp, 0.0) for sp in sp_list}
    S = dict(media_init)  # mutable copy of media [mM = mmol/L, V=1L]

    o2 = S.get("o2[e]", 0.0)

    # Record arrays
    biomass_hist = {sp: np.zeros(n_steps + 1) for sp in sp_list}
    for sp in sp_list:
        biomass_hist[sp][0] = X[sp]

    media_hist = {}

    for step in range(n_steps):
        dX = {sp: 0.0 for sp in sp_list}
        dS: dict[str, float] = {}

        for sp in sp_list:
            mp = MONOD_PARAMS[sp]
            x = X[sp]
            if x <= 0:
                continue

            # O2 inhibition for strict anaerobes
            o2_factor = 1.0
            if mp["o2_inhibit"] and o2 > 0.05:
                o2_factor = o2_inhibit_factor

            # Compute per-substrate uptake fluxes [mmol/gDW/h]
            sub_fluxes: dict[str, float] = {}
            sub_growth: dict[str, float] = {}
            for met, (q_max, Km, Y) in mp["uptake"].items():
                s = max(S.get(met, 0.0), 0.0)
                flux = q_max * s / (Km + s)  # Monod [mmol/gDW/h]
                sub_fluxes[met] = flux
                sub_growth[met] = flux * Y   # [gDW/gDW/h] = specific growth rate

            # Combine fluxes based on multi mode
            if mp["multi"] == "product":
                # needs ALL substrates; limited by minimum
                mu_sp = min(sub_growth.values()) if sub_growth else 0.0
                # scale individual fluxes proportionally
                if mu_sp > 0 and sub_growth:
                    primary = min(sub_growth, key=sub_growth.get)
                    scale = mu_sp / sub_growth[primary] if sub_growth[primary] > 0 else 0.0
                    sub_fluxes = {m: f * scale for m, f in sub_fluxes.items()}
            else:  # "sum"
                mu_sp = sum(sub_growth.values())

            # Cap at mu_max
            if mu_sp > mp["mu_max"]:
                cap = mp["mu_max"] / mu_sp if mu_sp > 0 else 1.0
                sub_fluxes = {m: f * cap for m, f in sub_fluxes.items()}
                mu_sp = mp["mu_max"]

            mu_sp *= o2_factor
            sub_fluxes = {m: f * o2_factor for m, f in sub_fluxes.items()}

            dX[sp] += mu_sp * x * dt  # [gDW]

            # Nutrient depletion
            for met, flux in sub_fluxes.items():
                consumed = flux * x * dt  # [mmol]
                dS[met] = dS.get(met, 0.0) - consumed

            # Secretion (cross-feeding)
            primary = mp.get("primary_sub")
            if primary and primary in sub_fluxes:
                r_primary = sub_fluxes[primary]
                for sec_met, stoich in mp.get("secretion", {}).items():
                    secreted = stoich * r_primary * x * dt  # [mmol]
                    dS[sec_met] = dS.get(sec_met, 0.0) + secreted

        # Update state
        for sp in sp_list:
            X[sp] = max(X[sp] + dX[sp], 1e-15)
        for met, delta in dS.items():
            S[met] = max(S.get(met, 0.0) + delta, 0.0)
        # O2 is continuously supplied in healthy (simplification)
        if "o2[e]" in media_init and media_init["o2[e]"] > 0:
            S["o2[e]"] = min(S.get("o2[e]", 0.0) + 0.001, media_init["o2[e]"])
        o2 = S.get("o2[e]", 0.0)

        for sp in sp_list:
            biomass_hist[sp][step + 1] = X[sp]

    cycles = np.arange(n_steps + 1)
    bm_df = pd.DataFrame({"cycle": cycles, **{sp: biomass_hist[sp] for sp in sp_list}})
    return bm_df, pd.DataFrame()


def final_di(bm_df: pd.DataFrame) -> float:
    di_df = compute_di(bm_df)
    return float(di_df["DI"].iloc[-1])


def di_series(bm_df: pd.DataFrame) -> np.ndarray:
    di_df = compute_di(bm_df)
    return di_df["DI"].values


# ─────────────────────────────────────────────────────────────────────────────
# Sweep 1: グルコース濃度  →  DI 軌跡
# ─────────────────────────────────────────────────────────────────────────────

def sweep_glucose():
    """
    Sweep initial glucose concentration [glc_D[e]] over log-spaced values.
    Shows how nutrient availability controls community DI.
    """
    glc_values = np.logspace(np.log10(0.01), np.log10(2.0), 8)  # 0.01 → 2.0 mM
    base_fracs = INIT_FRACTIONS["healthy"].copy()
    base_media = dict(MEDIA_HEALTHY)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = cm.get_cmap("viridis", len(glc_values))

    final_dis = []
    for i, glc in enumerate(glc_values):
        media = dict(base_media)
        media["glc_D[e]"] = glc
        bm, _ = simulate_monod_0d(media, base_fracs)
        di = di_series(bm)
        final_dis.append(di[-1])
        color = cmap(i)
        axes[0].plot(di, color=color, lw=1.5, alpha=0.85,
                     label=f"{glc:.3f} mM")

    axes[0].set_xlabel("Time step (× 0.02 h)")
    axes[0].set_ylabel("Dysbiosis Index (DI)")
    axes[0].set_title("DI trajectory\nvs glucose concentration")
    axes[0].set_ylim(0, 1)
    cbar0 = fig.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(glc_values[0], glc_values[-1]), cmap="viridis"),
        ax=axes[0])
    cbar0.set_label("Glucose [mM]")

    # Final DI vs glucose
    axes[1].semilogx(glc_values, final_dis, "o-", color="steelblue", lw=2, ms=7)
    axes[1].set_xlabel("Initial glucose [mM]")
    axes[1].set_ylabel("Final DI")
    axes[1].set_title("Final DI vs glucose\n(lower glucose → higher DI = more dysbiotic)")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color="gray", ls="--", lw=1, label="DI=0.5 threshold")
    axes[1].legend(fontsize=8)

    # Species composition at high vs low glucose
    glc_lo, glc_hi = glc_values[0], glc_values[-1]
    for ax, glc, label in zip([axes[2]], [glc_lo], ["Low glucose (dysbiotic)"]):
        media = dict(base_media)
        media["glc_D[e]"] = glc
        bm, _ = simulate_monod_0d(media, base_fracs)
        sp_cols = [c for c in bm.columns if c in SPECIES]
        total = bm[sp_cols].sum(axis=1)
        bottom = np.zeros(len(bm))
        for sp in sp_cols:
            frac = bm[sp] / total.replace(0, np.nan).fillna(1)
            ax.fill_between(bm["cycle"], bottom, bottom + frac,
                            color=COLORS[sp], alpha=0.8, label=sp)
            bottom += frac
        ax.set_xlabel("Time step")
        ax.set_ylabel("Biomass fraction")
        ax.set_title(f"Species composition\n{label} ({glc:.3f} mM)")
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("Sweep 1: Glucose concentration\n"
                 "COMETS 内部 = 各ステップで Monod FBA → biomass 更新 → 栄養更新",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = OUTDIR / "sweep_glucose.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep 2: 初期 Pg 比率  →  bifurcation 図
# ─────────────────────────────────────────────────────────────────────────────

def sweep_pg_init():
    """
    Sweep initial Pg fraction from 0.01 to 0.50.
    Shows bifurcation: beyond a threshold, Pg dominates and DI jumps.
    """
    pg_values = np.linspace(0.01, 0.50, 12)
    base_fracs_healthy = INIT_FRACTIONS["healthy"].copy()
    base_fracs_diseased = INIT_FRACTIONS["diseased"].copy()

    # Use diseased media (anaerobic, succinate present) to see Pg effect
    base_media_h = dict(MEDIA_HEALTHY)
    base_media_d = {k: v for k, v in {
        **MEDIA_HEALTHY,
        "glc_D[e]": 0.05,    # depleted glucose
        "succ[e]": 0.10,     # succinate for Pg
        "pheme[e]": 0.50,    # hemin for Pg
        "lac_L[e]": 0.20,    # accumulated lactate
    }.items()}
    # Remove O2 for diseased
    base_media_d.pop("o2[e]", None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row, (media, media_label) in enumerate([
        (base_media_h, "Healthy media (aerobic, glucose-rich)"),
        (base_media_d, "Diseased media (anaerobic, glucose-depleted)"),
    ]):
        final_dis = []
        cmap = cm.get_cmap("plasma", len(pg_values))

        for i, pg0 in enumerate(pg_values):
            # Renormalize fracs so total = 1, Pg = pg0
            fracs = dict(base_fracs_healthy)
            rest = 1.0 - pg0
            total_rest = sum(v for k, v in fracs.items() if k != "Pg")
            for sp in fracs:
                if sp != "Pg":
                    fracs[sp] = fracs[sp] / total_rest * rest
            fracs["Pg"] = pg0

            bm, _ = simulate_monod_0d(media, fracs)
            di = di_series(bm)
            final_dis.append(di[-1])
            axes[row, 0].plot(di, color=cmap(i), lw=1.5, alpha=0.85)

        # DI trajectory panel
        axes[row, 0].set_xlabel("Time step (× 0.02 h)")
        axes[row, 0].set_ylabel("DI")
        axes[row, 0].set_title(f"DI trajectories\n{media_label}")
        axes[row, 0].set_ylim(0, 1)
        sm = cm.ScalarMappable(norm=plt.Normalize(pg_values[0], pg_values[-1]), cmap="plasma")
        cbar = fig.colorbar(sm, ax=axes[row, 0])
        cbar.set_label("Initial Pg fraction")

        # Bifurcation panel
        axes[row, 1].plot(pg_values, final_dis, "o-", color="crimson", lw=2, ms=7)
        axes[row, 1].set_xlabel("Initial Pg fraction φ_Pg(0)")
        axes[row, 1].set_ylabel("Final DI")
        axes[row, 1].set_title(f"Final DI vs φ_Pg(0)\n{media_label}")
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].axhline(0.5, color="gray", ls="--", lw=1)

        # Annotate threshold
        threshold_idx = next(
            (i for i, d in enumerate(final_dis) if d > 0.5), None
        )
        if threshold_idx is not None and threshold_idx > 0:
            thresh = pg_values[threshold_idx]
            axes[row, 1].axvline(thresh, color="orange", ls="--", lw=1.5,
                                 label=f"DI>0.5 at φ_Pg={thresh:.2f}")
            axes[row, 1].legend(fontsize=8)

    fig.suptitle(
        "Sweep 2: 初期 Pg 比率 vs 最終 DI\n"
        "COMETS cross-feeding chain: So/An →[lactate]→ Vp/Fn / An →[succinate]→ Pg",
        y=1.01, fontsize=11
    )
    fig.tight_layout()
    out = OUTDIR / "sweep_pg_init.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-feeding visualization (教育用: COMETS の核心)
# ─────────────────────────────────────────────────────────────────────────────

def plot_crossfeeding():
    """
    Show metabolite concentrations over time for healthy and diseased.
    This is exactly what COMETS tracks internally at each cycle.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (cond, media_init, fracs) in enumerate([
        ("Healthy", dict(MEDIA_HEALTHY), INIT_FRACTIONS["healthy"].copy()),
        ("Diseased", {**MEDIA_HEALTHY, "glc_D[e]": 0.05, "succ[e]": 0.10,
                      "pheme[e]": 0.50, "lac_L[e]": 0.20}, INIT_FRACTIONS["diseased"].copy()),
    ]):
        if cond == "Diseased":
            media_init.pop("o2[e]", None)

        bm, _ = simulate_monod_0d(media_init, fracs, T=60)
        sp_cols = [c for c in bm.columns if c in SPECIES]
        cycles = bm["cycle"].values

        # Compute metabolite traces directly by re-running with tracking
        n_steps = len(cycles) - 1
        X = {sp: TOTAL_INIT_BIOMASS * fracs.get(sp, 0.0) for sp in sp_cols}
        S = dict(media_init)
        o2 = S.get("o2[e]", 0.0)
        dt = 0.02

        glc_h, lac_h, succ_h = [S.get("glc_D[e]", 0.0)], [S.get("lac_L[e]", 0.0)], [S.get("succ[e]", 0.0)]

        for step in range(min(n_steps, 3000)):
            dX = {sp: 0.0 for sp in sp_cols}
            dS: dict[str, float] = {}
            for sp in sp_cols:
                mp = MONOD_PARAMS[sp]
                x = X[sp]
                if x <= 0:
                    continue
                o2_factor = 0.05 if mp["o2_inhibit"] and o2 > 0.05 else 1.0
                sub_fluxes = {}
                sub_growth = {}
                for met, (q_max, Km, Y) in mp["uptake"].items():
                    s = max(S.get(met, 0.0), 0.0)
                    flux = q_max * s / (Km + s)
                    sub_fluxes[met] = flux
                    sub_growth[met] = flux * Y
                if mp["multi"] == "product":
                    mu_sp = min(sub_growth.values()) if sub_growth else 0.0
                else:
                    mu_sp = sum(sub_growth.values())
                if mu_sp > mp["mu_max"]:
                    cap = mp["mu_max"] / mu_sp
                    sub_fluxes = {m: f * cap for m, f in sub_fluxes.items()}
                    mu_sp = mp["mu_max"]
                mu_sp *= o2_factor
                sub_fluxes = {m: f * o2_factor for m, f in sub_fluxes.items()}
                dX[sp] += mu_sp * x * dt
                for met, flux in sub_fluxes.items():
                    dS[met] = dS.get(met, 0.0) - flux * x * dt
                primary = mp.get("primary_sub")
                if primary and primary in sub_fluxes:
                    for sec_met, stoich in mp.get("secretion", {}).items():
                        dS[sec_met] = dS.get(sec_met, 0.0) + stoich * sub_fluxes[primary] * x * dt
            for sp in sp_cols:
                X[sp] = max(X[sp] + dX[sp], 1e-15)
            for met, delta in dS.items():
                S[met] = max(S.get(met, 0.0) + delta, 0.0)
            if "o2[e]" in media_init and media_init["o2[e]"] > 0:
                S["o2[e]"] = min(S.get("o2[e]", 0.0) + 0.001, media_init["o2[e]"])
            o2 = S.get("o2[e]", 0.0)
            glc_h.append(S.get("glc_D[e]", 0.0))
            lac_h.append(S.get("lac_L[e]", 0.0))
            succ_h.append(S.get("succ[e]", 0.0))

        t_trim = np.arange(len(glc_h)) * dt

        # Biomass
        ax = axes[row, 0]
        total = bm[sp_cols].sum(axis=1)
        bottom = np.zeros(len(bm))
        for sp in sp_cols:
            frac = bm[sp] / total.replace(0, np.nan).fillna(1)
            ax.fill_between(cycles * dt, bottom, bottom + frac,
                            color=COLORS[sp], alpha=0.8, label=sp)
            bottom += frac
        ax.set_title(f"{cond}: Species fractions")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Biomass fraction")
        ax.legend(loc="upper right", fontsize=7)

        # Metabolites
        ax2 = axes[row, 1]
        ax2.plot(t_trim, glc_h, color="blue", lw=1.5, label="Glucose")
        ax2.plot(t_trim, lac_h, color="green", lw=1.5, label="Lactate")
        ax2.plot(t_trim, succ_h, color="orange", lw=1.5, label="Succinate")
        ax2.set_title(f"{cond}: Metabolite concentrations\n(COMETSが各サイクルで追跡)")
        ax2.set_xlabel("Time [h]")
        ax2.set_ylabel("[mM]")
        ax2.legend(fontsize=8)

        # DI
        di = di_series(bm)
        ax3 = axes[row, 2]
        ax3.plot(cycles * dt, di, color="crimson", lw=2)
        ax3.set_title(f"{cond}: Dysbiosis Index")
        ax3.set_xlabel("Time [h]")
        ax3.set_ylabel("DI")
        ax3.set_ylim(0, 1)
        ax3.axhline(0.5, color="gray", ls="--", lw=1, label="DI=0.5")
        ax3.text(0.05, 0.90, f"Final DI={di[-1]:.2f}", transform=ax3.transAxes, fontsize=9)
        ax3.legend(fontsize=8)

    fig.suptitle(
        "COMETS 内部動作の可視化\n"
        "左: species 比率  |  中: 栄養 (cross-feeding で増減)  |  右: DI\n"
        "Vp は So/An の乳酸を食べる → cross-feeding が commensal community を安定化",
        y=1.02, fontsize=11
    )
    fig.tight_layout()
    out = OUTDIR / "sweep_crossfeeding.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Sobol 感度解析  (SALib の Saltelli sampler)
# 対象パラメータ: 12 個の Monod キネティクス / 初期条件 / 分泌係数
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_with_params(theta: np.ndarray) -> float:
    """
    theta: 12-dim vector (see SOBOL_PROBLEM below).
    Returns final DI (0–1) under diseased-like media.
    """
    (
        So_mu, An_mu, Vp_mu, Fn_mu, Pg_mu,  # 0-4: mu_max [h⁻¹]
        So_Km, Vp_Km, Pg_Km,                 # 5-7: Km for primary substrate [mM]
        glc_init,                             # 8:   initial glucose [mM]
        Pg_frac,                              # 9:   initial Pg fraction
        So_lac_sec,                           # 10:  So lactate secretion stoichiometry
        An_succ_sec,                          # 11:  An succinate secretion stoichiometry
    ) = theta

    # Build modified MONOD_PARAMS (deep copy to avoid mutation)
    from copy import deepcopy
    mp = deepcopy(MONOD_PARAMS)
    mp["So"]["mu_max"] = So_mu
    mp["An"]["mu_max"] = An_mu
    mp["Vp"]["mu_max"] = Vp_mu
    mp["Fn"]["mu_max"] = Fn_mu
    mp["Pg"]["mu_max"] = Pg_mu

    # Km modification: (q_max, Km, Y) → replace Km
    q, _, Y = mp["So"]["uptake"]["glc_D[e]"]
    mp["So"]["uptake"]["glc_D[e]"] = (q, So_Km, Y)
    q, _, Y = mp["Vp"]["uptake"]["lac_L[e]"]
    mp["Vp"]["uptake"]["lac_L[e]"] = (q, Vp_Km, Y)
    q, _, Y = mp["Pg"]["uptake"]["succ[e]"]
    mp["Pg"]["uptake"]["succ[e]"] = (q, Pg_Km, Y)

    # Secretion stoichiometry
    mp["So"]["secretion"]["lac_L[e]"] = So_lac_sec
    mp["An"]["secretion"]["succ[e]"] = An_succ_sec

    # Initial fractions (Pg given, rest proportional to healthy baseline)
    base = INIT_FRACTIONS["healthy"].copy()
    rest = 1.0 - Pg_frac
    total_rest = sum(v for k, v in base.items() if k != "Pg")
    fracs = {sp: (base[sp] / total_rest * rest if sp != "Pg" else Pg_frac)
             for sp in base}

    # Media (diseased-like, use given glc_init)
    media = {
        "glc_D[e]": glc_init,
        "lac_L[e]": 0.05,
        "succ[e]": 0.10,
        "pheme[e]": 0.50,
        "nh4[e]": 10.0, "pi[e]": 10.0, "h2o[e]": 1000.0,
        "ca2[e]": 2.0, "mg2[e]": 1.0,
    }  # anaerobic (no o2) → anaerobes less inhibited

    # Inline Monod ODE (same logic as simulate_monod_0d but uses mp)
    T, dt = 40.0, 0.05
    n_steps = int(T / dt)
    sp_list = list(SPECIES.keys())
    X = {sp: TOTAL_INIT_BIOMASS * fracs.get(sp, 0.0) for sp in sp_list}
    S = dict(media)

    biomass_hist = {sp: np.zeros(n_steps + 1) for sp in sp_list}
    for sp in sp_list:
        biomass_hist[sp][0] = X[sp]

    for step in range(n_steps):
        dX = {sp: 0.0 for sp in sp_list}
        dS: dict[str, float] = {}
        for sp in sp_list:
            p = mp[sp]
            x = X[sp]
            if x <= 0:
                continue
            sub_fluxes, sub_growth = {}, {}
            for met, (q_max, Km, Y) in p["uptake"].items():
                s = max(S.get(met, 0.0), 0.0)
                flux = q_max * s / (Km + s + 1e-15)
                sub_fluxes[met] = flux
                sub_growth[met] = flux * Y
            if p["multi"] == "product":
                mu_sp = min(sub_growth.values()) if sub_growth else 0.0
            else:
                mu_sp = sum(sub_growth.values())
            if mu_sp > p["mu_max"]:
                cap = p["mu_max"] / mu_sp
                sub_fluxes = {m: f * cap for m, f in sub_fluxes.items()}
                mu_sp = p["mu_max"]
            dX[sp] += mu_sp * x * dt
            for met, flux in sub_fluxes.items():
                dS[met] = dS.get(met, 0.0) - flux * x * dt
            primary = p.get("primary_sub")
            if primary and primary in sub_fluxes:
                for sec_met, stoich in p.get("secretion", {}).items():
                    dS[sec_met] = dS.get(sec_met, 0.0) + stoich * sub_fluxes[primary] * x * dt
        for sp in sp_list:
            X[sp] = max(X[sp] + dX[sp], 1e-15)
        for met, delta in dS.items():
            S[met] = max(S.get(met, 0.0) + delta, 0.0)
        for sp in sp_list:
            biomass_hist[sp][step + 1] = X[sp]

    cycles = np.arange(n_steps + 1)
    import pandas as pd
    bm_df = pd.DataFrame({"cycle": cycles, **{sp: biomass_hist[sp] for sp in sp_list}})
    return final_di(bm_df)


# Parameter space definition
SOBOL_PROBLEM = {
    "num_vars": 12,
    "names": [
        "So_mu_max", "An_mu_max", "Vp_mu_max", "Fn_mu_max", "Pg_mu_max",
        "So_Km_glc",  "Vp_Km_lac", "Pg_Km_succ",
        "glc_init",
        "Pg_frac_init",
        "So_lac_sec",
        "An_succ_sec",
    ],
    "bounds": [
        [0.20, 1.00],  # So_mu_max  (nominal 0.50)
        [0.15, 0.70],  # An_mu_max  (nominal 0.35)
        [0.15, 0.80],  # Vp_mu_max  (nominal 0.40)
        [0.12, 0.64],  # Fn_mu_max  (nominal 0.32)
        [0.05, 0.40],  # Pg_mu_max  (nominal 0.20)
        [0.01, 0.20],  # So_Km_glc  (nominal 0.05)
        [0.05, 0.60],  # Vp_Km_lac  (nominal 0.15)
        [0.02, 0.32],  # Pg_Km_succ (nominal 0.08)
        [0.05, 1.00],  # glc_init   (nominal 0.20)
        [0.01, 0.30],  # Pg_frac_init (nominal ~0.02)
        [0.90, 2.70],  # So_lac_sec (nominal 1.80)
        [0.10, 0.60],  # An_succ_sec (nominal 0.30)
    ],
    # Parameter categories for coloring
    "_categories": {
        "mu_max": ["So_mu_max", "An_mu_max", "Vp_mu_max", "Fn_mu_max", "Pg_mu_max"],
        "Km": ["So_Km_glc", "Vp_Km_lac", "Pg_Km_succ"],
        "init": ["glc_init", "Pg_frac_init"],
        "secretion": ["So_lac_sec", "An_succ_sec"],
    },
}

CAT_COLORS = {
    "mu_max":    "#2196F3",  # blue
    "Km":        "#FF9800",  # orange
    "init":      "#4CAF50",  # green
    "secretion": "#9C27B0",  # purple
}


def sobol_sensitivity(N: int = 256):
    """
    Sobol 感度解析 (Saltelli 2010 estimator, SALib).

    N=256 → N×(2D+2) = 256×26 = 6656 simulations
    各シミュレーション T=40h, dt=0.05h → 約30秒 (CPU 1コア)

    Output: pipeline_results/sobol_sensitivity.png
    """
    try:
        try:
            from SALib.sample.sobol import sample as sobol_sample  # SALib ≥ 1.4.5
        except ImportError:
            from SALib.sample.saltelli import sample as sobol_sample  # SALib < 1.4.5
        from SALib.analyze import sobol as sobol_analyze
    except ImportError:
        print("  SALib not found. Run: pip install SALib")
        return

    problem = {k: v for k, v in SOBOL_PROBLEM.items() if not k.startswith("_")}

    print(f"  Generating Saltelli sample (N={N}, D={problem['num_vars']})...")
    param_values = sobol_sample(problem, N, calc_second_order=False)
    n_sims = param_values.shape[0]
    print(f"  Running {n_sims} simulations (T=40h, dt=0.05h)...")

    Y = np.array([_simulate_with_params(row) for row in param_values])

    print("  Computing Sobol indices...")
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    S1   = Si["S1"]
    ST   = Si["ST"]
    S1_ci = Si["S1_conf"]
    ST_ci = Si["ST_conf"]
    names = problem["names"]

    # ── Category color mapping ──
    cats = SOBOL_PROBLEM["_categories"]
    name_to_cat = {n: cat for cat, ns in cats.items() for n in ns}
    colors_bar = [CAT_COLORS[name_to_cat[n]] for n in names]

    # ── Figure: two panels (S1, ST) + scatter S1 vs ST ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(names))
    w = 0.35
    short_names = [n.replace("_mu_max", "\nμmax").replace("_Km_", "\nKm_")
                   .replace("_frac_init", "\nφ₀").replace("_lac_sec", "\nlac_sec")
                   .replace("_succ_sec", "\nsucc_sec").replace("glc_init", "glc\ninit")
                   for n in names]

    # S1 panel
    bars1 = axes[0].bar(x, S1, color=colors_bar, alpha=0.85, width=0.6)
    axes[0].errorbar(x, S1, yerr=S1_ci, fmt="none", color="k", capsize=3, lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=7)
    axes[0].set_ylabel("First-order Sobol index S₁")
    axes[0].set_title("S₁ (主効果)\n各パラメータ単独の DI 分散寄与率")
    axes[0].set_ylim(0, max(0.6, (S1 + S1_ci).max() * 1.15))
    axes[0].axhline(0, color="k", lw=0.5)

    # ST panel
    axes[1].bar(x, ST, color=colors_bar, alpha=0.85, width=0.6)
    axes[1].errorbar(x, ST, yerr=ST_ci, fmt="none", color="k", capsize=3, lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=7)
    axes[1].set_ylabel("Total-order Sobol index Sᵀ")
    axes[1].set_title("Sᵀ (交互作用含む総効果)\n高い → キャリブレーション優先度高")
    axes[1].set_ylim(0, max(0.8, (ST + ST_ci).max() * 1.15))
    axes[1].axhline(0, color="k", lw=0.5)

    # Scatter S1 vs ST
    for i, (n, s1v, stv, col) in enumerate(zip(names, S1, ST, colors_bar)):
        axes[2].scatter(s1v, stv, color=col, s=80, zorder=3)
        axes[2].annotate(short_names[i].replace("\n", " "), (s1v, stv),
                         fontsize=6.5, textcoords="offset points", xytext=(4, 2))
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="S1=ST (no interaction)")
    axes[2].set_xlabel("S₁ (first-order)")
    axes[2].set_ylabel("Sᵀ (total-order)")
    axes[2].set_title("交互作用マップ\nS1≪ST → 他パラメータとの相互作用大")
    axes[2].set_xlim(-0.02, 0.75)
    axes[2].set_ylim(-0.02, 0.85)
    axes[2].legend(fontsize=8)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CAT_COLORS[cat], label=cat)
                       for cat in ["mu_max", "Km", "init", "secretion"]]
    axes[0].legend(handles=legend_elements, fontsize=8, loc="upper right")

    # Print top 5 ranking
    ranking = sorted(zip(names, ST), key=lambda x: x[1], reverse=True)
    print("  === Sobol ST ranking (total-order, anaerobic media) ===")
    for rank, (n, st) in enumerate(ranking[:5], 1):
        print(f"    {rank}. {n:<20} ST={st:.3f}")

    fig.suptitle(
        f"Sobol 感度解析: 最終 DI への各パラメータの寄与率\n"
        f"N={N}, n_sims={n_sims}, 嫌気性培地 (Pg 優勢条件)\n"
        "色: 青=μmax, オレンジ=Km, 緑=初期条件, 紫=分泌係数",
        y=1.02, fontsize=11
    )
    fig.tight_layout()
    out = OUTDIR / "sobol_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sobol-only", action="store_true", help="Run Sobol analysis only")
    ap.add_argument("--sobol-n", type=int, default=256, help="Saltelli N (default 256)")
    ap.add_argument("--all", action="store_true", help="Run all sweeps including Sobol")
    args = ap.parse_args()

    if args.sobol_only:
        print("=== Sobol 感度解析 ===")
        sobol_sensitivity(N=args.sobol_n)
    else:
        print("=== COMETS 0D Parameter Sweep ===")
        print("Sweep 1: glucose concentration...")
        sweep_glucose()
        print("Sweep 2: initial Pg fraction...")
        sweep_pg_init()
        print("Cross-feeding visualization...")
        plot_crossfeeding()
        if args.all:
            print(f"Sobol sensitivity (N={args.sobol_n})...")
            sobol_sensitivity(N=args.sobol_n)
        print("Done. Check pipeline_results/sweep_*.png")
