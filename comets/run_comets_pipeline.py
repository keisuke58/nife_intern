#!/usr/bin/env python3
"""
run_comets_pipeline.py
======================
A → B → C の COMETS パイプライン（順番に実行）

  A: 5種 0D COMETS  healthy vs diseased  (well-mixed, grid 1×1)
  B: 2D 空間 COMETS                      (grid NX×NZ, 拡散あり)
  C: MetaPhlAn init_comp.json → 患者 A_3 初期値で COMETS

Usage:
  python nife/comets/run_comets_pipeline.py          # A のみ
  python nife/comets/run_comets_pipeline.py --all    # A + B + C
  python nife/comets/run_comets_pipeline.py --step B # B のみ
  python nife/comets/run_comets_pipeline.py --step C --init-comp path/to/init_comp.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from nife.comets.oral_biofilm import (
    OralBiofilmComets,
    SPECIES,
    SPECIES_ORDER,
    INIT_FRACTIONS,
    TOTAL_INIT_BIOMASS,
    MEDIA_HEALTHY,
    MEDIA_DISEASED,
    compute_di,
    metabolic_interaction_prior,
)

COLORS = {
    "So": "#2196F3",
    "An": "#4CAF50",
    "Vp": "#FF9800",
    "Fn": "#9C27B0",
    "Pg": "#F44336",
}
SPECIES_LABELS = {
    "So": "S. oralis",
    "An": "A. naeslundii",
    "Vp": "V. parvula",
    "Fn": "F. nucleatum",
    "Pg": "P. gingivalis",
}

OUTDIR = Path(__file__).parent / "pipeline_results"


# ─────────────────────────────────────────────────────────────────────────────
# A: 5種 0D
# ─────────────────────────────────────────────────────────────────────────────

def run_A(max_cycles: int = 500, time_step: float = 0.01, outdir: Path = OUTDIR):
    """5種 0D COMETS: healthy vs diseased."""
    print("\n=== STEP A: 5-species 0D COMETS ===")
    outdir.mkdir(parents=True, exist_ok=True)

    sim = OralBiofilmComets(grid=(1, 1))
    results = {}
    for cond in ("healthy", "diseased"):
        print(f"  Running {cond}...")
        r = sim.run(
            condition=cond,
            max_cycles=max_cycles,
            output_dir=outdir / "comets_0d",
        )
        results[cond] = r
        print(f"  {cond}: {'COMETS' if not getattr(r, '_is_mock', False) else 'mock/cobra'} OK")

    _plot_A(results, max_cycles, time_step, outdir)
    print(f"  → {outdir}/A_0d_comparison.png")
    return results


def _plot_A(results: dict, max_cycles: int, time_step: float, outdir: Path):
    t_unit = "h"
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    for col, cond in enumerate(("healthy", "diseased")):
        r = results[cond]
        bm = r.total_biomass
        sp_cols = [c for c in bm.columns if c in SPECIES]
        cycles = bm["cycle"].values
        t = cycles * time_step

        # Row 0: biomass stack
        ax = fig.add_subplot(gs[0, col])
        bottom = np.zeros(len(t))
        for sp in SPECIES:
            if sp not in sp_cols:
                continue
            vals = bm[sp].values
            ax.fill_between(t, bottom, bottom + vals, color=COLORS[sp],
                            alpha=0.75, label=SPECIES_LABELS[sp])
            bottom += vals
        ax.set_title(f"{cond.upper()}\nBiomass (stacked)", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Time ({t_unit})")
        ax.set_ylabel("Biomass (g)")
        if col == 1:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

        # Row 1: individual species lines
        ax2 = fig.add_subplot(gs[1, col])
        for sp in SPECIES:
            if sp not in sp_cols:
                continue
            ax2.semilogy(t, np.maximum(bm[sp].values, 1e-15),
                         color=COLORS[sp], label=SPECIES_LABELS[sp], lw=1.5)
        ax2.set_title("Species (log scale)")
        ax2.set_xlabel(f"Time ({t_unit})")
        ax2.set_ylabel("Biomass (g)")
        ax2.legend(fontsize=7, ncol=2)

        # Row 2: DI
        ax3 = fig.add_subplot(gs[2, col])
        di_df = compute_di(bm)
        ax3.plot(di_df["cycle"].values * time_step, di_df["DI"].values,
                 color="k", lw=2)
        ax3.axhline(0.5, color="gray", ls="--", lw=1, label="DI=0.5")
        ax3.set_ylim(0, 1)
        ax3.set_title("Dysbiosis Index (DI)")
        ax3.set_xlabel(f"Time ({t_unit})")
        ax3.set_ylabel("DI")
        ax3.legend(fontsize=8)

    fig.suptitle("5-species Oral Biofilm — COMETS 0D (A)", fontsize=13, fontweight="bold")
    fig.savefig(outdir / "A_0d_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# B: 2D 空間 COMETS
# ─────────────────────────────────────────────────────────────────────────────

def run_B(
    nx: int = 10,
    nz: int = 20,
    max_cycles: int = 800,
    time_step: float = 0.1,
    outdir: Path = OUTDIR,
    force_cobra: bool = False,
):
    """
    2D spatial COMETS.
    Grid: NX (lateral) × NZ (depth from bulk to surface).
    z=0: GCF/bulk reservoir (nutrient source)
    z=NZ-1: implant surface (bacteria start here)
    """
    print(f"\n=== STEP B: 2D spatial COMETS ({nx}×{nz} grid) ===")
    outdir.mkdir(parents=True, exist_ok=True)

    sim = OralBiofilmComets(grid=(nx, nz))
    results_spatial = {}

    for cond in ("healthy", "diseased"):
        if force_cobra or sim.comets_home is None:
            if sim.comets_home is None:
                print("  COMETS_HOME not found → running COBRApy dFBA (0D fallback)", flush=True)
            else:
                print("  force_cobra=True → running COBRApy dFBA (0D fallback)", flush=True)
            r = sim.run(condition=cond, max_cycles=max_cycles, output_dir=outdir / "comets_0d")
            results_spatial[cond] = {"total_biomass": r.total_biomass, "exp": None, "model_ids": {}, "log_rate": 1}
            continue

        import cometspy as c
        import cobra

        print(f"  Building spatial layout: {cond}...")
        layout = c.layout()
        layout.grid = [nx, nz]

        # Diffusion coefficients (cm²/s × COMETS scale factor)
        diff_coeffs = {
            "glc_D[e]": 6e-6,
            "o2[e]": 2.1e-5,
            "lac_L[e]": 6e-6,
            "succ[e]": 5e-6,
            "pheme[e]": 1e-6,
            "nh4[e]": 1.9e-5,
            "pi[e]": 8e-6,
            "h2o[e]": 1e-4,
            "ca2[e]": 7e-6,
            "mg2[e]": 7e-6,
        }

        media = MEDIA_HEALTHY if cond == "healthy" else MEDIA_DISEASED

        # Set media at bulk (z=0 row) as Dirichlet source
        for met, val in media.items():
            layout.set_specific_metabolite(met, float(val))

        # Set diffusion rates
        for met, dcoeff in diff_coeffs.items():
            if met in media:
                layout.set_specific_metabolite_diffusion(met, dcoeff)

        # Add species: place biomass near surface (z = NZ-1)
        fracs = INIT_FRACTIONS[cond]
        model_ids = {}
        for sp_key, frac in fracs.items():
            cobra_model = sim._load_agora_model(sp_key, cond)
            sp_model = c.model(cobra_model)
            model_ids[sp_key] = cobra_model.id
            # Distribute along surface row at z=NZ-1, spread across x
            pop_list = []
            for xi in range(nx):
                pop_list.append([xi, nz - 1, TOTAL_INIT_BIOMASS * frac / nx])
            sp_model.initial_pop = pop_list
            sp_model.obj_style = "MAX_OBJECTIVE_MIN_TOTAL"
            sp_model.change_optimizer("GLOP")
            layout.add_model(sp_model)

        log_rate = max(1, max_cycles // 4)
        params = sim.build_params(
            max_cycles=max_cycles,
            time_step=time_step,
            write_media_log=True,
            write_biomass_log=True,
            biomass_log_rate=log_rate,
        )
        params.set_param("numRunThreads", 4)

        run_dir = outdir / "comets_2d" / f"spatial_{cond}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Running spatial COMETS {cond}...")
        try:
            exp = c.comets(layout, params, relative_dir=str(run_dir.relative_to(Path.cwd())) + "/")
            exp.run(delete_files=False)
            bm = sim._rename_biomass_cols(exp.total_biomass)
            results_spatial[cond] = {"total_biomass": bm, "exp": exp, "model_ids": model_ids, "log_rate": log_rate}
            print(f"  {cond}: COMETS spatial OK, cycles={len(bm)}")
        except Exception as e:
            print(f"  WARNING: spatial COMETS failed ({e}), using 0D fallback")
            r = sim.run(condition=cond, max_cycles=max_cycles, output_dir=outdir / "comets_0d")
            results_spatial[cond] = {"total_biomass": r.total_biomass, "exp": None, "model_ids": model_ids, "log_rate": log_rate}

    _plot_B(results_spatial, nx, nz, time_step, outdir, max_cycles)
    print(f"  → {outdir}/B_spatial_comparison.png")
    return results_spatial


def run_B_sweep_o2(
    o2_values: list[float],
    nx: int = 10,
    nz: int = 20,
    max_cycles: int = 800,
    time_step: float = 0.1,
    outdir: Path = OUTDIR,
    force_cobra: bool = False,
):
    print(f"\n=== STEP B-sweep: o2 sweep ({len(o2_values)} points) ===")
    outdir.mkdir(parents=True, exist_ok=True)

    sim = OralBiofilmComets(grid=(nx, nz))

    def _get_last_di(bm):
        di_df = compute_di(bm)
        if isinstance(di_df, dict):
            return float(np.asarray(di_df["DI"])[-1])
        return float(di_df["DI"].iloc[-1])

    def _get_last_phi_pg(bm):
        if "Pg" not in bm.columns:
            return 0.0
        sp_cols = [sp for sp in SPECIES if sp in bm.columns]
        denom = float(bm[sp_cols].iloc[-1].sum()) if sp_cols else 0.0
        return float(bm["Pg"].iloc[-1] / denom) if denom > 0 else 0.0

    sweep_out = outdir / "comets_2d_sweep_o2"
    sweep_out.mkdir(parents=True, exist_ok=True)

    summary = {"nx": nx, "nz": nz, "max_cycles": max_cycles, "time_step": time_step, "points": []}

    for o2 in o2_values:
        print(f"  o2={o2} ...")
        point = {"o2": float(o2), "healthy": {}, "diseased": {}}

        for cond in ("healthy", "diseased"):
            if force_cobra or sim.comets_home is None:
                bm, _ = sim.run_dfba_cobra(
                    condition=cond,
                    max_cycles=max_cycles,
                    time_step=time_step,
                    media_override={"o2[e]": float(o2)},
                )
                point[cond] = {
                    "di_last": _get_last_di(bm),
                    "phi_pg_last": _get_last_phi_pg(bm),
                    "biomass_total_last": float(bm[[sp for sp in SPECIES if sp in bm.columns]].iloc[-1].sum()),
                }
                continue

            import cometspy as c
            import cobra

            layout = c.layout()
            layout.grid = [nx, nz]

            diff_coeffs = {
                "glc_D[e]": 6e-6,
                "o2[e]": 2.1e-5,
                "lac_L[e]": 6e-6,
                "succ[e]": 5e-6,
                "pheme[e]": 1e-6,
                "nh4[e]": 1.9e-5,
                "pi[e]": 8e-6,
                "h2o[e]": 1e-4,
                "ca2[e]": 7e-6,
                "mg2[e]": 7e-6,
            }

            media = dict(MEDIA_HEALTHY if cond == "healthy" else MEDIA_DISEASED)
            media["o2[e]"] = float(o2)

            for met, val in media.items():
                layout.set_specific_metabolite(met, float(val))
            for met, dcoeff in diff_coeffs.items():
                if met in media:
                    layout.set_specific_metabolite_diffusion(met, dcoeff)

            fracs = INIT_FRACTIONS[cond]
            model_ids = {}
            for sp_key, frac in fracs.items():
                cobra_model = sim._load_agora_model(sp_key, cond)
                sp_model = c.model(cobra_model)
                model_ids[sp_key] = cobra_model.id
                pop_list = []
                for xi in range(nx):
                    pop_list.append([xi, nz - 1, TOTAL_INIT_BIOMASS * frac / nx])
                sp_model.initial_pop = pop_list
                sp_model.obj_style = "MAX_OBJECTIVE_MIN_TOTAL"
                sp_model.change_optimizer("GLOP")
                layout.add_model(sp_model)

            log_rate = max(1, max_cycles // 4)
            params = sim.build_params(
                max_cycles=max_cycles,
                time_step=time_step,
                write_media_log=True,
                write_biomass_log=True,
                biomass_log_rate=log_rate,
            )
            params.set_param("numRunThreads", 4)
            params.set_param("spatialKill", 1e-13)

            run_dir = sweep_out / f"o2_{o2:g}" / f"spatial_{cond}"
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                exp = c.comets(layout, params, relative_dir=str(run_dir.relative_to(Path.cwd())) + "/")
                exp.run(delete_files=False)
                bm = sim._rename_biomass_cols(exp.total_biomass)
            except Exception as e:
                print(f"    {cond}: WARNING spatial failed ({e}); fallback to 0D/cobrapy")
                r = sim.run(condition=cond, max_cycles=max_cycles, output_dir=outdir / "comets_0d")
                bm = r.total_biomass

            point[cond] = {
                "di_last": _get_last_di(bm),
                "phi_pg_last": _get_last_phi_pg(bm),
                "biomass_total_last": float(bm[[sp for sp in SPECIES if sp in bm.columns]].iloc[-1].sum()),
            }

        summary["points"].append(point)

    out_json = outdir / "B_sweep_o2_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    _plot_B_sweep_o2(summary, outdir)
    print(f"  → {out_json}")
    print(f"  → {outdir}/B_sweep_o2_phase.png")
    return summary


def _plot_B_sweep_o2(summary: dict, outdir: Path):
    pts = summary.get("points", [])
    if not pts:
        return
    o2 = np.array([p["o2"] for p in pts], dtype=float)
    di_h = np.array([p["healthy"]["di_last"] for p in pts], dtype=float)
    di_d = np.array([p["diseased"]["di_last"] for p in pts], dtype=float)
    pg_h = np.array([p["healthy"]["phi_pg_last"] for p in pts], dtype=float)
    pg_d = np.array([p["diseased"]["phi_pg_last"] for p in pts], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax0, ax1 = axes

    ax0.plot(o2, di_h, marker="o", color="#2E7D32", label="healthy")
    ax0.plot(o2, di_d, marker="o", color="#C62828", label="diseased")
    ax0.set_xscale("log")
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("Bulk O2 (mM) [log]")
    ax0.set_ylabel("DI (last)")
    ax0.set_title("DI vs O2")
    ax0.legend()
    ax0.grid(alpha=0.25)

    ax1.plot(o2, pg_h, marker="o", color="#2E7D32", label="healthy")
    ax1.plot(o2, pg_d, marker="o", color="#C62828", label="diseased")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Bulk O2 (mM) [log]")
    ax1.set_ylabel("φ_Pg (last)")
    ax1.set_title("Pg fraction vs O2")
    ax1.legend()
    ax1.grid(alpha=0.25)

    fig.suptitle(f"COMETS sweep (nx={summary.get('nx')}, nz={summary.get('nz')})", fontweight="bold")
    fig.savefig(outdir / "B_sweep_o2_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_E_metabolic_prior(outdir: Path = OUTDIR):
    outdir.mkdir(parents=True, exist_ok=True)
    for cond in ("healthy", "diseased"):
        prof = metabolic_interaction_prior(cond, flux_threshold=1e-7, min_count=1)
        out_json = outdir / f"E_metabolic_prior_{cond}.json"
        with open(out_json, "w") as f:
            json.dump(prof, f, indent=2)

        sign = np.array(prof["prior"]["sign"], dtype=float)
        cross = np.array(prof["prior"]["crossfeed_count"], dtype=float)
        comp = np.array(prof["prior"]["competition_count"], dtype=float)
        score = cross - comp

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax0, ax1 = axes
        im0 = ax0.imshow(score, cmap="bwr", vmin=-max(1.0, np.max(np.abs(score))), vmax=max(1.0, np.max(np.abs(score))))
        ax0.set_title(f"{cond}: cross - competition")
        ax0.set_xticks(range(len(SPECIES_ORDER)))
        ax0.set_yticks(range(len(SPECIES_ORDER)))
        ax0.set_xticklabels(list(SPECIES_ORDER))
        ax0.set_yticklabels(list(SPECIES_ORDER))
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(sign, cmap="bwr", vmin=-1, vmax=1)
        ax1.set_title(f"{cond}: sign prior")
        ax1.set_xticks(range(len(SPECIES_ORDER)))
        ax1.set_yticks(range(len(SPECIES_ORDER)))
        ax1.set_xticklabels(list(SPECIES_ORDER))
        ax1.set_yticklabels(list(SPECIES_ORDER))
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        fig.suptitle("Metabolic interaction prior (COBRApy; AGORA exchanges)", fontweight="bold")
        fig.savefig(outdir / f"E_metabolic_prior_{cond}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_json}")
        print(f"  → {outdir}/E_metabolic_prior_{cond}.png")


def _plot_B(results: dict, nx: int, nz: int, time_step: float, outdir: Path, max_cycles: int = 800):
    # Determine snapshot cycles (25%, 50%, 100% of sim)
    log_rate = results["healthy"].get("log_rate", max(1, max_cycles // 4))
    snap_cycles = [
        log_rate * max(1, max_cycles // (4 * log_rate)),
        log_rate * max(1, max_cycles // (2 * log_rate)),
        log_rate * (max_cycles // log_rate),
    ]
    snap_cycles = sorted(set(snap_cycles))

    n_snaps = len(snap_cycles)
    fig = plt.figure(figsize=(14, 4 + 3 * n_snaps))
    n_rows = 2 + n_snaps
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.55, wspace=0.35)
    fig.suptitle(f"5-species Oral Biofilm — COMETS 2D Spatial ({nx}×{nz}) (B)",
                 fontsize=12, fontweight="bold")

    for col, cond in enumerate(("healthy", "diseased")):
        bm = results[cond]["total_biomass"]
        sp_cols = [sp for sp in SPECIES if sp in bm.columns]
        cycles = bm["cycle"].values
        t = cycles * time_step

        # Row 0: total biomass time series
        ax = fig.add_subplot(gs[0, col])
        for sp in SPECIES:
            if sp not in sp_cols:
                continue
            ax.semilogy(t, np.maximum(bm[sp].values, 1e-15),
                        color=COLORS[sp], label=SPECIES_LABELS[sp], lw=1.5)
        ax.set_title(f"{cond.upper()} — Total Biomass")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Biomass (g)")
        ax.legend(fontsize=7, ncol=2)

        # Row 1: DI time series
        ax2 = fig.add_subplot(gs[1, col])
        di_df = compute_di(bm)
        ax2.plot(di_df["cycle"].values * time_step, di_df["DI"].values,
                 color="k", lw=2)
        ax2.axhline(0.5, color="gray", ls="--", lw=1)
        ax2.set_ylim(0, 1)
        ax2.set_title("Dysbiosis Index")
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("DI")

        # Rows 2+: spatial DI heatmaps at snapshot cycles
        exp = results[cond].get("exp")
        model_ids = results[cond].get("model_ids", {})
        for ri, snap_c in enumerate(snap_cycles):
            ax3 = fig.add_subplot(gs[2 + ri, col])
            if exp is not None and model_ids:
                try:
                    # Collect biomass per species at this cycle → compute DI spatially
                    grid_list = []
                    for sp in list(SPECIES.keys()):
                        mid = model_ids.get(sp)
                        if mid is None:
                            continue
                        g = exp.get_biomass_image(mid, snap_c)
                        grid_list.append(np.array(g, dtype=float))
                    if grid_list:
                        stacked = np.stack(grid_list, axis=0)  # (n_sp, nx, nz)
                        total = stacked.sum(axis=0) + 1e-30
                        fracs_grid = stacked / total
                        # Shannon entropy DI
                        with np.errstate(divide="ignore", invalid="ignore"):
                            H = -np.nansum(fracs_grid * np.log(fracs_grid + 1e-30), axis=0)
                        H_max = np.log(len(grid_list))
                        di_grid = 1.0 - H / H_max if H_max > 0 else np.zeros_like(H)
                        im = ax3.imshow(di_grid.T, origin="lower", aspect="auto",
                                        cmap="RdYlGn_r", vmin=0, vmax=1)
                        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                        ax3.set_title(f"DI grid  t={snap_c * time_step:.0f}h")
                    else:
                        ax3.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax3.transAxes)
                except Exception as e_snap:
                    ax3.text(0.5, 0.5, f"snap err\n{e_snap}", ha="center", va="center",
                             transform=ax3.transAxes, fontsize=7)
            else:
                ax3.text(0.5, 0.5, "spatial N/A\n(0D fallback)", ha="center", va="center",
                         transform=ax3.transAxes, fontsize=8)
            ax3.set_xlabel("x")
            ax3.set_ylabel("z (depth)")

    fig.savefig(outdir / "B_spatial_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# C: MetaPhlAn init_comp.json → COMETS 初期値
# ─────────────────────────────────────────────────────────────────────────────

def run_C(
    init_comp_path: Path,
    max_cycles: int = 500,
    time_step: float = 0.01,
    outdir: Path = OUTDIR,
):
    """
    患者 A_3 の MetaPhlAn 組成を COMETS 初期値として使用。
    init_comp.json: {Str, Act, Vel, Hae, Rot, Fus, Por} → {So, An, Vp, Fn, Pg} にマップ
    """
    print(f"\n=== STEP C: MetaPhlAn init_comp → COMETS ===")

    if not init_comp_path.exists():
        print(f"  init_comp.json not found: {init_comp_path}")
        print("  MetaPhlAn pipeline (PBS job 29511) が完了してから再実行してください。")
        return None

    outdir.mkdir(parents=True, exist_ok=True)

    with open(init_comp_path) as f:
        ic = json.load(f)

    # init_comp: Str/Act/Vel/Hae/Rot/Fus/Por → COMETS 5種へマッピング
    # Hae (Haemophilus) と Rot (Rothia) は AGORA GEM なし → Fn に合算
    sp_fracs = {
        "So": ic.get("Str", 0.0),
        "An": ic.get("Act", 0.0),
        "Vp": ic.get("Vel", 0.0),
        "Fn": ic.get("Fus", 0.0) + ic.get("Hae", 0.0) + ic.get("Rot", 0.0),
        "Pg": ic.get("Por", 0.0),
    }

    # 正規化
    total = sum(sp_fracs.values())
    if total > 0:
        sp_fracs = {k: v / total for k, v in sp_fracs.items()}

    print(f"  Sample fractions: {json.dumps({k: round(v,3) for k,v in sp_fracs.items()})}")

    sim = OralBiofilmComets(grid=(1, 1))

    # Init fracs を一時的に "healthy" キーに上書き（COMETS / COBRApy 両方で有効）
    import nife.comets.oral_biofilm as ob_mod
    _orig_healthy = ob_mod.INIT_FRACTIONS["healthy"].copy()
    ob_mod.INIT_FRACTIONS["healthy"] = sp_fracs

    # media: healthy を基準に
    r = sim.run(
        condition="healthy",
        max_cycles=max_cycles,
        output_dir=outdir / "comets_patient",
    )

    _plot_C(r, sp_fracs, time_step, outdir, init_comp_path.stem)
    print(f"  → {outdir}/C_patient_A3.png")
    return r


def _plot_C(r, sp_fracs, time_step, outdir, label):
    bm = r.total_biomass
    sp_cols = [c for c in bm.columns if c in SPECIES]
    cycles = bm["cycle"].values
    t = cycles * time_step

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Patient A_3 (MetaPhlAn init) — COMETS 0D  (C)\n{label}",
                 fontsize=11, fontweight="bold")

    # 初期組成 pie
    ax0 = axes[0]
    vals = [sp_fracs[sp] for sp in SPECIES if sp_fracs[sp] > 0]
    lbls = [SPECIES_LABELS[sp] for sp in SPECIES if sp_fracs[sp] > 0]
    cols = [COLORS[sp] for sp in SPECIES if sp_fracs[sp] > 0]
    ax0.pie(vals, labels=lbls, colors=cols, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 8})
    ax0.set_title("Initial composition\n(MetaPhlAn)")

    # バイオマス time course
    ax1 = axes[1]
    for sp in SPECIES:
        if sp not in sp_cols:
            continue
        ax1.semilogy(t, np.maximum(bm[sp].values, 1e-15),
                     color=COLORS[sp], label=SPECIES_LABELS[sp], lw=1.5)
    ax1.set_title("Biomass over time")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Biomass (g)")
    ax1.legend(fontsize=7)

    # φ_Pg (Porphyromonas relative abundance)
    ax2 = axes[2]
    sp_total = bm[sp_cols].sum(axis=1).replace(0, np.nan)
    if "Pg" in sp_cols:
        phi_pg = (bm["Pg"] / sp_total).fillna(0.0).values
    else:
        phi_pg = np.zeros(len(t))
    ax2.plot(t, phi_pg, color=COLORS.get("Pg", "purple"), lw=2)
    ax2.axhline(0.1, color="gray", ls="--", lw=1, label="φ_Pg=0.1")
    ax2.set_ylim(0, 1)
    ax2.set_title("φ_Pg (Porphyromonas fraction)")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("φ_Pg")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outdir / "C_patient_A3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Step D: Mukherjee 2025 (DHNA) inspired in vitro–in silico comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_D(
    init_comp_path: Path,
    max_cycles: int = 24 * 6,
    time_step: float = 1.0,
    outdir: Path = OUTDIR,
):
    """
    Mukherjee et al. 2025 (Microbiol Spectr) の DHNA 条件に合わせた介入比較（Pg 単独系）:
      baseline : hemin 添加培地 + Pg
      dhna     : baseline + DHNA による Pg 成長促進（mu_max 上方補正の proxy）
    """
    import nife.comets.oral_biofilm as ob_mod

    outdir.mkdir(parents=True, exist_ok=True)

    sp_fracs = ob_mod.INIT_FRACTIONS["pg_single"].copy()
    print(f"  [D] Using pg_single init: {sp_fracs}")

    SCENARIOS = {
        "baseline": {"media_override": {}, "mu_scale": {"Pg": 1.0}},
        "dhna": {"media_override": {}, "mu_scale": {"Pg": 1.2}},
    }

    results = {}
    for name, cfg in SCENARIOS.items():
        print(f"  [D] Running {name}...")

        orig_frac = ob_mod.INIT_FRACTIONS["pg_single"].copy()
        ob_mod.INIT_FRACTIONS["pg_single"] = sp_fracs.copy()

        orig_media = ob_mod.MEDIA_PG_SINGLE.copy()
        for k, v in cfg["media_override"].items():
            ob_mod.MEDIA_PG_SINGLE[k] = v

        orig_mu = {sp: ob_mod.MONOD_PARAMS[sp]["mu_max"] for sp in cfg["mu_scale"]}
        for sp, scale in cfg["mu_scale"].items():
            ob_mod.MONOD_PARAMS[sp]["mu_max"] *= scale

        try:
            sim = OralBiofilmComets(grid=(1, 1))
            bm, med = sim.run_dfba_cobra(
                condition="pg_single",
                max_cycles=max_cycles,
                time_step=time_step,
            )

            class _Result:
                pass
            r = _Result()
            r.total_biomass = bm
            r.media = med
            r.is_cobra = True
            results[name] = r
        finally:
            ob_mod.INIT_FRACTIONS["pg_single"] = orig_frac
            ob_mod.MEDIA_PG_SINGLE.update(orig_media)
            for sp, mu in orig_mu.items():
                ob_mod.MONOD_PARAMS[sp]["mu_max"] = mu

    _plot_D(results, sp_fracs, time_step=time_step, outdir=outdir)
    print(f"  → {outdir}/D_intervention.png")


def _plot_D(results: dict, sp_fracs: dict, time_step: float, outdir: Path):
    scenario_styles = {
        "baseline":      {"color": "#555555", "ls": "-",  "label": "Baseline"},
        "dhna":          {"color": "#9C27B0", "ls": "--", "label": "DHNA (μmax×1.2)"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Step D — Mukherjee 2025 inspired: Pg single-species + DHNA proxy\n"
                 "(COBRApy Monod dFBA; hemin present; NH4 tracked as assimilation marker)",
                 fontsize=11, fontweight="bold")

    # Panel 1: Pg biomass
    ax0 = axes[0]
    for name, r in results.items():
        bm = r.total_biomass
        t = bm["cycle"].values * time_step
        pg = bm["Pg"].values if "Pg" in bm.columns else np.zeros(len(t))
        s = scenario_styles[name]
        ax0.semilogy(t, np.maximum(pg, 1e-15), color=s["color"], ls=s["ls"],
                     lw=2, label=s["label"])
    ax0.set_title("P. gingivalis biomass")
    ax0.set_xlabel("Time (h)")
    ax0.set_ylabel("Biomass (g)")
    ax0.legend(fontsize=7)

    # Panel 2: NH4 concentration
    ax1 = axes[1]
    for name, r in results.items():
        med = r.media
        if med is None or med.empty:
            continue
        nh4 = med[med["metabolite"] == "nh4[e]"]
        if nh4.empty:
            continue
        t = nh4["cycle"].values * time_step
        y = nh4["conc_mmol"].values
        s = scenario_styles[name]
        ax1.plot(t, y, color=s["color"], ls=s["ls"], lw=2, label=s["label"])
    ax1.set_title("NH4 in conditioned medium")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("NH4 (mM)")
    ax1.legend(fontsize=7)

    # Panel 3: bar chart — final Pg biomass ratio
    ax2 = axes[2]
    names = list(results.keys())
    final_pg = []
    for name in names:
        bm = results[name].total_biomass
        pg_last = bm["Pg"].iloc[-1] if "Pg" in bm.columns else 0.0
        final_pg.append(pg_last)
    colors = [scenario_styles[n]["color"] for n in names]
    bars = ax2.bar(names, final_pg, color=colors, edgecolor="k", linewidth=0.5)
    ax2.set_ylim(0, max(max(final_pg) * 1.3, 1e-12))
    ax2.set_title(f"Final Pg biomass (t={results['baseline'].total_biomass['cycle'].iloc[-1] * time_step:.0f}h)")
    ax2.set_ylabel("Biomass (g)")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([scenario_styles[n]["label"] for n in names],
                        rotation=15, ha="right", fontsize=7)
    for bar, val in zip(bars, final_pg):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(final_pg) * 0.02,
                 f"{val:.2e}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(outdir / "D_intervention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["A", "B", "C", "D", "E"], default=None,
                    help="Run only this step")
    ap.add_argument("--all", action="store_true", help="Run A + B + C")
    ap.add_argument("--cycles", type=int, default=500)
    ap.add_argument("--nx", type=int, default=10, help="Spatial grid X (for B)")
    ap.add_argument("--nz", type=int, default=20, help="Spatial grid Z/depth (for B)")
    ap.add_argument("--sweep-o2", default=None,
                    help="Comma-separated bulk O2 values (mM) for STEP B sweep (e.g. 0.001,0.01,0.1,1)")
    ap.add_argument("--force-cobra", action="store_true",
                    help="Skip COMETS Java run and use COBRApy dFBA fallback (useful for quick sweeps)")
    ap.add_argument("--init-comp", type=Path,
                    default=Path("/home/nishioka/IKM_Hiwi/nife/data/metaphlan_profiles/init_comp_ERR13166576_A_3.json"),
                    help="init_comp.json path (for C)")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args(argv)

    steps = []
    if args.all:
        steps = ["A", "B", "C"]
    elif args.step:
        steps = [args.step]
    else:
        steps = ["A"]  # デフォルトは A のみ

    print(f"Running steps: {steps}")
    print(f"Output dir: {args.outdir}")

    if "A" in steps:
        run_A(max_cycles=args.cycles, outdir=args.outdir)

    if "B" in steps:
        if args.sweep_o2:
            o2_values = [float(x) for x in args.sweep_o2.split(",") if x.strip()]
            run_B_sweep_o2(
                o2_values=o2_values,
                nx=args.nx,
                nz=args.nz,
                max_cycles=args.cycles,
                outdir=args.outdir,
                force_cobra=args.force_cobra,
            )
        else:
            run_B(
                nx=args.nx,
                nz=args.nz,
                max_cycles=args.cycles,
                outdir=args.outdir,
                force_cobra=args.force_cobra,
            )

    if "C" in steps:
        run_C(init_comp_path=args.init_comp, max_cycles=args.cycles, outdir=args.outdir)

    if "D" in steps:
        run_D(init_comp_path=args.init_comp, max_cycles=24 * 6, time_step=1.0, outdir=args.outdir)

    if "E" in steps:
        run_E_metabolic_prior(outdir=args.outdir)

    print("\n=== All done ===")
    print(f"Results in: {args.outdir}")


if __name__ == "__main__":
    main()
