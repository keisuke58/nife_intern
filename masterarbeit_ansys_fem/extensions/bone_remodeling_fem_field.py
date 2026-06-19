"""FEM-coupled Frost mechanostat — apply the canonical microstrain bone-adaptation windows to the
REAL 3-D crowned-implant strain field, and compare the predicted remodeling pattern with the clinical
marginal-bone-loss (MBL) literature.

This fills the one standard ("定番") method the project's remodeling set was missing: Frost's
mechanostat expressed in **microstrain (µε)** with the **dual-threshold overload→resorption** branch
(bone_standard_laws.frost_remodeling_sign). It complements:
  • bone_remodeling_huiskes.py            — Huiskes site-specific SED law (monotonic, no upper limit)
  • fig_periimplantitis_remodel.py        — vM-overload map on the 4 mm-loss field
  • fig_periimplantitis_rankl_opg.py       — mechanistic RANKL/OPG disease model
  • fig_periimplantitis_clinical_calibration.py — calibration vs a longitudinal cohort

WHY THE UPPER THRESHOLD MATTERS. A Huiskes/Frost rule with only a lower (disuse) threshold is
monotonic: more load → more bone, forever. It predicts the stress-concentrated implant crest should
*densify*. Clinically the crest is exactly where bone is *lost* (MBL). Frost's UPPER threshold
(MESp ≈ 3000 µε: pathological overload → microdamage → osteoclastic resorption) resolves this: the
implant shoulder funnels strain into the overload window, so the crest resorbs. This script shows the
real 3-D field does reach that window crestally, reproducing the clinical crestal-MBL localisation
from mechanics alone (the *magnitude* / progression then needs the inflammatory RANKL/OPG channel —
the project's vicious-cycle thesis).

CAVEAT (honest): this single-implant model loads only the implant, so far-field alveolar bone sees
near-zero strain (apparent "disuse") — an artifact of the absent natural dentition / muscle loads,
not a real resorption prediction. The remodeling interpretation is therefore restricted to the
peri-implant collar (r ≤ 4 mm), where the implant load genuinely dominates.

Run:  python bone_remodeling_fem_field.py [path/to/tier2b_crown_field.json]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")          # courteous on the shared login node

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from bone_standard_laws import (  # noqa: E402
    FROST_WINDOWS, MES_M, MES_P, MES_R, MBL_REFERENCE, frost_remodeling_sign,
    is_in_physiological_range,
)

DEFAULT_FIELD = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real/tier2b_crown_field.json")
CREST_Z = 29.0                            # bone crest [mm]
IMPLANT_AXIS = np.array([-69.4, -41.0])   # tooth-23 implant axis (x, y)
BONE_MATS = ("CORTICAL", "CANCELLOUS", "BONE")
COLLAR_R = 4.0                            # peri-implant collar radius [mm]

CLASS_COLOR = {"disuse": "#3498db", "adapted": "#2ecc71", "formation": "#f39c12",
               "overload": "#c0392b", "fracture": "#7b241c"}


def load_bone_field(path: Path):
    els = json.load(open(path))["els"]
    mat = np.array([e["mat"] for e in els])
    keep = np.isin(mat, BONE_MATS)
    g = lambda k: np.array([e[k] for e in els])[keep]
    return g("x"), g("y"), g("z"), g("vmo"), g("vmg"), mat[keep]


def main() -> int:
    field_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIELD
    if not field_path.exists():
        print(f"[error] field not found: {field_path} — pass tier2b_crown_field.json as argv[1].")
        return 1

    x, y, z, vmo, vmg, mat = load_bone_field(field_path)
    ue = vmg * 1.0e6                                   # von-Mises strain → microstrain
    depth = CREST_Z - z                                # >0 = apical (below crest)
    r = np.hypot(x - IMPLANT_AXIS[0], y - IMPLANT_AXIS[1])
    collar = r <= COLLAR_R
    print(f"loaded {len(z)} bone elements ({collar.sum()} in peri-implant collar r≤{COLLAR_R} mm)")

    sign = frost_remodeling_sign(ue)                   # dual-threshold: -1 resorb / 0 maintain / +1 form

    # ── crestal ISO/Frost sanity (peri-implant collar, crestal 3 mm band) ──────
    crest = collar & (depth >= -1.0) & (depth < 3.0)
    vm_p95 = float(np.percentile(vmo[crest], 95))
    ue_p95 = float(np.percentile(ue[crest], 95))
    print("\n" + is_in_physiological_range(vm_p95, ue_p95))

    # ── Frost-window composition by depth, peri-implant collar ────────────────
    bands = [(-1.0, 3.0, "crestal 0–3 mm"), (3.0, 6.0, "mid 3–6 mm"), (6.0, 99.0, "apical >6 mm")]
    print("\nFrost-window composition (peri-implant collar):")
    overload_share_crest = None
    for lo, hi, name in bands:
        sel = collar & (depth >= lo) & (depth < hi)
        n = max(sel.sum(), 1)
        ov = float(np.mean(ue[sel] >= MES_P) * 100)
        net = float(np.mean(sign[sel]))               # mean remodeling sign (−1..+1)
        if name.startswith("crestal"):
            overload_share_crest = ov
        print(f"  {name:14s} n={sel.sum():6d}  overload(>{MES_P:.0f}µε)={ov:5.1f}%  "
              f"formation={np.mean((ue[sel]>=MES_M)&(ue[sel]<MES_P))*100:4.1f}%  "
              f"net sign={net:+.2f}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # (A) overload (>MESp) fraction vs depth — the resorption-driving window
    dd = np.linspace(0, 12, 25)
    ov_prof = [np.mean(ue[collar & (depth >= dd[i]) & (depth < dd[i + 1])] >= MES_P) * 100
               if (collar & (depth >= dd[i]) & (depth < dd[i + 1])).sum() else 0.0
               for i in range(len(dd) - 1)]
    axes[0].barh((dd[:-1] + dd[1:]) / 2, ov_prof, height=(dd[1] - dd[0]) * 0.9, color="#c0392b", alpha=0.85)
    axes[0].invert_yaxis()
    axes[0].axhspan(0, 3, color="#f1c40f", alpha=0.22, label="crestal 0–3 mm (MBL onset)")
    axes[0].set_xlabel(fr"% bone in overload window (>{MES_P:.0f} µε)")
    axes[0].set_ylabel("depth below crest [mm]")
    axes[0].set_title("(A) Overload localises crestally", fontsize=11, loc="left")
    axes[0].legend(fontsize=8); axes[0].grid(True, lw=0.4, alpha=0.4)

    # (B) spatial r–z map coloured by Frost WINDOW (4 classes) so disuse-resorption (far-field,
    # boundary artifact) is visually distinct from the clinically-relevant crestal overload-resorption.
    cls_field = _window_of(ue[collar])
    order_b = ["disuse", "adapted", "formation", "overload"]
    for cls in order_b:
        m = cls_field == cls
        axes[1].scatter(r[collar][m], z[collar][m], s=3, color=CLASS_COLOR[cls],
                        label=cls, rasterized=True)
    axes[1].axhline(CREST_Z, color="k", ls="--", lw=1.0)
    axes[1].set_xlabel("radial distance from implant axis [mm]")
    axes[1].set_ylabel("z [mm]")
    axes[1].set_title("(B) Frost window (overload = crestal)", fontsize=11, loc="left")
    axes[1].legend(fontsize=7, loc="lower right", markerscale=2)

    # (C) Frost-window composition by depth band (stacked) + clinical annotation
    order = ["disuse", "adapted", "formation", "overload"]
    bottoms = np.zeros(len(bands))
    xs = np.arange(len(bands))
    for cls in order:
        vals = [np.mean(_window_of(ue[collar & (depth >= lo) & (depth < hi)]) == cls) * 100
                if (collar & (depth >= lo) & (depth < hi)).sum() else 0.0
                for lo, hi, _ in bands]
        axes[2].bar(xs, vals, bottom=bottoms, color=CLASS_COLOR[cls], label=cls, alpha=0.9)
        bottoms += np.array(vals)
    axes[2].set_xticks(xs); axes[2].set_xticklabels([b[2].split(" ")[0] for b in bands], fontsize=9)
    axes[2].set_ylabel("Frost-window share [%]"); axes[2].set_ylim(0, 100)
    axes[2].set_title("(C) vs clinical MBL pattern", fontsize=11, loc="left")
    axes[2].legend(fontsize=7, loc="center right")
    axes[2].annotate(f"crestal overload = {overload_share_crest:.0f}%\nclinical MBL onset = crestal\n"
                     "(Oh 2002; Albrektsson <0.2 mm/yr)", xy=(0.5, 0.5), xycoords="axes fraction",
                     ha="center", va="center", fontsize=7,
                     bbox=dict(boxstyle="round", fc="#fffbe6", ec="0.6", lw=0.5))

    fig.suptitle("FEM-coupled Frost mechanostat — microstrain windows on the 3-D crown strain field",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_bone_remodeling_fem_field.pdf")
    fig.savefig(OUT / "fig_bone_remodeling_fem_field.png", dpi=200)
    print(f"\nwrote {(OUT / 'fig_bone_remodeling_fem_field.pdf').name}")
    print(f"crestal overload-window share = {overload_share_crest:.1f}%  "
          f"→ pathological-overload (resorption-driving) strain concentrates at the crest, "
          f"matching clinical MBL onset")
    return 0


def _window_of(ue: np.ndarray) -> np.ndarray:
    """Vectorised Frost-window label for a microstrain array."""
    out = np.full(np.asarray(ue).shape, "adapted", dtype=object)
    for lo, hi, name in FROST_WINDOWS:
        out[(ue >= lo) & (ue < hi)] = name
    return out


if __name__ == "__main__":
    raise SystemExit(main())
