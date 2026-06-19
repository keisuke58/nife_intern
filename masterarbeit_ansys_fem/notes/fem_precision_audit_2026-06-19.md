# FEM precision audit — Stage A package (2026-06-19)

Extends `notes/fem_element_fidelity_2026-06-18.md` (element-formulation + C3D4→C3D10 order audits) with
the three precision items still missing for a defensible thesis-grade FEM headline:

  A1. **h-convergence** on the crowned tier2b assembly at the EXACT thesis metric
  A2. **Material-parameter sensitivity** (one-at-a-time, OAT) on the soft / uncertain materials
  A3. **ISO-14801 / literature validation** of the crestal-p95 absolute level

All three reuse the existing build / solve / extract toolchain (`mesh_crown.py` → `build_assembly.py` →
`abaqus job=...` → `extract_pimp_field.py` → `crestal_p95_thesis_metric.py`). No headline number is
changed; the new scripts add bracketing / cross-checking on top.

---

## A1 — h-convergence (crown LC sweep)

**Knob.** `mesh_crown.py` LC (crown tet edge, mm). The headline `mesh_crown.py` is preserved untouched
(per the project new-variant convention); a thin wrapper `mesh_crown_lc.py` accepts LC from `sys.argv[1]`,
monkey-patches `mesh_crown.LC`, and calls `mesh_crown.main()` so the geometry / OCC steps / cache
filename are byte-identical.

**Sweep.** `run_crown_hconvergence.sh` halves/doubles LC ∈ {0.55, 0.40, 0.30, 0.22}. For each LC the
crowned and bare-generic jobs are assembled, solved at C3D4 (`build_assembly.py`) AND quadratically
re-solved at C3D10 (`convert_c3d4_to_c3d10.py`) — so each LC delivers four data points (crown / generic
× C3D4 / C3D10) at the EXACT thesis metric (p95 of occlusal vM over BONE elements in the crestal disk
r ≤ 3 mm, z ∈ [crest−3, crest+1.5]).

**Compute.** 4 LCs × 2 jobs × 2 orders × ~37 min ≈ 10 h serial. The QSD=50-token DSLS forces strict
serialisation, so the sweep occupies one solver slot for the full duration; submit overnight.

**Output.** `hconv_results.jsonl`, `hconv_summary.csv`, `fig_crown_hconvergence.{pdf,png}`. The figure
shows p95 vs LC (log) for all four series and overlays the Richardson-extrapolated `p95∞` (3-grid
Roache apparent order when monotone, p=2 fallback). Panel C is the thesis-headline moment-arm ratio
(crown ÷ generic) vs LC — what we need to certify h-invariant.

**Acceptance criteria.**

- The C3D10 crown / generic series should approach `p95∞` from below with apparent order p̂ ≈ 2.
- The headline LC=0.40 C3D4 number should sit within 5 % of `p95∞` for either order — that bounds the
  residual h-discretisation error on the absolute peak.
- The moment-arm ratio (Panel C) should vary by <2 % across the swept LC range at either order.
  Combined with the C3D4↔C3D10 ratio shift of −0.9 % (audit-2026-06-18 §2), this puts a total
  h+order envelope of ~±3 % on the headline ×1.5 ratio.

If any of these are violated the figure will say so loudly (Richardson lines vs data points) and we
re-run with LC ∈ {0.22, 0.17, 0.13} until the criteria are met.

## A2 — Material-parameter OAT sensitivity

**Knob.** Wrapper script `build_assembly_override.py` honours `MATS_OVERRIDE` env var (JSON
`{mat_name: [E_MPa, nu]}`), mutates `build_assembly.MATS`, then runpy-invokes the headline CLI. The
headline `build_assembly.py` is preserved untouched; with `MATS_OVERRIDE` unset the wrapper is
byte-identical to running `build_assembly.py` directly.

**Sweep.** `run_crown_sensitivity.sh` perturbs each of the soft / uncertain materials at E×0.5 and
E×2.0 (nu fixed at headline value), one at a time, holding all others at headline:

| material | headline E [MPa] | low (×0.5) | high (×2.0) | lit. range we sweep covers |
|---|---:|---:|---:|---|
| GINGIVA  | 3.0   | 1.5   | 6.0   | 1.5–30 MPa (Picton & Wills 1978; Goktas 2011) |
| PDL      | 50.0  | 25.0  | 100.0 | 10–100 MPa (Rees & Jacobsen 1997 review)      |
| CEMENTUM | 15000 | 7500  | 30000 | 8–25 GPa (Ho et al. 2009)                     |
| BIOFILM  | 1.0   | 0.5   | 2.0   | shear-rheology rough order (Stoodley 2002)    |

→ 4 mats × 2 levels × 2 jobs = 16 solves ≈ 10 h serial.

**Output.** `sens_results.jsonl`, `sens_summary.csv`, `fig_crown_sensitivity.{pdf,png}` (tornado).
The tornado ranks the materials by `|Δp95|`-per-E-doubling; the material on top is the one to refine
modelling / measurement for first.

**Hypothesis (to falsify).** Crestal p95 is dominated by the BONE materials (already at well-known E
≈ 1–14 GPa with low literature scatter) → the soft materials should each contribute < 5 % to the
crestal p95, in which case the headline is robust to GINGIVA/PDL/BIOFILM uncertainty and the model
needs no soft-tissue measurement campaign. If a soft material does drive a > 10 % swing, that's a
finding ("crown moment-arm × biofilm stiffness coupling") and goes in the discussion.

## A3 — ISO-14801 / literature validation

**Script.** `crown_iso14801_validation.py` reads the headline crestal p95 (from `hconv_results.jsonl`
LC=0.40, or audit-note fallback), prints the headline ISO-14801 geometry (lever arm, load magnitude,
30° angle), and writes a `iso14801_validation.csv` + `.txt` comparing against a curated set of
100-N/30° literature crestal-vM bands:

- Sahin et al. 2002 (J Dent): 16–28 MPa
- Geng et al. 2004 (J Prosthet Dent), lit. review: 18–35 MPa
- Cicciu et al. 2018 (Materials, 150 N → linearly scaled): 22–38 MPa
- Bozkaya et al. 2004 (J Prosthet Dent): 14–32 MPa

**Acceptance.** The crown C3D10 number (29.4 MPa per the 2026-06-18 audit) should sit inside every
band — upper third of the Sahin/Bozkaya bands (as expected for a crown moment-arm geometry), and
lower edge of the Cicciu band after 150→100 N linear scaling. The generic (no-crown moment arm)
number (18.9 MPa C3D10) should sit in the lower third of the same bands. Both regions reproduced →
the model is in the published physiological band, and the ×1.5 moment-arm ratio is a within-cohort,
dimensionally-controlled increment that does not depend on the absolute band uncertainty.

Adding a new literature reference: append a dict to the `LITERATURE` list in the script. The
comparison is deliberately qualitative ("upper / middle / lower third of band") so that reviewer
disagreement on a specific number does not invalidate the structure of the argument.

---

## Run order

```
cd /home/nishioka/IKM_Hiwi/FEM/tier2b_real
bash run_crown_hconvergence.sh        # ~10 h, produces hconv_results.jsonl
python fig_crown_hconvergence.py      # → Panel A/B/C + Richardson
bash run_crown_sensitivity.sh         # ~10 h, produces sens_results.jsonl
python fig_crown_sensitivity.py       # → tornado
python crown_iso14801_validation.py   # instant, produces iso14801_validation.{csv,txt}
```

Stage A is complete when:

1. `fig_crown_hconvergence.pdf` shows Roache p̂ ≈ 2 on the C3D10 series and < 5 % residual error at LC=0.40.
2. `fig_crown_sensitivity.pdf` ranks the materials with the BONE materials (or rather: their absence
   from the list, by design) above the soft tissues, all soft-tissue swings ≤ 10 %.
3. `iso14801_validation.txt` shows crown C3D10 in band for every reference.

Stage B (mechanics → biology bridge productionisation) is gated on Stage A — see the existing
`coupling_prototype/*umat_server*.py` family.
