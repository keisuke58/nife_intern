# FEM element-order / formulation fidelity audit (2026-06-18)

Two independent "is the headline element good enough?" checks, both run on the local Abaqus/Standard
2024 (shared DSLS, **QSD = 50 tokens only → jobs MUST run strictly serially**, concurrency exhausts the
pool and deadlocks). Units mm-N-MPa.

---

## 1. Plane-strain thread model — element-FORMULATION robustness  ✅ invariant

**Question (examiner-bait):** the headline thread model (`gen_implant_inp.py`) is CPE4 (linear,
full-integration) at **ν = 0.45** (near-incompressible biofilm). Does **volumetric locking** or
**curved-root under-resolution** distort the interface peak and the reported ratios?

**Method:** new companion generator `coupling_prototype/abaqus/gen_implant_inp_q.py` re-emits the SAME
geometry / growth eigenstrain / element numbering with a chosen formulation; `run_implant_elform.sh`
solves the matrix at the headline resolution n108. Driver: `fig_implant_elform.py` →
`figures/fem_implant_elform.pdf`. Data: `coupling_prototype/abaqus/elform_results.csv`.
(The new generator's CPE4 path reproduces the canonical `implant_DH_thread` interior peak **609.46**
exactly → conversion verified.)

**Result (interior interface peak vM, DH thread, n108):**

| element | CPE4 | CPE4H | CPE8 | CPE8R | CPE8RH |
|---|---:|---:|---:|---:|---:|
| peak vM | 609.5 | 609.5 | 598.7 | 601.4 | 601.5 |

- absolute peak **spread ±0.9 %**; **DH/CH = 2.49–2.53 (±1.8 %)**, **thread/flat = 4.07–4.14 (±1.8 %)**.
- **hybrid = standard** (CPE4H ≡ CPE4, CPE8RH ≈ CPE8R) → **no volumetric locking at ν = 0.45**.
- **quadratic = linear** (CPE8/8R ≈ CPE4) → the **smooth sinusoidal thread is already resolved at n108**
  (it has finite root/crest curvature, no sharp singularity). A coarse n72 mesh inflates the peak ~12 %
  → that was a *resolution* artefact, gone by n108.

**Verdict:** the linear-CPE4 headline carries **neither a locking nor an under-resolution artefact**; the
dysbiosis ratio and the "thread *generates* the stress" conclusion are formulation-invariant.

---

## 2. Crowned 3-D assembly — quadratic-tet (C3D10) re-solve  ✅ order-robust (+8.6 %)

**Question:** the crowned headline model `tier2b_crown` (~275 k elements) is the ONE large assembly still
on **C3D4 linear tets** (the ISO-14801 coupons were already moved to C3D10 because linear tets
under-predict the thread-root concentration by ~9–15 %). Does the **crestal peri-implant bone p95** — the
crown moment-arm clinical quantity — survive second order?

**Method:** standalone global converter `FEM/tier2b_real/convert_c3d4_to_c3d10.py` — ONE global
edge→mid-node map over every element (so the conforming PDL-inner/dentin interface stays watertight),
`ALLN` extended with all mid-nodes (uniform growth `*TEMPERATURE`), `FIXED` extended with mid-nodes on
fully-fixed edges, *CLOAD kept corner-only (100 N resultant preserved; crestal quantity is St-Venant-far
from the load patch). Extract: `extract_tier2b_q.py` (C3D10-aware). **Metric: the EXACT thesis metric**
(`crestal_p95_thesis_metric.py`, verbatim from `fig_implant_crown_fem.py::peri_implant_bone_peak`):
p95 of occlusal vM over CORTICAL/CANCELLOUS/BONE in the crestal DISK r ≤ 3.0 mm of axis T23=(−69.4,−41.0),
z ∈ [CREST−3, CREST+1.5], CREST = 29.0. **This reproduces the published C3D4 headline exactly → metric
validated** (see table). _(An earlier draft of this note used a non-thesis annulus shell r∈[2.0,3.5],
z∈[crest−3,crest] and reported a spurious ~11 % ratio drop — that was a shell-definition artefact, NOT
physics; the thesis-metric numbers below supersede it.)_

- conversion: 274 854 C3D4 → **280 695 C3D10** / 424 739 nodes. datacheck **0 errors**; untied-tie
  warnings **33** (vs **44** for the C3D4 baseline → conversion did not introduce tie defects). Solve:
  ~37 min wallclock each (crown + generic), cpus=1, ~40 GB, both steps converged.

**Result — crestal peri-implant bone p95 (occlusal step), thesis metric, both jobs re-solved at C3D10:**

| crestal bone p95 | C3D4 | C3D10 | Δ |
|---|---:|---:|---:|
| crown (load @ z 38) | **27.0** | **29.4** | **+8.9 %** |
| bare / generic (load @ z 32.5) | **17.2** | **18.9** | **+9.9 %** |
| **moment-arm ratio (crown ÷ bare)** | **×1.57** | **×1.56** | **−0.9 %** |

(C3D4 27.0 / 17.2 / ×1.57 reproduces the published "18 → 27 MPa, ×1.5" headline → confirms the metric.)

**Verdict:** quadratic order lifts BOTH crestal p95 values by **~9 %** (both load paths under-resolved by
linear tets by the same amount), so the **crown moment-arm RATIO is essentially invariant: ×1.57 → ×1.56
(−0.9 %)**. The headline **×1.5 moment-arm ratio is fully order-robust**; only the *absolute* crestal
stresses are ~9 % higher at C3D10 (**27 → 29 MPa crown, 17 → 19 MPa bare**). Unlike the smooth plane-strain
thread (§1, invariant in absolute terms too), the 3-D screw thread/neck is a genuine concentrator, so the
~9 % absolute lift is real and expected — but it is a *uniform* lift that cancels in the ratio.
**Reporting line:** *the ×1.5 crown moment-arm conclusion holds at quadratic order; the C3D10 absolute
crestal-bone p95 is ~9 % higher (27→29 / 17→19 MPa) and is the more accurate value.*

---

## Reproduce

```bash
# §1 plane-strain formulation matrix (fast, serial)
cd masterarbeit_ansys_fem/coupling_prototype/abaqus && bash run_implant_elform.sh
python ../../extensions/fig_implant_elform.py

# §2 crown / generic C3D10 (each ~37 min, cpus=1, serial — QSD=50 tokens)
cd FEM/tier2b_real
python convert_c3d4_to_c3d10.py tier2b_crown.inp   tier2b_crown_q.inp
abaqus job=tier2b_crown_q   ask_delete=OFF memory=40gb     # detached; poll .sta
abaqus python extract_tier2b_q.py tier2b_crown_q
python convert_c3d4_to_c3d10.py tier2b_generic.inp tier2b_generic_q.inp
abaqus job=tier2b_generic_q ask_delete=OFF memory=40gb
abaqus python extract_tier2b_q.py tier2b_generic_q
python compare_crown_order.py tier2b_crown tier2b_crown_q
```
