# Masterarbeit — Python material model × ANSYS FEM

Working directory for the Master's-thesis topic proposed by **Meisam Soleimani (IKM)** on
2026-06-12: integrate Keisuke's Python-based **constitutive / material model** (the basis of
the recent IKM publication) into **Felix's existing ANSYS finite-element framework**, replacing
the current phenomenological constitutive model — invoked at each Gauss point.

- **Erstprüfer (1st):** Soleimani (proposed the topic) · **Zweitprüfer (2nd):** Geisler · **PI:** Junker (too busy to examine) · FEM codebase owner: Felix (IKM)
- **Registration form:** "Antrag auf Zulassung zur Masterarbeit" (2 pages)
  https://www.uni-hannover.de/studium/im-studium/pruefungsinfos-fachberatung/maschinenbau-msc/formulare
  - Page 1 → Prüfungsamt verifies formal eligibility → then Page 2 is completed & signed.
- **Formal period:** 6 months from registration.

## ⚠️ Open decision — title / scope reconciliation
Title floated to Prof. Junker (2026-06-04): *Hamilton-Principle Models of Oral Biofilm Dysbiosis —
GPU-Accelerated Inference, Metabolic Priors, and Spatial Stratification* (inference-centric).

Soleimani's ANSYS proposal is computational **solid mechanics** (material model at Gauss points).
**Proposed retitle (FEM-forward, keeps Hamilton[=Junker] + biofilm[=NIFE continuity]):**
> *Finite-Element Implementation of a Hamilton-Principle Material Model for Oral Biofilms —
> Gauss-Point Coupling of a Python Constitutive Model in ANSYS*

**Before Page 1 goes to the Prüfungsamt, Junker + Soleimani must agree on this ONE title.**

### ANSYS vs Abaqus
- **ANSYS** (target — Felix's framework): not a personal license; request a seat via the **IKM/LUH
  campus/Academic license**. Confirm early so licensing doesn't delay the start.
- **Abaqus** (own access): use as a **prototyping sandbox** for the Python↔constitutive-model
  coupling while ANSYS access is being set up. Deliverable must run in Felix's ANSYS model.

## Open questions to Soleimani (sent in reply 2026-06-12)
0. Title/scope alignment with Junker's registered direction.
1. **Coupling approach** — per-Gauss-point CPython via UPF/UserMat, Fortran ctypes bridge, IPC,
   or an **offline-tabulated surrogate** of the model (perf-robust). Prototype the best option first.
2. **Handover** — meeting with Felix; acknowledge his PhD FEM contribution in the thesis.
3. **Validation** — existing reference cases, or define 2–3 standard benchmarks.
4. **Timeline** — submit Page 1 in coming weeks; begin 6-month period in [Month].
5. **Scheduling** — keep heaviest implementation outside the 3-month NIFE internship + HiWi window.

## Subdirs
- `notes/`            — meeting notes, decisions, the email thread
- `coupling_prototype/` — Python↔ANSYS coupling experiments (UserMat / surrogate / IPC)
- `benchmarks/`       — verification cases against the current phenomenological model

## Status
- [ ] ANSYS seat obtained via IKM/LUH campus license (Abaqus prototyping in the meantime)
- [ ] Topic/title agreed by Junker + Soleimani
- [ ] Page 1 submitted to Prüfungsamt
- [ ] Felix handover meeting
- [~] Coupling prototype — `coupling_prototype/`. Local-reproducible (JAX), all PASS:
      rung 1 placeholder tangent · 1b REAL Heine NSP tangent (5e-5) · 1c gLV-vs-NSP comparison
      (gLV 5-state, NSP 12-state) · 1d φ_Pg→growth eigenstrain→Cauchy stress (mechanically closed) ·
      1e CLSM β calibration (DH corr=0.88, β_depth=17.2, anisotropic depth growth) ·
      1f finite-strain F=FₑF_g + neo-Hookean (λ_z~2.4; confined ~−18μ, free-top ~0) ·
      2 socket bridge (exact) · **3 Abaqus UMAT — VERIFIED on Abaqus 2024 (σ_xx=E·ε)** ·
      **3b Abaqus NSP finite-strain — VERIFIED on Abaqus 2024 (NLGEOM, U3→λ_z−1, σ=0, SDV1 0→20)** ·
      4 ANSYS USERMAT skeleton (shear remap, after seat). Build notes: `coupling_prototype/BUILD.md`.
- [ ] Benchmarks defined
- [ ] Page 2 signed → registration complete
