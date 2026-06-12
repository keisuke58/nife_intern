# DRAFT reply → Soleimani (2026-06-12)

Proposed thesis title (FEM-forward, keeps Hamilton + biofilm):
**Finite-Element Implementation of a Hamilton-Principle Material Model for Oral Biofilms —
Gauss-Point Coupling of a Python Constitutive Model in ANSYS**

---

Dear Meisam,

Thank you for the proposal — I'm enthusiastic about it. Integrating my material model into the
existing ANSYS framework is a natural and exciting continuation of my HiWi and research-project
work, and I'd be glad to proceed. In fact, my ongoing NIFE internship research addresses
essentially the same theme — the modelling of oral biofilm behaviour — so I see a strong
opportunity to feed those results directly into this thesis, reinforcing the continuity you
mention rather than running two separate tracks.

Before I finalize Page 1 for the Prüfungsamt, a few points to keep the scope and timeline realistic:

0. **Title / scope alignment** — I'd registered a working title with Prof. Junker on the biofilm
   side. Since the NIFE biofilm work and the ANSYS material model are closely related, could we
   agree on a single title that frames the material-model integration as the mechanical extension
   of that line of work? My current suggestion: *"Finite-Element Implementation of a
   Hamilton-Principle Material Model for Oral Biofilms — Gauss-Point Coupling of a Python
   Constitutive Model in ANSYS."* Happy to adjust.

1. **Coupling approach** — Which Python↔ANSYS route do you envisage? Per-Gauss-point CPython via
   UPF/UserMat is possible but performance-sensitive; an offline-tabulated surrogate of the model,
   or a Fortran/IPC bridge, may be more robust. I'd be glad to prototype the most promising option
   first.

2. **ANSYS access & handover** — Could we arrange a short meeting with Felix to align on his FEM
   codebase (and acknowledge his PhD contribution in the thesis)? I'd also like to confirm how I
   obtain my own ANSYS seat via the IKM/LUH campus license, so licensing doesn't delay the start.
   In the meantime I have my own Abaqus access and can prototype the Python↔constitutive-model
   coupling pattern there, so I'm not blocked while ANSYS access is set up.

3. **Validation** — Are there existing reference cases for the current constitutive model, or shall
   we define 2–3 standard benchmarks for verification?

4. **Timeline** — I aim to submit Page 1 in the coming weeks and begin the formal 6-month period in
   [Month]. Does that fit your and Felix's availability?

5. **Scheduling** — With my 3-month NIFE internship and ongoing HiWi duties, I'd like to sketch a
   milestone plan that keeps the heaviest implementation outside the internship window. Where the
   NIFE results overlap with the thesis, I'll reuse them to save time rather than duplicate effort.

Best regards,
Keisuke Nishioka
