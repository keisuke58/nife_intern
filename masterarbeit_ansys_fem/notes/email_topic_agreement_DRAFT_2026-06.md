# DRAFT — topic/scope agreement email (to send next week)

**To:** soleimani@ikm.uni-hannover.de (Erstprüfer + topic proposer)
**Cc:** junker@ikm.uni-hannover.de (PI, informed) ; Hendrik ; Felix
**(Examiner update 2026-06-13: Soleimani → Erstprüfer, Geisler → Zweitprüfer, Junker → PI only.)**
**Subject:** Master's thesis — scope & title alignment + ANSYS access (with first verified results)
**Attach:** slides_fem_coupling_en.pdf

---

Dear Meisam, dear Prof. Junker,

Following our discussion on integrating my material model into the IKM finite-element framework, I
have built and **verified a first working prototype**, and I would like to align the scope and the
thesis title before I submit Page 1 to the Prüfungsamt.

**What is already working (verified on Abaqus 2024):**
- My Python biofilm material model is invoked **at each Gauss point** of an FE model through a
  self-contained user material (UMAT), replacing the phenomenological constitutive law — the core of
  the proposed topic.
- Small-strain check passes exactly (σ = E·ε); the finite-strain, composition-driven growth
  (multiplicative split F = Fₑ·F_g) runs through NLGEOM and reproduces the expected biofilm thickening.
- The growth is **grounded in our CLSM data** (biofilm thickness vs P. gingivalis fraction,
  correlation ≈ 0.88).
- First scientific result a single element cannot give: a multi-element column with depth-graded
  growth develops a **depth-resolved residual-stress profile**, and a **dysbiotic** community carries
  markedly higher and more stratified residual stress than a **commensal** one — a mechanically
  testable link to detachment/peri-implantitis. A mesh-convergence study confirms the interior
  result. (Summary in the attached slides.)

**Proposed way forward — and the one decision I need from you:**
I would like to **develop the science in Abaqus first** and **port the final coupling to ANSYS
USERMAT** as the integration step into Felix's framework. My reasons:
1. It removes the ANSYS-license wait from the critical path, so the modelling work can proceed now.
2. The coupling is designed to be solver-agnostic — the ANSYS USERMAT port is essentially a copy of
   the verified Abaqus UMAT (same material server, same protocol; only the interface signature differs).
3. Abaqus is also the platform of my subsequent research stay, giving continuity beyond the thesis.

Accordingly I propose a **solver-neutral title**, e.g.:
> *Finite-Element Implementation of a Hamilton-Principle Material Model for Oral Biofilms —
> Gauss-Point Coupling of a Data-Driven Constitutive Model*

so that both the Abaqus development and the ANSYS integration fall within scope. I am happy to adjust
the wording — the key point is a **single agreed title** before Page 1.

**Could you help me with three things?**
1. **Title** — Meisam, since you proposed the topic and will supervise it, could you confirm/adjust
   the title above so Page 1 and the Prüfungsamt registration are consistent? (Prof. Junker, please
   chime in if you would like any change.)
2. **ANSYS access** — how do I obtain a seat for the integration step: an account on the machine
   where Felix's model already runs (license server attached), or an install + license-server details
   from IKM IT? I would like to match Felix's ANSYS version.
3. **Handover** — a short meeting with Felix to align on his FE codebase (and to acknowledge his PhD
   contribution properly in the thesis).

I aim to submit Page 1 in the coming weeks and to begin the formal 6-month period in **August**,
keeping the heaviest implementation outside my NIFE internship window. Does that fit your and Felix's
availability?

Thank you — I am very enthusiastic about where this is going.

Best regards,
Keisuke Nishioka

---
**Pre-send checklist:** confirm Junker's address; fill [Month]; attach
`masterarbeit_ansys_fem/slides/slides_fem_coupling_en.pdf`; optionally attach
`fig_residual_stress_DHvsCH.png`. Keep claims qualitative on stress *magnitude* (shape is calibrated,
γ-magnitude still parametric).
