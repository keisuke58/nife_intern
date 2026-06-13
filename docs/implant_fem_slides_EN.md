---
title: "Implant FEM: Composition-Resolved Mechanics of the Peri-Implant Biofilm"
subtitle: "Composition $\\varphi_{Pg} \\rightarrow$ growth eigenstrain $\\rightarrow$ implant-interface residual stress $\\rightarrow$ detachment"
author: "Keisuke Nishioka — NIFE / SFB TRR-298"
date: "2026-06-13"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## Where this deck sits

**Implant FEM** — composition-resolved mechanics of the peri-implant biofilm.

Data flow:

raw 16S → guild $\varphi$ → gLV/Hamilton (+ sign prior) → spatial PDE → **FEM mechanics**

Supervisor context:

- **Soleimani** (IKM, FEM) — continuum-mechanics / finite-element framework
- **Junker** (IKM, Hamilton principle) — variational material-model formulation
- **Szafrański** (MHH / NIFE) — oral biofilm / clinical

\vspace{0.4em}
**One-line thesis:** dysbiosis is read out as a *mechanical* implant-interface risk.

---

## Clinical problem: peri-implantitis

Peri-implantitis is biofilm-driven inflammation that drives loss of the bone
supporting the implant.

- Affects **20−35%** of implant patients — leading cause of implant failure.
- **NIFE** = implant R&D centre. The substratum is **titanium**.
- **Dieckow cohort** = abutment biofilm followed longitudinally.

\vspace{0.4em}
Mechanically: the biofilm grows at the Ti interface, generates interface
**stress**, and eventually **detaches**. We quantify this with continuum mechanics.

---

## The idea: a composition-to-mechanics chain

The central causal chain:

$$\varphi_{Pg}\;\rightarrow\;\text{eigenstrain}\;\rightarrow\;\text{interface stress}\;\rightarrow\;\text{detachment}\;\rightarrow\;\text{peri-implantitis}$$

![](masterarbeit_ansys_fem/figures/fem_clinical_schematic.pdf){ height=44% }

Ti implant thread under CH (commensal) vs DH (dysbiotic) biofilm: growth-induced interface stress and delamination.

---

## Method: a Hamilton-principle material model

At each Gauss point we invoke a Python Hamilton-principle material model
**instead of the phenomenological law**.

- Multiplicative growth split $F = F_e\,F_g$ + neo-Hookean elasticity.
- Growth $F_g$ is driven by a composition-($\varphi_{Pg}$)-dependent eigenstrain.
- The elastic response $F_e$ carries the stress, stored as residual stress at the interface.

\vspace{0.3em}
**Verification (Abaqus):**

- Small strain: matches $\sigma = E\,\varepsilon$.
- Finite strain (NLGEOM): matches the neo-Hookean solution.

\vspace{0.3em}
$\Rightarrow$ the material law drops unchanged into an external FEM solver (verified as a kernel).

---

## Growth calibration

The growth eigenstrain is calibrated from **CLSM depth profiles**.

![](masterarbeit_ansys_fem/figures/F2_growth_calibration.pdf){ height=52% }

- DH: depth correlation $0.88$, $\beta_{\text{depth}} \approx 17.2$.
- The commensal control has the **opposite sign** — not mere volumetric swelling.

---

## Result: residual-stress column $S_{11}(z)$

![](masterarbeit_ansys_fem/figures/F4_residual_stress_DHvsCH.pdf){ height=54% }

- Stress concentrates at the **substratum** (implant-abutment wall); the free top $\approx 0$.
- $S_{11}(z)$ **tracks** $\varphi_{Pg}(z)$ — deep Pg load builds the interface stress.

---

## Result: delamination (cohesive interface)

![](masterarbeit_ansys_fem/figures/F6_delamination_DHvsCH.pdf){ height=52% }

Cohesive-interface FEM with **identical** interface strength:

- **dysbiotic 83%** vs **commensal 25%** of interface delaminated (**3.3×**).
- Sole cause = the **Pg depth profile** (strength and load held identical).

---

## 3D hero: full peri-implant assembly

![](masterarbeit_ansys_fem/figures/fem_implant_screw3d.pdf){ height=50% }

Implant + neighbouring tooth + shared alveolar bone. The occlusal load
concentrates at the **crestal peri-implant bone** (marginal-bone-loss
signature). The composition mechanism **transfers unchanged** into realistic anatomy.

---

## 2D coupled section (supporting)

![](masterarbeit_ansys_fem/figures/fem_tier2b_real.pdf){ height=52% }

Coupled implant+tooth+bone mesiodistal section. (A) multi-material anatomy,
(B) occlusal von Mises shared-bone coupling — **crestal stress = peri-implant
marginal-bone-loss signature**.

---

## Patient bridge

![](masterarbeit_ansys_fem/figures/F1c_patient_bridge_validation.pdf){ height=48% }

Projecting the 10 Dieckow patients through the verified detachment kernel $J$:

- Inter-patient spread **$CV = 0.86$** is set by the *measured* Pg load.
- The borrowed in-vitro depth shape is a **34× weaker lever** ($\pm 8\%$).

\vspace{0.2em}
$\Rightarrow$ a **0-D clinical Pg readout** is an adequate ordinal detachment-risk surrogate.

---

## Validation, limits & outlook

**Honest assessment:**

- The growth **shape** is calibrated from CLSM.
- The stress **magnitude** is parametric (biofilm modulus spans $\approx 7$
  orders → reported as $\sigma/\mu$).
- The claim is **ordinal / mechanistic**, not predictive. The differentiation
  is the *biofilm coupling*, not generic implant biomechanics.

**Validating experiment:** depth-resolved AFM / shear-flow detachment vs Pg load.

\vspace{0.3em}
**Outlook:** the growth-eigenstrain → interface-stress → delamination competency
transfers to Keio / Muramatsu computational solid mechanics and to semiconductor
thin-film stress / dicing-street delamination (DISCO).
