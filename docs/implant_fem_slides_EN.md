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

\begin{center}
\includegraphics[width=0.78\textwidth,height=0.5\textheight,keepaspectratio]{masterarbeit_ansys_fem/figures/F2_growth_calibration.pdf}
\end{center}

\footnotesize Calibration of the anisotropic depth-growth strain $\varepsilon_{zz}=\beta\,\phi_{Pg}$ from CLSM: dysbiotic biofilm swells with P. gingivalis (DH $\beta\approx+17$, $r=0.88$); commensal does not (CH $\beta\approx-10$). (B) shows the source z-stack thicknesses. \normalsize

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

## Interface mechanics: the thread generates the stress

\begin{center}
\includegraphics[height=0.54\textheight,keepaspectratio]{masterarbeit_ansys_fem/figures/fem_implant_thread.pdf}
\end{center}

\footnotesize Coupled implant–thread interface model (CPE4, real V-thread profile). (A) the
von Mises field concentrates at the bonded thread flanks; (B) the interior-peak
interface stress is mesh-converged (the free-edge corner singularity is
excluded — it diverges with refinement, the interior plateau does not); (C) the
thread geometry itself generates the stress — a flat interface stays near
baseline; (D) dysbiotic (DH) vs commensal (CH) biofilm scales it. Interior peak
$\sigma_\mathrm{vM} \approx 0.6$–$0.7$ GPa. \normalsize

---

## 3D hero: full peri-implant assembly

\begin{center}
\includegraphics[height=0.5\textheight,keepaspectratio]{masterarbeit_ansys_fem/figures/fem_implant_paraview_oblique_annot.png}\hfill
\includegraphics[height=0.5\textheight,keepaspectratio]{masterarbeit_ansys_fem/figures/fem_implant_crown_section.pdf}
\end{center}

\footnotesize Left: physically-based (PBR) ParaView render of the full
peri-implant assembly (implant + crown + alveolar bone), oblique view,
annotated. Right: updated crown–implant cross-section (load path through the
abutment). The occlusal load concentrates at the **crestal peri-implant bone**
(marginal-bone-loss signature). The composition mechanism **transfers
unchanged** into realistic anatomy. \normalsize

---

## Crown load transfer: the moment-arm effect

![](masterarbeit_ansys_fem/figures/fem_implant_crown_fem.pdf){ height=50% }

A real-tooth ceramic crown on the abutment carries the bite at the **occlusal
table (z≈38, the neighbour's plane)**, raising the bending moment at the implant
neck **~3.4×**. Even under physiological occlusion (100 N, 30°, ISO 14801) the
**crestal peri-implant bone stress (p95) rises ×1.5 (18→27 MPa)**.
$\Rightarrow$ prosthetic design (crown height, load position) **directly drives marginal-bone risk**.

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

## Disease-model calibration: RANKL/OPG vs clinical GCF

\begin{center}
\includegraphics[height=0.54\textheight,keepaspectratio]{masterarbeit_ansys_fem/figures/fem_periimplantitis_rankl_opg_calib.pdf}
\end{center}

\footnotesize The biology$\rightarrow$bone-loss link is anchored to clinical data. The
RANKL/OPG inflammatory switch is recalibrated so its dysbiotic fold-change
matches periodontitis crevicular-fluid (GCF) measurements (median $\approx 17×$;
Bostanci / consensus band). The model's RANKL/OPG-ratio $\leftrightarrow$
marginal-bone-loss correlation ($r=0.99$) brackets the clinical value
($r=0.81$), and reversing the dysbiosis (treatment) restores the ratio — the
mechanical driver is now clinically calibrated. \normalsize

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

---

## Appendix: FEM conditions (materials, BCs, loads)

| Material | $E$ (MPa) | $\nu$ |
|---|---:|---:|
| Titanium (Ti-6Al-4V) | 110 000 | 0.34 |
| Cortical / cancellous bone | 13 700 / 1 000 | 0.30 |
| Dentin | 18 000 | 0.31 |
| PDL (0.25 mm, linear idealisation) | 50 | 0.45 |
| Crown (lithium-disilicate / zirconia) | 95 000 / 210 000 | 0.30 |
| Gingiva (mucosa) | 3 | 0.45 |
| Biofilm (+ growth $\varepsilon_g=0.19$) | 1.0 | 0.45 |

\footnotesize
**BCs:** bone fully fixed on the crop faces ($U_{1,2,3}=0$). **Interfaces:** osseointegration = \*TIE, crown tied to abutment top, frictional contact $\mu=0.3$ only for the de-integrated variant.
**Load:** ISO 14801, 100 N at 30$^\circ$ (lat/ax = 0.577). Step 1 = growth eigenstrain, Step 2 = occlusal (with crown: at the **occlusal table z=38** = moment arm).
**Elements:** assembly = C3D4 (${\sim}270$k), ISO-14801 coupon = C3D10 (linear under-predicts the concentration by 9–15%), convergence $\pm7\%$. Units mm-N-MPa. Full spec in `FEM_CONDITIONS.md`.
\normalsize
