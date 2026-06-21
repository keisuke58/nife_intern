# Paste-ready thesis snippets — FEM literature validation + GDI scope (2026-06-16)

> **Freeze respected.** Nothing in `/home/nishioka/LUHsummer26/30_Masterarbeit` is edited. These are
> paste-ready LaTeX fragments to drop into the **Overleaf** canonical copy. Backed by
> `notes/fem_literature_validation_2026-06-16.md` (verified citations).

Audit of the thesis found: the ch5 FEM material table is **uncited**; micromotion and RANKL wording are
**already appropriately cautious** (no over-claim to fix); ch6 Outlook already frames GDI as an
*ordering*. So only two edits add value — material citations, and one GDI-scope sentence.

---

## 1. ch5_integration.tex — cite the material table (≈ line 621–624)

REPLACE:
```latex
implant finite-element analysis; the periodontal ligament (PDL) and the biofilm are
deliberate linear idealisations of nonlinear/uncertain materials,
```
WITH:
```latex
implant finite-element analysis \cite{Geng2001,Sevimay2005}; the periodontal ligament
(PDL) takes the linear secant modulus of \citet{ReesJacobsen1997} and the dentin value
of \citet{Kinney2003}. The PDL and the biofilm are deliberate linear idealisations of
nonlinear/uncertain materials (the true PDL is nonlinear \cite{Cattaneo2005}),
```
*(Cancellous $E=1000$\,MPa sits between Sevimay's D3$=1600$ and D4$=690$\,MPa; it is a D3-leaning
"type III–IV" value. Optionally add "(type III; a D4 $\approx 690$\,MPa case bounds the soft end)".)*

## 2. ch6_conclusion.tex — GDI ordinal-scope sentence (Outlook "clinical translation", ≈ line 104–108)

The paragraph already says the index is a "guild-level dysbiosis **ordering**". APPEND after the
"...commensal–dysbiotic axis" sentence:
```latex
Crucially, this index is interpretable only as a \emph{relative ordering}, not on an
absolute scale: on the longitudinal periodontitis cohort every patient scores a
class-level $\mathrm{GDI}<0$ (the abundant commensal classes \emph{Streptococcus},
\emph{Veillonella} and \emph{Actinomyces} numerically dominate even in disease), so a
fixed $\mathrm{GDI}=0$ threshold would mis-classify the entire diseased cohort as stable.
The predicted bone-loss \emph{ranking} is nonetheless invariant to the threshold choice
(Spearman $\rho=0.97$ between a cohort-relative and an absolute cut), so a deployable index
must be anchored to a healthy-cohort reference rather than to an absolute zero.
```
*(Figure available: `masterarbeit_ansys_fem/figures/fem_periimplantitis_gdi_threshold.pdf` — can be
cited as the supporting sensitivity analysis if a figure is wanted.)*

## 3. references.bib — add these (verified)

```bibtex
@article{Geng2001, author={Geng, Jian-Ping and Tan, Keson B.C. and Liu, Gui-Rong},
  title={Application of finite element analysis in implant dentistry: a review of the literature},
  journal={The Journal of Prosthetic Dentistry}, volume={85}, number={6}, pages={585--598}, year={2001}}

@article{Sevimay2005, author={Sevimay, M. and Turhan, F. and K{\i}l{\i}{\c{c}}arslan, M.A. and Eskitascioglu, G.},
  title={Three-dimensional finite element analysis of the effect of different bone quality on stress distribution in an implant-supported crown},
  journal={The Journal of Prosthetic Dentistry}, volume={93}, number={3}, pages={227--233}, year={2005}}

@article{ReesJacobsen1997, author={Rees, J.S. and Jacobsen, P.H.},
  title={Elastic modulus of the periodontal ligament},
  journal={Biomaterials}, volume={18}, number={14}, pages={995--999}, year={1997}}

@article{Cattaneo2005, author={Cattaneo, P.M. and Dalstra, M. and Melsen, B.},
  title={The finite element method: a tool to study orthodontic tooth movement},
  journal={Journal of Dental Research}, volume={84}, number={5}, pages={428--433}, year={2005}}

@article{Kinney2003, author={Kinney, J.H. and Marshall, S.J. and Marshall, G.W.},
  title={The mechanical properties of human dentin: a critical review and re-evaluation of the dental literature},
  journal={Critical Reviews in Oral Biology \& Medicine}, volume={14}, number={1}, pages={13--29}, year={2003}}
```
*(For the micromotion 50–150\,µm range and the GDI guild-split grounding, the full verified entries —
Pilliar 1986, Szmukler-Moncler 1998, Gevers 2014, Socransky 1998, Abusleme 2013 — are listed in
`notes/fem_literature_validation_2026-06-16.md` §5 if those claims are added to the thesis later.)*

## 4. No change needed (already correct in thesis)
- **Micromotion**: ch5 mentions the de-integrated variant but states no "150\,µm" threshold in the body,
  so there is no Brunski misattribution to fix. (The 50–150\,µm range only needs a cite if you add the
  numeric claim to the text.)
- **RANKL/OPG**: ch5 (l.1146, l.1205) frames it as a mechanistic balance driving osteoclastic loss — not
  as an "established biomarker / master switch" — so the cautious wording is already in place.
