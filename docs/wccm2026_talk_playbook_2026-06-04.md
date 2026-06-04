# WCCM 2026 — TALK-LEVEL playbook (2nd pass, 2026-06-04)

Deep-mine of individual talks/abstracts in the top target minisymposia (2nd pass).
Companion to `wccm2026_playbook_2026-06-04.md`. Congress: Munich, 19–24 Jul 2026.

## 1. Data caveat — verified vs inferred
| MS | Topic | Talk-list status |
|---|---|---|
| **MS372** (phase-field fracture, under MS297 area) | PF fracture | **MS372D fully verified (6 named talks, fetched)**; MS372A/B/C/E times+chairs verified, speakers not yet public |
| MS297 | PF modeling (fluid sub-sessions) | organizer-inferred only |
| MS096 | UQ in Materials Science | 1 talk verified (Fric), rest inferred |
| MS106 | MOR × ML | 1 title verified (Choi DD-FEM), rest inferred |
| MS237 | SciML for DEs | inferred only |
| MS083 | Multiscale homogenization (FE²) | inferred only |
| MS266 | Computational fracture (JP) | inferred only |

**Bottom line:** only **MS372D** has a verified speaker program. All other "talks" below are inferred from the named organizer's 2024–2026 papers (each cites a real paper; actual WCCM titles may differ) — read as "who's in the room + what they work on," not a confirmed schedule. Re-fetch session pages ~early July to confirm.

## 2. Must-attend talks (ranked)
| # | Talk / presenter | MS | Status | Why + asset |
|---|---|---|---|---|
| 1 | **Heinzmann, Vicentini, Carrara, De Lorenzis — Variational PF Cohesive Fracture in Elastodynamics** | MS372D (Mon 20 Jul 17:00, ICM-13a) | VERIFIED | strength surface decoupled from ℓ = the exact Gc/ℓ/σc identifiability problem. → TMCMC→fracture |
| 2 | **De Lorenzis — variational PF cohesive fracture / nucleation (overview)** | MS372A (Mon 20 Jul 11:30) | inferred(high) | field's voice on what's *open* in ℓ-identifiability; best orientation talk pre-Keio |
| 3 | **Kiyani, Manav, De Lorenzis, Karniadakis — DeepONet (KAN trunk) for crack nucleation/propagation** (arXiv:2501.00016) | MS372A/B | inferred(high) | operator learning *on* PF fracture = FOL precedent. Ask: fixed or variable ℓ? |
| 4 | **Wada / Muraoka — PDE-discovery-regularized NN for crack propagation** (IJNME 2025, nme.7665) | MS266 | inferred | SINDy/AI-Feynman PDE discovery as regularizer; closest published FOL analog; Terada/JSCES network |
| 5 | **Terada group — RBF-surrogate FE² for elastoplastic composites** (CMAME 436, 2025) | MS083 | inferred | DB-interpolation surrogate = posterior-emulator; **Terada = Muramatsu mentor (warmest intro)** |
| 6 | **ten Eikelder — mixture-aware N-phase Navier-Stokes-Cahn-Hilliard closure** (arXiv:2604.27999) | MS297 fluid | inferred(high) | thermodynamically-consistent multiphase PDE → upgrades heuristic Darcy-CH biofilm PDE |
| 7 | **Riccius, Rocha, van der Meer — active-learning + MCMC for Bayesian calibration** (arXiv:2411.13361) | MS083 | inferred | forward model (not sampler) is bottleneck; emulator+sampler co-evolve. Citable motivation |
| 8 | **Lipton — Dynamic Quasi-Brittle Fracture: A Blended Approach** | MS372E (Tue 21 Jul 09:45, ICM-13a) | VERIFIED | nonlocal/peridynamic-blended PF → better-conditioned posterior on ℓ |

## 3. People to meet (ranked; Muramatsu network first)
1. **Kenjiro Terada (Tohoku)** — RBF-surrogate FE² (CMAME 2025). *Muramatsu's mentor → top priority.* Ask: "Do you propagate micro-uncertainty to macro QoIs / tried TMCMC over the RBF coefficients?" Bring the NIFE sign-prior one-pager as an effective-constitutive-law analog.
2. **Yoshitaka Wada (Kindai)** — PDE-discovery NN (IJNME 2025). JSCES, FOL precedent. Ask about regime-change rediscovery + 3D generalization.
3. **Laura De Lorenzis (ETH)** — MS372A. CP+PF overlaps Muramatsu; open positions. Ask: "Bayesian posterior on ℓ from noisy load–displacement? best dataset (Damage Mechanics Challenge)?"
4. **Hiroshi Okada (Tokyo Univ Science)** — redefined 3D J-integral (TAFM 2024). Senior JSCES/APACM, possible letter-writer.
5. **Heinzmann / Carrara (ETH, De Lorenzis grp)** — MS372D implementers. Ask: "ℓ/ℓ_ch identifiable from mode-I alone? solver FEniCSx or JAX?"
6. **ten Eikelder (TU Darmstadt)** — MS297 fluid. Ask: mixture reduction → Darcy-CH biofilm?
7. **Hector Gomez (Purdue)** — Phase-Field DeepONet creator. Ask: "one DeepONet queried at unseen ℓ?" (= the FOL setup).
8. **Masayuki Yano (Toronto)** — component-based hyperreduced ROM (MS106). Ask: certified LOO-CV error bounds via component fidelity.

## 4. Talk-level ideas to carry home (ranked)
1. **PDE-discovery regularization for physics-constrained surrogates** (Wada/Muraoka, MS266) → discover gLV/Hamilton ODE from trajectories as a regularizer; extend to the 1D biofilm PDE. *Advances FOL + gLV interpretability.*
2. **Decoupled (Gc, σc, ℓ) → identifiable Bayesian fracture calibration** (Vicentini/De Lorenzis JMPS 2025, arXiv:2506.12188) → TMCMC on synthetic 3PB/DIC; separate posteriors. *= first Keio paper topic.*
3. **Active-learning-guided MCMC (emulator+sampler co-evolve)** (Riccius/van der Meer, arXiv:2411.13361) → corrects naive "LHS emulator then TMCMC". *Surrogate-accelerated TMCMC.*
4. **Thermodynamically-consistent mixture closure for the biofilm PDE** (ten Eikelder & Brunk, arXiv:2604.27999) → derive Darcy-CH biofilm from reduction axiom. *Physical justification of the spatial PDE chapter.*
5. **Variationally-correct (FOSLS) operator learning with computable error bound** (RBNO, arXiv:2512.21319) → FOSLS loss = convergence certificate; predict reduced-basis coeffs. *Certified FOL + PDE-verification work.*
6. **Symmetry-equivariant GNN/neural-operator surrogates over microstructures** (Hendriks/Geers SimEGNN, arXiv:2404.17365) → equivariance by architecture. *Data-efficient FOL design for CP RVEs.*

## 5. Six-week prep list (before 19 Jul)
**Read:** Wk1 — arXiv:2506.12188 (tunable strength surface) + Heinzmann elastodynamics; IJNME nme.7665 (Wada PDE-discovery). Wk2 — arXiv:2501.00016 (DeepONet/KAN fracture), arXiv:2302.13368 (PF-DeepONet), arXiv:2411.13361 (AL+MCMC). Wk3 — arXiv:2604.27999 (mixture NSCH), arXiv:2512.21319 (RBNO). Wk4 — CMAME 436/117708 (Terada RBF-FE²), arXiv:2404.17365 (SimEGNN), TAFM 130 2024 (Okada J-integral).

**Artifacts (Wk5–6):**
- NIFE one-pager: sign-prior gLV A reframed as an "effective constitutive law inferred by TMCMC" (bridge for Terada/Wada/De Lorenzis).
- **Toy demo: TMCMC calibrating (Gc, σc, ℓ) on a synthetic 1D PF bar — ALREADY EXISTS in `~/pf-fracture-uq` (just polish + the decoupled-σc parameterization from idea #2).**
- Question card (the §3 questions) grouped by room (ICM-13a Mon/Tue for MS372).
- ~Early July: re-fetch MS372A/B/C/E, MS297-fluid, MS083, MS106, MS237, MS266 pages to confirm slots.

**Scheduling:** MS372 anchors **Mon 20 Jul (ICM-13a, all day) + Tue 21 Jul AM** = densest day. Prioritize verified PF-fracture over inferred parallel sessions on clashes.

---
*Provenance: workflow `wccm-2ndpass-talkmine` (run wf_2906705b-137), 7 agents, 2026-06-04. Re-confirm inferred talks against the final program before relying on them.*
