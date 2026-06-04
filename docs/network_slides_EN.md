---
title: "Reading the Interaction Matrix as a Network"
subtitle: "Beyond fitting A — keystones, bridges, trophic layers, and rewiring"
author: "Keisuke Nishioka — NIFE"
date: "2026-06-03"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## Where this deck sits

The project = three pillars + a spatial extension. Data flow:

raw 16S → guild $\varphi$ → gLV/Hamilton (+ sign prior) → LOO validation → spatial PDE

The decks (you-are-here in **bold marker**):

- **Overview** (umbrella) — the whole picture, three pillars
- **AGORA** — metabolism → sign prior (input to the ecological model)
- **Dieckow** — in-vivo interaction inference & validation (the model)
- **Network** — structural analysis of the interaction matrix $A$  — **(you are here)**
- **Spatial-PDE** — reaction-diffusion of the FISH depth profiles
- **FISH pipeline** — .lif → 5-species depth composition

---

## Motivation: from fitting $A$ to reading it

Beyond inferring the interaction matrix $A$, read its **graph structure**.

- **keystone** — a taxon that governs the community despite low abundance (high centrality).
- **bridge** — a mediator linking trophic layers (high betweenness).
- **trophic layer** — producer $\to$ consumer hierarchy of cross-feeding.
- **rewiring** — edge signs that flip between health and dysbiosis.

\vspace{0.4em}
The textbook picture of oral biofilms is "*P. gingivalis* = keystone, *F. nucleatum* = bridge."
This deck tests whether that picture is **supported** at the in-vivo class level, using
network statistics.

---

## Gauge invariance: read $A_{\text{eff}}$, not $A$

The replicator dynamics
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big],\qquad \sum_i\varphi_i=1$$
are invariant under **column-wise constant shifts** $A_{ij}\to A_{ij}+c_j$ (they cancel on $\sum_i\varphi_i=1$).

$\Rightarrow$ the sign/magnitude of the raw $A_{ij}$ are **gauge-dependent** and meaningless. Only the
**column-centered** matrix
$$A_{\text{eff}}[i,j] = A_{ij} - \frac{1}{S}\sum_{k} A_{kj}$$
is gauge-invariant. **All sign and centrality interpretation below uses $A_{\text{eff}}$.**

Define a directed weighted graph $G=(V,E)$ with $w_{ij}=A_{\text{eff}}[i,j]$ ($|V|=S=10$ guilds).

---

## Centrality definitions

Compute several centralities from $A_{\text{eff}}$:

- **eigenvector centrality** — leading eigenvector $v$ of $A\,v = \lambda\,v$;
  recursively, "a taxon is central if it connects to central taxa."
- **betweenness** — fraction of shortest paths through vertex $i$,
  $$g(i)=\sum_{s\neq i\neq t}\frac{\sigma_{st}(i)}{\sigma_{st}}.$$
- **PageRank** — stationary law $\pi = (1-d)\,\mathbf{1}/S + d\,M\pi$ ($M$ = column-normalised transition).
- **in/out strength** — $\;s^{\text{in}}_i=\sum_j |A_{\text{eff}}[i,j]|,\quad s^{\text{out}}_j=\sum_i |A_{\text{eff}}[i,j]|.$

---

## Centrality results (Dieckow 10-guild, no-prior consensus)

![](results/guild_network/guild_centrality_summary.png){ height=50% }

Eigenvector centrality: Bacilli **0.61**, Actinobacteria **0.58**, Betaproteobacteria 0.44,
Negativicutes 0.22, Bacteroidia ($\approx$ *P. gingivalis*) 0.19.
Betweenness: Bacilli **0.78**, Actinobacteria 0.22, all others 0.
Mean abundance: Bacilli **0.547** (dominant), Actinobacteria 0.203, Bacteroidia 0.047.

---

## Keystone / bridge test: does the textbook hold?

Cross-check centrality against abundance ranks:

- *P. gingivalis* (Bacteroidia) is eigen-centrality rank $\sim$**4** and abundance rank $\sim$**5**.
  $\Rightarrow$ **not a structural keystone** (low abundance and low centrality).
- the bridge (highest betweenness) is **Bacilli (*Streptococcus*)**, **not Fusobacterium**.

\vspace{0.3em}
$\Rightarrow$ the classic "Pg-keystone / Fn-bridge" picture is **unsupported at the in-vivo class level**.

\vspace{0.3em}
\textcolor{red}{Caveat (do not over-read): class $\neq$ species; a dense graph degenerates
betweenness; rare guilds are noisy.}

---

## Influence vs vulnerability (out- vs in-strength)

![](results/guild_network/influence_vulnerability.png){ height=55% }

Out-strength $s^{\text{out}}_j=\sum_i|A_{\text{eff}}[i,j]|$ measures the **influence** $j$ exerts on others;
in-strength $s^{\text{in}}_i=\sum_j|A_{\text{eff}}[i,j]|$ measures the **vulnerability** of $i$.
Their asymmetry separates drivers from driven taxa.

---

## Trophic coherence (directed AGORA cross-feeding)

On the directed cross-feeding graph, measure MacKay's **trophic incoherence** $F_0$
(assign trophic levels $h_i$; variance of level gaps across edges):
$$F_0 = \frac{1}{w}\sum_{ij} W_{ij}\,(h_i - h_j - 1)^2 .$$

- Observed $F_0 = \mathbf{0.652}$ vs random $0.646$, $p=0.50$.
- $\Rightarrow$ **trophically incoherent** = no hierarchy, **cyclic mutual cross-feeding**.

\vspace{0.3em}
Biologically sensible: Fusobacterium is a basal producer and Veillonella a high consumer, yet
they entangle in loops rather than forming a clean one-way food chain.

---

## Concordant backbone: ecology layer $\times$ metabolic layer

Overlay two independent layers — the ecological (Hamilton) layer and the AGORA metabolic layer —
and keep edges where both **agree in sign**:

- **11 edges, all positive (facilitation)**, $p=4\times10^{-4}$ (the same independent validation as the AGORA deck).
- Negativicutes (*Veillonella*) is the main metabolic **sink**:
$$\{\text{Bacilli, Bacteroidia, }\beta\text{Proteo, Fusobacteria}\}\ \longrightarrow\ \text{Negativicutes}.$$
- This corresponds to reconstructed **lactate cross-feeding**.

\vspace{0.3em}
**The strongest network result.** The sign-level backbone is supported independently by both layers.

---

## LOO stability of the network

![](results/guild_network/loo_stability.png){ height=55% }

Re-estimate $A_{\text{eff}}$ under leave-one-patient-out and measure each pair's sign-consistency.
**All pairs have sign-consistency $\geq 0.70$** $\Rightarrow$ the network's sign structure is robust to
individual patients.

---

## Permutation test of edge magnitudes (honestly)

![](results/guild_network/permutation_test.png){ height=48% }

Compare each off-diagonal edge **magnitude** to a permutation null:
**only 2 of 90 pairs** individually exceed it at $p<0.05$.

\vspace{0.3em}
$\Rightarrow$ the network is dense and individual edge **magnitudes are weakly resolved**.
What is robust is the **sign-level backbone** of the previous slide, not the individual magnitudes.

---

## CS $\leftrightarrow$ DH rewiring (Heine 5-species, posterior)

Compare the four-attractor posterior of the **(1) Heine 5-species GPU-Bayesian TMCMC**
(CS = commensal-static, DH = dysbiotic-HOBIC, `ultimate_10000p`) in the
column-centered $A_{\text{eff}}$. Define the **facilitation probability**
$$P_{\text{facilitation}}(i\!\leftarrow\! j) = \Pr\big[\,A_{\text{eff}}[i,j] > 0\,\big]$$
(the posterior probability that $A_{\text{eff}}[i,j]>0$).

$P_{\text{fac}}\to 1$ means facilitation; $\to 0$ means competition / suppression.
How $P_{\text{fac}}$ moves between CS and DH quantifies the rewiring.

---

## Rewiring results

From CS $\to$ DH the facilitation probability $P_{\text{fac}}$ changes dramatically:

- **So $\leftrightarrow$ Vd mutualism collapses**: *Streptococcus*–*Veillonella* symbiosis becomes
  competition in dysbiosis.
  $$P_{\text{fac}}(\text{So}\!\to\!\text{Vd}):\ 1.00 \to 0.03,\qquad
    P_{\text{fac}}(\text{Vd}\!\to\!\text{So}):\ 0.9995 \to 0.00.$$
- **Fusobacterium becomes supported**: An$\to$Fn and Vd$\to$Fn flip competition $\to$ facilitation.
  $$P_{\text{fac}}(\text{An}\!\to\!\text{Fn}):\ 0.23 \to 0.97,\qquad
    P_{\text{fac}}(\text{Vd}\!\to\!\text{Fn}):\ 0.09 \to 0.77.$$

---

## *P. gingivalis* centralisation

Eigenvector centrality: CS **0.32** $\to$ DH **0.51** (highest in DH).

- Yet in DH, Pg is net suppressive: the $P_{\text{facilitation}}$ of its outgoing edges is $0.27$.
- $\Rightarrow$ Pg is **not a static keystone**, but **becomes keystone-like** in the dynamic
  dysbiotic state.

\vspace{0.3em}
Consistent with the FISH finding: in dysbiosis Pg sinks deep, loses neighbour-dependence, and
**goes autonomous (centralises)**. A keystone is a function of state, not a fixed attribute.

---

## Conclusions

1. **The metabolic backbone is robust**: lactate cross-feeding with Veillonella as the main sink is
   supported in sign independently by the ecological and metabolic layers
   (11 edges, all positive, $p=4\times10^{-4}$).
2. **The textbook keystone is not supported**: at the in-vivo class level Pg is not a structural
   keystone, and the bridge is Bacilli, not Fusobacterium.
3. **Dysbiosis is a rewiring, not a mere abundance shift**: loss of So–Vd mutualism plus Pg centralisation.
4. Individual edge magnitudes are weakly resolved (2/90 by permutation) — only the sign structure is robust.
   class $\neq$ species, dense-graph degeneracy, and rare-guild noise remain limitations.
