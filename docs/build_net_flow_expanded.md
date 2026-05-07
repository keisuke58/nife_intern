# `build_net_flow_expanded.py` — How Signs Are Estimated

> Builds a **10×10 signed interaction matrix** between microbial guilds,
> used as a Bayesian sign-prior in the Hamilton ODE model.

---

## 1. The Core Idea in One Sentence

> **If guild A produces a metabolite that guild B consumes → A helps B → net_flow[B, A] > 0**

The matrix encodes *who feeds whom* (and *who poisons whom*) based on published microbe–metabolite relationships.

---

## 2. What the Matrix Looks Like (L1+L2 only, actual values)

```
              Actin.  Bacil.  Bact.   β-Prot. Clost.  Corio.  Fusob.  γ-Prot. Negat.  Other
Actinobact.     0.0     2.0    2.0     0.0     0.0     0.0     0.0     0.0     5.5     0.0
Bacilli         4.0     0.0    2.0     2.0     0.0     0.0     0.0     2.0     4.0     0.0
Bacteroidia     3.5     0.0    0.0     1.5     0.0     0.0     0.0     0.0     3.5     0.0
β-Proteobact.   4.0     4.0    2.0     0.0     0.0     0.0     0.0     2.0     6.0     0.0
Clostridia      0.0     0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
Coriobacteriia  0.0     0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
Fusobacteriia   0.0     2.0    0.0     0.0     0.0     0.0     0.0     0.0     2.0     0.0
γ-Proteobact.   0.0     0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
Negativicutes   2.0     2.0    2.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
Other           0.0     0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0

  row = "who benefits"    col = "who provides"
  net_flow[i, j] > 0  →  guild j  helps  guild i
  net_flow[i, j] < 0  →  guild j  harms  guild i
  net_flow[i, j] = 0  →  no evidence (prior: unconstrained)
```

**Notable**: Clostridia, Coriobacteriia, γ-Proteobacteria, Other are all zero — no guild in the dataset was mapped to these.

---

## 3. The Raw Input: Szafranski Supplementary File

The Excel file (351 rows) records one microbe–metabolite relationship per row:

```
TAXON                    RELATIONSHIP    OBJECT         EVIDENCE      HMDB_ID
─────────────────────────────────────────────────────────────────────────────
Abiotrophia defectiva    PRODUCES        lactic acid    experimental  HMDB0000190
Veillonella parvula      USES            lactic acid    experimental  HMDB0000190
Actinomyces israelii     IS_INHIBITED_BY oxygen         experimental  HMDB0001377
Streptococcus gordonii   PRODUCES        lactic acid    experimental  HMDB0000190
Aggregatibacter aphro.   USES            lactic acid    prediction    HMDB0000190
```

### Relationship types in the file

| RELATIONSHIP | Count | Treated as | Handled? |
|---|---|---|---|
| PRODUCES | 173 | producer | ✓ |
| USES | 143 | consumer | ✓ |
| IS_INHIBITED_BY | 22 | inhibition target | ⚠️ see §6 |
| DEGRADES | 4 | consumer | ✓ |
| RELEASES | 4 | producer | ✓ |
| HYDROLYSES | 2 | consumer | ✓ |
| DEPENDS_ON | 1 | consumer | ✓ |
| IS_A_HOST_FOR | 2 | — | ✗ silently dropped |

### Object types in the file

| OBJECT_TYPE | Count | Note |
|---|---|---|
| metabolite | 277 | main signal |
| enzyme | 65 | treated identically to metabolite ⚠️ |
| biofilm component | 4 | treated identically |
| taxon | 3 | treated identically |
| cell component | 2 | treated identically |

---

## 4. Step-by-Step: How One Metabolite Becomes Matrix Values

Using **lactic acid** as a concrete example:

### Step 1 — Group all rows by metabolite

```
Metabolite: lactic acid
┌────────────────────────────────┬─────────────┬──────────────┬──────────┐
│ TAXON                          │ RELATIONSHIP│ EVIDENCE     │ HMDB     │
├────────────────────────────────┼─────────────┼──────────────┼──────────┤
│ Streptococcus gordonii (Bacil.)│ PRODUCES    │ experimental │ present  │ → w=2.0
│ Streptococcus sanguinis (Bacil)│ PRODUCES    │ experimental │ present  │ → w=2.0
│ Prevotella nanceiensis (Bact.) │ PRODUCES    │ experimental │ present  │ → w=2.0
│ Veillonella parvula (Negat.)   │ USES        │ experimental │ present  │ → w=2.0
│ Arachnia propionica (Actino.)  │ USES        │ experimental │ present  │ → w=2.0
│ Aggregatibacter (β-Prot.)      │ USES        │ prediction   │ present  │ → w=1.0
│ Veillonella sp. OTU_16 (Negat.)│ USES        │ prediction   │ present  │ → w=1.0
│ ...                            │ ...         │ ...          │ ...      │
└────────────────────────────────┴─────────────┴──────────────┴──────────┘
```

### Step 2 — Compute a single weight `w` for this metabolite

```python
w = max(weight of all rows for this metabolite)
  = max(2.0, 2.0, 2.0, ..., 1.0, 1.0)
  = 2.0
```

> ⚠️ **Important**: `w` is one number for the entire metabolite, not per-row.
> Because lactic acid has at least one experimental+annotated row, **ALL** pairs
> involving lactic acid (including prediction rows) are weighted at **2.0**.

### Step 3 — Identify producer and consumer guilds

```
producers  = { Bacilli, Bacteroidia, Actinobacteria, Negativicutes, ... }
consumers  = { Negativicutes, Actinobacteria, Betaproteobacteria, ... }
```

### Step 4 — Add to pos matrix for every (producer, consumer) pair

```
For each src in producers:
  For each tgt in consumers:
    if src ≠ tgt:
      pos[tgt, src] += 2.0
```

Concrete additions for lactic acid (partial):

```
pos[Negativicutes, Bacilli]      += 2.0   ← Bacilli feeds Negativicutes
pos[Actinobacteria, Bacilli]     += 2.0   ← Bacilli feeds Actinobacteria
pos[Negativicutes, Bacteroidia]  += 2.0   ← Bacteroidia feeds Negativicutes
pos[Betaproteo., Bacilli]        += 2.0   ← Bacilli feeds β-Proteobacteria
...
```

---

## 5. Weight Assignment Rules

```
Each row of Szafranski file gets a per-row weight:

  experimental  +  KEGG or HMDB known  →  2.0   (L1, high confidence)
  experimental  +  no DB annotation    →  1.5   (L1, lower)
  prediction    +  anything            →  1.0   (L2)

Then for each metabolite:
  w = MAX of all per-row weights for that metabolite
```

```
EVIDENCE column = 'experimental'
       │
       ├─ Yes ──► HMDB_ID or KEGG present?
       │            ├─ Yes → w_row = 2.0
       │            └─ No  → w_row = 1.5
       │
       └─ No  ──► w_row = 1.0

w_metabolite = max(w_row for all rows of this metabolite)
```

---

## 6. IS_INHIBITED_BY — What Actually Fires

The inhibition logic is:

```python
for src in producers:       # guild that makes the metabolite
    for tgt in inhibited:   # guild that IS_INHIBITED_BY the metabolite
        neg[tgt, src] += w
```

### Reality check — which inhibition signals fire?

```
22 IS_INHIBITED_BY rows in the file:
  - 20 rows: IS_INHIBITED_BY  oxygen
  - 2 rows:  IS_INHIBITED_BY  hydrogen peroxide

Oxygen:            No guild PRODUCES oxygen → zero neg signals
Hydrogen peroxide: Streptococcus spp. (Bacilli) produce H₂O₂
                   Bacteroidia and Actinobacteria are inhibited by H₂O₂
                   → neg[Bacteroidia, Bacilli]   += 2.0  ✓ fires
                   → neg[Actinobacteria, Bacilli] += 2.0  ✓ fires
```

**Result**: of 22 IS_INHIBITED_BY rows, only **2 pairs** produce a neg signal.
The 20 oxygen rows are effectively dead — the prior table contains no oxygen producer.

This is why `net_flow[Bacteroidia, Bacilli] = 0.0` even though Bacilli produces things
Bacteroidia consumes: the H₂O₂ penalty exactly cancels the lactic-acid benefit.

---

## 7. Three Evidence Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Evidence Sources                              │
│                                                                     │
│  Layer 1 (L1)           Layer 2 (L2)           Layer 3 (L3)        │
│  Szafranski             Szafranski             AGORA2 FBA           │
│  experimental           predicted              genome-scale models  │
│  w = 2.0 / 1.5          w = 1.0                w = 0.5             │
│       │                      │                      │               │
│       └──────────────────────┴──────────────────────┘               │
│                              │                                      │
│                   pos[i,j] and neg[i,j]                             │
│                              │                                      │
│                   net = pos − neg  (10×10)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### L3: AGORA2 FBA Cross-Feeding (when `use_agora=True`)

Each guild's representative SBML model is solved with pFBA under oral-fluid medium:

```
secretion flux > +0.05  →  guild secretes this metabolite
uptake   flux  < −0.05  →  guild consumes this metabolite

For each (j secretes X) × (i uptakes X):
  X is H₂O₂ or H₂S?   → neg[i, j] += 0.5    (j poisons i)
  Otherwise             → pos[i, j] += 0.5    (j feeds i)
```

---

## 8. How This Becomes a Sign Prior

```python
net_flow = build_net_flow_expanded(use_agora=True)

# In guild_replicator_dieckow.py:
sign_prior[i, j] = +1  if net_flow[i, j] > 0   # A[i,j] must be positive
sign_prior[i, j] = -1  if net_flow[i, j] < 0   # A[i,j] must be negative
sign_prior[i, j] =  0  if net_flow[i, j] == 0  # unconstrained

# During TMCMC log-likelihood:
penalty = σ * sum( max(0, -sign_prior[i,j] * A[i,j]) for all i,j )
log_posterior -= penalty
```

The **sign** of `net_flow` matters; the magnitude only determines weight accumulation,
not the eventual +1/−1/0 classification.

---

## 9. Known Limitations

| Issue | Detail |
|---|---|
| **IS_INHIBITED_BY mostly dead** | 20/22 rows are for oxygen; no guild produces oxygen → zero neg signal |
| **Weight is per-metabolite max** | Prediction rows for a well-studied metabolite get bumped to w=2.0 |
| **Enzymes treated as metabolites** | 65 enzyme rows ("fucosidase" etc.) use the same logic as metabolites |
| **IS_A_HOST_FOR silently dropped** | 2 rows with this relationship type are ignored |
| **Magnitude unused** | Only sign(net_flow) enters the prior; accumulated weights are discarded |
| **Many guilds get zero prior** | Clostridia, Coriobacteriia, γ-Proteobacteria, Other: no genus in file maps to them |

---

## 10. Full Data Flow

```
Szafranski Excel (351 rows)
  TAXON × RELATIONSHIP × OBJECT × EVIDENCE × HMDB_ID × KEGG
         │
         ▼
  for each unique metabolite:
    w = max(per-row weight)              ← one weight per metabolite
    producers = guilds with PRODUCES / RELEASES
    consumers = guilds with USES / DEPENDS_ON / HYDROLYSES / DEGRADES
    inhibited = guilds with IS_INHIBITED_BY
         │
         ├──► pos[consumer, producer] += w   for all pairs
         └──► neg[inhibited, producer] += w  for pairs where producer exists
                                             (only H₂O₂ fires in practice)
         │
  + AGORA2 FBA (use_agora=True):
    pFBA secretions × uptakes → pos or neg += 0.5
         │
         ▼
  net_flow = pos − neg  (10×10 float)
         │
         ▼
  sign(net_flow) → sign_prior matrix
         │
         ▼
  TMCMC log-posterior penalty
```
