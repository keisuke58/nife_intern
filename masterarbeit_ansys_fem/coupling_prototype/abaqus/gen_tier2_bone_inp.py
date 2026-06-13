"""Tier-2 (a): parametric VOLUMETRIC alveolar-bone segment coupling an implant to an adjacent tooth.

A structured-hex (C3D8) alveolar bone block containing, side by side:
  * an osseointegrated titanium IMPLANT (stiff cylinder, bonded to bone by mesh continuity), and
  * a natural TOOTH root (dentin) seated in a soft PERIODONTAL-LIGAMENT (PDL) shell,
with peri-implant and peri-tooth BIOFILM collars (soft, dysbiotic growth eigenstrain) at the crest.
Occlusal (masticatory) load is applied to both crown tops; because both are embedded in the SAME bone,
the load is transmitted implant<->bone<->tooth -- the genuine Tier-2 coupling that Tier-1 (rigid bonded
boundary, independent patches) cannot capture. Two steps: (1) dysbiotic biofilm growth; (2) occlusal load.

Units: mm, MPa. Materials: bone 13.7 GPa, Ti 110 GPa, dentin 18 GPa, PDL 50 MPa, biofilm 1 MPa.
Run:  python gen_tier2_bone_inp.py <out.inp> [nx ny nz]
"""
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "tier2_bone.inp"
nx = int(sys.argv[2]) if len(sys.argv) > 2 else 36
ny = int(sys.argv[3]) if len(sys.argv) > 3 else 20
nz = int(sys.argv[4]) if len(sys.argv) > 4 else 26
Lx, Ly, Lz = 18.0, 10.0, 13.0
Z_CREST = 10.0                              # bone crest height
IMP = (5.0, 5.0); R_IMP = 2.0               # implant centre, radius
TOO = (13.0, 5.0); R_ROOT = 2.5; R_PDL = 3.0  # tooth centre, root + pdl outer radius
Z_ROOT_APEX = 3.0
EPS_GROWTH = 0.19                           # dysbiotic biofilm growth (eps = alpha/3, DH)

hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
MATS = {"BONE": (13700., 0.30), "TI": (110000., 0.34), "DENTIN": (18000., 0.31),
        "PDL": (50., 0.45), "BIOFILM": (1.0, 0.45)}


def nid(i, j, k):
    return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1


def region(xc, yc, zc):
    ri = np.hypot(xc - IMP[0], yc - IMP[1])
    rt = np.hypot(xc - TOO[0], yc - TOO[1])
    # biofilm collars at the crest (transmucosal sulcus)
    if 8.0 <= zc <= 11.0 and (R_IMP <= ri <= R_IMP + 0.5 or R_ROOT <= rt <= R_ROOT + 0.5):
        return "BIOFILM"
    if ri < R_IMP:
        return "TI"                          # implant (full height: root + abutment)
    if rt < R_ROOT and zc >= Z_ROOT_APEX:
        return "DENTIN"                       # tooth root/crown
    if R_ROOT <= rt < R_PDL and zc < Z_CREST and zc >= Z_ROOT_APEX:
        return "PDL"                          # periodontal ligament shell (in bone)
    if zc < Z_CREST:
        return "BONE"
    return None                               # air / sulcus space above crest -> skip element


# build elements (skip air); collect used materials per element
elems = {m: [] for m in MATS}
e = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            xc = (i + 0.5) * hx; yc = (j + 0.5) * hy; zc = (k + 0.5) * hz
            reg = region(xc, yc, zc)
            if reg is None:
                continue
            e += 1
            conn = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
            elems[reg].append((e, conn))

L = []
ap = L.append
ap("*HEADING"); ap(" Tier-2 parametric alveolar bone: implant + adjacent tooth coupled through bone")
ap("*NODE")
for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            ap(" %d, %.5f, %.5f, %.5f" % (nid(i, j, k), i * hx, j * hy, k * hz))
for m, els in elems.items():
    if not els:
        continue
    ap("*ELEMENT, TYPE=C3D8, ELSET=%s" % m)
    for eid, conn in els:
        ap(" %d, %s" % (eid, ",".join(str(c) for c in conn)))
for m, (E, nu) in MATS.items():
    if not elems[m]:
        continue
    ap("*SOLID SECTION, ELSET=%s, MATERIAL=%s" % (m, m))
    ap("*MATERIAL, NAME=%s" % m)
    ap("*ELASTIC"); ap(" %.1f, %.3f" % (E, nu))
    if m == "BIOFILM":
        ap("*EXPANSION"); ap(" %.6f" % EPS_GROWTH)

# node sets: bone bottom + outer faces fixed; implant-top + tooth-top loaded
fixed, imp_top, too_top = [], [], []
for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y, z = i * hx, j * hy, k * hz
            if z <= 1e-6 or (z < Z_CREST and (x <= 1e-6 or x >= Lx - 1e-6 or y <= 1e-6 or y >= Ly - 1e-6)):
                fixed.append(nid(i, j, k))
            if abs(z - Lz) < 1e-6:
                if np.hypot(x - IMP[0], y - IMP[1]) < R_IMP:
                    imp_top.append(nid(i, j, k))
                if np.hypot(x - TOO[0], y - TOO[1]) < R_ROOT:
                    too_top.append(nid(i, j, k))
for name, ids in (("FIXED", fixed), ("IMPTOP", imp_top), ("TOOTOP", too_top),
                  ("ALLN", [nid(i, j, k) for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)])):
    ap("*NSET, NSET=%s" % name)
    ids = sorted(set(ids))
    for m in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[m:m + 16]))

ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE"); ap(" ALLN, 0.0")
ap("*BOUNDARY"); ap(" FIXED, 1, 3")
ap("*STEP"); ap(" 1) dysbiotic biofilm growth"); ap("*STATIC")
ap("*TEMPERATURE"); ap(" ALLN, 1.0")
ap("*OUTPUT, FIELD"); ap("*NODE OUTPUT"); ap(" U"); ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD")
ap("*END STEP")
ap("*STEP"); ap(" 2) occlusal (masticatory) load on both crowns -> coupled through bone"); ap("*STATIC")
ap("*CLOAD")
for n in sorted(set(imp_top)):
    ap(" %d, 3, -50.0" % n)        # downward occlusal load on implant crown
    ap(" %d, 1, 10.0" % n)         # + lateral (oblique mastication)
for n in sorted(set(too_top)):
    ap(" %d, 3, -50.0" % n)        # downward occlusal load on tooth crown
    ap(" %d, 1, 10.0" % n)
ap("*OUTPUT, FIELD"); ap("*NODE OUTPUT"); ap(" U"); ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD")
ap("*END STEP")

open(OUT, "w").write("\n".join(L) + "\n")
tot = sum(len(v) for v in elems.values())
print("wrote %s (%d C3D8: %s)" % (OUT, tot, {m: len(v) for m, v in elems.items() if v}))
