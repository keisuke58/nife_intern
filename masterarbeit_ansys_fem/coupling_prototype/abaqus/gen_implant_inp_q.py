"""Element-FORMULATION robustness deck for the implant-thread biofilm (companion to gen_implant_inp.py).

The headline thread model (gen_implant_inp.py) uses CPE4 (linear, full-integration) at nu=0.45 -- a
near-incompressible biofilm. Two textbook artefacts could distort the interface peak stress and hence the
DH/CH and thread/flat ratios:

  (1) VOLUMETRIC LOCKING  -- full-integration linear quads lock as nu -> 0.5, over-stiffening the
      near-incompressible film and biasing the interface stress.
  (2) ROOT UNDER-RESOLUTION -- linear elements under-predict the stress concentration at the curved
      thread root (the same reason the 3-D coupons were upgraded CPE4 -> C3D10).

This deck re-solves the SAME geometry / SAME volume-matched growth eigenstrain with a chosen element
formulation so the canonical CPE4 result can be bracketed:

  CPE4    linear quad, full integration            (the canonical headline element)
  CPE4H   linear quad, HYBRID (pressure dof)        -> removes volumetric locking, same order
  CPE8    quadratic serendipity, full integration   -> resolves the curved root, no locking-free claim
  CPE8R   quadratic, reduced integration            -> locking-free + shear-lock-free general element
  CPE8RH  quadratic, reduced + HYBRID               -> gold standard for near-incompressible

If the interior interface peak and (more importantly) the dimensionless DH/CH & thread/flat ratios are
stable across these, the linear-CPE4 conclusion is NOT a locking / under-resolution artefact.

Geometry, BCs, growth law and element NUMBERING are kept identical to gen_implant_inp.py (bottom-row
elements are 1..Nx) so extract_implant.py works unchanged.

Run:  python gen_implant_inp_q.py <cond DH|CH> <thread|flat> <out.inp> [Nx Nz periods amp eltype]
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))

cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
profile = sys.argv[2] if len(sys.argv) > 2 else "thread"
OUT = sys.argv[3] if len(sys.argv) > 3 else "implantq_%s_%s.inp" % (cond, profile)
Nx = int(sys.argv[4]) if len(sys.argv) > 4 else 72
Nz = int(sys.argv[5]) if len(sys.argv) > 5 else 12
PERIODS = int(sys.argv[6]) if len(sys.argv) > 6 else 3
AMP = float(sys.argv[7]) if len(sys.argv) > 7 else (0.0 if profile == "flat" else 0.18)
ELTYPE = sys.argv[8] if len(sys.argv) > 8 else "CPE8R"

QUAD = {"CPE4", "CPE4H"}              # 4-node (corner-only) families
SERENDIP = {"CPE8", "CPE8R", "CPE8H", "CPE8RH"}  # 8-node (corner + mid-edge) families
if ELTYPE not in QUAD | SERENDIP:
    raise SystemExit("unsupported eltype %r (use %s)" % (ELTYPE, sorted(QUAD | SERENDIP)))
is_quad8 = ELTYPE in SERENDIP

E, NU = 5000.0, 0.45
W = float(PERIODS)
HF = 1.0
BETA, IC = CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"]

sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "extensions"))
from b4_viscoelastic_growth_stress import depth_phi_pg  # noqa: E402
_zn, _pg = depth_phi_pg(cond)


def eps_g(zfrac):
    pg = float(np.interp(zfrac, _zn, _pg))
    return max(0.0, BETA * pg + IC)


def y_thread(x):
    return AMP * 0.5 * (1.0 - np.cos(2.0 * np.pi * x))


# --- fine-grid node lattice (I in 0..2Nx, J in 0..2Nz); CPE4 uses only even/even corners ---
FI, FJ = 2 * Nx, 2 * Nz
xs = np.linspace(0.0, W, FI + 1)


def used(I, J):
    """Which lattice nodes carry a real dof. CPE4: corners only (I,J both even). CPE8 serendipity:
    corners + mid-edge (NOT the centre node where both are odd)."""
    if not is_quad8:
        return (I % 2 == 0) and (J % 2 == 0)
    return not (I % 2 == 1 and J % 2 == 1)


# assign contiguous node ids to used lattice points
gid = {}
nid_xy = []
for J in range(FJ + 1):
    for I in range(FI + 1):
        if used(I, J):
            n = len(gid) + 1
            gid[(I, J)] = n
            x = xs[I]
            y = y_thread(x) + (J / FJ) * HF
            nid_xy.append((n, x, y))

L = []
ap = L.append
ap("*HEADING")
ap(" Implant-thread biofilm eigenstrain deck (element-formulation study): cond=%s profile=%s amp=%.3f el=%s" %
   (cond, profile, AMP, ELTYPE))

ap("*NODE")
for n, x, y in nid_xy:
    ap(" %d, %.6f, %.6f" % (n, x, y))

# element numbering e = j*Nx + i + 1  (bottom row = 1..Nx, identical to gen_implant_inp.py)
ap("*ELEMENT, TYPE=%s" % ELTYPE)
for j in range(Nz):
    for i in range(Nx):
        e = j * Nx + i + 1
        I0, J0 = 2 * i, 2 * j
        c1 = gid[(I0, J0)]
        c2 = gid[(I0 + 2, J0)]
        c3 = gid[(I0 + 2, J0 + 2)]
        c4 = gid[(I0, J0 + 2)]
        if not is_quad8:
            ap(" %d, %d, %d, %d, %d" % (e, c1, c2, c3, c4))
        else:
            m5 = gid[(I0 + 1, J0)]        # between c1-c2
            m6 = gid[(I0 + 2, J0 + 1)]    # between c2-c3
            m7 = gid[(I0 + 1, J0 + 2)]    # between c3-c4
            m8 = gid[(I0, J0 + 1)]        # between c4-c1
            ap(" %d, %d, %d, %d, %d, %d, %d, %d, %d" % (e, c1, c2, c3, c4, m5, m6, m7, m8))

# one elset + material per depth ROW (constant volume-matched isotropic eigenstrain per row)
for j in range(Nz):
    els = [j * Nx + i + 1 for i in range(Nx)]
    ap("*ELSET, ELSET=ROW%d" % j)
    for k in range(0, len(els), 16):
        ap(" " + ",".join(str(x) for x in els[k:k + 16]))
ap("*ORIENTATION, NAME=ORI, SYSTEM=RECTANGULAR, DEFINITION=COORDINATES")
ap(" 1.0, 0.0, 0.0, 0.0, 1.0, 0.0")
ap(" 3, 0.0")
for j in range(Nz):
    ap("*SOLID SECTION, ELSET=ROW%d, MATERIAL=MAT%d, ORIENTATION=ORI" % (j, j))
for j in range(Nz):
    zc = (j + 0.5) / Nz
    g = eps_g(zc)
    a = (1.0 + g) ** (1.0 / 3.0) - 1.0           # volume-matched isotropic linear strain (Jg = 1+g)
    ap("*MATERIAL, NAME=MAT%d" % j)
    ap("*ELASTIC")
    ap(" %.1f, %.3f" % (E, NU))
    ap("*EXPANSION, TYPE=ORTHO, ZERO=0.0")
    ap(" %.6f, %.6f, %.6f" % (a, a, a))

# node sets for BCs (all USED lattice nodes; bottom row J=0)
bottom = [gid[(I, 0)] for I in range(FI + 1) if (I, 0) in gid]
alln = [n for n, _, _ in nid_xy]
for name, ids in (("BOTTOM", bottom), ("ALLN", alln)):
    ap("*NSET, NSET=%s" % name)
    for k in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[k:k + 16]))

ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE")
ap(" ALLN, 0.0")
ap("*BOUNDARY")
ap(" BOTTOM, 1, 2")
ap("*STEP")
ap(" Confined depth growth via thermal eigenstrain -> interface residual stress (el=%s)" % ELTYPE)
ap("*STATIC")
ap("*TEMPERATURE")
ap(" ALLN, 1.0")
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, COORD")
ap("*END STEP")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote %s  (%dx%d %s, %d nodes, cond=%s, %s, amp=%.3f)"
      % (OUT, Nx, Nz, ELTYPE, len(nid_xy), cond, profile, AMP))
