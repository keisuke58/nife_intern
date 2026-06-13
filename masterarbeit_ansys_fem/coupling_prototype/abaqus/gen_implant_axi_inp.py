"""Axisymmetric implant-SCREW biofilm: the hoop-confinement (genuinely 3-D) effect at 2-D cost.

The plane-strain thread deck misses the cylindrical curvature of the implant body/screw: a biofilm
growing on a cylinder of radius R is confined in the HOOP (circumferential) direction --- it cannot
expand around the cylinder without stretching --- so even a SMOOTH cylindrical surface carries an
interface stress that a flat (plane-strain) surface does not, scaling with the implant curvature 1/R.

This CAX4 axisymmetric deck captures that: a conformal biofilm of thickness Hf on a cylinder of radius
R0 (inner radius = the bonded titanium surface), isotropic volume-matched growth (thermal eigenstrain,
no orientation needed for isotropic expansion), inner radius bonded, outer free, axial slice confined
(u_z=0 at both z-faces -> representative slice of a long screw). Output the inner-interface hoop stress
S33 and von Mises. A sweep over R0 shows when the cylindrical curvature matters (microthread/small R)
versus the flat limit (implant body, large R).

Run:  python gen_implant_axi_inp.py <cond DH|CH> <R0> <out.inp> [Nr Nz]
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))

cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
R0 = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "axi_%s_R%s.inp" % (cond, str(R0).replace(".", "p"))
Nr = int(sys.argv[4]) if len(sys.argv) > 4 else 16
Nz = int(sys.argv[5]) if len(sys.argv) > 5 else 8

E, NU = 5000.0, 0.45
HF, LZ = 1.0, 0.5                       # film thickness (radial), axial slice height
BETA, IC = CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"]

sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "extensions"))
from b4_viscoelastic_growth_stress import depth_phi_pg  # noqa: E402
_zn, _pg = depth_phi_pg(cond)


def eps_g(zfrac):
    return max(0.0, BETA * float(np.interp(zfrac, _zn, _pg)) + IC)


def nid(i, j):
    return j * (Nr + 1) + i + 1


L = []
ap = L.append
ap("*HEADING")
ap(" Axisymmetric implant-screw biofilm, cond=%s, R0=%.3f (hoop confinement)" % (cond, R0))
ap("*NODE")
for j in range(Nz + 1):
    for i in range(Nr + 1):
        r = R0 + (i / Nr) * HF
        z = (j / Nz) * LZ
        ap(" %d, %.6f, %.6f" % (nid(i, j), r, z))
ap("*ELEMENT, TYPE=CAX4")
for j in range(Nz):
    for i in range(Nr):
        e = j * Nr + i + 1
        ap(" %d, %d, %d, %d, %d" % (e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))

# isotropic eigenstrain per RADIAL layer i (depth = radial distance from implant surface)
for i in range(Nr):
    els = [j * Nr + i + 1 for j in range(Nz)]
    ap("*ELSET, ELSET=COL%d" % i)
    ap(" " + ",".join(str(x) for x in els))
for i in range(Nr):
    ap("*SOLID SECTION, ELSET=COL%d, MATERIAL=MAT%d" % (i, i))
for i in range(Nr):
    a = (1.0 + eps_g((i + 0.5) / Nr)) ** (1.0 / 3.0) - 1.0
    ap("*MATERIAL, NAME=MAT%d" % i)
    ap("*ELASTIC")
    ap(" %.1f, %.3f" % (E, NU))
    ap("*EXPANSION")          # isotropic -> single alpha, no orientation needed
    ap(" %.6f" % a)

inner = [nid(0, j) for j in range(Nz + 1)]
zbot = [nid(i, 0) for i in range(Nr + 1)]
ztop = [nid(i, Nz) for i in range(Nr + 1)]
alln = [nid(i, j) for j in range(Nz + 1) for i in range(Nr + 1)]
for name, ids in (("INNER", inner), ("ZBOT", zbot), ("ZTOP", ztop), ("ALLN", alln)):
    ap("*NSET, NSET=%s" % name)
    for k in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[k:k + 16]))
ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE")
ap(" ALLN, 0.0")
ap("*BOUNDARY")
ap(" INNER, 1, 2")            # bonded titanium surface (r and z)
ap(" ZBOT, 2, 2")            # axial slice: u_z=0 at both ends (long-screw representative slice)
ap(" ZTOP, 2, 2")
ap("*STEP")
ap(" Axisymmetric confined growth -> hoop + radial interface stress")
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
print("wrote %s (CAX4 %dx%d, cond=%s, R0=%.3f, Hf/R0=%.3f)" % (OUT, Nr, Nz, cond, R0, HF / R0))
