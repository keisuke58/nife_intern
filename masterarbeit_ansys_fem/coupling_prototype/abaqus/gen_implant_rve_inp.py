"""Periodic RVE unit cell of the threaded biofilm-implant interface (resolves the lateral-BC caveat).

The base-bonded patch (gen_implant_inp.py, free sides) is a FINITE biofilm; its interface stress mixes
the thread response with a free-edge corner singularity. A real peri-implant biofilm is laterally
EXTENSIVE. This deck models that limit rigorously: a single thread period with PERIODIC lateral boundary
conditions (left edge tied to right edge by *EQUATION), bottom bonded to the titanium thread, top free.
No free edge -> no corner singularity -> the whole bonded interface is interior and the result is clean.

Isotropic volume-matched growth as an orthotropic thermal eigenstrain (per depth layer), matching the
thesis column. In this laterally-confined (infinite-film) limit the interface stress is membrane-driven;
comparing thread vs flat here tells whether implant thread geometry still matters once the film is
laterally infinite, and whether the dysbiosis (DH/CH) contrast survives.

Run:  python gen_implant_rve_inp.py <cond DH|CH> <profile thread|flat> <out.inp> [Nx Nz amp]
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))

cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
profile = sys.argv[2] if len(sys.argv) > 2 else "thread"
OUT = sys.argv[3] if len(sys.argv) > 3 else "rve_%s_%s.inp" % (cond, profile)
Nx = int(sys.argv[4]) if len(sys.argv) > 4 else 48
Nz = int(sys.argv[5]) if len(sys.argv) > 5 else 18
AMP = float(sys.argv[6]) if len(sys.argv) > 6 else (0.0 if profile == "flat" else 0.18)

E, NU = 5000.0, 0.45
W, HF = 1.0, 1.0                       # ONE thread period
BETA, IC = CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"]

sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "extensions"))
from b4_viscoelastic_growth_stress import depth_phi_pg  # noqa: E402
_zn, _pg = depth_phi_pg(cond)


def eps_g(zfrac):
    return max(0.0, BETA * float(np.interp(zfrac, _zn, _pg)) + IC)


def y_thread(x):
    return AMP * 0.5 * (1.0 - np.cos(2.0 * np.pi * x))    # period 1; smooth


def nid(i, j):
    return j * (Nx + 1) + i + 1


L = []
ap = L.append
ap("*HEADING")
ap(" Periodic RVE of threaded biofilm-implant interface, cond=%s profile=%s amp=%.3f" % (cond, profile, AMP))
ap("*NODE")
xs = np.linspace(0.0, W, Nx + 1)
for j in range(Nz + 1):
    for i in range(Nx + 1):
        ap(" %d, %.6f, %.6f" % (nid(i, j), xs[i], y_thread(xs[i]) + (j / Nz) * HF))
ap("*ELEMENT, TYPE=CPE4")
for j in range(Nz):
    for i in range(Nx):
        e = j * Nx + i + 1
        ap(" %d, %d, %d, %d, %d" % (e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))

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
    a = (1.0 + eps_g((j + 0.5) / Nz)) ** (1.0 / 3.0) - 1.0     # isotropic volume-matched
    ap("*MATERIAL, NAME=MAT%d" % j)
    ap("*ELASTIC")
    ap(" %.1f, %.3f" % (E, NU))
    ap("*EXPANSION, TYPE=ORTHO, ZERO=0.0")
    ap(" %.6f, %.6f, %.6f" % (a, a, a))

bottom = [nid(i, 0) for i in range(Nx + 1)]
alln = [nid(i, j) for j in range(Nz + 1) for i in range(Nx + 1)]
for name, ids in (("BOTTOM", bottom), ("ALLN", alln)):
    ap("*NSET, NSET=%s" % name)
    for k in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[k:k + 16]))

# periodic lateral BC: tie left edge (i=0) to right edge (i=Nx) for j=1..Nz (bonded row j=0 excluded)
ap("** periodic left-right: u(left,j) = u(right,j)")
for j in range(1, Nz + 1):
    for dof in (1, 2):
        ap("*EQUATION")
        ap(" 2")
        ap(" %d, %d, 1.0, %d, %d, -1.0" % (nid(0, j), dof, nid(Nx, j), dof))

ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE")
ap(" ALLN, 0.0")
ap("*BOUNDARY")
ap(" BOTTOM, 1, 2")
ap("*STEP")
ap(" Periodic-RVE confined growth -> interface stress (laterally-infinite limit)")
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
print("wrote %s (periodic RVE, %dx%d CPE4, cond=%s, %s, amp=%.3f)" % (OUT, Nx, Nz, cond, profile, AMP))
