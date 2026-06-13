"""Generate a FINITE-STRAIN (F=Fe.Fg) implant-thread biofilm deck for umat_growth_phi.f.

Same threaded titanium cross-section as gen_implant_inp.py, but the growth is the rigorous
multiplicative split F = Fe.Fg (compressible neo-Hookean on Fe) evaluated by the socket-free UMAT
umat_growth_phi.f under NLGEOM=YES -- NOT the small-strain thermal-eigenstrain linearisation.
This removes the small-strain caveat (lambda_z ~ 1.4 is finite strain) and is the version a continuum-
mechanics examiner expects. phi_Pg(z) enters as predefined FIELD variable 1 (ramped 0 -> profile),
growth mode 1 = isotropic volume-matched (matches the thesis column, gen_column_phi_inp.py).

CPE4 plane strain, biofilm patch base-bonded to a smooth sinusoidal thread, sides free, top free.

Run:  python gen_implant_phi_inp.py <cond DH|CH> <out.inp> [Nx Nz periods amp]
Then: abaqus job=<job> user=umat_growth_phi.f interactive
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))

cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
OUT = sys.argv[2] if len(sys.argv) > 2 else "implantphi_%s_thread.inp" % cond
Nx = int(sys.argv[3]) if len(sys.argv) > 3 else 108
Nz = int(sys.argv[4]) if len(sys.argv) > 4 else 18
PERIODS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
AMP = float(sys.argv[6]) if len(sys.argv) > 6 else 0.18

E, NU = 5000.0, 0.45
W, HF = float(PERIODS), 1.0
BETA, IC = CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"]   # Pg-driven law (both conds)

sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "extensions"))
from b4_viscoelastic_growth_stress import depth_phi_pg  # noqa: E402
_zn, _pg = depth_phi_pg(cond)


def phi_at(zfrac):
    return float(np.interp(zfrac, _zn, _pg))


def y_thread(x):
    return AMP * 0.5 * (1.0 - np.cos(2.0 * np.pi * x))


def nid(i, j):
    return j * (Nx + 1) + i + 1


L = []
ap = L.append
ap("*HEADING")
ap(" Finite-strain (F=Fe.Fg) implant-thread biofilm, cond=%s, umat_growth_phi, NLGEOM" % cond)

ap("*NODE")
xs = np.linspace(0.0, W, Nx + 1)
node_phi = {}
for j in range(Nz + 1):
    for i in range(Nx + 1):
        x = xs[i]
        y = y_thread(x) + (j / Nz) * HF
        ap(" %d, %.6f, %.6f" % (nid(i, j), x, y))
        node_phi[nid(i, j)] = phi_at(j / Nz)

ap("*ELEMENT, TYPE=CPE4, ELSET=FILM")
for j in range(Nz):
    for i in range(Nx):
        e = j * Nx + i + 1
        ap(" %d, %d, %d, %d, %d" % (e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))

ap("*SOLID SECTION, ELSET=FILM, MATERIAL=BIOFILM")
ap("*MATERIAL, NAME=BIOFILM")
ap("*USER MATERIAL, CONSTANTS=5")
ap("** E, nu, beta_depth(CLSM), intercept(CLSM), mode(1=iso volume-matched)")
ap(" %.1f, %.3f, %.6f, %.6f, 1.0" % (E, NU, BETA, IC))
ap("*DEPVAR")
ap(" 3")

bottom = [nid(i, 0) for i in range(Nx + 1)]
alln = [nid(i, j) for j in range(Nz + 1) for i in range(Nx + 1)]
for name, ids in (("BOTTOM", bottom), ("ALLN", alln)):
    ap("*NSET, NSET=%s" % name)
    for k in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[k:k + 16]))

ap("*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=1")
ap(" ALLN, 0.0")
ap("*BOUNDARY")
ap(" BOTTOM, 1, 2")               # bonded to Ti thread (sides free -> geometry governs)
ap("*STEP, NLGEOM=YES, INC=300")
ap(" Ramp phi_Pg(z) 0 -> profile; finite-strain growth F=Fe.Fg -> interface stress")
ap("*STATIC")
ap(" 0.02, 1.0, 1.0E-5, 0.05")
ap("*FIELD, VARIABLE=1")
for n in alln:
    ap(" %d, %.6f" % (n, node_phi[n]))
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote %s (%dx%d CPE4, cond=%s, FINITE-STRAIN umat_growth_phi, amp=%.3f)" % (OUT, Nx, Nz, cond, AMP))
print("phi_Pg(z) by layer:", [round(phi_at((j + 0.5) / Nz), 3) for j in range(Nz)])
