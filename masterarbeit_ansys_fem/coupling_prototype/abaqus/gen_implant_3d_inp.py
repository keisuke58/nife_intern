"""Full 3-D HELICAL implant-screw biofilm: growth + (optional) occlusal load + flow shear.

A genuine 3-D model (C3D8) of a conformal biofilm on a helically-threaded cylindrical titanium implant
screw -- the axisymmetric model's circular-groove approximation replaced by a real helix. Node grid in
(theta, radial, z); the implant surface radius follows a helical thread r_s(theta,z)=R0+A/2(1-cos(2*pi*z/P - theta)).
Biofilm of thickness Hf bonded at the implant surface (inner), free outer; z-ends u_z=0.

Loads (additive steps): (1) growth eigenstrain (isotropic volume-matched thermal); (2) optional OCCLUSAL
pressure on the top (mastication, for the loaded/cyclic-fatigue case); (3) optional FLOW SHEAR traction
on the outer biofilm surface (saliva/HOBIC flow). Output S (von Mises etc.) + COORD for 3-D visualisation.

Run:  python gen_implant_3d_inp.py <cond> <out.inp> [Nt Nz Nr nturns amp occlusal flowshear]
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))

cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
OUT = sys.argv[2] if len(sys.argv) > 2 else "screw3d_%s.inp" % cond
Nt = int(sys.argv[3]) if len(sys.argv) > 3 else 48      # circumferential divisions
Nz = int(sys.argv[4]) if len(sys.argv) > 4 else 60      # axial divisions
Nr = int(sys.argv[5]) if len(sys.argv) > 5 else 4       # radial (biofilm thickness) divisions
NTURNS = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
AMP = float(sys.argv[7]) if len(sys.argv) > 7 else 0.18
OCCLUSAL = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0    # occlusal pressure (mastication)
FLOWSHEAR = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0   # flow shear traction on outer surface

E, NU = 5000.0, 0.45
R0, HF, P = 2.0, 1.0, 1.0                 # core radius, biofilm thickness, thread pitch (film-thickness units)
HZ = NTURNS * P
BETA, IC = CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"]

sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "extensions"))
from b4_viscoelastic_growth_stress import depth_phi_pg  # noqa: E402
_zn, _pg = depth_phi_pg(cond)


def eps_g(zf):
    return max(0.0, BETA * float(np.interp(zf, _zn, _pg)) + IC)


def r_surf(theta, z):
    return R0 + AMP * 0.5 * (1.0 - np.cos(2.0 * np.pi * z / P - theta))   # helical thread


def nid(i, j, k):
    return k * (Nt * (Nr + 1)) + j * Nt + (i % Nt) + 1


L = []
ap = L.append
ap("*HEADING")
ap(" 3-D helical implant-screw biofilm, cond=%s, %d turns, occlusal=%.3g flowshear=%.3g"
   % (cond, int(NTURNS), OCCLUSAL, FLOWSHEAR))
ap("*NODE")
thetas = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
zs = np.linspace(0, HZ, Nz + 1)
for k in range(Nz + 1):
    z = zs[k]
    for j in range(Nr + 1):
        for i in range(Nt):
            th = thetas[i]
            r = r_surf(th, z) + (j / Nr) * HF
            ap(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), r * np.cos(th), r * np.sin(th), z))

ap("*ELEMENT, TYPE=C3D8, ELSET=FILM")
eid = 0
for k in range(Nz):
    for j in range(Nr):
        for i in range(Nt):
            eid += 1
            # winding reversed in the (theta,radial) face so the hex is right-handed (positive volume)
            n = [nid(i, j, k), nid(i, j + 1, k), nid(i + 1, j + 1, k), nid(i + 1, j, k),
                 nid(i, j, k + 1), nid(i, j + 1, k + 1), nid(i + 1, j + 1, k + 1), nid(i + 1, j, k + 1)]
            ap(" %d, %s" % (eid, ",".join(str(x) for x in n)))

# radial-layer elsets (element ordering is k-major, then j, then i) + isotropic growth materials
for j in range(Nr):
    els = []
    e2 = 0
    for k in range(Nz):
        for jj in range(Nr):
            for i in range(Nt):
                e2 += 1
                if jj == j:
                    els.append(e2)
    ap("*ELSET, ELSET=RLAY%d" % j)
    for m in range(0, len(els), 16):
        ap(" " + ",".join(str(x) for x in els[m:m + 16]))
for j in range(Nr):
    ap("*SOLID SECTION, ELSET=RLAY%d, MATERIAL=MAT%d" % (j, j))
for j in range(Nr):
    a = (1.0 + eps_g((j + 0.5) / Nr)) ** (1.0 / 3.0) - 1.0
    ap("*MATERIAL, NAME=MAT%d" % j)
    ap("*ELASTIC"); ap(" %.1f, %.3f" % (E, NU))
    ap("*EXPANSION"); ap(" %.6f" % a)

inner = [nid(i, 0, k) for k in range(Nz + 1) for i in range(Nt)]
zends = [nid(i, j, 0) for j in range(Nr + 1) for i in range(Nt)] + \
        [nid(i, j, Nz) for j in range(Nr + 1) for i in range(Nt)]
alln = [nid(i, j, k) for k in range(Nz + 1) for j in range(Nr + 1) for i in range(Nt)]
top = [nid(i, j, Nz) for j in range(Nr + 1) for i in range(Nt)]
outer = [nid(i, Nr, k) for k in range(Nz + 1) for i in range(Nt)]
for name, ids in (("INNER", inner), ("ZENDS", sorted(set(zends))), ("ALLN", sorted(set(alln))),
                  ("TOP", sorted(set(top))), ("OUTER", sorted(set(outer)))):
    ap("*NSET, NSET=%s" % name)
    for m in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[m:m + 16]))

zbot = [nid(i, j, 0) for j in range(Nr + 1) for i in range(Nt)]
ap("*NSET, NSET=ZBOT")
for m in range(0, len(zbot), 16):
    ap(" " + ",".join(str(x) for x in sorted(set(zbot))[m:m + 16]))

loaded = (OCCLUSAL > 0.0 or FLOWSHEAR > 0.0)
ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE"); ap(" ALLN, 0.0")
ap("*BOUNDARY")
ap(" INNER, 1, 3")                 # bonded to titanium implant (helical surface)
ap(" ZBOT, 3, 3")                  # bottom end u_z=0
if not loaded:
    ap(" TOP, 3, 3")               # growth-only: representative slice, top u_z=0 too

ap("*STEP"); ap(" growth eigenstrain")
ap("*STATIC")
ap("*TEMPERATURE"); ap(" ALLN, 1.0")
ap("*OUTPUT, FIELD"); ap("*NODE OUTPUT"); ap(" U")
ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD")
ap("*END STEP")
if loaded:
    ap("*STEP"); ap(" occlusal / flow load on top of growth (held)")
    ap("*STATIC")
    if OCCLUSAL > 0.0:
        ap("*CLOAD")
        for n in sorted(set(top)):
            ap(" %d, 3, %.6g" % (n, -OCCLUSAL))     # downward occlusal load on the free top
    if FLOWSHEAR > 0.0:
        ap("*CLOAD")
        for n in sorted(set(outer)):
            ap(" %d, 1, %.6g" % (n, FLOWSHEAR))     # lateral flow-drag proxy on outer surface
    ap("*OUTPUT, FIELD"); ap("*NODE OUTPUT"); ap(" U")
    ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD")
    ap("*END STEP")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote %s (C3D8 %dx%dx%d=%d elems, cond=%s, %d turns, amp=%.3f)"
      % (OUT, Nt, Nr, Nz, eid, cond, int(NTURNS), AMP))
