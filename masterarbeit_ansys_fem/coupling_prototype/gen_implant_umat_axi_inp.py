"""Axisymmetric implant-screw biofilm with NSP UMAT (hoop confinement).

Uses CGAX4 (generalised axisymmetric, NTENS=6) so the existing umat_socket_col.f works unchanged.
Models a smooth cylindrical implant surface of radius R0; biofilm grows radially outward.
Hoop (circumferential) confinement is a genuinely 3-D effect invisible to 2-D plane-strain: a film
on a cylinder cannot expand around the circumference without stretching, generating an interface hoop
stress that scales with curvature 1/R0. Run with varying R0 to show when curvature matters.

Twist DOF (CGAX4's 5th DOF) is constrained to zero -> pure axisymmetric growth without torsion.

Usage:
    python gen_implant_umat_axi_inp.py [cond DH|CH] [R0 float] [out.inp] [Nr Nz]
Server: umat_server_col.py --coord-axis 0 --coord-offset <R0> --height 1.0
            --profile phipg_depth_DH.json --gamma 0.4 --nramp 20 --growth_mode iso
UMAT:   umat_socket_col.f (unchanged, NTENS=6 for CGAX4)
"""
import sys, json
from pathlib import Path
import numpy as np

HERE  = Path(__file__).resolve().parent
CALIB = json.load(open(HERE / "beta_calibration.json"))

cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
R0   = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0    # implant core radius [film-thickness units]
OUT  = sys.argv[3] if len(sys.argv) > 3 else "implant_umat_axi_%s_R%.1f.inp" % (cond, R0)
Nr   = int(sys.argv[4]) if len(sys.argv) > 4 else 12       # radial divisions (biofilm)
Nz   = int(sys.argv[5]) if len(sys.argv) > 5 else 6        # axial divisions (representative slice)

E, NU = 5000.0, 0.45
HF    = 1.0          # biofilm thickness
LZ    = 1.0          # axial length of representative slice

def nid(i, j):       # i=radial (0=inner), j=axial
    return j * (Nr + 1) + i + 1

L = []; ap = L.append
ap("*HEADING")
ap(" Implant biofilm NSP UMAT axisymmetric (CGAX4): cond=%s R0=%.2f HF=%.2f" % (cond, R0, HF))
ap("** Hoop confinement: film on cylinder of radius R0; depth = r - R0")
ap("** CGAX4: generalized axisymmetric, NTENS=6, twist DOF constrained")

# --- nodes in (r, z) plane ---
ap("*NODE")
for j in range(Nz + 1):
    z = (j / Nz) * LZ
    for i in range(Nr + 1):
        r = R0 + (i / Nr) * HF
        ap(" %d, %.6f, %.6f" % (nid(i, j), r, z))

# --- CGAX4 elements ---
ap("*ELEMENT, TYPE=CGAX4, ELSET=BIOFILM")
for j in range(Nz):
    for i in range(Nr):
        e = j * Nr + i + 1
        ap(" %d, %d, %d, %d, %d" % (
            e, nid(i, j), nid(i+1, j), nid(i+1, j+1), nid(i, j+1)))

# --- single NSP UMAT material ---
ap("*SOLID SECTION, ELSET=BIOFILM, MATERIAL=NSP_BIOFILM")
ap("*MATERIAL, NAME=NSP_BIOFILM")
ap("*USER MATERIAL, CONSTANTS=5")
ap(" %.1f, %.3f, %.4f, %.6f, %.4f" % (
    E, NU, CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"], HF))
ap("*DEPVAR")
ap(" 1")

# --- node sets ---
inner_ns = [nid(0,  j) for j in range(Nz + 1)]
zbot_ns  = [nid(i,  0) for i in range(Nr + 1)]
ztop_ns  = [nid(i, Nz) for i in range(Nr + 1)]
alln_ns  = [nid(i,  j) for j in range(Nz + 1) for i in range(Nr + 1)]
for name, ids in (("INNER", inner_ns), ("ZBOT", zbot_ns),
                  ("ZTOP", ztop_ns),   ("ALLN", alln_ns)):
    ap("*NSET, NSET=%s" % name)
    for kk in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[kk:kk+16]))

# --- BCs ---
ap("*BOUNDARY")
ap(" INNER, 1, 2")   # bonded titanium: u_r=0 (DOF1), u_z=0 (DOF2)
ap(" INNER, 5, 5")   # no twist at inner surface
ap(" ZBOT,  2, 2")   # axial symmetry (representative slice)
ap(" ZTOP,  2, 2")
ap(" ALLN,  5, 5")   # suppress twist everywhere -> pure axisymmetric

# --- step ---
ap("*STEP, NLGEOM=YES, INC=200")
ap(" NSP UMAT axisymmetric: hoop-confined growth, R0=%.2f" % R0)
ap("*STATIC")
ap(" 0.05, 1.0, 1e-5, 0.05")
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U, COORD")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

Path(OUT).write_text("\n".join(L) + "\n")
print("wrote %s  (%dx%d CGAX4, cond=%s, R0=%.2f)" % (OUT, Nr, Nz, cond, R0))
print("run server: umat_server_col.py --coord-axis 0 --coord-offset %.2f --height %.2f ..." % (R0, HF))
