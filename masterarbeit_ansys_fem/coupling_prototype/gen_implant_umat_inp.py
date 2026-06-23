"""Generate Abaqus C3D8 implant-thread biofilm inp with NSP UMAT socket coupling.

Extrudes the 2-D thread profile by one element in z (pseudo-plane-strain slice).
Uses the existing umat_socket_col.f unchanged (hardcoded 6-stress/36-tangent protocol).
Depth = y-coordinate (axis 1), so run server with --coord-axis 1.

Usage:
    python gen_implant_umat_inp.py [cond DH|CH] [profile thread|flat] [out.inp] [Nx Nz periods amp]

Server: umat_server_col.py --coord-axis 1 --height 1.0 --profile phipg_depth_DH.json \\
            --gamma 0.4 --nramp 20 --growth_mode iso
UMAT:   umat_socket_col.f  (no changes needed)
"""
import sys
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE / "beta_calibration.json"))

cond    = sys.argv[1] if len(sys.argv) > 1 else "DH"
profile = sys.argv[2] if len(sys.argv) > 2 else "thread"
OUT     = sys.argv[3] if len(sys.argv) > 3 else "implant_umat_%s_%s.inp" % (cond, profile)
Nx      = int(sys.argv[4])   if len(sys.argv) > 4 else 72
Nz      = int(sys.argv[5])   if len(sys.argv) > 5 else 12
PERIODS = int(sys.argv[6])   if len(sys.argv) > 6 else 3
AMP     = float(sys.argv[7]) if len(sys.argv) > 7 else (0.0 if profile == "flat" else 0.18)

E, NU   = 5000.0, 0.45
W       = float(PERIODS)   # total width (period = 1.0)
HF      = 1.0              # biofilm thickness above thread surface
DZ      = 0.1              # out-of-plane thickness (thin slice for pseudo-plane-strain)

# thread surface: y_bot(x) = AMP * sin^2(pi*x/period), period=1
def y_thread(x):
    return AMP * np.sin(np.pi * x) ** 2

# --- node numbering: (i, j, k), k in {0,1} (front/back z-plane) ---
# node id = k*(Nx+1)*(Nz+1) + j*(Nx+1) + i + 1
def nid(i, j, k):
    return k * (Nx + 1) * (Nz + 1) + j * (Nx + 1) + i + 1

L = []
ap = L.append

ap("*HEADING")
ap(" Implant biofilm NSP UMAT: cond=%s profile=%s amp=%.3f  (C3D8 thin slice, z-symmetry)" %
   (cond, profile, AMP))

# --- nodes (2 z-planes) ---
ap("*NODE")
xs = np.linspace(0.0, W, Nx + 1)
for k in range(2):
    zv = k * DZ
    for j in range(Nz + 1):
        for i in range(Nx + 1):
            x   = xs[i]
            ybot = y_thread(x)
            y   = ybot + (j / Nz) * HF
            ap(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), x, y, zv))

# --- C3D8 elements ---
ap("*ELEMENT, TYPE=C3D8, ELSET=BIOFILM")
for j in range(Nz):
    for i in range(Nx):
        e = j * Nx + i + 1
        # bottom face (k=0) then top face (k=1), Abaqus C3D8 connectivity
        n = [nid(i,   j,   0), nid(i+1, j,   0),
             nid(i+1, j+1, 0), nid(i,   j+1, 0),
             nid(i,   j,   1), nid(i+1, j,   1),
             nid(i+1, j+1, 1), nid(i,   j+1, 1)]
        ap(" %d, %s" % (e, ", ".join(str(x) for x in n)))

# --- single UMAT material ---
ap("*SOLID SECTION, ELSET=BIOFILM, MATERIAL=NSP_BIOFILM")
ap("*MATERIAL, NAME=NSP_BIOFILM")
ap("*USER MATERIAL, CONSTANTS=5")
ap(" %.1f, %.3f, %.4f, %.6f, %.4f" % (
    E, NU, CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"], HF))
ap("*DEPVAR")
ap(" 1")   # STATEV(1) = growth ramp counter

# --- node sets ---
bottom_k0 = [nid(i, 0, 0) for i in range(Nx + 1)]
bottom_k1 = [nid(i, 0, 1) for i in range(Nx + 1)]
front_face = [nid(i, j, 0) for j in range(Nz + 1) for i in range(Nx + 1)]
back_face  = [nid(i, j, 1) for j in range(Nz + 1) for i in range(Nx + 1)]

for name, ids in (("BOTTOM", bottom_k0 + bottom_k1),
                  ("FRONT",  front_face),
                  ("BACK",   back_face)):
    ap("*NSET, NSET=%s" % name)
    for k in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[k:k+16]))

# --- BCs ---
ap("*BOUNDARY")
ap(" BOTTOM, 1, 3")   # bonded: u_x=u_y=u_z=0 at titanium interface
ap(" FRONT,  3, 3")   # u_z=0  (plane-strain front face)
ap(" BACK,   3, 3")   # u_z=0  (plane-strain back face)

# --- step ---
ap("*STEP, NLGEOM=YES, INC=200")
ap(" NSP socket UMAT: depth-graded growth on implant-thread geometry")
ap("*STATIC")
ap(" 0.05, 1.0, 1e-5, 0.05")
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U, COORD")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

Path(OUT).write_text("\n".join(L) + "\n")
print("wrote %s  (%d x %d x 1 C3D8, cond=%s, %s, amp=%.3f)" % (OUT, Nx, Nz, cond, profile, AMP))
print("run:  bash run_implant_umat.sh %s %s" % (cond, profile))
