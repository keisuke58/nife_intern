"""RVE (periodic lateral BC) implant-thread biofilm with NSP UMAT.

Single thread period; left/right edges tied by *EQUATION -> laterally-infinite film.
Eliminates the free-edge corner singularity in gen_implant_umat_inp.py.
Use this to verify that DH/CH and thread/flat contrasts survive in the periodic limit.

Usage:
    python gen_implant_umat_rve_inp.py [cond DH|CH] [profile thread|flat] [out.inp] [Nx Nz amp]
Server: umat_server_col.py --coord-axis 1 --height 1.0 --profile phipg_depth_DH.json
            --gamma 0.4 --nramp 20 --growth_mode iso
UMAT:   umat_socket_col.f (unchanged)
"""
import sys, json
from pathlib import Path
import numpy as np

HERE   = Path(__file__).resolve().parent
CALIB  = json.load(open(HERE / "beta_calibration.json"))

cond    = sys.argv[1] if len(sys.argv) > 1 else "DH"
profile = sys.argv[2] if len(sys.argv) > 2 else "thread"
OUT     = sys.argv[3] if len(sys.argv) > 3 else "implant_umat_rve_%s_%s.inp" % (cond, profile)
Nx      = int(sys.argv[4])   if len(sys.argv) > 4 else 48
Nz      = int(sys.argv[5])   if len(sys.argv) > 5 else 12
AMP     = float(sys.argv[6]) if len(sys.argv) > 6 else (0.0 if profile == "flat" else 0.18)

E, NU  = 5000.0, 0.45
W      = 1.0    # single thread period
HF     = 1.0
DZ     = 0.1   # thin slice thickness (pseudo-plane-strain)

def y_thread(x):
    return AMP * np.sin(np.pi * x) ** 2

def nid(i, j, k):
    return k * (Nx + 1) * (Nz + 1) + j * (Nx + 1) + i + 1

L = []; ap = L.append
ap("*HEADING")
ap(" Implant biofilm NSP UMAT RVE (periodic BC): cond=%s profile=%s amp=%.3f" % (cond, profile, AMP))

# --- nodes (2 z-planes) ---
ap("*NODE")
xs = np.linspace(0.0, W, Nx + 1)
for k in range(2):
    zv = k * DZ
    for j in range(Nz + 1):
        for i in range(Nx + 1):
            x    = xs[i]
            ybot = y_thread(x)
            y    = ybot + (j / Nz) * HF
            ap(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), x, y, zv))

# --- C3D8 elements ---
ap("*ELEMENT, TYPE=C3D8, ELSET=BIOFILM")
for j in range(Nz):
    for i in range(Nx):
        e = j * Nx + i + 1
        n = [nid(i,   j,   0), nid(i+1, j,   0),
             nid(i+1, j+1, 0), nid(i,   j+1, 0),
             nid(i,   j,   1), nid(i+1, j,   1),
             nid(i+1, j+1, 1), nid(i,   j+1, 1)]
        ap(" %d, %s" % (e, ", ".join(str(x) for x in n)))

# --- single NSP UMAT material ---
ap("*SOLID SECTION, ELSET=BIOFILM, MATERIAL=NSP_BIOFILM")
ap("*MATERIAL, NAME=NSP_BIOFILM")
ap("*USER MATERIAL, CONSTANTS=5")
ap(" %.1f, %.3f, %.4f, %.6f, %.4f" % (
    E, NU, CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"], HF))
ap("*DEPVAR")
ap(" 1")

# --- node sets ---
bottom = [nid(i, 0, k) for k in range(2) for i in range(Nx + 1)]
front  = [nid(i, j, 0) for j in range(Nz + 1) for i in range(Nx + 1)]
back   = [nid(i, j, 1) for j in range(Nz + 1) for i in range(Nx + 1)]
for name, ids in (("BOTTOM", bottom), ("FRONT", front), ("BACK", back)):
    ap("*NSET, NSET=%s" % name)
    for kk in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[kk:kk+16]))

# --- BCs: bonded bottom + pseudo-plane-strain ---
ap("*BOUNDARY")
ap(" BOTTOM, 1, 3")   # bonded titanium interface
ap(" FRONT,  3, 3")   # u_z=0 (front face)
ap(" BACK,   3, 3")   # u_z=0 (back face)

# --- periodic lateral BC: u(left, j, k) = u(right, j, k) for j=1..Nz ---
ap("** Periodic lateral BC: u_x and u_y are equal on left (i=0) and right (i=Nx) edges")
for j in range(1, Nz + 1):
    for k in range(2):
        for dof in (1, 2):
            ap("*EQUATION")
            ap(" 2")
            ap(" %d, %d, 1.0, %d, %d, -1.0" % (nid(0, j, k), dof, nid(Nx, j, k), dof))

# --- step ---
ap("*STEP, NLGEOM=YES, INC=200")
ap(" NSP UMAT RVE: periodic lateral BC, single thread period")
ap("*STATIC")
ap(" 0.05, 1.0, 1e-5, 0.05")
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U, COORD")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

Path(OUT).write_text("\n".join(L) + "\n")
print("wrote %s  (%dx%d C3D8 RVE, periodic BC, cond=%s, %s, amp=%.3f)" % (
    OUT, Nx, Nz, cond, profile, AMP))
print("run:  umat_server_col.py --coord-axis 1 --height 1.0 --profile phipg_depth_%s.json ..." % cond)
