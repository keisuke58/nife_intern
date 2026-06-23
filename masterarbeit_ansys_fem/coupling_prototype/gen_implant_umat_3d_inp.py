"""Full 3-D helical implant-screw biofilm with NSP UMAT.

True helical screw thread: biofilm of thickness HF conformal on a titanium cylinder of radius R0
with a helical groove r_surface(theta,z) = R0 + AMP/2*(1 - cos(2*pi*z/P - theta)).
C3D8 elements; depth from the local thread surface computed by the server from (r, theta, z).

The server needs --geometry helical mode to compute the local depth:
    depth = r - r_surface(theta, z),  depth_frac = depth / HF
where r = sqrt(x^2+y^2), theta = atan2(y,x) from COORDS.

Usage:
    python gen_implant_umat_3d_inp.py [cond DH|CH] [out.inp] [Nt Nz Nr nturns amp]
Server: umat_server_col.py --geometry helical --R0 2.0 --pitch 1.0 --amp 0.18 --height 1.0
            --profile phipg_depth_DH.json --gamma 0.4 --nramp 20
UMAT:   umat_socket_col.f (unchanged; COORDS(3) carries x,y,z in Cartesian)
"""
import sys, json
from pathlib import Path
import numpy as np

HERE   = Path(__file__).resolve().parent
CALIB  = json.load(open(HERE / "beta_calibration.json"))

cond   = sys.argv[1] if len(sys.argv) > 1 else "DH"
OUT    = sys.argv[2] if len(sys.argv) > 2 else "implant_umat_3d_%s.inp" % cond
Nt     = int(sys.argv[3])   if len(sys.argv) > 3 else 48   # circumferential divisions
Nz     = int(sys.argv[4])   if len(sys.argv) > 4 else 60   # axial divisions
Nr     = int(sys.argv[5])   if len(sys.argv) > 5 else 4    # radial (biofilm thickness)
NTURNS = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
AMP    = float(sys.argv[7]) if len(sys.argv) > 7 else 0.18

E, NU  = 5000.0, 0.45
R0, HF, P = 2.0, 1.0, 1.0     # core radius, biofilm thickness, thread pitch
HZ = NTURNS * P                # total axial height

def r_surf(theta, z):
    """Helical thread surface radius."""
    return R0 + AMP * 0.5 * (1.0 - np.cos(2.0 * np.pi * z / P - theta))

def nid(i, j, k):
    """i=circumferential (0..Nt-1, periodic), j=radial (0=inner), k=axial (0..Nz)."""
    return k * (Nt * (Nr + 1)) + j * Nt + (i % Nt) + 1

L = []; ap = L.append
ap("*HEADING")
ap(" 3-D helical implant screw NSP UMAT: cond=%s %.1f turns R0=%.2f amp=%.3f" % (
    cond, NTURNS, R0, AMP))

# --- nodes ---
ap("*NODE")
thetas = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
zs     = np.linspace(0, HZ, Nz + 1)
for k in range(Nz + 1):
    z = zs[k]
    for j in range(Nr + 1):
        for i in range(Nt):
            th  = thetas[i]
            r   = r_surf(th, z) + (j / Nr) * HF
            ap(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k),
               r * np.cos(th), r * np.sin(th), z))

# --- C3D8 elements (hexahedral, right-handed winding) ---
ap("*ELEMENT, TYPE=C3D8, ELSET=FILM")
eid = 0
for k in range(Nz):
    for j in range(Nr):
        for i in range(Nt):
            eid += 1
            n = [nid(i,   j,   k), nid(i,   j+1, k),
                 nid(i+1, j+1, k), nid(i+1, j,   k),
                 nid(i,   j,   k+1), nid(i,   j+1, k+1),
                 nid(i+1, j+1, k+1), nid(i+1, j,   k+1)]
            ap(" %d, %s" % (eid, ", ".join(str(x) for x in n)))

# --- NSP UMAT material ---
ap("*SOLID SECTION, ELSET=FILM, MATERIAL=NSP_BIOFILM")
ap("*MATERIAL, NAME=NSP_BIOFILM")
ap("*USER MATERIAL, CONSTANTS=5")
ap(" %.1f, %.3f, %.4f, %.6f, %.4f" % (
    E, NU, CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"], HF))
ap("*DEPVAR")
ap(" 1")

# --- node sets: inner cylindrical surface (j=0) ---
inner_ids = [nid(i, 0, k) for k in range(Nz + 1) for i in range(Nt)]
zbot_ids  = [nid(i, j, 0)  for j in range(Nr + 1) for i in range(Nt)]
ztop_ids  = [nid(i, j, Nz) for j in range(Nr + 1) for i in range(Nt)]
for name, ids in (("INNER", inner_ids), ("ZBOT", zbot_ids), ("ZTOP", ztop_ids)):
    ap("*NSET, NSET=%s" % name)
    for kk in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[kk:kk+16]))

# --- BCs: bonded inner surface, axial ends fixed ---
ap("*BOUNDARY")
ap(" INNER, 1, 3")    # bonded to titanium: u_x=u_y=u_z=0
ap(" ZBOT,  3, 3")    # axial constraint at z=0 (representative helix turns)
ap(" ZTOP,  3, 3")    # axial constraint at z=HZ

# --- step ---
ap("*STEP, NLGEOM=YES, INC=200")
ap(" 3-D helical NSP UMAT: depth-graded growth on true helical thread geometry")
ap("*STATIC")
ap(" 0.05, 1.0, 1e-5, 0.05")
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U, COORD")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

Path(OUT).write_text("\n".join(L) + "\n")
print("wrote %s  (%d x %d x %d C3D8, cond=%s, %.1f turns, amp=%.3f)" % (
    OUT, Nt, Nr, Nz, cond, NTURNS, AMP))
print("server: umat_server_col.py --geometry helical --R0 %.2f --pitch %.2f --amp %.3f "
      "--height %.2f --coord-axis 0 --profile phipg_depth_%s.json ..." % (R0, P, AMP, HF, cond))
