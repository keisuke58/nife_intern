"""Generate film_buckle.inp — B1: growth-induced out-of-plane buckling / wrinkling of the film strip.

Same bonded film strip as gen_film_inp.py (bottom z=0 fixed, free top, free right edge, x=0 symmetry,
plane strain), but (i) a small BENDING IMPERFECTION is seeded into the geometry — every node is shifted
in z by  amp * (z/H) * sin(pi x / L)  so the strip starts very slightly wavy (largest at the top, zero
at the bonded base), and (ii) it is run at ELEVATED growth (gamma passed via the server) with
*STATIC, STABILIZE so the in-plane compression can drive the seeded mode. As growth increases the
geometrically-softening film amplifies the out-of-plane deflection max|U3|; the dysbiotic profile (DH,
higher phi_Pg -> stronger compression) amplifies earlier/more than CH = growth-instability onset.

Run:  python gen_film_buckle_inp.py 16 8 film_buckle.inp [amp]   (default amp=5e-3)
"""
import sys, math
Nx  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
Nz  = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = sys.argv[3] if len(sys.argv) > 3 else "film_buckle.inp"
AMP = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0e-3
L, T, H = 4.0, 1.0, 1.0
dx, dz = L / Nx, H / Nz

def nid(i, j, k):
    return 1 + i + (Nx + 1) * (j + 2 * k)

lines = ["*HEADING", " Biofilm film strip, seeded imperfection -> growth-induced buckling (B1)", "*NODE"]
for k in range(Nz + 1):
    for j in range(2):
        for i in range(Nx + 1):
            x, z = i * dx, k * dz
            zp = z + AMP * (z / H) * math.sin(math.pi * x / L)   # bending imperfection (max at top)
            lines.append(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), x, j * T, zp))
lines.append("*ELEMENT, TYPE=C3D8, ELSET=FILM")
e = 0
for k in range(Nz):
    for i in range(Nx):
        e += 1
        n = [nid(i,0,k), nid(i+1,0,k), nid(i+1,1,k), nid(i,1,k),
             nid(i,0,k+1), nid(i+1,0,k+1), nid(i+1,1,k+1), nid(i,1,k+1)]
        lines.append(" %d, %s" % (e, ",".join(str(x) for x in n)))
lines += ["*SOLID SECTION, ELSET=FILM, MATERIAL=BIOFILM",
          "*MATERIAL, NAME=BIOFILM", "*USER MATERIAL, CONSTANTS=2", " 5000., 0.45", "*DEPVAR", " 1"]
xsym = [nid(0, j, k) for k in range(Nz + 1) for j in range(2)]
ysym = [nid(i, j, k) for k in range(Nz + 1) for j in range(2) for i in range(Nx + 1)]
zbot = [nid(i, j, 0) for j in range(2) for i in range(Nx + 1)]
def nset(name, ids):
    lines.append("*NSET, NSET=%s" % name)
    for s in range(0, len(ids), 12):
        lines.append(" " + ",".join(str(x) for x in ids[s:s+12]))
nset("XSYM", xsym); nset("YSYM", ysym); nset("ZBOT", zbot)
lines += ["*BOUNDARY", " XSYM, 1, 1", " YSYM, 2, 2", " ZBOT, 3, 3",
          "*STEP, NLGEOM=YES, INC=500", " elevated growth, seeded imperfection",
          "*STATIC, STABILIZE=1.0E-4", " 0.02, 1.0, 1.0E-6, 0.02",
          "*OUTPUT, FIELD, FREQUENCY=1", "*NODE OUTPUT", " U",
          "*ELEMENT OUTPUT, POSITION=CENTROID", " S", "*END STEP"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (%d elem, imperfection amp=%g)" % (OUT, e, AMP))
