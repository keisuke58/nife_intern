"""Generate film_fs.inp — a plane-strain biofilm FILM STRIP with a FREE EDGE.

Unlocks B1 (buckling) and B2 (delamination): unlike the laterally-confined column, a strip with a
free right edge lets the growth-induced in-plane compression concentrate at the edge — the driver
for both edge delamination and edge buckling.

Geometry: Nx (length x) x 1 (thickness y, plane strain via y-symmetry) x Nz (depth z), height 1.
BCs: x=0 symmetry (u_x=0); y=0 & y=t faces u_y=0 (plane strain); bottom z=0 u_z=0 (bonded base);
free right edge (x=L) and free top (z=1). Growth via the col server (COORDS-driven, depth-graded).

Run:  python gen_film_inp.py 12 8 film_fs.inp   # Nx=12, Nz=8
Recipes to extend (see EXTENSIONS_RECIPES.md):
  - delamination: replace bottom u_z=0 with cohesive surface to a rigid base (*COHESIVE BEHAVIOR).
  - buckling: keep bottom fixed, raise gamma to ~17, add a small z-imperfection at the free edge,
    use *STATIC, STABILIZE (or *BUCKLE for the linearised eigenmode).
"""
import sys

Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 12
Nz = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = sys.argv[3] if len(sys.argv) > 3 else "film_fs.inp"
L, T, H = 4.0, 1.0, 1.0           # length, y-thickness, height
dx, dz = L / Nx, H / Nz

def nid(i, j, k):                 # i in x(0..Nx), j in y(0..1), k in z(0..Nz)
    return 1 + i + (Nx + 1) * (j + 2 * k)

lines = ["*HEADING",
         " Biofilm film strip (%dx1x%d C3D8) free edge -> edge stress (B1 buckling / B2 delam base)" % (Nx, Nz),
         "*NODE"]
for k in range(Nz + 1):
    for j in range(2):
        for i in range(Nx + 1):
            lines.append(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), i * dx, j * T, k * dz))
lines.append("*ELEMENT, TYPE=C3D8, ELSET=FILM")
e = 0
for k in range(Nz):
    for i in range(Nx):
        e += 1
        n = [nid(i, 0, k), nid(i+1, 0, k), nid(i+1, 1, k), nid(i, 1, k),
             nid(i, 0, k+1), nid(i+1, 0, k+1), nid(i+1, 1, k+1), nid(i, 1, k+1)]
        lines.append(" %d, %s" % (e, ",".join(str(x) for x in n)))
lines += ["*SOLID SECTION, ELSET=FILM, MATERIAL=BIOFILM",
          "*MATERIAL, NAME=BIOFILM", "*USER MATERIAL, CONSTANTS=2", " 5000., 0.45",
          "*DEPVAR", " 1"]
# node sets
xsym = [nid(0, j, k) for k in range(Nz + 1) for j in range(2)]            # x=0 -> u_x=0
ysym = [nid(i, j, k) for k in range(Nz + 1) for j in range(2) for i in range(Nx + 1)]  # all -> u_y=0 (plane strain)
zbot = [nid(i, j, 0) for j in range(2) for i in range(Nx + 1)]           # z=0 -> u_z=0
def nset(name, ids):
    lines.append("*NSET, NSET=%s" % name)
    for s in range(0, len(ids), 12):
        lines.append(" " + ",".join(str(x) for x in ids[s:s+12]))
nset("XSYM", xsym); nset("YSYM", ysym); nset("ZBOT", zbot)
lines += ["*BOUNDARY", " XSYM, 1, 1", " YSYM, 2, 2", " ZBOT, 3, 3",
          "*STEP, NLGEOM=YES, INC=200", " Depth-graded growth, free edge",
          "*STATIC", " 0.05, 1.0, 1.0E-4, 0.05",
          "*OUTPUT, FIELD", "*NODE OUTPUT", " U",
          "*ELEMENT OUTPUT, POSITION=CENTROID", " S, E, SDV, COORD",
          "*OUTPUT, HISTORY, VARIABLE=PRESELECT", "*END STEP"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (%d elements, %d nodes; free edge at x=%.1f)" % (OUT, e, 2 * (Nx + 1) * (Nz + 1), L))
