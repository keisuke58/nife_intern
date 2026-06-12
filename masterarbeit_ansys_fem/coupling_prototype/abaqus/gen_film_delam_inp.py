"""Generate film_delam.inp — B2: a biofilm FILM STRIP on a COHESIVE base (delamination).

Same film strip as gen_film_inp.py (free right edge, depth-graded growth via the socket UMAT), but
the bottom is no longer perfectly bonded (u_z=0). Instead a single layer of COH3D8 cohesive elements
sits UNDER the film (z in [-h_coh, 0]); their top face shares the film's z=0 nodes, their bottom face
nodes are fully fixed (the substratum). The cohesive layer uses Abaqus' built-in traction-separation
law (NOT the user material) with a QUADS damage-initiation criterion + energy-based evolution.

Mechanism: growth-induced in-plane compression peels/shears the film off the base, concentrating
interface traction at the FREE EDGE. When the QUADS criterion is met there, cohesive damage (SDEG)
initiates and the delamination front propagates inward. DH (higher residual stress) delaminates
earlier / further than CH -> a direct FEM test of the dysbiosis -> detachment claim.

The film z-coords stay in [0,1] so the depth->phi_Pg mapping in umat_server_col.py is unchanged.

Run:  python gen_film_delam_inp.py 12 8 film_delam.inp [tn0] [Gc] [K]
  tn0  cohesive strength (peel & shear, MPa-equiv stress units)   default 60
  Gc   fracture energy (energy/area)                              default 4.0
  K    interface penalty stiffness (stress/length)                default 5.0e4
"""
import sys

Nx  = int(sys.argv[1]) if len(sys.argv) > 1 else 12
Nz  = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = sys.argv[3] if len(sys.argv) > 3 else "film_delam.inp"
TN0 = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
GC  = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
K   = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0e4
L, T, H = 4.0, 1.0, 1.0
HCOH = 0.02                       # cohesive layer geometric thickness (constitutive thickness = 1)
dx, dz = L / Nx, H / Nz

def nid(i, j, k):                 # film node: i in x(0..Nx), j in y(0..1), k in z(0..Nz)
    return 1 + i + (Nx + 1) * (j + 2 * k)

FILM_MAX = nid(Nx, 1, Nz)
CBASE = FILM_MAX + 100            # cohesive bottom-face node base (z = -HCOH, fixed)
def cnid(i, j):
    return CBASE + i + (Nx + 1) * j

lines = ["*HEADING",
         " Biofilm film strip on COHESIVE base (%dx1x%d C3D8 + %d COH3D8) -> delamination" % (Nx, Nz, Nx),
         "*NODE"]
# film nodes (z in [0,1])
for k in range(Nz + 1):
    for j in range(2):
        for i in range(Nx + 1):
            lines.append(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), i * dx, j * T, k * dz))
# cohesive bottom nodes (z = -HCOH)
for j in range(2):
    for i in range(Nx + 1):
        lines.append(" %d, %.6f, %.6f, %.6f" % (cnid(i, j), i * dx, j * T, -HCOH))

# film elements
lines.append("*ELEMENT, TYPE=C3D8, ELSET=FILM")
e = 0
for k in range(Nz):
    for i in range(Nx):
        e += 1
        n = [nid(i, 0, k), nid(i+1, 0, k), nid(i+1, 1, k), nid(i, 1, k),
             nid(i, 0, k+1), nid(i+1, 0, k+1), nid(i+1, 1, k+1), nid(i, 1, k+1)]
        lines.append(" %d, %s" % (e, ",".join(str(x) for x in n)))
# cohesive elements: bottom face = fixed base nodes, top face = film z=0 nodes
lines.append("*ELEMENT, TYPE=COH3D8, ELSET=COH")
ce0 = e
for i in range(Nx):
    e += 1
    n = [cnid(i, 0), cnid(i+1, 0), cnid(i+1, 1), cnid(i, 1),           # bottom (fixed base)
         nid(i, 0, 0), nid(i+1, 0, 0), nid(i+1, 1, 0), nid(i, 1, 0)]   # top (film bottom)
    lines.append(" %d, %s" % (e, ",".join(str(x) for x in n)))
# ELSET of cohesive elements ordered edge(x=0)->free(x=L) for delam-length extraction
lines.append("*ELSET, ELSET=COH_X, GENERATE")
lines.append(" %d, %d, 1" % (ce0 + 1, e))

lines += ["*SOLID SECTION, ELSET=FILM, MATERIAL=BIOFILM",
          "*MATERIAL, NAME=BIOFILM", "*USER MATERIAL, CONSTANTS=2", " 5000., 0.45", "*DEPVAR", " 1",
          "*COHESIVE SECTION, ELSET=COH, MATERIAL=COHMAT, RESPONSE=TRACTION SEPARATION", " 1.0",
          "*MATERIAL, NAME=COHMAT",
          "*ELASTIC, TYPE=TRACTION", " %g, %g, %g" % (K, K, K),
          "*DAMAGE INITIATION, CRITERION=QUADS", " %g, %g, %g" % (TN0, TN0, TN0),
          "*DAMAGE EVOLUTION, TYPE=ENERGY, MIXED MODE BEHAVIOR=BK, POWER=1.45", " %g, %g, %g" % (GC, GC, GC),
          "*DAMAGE STABILIZATION", " 1.0E-4"]
# node sets
xsym = [nid(0, j, k) for k in range(Nz + 1) for j in range(2)]                          # x=0 sym -> u_x=0
ysym = ([nid(i, j, k) for k in range(Nz + 1) for j in range(2) for i in range(Nx + 1)]  # plane strain -> u_y=0
        + [cnid(i, j) for j in range(2) for i in range(Nx + 1)])
cbot = [cnid(i, j) for j in range(2) for i in range(Nx + 1)]                            # substratum -> fully fixed
def nset(name, ids):
    lines.append("*NSET, NSET=%s" % name)
    for s in range(0, len(ids), 12):
        lines.append(" " + ",".join(str(x) for x in ids[s:s+12]))
nset("XSYM", xsym); nset("YSYM", ysym); nset("CBOT", cbot)
lines += ["*BOUNDARY", " XSYM, 1, 1", " YSYM, 2, 2", " CBOT, 1, 3",
          "*STEP, NLGEOM=YES, INC=500", " Depth-graded growth on cohesive base -> edge delamination",
          "*STATIC", " 0.02, 1.0, 1.0E-5, 0.02",
          "*OUTPUT, FIELD", "*NODE OUTPUT", " U",
          "*ELEMENT OUTPUT, POSITION=CENTROID", " S, E, SDV, SDEG, COORD",
          "*OUTPUT, HISTORY, VARIABLE=PRESELECT", "*END STEP"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (%d film + %d cohesive elem; tn0=%g Gc=%g K=%g; free edge x=%.1f)"
      % (OUT, ce0, Nx, TN0, GC, K, L))
