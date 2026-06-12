"""Generate column_fs.inp — an N-element biofilm growth column (laterally-confined film).

A 1x1xN stack of C3D8 elements (height 1.0, z = depth). BCs make it a representative column of a
laterally-infinite substrate-bonded film: both x-faces u_x=0, both y-faces u_y=0 (so ε_xx=ε_yy=0),
bottom z=0 fixed in z, top free. Isotropic depth-graded growth (server) then develops an in-plane
residual-stress profile σ_xx(z). NLGEOM, automatic incrementation.

Run:  python gen_column_inp.py 8                # -> column_fs.inp
      python gen_column_inp.py 16 column16.inp  # -> column16.inp (mesh convergence)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = sys.argv[2] if len(sys.argv) > 2 else "column_fs.inp"
H = 1.0
dz = H / N

lines = []
ap = lines.append
ap("*HEADING")
ap(" Biofilm growth column (%d C3D8) — depth-graded isotropic growth -> residual stress" % N)
ap("** Laterally-confined film: u_x=0 on x-faces, u_y=0 on y-faces, u_z=0 at z=0, top free.")
ap("*NODE")
# node id = layer*100 + corner(1..4); layers 0..N at z = layer*dz
def nid(layer, c):
    return layer * 100 + c
for layer in range(N + 1):
    z = layer * dz
    ap(" %d, 0.0, 0.0, %.6f" % (nid(layer, 1), z))
    ap(" %d, 1.0, 0.0, %.6f" % (nid(layer, 2), z))
    ap(" %d, 1.0, 1.0, %.6f" % (nid(layer, 3), z))
    ap(" %d, 0.0, 1.0, %.6f" % (nid(layer, 4), z))
ap("*ELEMENT, TYPE=C3D8, ELSET=COL")
for e in range(N):
    b = [nid(e, 1), nid(e, 2), nid(e, 3), nid(e, 4),
         nid(e + 1, 1), nid(e + 1, 2), nid(e + 1, 3), nid(e + 1, 4)]
    ap(" %d, %s" % (e + 1, ",".join(str(x) for x in b)))
ap("*SOLID SECTION, ELSET=COL, MATERIAL=BIOFILM")
ap("*MATERIAL, NAME=BIOFILM")
ap("*USER MATERIAL, CONSTANTS=2")
ap(" 5000., 0.45")
ap("*DEPVAR")
ap(" 1")
# node sets for BCs
xfix = [nid(l, c) for l in range(N + 1) for c in (1, 2, 3, 4)]  # all (x=0 or x=1) -> u_x=0
yfix = xfix
zfix = [nid(0, c) for c in (1, 2, 3, 4)]
ap("*NSET, NSET=XF")
ap(" " + ",".join(str(x) for x in [nid(l, c) for l in range(N + 1) for c in (1, 4)]))  # x=0 faces
ap("*NSET, NSET=XR")
ap(" " + ",".join(str(x) for x in [nid(l, c) for l in range(N + 1) for c in (2, 3)]))  # x=1 faces
ap("*NSET, NSET=YF")
ap(" " + ",".join(str(x) for x in [nid(l, c) for l in range(N + 1) for c in (1, 2)]))  # y=0 faces
ap("*NSET, NSET=YR")
ap(" " + ",".join(str(x) for x in [nid(l, c) for l in range(N + 1) for c in (3, 4)]))  # y=1 faces
ap("*NSET, NSET=ZB")
ap(" " + ",".join(str(x) for x in zfix))
ap("*BOUNDARY")
ap(" XF, 1, 1")
ap(" XR, 1, 1")
ap(" YF, 2, 2")
ap(" YR, 2, 2")
ap(" ZB, 3, 3")
ap("*STEP, NLGEOM=YES, INC=100")
ap(" Depth-graded growth -> residual stress")
ap("*STATIC")
ap(" 0.05, 1.0, 1.0E-4, 0.05")
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, E, SDV, COORD")
ap("*OUTPUT, HISTORY, VARIABLE=PRESELECT")   # ALLSE/ALLWK/ALLIE for energy balance (A2)
ap("*END STEP")

open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (%d elements, %d nodes)" % (OUT, N, 4 * (N + 1)))
