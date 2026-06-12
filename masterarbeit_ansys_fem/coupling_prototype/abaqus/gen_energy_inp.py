"""Generate energy_test.inp — A2 energy-conservation verification (clean, EXTERNAL work).

Eigenstrain (growth) does internal work that Abaqus does NOT count in ALLWK, so a global
ALLSE=ALLWK balance is meaningless for the growth problem. Here growth is OFF (run with
phipg_zero.json -> Fg=I, pure compressible neo-Hookean) and the block is stretched by a PRESCRIBED
top displacement. With no dissipation, energy conservation requires ALLSE (stored strain energy) =
ALLWK (external work) to machine precision; both also match the analytic neo-Hookean energy.

Single C3D8 unit cube, bottom z=0 on rollers (u_z=0) + minimal rigid-body restraint, top z=1 pulled
to u_z = UZ (lateral faces free -> Poisson contraction). NLGEOM=YES.

Run:  python gen_energy_inp.py energy_test.inp [UZ]   (default 0.20)
"""
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "energy_test.inp"
UZ  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
P = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
lines = ["*HEADING", " A2 energy conservation: neo-Hookean unit cube, prescribed stretch (growth off)",
         "*NODE"]
for n, (x, y, z) in enumerate(P, 1):
    lines.append(" %d, %.1f, %.1f, %.1f" % (n, x, y, z))
lines += ["*ELEMENT, TYPE=C3D8, ELSET=BLK", " 1, 1,2,3,4,5,6,7,8",
          "*SOLID SECTION, ELSET=BLK, MATERIAL=BIOFILM",
          "*MATERIAL, NAME=BIOFILM", "*USER MATERIAL, CONSTANTS=2", " 5000., 0.45", "*DEPVAR", " 1",
          "*NSET, NSET=ZBOT", " 1,2,3,4", "*NSET, NSET=ZTOP", " 5,6,7,8",
          "*BOUNDARY", " ZBOT, 3, 3", " 1, 1, 2", " 2, 2, 2", " 4, 1, 1",   # rollers + kill rigid body
          "*STEP, NLGEOM=YES, INC=100", " uniaxial stretch by prescribed top displacement",
          "*STATIC", " 0.05, 1.0, 1.0E-4, 0.05",
          "*BOUNDARY", " ZTOP, 3, 3, %.4f" % UZ,
          "*OUTPUT, FIELD", "*NODE OUTPUT", " U, RF", "*ELEMENT OUTPUT, POSITION=CENTROID", " S, E",
          "*OUTPUT, HISTORY, VARIABLE=PRESELECT", "*END STEP"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (1 C3D8, prescribed u_z=%.3f, growth off)" % (OUT, UZ))
