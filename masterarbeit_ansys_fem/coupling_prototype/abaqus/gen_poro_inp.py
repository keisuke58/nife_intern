"""A2 — Poroelastic drainage of a biofilm column (Terzaghi consolidation).

The elastic residual stress (thesis) is the SHORT-time state; on longer times the biofilm (a fluid-
saturated porous EPS) relieves it by DRAINING pore fluid. We model the growth-induced pressurisation as
an initial excess pore pressure p0 in a saturated poroelastic column (C3D8P), drained at the top
(biofilm-bulk) surface and sealed at the substratum, and let it consolidate (*SOILS, CONSOLIDATION,
no external load). The excess pressure dissipates over the Terzaghi time tau = H^2/c_v, linking the
elastic (short-time, thesis) and relaxed (long-time, Bolea-Albero pressure-restricted) limits.

Column height H along z; drained top (z=H, pp=0), sealed sides + base. p0 = 1.0 initial excess pp.
Run:  python gen_poro_inp.py 16 poro_col.inp [k] [E]
"""
import sys
Nz  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
OUT = sys.argv[2] if len(sys.argv) > 2 else "poro_col.inp"
PERM = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e-4    # permeability (length/time)
E    = float(sys.argv[4]) if len(sys.argv) > 4 else 5000.0
H, w = 1.0, 0.25
dz = H / Nz
def nid(i, j, k):  # i,j in {0,1} (x,y), k in 0..Nz (z)
    return 1 + i + 2 * (j + 2 * k)
lines = ["*HEADING", " Biofilm poroelastic column - Terzaghi consolidation (A2)", "*NODE"]
for k in range(Nz + 1):
    for j in range(2):
        for i in range(2):
            lines.append(" %d, %.5f, %.5f, %.5f" % (nid(i, j, k), i * w, j * w, k * dz))
lines.append("*ELEMENT, TYPE=C3D8P, ELSET=COL")
for k in range(Nz):
    e = k + 1
    n = [nid(0,0,k), nid(1,0,k), nid(1,1,k), nid(0,1,k),
         nid(0,0,k+1), nid(1,0,k+1), nid(1,1,k+1), nid(0,1,k+1)]
    lines.append(" %d, %s" % (e, ",".join(str(x) for x in n)))
lines += ["*SOLID SECTION, ELSET=COL, MATERIAL=PORO",
          "*MATERIAL, NAME=PORO",
          "*ELASTIC", " %g, 0.3" % E,
          "*PERMEABILITY, SPECIFIC=10.0", " %g, 1.0" % PERM,   # k vs void ratio
          "*POROUS BULK MODULI", " 2.2E5, 2.2E9"]              # fluid, solid grain bulk moduli
# node sets
alln = [nid(i, j, k) for k in range(Nz + 1) for j in range(2) for i in range(2)]
topn = [nid(i, j, Nz) for j in range(2) for i in range(2)]
botn = [nid(i, j, 0) for j in range(2) for i in range(2)]
def nset(name, ids):
    lines.append("*NSET, NSET=%s" % name)
    for s in range(0, len(ids), 12):
        lines.append(" " + ",".join(str(x) for x in ids[s:s+12]))
nset("ALLN", alln); nset("TOPN", topn); nset("BOTN", botn)
lines += ["*INITIAL CONDITIONS, TYPE=RATIO", " ALLN, 1.0",
          "*INITIAL CONDITIONS, TYPE=PORE PRESSURE", " ALLN, 1.0",   # growth-induced excess pp
          # confined oedometer: lateral + base displacement fixed, drained top
          "*BOUNDARY", " ALLN, 1, 2", " BOTN, 3, 3",
          "*STEP, NLGEOM=NO, INC=500", " consolidation / drainage",
          "*SOILS, CONSOLIDATION, UTOL=0.1", " 1.0E-6, 50.0, 1.0E-10, 2.0",
          "*BOUNDARY", " TOPN, 8, 8, 0.0",                            # drained top: pp=0
          "*OUTPUT, FIELD, FREQUENCY=1", "*NODE OUTPUT", " U, POR", "*END STEP"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (%d C3D8P, perm=%g, drained top, p0=1.0)" % (OUT, Nz, PERM))
