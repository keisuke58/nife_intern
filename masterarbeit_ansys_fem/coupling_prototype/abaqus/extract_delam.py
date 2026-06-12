"""B2 delamination result: cohesive damage (SDEG) profile + delamination length from the free edge.
Usage: abaqus python extract_delam.py <job> [Nx] [Nz] [L]   (defaults film_delam 12 8 4.0)

Cohesive elements are labelled (Nz*Nx + 1 .. Nz*Nx + Nx), ordered x=0 (symmetry) -> x=L (free edge).
SDEG=0 intact, SDEG=1 fully delaminated. Delamination length = contiguous fully-damaged run measured
inward FROM THE FREE EDGE x=L (that is where growth-induced traction concentrates)."""
from __future__ import annotations
import sys
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "film_delam"
Nx  = int(sys.argv[2]) if len(sys.argv) > 2 else 12
Nz  = int(sys.argv[3]) if len(sys.argv) > 3 else 8
L   = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
dx  = L / Nx
o = openOdb(job + ".odb")
fr = list(o.steps.values())[-1].frames[-1]
tfin = fr.frameValue
sdeg = {v.elementLabel: v.data for v in fr.fieldOutputs['SDEG'].values}
coh0 = Nz * Nx                      # last film element label; cohesive are coh0+1 .. coh0+Nx
prof = [sdeg.get(coh0 + i + 1, 0.0) for i in range(Nx)]   # i=0 at x=0 sym, i=Nx-1 at free edge

print("step fraction reached: %.3f  (1.0 = full growth)" % tfin)
print("SDEG along x  (x=0 sym -> free edge):  " + "  ".join("%4.2f" % s for s in prof))
# delamination length: contiguous run with SDEG>0.5 starting at the free edge, going inward
delam = 0
for i in range(Nx - 1, -1, -1):
    if prof[i] > 0.5:
        delam += 1
    else:
        break
print("delaminated elements from free edge: %d / %d   ->   delam length = %.3f  (%.0f%% of L)"
      % (delam, Nx, delam * dx, 100.0 * delam / Nx))
print("max SDEG = %.3f   (>~0.99 => fully separated interface)" % (max(prof) if prof else 0.0))
o.close()
