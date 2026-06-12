"""Film strip results: edge stress + out-of-plane deflection (buckling/delam precursor).
Usage: abaqus python extract_film.py <job> [Nx] [Nz]   (defaults film_fs 12 8)
Selects the mid-depth element ROW by element-label arithmetic (robust under NLGEOM deformation,
unlike binning on deformed COORD): elem e = k*Nx + i + 1, row k, column i along x (free edge at i=Nx-1).
Also reports top-row vs bottom-row S11 (bending signature) and max |U3|."""
from __future__ import annotations
import sys
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "film_fs"
Nx = int(sys.argv[2]) if len(sys.argv) > 2 else 12
Nz = int(sys.argv[3]) if len(sys.argv) > 3 else 8
o = openOdb(job + ".odb")
fr = list(o.steps.values())[-1].frames[-1]
S = {v.elementLabel: v.data[0] for v in fr.fieldOutputs['S'].values}   # S11
U = fr.fieldOutputs['U']

def row(k):  # S11 along x for depth-row k
    return [S.get(k * Nx + i + 1, float('nan')) for i in range(Nx)]

for tag, k in [("bottom z~0", 0), ("mid", Nz // 2), ("top z~1", Nz - 1)]:
    r = row(k)
    print("S11 along x (%s):  edge->  " % tag + "  ".join("%6.0f" % s for s in r))
u3 = max(abs(v.data[2]) for v in U.values)
print("max |U3| (out-of-plane / buckling-delam indicator) = %.4f" % u3)
print("(in-plane stress relaxes toward the free edge at i=%d; sign change top vs bottom = bending/curl)" % (Nx - 1))
o.close()
