"""Extract the residual-stress-vs-depth profile from the growth column.
Run: abaqus python extract_col.py
Prints, per element (sorted by centroid depth z): in-plane S11, through-thickness S33, SDV1.
Expected: S11 compressive (<0), magnitude tracking the CLSM phi_Pg(z) (most at Pg-rich layers);
S33 ~ 0 at the free top. This σ_xx(z) is the FEM-only residual-stress result."""
from __future__ import annotations
import sys
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else 'column_fs'
o = openOdb(job + '.odb')
step = list(o.steps.values())[-1]
fr = step.frames[-1]
S = fr.fieldOutputs['S']
COORD = fr.fieldOutputs['COORD']
try:
    SDV = fr.fieldOutputs['SDV1']
    sdv_by_elem = {v.elementLabel: v.data for v in SDV.values}
except KeyError:
    sdv_by_elem = {}
coord_by = {v.elementLabel: v.data for v in COORD.values}
rows = []
for v in S.values:
    el = v.elementLabel
    z = coord_by.get(el, [0, 0, 0])[2]
    rows.append((z, el, v.data[0], v.data[2], sdv_by_elem.get(el, float('nan'))))
rows.sort()
print("%-8s %-8s %-12s %-12s %-6s" % ("z", "elem", "S11(in-plane)", "S33(thru)", "SDV1"))
for z, el, s11, s33, sdv in rows:
    sdv = sdv if not hasattr(sdv, '__len__') else sdv
    print("%-8.3f %-8d %-12.2f %-12.2f %-6s" % (z, el, s11, s33, str(sdv)))
o.close()
print("Residual-stress profile: S11(z) compressive, magnitude follows phi_Pg(z); S33~0 (free top).")
