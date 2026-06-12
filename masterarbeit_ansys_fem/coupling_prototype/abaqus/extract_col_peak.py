"""Print one convergence metric for a column odb: peak |S11| and S11 near the top.
Usage: abaqus python extract_col_peak.py <job>   (default column_fs)"""
from __future__ import annotations
import sys
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "column_fs"
o = openOdb(job + ".odb")
fr = list(o.steps.values())[-1].frames[-1]
S = fr.fieldOutputs['S']
COORD = fr.fieldOutputs['COORD']
cz = {v.elementLabel: v.data[2] for v in COORD.values}
vals = [(cz.get(v.elementLabel, 0.0), v.data[0]) for v in S.values]
peak = max(abs(s11) for _, s11 in vals)
top = max(vals, key=lambda zs: zs[0])[1]   # S11 of the top-most element
nel = len(vals)
print("%d  %.4f  %.4f" % (nel, peak, top))   # N  peak|S11|  S11_top
o.close()
