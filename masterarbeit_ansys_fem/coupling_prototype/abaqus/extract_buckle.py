"""B1 buckling: max|U3| vs growth fraction (frame value) over the whole step.
Usage: abaqus python extract_buckle.py <job>   -> writes <job>_u3.txt with  growthfrac  maxU3"""
from __future__ import annotations
import sys
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "film_buckle"
o = openOdb(job + ".odb")
step = list(o.steps.values())[-1]
out = []
for fr in step.frames:
    try:
        u3 = max(abs(v.data[2]) for v in fr.fieldOutputs['U'].values)
    except KeyError:
        continue
    out.append((fr.frameValue, u3))
with open(job + "_u3.txt", "w") as f:
    for t, u in out:
        f.write("%.4f  %.6e\n" % (t, u))
print("growth-fraction   max|U3|")
for t, u in out:
    print("  %.3f            %.5f" % (t, u))
print("FINAL max|U3| = %.5f  (wrote %s_u3.txt, %d frames)" % (out[-1][1], job, len(out)))
o.close()
