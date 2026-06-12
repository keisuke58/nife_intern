"""Extract & sanity-check the finite-strain growth run. Run: abaqus python extract_fs.py
Prints, per frame: top-face z-displacement U3 (the biofilm thickening), element von-Mises-ish
stress level, and SDV1 (the NSP step counter). Expected: U3 grows toward lambda_z-1 (~0.3 for the
5-step ramp), stress stays low (free top), SDV1 climbs 1..5."""
from __future__ import annotations
from odbAccess import openOdb

o = openOdb('one_element_fs.odb')
step = list(o.steps.values())[-1]
ra = o.rootAssembly
# a top-face node (z=1): instance node label 7 (corner 1,1,1)
print("%-6s %-10s %-12s %-10s" % ("frame", "U3(top)", "|S|max", "SDV1"))
for i, fr in enumerate(step.frames):
    U = fr.fieldOutputs['U']
    # node set: pick the value with the largest z-displacement (top face)
    u3 = max(v.data[2] for v in U.values)
    S = fr.fieldOutputs['S'].values[0].data
    smax = max(abs(x) for x in S)
    try:
        sdv = fr.fieldOutputs['SDV1'].values[0].data
    except KeyError:
        sdv = float('nan')
    print("%-6d %-10.4f %-12.2f %-10.1f" % (i, u3, smax, sdv))
o.close()
print("PASS criteria: U3(top) increases (z-thickening), SDV1 climbs per increment.")
