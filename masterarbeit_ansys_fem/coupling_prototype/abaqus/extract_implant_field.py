"""Dump every element's centroid + von Mises (and S components) to json for field plotting.
Run: abaqus python extract_implant_field.py <job>
"""
from __future__ import annotations
import sys
import json
import math
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "implant_DH_thread"
o = openOdb(job + ".odb")
fr = list(o.steps.values())[-1].frames[-1]
S = {v.elementLabel: v.data for v in fr.fieldOutputs["S"].values}
C = {v.elementLabel: v.data for v in fr.fieldOutputs["COORD"].values}
els = []
for el, s in S.items():
    s11, s22, s33, s12 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
    vm = math.sqrt(max(0.0, s11**2 + s22**2 + s33**2 - s11*s22 - s22*s33 - s33*s11 + 3*s12**2))
    x, y = float(C[el][0]), float(C[el][1])
    els.append({"x": x, "y": y, "vm": vm, "S22": s22})
json.dump({"job": job, "els": els}, open(job + "_field.json", "w"))
print("%s: %d elements, max vm=%.1f" % (job, len(els), max(e["vm"] for e in els)))
o.close()
