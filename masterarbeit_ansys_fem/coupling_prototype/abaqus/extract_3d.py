"""Dump 3-D element centroids + von Mises (and radius) for the helical screw, for visualisation.
Run: abaqus python extract_3d.py <job>"""
from __future__ import annotations
import sys, json, math
from odbAccess import openOdb
job = sys.argv[1]
o = openOdb(job + ".odb"); fr = list(o.steps.values())[-1].frames[-1]
S = {v.elementLabel: v.data for v in fr.fieldOutputs["S"].values}
C = {v.elementLabel: v.data for v in fr.fieldOutputs["COORD"].values}
els = []
for el, s in S.items():
    s11, s22, s33, s12, s13, s23 = [float(s[m]) for m in range(6)]
    vm = math.sqrt(0.5*((s11-s22)**2+(s22-s33)**2+(s33-s11)**2)+3*(s12**2+s13**2+s23**2))
    x, y, z = float(C[el][0]), float(C[el][1]), float(C[el][2])
    els.append({"x": x, "y": y, "z": z, "r": math.sqrt(x*x+y*y), "vm": vm})
json.dump({"job": job, "els": els}, open(job + "_3d.json", "w"))
print("%s: %d elements, max vM=%.1f, r range %.2f-%.2f" %
      (job, len(els), max(e["vm"] for e in els), min(e["r"] for e in els), max(e["r"] for e in els)))
o.close()
