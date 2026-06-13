"""Instantaneous (step1) vs relaxed (step2) interface peak vM for the VE thread deck.
Run: abaqus python extract_ve_thread.py <job> <Nx>"""
from __future__ import annotations
import sys, json, math
from odbAccess import openOdb
job = sys.argv[1]; Nx = int(sys.argv[2]) if len(sys.argv) > 2 else 108
o = openOdb(job + ".odb"); steps = list(o.steps.values())
def peakvm(fr):
    S = {v.elementLabel: v.data for v in fr.fieldOutputs["S"].values}
    best = 0.0
    for el in range(1, Nx + 1):          # bottom (interface) row
        if el in S:
            s = S[el]; s11, s22, s33, s12 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
            best = max(best, math.sqrt(max(0.0, s11**2+s22**2+s33**2-s11*s22-s22*s33-s33*s11+3*s12**2)))
    return best
inst = peakvm(steps[0].frames[-1]); rel = peakvm(steps[-1].frames[-1])
json.dump({"job": job, "instant": inst, "relaxed": rel, "ratio": rel/inst if inst else 0}, open(job+"_ve.json","w"))
print("%s: instantaneous peak vM=%.1f  relaxed peak vM=%.1f  (relaxed/instant=%.2f)" % (job, inst, rel, rel/inst if inst else 0))
o.close()
