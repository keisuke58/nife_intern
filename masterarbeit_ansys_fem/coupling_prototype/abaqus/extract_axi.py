"""Inner-interface hoop + vM stress for the axisymmetric screw deck.
Run: abaqus python extract_axi.py <job> <Nr>"""
from __future__ import annotations
import sys, json, math
from odbAccess import openOdb
job = sys.argv[1]; Nr = int(sys.argv[2]) if len(sys.argv) > 2 else 16
o = openOdb(job + ".odb"); fr = list(o.steps.values())[-1].frames[-1]
S = {v.elementLabel: v.data for v in fr.fieldOutputs["S"].values}
# innermost radial elements: e = j*Nr + 1
hoop = []; vmv = []
for el, s in S.items():
    if (el - 1) % Nr == 0:                      # i=0 column (inner interface)
        s11, s22, s33, s12 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        vm = math.sqrt(max(0.0, s11**2+s22**2+s33**2 - s11*s22-s22*s33-s33*s11 + 3*s12**2))
        hoop.append(abs(s33)); vmv.append(vm)
peak_hoop = max(hoop) if hoop else 0.0; peak_vm = max(vmv) if vmv else 0.0
json.dump({"job": job, "peak_hoop": peak_hoop, "peak_vm": peak_vm}, open(job + "_axi.json", "w"))
print("%s: inner-interface peak hoop|S33|=%.2f  peak vM=%.2f" % (job, peak_hoop, peak_vm))
o.close()
