"""Extract interface stress along the bonded bottom row of the implant-thread biofilm deck.
Run: abaqus python extract_implant.py <job> <Nx>
Writes <job>_iface.json: per bottom element x, S11, S22(peel), S12(shear), von Mises; and the peaks.
"""
from __future__ import annotations
import sys
import json
import math
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "implant_DH_thread"
Nx = int(sys.argv[2]) if len(sys.argv) > 2 else 72

o = openOdb(job + ".odb")
fr = list(o.steps.values())[-1].frames[-1]
S = fr.fieldOutputs["S"]
COORD = fr.fieldOutputs["COORD"]
coord_by = {v.elementLabel: v.data for v in COORD.values}
S_by = {v.elementLabel: v.data for v in S.values}

rows = []
for el in range(1, Nx + 1):                 # bottom row elements e = i+1
    if el not in S_by:
        continue
    s = S_by[el]
    s11, s22, s33, s12 = float(s[0]), float(s[1]), float(s[2]), float(s[3])
    vm = math.sqrt(max(0.0, s11**2 + s22**2 + s33**2 - s11*s22 - s22*s33 - s33*s11 + 3*s12**2))
    x = float(coord_by.get(el, [0, 0])[0])
    rows.append({"x": x, "S11": s11, "S22_peel": s22, "S12_shear": s12, "vm": vm})

rows.sort(key=lambda r: r["x"])
peak_vm = max(r["vm"] for r in rows) if rows else 0.0
peak_peel = max(abs(r["S22_peel"]) for r in rows) if rows else 0.0
# INTERIOR peak: exclude the outer thread period at each lateral end, where the bonded-base/free-edge
# corner is a BC stress singularity (mesh-divergent). The interior is the convergent, physical response.
xmax = max((r["x"] for r in rows), default=0.0)
lo, hi = 1.0, xmax - 1.0
inner = [r for r in rows if lo <= r["x"] <= hi]
ipeak_vm = max((r["vm"] for r in inner), default=0.0)
ipeak_peel = max((abs(r["S22_peel"]) for r in inner), default=0.0)
out = {"job": job, "Nx": Nx, "peak_vm": peak_vm, "peak_peel": peak_peel,
       "interior_peak_vm": ipeak_vm, "interior_peak_peel": ipeak_peel, "rows": rows}
json.dump(out, open(job + "_iface.json", "w"), indent=1)
print("%s: interior peak vM=%.2f  |peel|=%.2f   (edge-incl vM=%.2f |peel|=%.2f)  (%d elems)"
      % (job, ipeak_vm, ipeak_peel, peak_vm, peak_peel, len(rows)))
o.close()
