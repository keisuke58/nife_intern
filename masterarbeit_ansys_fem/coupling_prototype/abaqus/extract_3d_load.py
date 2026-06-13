"""Inner-interface vM at growth (step1) vs growth+load (step2) for the 3D screw load cases.
Delta_sigma (load-induced amplitude at the interface) is the cyclic-fatigue driver.
Run: abaqus python extract_3d_load.py <job> <Nt> <Nr>"""
from __future__ import annotations
import sys, json, math
from odbAccess import openOdb
job = sys.argv[1]; Nt = int(sys.argv[2]) if len(sys.argv)>2 else 48; Nr = int(sys.argv[3]) if len(sys.argv)>3 else 4
o = openOdb(job + ".odb"); steps = list(o.steps.values())
def vmmap(fr):
    out = {}
    for v in fr.fieldOutputs["S"].values:
        s = v.data
        out[v.elementLabel] = math.sqrt(0.5*((s[0]-s[1])**2+(s[1]-s[2])**2+(s[2]-s[0])**2)+3*(s[3]**2+s[4]**2+s[5]**2))
    return out
g = vmmap(steps[0].frames[-1])
L = vmmap(steps[-1].frames[-1]) if len(steps) > 1 else g
inner = [e for e in g if ((e-1)//Nt) % Nr == 0]      # radial layer 0 (bonded interface)
pg = max(g[e] for e in inner); pl = max(L[e] for e in inner)
dmax = max(abs(L[e]-g[e]) for e in inner)
json.dump({"job": job, "growth_peak": pg, "loaded_peak": pl, "delta_amp": dmax}, open(job+"_load.json","w"))
print("%s: interface vM growth=%.1f  growth+load=%.1f  load-amplitude Delta=%.1f" % (job, pg, pl, dmax))
o.close()
