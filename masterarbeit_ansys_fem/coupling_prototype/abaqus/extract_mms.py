"""A3 MMS error: discrete L2 norm of (u_x^h - u_x^exact), u_x^exact = alpha sin(pi X/L).
Usage: abaqus python extract_mms.py mms_N<Nx> <Nx>   -> prints  Nx  h  L2err"""
from __future__ import annotations
import sys, math
from odbAccess import openOdb

job = sys.argv[1]
Nx  = int(sys.argv[2])
L, ALPHA = 1.0, 1.0e-4
o = openOdb(job + ".odb")
inst = list(o.rootAssembly.instances.values())[0]
X = {n.label: n.coordinates[0] for n in inst.nodes}      # reference X of each node
fr = list(o.steps.values())[-1].frames[-1]
num = 0.0; den = 0.0
for v in fr.fieldOutputs['U'].values:
    ux = v.data[0]
    ue = ALPHA * math.sin(math.pi * X[v.nodeLabel] / L)
    num += (ux - ue) ** 2
    den += 1.0
l2 = math.sqrt(num / den)
print("%d  %.6f  %.6e" % (Nx, L / Nx, l2))
o.close()
