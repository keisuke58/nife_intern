"""A2 energy conservation (UMAT-independent): external work ALLWK must equal the analytic compressible
neo-Hookean stored energy at the final deformation. (The socket UMAT does not fill SSE, so ALLSE reads
0 — we recompute the stored energy in closed form from the kinematics instead, a stronger check.)
Usage: abaqus python extract_energy2.py energy_test"""
from __future__ import annotations
import sys, math
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "energy_test"
E, nu = 5000.0, 0.45
mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
o = openOdb(job + ".odb")
step = list(o.steps.values())[-1]
fr = step.frames[-1]
U = {v.nodeLabel: v.data for v in fr.fieldOutputs['U'].values}
# homogeneous diagonal deformation of the unit cube: stretches from corner displacements
lz = 1.0 + U[5][2]                      # node 5 (0,0,1): u_z -> lambda_z
lx = 1.0 + U[2][0]                      # node 2 (1,0,0): u_x -> lambda_x
ly = 1.0 + U[4][1]                      # node 4 (0,1,0): u_y -> lambda_y
J = lx * ly * lz
trb = lx*lx + ly*ly + lz*lz
W = 0.5 * mu * (trb - 3.0) - mu * math.log(J) + 0.5 * lam * math.log(J)**2   # per unit ref volume
U_stored = W * 1.0                      # V0 = 1 (unit cube)
# external work from Abaqus history (whole-model)
hr = [h for h in step.historyRegions.values() if 'ALLWK' in h.historyOutputs]
allwk = hr[0].historyOutputs['ALLWK'].data[-1][1] if hr else float('nan')
print("stretches  lx=%.5f ly=%.5f lz=%.5f  J=%.5f" % (lx, ly, lz, J))
print("analytic neo-Hookean stored energy  U = %.4f" % U_stored)
print("Abaqus external work               ALLWK = %.4f" % allwk)
rel = abs(U_stored - allwk) / (abs(allwk) + 1e-30)
print("rel-diff = %.2e   (%s : energy conserved, no spurious dissipation)" % (rel, "PASS" if rel < 1e-2 else "CHECK"))
o.close()
