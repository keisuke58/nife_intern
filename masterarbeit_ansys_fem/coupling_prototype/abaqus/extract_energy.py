"""A2 energy balance: hyperelastic -> ALLSE (strain energy) should equal ALLWK (external work),
ALLPD (plastic dissipation) ~ 0. Usage: abaqus python extract_energy.py <job>  (default column_dh)"""
from __future__ import annotations
import sys
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "column_dh"
o = openOdb(job + ".odb")
step = list(o.steps.values())[-1]
hr = list(step.historyRegions.values())[0]
def last(name):
    try:
        return hr.historyOutputs[name].data[-1][1]
    except KeyError:
        return float('nan')
se, wk, pd = last('ALLSE'), last('ALLWK'), last('ALLPD')
print("ALLSE=%.4g  ALLWK=%.4g  ALLPD=%.4g" % (se, wk, pd))
rel = abs(se - wk) / (abs(wk) + 1e-30)
print("strain energy vs external work: rel-diff %.2e  (%s)" % (rel, "PASS" if rel < 1e-2 else "check"))
o.close()
