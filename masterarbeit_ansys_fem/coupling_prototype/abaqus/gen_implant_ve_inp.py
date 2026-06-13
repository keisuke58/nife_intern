"""Viscoelastic (Prony) relaxation of the implant-thread interface stress.
Same threaded eigenstrain deck, but each material is *ELASTIC + *EXPANSION + *VISCOELASTIC (biofilm
Prony series, prony_biofilm.json). Step 1 STATIC applies the growth (instantaneous stress); step 2
VISCO holds it and lets the deviatoric stress relax over t >> tau. Shows how much of the thread
interface stress relaxes (deviatoric) vs persists (volumetric).
Run: python gen_implant_ve_inp.py <cond> <out.inp> [Nx Nz periods amp]
"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))
PRONY = json.load(open(HERE.parent / "prony_biofilm.json"))
cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
OUT = sys.argv[2] if len(sys.argv) > 2 else "ve_%s.inp" % cond
Nx = int(sys.argv[3]) if len(sys.argv) > 3 else 108
Nz = int(sys.argv[4]) if len(sys.argv) > 4 else 18
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 3
AMP = float(sys.argv[6]) if len(sys.argv) > 6 else 0.18
E, NU = 5000.0, 0.45
BETA, IC = CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"]
sys.path.insert(0, str(HERE.parent.parent.parent)); sys.path.insert(0, str(HERE.parent.parent / "extensions"))
from b4_viscoelastic_growth_stress import depth_phi_pg
_zn, _pg = depth_phi_pg(cond)
def eps_g(z): return max(0.0, BETA*float(np.interp(z,_zn,_pg))+IC)
def yth(x): return AMP*0.5*(1.0-np.cos(2.0*np.pi*x))
def nid(i,j): return j*(Nx+1)+i+1
gi = [p["g_i"] for p in PRONY["prony"]]; ti = [p["tau_i_s"] for p in PRONY["prony"]]
ssum = sum(gi); gi = [g/ssum*(1-PRONY["g_inf"]) for g in gi]   # normalize so sum gi = 1 - g_inf
L=[]; ap=L.append
ap("*HEADING"); ap(" Viscoelastic implant-thread, cond=%s"%cond)
ap("*NODE"); xs=np.linspace(0,float(PER),Nx+1)
for j in range(Nz+1):
  for i in range(Nx+1): ap(" %d, %.6f, %.6f"%(nid(i,j), xs[i], yth(xs[i])+(j/Nz)*1.0))
ap("*ELEMENT, TYPE=CPE4")
for j in range(Nz):
  for i in range(Nx):
    e=j*Nx+i+1; ap(" %d, %d, %d, %d, %d"%(e,nid(i,j),nid(i+1,j),nid(i+1,j+1),nid(i,j+1)))
for j in range(Nz):
  els=[j*Nx+i+1 for i in range(Nx)]; ap("*ELSET, ELSET=ROW%d"%j)
  for k in range(0,len(els),16): ap(" "+",".join(str(x) for x in els[k:k+16]))
ap("*ORIENTATION, NAME=ORI, SYSTEM=RECTANGULAR, DEFINITION=COORDINATES"); ap(" 1.0,0.0,0.0,0.0,1.0,0.0"); ap(" 3, 0.0")
for j in range(Nz): ap("*SOLID SECTION, ELSET=ROW%d, MATERIAL=MAT%d, ORIENTATION=ORI"%(j,j))
for j in range(Nz):
  a=(1.0+eps_g((j+0.5)/Nz))**(1.0/3.0)-1.0
  ap("*MATERIAL, NAME=MAT%d"%j); ap("*ELASTIC"); ap(" %.1f, %.3f"%(E,NU))
  ap("*EXPANSION, TYPE=ORTHO, ZERO=0.0"); ap(" %.6f, %.6f, %.6f"%(a,a,a))
  ap("*VISCOELASTIC, TIME=PRONY")
  for g,t in zip(gi,ti): ap(" %.6f, 0.0, %.6f"%(g,t))
bottom=[nid(i,0) for i in range(Nx+1)]; alln=[nid(i,j) for j in range(Nz+1) for i in range(Nx+1)]
for nm,ids in (("BOTTOM",bottom),("ALLN",alln)):
  ap("*NSET, NSET=%s"%nm)
  for k in range(0,len(ids),16): ap(" "+",".join(str(x) for x in ids[k:k+16]))
ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE"); ap(" ALLN, 0.0")
ap("*BOUNDARY"); ap(" BOTTOM, 1, 2")
ap("*STEP, INC=200"); ap(" instantaneous growth (t << tau)"); ap("*VISCO, CETOL=1.0E-2")
ap(" 1.0E-4, 1.0E-3, 1.0E-9, 1.0E-3"); ap("*TEMPERATURE"); ap(" ALLN, 1.0")
ap("*OUTPUT, FIELD"); ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD"); ap("*END STEP")
ap("*STEP, INC=500"); ap(" hold, relax (t >> tau)"); ap("*VISCO, CETOL=1.0E-2")
ap(" 1.0, 1.0E5, 1.0E-4, 1.0E4")
ap("*OUTPUT, FIELD"); ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD"); ap("*END STEP")
open(OUT,"w").write("\n".join(L)+"\n")
print("wrote %s (VE thread, cond=%s, Prony gi=%s tau=%s)"%(OUT,cond,[round(g,3) for g in gi],[round(t,1) for t in ti]))
