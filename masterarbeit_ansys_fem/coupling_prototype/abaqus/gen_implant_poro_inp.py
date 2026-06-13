"""Poroelastic (Terzaghi consolidation) relaxation of the implant-thread interface stress.
CPE4P (coupled pore-pressure/displacement) thread; growth (thermal eigenstrain) pressurises the
saturated biofilm matrix, which drains (top + sides pw=0; bonded base sealed) -> the effective
interface stress relaxes over the consolidation time. Companion to the column Terzaghi result.
Run: python gen_implant_poro_inp.py <cond> <out.inp> [Nx Nz periods amp]"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))
cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
OUT = sys.argv[2] if len(sys.argv) > 2 else "poro_%s.inp" % cond
Nx = int(sys.argv[3]) if len(sys.argv) > 3 else 72
Nz = int(sys.argv[4]) if len(sys.argv) > 4 else 12
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
L=[]; ap=L.append
ap("*HEADING"); ap(" Poroelastic implant-thread (Terzaghi), cond=%s"%cond)
ap("*NODE"); xs=np.linspace(0,float(PER),Nx+1)
for j in range(Nz+1):
  for i in range(Nx+1): ap(" %d, %.6f, %.6f"%(nid(i,j), xs[i], yth(xs[i])+(j/Nz)*1.0))
ap("*ELEMENT, TYPE=CPE4P, ELSET=FILM")
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
  ap("*PERMEABILITY, SPECIFIC=10.0"); ap(" 1.0E-4, 1.0")    # k at void ratio e=1.0
alln=[nid(i,j) for j in range(Nz+1) for i in range(Nx+1)]
bottom=[nid(i,0) for i in range(Nx+1)]
top=[nid(i,Nz) for i in range(Nx+1)]; left=[nid(0,j) for j in range(Nz+1)]; right=[nid(Nx,j) for j in range(Nz+1)]
drained=sorted(set(top+left+right))
for nm,ids in (("ALLN",alln),("BOTTOM",bottom),("DRAIN",drained)):
  ap("*NSET, NSET=%s"%nm)
  for k in range(0,len(ids),16): ap(" "+",".join(str(x) for x in ids[k:k+16]))
ap("*INITIAL CONDITIONS, TYPE=RATIO"); ap(" ALLN, 1.0")            # void ratio e0=1 (NODAL)
ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE"); ap(" ALLN, 0.0")
ap("*BOUNDARY"); ap(" BOTTOM, 1, 2"); ap(" DRAIN, 8, 8, 0.0")     # drained surfaces pw=0
ap("*STEP, INC=200"); ap(" growth pressurisation (fast)"); ap("*SOILS, CONSOLIDATION, UTOL=1.0E8")
ap(" 1.0E-3, 1.0E-3, 1.0E-8, 1.0E-3"); ap("*TEMPERATURE"); ap(" ALLN, 1.0")
ap("*OUTPUT, FIELD"); ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD"); ap("*NODE OUTPUT"); ap(" POR"); ap("*END STEP")
ap("*STEP, INC=500"); ap(" drainage / consolidation"); ap("*SOILS, CONSOLIDATION, UTOL=1.0E8")
ap(" 0.1, 1.0E4, 1.0E-5, 1.0E3")
ap("*OUTPUT, FIELD"); ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD"); ap("*NODE OUTPUT"); ap(" POR"); ap("*END STEP")
open(OUT,"w").write("\n".join(L)+"\n")
print("wrote %s (CPE4P poro thread, cond=%s)"%(OUT,cond))
