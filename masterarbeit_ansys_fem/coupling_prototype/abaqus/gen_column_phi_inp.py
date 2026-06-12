"""Generate a SOCKET-FREE depth-graded biofilm growth column for umat_growth_phi.f.

A 1x1xN stack of C3D8 (height 1.0, z = depth), laterally confined (u_x=0 on x-faces, u_y=0 on
y-faces, u_z=0 at z=0, top free) = a representative column of a substrate-bonded laterally-infinite
film. The composition field phi_Pg(z) is read from a depth-profile JSON (phipg_depth_{DH,CH}.json) and
written as a **predefined FIELD variable** node-by-node (interpolated to each layer's z). Isotropic
growth Fg = s*I, s = 1 + gamma*phi_Pg (mode=1), confined in-plane -> in-plane residual stress
sigma_xx(z) tracking phi_Pg(z). No Python server: the UMAT reads PREDEF(1) per Gauss point.

Run:  python gen_column_phi_inp.py phipg_depth_DH.json column_phi_DH.inp 8 [gamma]
"""
import sys, json
import numpy as np

prof_file = sys.argv[1] if len(sys.argv) > 1 else "phipg_depth_DH.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "column_phi_DH.inp"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CALIB = sys.argv[4] if len(sys.argv) > 4 else "beta_calibration.json"  # CLSM beta_depth, intercept
ELTYPE = sys.argv[5] if len(sys.argv) > 5 else "C3D8"  # C3D8H = hybrid (near-incompressible biofilm)
GSCALE = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0  # growth-magnitude scale (1=full CLSM)
HEIGHT = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0  # column height / width (>=1; tall = clean interior)
E, NU = 5000.0, 0.45

prof = json.load(open(prof_file))
cond = prof.get("cond", "DH")
# CLSM-calibrated growth: Jg = 1 + max(0, beta*phi + ic); iso volume-matched s = Jg^(1/3) (in UMAT).
# GSCALE<1 scales the eigenstrain down (moderate-growth regime) for the mesh-convergence study,
# isolating discretization error from the large-strain/load-interpolation sensitivity at full CLSM.
cal = json.load(open(CALIB))[cond]
BETA, IC = GSCALE * cal["beta_depth"], GSCALE * cal["intercept"]
zc, pc = np.array(prof["z_norm"], float), np.array(prof["phi_Pg"], float)
# Column is width 1 x 1, height HEIGHT. A tall column (HEIGHT>>1) keeps the free-surface Saint-Venant
# relaxation zone (~1 width deep) confined to the top, leaving a clean residual-stress interior.
H = HEIGHT
dz = H / N


def nid(layer, c):
    return layer * 1000 + c          # 1000 to allow N up to ~999 layers without id collision


def phi_at(z):
    return float(np.interp(z / H, zc, pc))   # profile z_norm 0..1 spans the full film height 0..H


L = []
ap = L.append
ap("*HEADING")
ap(" Socket-free depth-graded growth column (%d C3D8), cond=%s, CLSM beta=%.4g ic=%.4g" % (N, cond, BETA, IC))
ap("** Laterally confined; phi_Pg(z) as FIELD var 1; iso volume-matched growth -> sigma_xx(z).")
ap("*NODE")
for layer in range(N + 1):
    z = layer * dz
    ap(" %d, 0.0, 0.0, %.6f" % (nid(layer, 1), z))
    ap(" %d, 1.0, 0.0, %.6f" % (nid(layer, 2), z))
    ap(" %d, 1.0, 1.0, %.6f" % (nid(layer, 3), z))
    ap(" %d, 0.0, 1.0, %.6f" % (nid(layer, 4), z))
ap("*ELEMENT, TYPE=%s, ELSET=COL" % ELTYPE)
for e in range(N):
    b = [nid(e, 1), nid(e, 2), nid(e, 3), nid(e, 4),
         nid(e + 1, 1), nid(e + 1, 2), nid(e + 1, 3), nid(e + 1, 4)]
    ap(" %d, %s" % (e + 1, ",".join(str(x) for x in b)))
ap("*SOLID SECTION, ELSET=COL, MATERIAL=BIOFILM")
ap("*MATERIAL, NAME=BIOFILM")
ap("*USER MATERIAL, CONSTANTS=5")
ap("** E, nu, beta_depth(CLSM), intercept(CLSM), mode(1=iso volume-matched)")
ap(" %.1f, %.3f, %.6f, %.6f, 1.0" % (E, NU, BETA, IC))
ap("*DEPVAR")
ap(" 3")
alln = [nid(l, c) for l in range(N + 1) for c in (1, 2, 3, 4)]
ap("*NSET, NSET=ALLN")
for i in range(0, len(alln), 16):
    ap(" " + ",".join(str(x) for x in alln[i:i + 16]))
ap("*NSET, NSET=XF")
ap(" " + ",".join(str(nid(l, c)) for l in range(N + 1) for c in (1, 4)))
ap("*NSET, NSET=XR")
ap(" " + ",".join(str(nid(l, c)) for l in range(N + 1) for c in (2, 3)))
ap("*NSET, NSET=YF")
ap(" " + ",".join(str(nid(l, c)) for l in range(N + 1) for c in (1, 2)))
ap("*NSET, NSET=YR")
ap(" " + ",".join(str(nid(l, c)) for l in range(N + 1) for c in (3, 4)))
ap("*NSET, NSET=ZB")
ap(" " + ",".join(str(nid(0, c)) for c in (1, 2, 3, 4)))
# phi_Pg = 0 initially everywhere
ap("*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=1")
ap(" ALLN, 0.0")
ap("*BOUNDARY")
ap(" XF, 1, 1")
ap(" XR, 1, 1")
ap(" YF, 2, 2")
ap(" YR, 2, 2")
ap(" ZB, 3, 3")
ap("*STEP, NLGEOM=YES, INC=200")
ap(" Ramp phi_Pg(z) 0 -> profile; isotropic confined growth -> residual sigma_xx(z)")
ap("*STATIC")
ap(" 0.02, 1.0, 1.0E-5, 0.05")
# final phi_Pg(z) per node, ramped from 0 by the step
ap("*FIELD, VARIABLE=1")
for layer in range(N + 1):
    z = layer * dz
    p = phi_at(z)
    for c in (1, 2, 3, 4):
        ap(" %d, %.6f" % (nid(layer, c), p))
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, E, SDV, COORD")
ap("*END STEP")

open(OUT, "w").write("\n".join(L) + "\n")
Jg = lambda p: 1.0 + max(0.0, BETA * p + IC)
print("wrote %s (%d elem, cond=%s, CLSM beta=%.4g ic=%.4g)" % (OUT, N, cond, BETA, IC))
print("phi_Pg(z) centroids:", [round(phi_at((e + 0.5) * dz), 3) for e in range(N)])
print("Jg=lam_z centroids :", [round(Jg(phi_at((e + 0.5) * dz)), 3) for e in range(N)])
print("s=Jg^1/3 centroids :", [round(Jg(phi_at((e + 0.5) * dz)) ** (1 / 3.), 3) for e in range(N)])
