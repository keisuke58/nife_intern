"""Generate mms_N<Nx>.inp — A3 Method of Manufactured Solutions (observed order of accuracy).

Manufactured exact solution (uniaxial strain, growth OFF):
    u_x(X) = alpha * sin(pi X / L),   u_y = u_z = 0.
For the (small-strain limit of the) neo-Hookean operator with u_y=u_z=0 everywhere the equilibrium is
1-D: sigma_xx = (lambda+2mu) u_x', so the manufactured body force is
    B_x(X) = -div sigma = (lambda+2mu) * alpha * (pi/L)^2 * sin(pi X / L),
applied per element column via *DLOAD (BX). Both ends are pinned (sin(0)=sin(pi)=0), so the interior
bump is driven purely by the body force. Refining Nx and measuring ||u_h - u_exact||_L2 gives the
observed order p; for trilinear C3D8 the L2 displacement error decays as h^2 (p=2).

alpha is kept small (1e-3) so the neo-Hookean nonlinearity (O(alpha^2)) stays below the discretization
error over the meshes used, i.e. the manufactured linear body force is the correct forcing to O(alpha^2).

Run:  python gen_mms_inp.py <Nx> mms_N<Nx>.inp
"""
import sys, math
Nx  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = sys.argv[2] if len(sys.argv) > 2 else ("mms_N%d.inp" % Nx)
L, ALPHA = 1.0, 1.0e-4
E, nu = 5000.0, 0.45
mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
c = (lam + 2 * mu) * ALPHA * (math.pi / L) ** 2          # body-force amplitude
dx = L / Nx

def nid(i, j, k):                                        # i:0..Nx (x), j,k:0..1
    return 1 + i + (Nx + 1) * (j + 2 * k)

lines = ["*HEADING", " A3 MMS: u_x=alpha sin(pi X/L), body-force-driven, C3D8 (order of accuracy)",
         "*NODE"]
for k in range(2):
    for j in range(2):
        for i in range(Nx + 1):
            lines.append(" %d, %.8f, %.6f, %.6f" % (nid(i, j, k), i * dx, j * 1.0, k * 1.0))
lines.append("*ELEMENT, TYPE=C3D8, ELSET=ALL")
for i in range(Nx):                                      # element e=i+1 spans [i,i+1] in x
    n = [nid(i,0,0), nid(i+1,0,0), nid(i+1,1,0), nid(i,1,0),
         nid(i,0,1), nid(i+1,0,1), nid(i+1,1,1), nid(i,1,1)]
    lines.append(" %d, %s" % (i + 1, ",".join(str(x) for x in n)))
lines += ["*SOLID SECTION, ELSET=ALL, MATERIAL=BIOFILM",
          "*MATERIAL, NAME=BIOFILM", "*USER MATERIAL, CONSTANTS=2", " 5000., 0.45", "*DEPVAR", " 1"]
# per-element body force at the column centroid
for i in range(Nx):
    lines.append("*ELSET, ELSET=E%d" % (i + 1))
    lines.append(" %d" % (i + 1))
# node sets
alln = [nid(i, j, k) for k in range(2) for j in range(2) for i in range(Nx + 1)]
xend = [nid(0, j, k) for k in range(2) for j in range(2)] + [nid(Nx, j, k) for k in range(2) for j in range(2)]
def nset(name, ids):
    lines.append("*NSET, NSET=%s" % name)
    for s in range(0, len(ids), 12):
        lines.append(" " + ",".join(str(x) for x in ids[s:s+12]))
nset("ALLN", alln); nset("XEND", xend)
lines += ["*BOUNDARY", " ALLN, 2, 3", " XEND, 1, 1",     # u_y=u_z=0 everywhere; u_x=0 at both ends
          "*STEP, NLGEOM=YES, INC=50", " body-force-driven manufactured solution",
          "*STATIC", " 0.1, 1.0, 1.0E-4, 0.1", "*DLOAD"]
for i in range(Nx):
    Xc = (i + 0.5) * dx
    lines.append(" E%d, BX, %.10e" % (i + 1, c * math.sin(math.pi * Xc / L)))
lines += ["*OUTPUT, FIELD", "*NODE OUTPUT", " U", "*ELEMENT OUTPUT, POSITION=CENTROID", " S", "*END STEP"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (Nx=%d, alpha=%g, body-force amp c=%.4g)" % (OUT, Nx, ALPHA, c))
