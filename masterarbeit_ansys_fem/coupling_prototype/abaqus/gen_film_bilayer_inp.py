"""Generate film_bilayer.inp — A1: growth-induced WRINKLING of a stiff biofilm crust on a soft,
inert foundation (a genuine Biot/bilayer bifurcation, upgrading the B1 free-edge precursor).

Two materials: the top `NCRUST` element layers are a STIFF growing crust (the user-material growth
UMAT, E_crust); the rest is a SOFT, NON-growing foundation (`*ELASTIC`, E_base). The crust grows
isotropically in-plane but is bonded to the foundation that does not grow, so it is driven into
in-plane compression — the classical configuration for growth-induced surface wrinkling. A small
multi-wave imperfection is seeded on the top surface; below a critical growth the surface stays flat
(deflection ~ imperfection), above it a wrinkle of a selected wavelength grows super-linearly. The
dysbiotic crust (larger phi_Pg -> more growth -> more compression) wrinkles at a lower growth than the
commensal one.

Only the crust calls the growth UMAT, so the foundation is inert; growth magnitude is set by the
server (uniform phi profile, gamma). BCs: x=0 symmetry, plane strain, bottom fully fixed (bonded
substratum), free top + free right edge.

Run:  python gen_film_bilayer_inp.py 48 8 film_bilayer.inp [ncrust] [Ecrust] [Ebase] [m] [amp]
"""
import sys, math
Nx     = int(sys.argv[1]) if len(sys.argv) > 1 else 48
Nz     = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT    = sys.argv[3] if len(sys.argv) > 3 else "film_bilayer.inp"
NCRUST = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ECR    = float(sys.argv[5]) if len(sys.argv) > 5 else 50000.0
EBA    = float(sys.argv[6]) if len(sys.argv) > 6 else 2000.0
MWAVE  = int(sys.argv[7]) if len(sys.argv) > 7 else 6
AMP    = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0e-3
STAB   = float(sys.argv[9]) if len(sys.argv) > 9 else 2.0e-3
L, T, H = 8.0, 1.0, 1.0
dx, dz = L / Nx, H / Nz
kc = Nz - NCRUST                                     # first crust layer index

def nid(i, j, k):
    return 1 + i + (Nx + 1) * (j + 2 * k)

lines = ["*HEADING", " Biofilm bilayer: stiff growing crust on soft inert base -> wrinkling (A1)",
         "*NODE"]
for k in range(Nz + 1):
    for j in range(2):
        for i in range(Nx + 1):
            x, z = i * dx, k * dz
            zp = z + (AMP * math.sin(MWAVE * math.pi * x / L) if k == Nz else 0.0)   # top-surface seed
            lines.append(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), x, j * T, zp))
lines.append("*ELEMENT, TYPE=C3D8, ELSET=ALLEL")
e = 0; crust = []; base = []
for k in range(Nz):
    for i in range(Nx):
        e += 1
        n = [nid(i,0,k), nid(i+1,0,k), nid(i+1,1,k), nid(i,1,k),
             nid(i,0,k+1), nid(i+1,0,k+1), nid(i+1,1,k+1), nid(i,1,k+1)]
        lines.append(" %d, %s" % (e, ",".join(str(x) for x in n)))
        (crust if k >= kc else base).append(e)
def elset(name, ids):
    lines.append("*ELSET, ELSET=%s" % name)
    for s in range(0, len(ids), 12):
        lines.append(" " + ",".join(str(x) for x in ids[s:s+12]))
elset("CRUST", crust); elset("BASE", base)
lines += ["*SOLID SECTION, ELSET=CRUST, MATERIAL=CRUSTMAT",
          "*MATERIAL, NAME=CRUSTMAT", "*USER MATERIAL, CONSTANTS=2", " %g, 0.45" % ECR, "*DEPVAR", " 1",
          "*SOLID SECTION, ELSET=BASE, MATERIAL=BASEMAT",
          "*MATERIAL, NAME=BASEMAT", "*ELASTIC", " %g, 0.45" % EBA]
xsym = [nid(0, j, k) for k in range(Nz + 1) for j in range(2)]
ysym = [nid(i, j, k) for k in range(Nz + 1) for j in range(2) for i in range(Nx + 1)]
zbot = [nid(i, j, 0) for j in range(2) for i in range(Nx + 1)]
def nset(name, ids):
    lines.append("*NSET, NSET=%s" % name)
    for s in range(0, len(ids), 12):
        lines.append(" " + ",".join(str(x) for x in ids[s:s+12]))
nset("XSYM", xsym); nset("YSYM", ysym); nset("ZBOT", zbot)
lines += ["*BOUNDARY", " XSYM, 1, 1", " YSYM, 2, 2", " ZBOT, 1, 3",
          "*STEP, NLGEOM=YES, INC=1000", " differential growth of crust -> wrinkling",
          "*STATIC, STABILIZE=%g, ALLSDTOL=0.05, CONTINUE=NO" % STAB, " 0.01, 1.0, 1.0E-8, 0.02",
          "*OUTPUT, FIELD, FREQUENCY=1", "*NODE OUTPUT", " U",
          "*ELEMENT OUTPUT, POSITION=CENTROID", " S", "*END STEP"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s (%d crust + %d base elem, crust E=%g base E=%g, seed m=%d amp=%g)"
      % (OUT, len(crust), len(base), ECR, EBA, MWAVE, AMP))
