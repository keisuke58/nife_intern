"""Implant-thread biofilm with TRUE gLV UMAT (coupled bio-mechanics at each Gauss point).

STATEV(5) = [x₁..x₅] absolute species abundances, evolved per-GP by gLV ODE.
Same C3D8 thin-slice geometry as gen_implant_umat_inp.py.
Uses umat_socket_ve.f (DTIME-aware) + umat_server_glv.py.

Physics: gLV ODE drives φ_Pg(t) at each Gauss point independently.
         Growth: F=Fᵉ·Fᵍ, Fᵍ = (1+γ·φ_Pg)^(1/3) I (isotropic volumetric).
         Mechanics: neo-Hookean hyperelastic, E=5000 Pa, ν=0.45.

Difference vs Hamilton UMAT:
    Hamilton:  STATEV=12 (φᵢ,ψᵢ,γ), variational Hamilton dynamics, calibrated φ_Pg(z) profile
    gLV:       STATEV=5  (xᵢ),       true gLV ODE dxᵢ/dt=xᵢ(bᵢ+Σ Aᵢⱼxⱼ), spatially uniform IC

Usage:
    python gen_implant_umat_glv_inp.py [cond DH|CH] [profile thread|flat] [out.inp] [Nx Nz amp]
Server: umat_server_glv.py --cond DH --bio-days 21 --gamma 0.4 --coord-axis 1 --height 1.0
UMAT:   umat_socket_ve.f  (sends DTIME to server)
"""
import sys, json
from pathlib import Path
import numpy as np

HERE   = Path(__file__).resolve().parent
CALIB  = json.load(open(HERE / "beta_calibration.json"))

cond    = sys.argv[1] if len(sys.argv) > 1 else "DH"
profile = sys.argv[2] if len(sys.argv) > 2 else "thread"
OUT     = sys.argv[3] if len(sys.argv) > 3 else "implant_umat_glv_%s_%s.inp" % (cond, profile)
Nx      = int(sys.argv[4])   if len(sys.argv) > 4 else 72
Nz      = int(sys.argv[5])   if len(sys.argv) > 5 else 12
PERIODS = int(sys.argv[6])   if len(sys.argv) > 6 else 3
AMP     = float(sys.argv[7]) if len(sys.argv) > 7 else (0.0 if profile == "flat" else 0.18)

E, NU   = 5000.0, 0.45
W       = float(PERIODS)
HF      = 1.0
DZ      = 0.1
NSTATV  = 5    # [x1..x5] species abundances

def y_thread(x):
    return AMP * np.sin(np.pi * x) ** 2

def nid(i, j, k):
    return k * (Nx + 1) * (Nz + 1) + j * (Nx + 1) + i + 1

L = []; ap = L.append
ap("*HEADING")
ap(" Implant biofilm TRUE gLV UMAT: cond=%s profile=%s amp=%.3f  (umat_socket_ve.f + umat_server_glv.py)" % (
    cond, profile, AMP))
ap("** gLV: dxi/dt = xi*(bi + Aij*xj)  [NOT replicator]  DEPVAR=5")

# --- nodes ---
ap("*NODE")
xs = np.linspace(0.0, W, Nx + 1)
for k in range(2):
    zv = k * DZ
    for j in range(Nz + 1):
        for i in range(Nx + 1):
            x    = xs[i]
            ybot = y_thread(x)
            y    = ybot + (j / Nz) * HF
            ap(" %d, %.6f, %.6f, %.6f" % (nid(i, j, k), x, y, zv))

# --- C3D8 elements ---
ap("*ELEMENT, TYPE=C3D8, ELSET=BIOFILM")
for j in range(Nz):
    for i in range(Nx):
        e = j * Nx + i + 1
        n = [nid(i,   j,   0), nid(i+1, j,   0),
             nid(i+1, j+1, 0), nid(i,   j+1, 0),
             nid(i,   j,   1), nid(i+1, j,   1),
             nid(i+1, j+1, 1), nid(i,   j+1, 1)]
        ap(" %d, %s" % (e, ", ".join(str(x) for x in n)))

# --- gLV UMAT material: CONSTANTS=5 (E, nu, beta, intercept, HF) ---
ap("*SOLID SECTION, ELSET=BIOFILM, MATERIAL=GLV_BIOFILM")
ap("*MATERIAL, NAME=GLV_BIOFILM")
ap("*USER MATERIAL, CONSTANTS=5")
ap(" %.1f, %.3f, %.4f, %.6f, %.4f" % (
    E, NU, CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"], HF))
ap("*DEPVAR")
ap(" %d" % NSTATV)

# --- node sets ---
bottom = [nid(i, 0, k) for k in range(2) for i in range(Nx + 1)]
front  = [nid(i, j, 0) for j in range(Nz + 1) for i in range(Nx + 1)]
back   = [nid(i, j, 1) for j in range(Nz + 1) for i in range(Nx + 1)]
for name, ids in (("BOTTOM", bottom), ("FRONT", front), ("BACK", back)):
    ap("*NSET, NSET=%s" % name)
    for kk in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[kk:kk+16]))

ap("*BOUNDARY")
ap(" BOTTOM, 1, 3")
ap(" FRONT,  3, 3")
ap(" BACK,   3, 3")

# --- step (STATIC; gLV time mapped to biological days via --bio-days in server) ---
ap("*STEP, NLGEOM=YES, INC=200")
ap(" TRUE gLV UMAT: gLV ODE per Gauss point, phi_Pg drives neo-Hookean growth")
ap("*STATIC")
ap(" 0.05, 1.0, 1e-5, 0.05")
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U, COORD")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

Path(OUT).write_text("\n".join(L) + "\n")
print("wrote %s  (%dx%d C3D8 gLV-UMAT, DEPVAR=%d, cond=%s, %s)" % (
    OUT, Nx, Nz, NSTATV, cond, profile))
print("UMAT:   umat_socket_ve.f  (DTIME protocol, same as VE variant)")
print("server: umat_server_glv.py --cond %s --bio-days 21 --gamma 0.4 "
      "--coord-axis 1 --height 1.0 --port 50007" % cond)
