"""Viscoelastic implant-thread biofilm with NSP UMAT + Prony relaxation.

2-step analysis:
    Step 1 (STATIC, t=0..1): depth-graded growth builds the instantaneous elastic stress.
    Step 2 (VISCO, t=1..3601): holds geometry fixed; the two Maxwell arms (tau=6.6s, 903s)
            relax the deviatoric stress towards the long-time value g_inf * s_dev_elastic.

STATEV (19):
    [1]     growth ramp counter
    [2..7]  h1_dev (Prony arm 1)
    [8..13] h2_dev (Prony arm 2)
    [14..19] sigma_e_old (previous elastic Cauchy stress)

Uses umat_socket_ve.f (adds DTIME to protocol) + umat_server_ve.py (Prony integration).

Usage:
    python gen_implant_umat_ve_inp.py [cond DH|CH] [profile thread|flat] [out.inp] [Nx Nz amp]
Server: umat_server_ve.py --coord-axis 1 --height 1.0 --profile phipg_depth_DH.json
            --gamma 0.4 --nramp 20 --prony prony_biofilm.json
UMAT:   umat_socket_ve.f
"""
import sys, json
from pathlib import Path
import numpy as np

HERE   = Path(__file__).resolve().parent
CALIB  = json.load(open(HERE / "beta_calibration.json"))

cond    = sys.argv[1] if len(sys.argv) > 1 else "DH"
profile = sys.argv[2] if len(sys.argv) > 2 else "thread"
OUT     = sys.argv[3] if len(sys.argv) > 3 else "implant_umat_ve_%s_%s.inp" % (cond, profile)
Nx      = int(sys.argv[4])   if len(sys.argv) > 4 else 72
Nz      = int(sys.argv[5])   if len(sys.argv) > 5 else 12
PERIODS = int(sys.argv[6])   if len(sys.argv) > 6 else 3
AMP     = float(sys.argv[7]) if len(sys.argv) > 7 else (0.0 if profile == "flat" else 0.18)

E, NU  = 5000.0, 0.45
W      = float(PERIODS)
HF     = 1.0
DZ     = 0.1
NSTATV = 19    # 1 (ramp) + 6 (h1) + 6 (h2) + 6 (sigma_e_old)

def y_thread(x):
    return AMP * np.sin(np.pi * x) ** 2

def nid(i, j, k):
    return k * (Nx + 1) * (Nz + 1) + j * (Nx + 1) + i + 1

L = []; ap = L.append
ap("*HEADING")
ap(" Implant biofilm NSP UMAT + Prony VE: cond=%s profile=%s amp=%.3f  (umat_socket_ve.f)" % (
    cond, profile, AMP))

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

# --- NSP UMAT material (VE version, DEPVAR=19) ---
ap("*SOLID SECTION, ELSET=BIOFILM, MATERIAL=NSP_BIOFILM_VE")
ap("*MATERIAL, NAME=NSP_BIOFILM_VE")
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

# --- Step 1: growth (STATIC) ---
ap("*STEP, NLGEOM=YES, INC=200, NAME=GROWTH")
ap(" Step 1: NSP growth (instantaneous elastic stress)")
ap("*STATIC")
ap(" 0.05, 1.0, 1e-5, 0.05")
ap("*OUTPUT, FIELD, FREQUENCY=5")
ap("*NODE OUTPUT")
ap(" U, COORD")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

# --- Step 2: Prony relaxation (VISCO) ---
# tau_max = 903 s; run to 5*tau_max ~ 4500 s for ~99% relaxation of arm 2
ap("*STEP, NLGEOM=YES, INC=500, NAME=RELAX")
ap(" Step 2: Prony VE relaxation (tau1=6.6s, tau2=903s; t_end=4500s)")
ap("*VISCO")
ap(" 1.0, 4500.0, 0.01, 100.0")   # dt_init, t_end, dt_min, dt_max
ap("*OUTPUT, FIELD, FREQUENCY=10")
ap("*NODE OUTPUT")
ap(" U, COORD")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, SDV, COORD")
ap("*END STEP")

Path(OUT).write_text("\n".join(L) + "\n")
print("wrote %s  (%dx%d C3D8 VE, DEPVAR=%d, cond=%s, %s)" % (OUT, Nx, Nz, NSTATV, cond, profile))
print("UMAT:   umat_socket_ve.f  (DTIME protocol)")
print("server: umat_server_ve.py --coord-axis 1 --height 1.0 --profile phipg_depth_%s.json "
      "--gamma 0.4 --nramp 20 --prony prony_biofilm.json" % cond)
