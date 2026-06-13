"""Generate a UMAT-FREE 2-D plane-strain biofilm on an IMPLANT-THREAD substratum (eigenstrain deck).

The thesis growth column / tooth runs keep the substratum flat (or smoothly conformal). This deck puts
the SAME CLSM-calibrated depth growth on an actual threaded titanium cross-section, so the FEM itself
(not just the Hutchinson--Suo scaling argument) shows whether implant-thread geometry concentrates the
growth-induced interface stress.

No UMAT / no socket server: the anisotropic substratum-normal growth eigenstrain eps_zz(z) = max(0,
beta*phi_Pg(z)+ic) is applied as an ORTHOTROPIC thermal expansion (alpha_22 = eps_zz, alpha_11=alpha_33=0)
with a unit temperature rise, one material per depth layer. Linear elastic, one static step.

Geometry: a structured CPE4 mesh of the biofilm film of thickness Hf sitting on a sawtooth thread of
period P and amplitude A (A=0 -> the flat control). Bottom nodes bonded (u=0); lateral sides rollered
(u_x=0, laterally-infinite film); top free. Depth z is measured from the local thread surface, so the
growth law is identical for flat and threaded -> the contrast isolates the geometry effect.

Run:  python gen_implant_inp.py <cond DH|CH> <profile=thread|flat> <out.inp> [Nx Nz periods amp]
"""
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = json.load(open(HERE.parent / "beta_calibration.json"))

cond = sys.argv[1] if len(sys.argv) > 1 else "DH"
profile = sys.argv[2] if len(sys.argv) > 2 else "thread"
OUT = sys.argv[3] if len(sys.argv) > 3 else "implant_%s_%s.inp" % (cond, profile)
Nx = int(sys.argv[4]) if len(sys.argv) > 4 else 72
Nz = int(sys.argv[5]) if len(sys.argv) > 5 else 12
PERIODS = int(sys.argv[6]) if len(sys.argv) > 6 else 3
AMP = float(sys.argv[7]) if len(sys.argv) > 7 else (0.0 if profile == "flat" else 0.18)
# growth mode: "iso" = isotropic volume-matched (matches the thesis column fem_residual);
#              "aniso" = substratum-normal only (calibrate_beta's literal anisotropic law).
MODE = sys.argv[8] if len(sys.argv) > 8 else "iso"

E, NU = 5000.0, 0.45
W = float(PERIODS)               # width = PERIODS thread periods (period = 1.0)
HF = 1.0                          # biofilm thickness above the thread surface
# Pg-driven growth law (DH calibration, applied to BOTH conditions; see b4 / c5)
BETA, IC = CALIB["DH"]["beta_depth"], CALIB["DH"]["intercept"]

# measured endpoint phi_Pg(z) (pooled diffusion-fit profile; same source as the FEM stress)
sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "extensions"))
from b4_viscoelastic_growth_stress import depth_phi_pg  # noqa: E402
_zn, _pg = depth_phi_pg(cond)


def eps_g(zfrac):
    pg = float(np.interp(zfrac, _zn, _pg))
    return max(0.0, BETA * pg + IC)


def y_thread(x):
    """Smooth (sinusoidal) thread surface, period 1.0, peak-to-valley amplitude AMP.
    Smooth (C-infinity) -> finite root/crest curvature, no sharp-corner stress singularity,
    so the interface stress is mesh-convergent (cf. a triangular sawtooth, which is not)."""
    return AMP * 0.5 * (1.0 - np.cos(2.0 * np.pi * x))   # 0 at roots (x=int), AMP at crests


def nid(i, j):
    return j * (Nx + 1) + i + 1


L = []
ap = L.append
ap("*HEADING")
ap(" Implant-thread biofilm eigenstrain deck: cond=%s profile=%s amp=%.3f (CPE4, no UMAT)" %
   (cond, profile, AMP))

ap("*NODE")
xs = np.linspace(0.0, W, Nx + 1)
for j in range(Nz + 1):
    for i in range(Nx + 1):
        x = xs[i]
        ybot = y_thread(x)
        y = ybot + (j / Nz) * HF
        ap(" %d, %.6f, %.6f" % (nid(i, j), x, y))

ap("*ELEMENT, TYPE=CPE4")
for j in range(Nz):
    for i in range(Nx):
        e = j * Nx + i + 1
        ap(" %d, %d, %d, %d, %d" % (e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))

# one elset + material per depth layer (constant eigenstrain in that layer)
for j in range(Nz):
    els = [j * Nx + i + 1 for i in range(Nx)]
    ap("*ELSET, ELSET=ROW%d" % j)
    for k in range(0, len(els), 16):
        ap(" " + ",".join(str(x) for x in els[k:k + 16]))
ap("*ORIENTATION, NAME=ORI, SYSTEM=RECTANGULAR, DEFINITION=COORDINATES")
ap(" 1.0, 0.0, 0.0, 0.0, 1.0, 0.0")     # local 1 = global x, local 2 = global y (growth dir)
ap(" 3, 0.0")
for j in range(Nz):
    ap("*SOLID SECTION, ELSET=ROW%d, MATERIAL=MAT%d, ORIENTATION=ORI" % (j, j))
for j in range(Nz):
    zc = (j + 0.5) / Nz
    g = eps_g(zc)                       # volume growth measure max(0, beta*phi+ic)
    ap("*MATERIAL, NAME=MAT%d" % j)
    ap("*ELASTIC")
    ap(" %.1f, %.3f" % (E, NU))
    ap("*EXPANSION, TYPE=ORTHO, ZERO=0.0")
    if MODE == "iso":
        a = (1.0 + g) ** (1.0 / 3.0) - 1.0     # volume-matched isotropic linear strain (Jg = 1+g)
        ap(" %.6f, %.6f, %.6f" % (a, a, a))    # isotropic (matches thesis column UMAT mode=1)
    else:
        ap(" 0.0, %.6f, 0.0" % g)              # substratum-normal only (anisotropic)

# node sets for BCs
bottom = [nid(i, 0) for i in range(Nx + 1)]
left = [nid(0, j) for j in range(Nz + 1)]
right = [nid(Nx, j) for j in range(Nz + 1)]
alln = [nid(i, j) for j in range(Nz + 1) for i in range(Nx + 1)]
for name, ids in (("BOTTOM", bottom), ("LEFT", left), ("RIGHT", right), ("ALLN", alln)):
    ap("*NSET, NSET=%s" % name)
    for k in range(0, len(ids), 16):
        ap(" " + ",".join(str(x) for x in ids[k:k + 16]))

ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE")
ap(" ALLN, 0.0")
ap("*BOUNDARY")
ap(" BOTTOM, 1, 2")          # bonded to titanium (u=0); also anchors x -> no rigid-body mode
# Sides FREE: a finite biofilm patch bonded only at the (thread) base, so the bottom geometry --
# not a lateral-confinement BC -- governs the interface stress (implant-geometry sensitivity).
ap("*STEP")
ap(" Confined depth growth via thermal eigenstrain -> interface residual stress")
ap("*STATIC")
ap("*TEMPERATURE")
ap(" ALLN, 1.0")             # unit T rise -> eps = alpha
ap("*OUTPUT, FIELD")
ap("*NODE OUTPUT")
ap(" U")
ap("*ELEMENT OUTPUT, POSITION=CENTROID")
ap(" S, COORD")
ap("*END STEP")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote %s  (%dx%d CPE4, cond=%s, %s, amp=%.3f)" % (OUT, Nx, Nz, cond, profile, AMP))
print("eps_g(z) by layer:", [round(eps_g((j + 0.5) / Nz), 3) for j in range(Nz)])
