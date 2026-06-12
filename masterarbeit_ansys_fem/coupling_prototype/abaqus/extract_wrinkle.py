"""A1 wrinkling: top-surface deflection amplitude vs growth + selected wavelength.
Usage: abaqus python extract_wrinkle.py <job> [Nx] [Nz] [L]   (defaults film_bilayer 48 8 8.0)
Writes <job>_wrinkle.txt (growthfrac  amplitude). Amplitude = half peak-to-peak of top-surface U3
along x; wavelength from the dominant FFT mode of the final top-surface profile."""
from __future__ import annotations
import sys, math
from odbAccess import openOdb

job = sys.argv[1] if len(sys.argv) > 1 else "film_bilayer"
Nx  = int(sys.argv[2]) if len(sys.argv) > 2 else 48
Nz  = int(sys.argv[3]) if len(sys.argv) > 3 else 8
L   = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
def nid(i, j, k):
    return 1 + i + (Nx + 1) * (j + 2 * k)
top = [nid(i, 0, Nz) for i in range(Nx + 1)]                 # top-surface row (j=0)
o = openOdb(job + ".odb")
step = list(o.steps.values())[-1]
rows = []
for fr in step.frames:
    U = {v.nodeLabel: v.data[2] for v in fr.fieldOutputs['U'].values}
    prof = [U.get(n, 0.0) for n in top]
    amp = 0.5 * (max(prof) - min(prof))
    rows.append((fr.frameValue, amp, prof))
with open(job + "_wrinkle.txt", "w") as f:
    for t, a, _ in rows:
        f.write("%.4f  %.6e\n" % (t, a))
# dominant wavelength from the final profile (discrete sine transform, count interior modes)
prof = rows[-1][2]
n = len(prof); best_m, best_p = 0, 0.0
for m in range(1, n // 2):
    c = sum(prof[i] * math.sin(m * math.pi * i / (n - 1)) for i in range(n))
    if abs(c) > best_p:
        best_p, best_m = abs(c), m
lam = (2.0 * L / best_m) if best_m else float('nan')
print("growth-frac  top-surface amplitude")
for t, a, _ in rows:
    print("  %.3f         %.5f" % (t, a))
print("FINAL amplitude = %.5f   dominant mode m=%d  ->  wavelength lambda = %.3f"
      % (rows[-1][1], best_m, lam))
o.close()
