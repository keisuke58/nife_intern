"""Extract S11(x,y) from implant_umat_*.odb and plot residual stress on thread geometry.

Usage: python extract_implant_umat.py implant_umat_DH_thread [--show]
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# --- parse args ---
ap = argparse.ArgumentParser()
ap.add_argument("jobname", nargs="?", default="implant_umat_DH_thread")
ap.add_argument("--show", action="store_true")
ap.add_argument("--out-json", default=None)
args = ap.parse_args()

JOB  = args.jobname
ODB  = Path(JOB + ".odb")

# --- try Abaqus odbAccess ---
try:
    import odbAccess
    odb  = odbAccess.openOdb(str(ODB))
    step = list(odb.steps.values())[-1]
    frame = list(step.frames)[-1]
    S    = frame.fieldOutputs["S"]
    COORD = frame.fieldOutputs["COORD"]
    SDV  = frame.fieldOutputs.get("SDV1", None)

    xs, ys, s11, s22, sdv1 = [], [], [], [], []
    for v, c in zip(S.values, COORD.values):
        xs.append(c.data[0]); ys.append(c.data[1])
        s11.append(v.data[0]); s22.append(v.data[1])
    if SDV:
        for v in SDV.values:
            sdv1.append(float(v.data) if hasattr(v.data, '__len__') else float(v.data))
    odb.close()

    xs   = np.array(xs);   ys  = np.array(ys)
    s11  = np.array(s11);  s22 = np.array(s22)
    data = {"x": xs.tolist(), "y": ys.tolist(),
            "S11": s11.tolist(), "S22": s22.tolist(),
            "SDV1": sdv1 if sdv1 else []}
    out_json = args.out_json or JOB + "_field.json"
    Path(out_json).write_text(json.dumps(data, indent=2))
    print(f"extracted {len(xs)} Gauss points  ->  {out_json}")
    print(f"S11 range: {s11.min():.1f} .. {s11.max():.1f}")
    print(f"S22 range: {s22.min():.1f} .. {s22.max():.1f}")

except Exception as e:
    print(f"odbAccess failed ({e}), trying JSON fallback ...")
    json_file = JOB + "_field.json"
    if not Path(json_file).exists():
        sys.exit(f"no {json_file} found either — run Abaqus first")
    data = json.loads(Path(json_file).read_text())
    xs   = np.array(data["x"]); ys  = np.array(data["y"])
    s11  = np.array(data["S11"])
    print(f"loaded {len(xs)} pts from {json_file}")

# --- plot ---
import matplotlib
matplotlib.use("Agg" if not args.show else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, vals, label, cmap in zip(
        axes, [s11, np.array(data.get("S22", [0]*len(xs)))],
        ["$S_{11}$ (in-plane residual stress)", "$S_{22}$ (vertical stress)"],
        ["RdBu_r", "PuOr_r"]):
    triang = mtri.Triangulation(xs, ys)
    lim = max(abs(vals.min()), abs(vals.max()))
    tc = ax.tricontourf(triang, vals, levels=20, cmap=cmap, vmin=-lim, vmax=lim)
    plt.colorbar(tc, ax=ax, label="[MPa/$\\mu$]")
    ax.set_xlabel("x (thread width)")
    ax.set_ylabel("y (depth)")
    ax.set_title(label)
    ax.set_aspect("equal")

fig.suptitle(f"NSP UMAT residual stress — {JOB}", fontsize=10)
plt.tight_layout()
out_fig = JOB + "_stress.png"
plt.savefig(out_fig, dpi=150)
print(f"saved  {out_fig}")
if args.show:
    plt.show()
