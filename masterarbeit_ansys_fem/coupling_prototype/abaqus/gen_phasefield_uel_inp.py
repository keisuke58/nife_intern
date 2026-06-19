"""Generates a single-edge-notched tension (SENT) Abaqus deck driving the AT2 phase-field UEL.

Geometry (plane-strain, mm; properties unit-consistent with the UEL):
    rectangle L_x × L_y, NX × NY bilinear quads, half-height notch of length L_x/2 on the left edge
    midplane (y = L_y/2 ± h, x ∈ [0, L_x/2]) imposed via INITIAL CONDITIONS, TYPE=FIELD on the
    phase-field DOF = 1 along the notch (cracked at start).

DOFs per node:
    1 = ux (Abaqus DOF 1)
    2 = uy (Abaqus DOF 2)
    11 = d  (Abaqus generalised DOF for the user element)

Loading:
    *BOUNDARY     fix uy at y=0, fix ux at one bottom-left corner.
    *AMPLITUDE    quasi-static ramp 0 → u_top over NSTEP increments.
    *BOUNDARY     uy at y=L_y prescribed as u_top * amplitude.

Output:
    .inp file ready for   `abaqus job=<name> user=uel_phasefield_at2.f cpus=1 interactive`.

Usage:
    python gen_phasefield_uel_inp.py phasefield_sent 32 64 0.05 0.10 1.0 1.0 0.001 0.04
                                       name           NX NY Lx Ly E_mod  Gc l   u_top
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def gen_mesh(nx: int, ny: int, lx: float, ly: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    nid = lambda i, j: j * (nx + 1) + i + 1     # 1-based
    elems = []
    for j in range(ny):
        for i in range(nx):
            elems.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)])
    return nodes, np.asarray(elems, dtype=np.int32)


def notched_nodes(nodes: np.ndarray, lx: float, ly: float, half_band: float) -> list[int]:
    out = []
    y_mid = 0.5 * ly
    for k, (x, y) in enumerate(nodes, start=1):
        if x <= 0.5 * lx + 1e-9 and abs(y - y_mid) <= half_band + 1e-9:
            out.append(k)
    return out


def write_inp(path: Path, name: str, nodes: np.ndarray, elems: np.ndarray,
              lx: float, ly: float, e_mod: float, nu: float, gc: float, ell: float,
              u_top: float, notch_band: float, nstep: int) -> None:
    out = path / f"{name}.inp"
    with out.open("w") as f:
        f.write("*HEADING\n")
        f.write(f"Phase-field SENT, AT2, UEL=U1 (NDOFEL=12)  ({nx}×{ny} mesh)\n")

        f.write("*NODE, NSET=NALL\n")
        for k, (x, y) in enumerate(nodes, start=1):
            f.write(f"{k}, {x:.6f}, {y:.6f}\n")

        f.write("*USER ELEMENT, NODES=4, TYPE=U1, COORDINATES=2, "
                "PROPERTIES=4, VARIABLES=8\n")
        f.write("1, 2, 11\n")              # active DOFs per node: ux=1, uy=2, d=11

        f.write("*ELEMENT, TYPE=U1, ELSET=EL_UEL\n")
        for k, e in enumerate(elems, start=1):
            f.write(f"{k}, {e[0]}, {e[1]}, {e[2]}, {e[3]}\n")

        f.write("*UEL PROPERTY, ELSET=EL_UEL\n")
        f.write(f"{e_mod:.6e}, {nu:.6e}, {gc:.6e}, {ell:.6e}\n")

        # initial crack via field d=1 on the notched node set
        ncrack = notched_nodes(nodes, lx, ly, notch_band)
        f.write("*NSET, NSET=NCRACK\n")
        for chunk in (ncrack[i:i+8] for i in range(0, len(ncrack), 8)):
            f.write(", ".join(map(str, chunk)) + "\n")
        f.write("*INITIAL CONDITIONS, TYPE=SOLUTION\n")
        f.write("NCRACK, 0.0, 0.0, 1.0\n")   # initial (ux, uy, d) = (0, 0, 1)

        # boundary node sets
        nbot = [k for k, (x, y) in enumerate(nodes, start=1) if abs(y) <= 1e-9]
        ntop = [k for k, (x, y) in enumerate(nodes, start=1) if abs(y - ly) <= 1e-9]
        f.write("*NSET, NSET=NBOT\n")
        for chunk in (nbot[i:i+8] for i in range(0, len(nbot), 8)):
            f.write(", ".join(map(str, chunk)) + "\n")
        f.write("*NSET, NSET=NTOP\n")
        for chunk in (ntop[i:i+8] for i in range(0, len(ntop), 8)):
            f.write(", ".join(map(str, chunk)) + "\n")

        # quasi-static incremental ramp 0 → u_top
        f.write("*BOUNDARY\n")
        f.write("NBOT, 2, 2, 0.0\n")            # fix uy at y=0
        f.write(f"{nbot[0]}, 1, 1, 0.0\n")      # fix ux at one corner (anti-rigid-body)

        f.write(f"*STEP, NLGEOM=NO, NAME=PULL, INC={nstep*3}\n")
        f.write(f"*STATIC\n{1.0/nstep:.6f}, 1.0, {1.0/(nstep*100):.6e}, {1.0/nstep:.6f}\n")
        f.write("*BOUNDARY\n")
        f.write(f"NTOP, 2, 2, {u_top:.6e}\n")

        f.write("*OUTPUT, FIELD, FREQUENCY=1\n")
        # SDV via *NODE FILE not standard for UEL fields; use *EL FILE / *EL PRINT for SVARS=H.
        f.write("*EL PRINT, ELSET=EL_UEL, FREQUENCY=10\n")
        f.write("SDV\n")
        f.write("*NODE PRINT, NSET=NTOP, FREQUENCY=10\n")
        f.write("U, RF\n")
        f.write("*END STEP\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", help="job name without .inp")
    p.add_argument("nx", type=int)
    p.add_argument("ny", type=int)
    p.add_argument("lx", type=float)
    p.add_argument("ly", type=float)
    p.add_argument("e_mod", type=float)
    p.add_argument("gc", type=float)
    p.add_argument("ell", type=float)
    p.add_argument("u_top", type=float)
    p.add_argument("--nu", type=float, default=0.30)
    p.add_argument("--notch-band", type=float, default=0.02)
    p.add_argument("--nstep", type=int, default=40)
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()

    nx = args.nx
    nodes, elems = gen_mesh(nx, args.ny, args.lx, args.ly)
    write_inp(args.out_dir, args.name, nodes, elems,
              args.lx, args.ly, args.e_mod, args.nu, args.gc, args.ell,
              args.u_top, args.notch_band, args.nstep)
