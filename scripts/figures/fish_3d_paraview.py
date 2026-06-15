#!/usr/bin/env python3
"""
fish_3d_paraview.py — ParaView volume rendering of the 5-species FISH 3-D biofilm.

Takes a phi_xyz_*.npy voxel array (X, Y, Z, 5) produced by
scripts/analysis/fish_3d_profile.py / fish_3d_batch.py and renders each species as
a colour-coded translucent volume in one 3-D scene (and a side/depth view that
shows the F.nucleatum -> P.gingivalis depth separation).

Channel order / colours are the canonical FISH order (fish_decode.SPECIES_ORDER):
  0 S.oralis (blue)  1 A.naeslundii (green)  2 Vd/Vp (olive)
  3 F.nucleatum (purple)  4 P.gingivalis (red)

This is NOT a normal python script: it must be run with ParaView's pvbatch, e.g.

  PV=/home/nishioka/miniconda3/envs/pvrender/bin
  $PV/pvbatch scripts/figures/fish_3d_paraview.py \
      --npy results/fish_3d/phi_xyz_DH_d6_s0.npy \
      --out results/fish_3d/paraview --zscale 3

It writes per-species .vti (for opening in the ParaView GUI later) and PNG
screenshots into --out.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np

from paraview.simple import *  # noqa: F401,F403
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter
from vtkmodules.util import numpy_support

# canonical FISH 5-species order + tab10 colours (mirrors fish_decode.SPECIES_RGB)
SPECIES = ["S.oralis", "A.naeslundii", "Vd/Vp", "F.nucleatum", "P.gingivalis"]
SHORT = ["S.o", "A.n", "Vp", "F.n", "P.g"]
RGB = [
    (0.122, 0.467, 0.706),   # S.oralis  blue
    (0.173, 0.627, 0.173),   # A.naeslundii green
    (0.737, 0.741, 0.133),   # Vd/Vp  olive
    (0.580, 0.404, 0.741),   # F.nucleatum purple
    (0.839, 0.153, 0.157),   # P.gingivalis red
]


def box_smooth(vol: np.ndarray, n: int) -> np.ndarray:
    """Cheap separable 3-D box blur (n passes of a 3-tap mean per axis) to tame
    per-voxel decode speckle. Operates on the leading 3 spatial axes."""
    if n <= 0:
        return vol
    out = vol.astype(np.float32)
    for _ in range(n):
        for ax in (0, 1, 2):
            out = (out
                   + np.roll(out, 1, axis=ax)
                   + np.roll(out, -1, axis=ax)) / 3.0
    return out


def npy_to_vti(channel: np.ndarray, spacing, name: str, path: str) -> str:
    """Write one species scalar field (X,Y,Z) to a .vti file."""
    X, Y, Z = channel.shape
    img = vtkImageData()
    img.SetDimensions(X, Y, Z)
    img.SetSpacing(*spacing)
    # vtkImageData point order is x-fastest -> Fortran ravel of (X,Y,Z)
    flat = np.ascontiguousarray(channel.ravel(order="F"), dtype=np.float32)
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True)
    vtk_arr.SetName(name)
    img.GetPointData().SetScalars(vtk_arr)
    w = vtkXMLImageDataWriter()
    w.SetFileName(path)
    w.SetInputData(img)
    w.Write()
    return path


def add_volume(view, vti_path, name, rgb, lo, hi, opacity):
    """Opacity window [lo, hi] is shared across species (percentile-based, NOT
    the raw max which is a sparse outlier) so weak diffuse signal stays faint
    while concentrated species read solid; colour is a constant per-species hue."""
    src = XMLImageDataReader(FileName=[vti_path])
    rep = Show(src, view)
    rep.SetRepresentationType("Volume")
    ColorBy(rep, ("POINTS", name))
    ctf = GetColorTransferFunction(name)
    otf = GetOpacityTransferFunction(name)
    r, g, b = rgb
    ctf.RGBPoints = [0.0, r, g, b, hi, r, g, b]
    ctf.ColorSpace = "RGB"
    # transparent below lo, ramp to peak opacity at hi (clamps outliers)
    otf.Points = [0.0, 0.0, 0.5, 0.0, lo, 0.0, 0.5, 0.0, hi, opacity, 0.5, 0.0]
    rep.SetScalarBarVisibility(view, False)
    return src, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out", default="results/fish_3d/paraview")
    ap.add_argument("--zscale", type=float, default=3.0,
                    help="depth (z) spacing multiplier vs lateral (xy=1)")
    ap.add_argument("--downsample", type=int, default=1,
                    help="lateral downsample factor for speed")
    ap.add_argument("--opacity", type=float, default=0.7, help="peak volume opacity")
    ap.add_argument("--p-lo", type=float, default=85.0,
                    help="percentile (across shown voxels) for opacity onset")
    ap.add_argument("--p-hi", type=float, default=99.9,
                    help="percentile for peak opacity (clamps sparse outliers)")
    ap.add_argument("--basal-damp", type=float, default=0.5,
                    help="opacity multiplier for the diffuse basal S.oralis layer")
    ap.add_argument("--smooth", type=int, default=2,
                    help="box-blur passes to reduce speckle (0 = raw)")
    ap.add_argument("--size", type=int, nargs=2, default=[1800, 1400])
    ap.add_argument("--species", nargs="*", default=None,
                    help="subset of short names to show, e.g. F.n P.g")
    args = ap.parse_args()

    npy = Path(args.npy)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = npy.stem.replace("phi_xyz_", "")

    vol = np.load(npy).astype(np.float32)        # (X, Y, Z, 5)
    if vol.ndim != 4 or vol.shape[-1] != 5:
        raise SystemExit(f"expected (X,Y,Z,5), got {vol.shape}")
    ds = max(1, args.downsample)
    if ds > 1:
        vol = vol[::ds, ::ds]
    if args.smooth > 0:
        for i in range(5):
            vol[..., i] = box_smooth(vol[..., i], args.smooth)
        print(f"[fish3d] applied {args.smooth} box-blur pass(es)")
    X, Y, Z, _ = vol.shape
    spacing = (1.0, 1.0, float(args.zscale))
    print(f"[fish3d] {tag}: grid (X,Y,Z)=({X},{Y},{Z})  spacing={spacing}")

    pick = None
    if args.species:
        want = {s.lower() for s in args.species}
        pick = [i for i, s in enumerate(SHORT) if s.lower() in want]

    # global opacity window from percentiles across the species we will show
    # (raw max is a sparse outlier ~6x the 99.9th pct, so it is the wrong scale)
    shown = [i for i in range(5) if (pick is None or i in pick)]
    pool = np.concatenate([vol[..., i].ravel() for i in shown])
    lo = float(np.percentile(pool, args.p_lo))
    hi = float(np.percentile(pool, args.p_hi))
    if hi <= lo:
        hi = lo + 1e-3
    print(f"[fish3d] opacity window p{args.p_lo}={lo:.3f} .. p{args.p_hi}={hi:.3f} "
          f"(raw max={float(pool.max()):.2f})")

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = list(args.size)
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Single Color"
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 1
    view.OrientationAxesLabelColor = [0, 0, 0]

    keep = []
    for i, sp in enumerate(SPECIES):
        if pick is not None and i not in pick:
            continue
        ch = np.ascontiguousarray(vol[..., i])
        vmax = float(ch.max())
        if vmax <= 0:
            print(f"  - {SHORT[i]}: empty, skipped")
            continue
        vti = str(out / f"{tag}_{SHORT[i].replace('.', '')}.vti")
        npy_to_vti(ch, spacing, SHORT[i], vti)
        # dampen the diffuse basal commensal (S.oralis) so it reads as context
        op = args.opacity * (args.basal_damp if SHORT[i] == "S.o" else 1.0)
        src, rep = add_volume(view, vti, SHORT[i], RGB[i], lo, hi, op)
        keep.append((SHORT[i], src, rep))
        print(f"  + {SHORT[i]}: vmax={vmax:.2f}  -> {Path(vti).name}")

    if not keep:
        raise SystemExit("nothing to render")

    # geometry for explicit camera placement
    cx, cy, cz = (X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) * spacing[2] / 2.0
    span = max(X, Y, (Z - 1) * spacing[2])

    def shoot(name, pos, up):
        cam = GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(*pos)
        cam.SetViewUp(*up)
        ResetCamera(view)
        Render(view)
        p = str(out / f"{tag}_{name}.png")
        SaveScreenshot(p, view, ImageResolution=list(args.size),
                       TransparentBackground=0)
        print(f"[fish3d] wrote {p}")

    # view 1: oblique 3-D perspective
    shoot("3d", (cx + 1.4 * span, cy - 1.8 * span, cz + 1.2 * span), (0, 0, 1))
    # view 2: side (x-z) view -> depth stratification (F.n over P.g)
    shoot("side", (cx, cy - 2.6 * span, cz), (0, 0, 1))
    # view 3: top-down (x-y) -> lateral heterogeneity
    shoot("top", (cx, cy, cz + 2.6 * span), (0, 1, 0))


if __name__ == "__main__":
    main()
