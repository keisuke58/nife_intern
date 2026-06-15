#!/usr/bin/env python3
"""
fem_implant_paraview.py — ParaView (PBR) render of the restored-implant hemisection.

Loads the per-part PLYs from masterarbeit_ansys_fem/figures/blender_geom/ (the same
geometry the Blender script consumes), assigns PBR materials matching the Blender
material spec (metallic Ti, matte bone/dentin, translucent red biofilm, pink gingiva,
glossy crown), clips the solid FEM bodies at y = ymid for a flat hemisection, and
renders an oblique 3/4 view.

Run with pvbatch (offscreen via xvfb):
  PV=/home/nishioka/miniconda3/envs/pvrender/bin
  xvfb-run -a $PV/pvbatch scripts/figures/fem_implant_paraview.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from paraview.simple import *  # noqa: F401,F403
from vtkmodules.vtkIOPLY import vtkPLYReader, vtkPLYWriter
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData, vtkWindowedSincPolyDataFilter, vtkPolyDataNormals)


def smooth_ply(in_path, out_path, iters, passband):
    """Clean + windowed-sinc smooth a PLY and write it back. Windowed-sinc is
    VOLUME-PRESERVING (unlike Laplacian, which shrinks), so it rounds the coarse
    voxel/marching-cubes crown without pulling the implant/tooth tops to different
    heights. iters=0 just cleans + recomputes normals."""
    rd = vtkPLYReader(); rd.SetFileName(in_path); rd.Update()
    cl = vtkCleanPolyData(); cl.SetInputConnection(rd.GetOutputPort())
    last = cl
    if iters > 0:
        sm = vtkWindowedSincPolyDataFilter()
        sm.SetInputConnection(cl.GetOutputPort())
        sm.SetNumberOfIterations(iters)
        sm.SetPassBand(passband)        # smaller -> smoother (0.001..0.1)
        sm.SetFeatureAngle(120.0)
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOff()   # ON leaves coords in [-1,1] space -> clip drops everything
        sm.BoundarySmoothingOff()
        last = sm
    nrm = vtkPolyDataNormals(); nrm.SetInputConnection(last.GetOutputPort())
    nrm.SetFeatureAngle(60.0); nrm.SplittingOff(); nrm.ConsistencyOn(); nrm.Update()
    w = vtkPLYWriter(); w.SetFileName(out_path); w.SetInputConnection(nrm.GetOutputPort())
    w.Write()
    return out_path

# (base RGB, metallic, roughness, opacity) — mirrors render_implant_blender.MAT
MAT = {
    "bone":    ((0.86, 0.79, 0.62), 0.0, 0.65, 1.0),
    "implant": ((0.82, 0.84, 0.88), 1.0, 0.30, 1.0),
    # real tooth: warm ivory dentin root + a translucent reddish periodontal
    # ligament (PDL) so the ivory root reads through it, not a solid brown cone
    "dentin":  ((0.93, 0.85, 0.63), 0.0, 0.40, 1.0),   # warm ivory root (dominant)
    "pdl":     ((0.82, 0.46, 0.46), 0.0, 0.70, 0.28),  # faint pink ligament film
    "biofilm": ((0.83, 0.13, 0.14), 0.0, 0.45, 0.82),
    "biofilm_imp": ((0.83, 0.13, 0.14), 0.0, 0.45, 0.82),  # implant-side sulcus
    "biofilm_too": ((0.83, 0.13, 0.14), 0.0, 0.45, 0.82),  # tooth-side sulcus
    "crown":   ((0.97, 0.965, 0.95), 0.0, 0.10, 1.0),  # glossy enamel white
    "gingiva": ((0.91, 0.50, 0.55), 0.0, 0.50, 0.97),  # opaque pink mantle
}
# solid volumes get the flat hemisection cut; crown+gingiva are surface shells (kept whole)
CUT_BODIES = {"bone", "implant", "dentin", "pdl", "biofilm", "biofilm_imp", "biofilm_too"}
# draw order: opaque first, translucent last (correct alpha blending)
ORDER = ["bone", "implant", "dentin", "crown", "pdl", "gingiva",
         "biofilm", "biofilm_imp", "biofilm_too"]


def main():
    here = Path("/home/nishioka/IKM_Hiwi/nife/masterarbeit_ansys_fem/figures")
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default=str(here / "blender_geom"))
    ap.add_argument("--out", default=str(here / "fem_implant_paraview.png"))
    ap.add_argument("--size", type=int, nargs=2, default=[1800, 1450])
    ap.add_argument("--no-cut", action="store_true", help="skip the hemisection clip")
    ap.add_argument("--smooth-iters", type=int, default=20,
                    help="windowed-sinc smoothing iterations (volume-preserving, "
                         "uniform across parts; 0 = clean+normals only)")
    ap.add_argument("--passband", type=float, default=0.05,
                    help="windowed-sinc pass band (smaller = smoother; 0.001..0.2)")
    args = ap.parse_args()

    out = Path(args.out)
    geo = Path(args.geo)
    sc = json.load(open(geo / "scene.json"))
    ymid = sc["ymid"]
    bb = sc["bbox"]
    ctr = [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2]
    diag = sum((bb[i + 3] - bb[i]) ** 2 for i in range(3)) ** 0.5

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = list(args.size)
    view.UseColorPaletteForBackground = 0
    view.BackgroundColorMode = "Single Color"
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    # multi-light kit: a single headlight leaves rounded surfaces and the
    # hemisection cut faces dark; key + fill + back + bottom-fill give even
    # illumination so white reads white and the cut cap is lit.
    try:
        view.UseLight = 0          # disable the default headlight; use only the kit below
    except Exception:
        pass
    for pos, inten in [((-1.0, -1.2, 1.4), 0.85),  # key (upper front-left)
                       ((1.3, -0.9, 0.5), 0.48),   # fill (right)
                       ((0.2, 1.4, 0.9), 0.40),    # back/rim
                       ((0.0, -0.6, -1.3), 0.42)]: # bottom fill -> lights the cut cap
        L = AddLight(view=view)
        L.Coords = "Scene"
        L.Type = "Directional"
        L.Intensity = inten
        L.FocalPoint = ctr
        L.Position = [ctr[0] + pos[0] * diag, ctr[1] + pos[1] * diag,
                      ctr[2] + pos[2] * diag]

    # if the biofilm was split into implant-/tooth-side parts, skip the legacy combined one
    split_biofilm = (geo / "biofilm_imp.ply").exists() and (geo / "biofilm_too.ply").exists()

    for name in ORDER:
        p = geo / f"{name}.ply"
        if name not in MAT or not p.exists():
            continue
        if name == "biofilm" and split_biofilm:
            continue
        # volume-preserving windowed-sinc smooth (rounds the voxel crown WITHOUT
        # shrinking heights), uniform across all parts -> implant/tooth tops stay matched
        sp = str(out.parent / f"_smooth_{name}.ply")
        smooth_ply(str(p), sp, args.smooth_iters, args.passband)
        src = OpenDataFile(sp)
        if (not args.no_cut) and name in CUT_BODIES:
            clip = Clip(Input=src)
            clip.ClipType = "Plane"
            clip.ClipType.Origin = [ctr[0], ymid, ctr[2]]
            clip.ClipType.Normal = [0.0, 1.0, 0.0]
            clip.Invert = 1          # keep y >= ymid (front half toward the camera)
            src = clip
        rep = Show(src, view)
        rgb, metal, rough, opac = MAT[name]
        rep.DiffuseColor = list(rgb)
        rep.AmbientColor = list(rgb)
        # PBR needs an environment map to look right (metal/white go black without
        # IBL), so use Phong/Gouraud where the base colour reads true under the
        # headlight; map roughness -> specular so the crown is glossy, bone matte.
        rep.Interpolation = "Gouraud"
        rep.SpecularColor = [1.0, 1.0, 1.0]
        if metal >= 0.5:
            rep.Specular = 0.85
            rep.SpecularPower = 40
            rep.Ambient = 0.25
        else:
            rep.Specular = round(0.55 * (1.0 - rough), 3)
            rep.SpecularPower = 25
            rep.Ambient = 0.28
        rep.Diffuse = 0.85
        rep.Opacity = opac
        print(f"  + {name}: rgb={rgb} metal={metal} rough={rough} opac={opac}")

    fp = [ctr[0], ctr[1], ctr[2] + diag * 0.05]
    cam = GetActiveCamera()

    def save(suffix, pos, parallel):
        cam.SetFocalPoint(*fp)
        cam.SetPosition(*pos)
        cam.SetViewUp(0, 0, 1)
        view.CameraParallelProjection = 1 if parallel else 0
        ResetCamera(view)
        Render(view)
        p = str(out.with_name(out.stem + suffix + out.suffix))
        SaveScreenshot(p, view, ImageResolution=list(args.size),
                       TransparentBackground=0)
        info = {}
        if parallel:
            # world units (mm) per pixel = 2*ParallelScale / viewport height
            info["mm_per_px"] = 2.0 * view.CameraParallelScale / args.size[1]
        json.dump(info, open(p + ".json", "w"))
        print(f"[fem-implant] wrote {p}  {info}")

    # hero 3/4 oblique (perspective, matches Blender camera)
    save("_oblique", [ctr[0] - diag * 0.62, ctr[1] - diag * 1.45, ctr[2] + diag * 0.62], False)
    # clinical front view (orthographic -> carries a real mm scale bar)
    save("_front", [ctr[0], ctr[1] - diag * 1.7, ctr[2]], True)


if __name__ == "__main__":
    main()
