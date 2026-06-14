"""Blender (Cycles) photorealistic render of the restored-implant hemisection.
Run headless:  blender --background --python render_implant_blender.py
Imports the PLYs from figures/blender_geom/, assigns physically-based materials (metallic Ti,
sub-surface bone/dentin/gingiva/biofilm, glossy ceramic crown), lights + camera, renders to
figures/fem_implant_blender.png.
"""
import json
import math
from pathlib import Path

import bpy
import mathutils

GEO = Path("/home/nishioka/IKM_Hiwi/nife/masterarbeit_ansys_fem/figures/blender_geom")
OUT = "/home/nishioka/IKM_Hiwi/nife/masterarbeit_ansys_fem/figures/fem_implant_blender.png"
sc = json.load(open(GEO / "scene.json"))
bb = sc["bbox"]; ctr = mathutils.Vector(((bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2))
diag = math.dist(bb[:3], bb[3:])

# material spec: (base_color RGBA, metallic, roughness, subsurface_weight, sss_radius, transmission, alpha)
MAT = {
    "bone":    ((0.86, 0.79, 0.62, 1), 0.0, 0.5, 0.22, 0.5, 0.0, 1.0),
    "implant": ((0.46, 0.48, 0.52, 1), 1.0, 0.26, 0.0, 0.0, 0.0, 1.0),
    "dentin":  ((0.90, 0.82, 0.64, 1), 0.0, 0.40, 0.30, 0.9, 0.0, 1.0),   # natural ivory tooth, SSS
    "pdl":     ((0.80, 0.62, 0.58, 1), 0.0, 0.62, 0.30, 0.5, 0.0, 1.0),   # pale ligament (not orange)
    "biofilm": ((0.82, 0.12, 0.13, 1), 0.0, 0.30, 0.32, 0.4, 0.18, 0.95),    # wet sulcular film
    "crown":   ((0.95, 0.93, 0.88, 1), 0.0, 0.16, 0.14, 0.4, 0.05, 1.0),
    "gingiva": ((0.93, 0.55, 0.60, 1), 0.0, 0.46, 0.55, 1.2, 0.0, 0.92),
    "enamel":  ((0.93, 0.94, 0.93, 1), 0.0, 0.10, 0.20, 0.6, 0.12, 1.0),   # glossy translucent enamel
}


def make_mat(name, spec):
    col, metal, rough, sss, ssr, trans, alpha = spec
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = col
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if "Subsurface Weight" in b.inputs:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (ssr, ssr * 0.6, ssr * 0.4)
        if "Subsurface Scale" in b.inputs:
            b.inputs["Subsurface Scale"].default_value = 1.0
    if "Transmission Weight" in b.inputs:
        b.inputs["Transmission Weight"].default_value = trans
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"; m.use_screen_refraction = True
    return m


FEM_BODIES = {"bone", "implant", "dentin", "pdl", "crown", "biofilm", "enamel"}   # boolean-cut at y_mid


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ymid = sc["ymid"]; big = diag * 3
    bpy.ops.mesh.primitive_cube_add(size=1)                  # half-space cutter: keeps y >= y_mid
    cutter = bpy.context.object; cutter.name = "cutter"
    cutter.scale = (big, big, big); cutter.location = (ctr.x, ymid + big * 0.5, ctr.z)
    cutter.hide_render = True
    for name in MAT:
        p = GEO / ("%s.ply" % name)
        if not p.exists():
            continue
        bpy.ops.wm.ply_import(filepath=str(p))
        ob = bpy.context.selected_objects[0]; ob.name = name
        ob.data.materials.clear(); ob.data.materials.append(make_mat(name, MAT[name]))
        sm = ob.modifiers.new("smooth", "SMOOTH")         # relax coarse tet facets
        sm.iterations = 8 if name == "crown" else (2 if name in ("pdl", "biofilm") else 5)
        sm.factor = 0.5
        if name in FEM_BODIES:                              # clean flat hemisection cut
            bm = ob.modifiers.new("cut", "BOOLEAN"); bm.operation = "INTERSECT"
            bm.object = cutter; bm.solver = "EXACT"
        bpy.ops.object.shade_smooth()
        if hasattr(ob.data, "use_auto_smooth"):
            ob.data.use_auto_smooth = False

    # camera: 3/4 oblique, looking at the centre from the front (-y), above and to the side
    cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = ctr + mathutils.Vector((-diag * 0.5, -diag * 1.18, diag * 0.5))
    cam_data.lens = 52
    con = cam.constraints.new("TRACK_TO");
    tgt = bpy.data.objects.new("tgt", None); tgt.location = ctr + mathutils.Vector((0, 0, diag * 0.06))
    bpy.context.scene.collection.objects.link(tgt)
    con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam

    # lighting: soft world + key/fill/rim area lights
    world = bpy.data.worlds.new("w"); bpy.context.scene.world = world; world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.95, 0.95, 0.97, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    for nm, loc, en, sz in [("key", (-diag, -diag, diag * 1.3), 1500, diag * 0.9),
                            ("fill", (diag * 0.8, -diag * 0.9, diag * 0.4), 480, diag),
                            ("rim", (diag * 0.4, diag, diag), 950, diag * 0.7)]:
        ld = bpy.data.lights.new(nm, "AREA"); ld.energy = en; ld.size = sz
        lo = bpy.data.objects.new(nm, ld); lo.location = ctr + mathutils.Vector(loc)
        c = lo.constraints.new("TRACK_TO"); c.target = tgt; c.track_axis = "TRACK_NEGATIVE_Z"
        bpy.context.scene.collection.objects.link(lo)

    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"; s.cycles.samples = 200
    s.cycles.use_denoising = True
    s.render.resolution_x = 1700; s.render.resolution_y = 1350
    s.render.film_transparent = True       # transparent bg; composite over WHITE afterwards (PIL):
    #   im=Image.open(OUT).convert("RGBA"); bg=Image.new("RGBA",im.size,(255,)*4)
    #   Image.alpha_composite(bg,im).convert("RGB").save(OUT)
    try:
        s.view_settings.view_transform = "AgX"; s.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        s.view_settings.view_transform = "Filmic"
    s.render.filepath = OUT
    bpy.ops.render.render(write_still=True)
    print("RENDERED", OUT)


main()
