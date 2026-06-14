"""Export the restored-implant assembly surfaces (per material) to PLY for a photorealistic Blender
render. Hemisection: keep the back half (y >= y_mid) so the cut reveals the implant; the boundary
surface of the back-half tets gives both the cut face and the outer surface. Adds the real-tooth crown
STL and the gingiva mantle. Writes to masterarbeit_ansys_fem/figures/blender_geom/<mat>.ply
Run: python masterarbeit_ansys_fem/extensions/export_blender_geom.py
"""
import sys
import json
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXT))
import fig_implant_crown_fem as cf  # noqa: E402

FEMDIR = Path("/home/nishioka/IKM_Hiwi/FEM/tier2b_real")
OUT = Path("/home/nishioka/IKM_Hiwi/nife/masterarbeit_ansys_fem/figures/blender_geom")


def write_ply(tris, path):
    """tris: (n,3,3) triangle vertex coords -> dedup vertices, write ASCII PLY with vertex normals."""
    if not len(tris):
        return 0
    V = tris.reshape(-1, 3)
    key = np.round(V, 5)
    uniq, inv = np.unique(key, axis=0, return_index=False, return_inverse=True), None
    u, inv = np.unique(np.round(V, 5), axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    # per-vertex normals (area-weighted from incident faces)
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = np.zeros((len(u), 3))
    for k in range(3):
        np.add.at(nrm, F[:, k], fn)
    nl = np.linalg.norm(nrm, axis=1, keepdims=True); nl[nl == 0] = 1
    nrm = nrm / nl
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(u))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("element face %d\n" % len(F))
        f.write("property list uchar int vertex_indices\nend_header\n")
        for (p, nn) in zip(u, nrm):
            f.write("%.5f %.5f %.5f %.4f %.4f %.4f\n" % (p[0], p[1], p[2], nn[0], nn[1], nn[2]))
        for t in F:
            f.write("3 %d %d %d\n" % (t[0], t[1], t[2]))
    return len(F)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(FEMDIR / "tier2b_crown_meta.npz", allow_pickle=True)
    nodes, conn, mat = a["nodes"], a["conn"], a["mat"]
    ymid = (nodes[:, 1].min() + nodes[:, 1].max()) / 2
    bz = nodes[conn].mean(1)[:, 2]; by = nodes[conn].mean(1)[:, 1]

    # FULL closed boundary surfaces (the back-half cut is done cleanly in Blender with a boolean) ->
    # the cut shows a flat cross-section, not the porous tet interior. Bone = cortical+cancellous outer
    # shell (the internal cancellous faces are shared, so only the clean outer + socket walls remain).
    groups = {"bone": ["CORTICAL", "CANCELLOUS"], "implant": ["TI"], "dentin": ["DENTIN"],
              "pdl": ["PDL"], "biofilm": ["BIOFILM"], "enamel": ["ENAMEL"],
              "pulp": ["PULP"], "abutscrew": ["ABUTSCREW"]}
    for name, mats in groups.items():
        idx = np.where(np.isin(mat, mats))[0]
        fb, _ = cf.boundary_faces(conn[idx])
        n = write_ply(nodes[fb], OUT / ("%s.ply" % name))
        print("  %-8s %6d faces (full)" % (name, n))

    # real-tooth ceramic crown as a fully SOLID body: use the VOXELISED real crown (cache_crown_real),
    # not the raw STL (medical STLs self-intersect and break the boolean -> hollow). Its boundary is a
    # clean closed surface, so the section cut shows a solid crown cross-section.
    crw = np.load(FEMDIR / "cache_crown_real.npz")
    cfb, _ = cf.boundary_faces(crw["tets"])
    print("  crown   %6d faces (voxel solid)" % write_ply(crw["nodes"][cfb], OUT / "crown.ply"))

    # gingiva mantle, back half
    def neck_c(M, zl):
        c = nodes[conn[mat == M]].reshape(-1, 3); s = (c[:, 2] >= zl - 0.6) & (c[:, 2] <= zl + 0.6)
        ct = c[s, :2].mean(0)
        return ct, float(np.percentile(np.hypot(c[s, 0] - ct[0], c[s, 1] - ct[1]), 99))
    ic, ir = neck_c("TI", 31.0); tc, tr = neck_c("DENTIN", 31.0)
    gm = cf.gingiva_mound(ic, ir, tc, tr, ymid)
    gm = gm[gm.mean(1)[:, 1] >= ymid]
    # quads -> tris for PLY
    gt = np.concatenate([gm[:, [0, 1, 2]], gm[:, [0, 2, 3]]], axis=0) if gm.ndim == 3 and gm.shape[1] == 4 else gm
    print("  gingiva %6d faces" % write_ply(gt, OUT / "gingiva.ply"))

    json.dump({"ymid": float(ymid), "bbox": [float(nodes[:, k].min()) for k in range(3)] +
               [float(nodes[:, k].max()) for k in range(3)]}, open(OUT / "scene.json", "w"))
    print("wrote PLYs + scene.json to", OUT)


if __name__ == "__main__":
    main()
