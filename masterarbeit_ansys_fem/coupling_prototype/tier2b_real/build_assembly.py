"""Tier-2(b) assembly: real BONE + real DENTIN(tooth24) + conforming PDL layer + root-form IMPLANT
(tooth23, Ti), coupled by *TIE at the socket walls.  Writes tier2b_real.inp.

Parts (cached by prep_meshes.py):
  BONE     real mandible crop (real alveolar sockets are voids)
  DENTIN   real tooth-24 solid
  PDL      0.2 mm conforming offset layer on the tooth-24 surface (inner nodes SHARED with dentin
           -> tooth/PDL interface is conforming; standard uniform-PDL idealisation)
  TI       real tooth-23 solid, titanium (the natural tooth is "extracted"; a root-analog implant
           fills the socket and is OSSEOINTEGRATED -> tied directly to bone, no PDL)
  BIOFILM  thin dysbiotic-growth collar carved from the bone crest around each neck
Couplings (*TIE, ADJUST=YES):  PDL-outer <-> bone socket(24);  implant <-> bone socket(23).
Steps: (1) dysbiotic biofilm growth;  (2) occlusal load on both crowns -> transmitted through bone.
Units mm, MPa.
"""
import sys
import numpy as np

FEM = "/home/nishioka/IKM_Hiwi/FEM"
sys.path.insert(0, FEM)
from biofilm_conformal_tet import compute_vertex_normals  # noqa: E402

OUT = f"{FEM}/tier2b_real"
T24_AXIS = np.array([-63.9, -41.2])
T23_AXIS = np.array([-69.4, -41.0])
CREST_Z = 29.0
PDL_THICK = 0.25
EPS_GROWTH = 0.19
# crop-box (must match prep_meshes.py)
X0, X1, Y0, Y1, Z0, Z1 = -76.0, -59.0, -47.0, -37.5, 15.0, 31.0
MATS = {"BONE": (13700., 0.30), "CORTICAL": (13700., 0.30), "CANCELLOUS": (1000., 0.30),
        "TI": (110000., 0.34), "DENTIN": (18000., 0.31), "PDL": (50., 0.45), "BIOFILM": (1.0, 0.45),
        "CROWN": (95000., 0.30)}   # monolithic lithium-disilicate-class ceramic crown
CORTICAL_THICK = 1.8          # cortical shell / lamina-dura thickness (mm) for the two-layer bone
# C3D4 face -> local node triple (Abaqus 1-based)
FACE_NODES = {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)}


def clean_tets(conn, nodes, vol_min=1e-4):
    """Flip negative-volume tets (winding) and drop sliver tets (volume < vol_min mm^3, which
    Abaqus rejects as zero/small). conn 0-based into nodes."""
    p = nodes[conn]
    v6 = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    out = conn.copy()
    neg = v6 < 0
    out[neg] = out[neg][:, [0, 1, 3, 2]]          # swap last two -> flip orientation
    keep = np.abs(v6) / 6.0 > vol_min
    return out[keep], int((~keep).sum()), int(neg.sum())


def boundary_face_centroids(tets, nodes, exclude_planes):
    """Centroids of the free (boundary) faces of a tet block, excluding faces lying on the artificial
    crop planes (so 'outer surface' = real mandible cortex + socket lamina dura, not the cut faces)."""
    cnt = {}
    for t in tets:
        for (a, b, c) in FACE_NODES.values():
            key = tuple(sorted((int(t[a]), int(t[b]), int(t[c]))))
            cnt[key] = cnt.get(key, 0) + 1
    cents = []
    for key, n in cnt.items():
        if n != 1:
            continue
        fc = nodes[list(key)].mean(axis=0)
        on_plane = any(abs(fc[ax] - val) < 0.8 for ax, val in exclude_planes)
        if not on_plane:
            cents.append(fc)
    return np.array(cents)


def free_faces(tets, elem_gids):
    """Return dict sorted-face -> (gid, face_id) for faces that appear exactly once."""
    seen = {}
    for loc, (t, g) in enumerate(zip(tets, elem_gids)):
        for fid, (a, b, c) in FACE_NODES.items():
            key = tuple(sorted((int(t[a]), int(t[b]), int(t[c]))))
            if key in seen:
                seen[key] = None            # shared -> not free
            else:
                seen[key] = (g, fid)
    return {k: v for k, v in seen.items() if v is not None}


def main():
    # optional argv: <implant_cache.npz> <job_basename>  (defaults = original root-analog model)
    imp_cache = sys.argv[1] if len(sys.argv) > 1 else "cache_implant.npz"
    job = sys.argv[2] if len(sys.argv) > 2 else "tier2b_real"
    WITH_CROWN = "crown" in job          # ceramic crown seated on the abutment -> occlusal moment arm
    bone = np.load(f"{OUT}/cache_bone.npz")
    dent = np.load(f"{OUT}/cache_dentin.npz")
    imp = np.load(f"{OUT}/{imp_cache}")
    bn, bt = bone["nodes"], bone["tets"]
    dn, dt = dent["nodes"], dent["tets"]
    sverts, sfaces = dent["sverts"], dent["sfaces"]
    inn, it = imp["nodes"], imp["tets"]
    if WITH_CROWN:
        crown_cache = "cache_crown_real.npz" if "crownreal" in job else "cache_crown.npz"
        crw = np.load(f"{OUT}/{crown_cache}")
        cnn, ct = crw["nodes"], crw["tets"]

    # PDL: offset tooth-24 surface outward along vertex normals
    vn = compute_vertex_normals(sverts, sfaces)
    pdl_outer_xyz = sverts + PDL_THICK * vn

    # map surface verts -> dentin node indices (exact coords)
    key2idx = {tuple(np.round(p, 5)): i for i, p in enumerate(dn)}
    svert2dn = np.array([key2idx[tuple(np.round(p, 5))] for p in sverts], dtype=np.int64)
    V = len(sverts)

    # ---- global node numbering ----
    Nb, Nd, Ni = len(bn), len(dn), len(inn)
    off_d = Nb
    off_i = Nb + Nd
    off_p = Nb + Nd + Ni                 # pdl OUTER nodes (inner reuse dentin ids)
    off_c = off_p + V                    # crown nodes (after pdl-outer); 0 of them if no crown
    if WITH_CROWN:
        nodes = np.vstack([bn, dn, inn, pdl_outer_xyz, cnn])
    else:
        nodes = np.vstack([bn, dn, inn, pdl_outer_xyz])

    # ---- element connectivity (global, 0-based) ----
    bt_g = bt
    dt_g = dt + off_d
    it_g = it + off_i
    ct_g = (ct + off_c) if WITH_CROWN else None
    # pdl tets: inner idx<V -> dentin global; outer idx>=V -> pdl-outer global
    pdl_tris = sfaces
    pdl_tets = []
    for (i0, i1, i2) in pdl_tris:
        B0, B1, B2 = off_d + svert2dn[i0], off_d + svert2dn[i1], off_d + svert2dn[i2]
        T0, T1, T2 = off_p + i0, off_p + i1, off_p + i2
        pdl_tets += [[B0, B1, B2, T2], [B0, B1, T2, T1], [B0, T1, T2, T0]]
    pdl_tets = np.array(pdl_tets, dtype=np.int64)

    # ---- biofilm collar: carve from BONE crest tets near each neck ----
    bcent = bn[bt].mean(axis=1)
    r24 = np.hypot(bcent[:, 0] - T24_AXIS[0], bcent[:, 1] - T24_AXIS[1])
    r23 = np.hypot(bcent[:, 0] - T23_AXIS[0], bcent[:, 1] - T23_AXIS[1])
    band = (bcent[:, 2] >= CREST_Z - 2.0) & (bcent[:, 2] <= CREST_Z + 1.5)
    coll = band & ((r24 <= 2.6) | (r23 <= 2.6))
    print(f"biofilm collar tets: {coll.sum()}")

    # ---- two-layer bone (level-up): cortical shell / lamina dura vs cancellous core ----
    # only for the generic-implant job; tier2b_real stays single-layer (preserved original).
    bone_block = bt_g[~coll]
    if job != "tier2b_real":
        from scipy.spatial import cKDTree
        planes = [(0, X0), (0, X1), (1, Y0), (1, Y1), (2, Z0)]   # artificial crop faces to ignore
        surf_c = boundary_face_centroids(bt_g, bn, planes)       # real outer + socket-wall faces
        bc = bn[bone_block].mean(axis=1)
        dist, _ = cKDTree(surf_c).query(bc)
        is_cort = dist < CORTICAL_THICK
        print(f"two-layer bone: cortical={int(is_cort.sum())} cancellous={int((~is_cort).sum())}")
        blocks = [("CORTICAL", bone_block[is_cort]), ("CANCELLOUS", bone_block[~is_cort]),
                  ("BIOFILM", bt_g[coll]), ("DENTIN", dt_g), ("TI", it_g), ("PDL", pdl_tets)]
    else:
        blocks = [("BONE", bone_block), ("BIOFILM", bt_g[coll]),
                  ("DENTIN", dt_g), ("TI", it_g), ("PDL", pdl_tets)]
    if WITH_CROWN:
        blocks.append(("CROWN", ct_g))
    eid = {}; gid = 1; elem_rows = []
    for name, conn in blocks:
        conn, ndrop, nflip = clean_tets(conn, nodes)
        if ndrop or nflip:
            print(f"  clean {name}: flipped {nflip}, dropped {ndrop} degenerate")
        ids = np.arange(gid, gid + len(conn))
        eid[name] = (ids, conn); gid += len(conn)
        elem_rows.append((name, ids, conn))
    print("element counts:", {n: len(eid[n][1]) for n in eid})

    # ---- TIE surfaces ----
    # slave: PDL exterior faces (outer + rim) ; implant exterior faces
    pdl_free_all = free_faces(pdl_tets, eid["PDL"][0])
    # keep ONLY the outer (bone-facing) faces: all 3 nodes are PDL-outer nodes (>= off_p).
    # the inner faces sit on the tooth surface and must NOT be tied to bone (that would kill the
    # 0.25 mm PDL compliance), so they are dropped.
    pdl_free = {k: v for k, v in pdl_free_all.items() if min(k) >= off_p}
    imp_free = free_faces(it_g, eid["TI"][0])
    # master: BONE boundary faces near either socket wall (over all bone elsets: single or two-layer)
    bone_sets = [s for s in ("BONE", "CORTICAL", "CANCELLOUS", "BIOFILM") if s in eid]
    bone_all = np.vstack([eid[s][1] for s in bone_sets])
    bone_gids = np.concatenate([eid[s][0] for s in bone_sets])
    bone_free = free_faces(bone_all, bone_gids)
    # restrict master to socket region (near tooth axes, below crest). The tooth-23 alveolar socket
    # is wider (buccolingually) than a standard Ø4.1 implant, so the tooth-23 master radius is enlarged
    # and the implant tie uses a larger position tolerance (the standard screw is bonded to the socket
    # walls -- an explicit idealisation, since a real placement would be in a healed/drilled ridge).
    R23_MASTER = 4.8 if job != "tier2b_real" else 3.2
    master = {}
    for k, v in bone_free.items():
        fc = nodes[list(k)].mean(axis=0)
        rr24 = np.hypot(fc[0] - T24_AXIS[0], fc[1] - T24_AXIS[1])
        rr23 = np.hypot(fc[0] - T23_AXIS[0], fc[1] - T23_AXIS[1])
        if fc[2] <= CREST_Z + 0.5 and (rr24 <= 3.2 or rr23 <= R23_MASTER):
            master[k] = v
    print(f"TIE faces: master(bone socket)={len(master)} slave_pdl={len(pdl_free)} slave_imp={len(imp_free)}")

    # ---- crown <-> abutment TIE ----
    # The ceramic crown is seated on the abutment top: tie the crown intaglio (seat) faces to the
    # abutment-top faces. Crown = slave (finer, more compliant), abutment top = master (stiff Ti).
    # The crown is a hollow cap sheathing the abutment: the load-transfer seat is the bore-roof disk
    # at the abutment-top plane (z_abut, r<RBORE), NOT the crown's lowest faces (the cervical margin).
    crown_seat, abut_top = {}, {}
    if WITH_CROWN:
        z_abut = nodes[off_i:off_i + Ni, 2].max()              # abutment platform top z (= 32.5)
        crown_free = free_faces(ct_g, eid["CROWN"][0])
        for k, v in crown_free.items():
            fc = nodes[list(k)].mean(axis=0)
            r = np.hypot(fc[0] - T23_AXIS[0], fc[1] - T23_AXIS[1])
            if abs(fc[2] - z_abut) < 0.4 and r < 2.0:          # bore-roof seat disk only
                crown_seat[k] = v
        for k, v in imp_free.items():
            if nodes[list(k)][:, 2].min() >= z_abut - 0.4:     # abutment-top disk faces only
                abut_top[k] = v
        print(f"crown TIE: slave_seat={len(crown_seat)} master_abut_top={len(abut_top)} "
              f"(z_abut={z_abut:.2f})")

    # ---- node sets ----
    nx, ny, nz = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    tol = 0.6
    fixed = np.where((np.arange(len(nodes)) < Nb) &
                     ((nz <= Z0 + tol) | (nx <= X0 + tol) | (nx >= X1 - tol) |
                      (ny <= Y0 + tol) | (ny >= Y1 - tol)))[0]
    # crown tops (load): dentin nodes & implant nodes near their max z
    d_ids = np.arange(off_d, off_d + Nd)
    i_ids = np.arange(off_i, off_i + Ni)
    too_top = d_ids[nz[d_ids] >= nz[d_ids].max() - 1.5]
    imp_top = i_ids[nz[i_ids] >= nz[i_ids].max() - 1.5]
    # with a crown, the occlusal load is applied at the crown occlusal table (z~40.8), NOT the
    # abutment top (z=32.5) -> the bite force now acts through the crown-height moment arm.
    if WITH_CROWN:
        c_ids = np.arange(off_c, off_c + len(cnn))
        crown_top = c_ids[nz[c_ids] >= nz[c_ids].max() - 1.0]
        load_top = crown_top
    else:
        load_top = imp_top
    print(f"fixed={len(fixed)} tooth_top={len(too_top)} imp_top={len(imp_top)} "
          f"load_top={len(load_top)} (crown={WITH_CROWN})")

    # ---- write INP ----
    L = []; ap = L.append
    ap("*HEADING"); ap(" Tier-2(b) real-shape implant + tooth24 + PDL + mandible, TIE-coupled")
    ap("*NODE")
    for i, (x, y, z) in enumerate(nodes, start=1):
        ap(" %d, %.5f, %.5f, %.5f" % (i, x, y, z))
    for name, ids, conn in elem_rows:
        ap("*ELEMENT, TYPE=C3D4, ELSET=%s" % name)
        for e, c in zip(ids, conn + 1):
            ap(" %d, %d, %d, %d, %d" % (e, c[0], c[1], c[2], c[3]))
    for m, (E, nu) in MATS.items():
        if m not in eid:                      # only emit materials that have elements
            continue
        ap("*SOLID SECTION, ELSET=%s, MATERIAL=%s" % (m, m))
        ap("*MATERIAL, NAME=%s" % m)
        ap("*ELASTIC"); ap(" %.1f, %.3f" % (E, nu))
        if m == "BIOFILM":
            ap("*EXPANSION"); ap(" %.6f" % EPS_GROWTH)

    def surf(name, faces):
        ap("*SURFACE, NAME=%s, TYPE=ELEMENT" % name)
        for (g, fid) in faces.values():
            ap(" %d, S%d" % (g, fid))
    surf("BONE_SOCKET", master)
    surf("PDL_OUT", pdl_free)
    surf("IMP_OUT", imp_free)
    # ADJUST=NO: do NOT move slave nodes onto the master (that collapses the thin PDL / implant-tip
    # tets); the small initial socket gap is absorbed into the rigid tie constraint instead.
    imp_tol = 2.8 if job != "tier2b_real" else 1.0   # generic screw < natural socket -> larger gap
    ap("*TIE, NAME=T_PDL, ADJUST=NO, POSITION TOLERANCE=1.0")
    ap(" PDL_OUT, BONE_SOCKET")
    ap("*TIE, NAME=T_IMP, ADJUST=NO, POSITION TOLERANCE=%.1f" % imp_tol)
    ap(" IMP_OUT, BONE_SOCKET")
    if WITH_CROWN:
        surf("CROWN_SEAT", crown_seat)
        surf("ABUT_TOP", abut_top)
        ap("*TIE, NAME=T_CROWN, ADJUST=NO, POSITION TOLERANCE=0.5")
        ap(" CROWN_SEAT, ABUT_TOP")

    def nset(nm, ids):
        ids = np.unique(np.asarray(ids)) + 1
        ap("*NSET, NSET=%s" % nm)
        for k in range(0, len(ids), 16):
            ap(" " + ",".join(str(int(v)) for v in ids[k:k + 16]))
    nset("FIXED", fixed); nset("TOOTOP", too_top); nset("IMPTOP", imp_top)
    nset("ALLN", np.arange(len(nodes)))

    ap("*INITIAL CONDITIONS, TYPE=TEMPERATURE"); ap(" ALLN, 0.0")
    ap("*BOUNDARY"); ap(" FIXED, 1, 3")
    ap("*STEP, NLGEOM=NO"); ap(" 1) dysbiotic biofilm growth"); ap("*STATIC")
    ap("*TEMPERATURE"); ap(" ALLN, 1.0")
    ap("*OUTPUT, FIELD"); ap("*NODE OUTPUT"); ap(" U")
    ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD"); ap("*END STEP")
    # occlusal load: ISO 14801-style 30deg oblique for the leveled-up generic job (lateral/axial =
    # tan30 = 0.577); the preserved tier2b_real keeps its original ~11deg (0.2) load.
    lat = 0.577 if job != "tier2b_real" else 0.2
    ang = "30deg oblique (ISO 14801)" if job != "tier2b_real" else "near-axial"
    ap("*STEP, NLGEOM=NO"); ap(" 2) occlusal load on both crowns (%s)" % ang); ap("*STATIC")
    ap("*CLOAD")
    for ids in (too_top, load_top):
        if len(ids):
            f = 100.0 / len(ids)            # ~100 N occlusal resultant per crown
            for n in np.unique(ids) + 1:
                ap(" %d, 3, %.5f" % (n, -f)); ap(" %d, 1, %.5f" % (n, lat * f))
    ap("*OUTPUT, FIELD"); ap("*NODE OUTPUT"); ap(" U")
    ap("*ELEMENT OUTPUT, POSITION=CENTROID"); ap(" S, COORD"); ap("*END STEP")
    open(f"{OUT}/{job}.inp", "w").write("\n".join(L) + "\n")

    # meta for visualisation
    matarr = np.empty(gid - 1, dtype=object)
    for name, ids, conn in elem_rows:
        matarr[ids - 1] = name
    allconn = np.vstack([c for _, _, c in elem_rows])
    np.savez(f"{OUT}/{job}_meta.npz", nodes=nodes, conn=allconn,
             mat=matarr.astype(str), cent=nodes[allconn].mean(axis=1))
    print("wrote %s.inp  total elems=%d nodes=%d" % (job, gid - 1, len(nodes)))


if __name__ == "__main__":
    main()
