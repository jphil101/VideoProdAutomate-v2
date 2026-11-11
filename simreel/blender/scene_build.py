"""Runs *inside* Blender (bpy module or `blender -b -P scene_build.py -- job.json`).

Builds the site, structure, tower crane and camera from the manifest written by the
choreographer, sets up Bullet rigid-body physics, bakes it (two passes: elements
that become fixed -- bolted, or loaded by something heavier -- are locked at their
*simulated* settled pose), writes a physics QA report and renders the frames.

The script is deliberately dumb about story/timing: everything time-related arrives
as keyframes, so Blender only does geometry, physics and pixels.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import bpy  # noqa: E402  (must come first when running as the bpy module)
import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

T0 = time.time()


def log(msg):
    print(f"[blender {time.time() - T0:7.1f}s] {msg}", flush=True)


# =============================================================================== keys
def fcurves_for(id_data):
    """Channelbag fcurves of id_data's action (Blender 4.4+ layered actions, 5.x safe)."""
    from bpy_extras import anim_utils

    ad = id_data.animation_data or id_data.animation_data_create()
    if ad.action is None:
        act = bpy.data.actions.new(f"{id_data.name}_act")
        ad.action = act
    act = ad.action
    if getattr(ad, "action_slot", None) is None:
        slot = act.slots.new(id_type=id_data.id_type, name=id_data.name)
        ad.action_slot = slot
    return anim_utils.action_ensure_channelbag_for_slot(act, ad.action_slot).fcurves


def set_keys(id_data, data_path, index, frames, values, interp="LINEAR"):
    fcs = fcurves_for(id_data)
    fc = fcs.find(data_path, index=index) or fcs.new(data_path, index=index)
    n = len(frames)
    fc.keyframe_points.add(n)
    co = [0.0] * (2 * n)
    co[0::2] = [float(f) for f in frames]
    co[1::2] = [float(v) for v in values]
    fc.keyframe_points.foreach_set("co", co)
    ip = {"LINEAR": 1, "CONSTANT": 0, "BEZIER": 2}[interp]
    fc.keyframe_points.foreach_set("interpolation", [ip] * n)
    fc.update()
    return fc


def bf(f):
    """Manifest frame index -> Blender frame number (Blender's sim cache starts at 1)."""
    return int(f) + 1


# =============================================================================== materials
def hex_rgb(h, default=(0.5, 0.5, 0.5)):
    if not h:
        return default
    h = h.lstrip("#")
    try:
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return default

    def lin(c):  # sRGB -> linear
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return lin(r), lin(g), lin(b)


MAT_SPECS = {
    "steel": dict(color="#A8432C", metal=0.15, rough=0.5, noise=0.08),
    "galvanized_steel": dict(color="#A9B0B5", metal=0.9, rough=0.32, noise=0.15),
    "steel_dark": dict(color="#3A4046", metal=0.6, rough=0.45, noise=0.05),
    "concrete": dict(color="#A6A39C", metal=0.0, rough=0.9, noise=0.22),
    "precast_concrete": dict(color="#C6C3BA", metal=0.0, rough=0.82, noise=0.14),
    "timber": dict(color="#9C7448", metal=0.0, rough=0.7, noise=0.25),
    "glass": dict(color="#7FA3B8", metal=0.0, rough=0.05, glass=True),
    "module": dict(color="#E6E3DC", metal=0.0, rough=0.55, noise=0.04),
    "crane_yellow": dict(color="#E5A812", metal=0.25, rough=0.45, noise=0.06),
    "crane_dark": dict(color="#2A2D30", metal=0.5, rough=0.5),
    "cable": dict(color="#151515", metal=0.7, rough=0.4),
    "sling": dict(color="#D9561B", metal=0.0, rough=0.7),
    "counterweight": dict(color="#8E8C86", metal=0.0, rough=0.9, noise=0.2),
    "paint": dict(color="#F1F1EC", metal=0.0, rough=0.6),
    "paint_yellow": dict(color="#F2C230", metal=0.0, rough=0.6),
    "pad": dict(color="#9D9A93", metal=0.0, rough=0.92, noise=0.3, scale=0.6),
    "gravel": dict(color="#6E675C", metal=0.0, rough=0.97, noise=0.45, scale=3.0, bump=0.3),
    "soil": dict(color="#6E5A44", metal=0.0, rough=0.97, noise=0.4, scale=1.5, bump=0.3),
    "dunnage": dict(color="#7A5A3A", metal=0.0, rough=0.85, noise=0.3),
}
_MATS = {}
CHEAP = [False]  # preview quality: skip bump mapping (costly on CPU path tracing)


def material(name, color_override=None):
    key = (name, color_override)
    if key in _MATS:
        return _MATS[key]
    spec = dict(MAT_SPECS.get(name, MAT_SPECS["steel"]))
    if color_override:
        spec["color"] = color_override
    m = bpy.data.materials.new(f"{name}{'_' + color_override if color_override else ''}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    rgb = hex_rgb(spec["color"])
    base = (*rgb, 1.0)
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Metallic"].default_value = spec.get("metal", 0.0)
    bsdf.inputs["Roughness"].default_value = spec.get("rough", 0.5)
    m.diffuse_color = base  # workbench / viewport
    m.roughness = spec.get("rough", 0.5)
    m.metallic = spec.get("metal", 0.0)
    shader_out = bsdf.outputs["BSDF"]
    if spec.get("glass"):
        try:
            bsdf.inputs["Transmission Weight"].default_value = 0.85
        except KeyError:
            pass
        bsdf.inputs["Roughness"].default_value = 0.03
        m.diffuse_color = (*rgb, 0.6)
    if spec.get("noise"):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = spec.get("scale", 2.0)
        noise.inputs["Detail"].default_value = 3.0
        nt.links.new(tc.outputs["Object" if name not in ("gravel", "soil", "pad") else "Generated"], noise.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        a = spec["noise"]
        ramp.color_ramp.elements[0].color = (*(c * (1 - a) for c in rgb), 1)
        ramp.color_ramp.elements[1].color = (*(min(1.0, c * (1 + a)) for c in rgb), 1)
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        if spec.get("bump") and not CHEAP[0]:
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = spec["bump"]
            n2 = nt.nodes.new("ShaderNodeTexNoise")
            n2.inputs["Scale"].default_value = 60.0
            nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
            nt.links.new(n2.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # glow: per-object custom property "glow" drives an emissive highlight
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "OBJECT"
    attr.attribute_name = "glow"
    try:
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.62, 0.12, 1.0)
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 2.5
        nt.links.new(attr.outputs["Fac"], mul.inputs[0])
        nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])
    except KeyError:
        pass
    nt.links.new(shader_out, out.inputs["Surface"])
    _MATS[key] = m
    return m


# =============================================================================== geometry
def _bm_box(bm, size, offset=(0, 0, 0), mat_index=0, rot=None):
    lx, ly, lz = size
    geom = bmesh.ops.create_cube(bm, size=1.0)
    verts = geom["verts"]
    M = Matrix.Diagonal((lx, ly, lz, 1.0))
    if rot is not None:
        M = rot.to_4x4() @ M
    M = Matrix.Translation(Vector(offset)) @ M
    bmesh.ops.transform(bm, matrix=M, verts=verts)
    for f in {f for v in verts for f in v.link_faces}:
        f.material_index = mat_index
    return verts


def _bm_profile_extrude(bm, pts2d, length, axis, offset, mat_index):
    """Extrude a closed 2D profile (in the plane normal to `axis`) along `axis`."""
    vs = []
    for (u, v) in pts2d:
        if axis == "x":
            co = (-length / 2, u, v)
        elif axis == "y":
            co = (u, -length / 2, v)
        else:
            co = (u, v, -length / 2)
        vs.append(bm.verts.new(Vector(co) + Vector(offset)))
    f = bm.faces.new(vs)
    vec = {"x": (length, 0, 0), "y": (0, length, 0), "z": (0, 0, length)}[axis]
    ext = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=Vector(vec), verts=nv)
    faces = {fa for v in vs + nv for fa in v.link_faces}
    for fa in faces:
        fa.material_index = mat_index
    bmesh.ops.recalc_face_normals(bm, faces=list(faces))


def _ibeam_profile(b, d, tw, tf):
    hb, hd, htw = b / 2, d / 2, tw / 2
    return [(-hb, -hd), (hb, -hd), (hb, -hd + tf), (htw, -hd + tf), (htw, hd - tf), (hb, hd - tf), (hb, hd),
            (-hb, hd), (-hb, hd - tf), (-htw, hd - tf), (-htw, -hd + tf), (-hb, -hd + tf)]


def build_mesh(name, parts):
    bm = bmesh.new()
    mats = []
    for p in parts:
        mname = p["material"]
        if mname not in mats:
            mats.append(mname)
        mi = mats.index(mname)
        sh, sz, off, axis = p["shape"], p["size"], p.get("offset", (0, 0, 0)), p.get("axis", "x")
        if sh == "box":
            _bm_box(bm, sz, off, mi)
        elif sh == "ibeam":
            L, b, d, tw, tf = sz
            pts = _ibeam_profile(b, d, tw, tf)  # axis z: flanges parallel to X (column)
            _bm_profile_extrude(bm, pts, L, axis, off, mi)
        elif sh == "tube":
            L, b, d, t = sz
            dims = {"x": (L, b, d), "y": (b, L, d), "z": (b, d, L)}[axis]
            _bm_box(bm, dims, off, mi)
        elif sh == "cylinder":
            r, L = sz
            geom = bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=r, radius2=r, depth=L)
            rot = {"x": Matrix.Rotation(math.pi / 2, 4, "Y"), "y": Matrix.Rotation(math.pi / 2, 4, "X"),
                   "z": Matrix.Identity(4)}[axis]
            bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(off)) @ rot, verts=geom["verts"])
            for f in {f for v in geom["verts"] for f in v.link_faces}:
                f.material_index = mi
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = False
    return me, mats


def new_object(name, me, coll):
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def add_bevel(ob, width):
    if width and width > 0:
        m = ob.modifiers.new("bevel", "BEVEL")
        m.width = width
        m.segments = 2
        m.limit_method = "ANGLE"


# =============================================================================== rigid body
def rb_add(ob, kind="PASSIVE", mass=1.0, kinematic=False, shape="BOX", friction=0.7):
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob],
                                   selected_editable_objects=[ob]):
        bpy.ops.rigidbody.object_add()
    rb = ob.rigid_body
    rb.type = kind
    rb.collision_shape = shape
    rb.mass = max(1.0, float(mass))
    rb.friction = friction
    rb.restitution = 0.0
    rb.linear_damping = 0.05
    rb.angular_damping = 0.15
    rb.kinematic = kinematic
    rb.use_deactivation = kind == "ACTIVE"
    rb.use_start_deactivated = False
    rb.deactivate_linear_velocity = 0.02
    rb.deactivate_angular_velocity = 0.03
    return rb


# =============================================================================== site
def build_site(M, coll):
    site = M["site"]
    fx0, fy0, fx1, fy1 = site["footprint"]
    bx0, by0, bx1, by1 = site["bounds"]
    # ground
    ground_mat = "soil" if site["ground"] == "soil" else ("pad" if site["ground"] == "concrete" else "gravel")
    me, _ = build_mesh("ground", [dict(shape="box", size=(400, 400, 0.2), material=ground_mat)])
    g = new_object("Ground", me, coll)
    # origin at the box centre: Bullet centres BOX colliders on the object origin
    g.location = ((fx0 + fx1) / 2, (fy0 + fy1) / 2, -0.13)
    g.data.materials.append(material(ground_mat))
    rb_add(g, "PASSIVE")
    # ground slab under the building (top at z=0)
    m = 2.2
    me, _ = build_mesh("pad", [dict(shape="box", size=(fx1 - fx0 + 2 * m, fy1 - fy0 + 2 * m, 0.3),
                                    material="pad")])
    pad = new_object("GroundSlab", me, coll)
    pad.location = ((fx0 + fx1) / 2, (fy0 + fy1) / 2, -0.15)  # top surface at z = 0
    pad.data.materials.append(material("pad"))
    add_bevel(pad, 0.03)
    rb_add(pad, "PASSIVE")
    if site.get("grid"):
        build_grid_markings(site, coll)
    # laydown area
    for d in M.get("dunnage", []):
        me, _ = build_mesh("dunnage", [dict(shape="box", size=d["size"], material="dunnage")])
        ob = new_object("Dunnage", me, coll)
        ob.location = d["pos"]
        ob.data.materials.append(material("dunnage"))


def build_grid_markings(site, coll):
    xs, ys = site["xs"], site["ys"]
    ext = 1.6
    parts = []
    w = 0.05
    for x in xs:
        parts.append(dict(shape="box", size=(w, ys[-1] - ys[0] + 2 * ext, 0.004), offset=(x, (ys[0] + ys[-1]) / 2, 0.002),
                          material="paint"))
    for y in ys:
        parts.append(dict(shape="box", size=(xs[-1] - xs[0] + 2 * ext, w, 0.004), offset=((xs[0] + xs[-1]) / 2, y, 0.002),
                          material="paint"))
    me, mats = build_mesh("gridlines", parts)
    ob = new_object("Gridlines", me, coll)
    for mn in mats:
        ob.data.materials.append(material(mn))
    # bubbles + labels
    def bubble(label, x, y):
        me = bpy.data.meshes.new("bubble")
        bm = bmesh.new()
        n = 40
        outer = [bm.verts.new((0.46 * math.cos(2 * math.pi * k / n), 0.46 * math.sin(2 * math.pi * k / n), 0)) for k in range(n)]
        inner = [bm.verts.new((0.39 * math.cos(2 * math.pi * k / n), 0.39 * math.sin(2 * math.pi * k / n), 0)) for k in range(n)]
        for k in range(n):
            bm.faces.new((outer[k], outer[(k + 1) % n], inner[(k + 1) % n], inner[k]))
        bm.to_mesh(me)
        bm.free()
        ob = new_object(f"Bubble_{label}", me, coll)
        ob.location = (x, y, 0.004)
        ob.data.materials.append(material("paint"))
        cu = bpy.data.curves.new(f"lbl_{label}", "FONT")
        cu.body = label
        cu.align_x = "CENTER"
        cu.align_y = "CENTER"
        cu.size = 0.5
        cu.extrude = 0.002
        t = new_object(f"Label_{label}", cu, coll)
        t.location = (x, y, 0.006)
        t.data.materials.append(material("paint"))
    for i, x in enumerate(xs):
        bubble(chr(65 + i), x, ys[0] - 1.6 - 0.5)
    for j, y in enumerate(ys):
        bubble(str(j + 1), xs[0] - 1.6 - 0.5, y)


# =============================================================================== crane
def lattice_members(height, width, seg, chord=0.12, brace=0.06):
    """Square lattice mast parts (boxes) from z=0 to z=height."""
    parts = []
    h = width / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(dict(shape="box", size=(chord, chord, height), offset=(sx * h, sy * h, height / 2),
                              material="crane_yellow"))
    n = max(1, int(round(height / seg)))
    dz = height / n
    for k in range(n + 1):
        z = k * dz
        for (ox, oy, sx_, sy_) in ((0, h, width, brace), (0, -h, width, brace), (h, 0, brace, width), (-h, 0, brace, width)):
            parts.append(dict(shape="box", size=(sx_, sy_, brace), offset=(ox, oy, min(z, height - brace / 2)),
                              material="crane_yellow"))
    return parts, n, dz


def diag(p0, p1, t, material_name="crane_yellow"):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    L = d.length
    rot = d.to_track_quat("X", "Z").to_matrix()
    return dict(shape="box", size=(L, t, t), offset=tuple((p0 + p1) / 2), material=material_name, rot=rot)


def build_mesh_rot(name, parts):
    """Like build_mesh but honours an optional 3x3 'rot' per box part."""
    bm = bmesh.new()
    mats = []
    for p in parts:
        if p["material"] not in mats:
            mats.append(p["material"])
        _bm_box(bm, p["size"], p.get("offset", (0, 0, 0)), mats.index(p["material"]), p.get("rot"))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me, mats


def build_crane(M, coll):
    C = M["crane"]
    bx, by = C["base"]
    H = C["height"]
    jib = C["jib"]
    w = 1.6
    # foundation + mast
    parts, n, dz = lattice_members(H - 1.2, w, 1.6)
    for k in range(n):
        z0, z1 = k * dz, (k + 1) * dz
        for (a, b) in (((-w / 2, -w / 2), (w / 2, -w / 2)), ((w / 2, -w / 2), (w / 2, w / 2)),
                       ((w / 2, w / 2), (-w / 2, w / 2)), ((-w / 2, w / 2), (-w / 2, -w / 2))):
            p0 = (a[0], a[1], z0) if k % 2 == 0 else (b[0], b[1], z0)
            p1 = (b[0], b[1], z1) if k % 2 == 0 else (a[0], a[1], z1)
            parts.append(diag(p0, p1, 0.05))
    parts.append(dict(shape="box", size=(4.0, 4.0, 1.0), offset=(0, 0, -0.45), material="pad"))
    me, mats = build_mesh_rot("mast", parts)
    mast = new_object("CraneMast", me, coll)
    mast.location = (bx, by, 0)
    for mn in mats:
        mast.data.materials.append(material(mn))
    # slewing part: an empty at the slew ring, everything above is its child
    slew = bpy.data.objects.new("CraneSlew", None)
    coll.objects.link(slew)
    slew.location = (bx, by, H - 1.2)
    slew.rotation_mode = "XYZ"
    p = []
    p.append(dict(shape="box", size=(2.2, 2.2, 0.6), offset=(0, 0, 0.3), material="crane_dark"))  # slew ring
    p.append(dict(shape="box", size=(1.6, 1.4, 1.6), offset=(1.2, -1.4, 1.1), material="paint"))  # cab
    # jib: triangular truss along +X
    zj = 0.6
    jh = 1.3
    jw = 1.1
    nseg = max(4, int(jib / 2.5))
    for sy in (-1, 1):
        p.append(dict(shape="box", size=(jib, 0.1, 0.1), offset=(jib / 2, sy * jw / 2, zj), material="crane_yellow"))
    p.append(dict(shape="box", size=(jib, 0.1, 0.1), offset=(jib / 2, 0, zj + jh), material="crane_yellow"))
    for k in range(nseg + 1):
        x = k * jib / nseg
        p.append(dict(shape="box", size=(0.06, jw, 0.06), offset=(x, 0, zj), material="crane_yellow"))
        if k < nseg:
            x1 = (k + 1) * jib / nseg
            for sy in (-1, 1):
                p.append(diag((x, sy * jw / 2, zj), ((x + x1) / 2, 0, zj + jh), 0.05))
                p.append(diag(((x + x1) / 2, 0, zj + jh), (x1, sy * jw / 2, zj), 0.05))
    # counter-jib + counterweights
    cj = max(5.0, 0.32 * jib)
    for sy in (-1, 1):
        p.append(dict(shape="box", size=(cj, 0.12, 0.25), offset=(-cj / 2, sy * 0.6, zj), material="crane_yellow"))
    for k in range(int(cj / 1.2)):
        p.append(dict(shape="box", size=(0.25, 1.1, 0.06), offset=(-k * 1.2, 0, zj + 0.12), material="crane_yellow"))
    p.append(dict(shape="box", size=(1.6, 2.0, 1.6), offset=(-cj + 1.0, 0, zj - 0.4), material="counterweight"))
    # tower head + pendants
    apex = (0, 0, zj + 5.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.append(diag((sx * 0.6, sy * 0.6, zj), apex, 0.12))
    p.append(diag(apex, (jib * 0.62, 0, zj + jh), 0.04, "cable"))
    p.append(diag(apex, (-cj + 0.5, 0, zj + 0.1), 0.04, "cable"))
    me, mats = build_mesh_rot("slewing", p)
    top = new_object("CraneTop", me, coll)
    top.parent = slew
    for mn in mats:
        top.data.materials.append(material(mn))
    # trolley (child of slew, moves along local X)
    me, mats = build_mesh("trolley", [dict(shape="box", size=(1.0, 1.2, 0.35), material="crane_dark")])
    trolley = new_object("CraneTrolley", me, coll)
    trolley.data.materials.append(material("crane_dark"))
    trolley.parent = slew
    z_trolley_local = C["z_trolley"] - (H - 1.2)
    trolley.location = (C["trolley"][0], 0, z_trolley_local + 0.2)
    # hook block (world space)
    hb = C["hook_block"]
    me, mats = build_mesh("hook", [
        dict(shape="box", size=(0.45, 0.32, hb * 0.65), offset=(0, 0, hb * 0.55), material="crane_yellow"),
        dict(shape="box", size=(0.12, 0.12, hb * 0.35), offset=(0, 0, hb * 0.12), material="crane_dark"),
    ])
    hook = new_object("CraneHook", me, coll)
    for mn in mats:
        hook.data.materials.append(material(mn))
    # rope + slings: unit-length cylinders along +Z from their origin, scaled per frame
    def rod(name, radius, mat):
        bm = bmesh.new()
        geom = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=radius, radius2=radius, depth=1.0)
        bmesh.ops.translate(bm, vec=Vector((0, 0, 0.5)), verts=geom["verts"])
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        ob = new_object(name, me, coll)
        ob.data.materials.append(material(mat))
        ob.rotation_mode = "QUATERNION"
        return ob

    rope = rod("HoistRope", 0.022, "cable")
    slings = [rod(f"Sling{i}", 0.03, "sling") for i in range(len(C["slings"]))]

    # ---- animation
    nfr = len(C["slew"])
    frames = [bf(i) for i in range(nfr)]
    set_keys(slew, "rotation_euler", 2, frames, C["slew"])
    set_keys(trolley, "location", 0, frames, C["trolley"])
    hooks = C["hook"]
    for k in range(3):
        set_keys(hook, "location", k, frames, [h[k] for h in hooks])
    # hoist rope from trolley underside to hook top
    ztr = C["z_trolley"]
    rl = [[], [], []]
    rq = [[], [], [], []]
    rs = []
    for i in range(nfr):
        th, r = C["slew"][i], C["trolley"][i]
        top_p = Vector((bx + r * math.cos(th), by + r * math.sin(th), ztr))
        bot = Vector(hooks[i]) + Vector((0, 0, hb))
        d = top_p - bot
        q = d.to_track_quat("Z", "Y")
        for k in range(3):
            rl[k].append(bot[k])
        for k in range(4):
            rq[k].append(q[k])
        rs.append(max(0.01, d.length))
    for k in range(3):
        set_keys(rope, "location", k, frames, rl[k])
    for k in range(4):
        set_keys(rope, "rotation_quaternion", k, frames, rq[k])
    set_keys(rope, "scale", 2, frames, rs)
    for S, ob in zip(C["slings"], slings):
        vis = [False] * nfr
        for a, b in S["vis"]:
            for f in range(a, b + 1):
                if f < nfr:
                    vis[f] = True
        loc = [[], [], []]
        qq = [[], [], [], []]
        sc = []
        for i in range(nfr):
            A, Bp = Vector(S["a"][i]), Vector(S["b"][i])
            d = A - Bp
            if not vis[i] or d.length < 1e-4:
                d = Vector((0, 0, 0.01))
            q = d.to_track_quat("Z", "Y")
            for k in range(3):
                loc[k].append(Bp[k])
            for k in range(4):
                qq[k].append(q[k])
            sc.append(max(0.001, d.length))
        for k in range(3):
            set_keys(ob, "location", k, frames, loc[k])
        for k in range(4):
            set_keys(ob, "rotation_quaternion", k, frames, qq[k])
        set_keys(ob, "scale", 2, frames, sc)
        hid = [0.0 if v else 1.0 for v in vis]
        _keys_bool(ob, "hide_render", frames, hid)
        _keys_bool(ob, "hide_viewport", frames, hid)


def _keys_bool(ob, path, frames, values):
    """Constant-interpolated boolean keys, only where the value changes."""
    fr, vals = [], []
    prev = None
    for f, v in zip(frames, values):
        if v != prev:
            fr.append(f)
            vals.append(v)
            prev = v
    fcs = fcurves_for(ob)
    fc = fcs.find(path) or fcs.new(path)
    fc.keyframe_points.add(len(fr))
    co = []
    for f, v in zip(fr, vals):
        co += [float(f), float(v)]
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [0] * len(fr))
    fc.update()


# =============================================================================== elements
def build_elements(M, coll):
    objs = {}
    for e in M["elements"]:
        me, mats = build_mesh(e["id"], e["parts"])
        ob = new_object(e["id"], me, coll)
        for mn in mats:
            ob.data.materials.append(material(mn, e.get("color") if mn == e["material"] else None))
        if any(p.get("bevel") for p in e["parts"]):
            add_bevel(ob, max(p.get("bevel", 0) for p in e["parts"]))
        ob.rotation_mode = "QUATERNION"
        ob["glow"] = 0.0
        keys = e["keys"]
        k0 = keys[0]
        ob.location = k0[1:4]
        ob.rotation_quaternion = k0[4:8]
        if len(keys) > 1:
            fr = [bf(k[0]) for k in keys]
            for i in range(3):
                set_keys(ob, "location", i, fr, [k[1 + i] for k in keys])
            for i in range(4):
                set_keys(ob, "rotation_quaternion", i, fr, [k[4 + i] for k in keys])
        if e.get("scale_keys"):
            fr = [bf(f) for f, _ in e["scale_keys"]]
            for i in range(3):
                set_keys(ob, "scale", i, fr, [v for _, v in e["scale_keys"]])
        if e.get("visible_from") and not e.get("scale_keys"):
            vf = bf(e["visible_from"])
            _keys_bool(ob, "hide_render", [1, vf], [1.0, 0.0])
        if e.get("glow_keys"):
            fcs = fcurves_for(ob)
            fc = fcs.new('["glow"]')
            gk = e["glow_keys"]
            fc.keyframe_points.add(len(gk))
            co = []
            for f, v in gk:
                co += [float(bf(f)), float(v)]
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [1] * len(gk))
            fc.update()
        objs[e["id"]] = ob
    return objs


def setup_physics(M, objs, locks):
    """locks: element id -> (frame, loc, quat) at which it becomes fixed (kinematic)."""
    sc = bpy.context.scene
    if sc.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    w = sc.rigidbody_world
    w.substeps_per_frame = int(M["physics"]["substeps"])
    w.solver_iterations = int(M["physics"]["iterations"])
    w.use_split_impulse = True
    w.point_cache.frame_start = 1
    w.point_cache.frame_end = M["frames"]
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception:
        pass
    for e in M["elements"]:
        ob = objs[e["id"]]
        role = e["role"]
        moving = len(e["keys"]) > 1
        if role == "settle" and e.get("release_frame") is not None:
            rb = ob.rigid_body or rb_add(ob, "ACTIVE", e["mass"], True)
            rb.type = "ACTIVE"
            rel = bf(e["release_frame"])
            lk = locks.get(e["id"])
            _keys_bool_rb(ob, [1, rel] + ([bf(lk[0])] if lk else []), [1.0, 0.0] + ([1.0] if lk else []))
            if lk:
                f, loc, quat = lk
                f = bf(f)
                # re-enter kinematic mode, holding the pose physics produced
                for i in range(3):
                    fc = _fc(ob, "location", i)
                    fc.keyframe_points.insert(f - 1, loc[i]).interpolation = "CONSTANT"
                    fc.keyframe_points.insert(f, loc[i]).interpolation = "CONSTANT"
                for i in range(4):
                    fc = _fc(ob, "rotation_quaternion", i)
                    fc.keyframe_points.insert(f - 1, quat[i]).interpolation = "CONSTANT"
                    fc.keyframe_points.insert(f, quat[i]).interpolation = "CONSTANT"
        elif role in ("anchored", "static") or role == "settle":
            if not ob.rigid_body:
                rb_add(ob, "PASSIVE", e["mass"], kinematic=moving or bool(e.get("scale_keys")))


def _fc(ob, path, index):
    fcs = fcurves_for(ob)
    return fcs.find(path, index=index) or fcs.new(path, index=index)


def _keys_bool_rb(ob, frames, values):
    fcs = fcurves_for(ob)
    fc = fcs.find("rigid_body.kinematic")
    if fc:
        fcs.remove(fc)
    fc = fcs.new("rigid_body.kinematic")
    fc.keyframe_points.add(len(frames))
    co = []
    for f, v in zip(frames, values):
        co += [float(f), float(v)]
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [0] * len(frames))
    fc.update()


def bake(M):
    sc = bpy.context.scene
    w = sc.rigidbody_world
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception:
        pass
    t = time.time()
    try:
        with bpy.context.temp_override(scene=sc, point_cache=w.point_cache):
            bpy.ops.ptcache.bake(bake=True)
        how = "ptcache.bake"
    except Exception as exc:  # fall back to stepping through the frames
        log(f"ptcache.bake unavailable ({exc}); stepping frames")
        for f in range(1, M["frames"] + 1):
            sc.frame_set(f)
        how = "frame stepping"
    log(f"physics baked in {time.time() - t:.1f}s via {how}")


def world_pose(ob, f):
    sc = bpy.context.scene
    sc.frame_set(f)
    mw = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world
    loc, rot, _ = mw.decompose()
    return loc, rot


def physics_qa(M, objs, check_frames):
    out = []
    for e in M["elements"]:
        if e["role"] != "settle" or e.get("release_frame") is None:
            continue
        ob = objs[e["id"]]
        f = check_frames.get(e["id"], M["frames"])
        loc, rot = world_pose(ob, f)
        tgt = Vector(e["target"]["pos"])
        tq = Quaternion(e["target"]["quat"])
        dpos = (loc - tgt).length
        dang = math.degrees(rot.rotation_difference(tq).angle)
        if dang > 180:
            dang = 360 - dang
        status = "settled" if dpos < 0.015 and dang < 1.0 else ("drifted" if dpos < 0.08 and dang < 5 else "FAILED")
        out.append(dict(id=e["id"], frame=f - 1, pos=list(loc), target=list(tgt), error_m=round(dpos, 4),
                        error_deg=round(dang, 3), status=status, release_frame=e["release_frame"]))
    return out


# =============================================================================== world / camera / render
def build_world(M):
    sc = bpy.context.scene
    env = M["site"]["environment"]
    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    elev = {"clear_day": 36, "golden_hour": 9, "overcast": 50}[env]
    rot = {"clear_day": 150, "golden_hour": 120, "overcast": 150}[env]
    for t in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        try:
            sky.sky_type = t
            break
        except TypeError:
            continue
    try:
        sky.sun_elevation = math.radians(elev)
        sky.sun_rotation = math.radians(rot)
        sky.sun_disc = False
        sky.altitude = 50.0
        sky.aerosol_density = 2.0 if env == "overcast" else 1.0
    except Exception:
        pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = {"clear_day": 0.22, "golden_hour": 0.35, "overcast": 0.35}[env]
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # sun
    ld = bpy.data.lights.new("Sun", "SUN")
    ld.energy = {"clear_day": 4.2, "golden_hour": 3.0, "overcast": 0.9}[env]
    ld.angle = math.radians({"clear_day": 1.2, "golden_hour": 1.5, "overcast": 25}[env])
    ld.color = {"clear_day": (1.0, 0.96, 0.9), "golden_hour": (1.0, 0.72, 0.45), "overcast": (0.92, 0.95, 1.0)}[env]
    sun = bpy.data.objects.new("Sun", ld)
    sc.collection.objects.link(sun)
    # direction must match the sky's sun
    sun.rotation_euler = Euler((math.radians(90 - elev), 0, math.radians(rot + 90)), "XYZ")
    if env == "overcast":
        bg.inputs["Strength"].default_value = 0.6


def build_camera(M):
    sc = bpy.context.scene
    C = M["camera"]
    cd = bpy.data.cameras.new("Camera")
    cd.lens = C["focal"]
    cd.sensor_width = C["sensor"]
    cd.sensor_fit = "AUTO"
    cd.clip_start = 0.1
    cd.clip_end = 2000
    cam = bpy.data.objects.new("Camera", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    n = len(C["loc"])
    frames = [bf(i) for i in range(n)]
    for k in range(3):
        set_keys(cam, "location", k, frames, [p[k] for p in C["loc"]])
        set_keys(cam, "rotation_euler", k, frames, [r[k] for r in C["rot"]])
    return cam


def setup_render(M, frames_dir):
    sc = bpy.context.scene
    r = sc.render
    W, H = M["resolution"]
    r.resolution_x, r.resolution_y = W, H
    r.resolution_percentage = 100
    r.fps = M["fps"]
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.use_file_extension = True
    r.filepath = os.path.join(frames_dir, "f_#####")
    r.use_overwrite = False  # resume: frames already on disk are skipped
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Punchy"
    except Exception:
        pass
    eng = M["engine"]
    if eng == "cycles":
        r.engine = "CYCLES"
        cy = sc.cycles
        cy.samples = int(M["samples"])
        cy.use_adaptive_sampling = True
        cy.adaptive_threshold = 0.02 if M["quality"] != "final" else 0.01
        cy.use_denoising = True
        try:
            cy.denoiser = "OPENIMAGEDENOISE"
        except Exception:
            pass
        cy.max_bounces = 4 if M["quality"] != "final" else 8
        cy.diffuse_bounces = 2 if M["quality"] != "final" else 4
        cy.glossy_bounces = 2 if M["quality"] != "final" else 4
        cy.transmission_bounces = 4
        cy.caustics_reflective = False
        cy.caustics_refractive = False
        cy.device = "CPU"
        r.use_persistent_data = False  # long sequences with persistent data showed shading corruption
        cy.use_light_tree = True
        if M.get("gpu", "auto") != "off":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        prefs.compute_device_type = backend
                        prefs.get_devices()
                        devs = [d for d in prefs.devices if d.type == backend]
                        if devs:
                            for d in prefs.devices:
                                d.use = d.type == backend
                            cy.device = "GPU"
                            log(f"cycles: rendering on GPU ({backend})")
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        if cy.device == "CPU":
            log(f"cycles: rendering on CPU ({os.cpu_count()} threads)")
    elif eng == "eevee":
        r.engine = "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = 32 if M["quality"] != "final" else 96
        try:
            sc.eevee.use_raytracing = M["quality"] == "final"
            sc.eevee.use_shadows = True
        except Exception:
            pass
    else:
        r.engine = "BLENDER_WORKBENCH"
        sh = sc.display.shading
        sh.light = "STUDIO"
        try:
            sh.studio_light = "paint.sl"
        except Exception:
            pass
        sh.color_type = "MATERIAL"
        sh.show_shadows = True
        sh.shadow_intensity = 0.45
        sh.show_cavity = True
        sh.cavity_type = "WORLD"
        sh.show_specular_highlight = True
        sc.display.light_direction = (0.5, -0.4, 0.77)
        sc.display.render_aa = "8"
        try:
            sc.view_settings.view_transform = "Standard"
        except Exception:
            pass


def main(job_path):
    job = json.load(open(job_path))
    M = json.load(open(job["manifest"]))
    frames_dir = job["frames_dir"]
    os.makedirs(frames_dir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end = M["frames"]
    sc.render.fps = M["fps"]
    sc.unit_settings.system = "METRIC"
    coll = sc.collection
    CHEAP[0] = M["quality"] != "final"
    log("building site, structure and crane")
    build_world(M)
    build_site(M, coll)
    objs = build_elements(M, coll)
    if M.get("crane"):
        build_crane(M, coll)
    build_camera(M)
    setup_render(M, frames_dir)
    # ---- physics pass 1
    setup_physics(M, objs, {})
    bake(M)
    # lock frames: bolting, or the moment something is released onto the element
    lock_at = {}
    for e in M["elements"]:
        if e["role"] == "settle" and e.get("release_frame") is not None:
            cands = []
            if e.get("bolt_frame") is not None:
                cands.append(e["bolt_frame"])
            for o in M["elements"]:
                if e["id"] in (o.get("supports") or []) and o.get("release_frame") is not None:
                    cands.append(o["release_frame"] - int(0.6 * M["fps"]))
            cands = [c for c in cands if c > e["release_frame"] + 3]
            if cands:
                lock_at[e["id"]] = min(cands)
    locks = {}
    for k, f in lock_at.items():
        loc, rot = world_pose(objs[k], bf(f) - 1)
        locks[k] = (f, list(loc), list(rot))
    if locks:
        log(f"locking {len(locks)} settled element(s) at their simulated poses: {', '.join(locks)}")
        setup_physics(M, objs, locks)
        bake(M)
    qa = physics_qa(M, objs, {})
    json.dump(dict(elements=qa, locks={k: v[0] for k, v in locks.items()}), open(job["qa_out"], "w"), indent=2)
    for q in qa:
        log(f"QA {q['id']:>8}: {q['status']:8} error {q['error_m'] * 1000:6.1f} mm, {q['error_deg']:.2f} deg")
    if job.get("blend_out"):
        try:
            bpy.ops.wm.save_as_mainfile(filepath=job["blend_out"], compress=True)
        except Exception as exc:
            log(f"could not save .blend: {exc}")
    if job.get("no_render"):
        return
    # ---- render requested frame ranges
    count = [0]
    total = sum(1 for a, b in M["render_ranges"] for f in range(a, b + 1)
                if not os.path.exists(os.path.join(frames_dir, f"f_{bf(f):05d}.png")))
    if total < sum(b - a + 1 for a, b in M["render_ranges"]):
        log(f"resuming: {total} frame(s) left to render")
    t_r = time.time()

    def post(scene, *_):
        count[0] += 1
        if count[0] % 10 == 0 or count[0] == total:
            el = time.time() - t_r
            log(f"rendered {count[0]}/{total} frames ({el / count[0]:.2f}s/frame, eta {el / count[0] * (total - count[0]):.0f}s)")

    bpy.app.handlers.render_post.append(post)
    for a, b in M["render_ranges"]:
        sc.frame_start, sc.frame_end = bf(a), bf(b)
        bpy.ops.render.render(animation=True)
    log("render complete")


if __name__ == "__main__":
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    main(args[0])
