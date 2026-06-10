"""
Blender scene builder for the tau-tau entanglement reconstruction event display.

Run head-less inside the Blender docker image, e.g.

    blender -b -P build_scene.py -- --shot all --samples 128

It reads ``data/event.json`` (produced by ``extract_event.py`` from a *real*
Monte-Carlo event) and builds a cinematic, pedagogical 3-D diagram of the
analysis' reconstruction method:

  * the e+e- -> ZH -> mu+mu- tau+tau- event in the lab frame (event display);
  * the Jeans impact-parameter geometry for one tau (impact parameter d,
    opening angle alpha, decay length L = |d|/sin a) in its track plane;
  * the two translucent tau decay planes and the acoplanarity angle phi
    between them -- the heart of the angular analysis -- shown both *with*
    the resolved decay displacement and in an "angular-only" version where
    the displacement is left unresolved;
  * the boost into the Higgs rest frame, where the taus become back-to-back.

Look & feel (from the brief):
  * a modern label typeface (Inter), carefully placed to avoid overlaps;
  * warm lighting set with Blender's *native* light colour temperature
    (Kelvin), plus a softly lit dome and a ground that catches soft shadows;
  * the beam axis is horizontal in frame (a clean physics->scene axis remap);
  * satin-matte materials; decay planes shown as sophisticated translucent
    panels; relevant angles drawn with proper symbol labels;
  * cinematic cameras: shallow DOF (9-blade bokeh) and slow ease-in-out moves;
  * Cycles, 4K, tiled to exactly two tiles per still; motion blur on anims;
  * subjects composed on the left/right third for slide backgrounds;
  * one consistent colour palette.

Final renders are intended for a GPU box:  pass  --device GPU  to use
OPTIX/CUDA/HIP/METAL automatically.
"""
import bpy
import json
import math
import os
import sys
from mathutils import Vector, Matrix

# ---------------------------------------------------------------------------
# Argument parsing (everything after the "--" Blender separator)
# ---------------------------------------------------------------------------
def get_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = {"shot": "all", "samples": 128, "res_percent": 100,
            "frame_start": 1, "frame_end": 96, "fps": 24, "device": "CPU"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--shot":
            args["shot"] = argv[i + 1]; i += 2
        elif a == "--samples":
            args["samples"] = int(argv[i + 1]); i += 2
        elif a == "--res-percent":
            args["res_percent"] = int(argv[i + 1]); i += 2
        elif a == "--frame-start":
            args["frame_start"] = int(argv[i + 1]); i += 2
        elif a == "--frame-end":
            args["frame_end"] = int(argv[i + 1]); i += 2
        elif a == "--fps":
            args["fps"] = int(argv[i + 1]); i += 2
        elif a == "--device":
            args["device"] = argv[i + 1].upper(); i += 2
        else:
            i += 1
    return args


ARGS = get_args()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "event.json")))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# 1 Blender unit == 1 mm == 1000 micrometres.  Tau flight lengths are a few mm.
UM_TO_BU = 1.0 / 1000.0

# Physics -> scene axis remap.  Physics (x, y, z) with beam along z maps to
# Blender (z, x, y): the beam (phys z) becomes Blender +X -> HORIZONTAL in
# frame, and up in the scene is Blender +Z (= phys y).
RMAP = Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
BEAM_DIR = RMAP @ Vector((0, 0, 1))          # = (1, 0, 0)

# ---------------------------------------------------------------------------
# Consistent colour palette  (linear-ish sRGB)
# ---------------------------------------------------------------------------
PALETTE = {
    "higgs":      (1.00, 0.78, 0.28, 1.0),   # gold
    "muon":       (0.30, 0.74, 0.96, 1.0),   # sky blue
    "tau_minus":  (0.13, 0.66, 0.58, 1.0),   # teal
    "tau_plus":   (0.96, 0.52, 0.22, 1.0),   # warm orange
    "pion_minus": (0.48, 0.86, 0.79, 1.0),   # light teal  (tau- product)
    "pion_plus":  (0.99, 0.72, 0.42, 1.0),   # light orange (tau+ product)
    "neutrino":   (0.74, 0.78, 0.82, 1.0),   # pale grey (translucent)
    "reco":       (0.97, 0.33, 0.57, 1.0),   # magenta -- the "method" accent
    "angle":      (1.00, 0.84, 0.46, 1.0),   # warm amber -- angle symbols
    "beam":       (0.55, 0.58, 0.62, 1.0),   # soft steel
    "vertex":     (0.97, 0.96, 0.93, 1.0),   # warm near-white nodes
    "ground":     (0.032, 0.028, 0.030, 1.0),
    "text":       (0.95, 0.95, 0.97, 1.0),
}
TAUCOL = {"tau_minus": "tau_minus", "tau_plus": "tau_plus"}
PIONCOL = {"tau_minus": "pion_minus", "tau_plus": "pion_plus"}

# ---------------------------------------------------------------------------
# Modern label font (bundled into the docker image at /opt/fonts/label.ttf)
# ---------------------------------------------------------------------------
def _load_label_font():
    for path in ("/opt/fonts/label.ttf", os.path.join(HERE, "fonts", "label.ttf")):
        if os.path.exists(path):
            try:
                return bpy.data.fonts.load(path)
            except Exception:
                pass
    return None


LABEL_FONT = None   # set in main(), after the factory reset


# ---------------------------------------------------------------------------
# Scene / collection helpers
# ---------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    for name in ("Lab", "Reco", "Rest", "Lights"):
        c = bpy.data.collections.new(name)
        scn.collection.children.link(c)
    return scn


def col(name):
    return bpy.data.collections[name]


def link(obj, collection_name):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col(collection_name).objects.link(obj)


def clear(collection_name):
    """Remove every object from a phase collection (so a shot can rebuild)."""
    for o in list(col(collection_name).objects):
        bpy.data.objects.remove(o, do_unlink=True)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
_MATS = {}


def _set(bsdf, names, value):
    for n in names:
        if n in bsdf.inputs:
            bsdf.inputs[n].default_value = value
            return


def matte_material(name, rgba, roughness=0.6, alpha=None, emission=0.0):
    """Satin-matte Principled material: matte base with a whisper of clearcoat
    and sheen so the soft dome reads as one long gentle highlight."""
    key = ("matte", name, tuple(rgba), roughness, alpha, emission)
    if key in _MATS:
        return _MATS[key]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b, a = rgba
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    _set(bsdf, ("Specular IOR Level", "Specular"), 0.3)
    _set(bsdf, ("Coat Weight", "Clearcoat"), 0.25)
    _set(bsdf, ("Coat Roughness", "Clearcoat Roughness"), 0.28)
    _set(bsdf, ("Sheen Weight", "Sheen"), 0.15)
    if emission > 0.0:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (r, g, b, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission
    use_alpha = alpha if alpha is not None else a
    if use_alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = use_alpha
        mat.blend_method = 'BLEND'
        mat.show_transparent_back = False
    _MATS[key] = mat
    return mat


def glass_panel_material(name, rgba, alpha=0.14):
    """A sophisticated translucent panel: lightly tinted, smooth, slightly
    glassy so the warm dome glows through it -- for the decay planes."""
    key = ("glass", name, tuple(rgba), alpha)
    if key in _MATS:
        return _MATS[key]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b, _ = rgba
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.12
    bsdf.inputs["Metallic"].default_value = 0.0
    _set(bsdf, ("Specular IOR Level", "Specular"), 0.4)
    _set(bsdf, ("Coat Weight", "Clearcoat"), 0.4)
    # a touch of self-glow so the tint stays visible at grazing angles
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (r, g, b, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 0.18
    bsdf.inputs["Alpha"].default_value = alpha
    mat.blend_method = 'BLEND'
    mat.show_transparent_back = True
    mat.use_backface_culling = False
    _MATS[key] = mat
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------
def sphere(loc, radius, mat, name, collection):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc,
                                          segments=48, ring_count=24)
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    assign(o, mat)
    link(o, collection)
    return o


def cylinder_between(p0, p1, radius, mat, name, collection, caps=True):
    p0 = Vector(p0); p1 = Vector(p1)
    d = p1 - p0
    length = max(d.length, 1e-6)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length,
                                         location=(p0 + p1) / 2.0, vertices=32)
    o = bpy.context.active_object
    o.name = name
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    bpy.ops.object.shade_smooth()
    assign(o, mat)
    link(o, collection)
    if caps:
        for end in (p0, p1):
            sphere(end, radius, mat, name + "_cap", collection)
    return o


def arrow(p0, direction, length, shaft_r, mat, name, collection):
    direction = Vector(direction).normalized()
    head_len = min(length * 0.28, shaft_r * 9.0)
    shaft_len = max(length - head_len, length * 0.4)
    p_shaft_end = Vector(p0) + direction * shaft_len
    p_tip = Vector(p0) + direction * length
    cylinder_between(p0, p_shaft_end, shaft_r, mat, name + "_shaft",
                     collection, caps=False)
    bpy.ops.mesh.primitive_cone_add(radius1=shaft_r * 2.4, radius2=0.0,
                                    depth=head_len,
                                    location=(p_shaft_end + p_tip) / 2.0,
                                    vertices=32)
    h = bpy.context.active_object
    h.name = name + "_head"
    h.rotation_mode = 'QUATERNION'
    h.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(direction)
    bpy.ops.object.shade_smooth()
    assign(h, mat)
    link(h, collection)
    return h


def make_unit_arrow_x(radius, mat, name, collection):
    """Single-mesh unit arrow along +X (tail at origin) for length-only scaling."""
    head_len = 0.26
    shaft_len = 1.0 - head_len
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=shaft_len,
                                         location=(shaft_len / 2.0, 0, 0),
                                         vertices=32,
                                         rotation=(0, math.radians(90), 0))
    shaft = bpy.context.active_object
    bpy.ops.mesh.primitive_cone_add(radius1=radius * 2.4, radius2=0.0,
                                    depth=head_len,
                                    location=(shaft_len + head_len / 2.0, 0, 0),
                                    vertices=32, rotation=(0, math.radians(90), 0))
    head = bpy.context.active_object
    bpy.ops.object.select_all(action='DESELECT')
    shaft.select_set(True); head.select_set(True)
    bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join()
    o = bpy.context.active_object
    o.name = name
    o.location = (0, 0, 0)
    bpy.ops.object.shade_smooth()
    assign(o, mat)
    link(o, collection)
    return o


def dashed_line(p0, p1, radius, mat, name, collection, n_dashes=12, duty=0.55):
    p0 = Vector(p0); p1 = Vector(p1)
    seg = (p1 - p0) / n_dashes
    for i in range(n_dashes):
        a = p0 + seg * i
        cylinder_between(a, a + seg * duty, radius, mat, f"{name}_{i}",
                         collection, caps=False)


def draw_angle_arc(apex, dir_a, dir_b, radius, mat, name, collection,
                   segments=40, tube=0.03):
    """A tube arc between two directions, denoting the angle between them."""
    a = Vector(dir_a).normalized()
    b = Vector(dir_b).normalized()
    axis = a.cross(b)
    if axis.length < 1e-9:
        return None
    axis.normalize()
    total = a.angle(b)
    pts = [Vector(apex) + (Matrix.Rotation(total * i / segments, 4, axis) @ a)
           * radius for i in range(segments + 1)]
    for i in range(segments):
        cylinder_between(pts[i], pts[i + 1], tube, mat, f"{name}_{i}",
                         collection, caps=False)
    # midpoint direction (handy for placing the symbol label)
    mid = (Matrix.Rotation(total * 0.5, 4, axis) @ a)
    return mid.normalized()


def billboard_label(body, loc, size, collection, name="lbl", weight="reg"):
    """A flat, modern-font text label that turns to face the active camera."""
    cu = bpy.data.curves.new(name, type='FONT')
    if LABEL_FONT is not None:
        cu.font = LABEL_FONT
    cu.body = body
    cu.size = size
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    o = bpy.data.objects.new(name, cu)
    mat = matte_material("text", PALETTE["text"], roughness=0.6, emission=5.0)
    o.data.materials.append(mat)
    col(collection).objects.link(o)
    o.location = loc
    return o


def face_labels_to_camera(cam):
    """Screen-align every text label with the camera (matches roll & moves)."""
    for o in bpy.data.objects:
        if o.type != 'FONT':
            continue
        for c in list(o.constraints):
            o.constraints.remove(c)
        con = o.constraints.new('COPY_ROTATION')
        con.target = cam


# ---------------------------------------------------------------------------
# Geometry extraction (with the physics->scene remap applied)
# ---------------------------------------------------------------------------
def bu(vec_um):
    """um 3-vector (list) -> remapped Blender Vector in BU."""
    return RMAP @ (Vector((vec_um[0], vec_um[1], vec_um[2])) * UM_TO_BU)


def p3(p4):
    """Spatial part of a remapped 4-vector."""
    return RMAP @ Vector((p4[1], p4[2], p4[3]))


def rdir(vec3):
    """Remap + normalise a physics 3-direction."""
    v = RMAP @ Vector((vec3[0], vec3[1], vec3[2]))
    return v.normalized()


PV = bu(DATA["lab"]["production_vertex_um"])   # production vertex (= origin)


def tau_plane_basis(label):
    """Return (pion_dir, d_hat, normal) for a tau's decay (track) plane."""
    t = DATA["taus"][label]
    pdir = rdir(t["pion_dir"])
    dvec = RMAP @ Vector(t["impact_param_vec_um"])
    d_hat = dvec.normalized() if dvec.length > 1e-12 else Vector((0, 0, 1))
    # orthonormalise d_hat against pion dir so the plane basis is clean
    d_hat = (d_hat - d_hat.dot(pdir) * pdir).normalized()
    n = pdir.cross(d_hat).normalized()
    return pdir, d_hat, n


# ===========================================================================
#  World: warm dome + temperature-controlled lights + soft-shadow ground
# ===========================================================================
DOME_RADIUS = 200.0


def build_world_and_lights():
    scn = bpy.context.scene
    world = bpy.data.worlds.new("W")
    scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs[1].default_value = 0.0   # the dome provides all ambient light

    # --- the softly lit dome: a giant sphere with a warm vertical gradient,
    #     doubling as a seamless cyclorama backdrop ---
    bpy.ops.mesh.primitive_uv_sphere_add(radius=DOME_RADIUS, location=(0, 0, 0),
                                          segments=96, ring_count=48)
    dome = bpy.context.active_object
    dome.name = "studio_dome"
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("dome")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    tex = nt.nodes.new("ShaderNodeTexCoord")
    ramp.color_ramp.interpolation = 'B_SPLINE'
    e = ramp.color_ramp.elements
    e[0].position = 0.0
    e[0].color = (0.018, 0.014, 0.012, 1.0)   # warm deep brown-black below
    e[1].position = 1.0
    e[1].color = (1.00, 0.86, 0.66, 1.0)      # warm soft glow at the zenith
    m1 = ramp.color_ramp.elements.new(0.45)
    m1.color = (0.075, 0.060, 0.052, 1.0)     # long warm-dark sweep at eye level
    m2 = ramp.color_ramp.elements.new(0.80)
    m2.color = (0.42, 0.34, 0.26, 1.0)        # amber rise toward the light
    nt.links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], emis.inputs["Color"])
    emis.inputs["Strength"].default_value = 0.6
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    dome.data.materials.append(mat)
    link(dome, "Lights")

    # --- ground that catches soft shadows; tone matches the dome's lower
    #     sweep so the horizon stays seamless ---
    bpy.ops.mesh.primitive_plane_add(size=260, location=(0, 0, -7.0))
    ground = bpy.context.active_object
    ground.name = "ground"
    assign(ground, matte_material("ground", PALETTE["ground"], roughness=0.92))
    link(ground, "Lights")

    # --- lights set with Blender's NATIVE colour temperature (Kelvin) ---
    def area(loc, energy, size, rot, kelvin):
        l = bpy.data.lights.new("area", 'AREA')
        l.energy = energy
        l.shape = 'DISK'
        l.size = size
        if hasattr(l, "use_temperature"):
            l.use_temperature = True
            l.temperature = kelvin
        o = bpy.data.objects.new("area", l)
        o.location = loc
        o.rotation_euler = rot
        col("Lights").objects.link(o)
        return o

    # warm key (3100 K, large & soft), neutral-warm rim, gentle warm top fill
    area((16, -20, 22), 22000, 38, (math.radians(46), 0, math.radians(38)), 3100)
    area((-20, 14, 14), 8000, 30, (math.radians(-54), 0, math.radians(-150)), 4500)
    area((-4, -10, 26), 6000, 34, (math.radians(18), 0, math.radians(-10)), 3600)


# ===========================================================================
#  Cameras (cinematic: shallow DOF, optional roll, animatable moves)
# ===========================================================================
def _aim(location, look_at, up=None):
    direction = (Vector(look_at) - Vector(location)).normalized()
    if up is None:
        return direction.to_track_quat('-Z', 'Y')
    f = direction
    right = f.cross(Vector(up)).normalized()
    true_up = right.cross(f).normalized()
    return Matrix((right, true_up, -f)).transposed().to_quaternion()


def add_camera(name, location, look_at, lens=50, shift_x=0.0, shift_y=0.0,
               up=None, fstop=2.8, focus_at=None):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.shift_x = shift_x
    cam_data.shift_y = shift_y
    cam = bpy.data.objects.new(name, cam_data)
    col("Lights").objects.link(cam)
    cam.location = location
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = _aim(location, look_at, up)
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = fstop
    cam_data.dof.aperture_blades = 9
    focus = Vector(focus_at) if focus_at is not None else Vector(look_at)
    cam_data.dof.focus_distance = (focus - Vector(location)).length
    return cam


def animate_camera(cam, moves, look_at, up=None, focus_at=None):
    focus = Vector(focus_at) if focus_at is not None else Vector(look_at)
    for frame, loc in moves:
        loc = Vector(loc)
        cam.location = loc
        cam.rotation_quaternion = _aim(loc, look_at, up)
        cam.data.dof.focus_distance = (focus - loc).length
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_quaternion", frame=frame)
        cam.data.dof.keyframe_insert("focus_distance", frame=frame)
    for holder in (cam, cam.data.dof.id_data):
        ad = holder.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.easing = 'EASE_IN_OUT'


def _arc_moves(f0, f1, center, start, sweep_deg, push_in=0.92, lift=0.6):
    """Slow orbit (about world up = Z) + push-in + rise: the product-shot move."""
    center = Vector(center); start = Vector(start)
    rel = start - center
    moves = []
    n = 5
    for i in range(n):
        t = i / (n - 1)
        q = Matrix.Rotation(math.radians(sweep_deg) * t, 4, Vector((0, 0, 1)))
        r = (q @ rel) * (1.0 + (push_in - 1.0) * t)
        moves.append((round(f0 + t * (f1 - f0)),
                      center + r + Vector((0, 0, lift * t))))
    return moves


# ===========================================================================
#  Decay planes + acoplanarity (the angular analysis)
# ===========================================================================
def build_decay_plane(label, span_u=5.0, span_w=2.2, offset_u=1.4):
    """A sophisticated translucent decay plane through the PV for one tau,
    spanned by the pion direction and the in-plane impact-parameter axis,
    with a glowing rim and a short normal indicator. Returns the normal."""
    pdir, d_hat, n = tau_plane_basis(label)
    color = PALETTE[TAUCOL[label]]
    m_glass = glass_panel_material("plane_" + label, color, alpha=0.13)
    m_rim = matte_material("rim_" + label, color, roughness=0.5, emission=2.5)
    m_norm = matte_material("pn_" + label, color, roughness=0.5)

    center = Vector(PV) + pdir * offset_u
    verts = [center + pdir * a + d_hat * b for (a, b) in
             [(-span_u, -span_w), (span_u, -span_w),
              (span_u, span_w), (-span_u, span_w)]]
    mesh = bpy.data.meshes.new("plane_" + label)
    mesh.from_pydata([tuple(v) for v in verts], [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new("plane_" + label, mesh)
    obj.data.materials.append(m_glass)
    col("Lab").objects.link(obj)
    # glowing rim
    for i in range(4):
        cylinder_between(verts[i], verts[(i + 1) % 4], 0.02, m_rim,
                         "rim_%s_%d" % (label, i), "Lab", caps=False)
    # short normal indicator at the PV
    arrow(PV, n, 2.0, 0.025, m_norm, "normal_" + label, "Lab")
    return n


def build_acoplanarity(n_minus, n_plus):
    """Arc + symbol for the angle between the two decay-plane normals."""
    m_ang = matte_material("angle", PALETTE["angle"], roughness=0.5, emission=2.0)
    mid = draw_angle_arc(PV, n_minus, n_plus, radius=1.7, mat=m_ang,
                         name="acop_arc", collection="Lab", tube=0.035)
    phi = math.degrees(Vector(n_minus).angle(Vector(n_plus)))
    if mid is not None:
        billboard_label("φ = %.0f°  (acoplanarity)" % phi,
                        Vector(PV) + mid * 3.1 + Vector((0, 0, 0.8)),
                        0.5, "Lab", "lbl_acop")
    return phi


# ===========================================================================
#  Lab-frame event  (displaced and angular-only variants)
# ===========================================================================
def build_event(displaced=True, show_planes=False):
    """Build the lab event into the 'Lab' collection.

    displaced  : if True, taus fly to resolved decay vertices and the pions
                 start there; if False, only the pion *directions* from the PV
                 are shown (displacement unresolved -> angular analysis only).
    show_planes: draw the two translucent decay planes + acoplanarity angle.
    """
    clear("Lab")
    lab = DATA["lab"]
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    m_vtx = matte_material("vertex", PALETTE["vertex"], roughness=0.4)
    m_beam = matte_material("beam", PALETTE["beam"], roughness=0.7)
    m_mu = matte_material("muon", PALETTE["muon"], roughness=0.5)
    m_nu = matte_material("neutrino", PALETTE["neutrino"], roughness=0.8, alpha=0.4)

    # beam (horizontal, along +/-X)
    cylinder_between(Vector(PV) - BEAM_DIR * 9, Vector(PV) + BEAM_DIR * 9,
                     0.03, m_beam, "beam", "Lab", caps=False)
    billboard_label("e⁺ e⁻ beam", Vector(PV) + BEAM_DIR * 9.6, 0.5,
                    "Lab", "lbl_beam")

    # production / Higgs decay vertex
    sphere(PV, 0.22, m_higgs, "pv", "Lab")
    billboard_label("H, Z production", Vector(PV) + Vector((0.0, 0.0, -0.95)),
                    0.5, "Lab", "lbl_pv")

    # Z -> mu mu  (the tag that fixes the Higgs 4-momentum)
    for key, p4, sym in (("mu_plus", lab["mu_plus_p4"], "μ⁺"),
                         ("mu_minus", lab["mu_minus_p4"], "μ⁻")):
        d = p3(p4).normalized()
        arrow(PV, d, 6.0, 0.045, m_mu, key, "Lab")
        billboard_label(sym, Vector(PV) + d * 6.5, 0.55, "Lab", "lbl_" + key)

    sym = {"tau_minus": ("τ⁻", "π⁻"),
           "tau_plus":  ("τ⁺", "π⁺")}
    for label in ("tau_minus", "tau_plus"):
        t = DATA["taus"][label]
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        m_pi = matte_material("pi_" + label, PALETTE[PIONCOL[label]], roughness=0.5)
        pdir = rdir(t["pion_dir"])
        dv = bu(t["decay_vertex_reco_um"])

        if displaced:
            cylinder_between(PV, dv, 0.07, m_tau, label + "_flight", "Lab")
            sphere(dv, 0.15, m_vtx, label + "_dv", "Lab")
            billboard_label(sym[label][0],
                            (Vector(PV) + dv) * 0.5 + Vector((0, 0.0, 0.4)),
                            0.55, "Lab", "lbl_" + label)
            pion_start = dv
            arrow(pion_start, pdir, 5.5, 0.045, m_pi, label + "_pion", "Lab")
            ndir = p3(t["neutrino_p4_reco"]).normalized()
            dashed_line(dv, dv + ndir * 3.0, 0.03, m_nu, label + "_nu", "Lab")
            billboard_label("ν", dv + ndir * 3.5, 0.45, "Lab",
                            "lbl_nu_" + label)
        else:
            # unresolved displacement: pion direction straight from the PV
            arrow(PV, pdir, 6.5, 0.05, m_pi, label + "_pion", "Lab")

        billboard_label(sym[label][1], Vector(PV) + pdir *
                        (6.9 if not displaced else (dv.length + 5.9)),
                        0.55, "Lab", "lbl_pi_" + label)

    if show_planes:
        n_minus = build_decay_plane("tau_minus")
        n_plus = build_decay_plane("tau_plus")
        # labels sit out on each plane surface, away from the central cluster
        for lbl, txt in (("tau_minus", "τ⁻ decay plane"),
                         ("tau_plus", "τ⁺ decay plane")):
            pdir, d_hat, _ = tau_plane_basis(lbl)
            pos = Vector(PV) + pdir * 4.6 + d_hat * 1.9
            billboard_label(txt, pos, 0.44, "Lab", "lbl_plane_" + lbl)
        build_acoplanarity(n_minus, n_plus)


# ===========================================================================
#  Jeans reconstruction geometry (one tau; the impact-parameter triangle)
# ===========================================================================
def build_reco(label="tau_plus"):
    clear("Reco")
    t = DATA["taus"][label]
    m_reco = matte_material("reco", PALETTE["reco"], roughness=0.5)
    m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
    m_pi = matte_material("pi_" + label, PALETTE[PIONCOL[label]], roughness=0.5)
    m_vtx = matte_material("vertex", PALETTE["vertex"], roughness=0.4)
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    m_ang = matte_material("angle", PALETTE["angle"], roughness=0.5, emission=2.0)
    m_rim = matte_material("rim_reco", PALETTE["reco"], roughness=0.5, emission=2.2)

    dv = bu(t["decay_vertex_reco_um"])
    pca = bu(t["pca_point_um"])
    pdir = rdir(t["pion_dir"])
    taudir = rdir(t["tau_dir"])
    dhat = (pca - Vector(PV)).normalized()

    sphere(PV, 0.14, m_higgs, "pv", "Reco")
    sphere(dv, 0.12, m_vtx, "dv", "Reco")
    cylinder_between(PV, dv, 0.045, m_tau, "tau", "Reco")

    pion_back = dv - pdir * 4.0
    pion_fwd = dv + pdir * 6.5
    cylinder_between(pion_back, pion_fwd, 0.03, m_pi, "pion_track", "Reco")

    cylinder_between(PV, pca, 0.045, m_reco, "d", "Reco")
    sphere(pca, 0.09, m_reco, "pca", "Reco")

    # translucent track plane (sophisticated, not a flooding fill)
    span_u, span_w = 3.6, 1.9
    u, w = pdir, dhat
    center = Vector(PV) + u * 2.0
    verts = [center + u * a + w * b for (a, b) in
             [(-span_u, -span_w), (span_u, -span_w),
              (span_u, span_w), (-span_u, span_w)]]
    mesh = bpy.data.meshes.new("track_plane")
    mesh.from_pydata([tuple(v) for v in verts], [], [(0, 1, 2, 3)])
    mesh.update()
    plane = bpy.data.objects.new("track_plane", mesh)
    plane.data.materials.append(glass_panel_material("reco_plane",
                                                     PALETTE["reco"], alpha=0.10))
    col("Reco").objects.link(plane)
    for i in range(4):
        cylinder_between(verts[i], verts[(i + 1) % 4], 0.014, m_rim,
                         "pe_%d" % i, "Reco", caps=False)

    # alpha angle where the tracks cross, with a symbol
    draw_angle_arc(dv, taudir, pdir, radius=1.3, mat=m_ang, name="alpha",
                   collection="Reco", tube=0.03)

    # right-angle marker: d perpendicular to the pion track
    q = 0.5
    corner = pca - pdir * q
    cylinder_between(corner, corner + pdir * q, 0.014, m_reco, "ra_a", "Reco",
                     caps=False)
    cylinder_between(corner, corner - dhat * q, 0.014, m_reco, "ra_b", "Reco",
                     caps=False)

    # labels (placed along the in-plane axes pdir/dhat for predictable layout)
    d_um = t["impact_param_mag_um"]
    a_mrad = t["alpha_rad"] * 1e3
    L_um = t["decay_length_reco_um"]
    billboard_label("PV", Vector(PV) - dhat * 0.8, 0.42, "Reco", "rl_pv")
    billboard_label("τ decay vertex", dv - pdir * 1.4 + dhat * 1.0, 0.42,
                    "Reco", "rl_dv")
    billboard_label("impact parameter  d = %.0f µm" % d_um,
                    pca - pdir * 0.3 + dhat * 1.1, 0.4, "Reco", "rl_d")
    billboard_label("π track (measured)",
                    (pca + dv) * 0.5 + dhat * 0.55, 0.42, "Reco", "rl_pi")
    billboard_label("L = |d| / sin α = %.0f µm" % L_um,
                    (Vector(PV) + dv) * 0.5 - dhat * 0.9, 0.4, "Reco", "rl_L")
    billboard_label("α = %.0f mrad" % a_mrad,
                    dv - taudir * 1.7 - dhat * 0.4, 0.4, "Reco", "rl_a")
    return pdir, dhat


# ===========================================================================
#  Higgs rest frame (taus back-to-back)
# ===========================================================================
def build_rest():
    clear("Rest")
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    origin = Vector((0, 0, 0))
    sphere(origin, 0.18, m_higgs, "rest_h", "Rest")
    billboard_label("Higgs rest frame", origin + Vector((0, 0, 1.1)), 0.5,
                    "Rest", "restl_h")
    pscale = 3.8 / 60.0
    sym = {"tau_minus": "τ⁻", "tau_plus": "τ⁺"}
    for label in ("tau_minus", "tau_plus"):
        t = DATA["taus"][label]["rest_frame"]
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        p = p3(t["tau_p4_reco"])
        L = p.length * pscale
        arrow(origin, p.normalized(), L, 0.06, m_tau, "rest_" + label, "Rest")
        billboard_label("%s   |p| = %.0f GeV ≈ M_H/2" % (sym[label], p.length),
                        origin + p.normalized() * (L + 0.7), 0.42, "Rest",
                        "restl_" + label)
    billboard_label("taus emitted back-to-back", origin + Vector((0, 0, -1.3)),
                    0.42, "Rest", "restl_b2b")


# ===========================================================================
#  Render settings
# ===========================================================================
def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except KeyError:
        return None
    for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type != 'CPU']
        if gpus:
            for d in prefs.devices:
                d.use = (d.type != 'CPU')
            bpy.context.scene.cycles.device = 'GPU'
            print(f"Cycles GPU backend: {backend} "
                  f"({', '.join(d.name for d in gpus)})")
            return backend
    return None


def setup_render(samples):
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'CPU'
    if ARGS["device"] == "GPU" and enable_gpu() is None:
        print("No GPU found -- falling back to CPU.")
    scn.cycles.samples = samples
    scn.cycles.use_adaptive_sampling = True
    scn.cycles.adaptive_threshold = 0.01
    scn.cycles.use_denoising = True
    try:
        scn.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    # enough bounces for the translucent panels to read nicely
    scn.cycles.transparent_max_bounces = 24
    scn.cycles.transmission_bounces = 12

    scn.render.resolution_x = 3840
    scn.render.resolution_y = 2160
    scn.render.resolution_percentage = ARGS["res_percent"]
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'

    # exactly two tiles for a 3840x2160 frame
    scn.cycles.use_auto_tile = True
    scn.cycles.tile_size = 2160

    try:
        scn.view_settings.view_transform = 'AgX'
        scn.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        scn.view_settings.view_transform = 'Filmic'
        scn.view_settings.look = 'None'

    scn.render.film_transparent = False   # the warm dome is the backdrop
    scn.render.fps = ARGS["fps"]


def _frame_range():
    scn = bpy.context.scene
    scn.frame_start, scn.frame_end = ARGS["frame_start"], ARGS["frame_end"]
    return ARGS["frame_start"], ARGS["frame_end"]


def set_visibility(lab, reco, rest):
    for name, on in {"Lab": lab, "Reco": reco, "Rest": rest}.items():
        c = col(name)
        c.hide_render = not on
        c.hide_viewport = not on


def render_still(cam, filename):
    scn = bpy.context.scene
    scn.camera = cam
    scn.render.use_motion_blur = False
    scn.render.image_settings.file_format = 'PNG'
    scn.render.filepath = os.path.join(OUT, filename)
    print(f"  -> rendering {filename}")
    bpy.ops.render.render(write_still=True)


def render_animation(cam, filename):
    scn = bpy.context.scene
    scn.camera = cam
    # H.264 requires even pixel dimensions; snap the effective resolution to
    # even values (matters only for odd preview percentages -- 100% is even).
    p = scn.render.resolution_percentage
    rx = (scn.render.resolution_x * p) // 100
    ry = (scn.render.resolution_y * p) // 100
    scn.render.resolution_x = rx - (rx % 2)
    scn.render.resolution_y = ry - (ry % 2)
    scn.render.resolution_percentage = 100
    scn.render.use_motion_blur = True
    scn.render.motion_blur_shutter = 0.5
    scn.cycles.motion_blur_position = 'CENTER'
    scn.render.image_settings.file_format = 'FFMPEG'
    scn.render.ffmpeg.format = 'MPEG4'
    scn.render.ffmpeg.codec = 'H264'
    scn.render.ffmpeg.constant_rate_factor = 'HIGH'
    scn.render.filepath = os.path.join(OUT, filename)
    print(f"  -> rendering animation {filename}")
    bpy.ops.render.render(animation=True)


# ===========================================================================
#  Cameras per shot  (beam horizontal; subjects to one side for slides)
# ===========================================================================
def _cam_event(animated, focus=(0, 1.4, 2.2)):
    # front-ish view from -Y so the beam (X) stays horizontal; subject RIGHT.
    # looking a touch upward drops the horizon into the lower third.
    look = focus
    start = (0, -31, 6.0)
    cam = add_camera("cam_event", location=start, look_at=look, up=(0, 0, 1),
                     lens=40, shift_x=-0.28, fstop=2.6, focus_at=(0, 0, 1.0))
    if animated:
        f0, f1 = _frame_range()
        # gentle truck + rise, kept in the X=0 plane so the beam stays level
        moves = [(f0, (0, -31, 5.0)), (round((f0 + f1) / 2), (0, -29, 6.5)),
                 (f1, (0, -28, 8.0))]
        animate_camera(cam, moves, look, up=(0, 0, 1), focus_at=(0, 0, 1.0))
    face_labels_to_camera(cam)
    return cam


def _cam_reco(animated, pdir, dhat):
    t = DATA["taus"]["tau_plus"]
    dv = bu(t["decay_vertex_reco_um"])
    focus = (Vector(PV) + dv) / 2.0
    n = pdir.cross(dhat).normalized()
    if n.z < 0:                      # keep the camera above the ground plane
        n = -n
    dist, tilt = 20.0, math.radians(16)
    loc = focus + n * dist * math.cos(tilt) + dhat * dist * math.sin(tilt)
    cam = add_camera("cam_reco", location=tuple(loc), look_at=tuple(focus),
                     lens=44, shift_x=-0.04, up=dhat, fstop=4.0,
                     focus_at=tuple(Vector(PV)))
    if animated:
        f0, f1 = _frame_range()
        moves = []
        for i in range(5):
            tt = i / 4
            ang = math.radians(-8 + 16 * tt)
            d2 = dist * (1.0 - 0.05 * tt)
            l2 = focus + (n * math.cos(ang) + pdir * math.sin(ang)) \
                * d2 * math.cos(tilt) + dhat * d2 * math.sin(tilt)
            moves.append((round(f0 + tt * (f1 - f0)), l2))
        animate_camera(cam, moves, tuple(focus), up=dhat,
                       focus_at=tuple(Vector(PV)))
    face_labels_to_camera(cam)
    return cam


def _cam_rest(animated):
    look = (0, 0, 0)
    start = (0, -22, 5.0)        # back-to-back axis is X -> horizontal
    cam = add_camera("cam_rest", location=start, look_at=look, up=(0, 0, 1),
                     lens=50, shift_x=-0.30, fstop=2.2)
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=20,
                                       push_in=0.9, lift=-0.8), look, up=(0, 0, 1))
    face_labels_to_camera(cam)
    return cam


def _cam_planes(animated):
    # 3/4 view that opens both decay planes; subject LEFT (room for the angle).
    look = (0, 0.3, 0.6)
    start = (10, -19, 12.0)
    cam = add_camera("cam_planes", location=start, look_at=look, up=(0, 0, 1),
                     lens=46, shift_x=0.26, fstop=3.2, focus_at=(0, 0, 0))
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=20,
                                       push_in=0.9, lift=0.6), look, up=(0, 0, 1))
    face_labels_to_camera(cam)
    return cam


# ===========================================================================
#  Shots
# ===========================================================================
def shot_event(animated=False):
    build_event(displaced=True, show_planes=False)
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(animated)
    render_animation(cam, "01_event_display.mp4") if animated \
        else render_still(cam, "01_event_display")


def shot_reco(animated=False):
    pdir, dhat = build_reco()
    set_visibility(lab=False, reco=True, rest=False)
    cam = _cam_reco(animated, pdir, dhat)
    render_animation(cam, "02_reconstruction_geometry.mp4") if animated \
        else render_still(cam, "02_reconstruction_geometry")


def shot_rest(animated=False):
    build_rest()
    set_visibility(lab=False, reco=False, rest=True)
    cam = _cam_rest(animated)
    render_animation(cam, "03_higgs_rest_frame.mp4") if animated \
        else render_still(cam, "03_higgs_rest_frame")


def shot_planes(animated=False):
    """Full info: resolved displacement AND the two decay planes + phi."""
    build_event(displaced=True, show_planes=True)
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_planes(animated)
    render_animation(cam, "05_decay_planes.mp4") if animated \
        else render_still(cam, "05_decay_planes")


def shot_planes_angular(animated=False):
    """Same event, displacement UNRESOLVED: angular analysis of the decay
    planes only (no tau flight / decay vertices)."""
    build_event(displaced=False, show_planes=True)
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_planes(animated)
    render_animation(cam, "06_decay_planes_angular.mp4") if animated \
        else render_still(cam, "06_decay_planes_angular")


def shot_event_angular(animated=False):
    """The event display without resolved displacement (angular-only)."""
    build_event(displaced=False, show_planes=False)
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(animated)
    render_animation(cam, "07_event_angular.mp4") if animated \
        else render_still(cam, "07_event_angular")


def shot_boost():
    """Animation: lab-frame tau momenta morph into the back-to-back rest-frame
    configuration, with motion blur on and a slow drifting camera."""
    clear("Rest")
    set_visibility(lab=False, reco=False, rest=True)
    scn = bpy.context.scene
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    origin = Vector((0, 0, 0))
    sphere(origin, 0.20, m_higgs, "boost_h", "Rest")
    billboard_label("boost into the Higgs rest frame",
                    origin + Vector((0, 0, 1.3)), 0.5, "Rest", "bl_title")
    pscale = 5.0 / 60.0
    f0, f1 = _frame_range()
    for label in ("tau_minus", "tau_plus"):
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        p_lab = p3(DATA["taus"][label]["tau_p4_reco"])
        p_rest = p3(DATA["taus"][label]["rest_frame"]["tau_p4_reco"])
        arr = make_unit_arrow_x(0.07, m_tau, "boost_" + label, "Rest")
        arr.location = origin
        arr.rotation_mode = 'QUATERNION'

        def keyed(frame, p):
            arr.rotation_quaternion = \
                Vector((1, 0, 0)).rotation_difference(p.normalized())
            arr.scale = (p.length * pscale, 1.0, 1.0)
            arr.keyframe_insert("rotation_quaternion", frame=frame)
            arr.keyframe_insert("scale", frame=frame)
        keyed(f0, p_lab)
        keyed(f1, p_rest)

    for o in col("Rest").objects:
        if o.animation_data and o.animation_data.action:
            for fc in o.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.easing = 'EASE_IN_OUT'

    look, start = (0, 0, 0), (0, -22, 5.0)
    cam = add_camera("cam_boost", location=start, look_at=look, up=(0, 0, 1),
                     lens=50, shift_x=-0.18, fstop=2.2)
    animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=16,
                                   push_in=0.92, lift=0.6), look, up=(0, 0, 1))
    face_labels_to_camera(cam)
    render_animation(cam, "04_boost_to_rest_frame.mp4")


# ===========================================================================
#  Main
# ===========================================================================
def main():
    global LABEL_FONT
    reset_scene()
    LABEL_FONT = _load_label_font()
    setup_render(ARGS["samples"])
    build_world_and_lights()

    s = ARGS["shot"]
    stills = {"event": shot_event, "reco": shot_reco, "rest": shot_rest,
              "planes": shot_planes, "planes-angular": shot_planes_angular,
              "event-angular": shot_event_angular}
    anims = {"event-anim": shot_event, "reco-anim": shot_reco,
             "rest-anim": shot_rest, "planes-anim": shot_planes,
             "boost": shot_boost}

    if s == "all":
        for fn in stills.values():
            fn(animated=False)
        shot_boost()
    elif s == "stills":
        for fn in stills.values():
            fn(animated=False)
    elif s == "anims":
        shot_event(True); shot_reco(True); shot_rest(True)
        shot_planes(True); shot_boost()
    elif s in stills:
        stills[s](animated=False)
    elif s in anims:
        if s == "boost":
            shot_boost()
        else:
            anims[s](animated=True)
    else:
        print(f"Unknown shot '{s}'")

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "tautau_event.blend"))
    print("Done.")


if __name__ == "__main__":
    main()
