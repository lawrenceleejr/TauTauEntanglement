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
import shutil
import subprocess
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
        # blend_method / show_transparent_back are EEVEE-only and were dropped
        # in newer Blenders; Cycles honours the Alpha input regardless.
        if hasattr(mat, "blend_method"):
            mat.blend_method = 'BLEND'
        if hasattr(mat, "show_transparent_back"):
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
    # EEVEE-only knobs, gone in newer Blenders; harmless to skip under Cycles.
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'BLEND'
    if hasattr(mat, "show_transparent_back"):
        mat.show_transparent_back = True
    if hasattr(mat, "use_backface_culling"):
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
    head_len = min(length * 0.3, shaft_r * 9.0)
    shaft_len = length - head_len           # shaft always meets the head base
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


def billboard_label(body, loc, size, collection, name="lbl", align='CENTER'):
    """A flat, modern-font text label that turns to face the active camera.

    Understands a light TeX-ish markup so labels read like real maths rather
    than ASCII: ``_`` starts a subscript and ``^`` a superscript, each taking
    the following alphanumeric run or a ``{...}`` group ("p_H", "M_{H}/2",
    "x^2").  When such markup is present the label is composed from several
    sized + offset text pieces (see math_label); otherwise it is a single
    flat text object.
    """
    if ("_" in body or "^" in body) and any(
            seg[1] != 'n' for seg in _parse_math(body)):
        return math_label(body, loc, size, collection, name, align)
    cu = bpy.data.curves.new(name, type='FONT')
    if LABEL_FONT is not None:
        cu.font = LABEL_FONT
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = 'CENTER'
    o = bpy.data.objects.new(name, cu)
    mat = matte_material("text", PALETTE["text"], roughness=0.6, emission=5.0)
    o.data.materials.append(mat)
    col(collection).objects.link(o)
    o.location = loc
    return o


def _parse_math(text):
    """Split a label into runs: list of (substring, kind) with kind in
    {'n' normal, 'sub' subscript, 'sup' superscript}.  ``_x``/``^x`` take the
    next alphanumeric run; ``_{...}``/``^{...}`` take the braced group."""
    segs = []
    buf = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "_^" and i + 1 < n:
            if buf:
                segs.append((buf, 'n'))
                buf = ""
            kind = 'sub' if ch == '_' else 'sup'
            i += 1
            if text[i] == '{':
                j = text.find('}', i + 1)
                if j == -1:
                    j = n
                tok = text[i + 1:j]
                i = j + 1
            else:
                j = i
                while j < n and text[j].isalnum():
                    j += 1
                if j == i:           # a lone symbol after the marker
                    j = i + 1
                tok = text[i:j]
                i = j
            segs.append((tok, kind))
        else:
            buf += ch
            i += 1
    if buf:
        segs.append((buf, 'n'))
    return segs


def math_label(text, loc, size, collection, name="lbl", align='CENTER'):
    """A billboard label with real sub/superscripts.  Each run becomes its own
    text piece, sub/superscripts at 0.62x size and vertically offset; all
    pieces are parented to a single anchor empty so they screen-align as one
    unit (face_labels_to_camera constrains the anchor, the pieces inherit)."""
    segs = _parse_math(text)
    mat = matte_material("text", PALETTE["text"], roughness=0.6, emission=5.0)
    sub_scale = 0.62
    pieces = []
    for idx, (s, kind) in enumerate(segs):
        if s == "":
            continue
        cu = bpy.data.curves.new(name, type='FONT')
        if LABEL_FONT is not None:
            cu.font = LABEL_FONT
        cu.body = s
        cu.size = size * (sub_scale if kind != 'n' else 1.0)
        cu.align_x = 'LEFT'
        cu.align_y = 'CENTER'
        o = bpy.data.objects.new("%s_%d" % (name, idx), cu)
        o.data.materials.append(mat)
        col(collection).objects.link(o)
        o["mathchild"] = 1
        pieces.append((o, kind))

    # widths are only known once the text data is evaluated
    bpy.context.view_layer.update()

    x = 0.0
    kern = size * 0.04
    placed = []
    for o, kind in pieces:
        w = o.dimensions.x
        dy = {'sub': -0.20 * size, 'sup': 0.30 * size}.get(kind, 0.0)
        placed.append((o, x, dy))
        x += w + (kern if kind != 'n' else 0.0)
    total = x
    x0 = {'CENTER': -total / 2.0, 'RIGHT': -total}.get(align, 0.0)

    anchor = bpy.data.objects.new(name + "_anchor", None)
    anchor["mathlabel"] = 1
    anchor.empty_display_size = 0.01
    col(collection).objects.link(anchor)
    anchor.location = loc
    for o, x, dy in placed:
        o.parent = anchor
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.location = Vector((x0 + x, dy, 0.0))
    return anchor


def screen_caption(cam, lines, collection, size=0.34, frac_x=-0.62,
                   frac_y=0.66, d=11.0, name="caption"):
    """A multi-line caption pinned to a fixed spot in the camera frame (default
    upper-left) so a numbered storyboard reads with a steady 'slide title'."""
    q = cam.rotation_quaternion
    right = q @ Vector((1, 0, 0))
    up = q @ Vector((0, 1, 0))
    forward = q @ Vector((0, 0, -1))
    half_w = d * (cam.data.sensor_width * 0.5) / cam.data.lens
    half_h = half_w * 9.0 / 16.0
    # compensate for the camera's lens shift so the caption lands at the
    # intended screen fraction regardless of how the subject was shifted
    fx = frac_x + 2.0 * cam.data.shift_x
    fy = frac_y + 2.0 * cam.data.shift_y
    pos = (Vector(cam.location) + forward * d
           + right * (fx * half_w) + up * (fy * half_h))
    return billboard_label("\n".join(lines), pos, size, collection, name,
                           align='LEFT')


def face_labels_to_camera(cam):
    """Screen-align every text label with the camera (matches roll & moves).

    Plain labels are FONT objects and are constrained directly.  Composed
    maths labels are an anchor empty (tagged 'mathlabel') with FONT children
    (tagged 'mathchild'); only the anchor is constrained -- the children
    inherit its rotation, which is what keeps a subscript glued beside its
    base as the camera moves."""
    for o in bpy.data.objects:
        if o.get("mathchild"):
            continue
        if o.type != 'FONT' and not o.get("mathlabel"):
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


def _action_fcurves(action):
    """Yield an action's F-curves across Blender versions.  Legacy actions
    (<= 4.x) expose ``action.fcurves`` directly; slotted actions (4.4+/5.x)
    keep them under layers -> strips -> channelbags."""
    fcurves = getattr(action, "fcurves", None)
    if fcurves:
        for fc in fcurves:
            yield fc
        return
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for cbag in getattr(strip, "channelbags", []):
                for fc in cbag.fcurves:
                    yield fc


def _ease_action(action):
    """Make every keyframe of an action ease in/out with bezier handles."""
    for fc in _action_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.easing = 'EASE_IN_OUT'


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
            _ease_action(ad.action)


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
#  Higgs-rest-frame decay scene
#  (acoplanarity lives HERE: with the taus back-to-back there is exactly one
#   angle between the two decay planes, hinged on the common tau axis)
# ===========================================================================
_REST_M = None


def _rest_M():
    """Rotation that orients the displayed rest-frame space so that the
    common tau axis (k-hat = tau- direction, per spin_analysis) is EXACTLY
    world-X -- horizontal in frame, like the lab beam -- and the bisector of
    the two pion transverse directions is world-Z.  The decay planes then
    open symmetrically upward, and no camera roll is needed, so the world
    horizon stays level."""
    global _REST_M
    if _REST_M is None:
        k = rdir(DATA["taus"]["tau_minus"]["rest_frame"]["tau_dir"])

        def qt(lbl):
            p = rdir(DATA["taus"][lbl]["rest_frame"]["pion_dir"])
            t = p - p.dot(k) * k
            return t.normalized()

        e1 = k
        e3 = (qt("tau_minus") + qt("tau_plus")).normalized()  # already ⊥ k
        e2 = e3.cross(e1)
        _REST_M = Matrix((e1, e2, e3))
    return _REST_M


def rv(v):
    """Remapped vector -> axis-aligned display coordinates (rest scenes)."""
    return _rest_M() @ Vector(v)


def rest_axis():
    """The common tau axis in display coordinates: exactly world-X."""
    return Vector((1, 0, 0))


def common_qhat(label):
    """Unit transverse component of this tau's pion direction w.r.t. the
    COMMON axis.  This is the azimuthal reference the analysis itself uses
    (spin_analysis projects both pions onto the {n, r} plane normal to
    k-hat), so the angle between the two q-hats IS the acoplanarity.  Taken
    straight from the boosted reco pion direction, so it is independent of
    the (idealised) decay geometry built in rest_geo."""
    k = rest_axis()
    p = rv(rdir(DATA["taus"][label]["rest_frame"]["pion_dir"]))
    q = p - p.dot(k) * k
    return q.normalized()


def rest_geo(label):
    """Idealised rest-frame decay geometry for one tau, drawn on the common
    axis.

    In the Higgs rest frame the two taus are EXACTLY back-to-back -- the
    independently reconstructed momenta only fail to cancel because of
    reconstruction error, which we don't want the diagram to advertise.  So
    we put each tau's flight exactly on the common axis (tau- along +X, tau+
    along -X), at its reconstructed decay length, and place the pion in the
    plane spanned by that axis and the pion's azimuth q-hat, opened by the
    reconstructed opening angle alpha.  The neutrino balances the transverse
    momentum (p_nu = p_tau - p_pi, from the reco magnitudes), so it lies in
    the same plane on the far side of the axis.  The impact-parameter foot is
    then the exact perpendicular from the PV to the pion line, so the right
    triangle and L = |d| / sin(alpha) hold by construction."""
    rf = DATA["taus"][label]["rest_frame"]
    sign = 1.0 if label == "tau_minus" else -1.0
    axis = rest_axis() * sign
    L_bu = rf["decay_length_um"] * UM_TO_BU
    alpha = rf["alpha_rad"]
    dv = axis * L_bu
    q = common_qhat(label)
    pdir = (axis * math.cos(alpha) + q * math.sin(alpha)).normalized()

    # neutrino direction from momentum conservation (p_nu = p_tau - p_pi)
    ptau_mag = Vector(rf["tau_p4_reco"][1:4]).length
    ppi_mag = Vector(rf["pion_p4"][1:4]).length
    nuv = axis * ptau_mag - pdir * ppi_mag
    nudir = nuv.normalized() if nuv.length > 1e-9 else Vector(axis)

    # impact-parameter foot: perpendicular from the PV (origin) to the pion
    # line through dv (gives |d| = L sin(alpha) exactly)
    pca = dv + pdir * (-dv.dot(pdir))

    return {
        "dv":    dv,
        "kdir":  axis,
        "pdir":  pdir,
        "pca":   pca,
        "nudir": nudir,
        "L_bu":  L_bu,
        "L_um":  rf["decay_length_um"],
        "d_um":  rf["impact_param_um"],
        "alpha": alpha,
    }


def rest_acoplanarity_deg():
    """The single acoplanarity angle between the two decay planes, both
    referenced to the common tau axis (matches spin_analysis.py)."""
    return math.degrees(common_qhat("tau_minus").angle(common_qhat("tau_plus")))


def build_rest_scene(displaced=True, show_planes=False, show_ip=False,
                     show_L=False, title=True, particle_labels=True,
                     show_sep=False):
    """The Higgs decay in its own rest frame (no muons -- they only set the
    boost).  Layers:

    displaced   : taus fly to their (boosted) decay vertices, pions start
                  there; if False only the pion directions from the PV are
                  shown (the angular-only view).
    show_planes : the two translucent decay half-planes hinged on the tau
                  axis + the single acoplanarity angle phi between them.
    show_ip     : zoom-level geometry -- the pion tracks extended past their
                  PCAs, the impact-parameter segments d, right-angle markers
                  and the opening angles alpha.
    show_L      : the decay-length constraint labels L = |d| / sin(alpha).
    particle_labels : per-particle text (tau/pi/nu symbols + the PV tag).
                  Turn OFF for the down-the-axis view, where everything on
                  the tau axis projects onto the centre point and the labels
                  would pile up there.
    show_sep    : the PAPER'S PAYOFF -- an engineering-style dimension line
                  between the two decay vertices, calling out their spatial
                  separation Δx.  With the decays back-to-back the interval
                  is always spacelike (Δx = L⁺+L⁻ > c·Δt = |L⁺−L⁻|), which
                  is what makes the entanglement measurement interesting.
    """
    clear("Rest")
    origin = Vector((0, 0, 0))
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    m_vtx = matte_material("vertex", PALETTE["vertex"], roughness=0.4)
    m_nu = matte_material("neutrino", PALETTE["neutrino"], roughness=0.8,
                          alpha=0.4)
    m_reco = matte_material("reco", PALETTE["reco"], roughness=0.5)
    m_ang = matte_material("angle", PALETTE["angle"], roughness=0.5,
                           emission=2.0)

    sphere(origin, 0.18, m_higgs, "rest_h", "Rest")
    if title:
        billboard_label("Higgs decay", origin + Vector((0, 0, -0.9)), 0.44,
                        "Rest", "rs_pv")
        billboard_label("Higgs rest frame", origin + Vector((0, 0, 1.4)),
                        0.5, "Rest", "rs_title")
    elif particle_labels:
        # zoomed frames: a short PV tag, tucked down-left of the vertex
        # cluster (below the tau+ flight, clear of the phi label above)
        billboard_label("PV", origin - rest_axis() * 1.6
                        + Vector((0, 0, -0.95)), 0.42, "Rest", "rs_pv")

    sym = {"tau_minus": ("τ⁻", "π⁻"), "tau_plus": ("τ⁺", "π⁺")}
    sup = {"tau_minus": "⁻", "tau_plus": "⁺"}
    # vertical staggering: tau- labels above the axis, tau+ labels below
    vside = {"tau_minus": 1.0, "tau_plus": -1.0}

    if not displaced:
        # angular-only view: the back-to-back tau directions as one dashed
        # common axis through the PV (decay distances left unresolved)
        k = rest_axis()
        m_tm = matte_material("tau_minus", PALETTE["tau_minus"], roughness=0.5)
        m_tp = matte_material("tau_plus", PALETTE["tau_plus"], roughness=0.5)
        dashed_line(origin, origin + k * 8.0, 0.025, m_tm, "axis_m", "Rest")
        dashed_line(origin, origin - k * 8.0, 0.025, m_tp, "axis_p", "Rest")
        if particle_labels:
            billboard_label("τ⁻", origin + k * 8.6, 0.5, "Rest",
                            "rs_tau_minus")
            billboard_label("τ⁺", origin - k * 8.6, 0.5, "Rest",
                            "rs_tau_plus")
    for label in ("tau_minus", "tau_plus"):
        g = rest_geo(label)
        q = common_qhat(label)
        s = vside[label]
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        m_pi = matte_material("pi_" + label, PALETTE[PIONCOL[label]],
                              roughness=0.5)
        if displaced:
            # tau flight to the boosted decay vertex, then the pion + neutrino
            cylinder_between(origin, g["dv"], 0.06, m_tau,
                             label + "_flight", "Rest")
            sphere(g["dv"], 0.13, m_vtx, label + "_dv", "Rest")
            arrow(g["dv"], g["pdir"], 5.0, 0.04, m_pi, label + "_pion", "Rest")
            dashed_line(g["dv"], g["dv"] + g["nudir"] * 5.2, 0.028, m_nu,
                        label + "_nu", "Rest", n_dashes=22)
            if particle_labels:
                # mid-flight, so both tau tags stay inside even the zoomed
                # vertex-region framing
                billboard_label(sym[label][0],
                                g["dv"] * 0.55 + Vector((0, 0, 1.0 * s)), 0.5,
                                "Rest", "rs_" + label)
                # pi label above its arrow, nu label below its dashed line,
                # staggered in WORLD-Z (screen-up in all these cameras: the
                # pion and neutrino are nearly collinear for the small-alpha
                # tau, so an offset along q-hat projects onto the lines).
                # Both kept INBOARD of the arrow tip: the tau- tip sits
                # right at the 3/4-camera's frame edge
                billboard_label(sym[label][1],
                                g["dv"] + g["pdir"] * 4.2
                                + Vector((0, 0, 0.9)), 0.5,
                                "Rest", "rs_pi_" + label)
                billboard_label("ν",
                                g["dv"] + g["nudir"] * 2.6
                                + Vector((0, 0, -0.8)), 0.42,
                                "Rest", "rs_nu_" + label)
        else:
            # angular-only: pion direction straight from the PV
            arrow(origin, g["pdir"], 6.0, 0.045, m_pi, label + "_pion", "Rest")
            if particle_labels:
                billboard_label(sym[label][1],
                                origin + g["pdir"] * 6.0 + q * 0.7, 0.5,
                                "Rest", "rs_pi_" + label)

        if show_ip:
            # extend the pion track back past its PCA so the impact parameter
            # meets the line, then draw d with its right-angle marker
            s_pca = (g["pca"] - g["dv"]).dot(g["pdir"])
            cylinder_between(g["dv"] + g["pdir"] * (s_pca - 1.2),
                             g["dv"] + g["pdir"] * 1.5, 0.022, m_pi,
                             label + "_track_ext", "Rest", caps=False)
            cylinder_between(origin, g["pca"], 0.035, m_reco,
                             label + "_d", "Rest")
            sphere(g["pca"], 0.07, m_reco, label + "_pca", "Rest")
            dhat = g["pca"].normalized()
            q = min(0.45, g["pca"].length * 0.55)
            corner = g["pca"] - g["pdir"] * q
            cylinder_between(corner, corner + g["pdir"] * q, 0.012, m_reco,
                             label + "_ra_a", "Rest", caps=False)
            cylinder_between(corner, corner - dhat * q, 0.012, m_reco,
                             label + "_ra_b", "Rest", caps=False)
            # d labels: kicked out along each arm and split vertically so the
            # two never collide near the PV
            billboard_label("d%s" % sup[label],
                            g["dv"] * 0.28 + Vector((0, 0, 1.05 * s)), 0.42,
                            "Rest", "rs_d_" + label)

        if show_L:
            # opening angle alpha at the decay vertex: the interior angle of
            # the PV-dv-pca right triangle, between the tau line back to the PV
            # (-kdir) and the pion line back to its closest approach (-pdir)
            mid_a = draw_angle_arc(g["dv"], -g["kdir"], -g["pdir"], radius=0.9,
                                   mat=m_ang, name=label + "_alpha",
                                   collection="Rest", tube=0.022)
            if mid_a is not None:
                billboard_label("α%s" % sup[label],
                                g["dv"] + Vector(mid_a) * 1.35, 0.4,
                                "Rest", "rs_a_" + label)
            # L hugs the flight, vertically split from the d label
            billboard_label("L%s = |d%s| / sin α%s"
                            % (sup[label], sup[label], sup[label]),
                            g["dv"] * 0.62 + Vector((0, 0, -1.15 * s)), 0.4,
                            "Rest", "rs_L_" + label)

    if show_sep:
        # Engineering-style dimension line between the two decay vertices:
        # dashed drop guides from each vertex, a horizontal measure line with
        # end ticks below the axis, Δx on the line and the physics statement
        # underneath.  This is the observable the whole analysis runs against.
        m_sep = matte_material("sep", PALETTE["angle"], roughness=0.5,
                               emission=2.2)
        dv_m = rest_geo("tau_minus")["dv"]
        dv_p = rest_geo("tau_plus")["dv"]
        z_dim = -1.8
        for tag, dv in (("m", dv_m), ("p", dv_p)):
            dashed_line(dv + Vector((0, 0, -0.30)),
                        Vector((dv.x, dv.y, z_dim - 0.35)), 0.014, m_sep,
                        "sep_guide_" + tag, "Rest", n_dashes=7, duty=0.5)
        a = Vector((dv_p.x, dv_p.y, z_dim))
        b = Vector((dv_m.x, dv_m.y, z_dim))
        cylinder_between(a, b, 0.022, m_sep, "sep_line", "Rest", caps=False)
        for end, other in ((a, b), (b, a)):
            # small arrowheads pointing outward at each end
            outw = (end - other).normalized()
            bpy.ops.mesh.primitive_cone_add(radius1=0.09, radius2=0.0,
                                            depth=0.34,
                                            location=end - outw * 0.17,
                                            vertices=24)
            hcone = bpy.context.active_object
            hcone.name = "sep_tick"
            hcone.rotation_mode = 'QUATERNION'
            hcone.rotation_quaternion = \
                Vector((0, 0, 1)).rotation_difference(outw)
            bpy.ops.object.shade_smooth()
            assign(hcone, m_sep)
            link(hcone, "Rest")
        mid = (a + b) * 0.5
        billboard_label("Δx", mid + Vector((0, 0, 0.55)), 0.5,
                        "Rest", "rs_sep_dx")
        billboard_label("the two decays are spacelike separated",
                        mid + Vector((0, 0, -0.85)), 0.42,
                        "Rest", "rs_sep_txt")

    if show_planes:
        # when the impact-parameter geometry is also drawn, show the planes as
        # clean translucent surfaces only (their labels/phi arc would clash
        # with the d / alpha / L labels)
        build_rest_decay_planes(labels=not show_ip, acop=not show_ip)


def build_rest_decay_planes(labels=True, acop=True):
    """The two decay half-planes, BOTH hinged on the common back-to-back tau
    axis (the analysis's k-hat), each opened toward its pion's transverse
    direction.  Their dihedral angle about the axis is the SINGLE
    acoplanarity angle phi -- exactly the analysis observable."""
    m_ang = matte_material("angle", PALETTE["angle"], roughness=0.5,
                           emission=2.0)
    k = rest_axis()
    for label in ("tau_minus", "tau_plus"):
        g = rest_geo(label)
        color = PALETTE[TAUCOL[label]]
        m_glass = glass_panel_material("rplane_" + label, color, alpha=0.13)
        m_rim = matte_material("rrim_" + label, color, roughness=0.5,
                               emission=2.5)
        w = common_qhat(label)
        # the hinge edge (b = 0) runs ALONG this tau's momentum: from the PV
        # out through the decay vertex.  side picks which way along the common
        # axis this tau flies.
        side = 1.0 if g["pdir"].dot(k) >= 0 else -1.0
        alpha = g["alpha"]
        # the plane must ENTIRELY CONTAIN the drawn pion vector.  The pion is
        # drawn either from the decay vertex (displaced view, length ~5) or
        # from the PV (angular-only view, length ~6); cover the larger reach
        # plus a margin, in both the axial (a) and transverse (b) directions.
        a_reach = max(g["L_bu"] + 5.0 * math.cos(alpha), 6.0 * math.cos(alpha))
        b_reach = 6.0 * math.sin(alpha)
        span = a_reach + 1.4
        b1 = max(3.0, b_reach + 1.0)        # opening: toward the pion
        a0, a1 = (-0.8, span) if side > 0 else (-span, 0.8)
        verts = [k * a + w * b for (a, b) in
                 [(a0, 0.0), (a1, 0.0), (a1, b1), (a0, b1)]]
        mesh = bpy.data.meshes.new("rplane_" + label)
        mesh.from_pydata([tuple(v) for v in verts], [], [(0, 1, 2, 3)])
        mesh.update()
        obj = bpy.data.objects.new("rplane_" + label, mesh)
        obj.data.materials.append(m_glass)
        col("Rest").objects.link(obj)
        for i in range(4):
            cylinder_between(verts[i], verts[(i + 1) % 4], 0.018, m_rim,
                             "rpe_%s_%d" % (label, i), "Rest", caps=False)
        if labels:
            # toward the plane's outer end, lifted in WORLD-Z above the rim:
            # an offset along w can point toward the camera and projects to
            # nothing, while +Z always reads as screen-up in these views
            billboard_label("%s decay plane"
                            % {"tau_minus": "τ⁻", "tau_plus": "τ⁺"}[label],
                            k * (side * span * 0.8) + w * b1
                            + Vector((0, 0, 1.1)),
                            0.42, "Rest", "rs_plane_" + label)

    if acop:
        # ONE angle between the two half-planes, around the common axis.
        # A small arc tucked at the hinge reads as a single dihedral angle.
        mid = draw_angle_arc((0, 0, 0), common_qhat("tau_minus"),
                             common_qhat("tau_plus"), radius=0.65, mat=m_ang,
                             name="acop", collection="Rest", tube=0.03)
        if mid is not None:
            billboard_label("φ  (acoplanarity)",
                            Vector(mid) * 1.45, 0.46, "Rest", "rs_acop")


# ===========================================================================
#  Lab-frame event  (displaced and angular-only variants)
# ===========================================================================
def build_event(displaced=True, show_muons=True):
    """Build the lab event into the 'Lab' collection.

    displaced : if True, taus fly to resolved decay vertices and the pions
                start there; if False, only the pion *directions* from the PV
                are shown (displacement unresolved -> angular analysis only).
    (Decay planes / acoplanarity are deliberately NOT drawn in the lab frame:
     with the taus not back-to-back there is no single angle between them --
     see build_rest_scene.)
    """
    clear("Lab")
    lab = DATA["lab"]
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    m_vtx = matte_material("vertex", PALETTE["vertex"], roughness=0.4)
    m_beam = matte_material("beam", PALETTE["beam"], roughness=0.7)
    m_mu = matte_material("muon", PALETTE["muon"], roughness=0.5)
    m_nu = matte_material("neutrino", PALETTE["neutrino"], roughness=0.8, alpha=0.4)

    # beam (horizontal, along +/-X); label on the quiet LEFT end -- the right
    # end is where the tau- pion / neutrino lines exit the frame
    cylinder_between(Vector(PV) - BEAM_DIR * 9, Vector(PV) + BEAM_DIR * 9,
                     0.03, m_beam, "beam", "Lab", caps=False)
    billboard_label("e⁺ e⁻ beam", Vector(PV) - BEAM_DIR * 9.7, 0.5,
                    "Lab", "lbl_beam", align='RIGHT')

    # production / Higgs decay vertex; label in the empty lower-right
    # quadrant (mu+ exits lower-LEFT and would cross a centred label)
    sphere(PV, 0.22, m_higgs, "pv", "Lab")
    billboard_label("H, Z production" if show_muons else "Higgs decay",
                    Vector(PV) + Vector((1.3, 0.0, -1.0)), 0.5, "Lab",
                    "lbl_pv", align='LEFT')

    # Z -> mu mu  (the tag that fixes the Higgs 4-momentum)
    if show_muons:
        for key, p4, msym in (("mu_plus", lab["mu_plus_p4"], "μ⁺"),
                              ("mu_minus", lab["mu_minus_p4"], "μ⁻")):
            d = p3(p4).normalized()
            arrow(PV, d, 6.0, 0.045, m_mu, key, "Lab")
            billboard_label(msym, Vector(PV) + d * 6.5, 0.55, "Lab",
                            "lbl_" + key)

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
            dashed_line(dv, dv + ndir * 5.5, 0.03, m_nu, label + "_nu", "Lab",
                        n_dashes=22)
            # nu label mid-line, dropped below it (the line ends near the
            # pi label / frame corner, so the end position would collide)
            billboard_label("ν", dv + ndir * 4.5 + Vector((0, 0, -0.6)),
                            0.45, "Lab", "lbl_nu_" + label)
        else:
            # unresolved displacement: pion direction straight from the PV
            arrow(PV, pdir, 6.5, 0.05, m_pi, label + "_pion", "Lab")

        billboard_label(sym[label][1], Vector(PV) + pdir *
                        (6.9 if not displaced else (dv.length + 5.3)),
                        0.55, "Lab", "lbl_pi_" + label)


def build_measurable():
    """The lab event reduced to what a detector ACTUALLY measures: the four
    charged-particle tracks (mu+ mu- from the Z, pi+ pi- from the taus), the
    primary vertex, and the impact parameters of the pion tracks.  Everything
    that is *inferred* -- the tau flight paths, the decay vertices and the
    neutrinos -- is drawn as faint ghosts so the contrast is explicit."""
    clear("Lab")
    lab = DATA["lab"]
    m_pv = matte_material("vertex", PALETTE["vertex"], roughness=0.4)
    m_beam = matte_material("beam", PALETTE["beam"], roughness=0.7)
    m_mu = matte_material("muon", PALETTE["muon"], roughness=0.5)
    m_reco = matte_material("reco", PALETTE["reco"], roughness=0.5)
    m_ghost = matte_material("ghost", (0.62, 0.64, 0.68, 1.0),
                             roughness=0.85, alpha=0.16)

    cylinder_between(Vector(PV) - BEAM_DIR * 9, Vector(PV) + BEAM_DIR * 9,
                     0.025, m_beam, "beam", "Lab", caps=False)
    billboard_label("e⁺ e⁻ beam", Vector(PV) - BEAM_DIR * 9.7, 0.5,
                    "Lab", "lbl_beam", align='RIGHT')

    sphere(PV, 0.22, m_pv, "pv", "Lab")
    # anchored left of the vertex reading outward, clear of the mu+ track
    # (which exits lower-left through a centred label position)
    billboard_label("primary vertex", Vector(PV) - BEAM_DIR * 2.4
                    + Vector((0, 0, -0.8)), 0.46, "Lab", "lbl_pv",
                    align='RIGHT')

    for key, p4, msym in (("mu_plus", lab["mu_plus_p4"], "μ⁺"),
                          ("mu_minus", lab["mu_minus_p4"], "μ⁻")):
        d = p3(p4).normalized()
        arrow(PV, d, 6.0, 0.045, m_mu, key, "Lab")
        billboard_label(msym, Vector(PV) + d * 6.5, 0.55, "Lab", "lbl_" + key)

    sym = {"tau_minus": ("τ⁻", "π⁻"), "tau_plus": ("τ⁺", "π⁺")}
    sup = {"tau_minus": "⁻", "tau_plus": "⁺"}
    # split the two d labels vertically; tau+ low enough that the back-
    # extension of its pion track (lowest point z ~ -1.8) cannot cross it
    dside = {"tau_minus": 1.4, "tau_plus": -2.4}
    for label in ("tau_minus", "tau_plus"):
        t = DATA["taus"][label]
        m_pi = matte_material("pi_" + label, PALETTE[PIONCOL[label]],
                              roughness=0.5)
        pdir = rdir(t["pion_dir"])
        dv = bu(t["decay_vertex_reco_um"])
        pca = bu(t["pca_point_um"])
        ndir = p3(t["neutrino_p4_reco"]).normalized()

        # --- inferred (ghosts): tau flight, decay vertex, neutrino ---
        dashed_line(PV, dv, 0.03, m_ghost, label + "_flight", "Lab")
        sphere(dv, 0.12, m_ghost, label + "_dv", "Lab")
        dashed_line(dv, dv + ndir * 5.5, 0.025, m_ghost, label + "_nu", "Lab",
                    n_dashes=22)

        # --- measured: the pion track (a line) + its impact parameter ---
        s_dv = (dv - pca).dot(pdir)
        cylinder_between(pca - pdir * 2.0, pca + pdir * (s_dv + 5.0), 0.04,
                         m_pi, label + "_track", "Lab")
        billboard_label(sym[label][1] + " track",
                        pca + pdir * (s_dv + 5.5), 0.5, "Lab", "lbl_pi_" + label)
        # impact parameter PV -> PCA (perpendicular to the track)
        cylinder_between(PV, pca, 0.04, m_reco, label + "_d", "Lab")
        sphere(pca, 0.07, m_reco, label + "_pca", "Lab")
        # the two d labels split vertically so they never collide near the PV
        billboard_label("d%s  (measured)" % sup[label],
                        Vector(PV) + Vector((0, 0, dside[label])), 0.42,
                        "Lab", "lbl_d_" + label)

    # one label crediting the ghosts as inferred, near a neutrino
    g = DATA["taus"]["tau_plus"]
    dv = bu(g["decay_vertex_reco_um"])
    ndir = p3(g["neutrino_p4_reco"]).normalized()
    # anchored at the nu line's end reading LEFT, so the wide text cannot
    # cross the (diagonal) measured pi+ track to its right
    billboard_label("ν, τ flight, decay point:  inferred, not measured",
                    dv + ndir * 6.3 + Vector((-1.2, 0, -0.4)), 0.42,
                    "Lab", "lbl_inferred", align='RIGHT')


def build_measure_muons():
    """Highlight the Z -> mu+ mu- measurement that fixes the Higgs 4-momentum.
    The muons are drawn bright; the Higgs recoil p_H = p_beam - p_Z is shown
    as a gold arrow; the tau side (the Higgs decay we are after) is ghosted so
    the slide reads 'measure the muons -> we know the Higgs -> boost into its
    rest frame'."""
    clear("Lab")
    lab = DATA["lab"]
    m_pv = matte_material("vertex", PALETTE["vertex"], roughness=0.4)
    m_beam = matte_material("beam", PALETTE["beam"], roughness=0.7)
    m_mu = matte_material("muon", PALETTE["muon"], roughness=0.5)
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    m_ghost = matte_material("ghost", (0.62, 0.64, 0.68, 1.0),
                             roughness=0.85, alpha=0.16)

    cylinder_between(Vector(PV) - BEAM_DIR * 9, Vector(PV) + BEAM_DIR * 9,
                     0.025, m_beam, "beam", "Lab", caps=False)
    billboard_label("e⁺ e⁻ beam", Vector(PV) - BEAM_DIR * 9.7, 0.5,
                    "Lab", "lbl_beam", align='RIGHT')
    sphere(PV, 0.22, m_pv, "pv", "Lab")
    billboard_label("primary vertex", Vector(PV) - BEAM_DIR * 2.4
                    + Vector((0, 0, -0.8)), 0.46, "Lab", "lbl_pv",
                    align='RIGHT')

    # bright muons -- the measurement
    for key, p4, msym in (("mu_plus", lab["mu_plus_p4"], "μ⁺"),
                          ("mu_minus", lab["mu_minus_p4"], "μ⁻")):
        d = p3(p4).normalized()
        arrow(PV, d, 6.5, 0.055, m_mu, key, "Lab")
        billboard_label(msym, Vector(PV) + d * 7.0, 0.6, "Lab", "lbl_" + key)
    # past the mu+ arrow tip (the Z direction is nearly parallel to mu+, so
    # anywhere along it the label would be crossed by the bright track)
    billboard_label("Z → μ⁺μ⁻  (measured)",
                    Vector(PV) + Vector((-5.1, 0, -5.75)), 0.48,
                    "Lab", "lbl_z")

    # the reconstructed Higgs momentum (recoil against the Z); label lifted
    # off the arrow axis so the arrowhead doesn't point into the text
    pH = p3(lab["p_H"]).normalized()
    arrow(PV, pH, 5.0, 0.06, m_higgs, "pH", "Lab")
    billboard_label("Higgs  (p_H = p_beam − p_Z)",
                    Vector(PV) + pH * 5.6 + Vector((0, 0, 0.7)),
                    0.48, "Lab", "lbl_pH")

    # ghosted tau side (the decay we will study, after the boost)
    for label in ("tau_minus", "tau_plus"):
        t = DATA["taus"][label]
        pdir = rdir(t["pion_dir"])
        dv = bu(t["decay_vertex_reco_um"])
        dashed_line(PV, dv, 0.03, m_ghost, label + "_flight", "Lab")
        dashed_line(dv, dv + pdir * 5.0, 0.028, m_ghost, label + "_pi", "Lab")


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

    # extend the track back past the PCA (near the PV) so d meets the line
    s_pca = (pca - dv).dot(pdir)
    cylinder_between(dv + pdir * (s_pca - 1.8), dv + pdir * 6.5, 0.03, m_pi,
                     "pion_track", "Reco")

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
    billboard_label("PV", Vector(PV) - dhat * 0.8, 0.42, "Reco", "rl_pv")
    billboard_label("τ decay vertex", dv - pdir * 1.4 + dhat * 1.0, 0.42,
                    "Reco", "rl_dv")
    billboard_label("impact parameter  d",
                    pca - pdir * 0.3 + dhat * 1.1, 0.4, "Reco", "rl_d")
    billboard_label("π track (measured)",
                    (pca + dv) * 0.5 + dhat * 0.55, 0.42, "Reco", "rl_pi")
    billboard_label("L = |d| / sin α",
                    (Vector(PV) + dv) * 0.5 - dhat * 0.9, 0.4, "Reco", "rl_L")
    billboard_label("α", dv - taudir * 1.7 - dhat * 0.4, 0.4, "Reco", "rl_a")
    return pdir, dhat


# ===========================================================================
#  Higgs rest frame (taus back-to-back)
# ===========================================================================
def build_rest(show_muons=False):
    clear("Rest")
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    origin = Vector((0, 0, 0))
    sphere(origin, 0.18, m_higgs, "rest_h", "Rest")
    billboard_label("Higgs rest frame", origin + Vector((0, 0, 1.1)), 0.5,
                    "Rest", "restl_h")
    pscale = 3.8 / 60.0
    sym = {"tau_minus": "τ⁻", "tau_plus": "τ⁺"}
    # In the Higgs rest frame the taus are EXACTLY back-to-back with equal
    # momenta (|p| = M_H/2); the independent reconstructions differ only by
    # reco error, so draw them along the common axis with a single magnitude.
    mags = [rv(p3(DATA["taus"][l]["rest_frame"]["tau_p4_reco"])).length
            for l in ("tau_minus", "tau_plus")]
    L = (sum(mags) / 2.0) * pscale
    for label in ("tau_minus", "tau_plus"):
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        pdir = rest_axis() * (1.0 if label == "tau_minus" else -1.0)
        arrow(origin, pdir, L, 0.06, m_tau, "rest_" + label, "Rest")
        p = pdir
        # labels anchored just past each arrow tip but reading back INWARD
        # (tau- below its arrow, tau+ above), so neither runs off the frame
        # edge nor collides with the nearby mu labels
        zoff, anch = ((-0.95, 'RIGHT') if label == "tau_minus"
                      else (0.95, 'LEFT'))
        billboard_label("%s   |p| ≈ M_H/2" % sym[label],
                        origin + p.normalized() * (L + 0.6)
                        + Vector((0, 0, zoff)), 0.42, "Rest",
                        "restl_" + label, align=anch)
    billboard_label("taus emitted back-to-back",
                    origin + Vector((1.6, 0, -2.0)),
                    0.42, "Rest", "restl_b2b")

    # the measured Z -> mu mu that defines the boost (lab directions, shown
    # faint so they read as "the tag we used", then dropped in later diagrams)
    if show_muons:
        m_mu = matte_material("muon", PALETTE["muon"], roughness=0.5)
        for key, p4, msym in (("mu_plus", DATA["lab"]["mu_plus_p4"], "μ⁺"),
                              ("mu_minus", DATA["lab"]["mu_minus_p4"], "μ⁻")):
            d = rv(p3(p4).normalized())
            arrow(origin, d, 3.6, 0.04, m_mu, "rest_" + key, "Rest")
            zoff = 0.45 if d.z >= 0 else -0.45
            billboard_label(msym, origin + d * 4.3 + Vector((0, 0, zoff)),
                            0.42, "Rest", "restl_" + key)
        billboard_label("Z → μ⁺μ⁻ (measured)", origin + Vector((0, 0, 2.0)),
                        0.42, "Rest", "restl_z")


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

    # Render the whole 4K frame as a SINGLE tile.  Multi-tile rendering makes
    # Cycles spill its accumulation buffer to a temporary .exr on disk, which
    # fails on some systems (notably macOS, where the temp dir can be cleaned
    # mid-render): "Error writing tile to file".  The scene peaks at ~3 GB, so
    # one tile fits comfortably in memory/VRAM and avoids the disk round-trip.
    if hasattr(scn.cycles, "use_auto_tile"):
        scn.cycles.use_auto_tile = False
    if hasattr(scn.cycles, "tile_size"):
        scn.cycles.tile_size = 4096

    # Prefer AgX (default since 4.0); fall back quietly if a name differs in a
    # given build rather than forcing 'Filmic', which newer Blenders may drop.
    for vt in ('AgX', 'Filmic', 'Standard'):
        try:
            scn.view_settings.view_transform = vt
            break
        except Exception:
            continue
    for lk in ('AgX - Medium High Contrast', 'None'):
        try:
            scn.view_settings.look = lk
            break
        except Exception:
            continue

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
    # Cycles read motion_blur_position from scene.cycles up to 4.x and from
    # scene.render in 5.x; set it on every owner that exposes it (harmless on
    # the one the active version ignores).
    for owner in (scn.render, scn.cycles):
        if hasattr(owner, "motion_blur_position"):
            owner.motion_blur_position = 'CENTER'
    # Try to produce a single .mp4.  The static RNA enum lists 'FFMPEG' even in
    # builds without FFmpeg, and only the *assignment* fails there, so probe by
    # actually setting it.  If that throws, fall back to a numbered PNG
    # sequence (some macOS 5.x builds ship without FFmpeg).
    have_ffmpeg = True
    try:
        scn.render.image_settings.file_format = 'FFMPEG'
    except (TypeError, RuntimeError):
        have_ffmpeg = False
    if have_ffmpeg:
        scn.render.ffmpeg.format = 'MPEG4'
        scn.render.ffmpeg.codec = 'H264'
        scn.render.ffmpeg.constant_rate_factor = 'HIGH'
        scn.render.filepath = os.path.join(OUT, filename)
        print(f"  -> rendering animation {filename}")
    else:
        # Blender uses its own compiled-in FFmpeg, not the system binary, so a
        # build without it can't write video directly.  Render a numbered PNG
        # sequence into output/<stem>/ and, if a system ffmpeg is on PATH,
        # encode it to output/<stem>.mp4 ourselves.
        stem = os.path.splitext(filename)[0]
        seq_dir = os.path.join(OUT, stem)
        os.makedirs(seq_dir, exist_ok=True)
        scn.render.image_settings.file_format = 'PNG'
        scn.render.image_settings.color_mode = 'RGBA'
        scn.render.filepath = os.path.join(seq_dir, stem + "_####")
        print(f"  -> Blender has no FFmpeg; rendering PNG sequence into "
              f"{seq_dir}/")
        bpy.ops.render.render(animation=True)
        _encode_with_system_ffmpeg(seq_dir, stem, filename)
        return
    bpy.ops.render.render(animation=True)


def _encode_with_system_ffmpeg(seq_dir, stem, filename):
    """Assemble the rendered PNG sequence into output/<filename> using the
    system ffmpeg, if available.  Leaves the frames in place either way."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("  -> system 'ffmpeg' not found on PATH; leaving the PNG "
              f"sequence in {seq_dir}/ (encode it yourself).")
        return
    pattern = os.path.join(seq_dir, stem + "_%04d.png")
    out_mp4 = os.path.join(OUT, filename)
    fps = bpy.context.scene.render.fps
    cmd = [ffmpeg, "-y",
           "-framerate", str(fps),
           "-start_number", str(ARGS["frame_start"]),
           "-i", pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "16", "-movflags", "+faststart",
           out_mp4]
    print(f"  -> encoding {out_mp4} with system ffmpeg")
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"  -> wrote {out_mp4}")
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  -> ffmpeg encode failed ({e}); the PNG frames remain in "
              f"{seq_dir}/")


# ===========================================================================
#  Cameras per shot  (beam horizontal; subjects to one side for slides)
# ===========================================================================
def _cam_event(animated, focus=(0, 0.8, 0.8)):
    # front view from -Y so the beam (X) stays horizontal; pulled back and
    # centred so the whole vertical spray (mu+ down, tau/pi up) stays in frame.
    look = focus
    start = (0, -36, 5.0)
    cam = add_camera("cam_event", location=start, look_at=look, up=(0, 0, 1),
                     lens=35, shift_x=-0.16, fstop=2.8, focus_at=(0, 0, 0.8))
    if animated:
        f0, f1 = _frame_range()
        # gentle truck + rise, kept in the X=0 plane so the beam stays level
        moves = [(f0, (0, -36, 4.0)), (round((f0 + f1) / 2), (0, -34, 5.5)),
                 (f1, (0, -33, 7.0))]
        animate_camera(cam, moves, look, up=(0, 0, 1), focus_at=(0, 0, 0.8))
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


# --- rest-frame cameras ----------------------------------------------------
# In display coordinates the common tau axis IS world-X and the decay planes
# open symmetrically about world-Z (see _rest_M), so simple unrolled cameras
# give a horizontal axis AND a level horizon.  Offsetting the camera purely
# in Y/Z from an on-axis look-at point keeps the view direction perpendicular
# to the axis: every axis-parallel line renders exactly horizontal.

def _cam_rest(animated):
    look = (0, 0, 0)
    start = (0, -22, 4.5)
    cam = add_camera("cam_rest", location=start, look_at=look, up=(0, 0, 1),
                     lens=50, shift_x=-0.15, fstop=2.2)
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=20,
                                       push_in=0.9, lift=-0.8), look,
                       up=(0, 0, 1))
    face_labels_to_camera(cam)
    return cam


def _cam_rest_wide(animated):
    """Whole rest-frame decay in frame (both vertices + pions)."""
    look = (-2.4, 0, 0.4)
    start = (-2.4, -32.0, 5.6)
    cam = add_camera("cam_rest_wide", location=start, look_at=look,
                     up=(0, 0, 1), lens=42, shift_x=-0.04,
                     fstop=3.5, focus_at=(0, 0, 0))
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=14,
                                       push_in=0.93, lift=1.0), look,
                       up=(0, 0, 1), focus_at=(0, 0, 0))
    face_labels_to_camera(cam)
    return cam


def _cam_rest_3q(animated):
    """Oblique 3/4 establishing view: elevated and off the axis so both decay
    arms are visible AND the decay planes read clearly as 3-D surfaces."""
    look = (-1.5, 0, 1.0)
    start = (8.0, -22.0, 12.0)
    cam = add_camera("cam_rest_3q", location=start, look_at=look,
                     up=(0, 0, 1), lens=40, shift_x=-0.08, fstop=4.0,
                     focus_at=(0, 0, 0))
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=14,
                                       push_in=0.93, lift=0.6), look,
                       up=(0, 0, 1), focus_at=(0, 0, 0))
    face_labels_to_camera(cam)
    return cam


def _cam_rest_acop(animated):
    """Half-axial 3/4 view: enough down-the-axis component that the phi arc
    opens up, while the two half-planes still read as 3-D surfaces."""
    look = (-1.5, 0, 1.2)
    start = (15.0, -18.0, 8.0)
    cam = add_camera("cam_rest_acop", location=start, look_at=look,
                     up=(0, 0, 1), lens=40, shift_x=-0.10,
                     fstop=4.0, focus_at=(0, 0, 0))
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=14,
                                       push_in=0.92, lift=0.8), look,
                       up=(0, 0, 1), focus_at=(0, 0, 0))
    face_labels_to_camera(cam)
    return cam


def _cam_rest_axial(animated):
    """Looking down the tau axis: the two decay planes collapse to two rays
    from the centre -- the single angle phi is unmistakable."""
    look = (0, 0, 1.4)                    # centre of the opened 'clock face'
    start = (17.0, 0, 1.4)
    cam = add_camera("cam_rest_axial", location=start, look_at=look,
                     up=(0, 0, 1), lens=48, shift_x=-0.06, fstop=5.6,
                     focus_at=(0, 0, 0))
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=10,
                                       push_in=0.94, lift=0.5), look,
                       up=(0, 0, 1))
    face_labels_to_camera(cam)
    return cam


def _cam_rest_zoom(animated):
    """Zoomed in on the vertex region: the impact parameters."""
    look = (-1.3, 0, 0.15)
    start = (-1.3, -11.3, 1.6)
    cam = add_camera("cam_rest_zoom", location=start, look_at=look,
                     up=(0, 0, 1), lens=46, shift_x=-0.08,
                     fstop=4.5, focus_at=(0, 0, 0))
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=10,
                                       push_in=0.94, lift=0.4), look,
                       up=(0, 0, 1), focus_at=(0, 0, 0))
    face_labels_to_camera(cam)
    return cam


# ===========================================================================
#  Shots
# ===========================================================================
def shot_event(animated=False):
    """The lab event -- same scene as storyboard frame 00, so the still IS
    frame 00 and the animation is its drop-in animated companion."""
    build_event(displaced=True)
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(animated)
    render_animation(cam, "00_lab_event.mp4") if animated \
        else render_still(cam, "00_lab_event")


def shot_reco(animated=False):
    """Appendix deep-dive: the Jeans impact-parameter right triangle for one
    tau in its lab-frame track plane (backup slide for questions)."""
    pdir, dhat = build_reco()
    set_visibility(lab=False, reco=True, rest=False)
    cam = _cam_reco(animated, pdir, dhat)
    render_animation(cam, "A1_reco_triangle.mp4") if animated \
        else render_still(cam, "A1_reco_triangle")


def shot_rest(animated=False):
    """The back-to-back taus (storyboard frame 03).  The still is frame 03
    itself; the animation is a slow orbit of the same scene (the boost MORPH
    animation is shot_boost -> 03_boost_to_rest_frame.mp4)."""
    build_rest()
    set_visibility(lab=False, reco=False, rest=True)
    cam = _cam_rest(animated)
    render_animation(cam, "03_rest_frame_orbit.mp4") if animated \
        else render_still(cam, "03_boost_to_rest_frame")


def shot_planes(animated=False):
    """Rest frame, decay planes + the single acoplanarity angle phi (same
    scene and camera as storyboard frame 06).  Drawn angular-only (pion
    directions from the PV) so each pion lies exactly in its plane."""
    build_rest_scene(displaced=False, show_planes=True, title=False)
    set_visibility(lab=False, reco=False, rest=True)
    cam = _cam_rest_acop(animated)
    render_animation(cam, "06_acoplanarity.mp4") if animated \
        else render_still(cam, "06_acoplanarity")


def shot_planes_axial(animated=False):
    """The same decay planes viewed down the tau axis: the 'clock face'
    (storyboard frame 07)."""
    build_rest_scene(displaced=False, show_planes=True, title=False,
                     particle_labels=False)
    set_visibility(lab=False, reco=False, rest=True)
    cam = _cam_rest_axial(animated)
    render_animation(cam, "07_acoplanarity_axial.mp4") if animated \
        else render_still(cam, "07_acoplanarity_axial")


def shot_event_angular(animated=False):
    """Appendix: the lab event without resolved displacement (pions straight
    from the PV) -- the 'angular analysis only' comparison."""
    build_event(displaced=False)
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(animated)
    render_animation(cam, "A2_event_angular_only.mp4") if animated \
        else render_still(cam, "A2_event_angular_only")


def shot_measurable(animated=False):
    """What a detector actually measures (storyboard frame 01)."""
    build_measurable()
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(animated)
    render_animation(cam, "01_what_is_measured.mp4") if animated \
        else render_still(cam, "01_what_is_measured")


def shot_measure_muons(animated=False):
    """Highlight the Z -> mu mu measurement (storyboard frame 02)."""
    build_measure_muons()
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(animated)
    render_animation(cam, "02_measure_the_muons.mp4") if animated \
        else render_still(cam, "02_measure_the_muons")


def shot_separation(animated=False):
    """THE PAYOFF (storyboard frame 10): the dimension line between the two
    decay vertices -- the spacetime separation the spin correlation is
    measured against."""
    build_rest_scene(displaced=True, title=False, show_sep=True)
    set_visibility(lab=False, reco=False, rest=True)
    cam = _cam_rest_wide(animated)
    render_animation(cam, "10_spacetime_separation.mp4") if animated \
        else render_still(cam, "10_spacetime_separation")


def shot_zoom():
    """Animated dolly from the wide rest-frame decay view down into the
    vertex region where the impact parameters live (motion blur on).  Planes
    shown, so the dolly lands exactly on storyboard frame 08's look."""
    build_rest_scene(displaced=True, show_ip=True, show_planes=True,
                     title=False)
    set_visibility(lab=False, reco=False, rest=True)
    f0, f1 = _frame_range()
    look = (-1.8, 0, 0.3)
    cam = add_camera("cam_zoom", location=(-2.7, -28, 7.5), look_at=look,
                     up=(0, 0, 1), lens=44, shift_x=-0.10, fstop=4.0,
                     focus_at=(0, 0, 0))
    animate_camera(cam, [(f0, (-2.7, -28, 7.5)),
                         (round(f0 + 0.55 * (f1 - f0)), (-2.0, -18, 4.0)),
                         (f1, (-1.3, -11.3, 1.6))],
                   look, up=(0, 0, 1), focus_at=(0, 0, 0))
    face_labels_to_camera(cam)
    render_animation(cam, "08_zoom_to_impact_parameters.mp4")


def shot_boost():
    """Animation: lab-frame tau momenta morph into the back-to-back rest-frame
    configuration, with motion blur on and a slow drifting camera."""
    clear("Rest")
    set_visibility(lab=False, reco=False, rest=True)
    scn = bpy.context.scene
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.45)
    origin = Vector((0, 0, 0))
    sphere(origin, 0.20, m_higgs, "boost_h", "Rest")
    # no caption -- like the stills, the animation carries no narration text
    pscale = 5.0 / 60.0
    f0, f1 = _frame_range()
    # the animation lands on the EXACT back-to-back rest configuration: a
    # single common axis (bisecting the two reco directions) and one shared
    # magnitude, so the two arrows finish perfectly antiparallel.
    pr = {l: p3(DATA["taus"][l]["rest_frame"]["tau_p4_reco"])
          for l in ("tau_minus", "tau_plus")}
    rest_mag = (pr["tau_minus"].length + pr["tau_plus"].length) / 2.0
    rest_ax = (pr["tau_minus"].normalized()
               - pr["tau_plus"].normalized()).normalized()
    p_rest_of = {"tau_minus": rest_ax * rest_mag,
                 "tau_plus": -rest_ax * rest_mag}
    for label in ("tau_minus", "tau_plus"):
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        p_lab = p3(DATA["taus"][label]["tau_p4_reco"])
        p_rest = p_rest_of[label]
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
            _ease_action(o.animation_data.action)

    look, start = (0, 0, 0), (0, -22, 5.0)
    cam = add_camera("cam_boost", location=start, look_at=look, up=(0, 0, 1),
                     lens=50, shift_x=-0.18, fstop=2.2)
    animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=16,
                                   push_in=0.92, lift=0.6), look, up=(0, 0, 1))
    face_labels_to_camera(cam)
    render_animation(cam, "04_boost_to_rest_frame.mp4")


def shot_steps():
    """The pedagogical storyboard, in narrative slide order:

      0. the lab event: mu+ mu- (Z) and tau+ tau- (Higgs)
      1. what a detector actually measures (tracks + PV + impact parameters)
      2. HIGHLIGHT the muons: 'we measure the muons from the Z decay, which
         fixes the Higgs 4-momentum...'
      3. '...and that lets us boost into the Higgs rest frame'
      --- from here on, muons removed; everything is in the Higgs rest frame ---
      4. the Higgs decay in its own frame
      5. the two decay planes hinged on the tau axis + the single angle phi
      6. the same, viewed down the tau axis (the 'clock face' view)
      7. zoom in: the impact parameters of the two pion tracks
      8. d and alpha pin down where each tau decayed
      9. THE PAYOFF: the spacetime separation Δx between the two decays --
         always spacelike for the back-to-back pair -- against which the
         spin correlation (entanglement) is measured

    No step captions and no specific numbers: each frame is a clean, generic
    stand-in carrying only symbolic physics labels, free for the user to
    narrate."""
    # 0. The lab event: the Z (mu mu) and the Higgs (tau tau).
    build_event(displaced=True, show_muons=True)
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(False)
    face_labels_to_camera(cam)
    render_still(cam, "00_lab_event")

    # 1. What is actually measurable: tracks + PV + impact parameters.
    build_measurable()
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(False)
    face_labels_to_camera(cam)
    render_still(cam, "01_what_is_measured")

    # 2. Highlight the muons: the Z -> mu mu measurement fixes the Higgs.
    build_measure_muons()
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(False)
    face_labels_to_camera(cam)
    render_still(cam, "02_measure_the_muons")

    # 3. Boost into the Higgs rest frame.  The muons did their job in frame 2
    #    (they fixed the Higgs momentum); this is a rest-frame view, so they
    #    are gone -- just the two taus, now exactly back-to-back.
    build_rest(show_muons=False)
    set_visibility(lab=False, reco=False, rest=True)
    cam = _cam_rest(False)
    face_labels_to_camera(cam)
    render_still(cam, "03_boost_to_rest_frame")

    # 4. Muons removed: the Higgs decay in its own rest frame.
    build_rest_scene(displaced=True)
    set_visibility(lab=False, reco=False, rest=True)
    render_still(_cam_rest_wide(False), "04_higgs_rest_frame")

    # 5. Each tau decay spans a plane -- shown with the real (displaced)
    #    decays sitting inside the two translucent decay planes.
    build_rest_scene(displaced=True, show_planes=True, title=False)
    set_visibility(lab=False, reco=False, rest=True)
    render_still(_cam_rest_3q(False), "05_decay_planes")

    # 6. The single acoplanarity angle phi between the planes (angular-only,
    #    so each pion lies exactly in its plane).
    build_rest_scene(displaced=False, show_planes=True, title=False)
    set_visibility(lab=False, reco=False, rest=True)
    render_still(_cam_rest_acop(False), "06_acoplanarity")

    # 7. Same planes, looking down the tau axis: one angle, unmistakably.
    #    Rebuilt without the per-particle labels: on-axis text all projects
    #    onto the centre point in this view and would pile up there.
    build_rest_scene(displaced=False, show_planes=True, title=False,
                     particle_labels=False)
    set_visibility(lab=False, reco=False, rest=True)
    render_still(_cam_rest_axial(False), "07_acoplanarity_axial")

    # 8. Zoom in: the impact parameters, sitting inside the decay planes.
    build_rest_scene(displaced=True, show_ip=True, show_planes=True,
                     title=False)
    set_visibility(lab=False, reco=False, rest=True)
    render_still(_cam_rest_zoom(False), "08_impact_parameters")

    # 9. The constraint: d and alpha fix the decay locations (with planes).
    build_rest_scene(displaced=True, show_ip=True, show_L=True,
                     show_planes=True, title=False)
    set_visibility(lab=False, reco=False, rest=True)
    render_still(_cam_rest_3q(False), "09_decay_locations")

    # 10. THE PAYOFF: knowing where each tau decayed gives the spacetime
    #     separation between the two decays -- always spacelike for the
    #     back-to-back pair -- against which the spin correlation is measured.
    build_rest_scene(displaced=True, title=False, show_sep=True)
    set_visibility(lab=False, reco=False, rest=True)
    render_still(_cam_rest_wide(False), "10_spacetime_separation")


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
              "planes": shot_planes, "planes-axial": shot_planes_axial,
              "event-angular": shot_event_angular, "measurable": shot_measurable,
              "measure-muons": shot_measure_muons, "separation": shot_separation}
    anims = {"event-anim": shot_event, "reco-anim": shot_reco,
             "rest-anim": shot_rest, "planes-anim": shot_planes,
             "separation-anim": shot_separation,
             "boost": shot_boost, "zoom": shot_zoom}
    # the four animations that slot straight into the talk, in beat order:
    # 00 lab event orbit, 03 boost morph, 06 acoplanarity orbit, 08 IP dolly
    talk_anims = [lambda: shot_event(True), shot_boost,
                  lambda: shot_planes(True), shot_zoom]

    if s == "steps":
        shot_steps()
    elif s == "all":
        shot_steps()
        for fn in talk_anims:
            fn()
    elif s == "stills":
        for fn in stills.values():
            fn(animated=False)
    elif s == "anims":
        for fn in talk_anims:
            fn()
    elif s == "anims-extra":
        shot_reco(True); shot_rest(True)
        shot_planes_axial(True); shot_separation(True)
    elif s in stills:
        stills[s](animated=False)
    elif s in anims:
        if s in ("boost", "zoom"):
            anims[s]()
        else:
            anims[s](animated=True)
    else:
        print(f"Unknown shot '{s}'")

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "tautau_event.blend"))
    print("Done.")


if __name__ == "__main__":
    main()
