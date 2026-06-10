"""
Blender scene builder for the tau-tau entanglement reconstruction event display.

Run head-less inside the Blender docker image, e.g.

    blender -b -P build_scene.py -- --shot all --samples 128

It reads ``data/event.json`` (produced by ``extract_event.py`` from a *real*
Monte-Carlo event) and constructs a pedagogical 3-D diagram of the analysis'
reconstruction method:

  * the e+e- -> ZH -> mu+mu- tau+tau- event in the lab frame (an event display);
  * the Jeans impact-parameter reconstruction geometry for one tau (impact
    parameter, track plane, opening angle alpha, decay length L = |d|/sin a);
  * the boost into the Higgs rest frame, where the two taus become exactly
    back-to-back, each carrying M_H / 2 -- the constraint that, together with
    the per-tau track planes and the tau-mass constraint, pins down *where*
    each tau decayed.

Design rules baked in (from the brief):
  * every material is satin-matte (Principled BSDF, no metalness, real surface
    roughness, a whisper of clearcoat for a slick automotive-matte finish);
  * the whole scene lives inside a softly lit dome -- a giant wraparound
    softbox whose interior gradient is also the clean cyclorama backdrop;
  * cinematic cameras: shallow photographic depth of field (9-blade bokeh)
    and slow ease-in-out arc moves on every animated shot;
  * Cycles, 4K (3840 x 2160), tiled so each still needs exactly two tiles
    (tile_size = 2160  =>  ceil(3840/2160) * ceil(2160/2160) = 2);
  * motion blur on for all animations;
  * cameras compose the subject on the left or right third of the frame so the
    empty side can hold slide text;
  * one consistent colour palette, reused everywhere.

Final renders are intended to run on a GPU box:  pass  --device GPU  to use
OPTIX/CUDA/HIP/METAL automatically.
"""
import bpy
import bmesh
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
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
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

# ---------------------------------------------------------------------------
# Consistent colour palette  (linear-ish sRGB, alpha)
# ---------------------------------------------------------------------------
PALETTE = {
    "higgs":      (1.00, 0.78, 0.28, 1.0),   # gold
    "z":          (0.64, 0.42, 0.95, 1.0),   # violet
    "muon":       (0.25, 0.72, 0.95, 1.0),   # sky blue
    "tau_minus":  (0.13, 0.66, 0.58, 1.0),   # teal
    "tau_plus":   (0.95, 0.55, 0.25, 1.0),   # warm orange
    "pion_minus": (0.45, 0.85, 0.78, 1.0),   # light teal (tau- product)
    "pion_plus":  (0.99, 0.74, 0.45, 1.0),   # light orange (tau+ product)
    "neutrino":   (0.72, 0.76, 0.80, 1.0),   # pale grey (translucent)
    "reco":       (0.96, 0.30, 0.55, 1.0),   # magenta -- the "method" accent
    "beam":       (0.45, 0.50, 0.55, 1.0),   # steel
    "vertex":     (0.97, 0.97, 0.97, 1.0),   # near-white nodes
    "floor":      (0.05, 0.06, 0.08, 1.0),   # very dark slate
    "plane":      (0.96, 0.30, 0.55, 1.0),   # track plane = reco accent
    "text":       (0.92, 0.93, 0.95, 1.0),
}

# colour per tau label, for convenience
TAUCOL = {"tau_minus": "tau_minus", "tau_plus": "tau_plus"}
PIONCOL = {"tau_minus": "pion_minus", "tau_plus": "pion_plus"}


# ---------------------------------------------------------------------------
# Scene reset
# ---------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    # collections we toggle per shot
    for name in ("Lab", "Reco", "Rest", "Labels", "Lights"):
        col = bpy.data.collections.new(name)
        scn.collection.children.link(col)
    return scn


def col(name):
    return bpy.data.collections[name]


def link(obj, collection_name):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col(collection_name).objects.link(obj)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
_MATS = {}


def _set(bsdf, names, value):
    """Set the first matching Principled input (handles 4.x / 3.x naming)."""
    for n in names:
        if n in bsdf.inputs:
            bsdf.inputs[n].default_value = value
            return


def matte_material(name, rgba, roughness=0.6, alpha=None, emission=0.0):
    """Satin-matte Principled material: matte base with a whisper of clearcoat
    so the big soft dome reads as long, soft highlights -- the slick
    'automotive matte' look, without ever going glossy or metallic."""
    key = (name, tuple(rgba), roughness, alpha, emission)
    if key in _MATS:
        return _MATS[key]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b, a = rgba
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    # Soft, broad specular response
    _set(bsdf, ("Specular IOR Level", "Specular"), 0.3)
    # Thin satin clearcoat: catches the dome as one long soft highlight
    _set(bsdf, ("Coat Weight", "Clearcoat"), 0.25)
    _set(bsdf, ("Coat Roughness", "Clearcoat Roughness"), 0.28)
    # Hint of sheen lifts grazing angles like fine velvet-matte paint
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


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------
def sphere(loc, radius, mat, name, collection, subsurf=True):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc,
                                          segments=48, ring_count=24)
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    assign(o, mat)
    link(o, collection)
    return o


def cylinder_between(p0, p1, radius, mat, name, collection, caps=True):
    """A cylinder spanning two points (Blender Vectors), with rounded caps."""
    p0 = Vector(p0); p1 = Vector(p1)
    d = p1 - p0
    length = d.length
    if length < 1e-9:
        length = 1e-6
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length,
                                         location=(p0 + p1) / 2.0, vertices=32)
    o = bpy.context.active_object
    o.name = name
    # orient +Z to direction d
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
    """An arrow (shaft cylinder + cone head) from p0 along `direction`."""
    direction = Vector(direction).normalized()
    head_len = min(length * 0.28, shaft_r * 9.0)
    shaft_len = max(length - head_len, length * 0.4)
    p_shaft_end = Vector(p0) + direction * shaft_len
    p_tip = Vector(p0) + direction * length
    cylinder_between(p0, p_shaft_end, shaft_r, mat, name + "_shaft",
                     collection, caps=False)
    # cone head
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
    """A single-mesh arrow of unit length pointing along +X, origin at the tail.

    Built as one mesh so it can be keyframed with a length-only scale (scale.x)
    and a rotation without shearing -- ideal for an animated momentum vector.
    """
    head_len = 0.26
    shaft_len = 1.0 - head_len
    # shaft cylinder along +X
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=shaft_len,
                                         location=(shaft_len / 2.0, 0, 0),
                                         vertices=32,
                                         rotation=(0, math.radians(90), 0))
    shaft = bpy.context.active_object
    # head cone along +X
    bpy.ops.mesh.primitive_cone_add(radius1=radius * 2.4, radius2=0.0,
                                    depth=head_len,
                                    location=(shaft_len + head_len / 2.0, 0, 0),
                                    vertices=32,
                                    rotation=(0, math.radians(90), 0))
    head = bpy.context.active_object
    # join into one object
    bpy.ops.object.select_all(action='DESELECT')
    shaft.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join()
    o = bpy.context.active_object
    o.name = name
    o.location = (0, 0, 0)
    bpy.ops.object.shade_smooth()
    assign(o, mat)
    link(o, collection)
    return o


def dashed_line(p0, p1, radius, mat, name, collection, n_dashes=14, duty=0.55):
    """A dashed cylinder (for neutrinos / invisible momentum)."""
    p0 = Vector(p0); p1 = Vector(p1)
    d = p1 - p0
    seg = d / n_dashes
    objs = []
    for i in range(n_dashes):
        a = p0 + seg * i
        b = a + seg * duty
        objs.append(cylinder_between(a, b, radius, mat, f"{name}_{i}",
                                     collection, caps=False))
    return objs


def billboard_label(body, loc, size, collection, name="lbl"):
    """A flat text label that always turns to face the active camera."""
    cu = bpy.data.curves.new(name, type='FONT')
    cu.body = body
    cu.size = size
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    o = bpy.data.objects.new(name, cu)
    # self-lit so labels stay crisp and legible against any lighting / slide
    mat = matte_material("text", PALETTE["text"], roughness=0.7, emission=6.0)
    o.data.materials.append(mat)
    col(collection).objects.link(o)
    o.location = loc
    return o


def face_labels_to_camera(cam):
    """Screen-align every text label with the camera via a COPY_ROTATION
    constraint -- matches camera roll and follows animated camera moves, so
    labels stay horizontal and readable in every frame."""
    for o in bpy.data.objects:
        if o.type != 'FONT':
            continue
        for c in list(o.constraints):
            o.constraints.remove(c)
        con = o.constraints.new('COPY_ROTATION')
        con.target = cam


# ---------------------------------------------------------------------------
# Geometry extraction from the event
# ---------------------------------------------------------------------------
def bu(vec_um):
    """um 3-vector (list) -> Blender Vector in BU."""
    return Vector((vec_um[0], vec_um[1], vec_um[2])) * UM_TO_BU


PV = bu(DATA["lab"]["production_vertex_um"])  # = origin


def p3(p4):
    return Vector((p4[1], p4[2], p4[3]))


# ===========================================================================
#  BUILD: lab-frame event display
# ===========================================================================
def build_lab():
    lab = DATA["lab"]
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.5)
    m_vtx = matte_material("vertex", PALETTE["vertex"], roughness=0.45)
    m_beam = matte_material("beam", PALETTE["beam"], roughness=0.7)
    m_z = matte_material("z", PALETTE["z"], roughness=0.55)
    m_mu = matte_material("muon", PALETTE["muon"], roughness=0.55)
    m_nu = matte_material("neutrino", PALETTE["neutrino"], roughness=0.8,
                          alpha=0.32)

    # --- beam line through PV along z ---
    beamlen = 8.0
    cylinder_between(PV + Vector((0, 0, -beamlen)), PV + Vector((0, 0, beamlen)),
                     0.028, m_beam, "beam_axis", "Lab", caps=False)
    billboard_label("e+e- beam", PV + Vector((0, 0, beamlen + 0.7)), 0.5,
                    "Lab", "lbl_beam")

    # --- production / Higgs decay vertex ---
    sphere(PV, 0.22, m_higgs, "higgs_vertex", "Lab")
    billboard_label("H, Z production", PV + Vector((1.2, -0.7, 0.6)), 0.52,
                    "Lab", "lbl_pv")

    # --- Z -> mu+ mu-  (the tag). Long matte tracks along muon momenta. ---
    mu_len = 6.5
    for key, p4, sym in (("mu_plus", lab["mu_plus_p4"], "mu+"),
                         ("mu_minus", lab["mu_minus_p4"], "mu-")):
        d = p3(p4).normalized()
        arrow(PV, d, mu_len, 0.05, m_mu, key, "Lab")
        billboard_label(sym, PV + d * (mu_len + 0.5), 0.5, "Lab", "lbl_" + key)

    # --- taus: flight from PV to decay vertex, then pion track + neutrino ---
    tausym = {"tau_minus": ("tau-", "pi-"), "tau_plus": ("tau+", "pi+")}
    for label in ("tau_minus", "tau_plus"):
        t = DATA["taus"][label]
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        m_pion = matte_material("pion_" + label, PALETTE[PIONCOL[label]],
                                roughness=0.5)

        dv = bu(t["decay_vertex_reco_um"])

        # tau flight path (thick, the "lifetime" segment)
        cylinder_between(PV, dv, 0.075, m_tau, label + "_flight", "Lab")
        # decay vertex node
        sphere(dv, 0.15, m_vtx, label + "_dv", "Lab")
        midt = (Vector(PV) + dv) * 0.5
        billboard_label(tausym[label][0], midt + Vector((0.45, 0.0, 0.0)), 0.5,
                        "Lab", "lbl_" + label)

        # charged pion track: a long straight line from the decay vertex
        pdir = Vector(t["pion_dir"]).normalized()
        pion_len = 6.0
        arrow(dv, pdir, pion_len, 0.045, m_pion, label + "_pion", "Lab")
        billboard_label(tausym[label][1], dv + pdir * (pion_len + 0.5), 0.55,
                        "Lab", "lbl_pi_" + label)

        # neutrino: dashed translucent arrow (carries the missing momentum)
        ndir = p3(t["neutrino_p4_reco"]).normalized()
        dashed_line(dv, dv + ndir * 3.2, 0.03, m_nu, label + "_nu", "Lab")
        arrow(dv + ndir * 3.2, ndir, 0.8, 0.03, m_nu, label + "_nu_tip", "Lab")
        billboard_label("nu", dv + ndir * 4.3, 0.45, "Lab", "lbl_nu_" + label)


# ===========================================================================
#  BUILD: Jeans reconstruction geometry (focus on tau_plus -- large |d|)
# ===========================================================================
def build_reco(label="tau_plus"):
    t = DATA["taus"][label]
    m_reco = matte_material("reco", PALETTE["reco"], roughness=0.5)
    m_plane = matte_material("plane", PALETTE["plane"], roughness=0.7, alpha=0.10)
    m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
    m_pion = matte_material("pion_" + label, PALETTE[PIONCOL[label]], roughness=0.5)
    m_vtx = matte_material("vertex", PALETTE["vertex"], roughness=0.45)
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.5)

    dv = bu(t["decay_vertex_reco_um"])
    pca = bu(t["pca_point_um"])
    pdir = Vector(t["pion_dir"]).normalized()
    taudir = Vector(t["tau_dir"]).normalized()
    dvec = bu(t["impact_param_vec_um"])  # PV -> PCA, perpendicular to pion

    # production vertex
    sphere(PV, 0.14, m_higgs, "reco_pv", "Reco")
    # decay vertex
    sphere(dv, 0.12, m_vtx, "reco_dv", "Reco")
    # tau flight (true direction) PV -> DV
    cylinder_between(PV, dv, 0.045, m_tau, "reco_tau", "Reco")

    # pion track: long line through the decay vertex; it misses the PV by |d|
    pion_back = dv - pdir * 4.0
    pion_fwd = dv + pdir * 6.5
    cylinder_between(pion_back, pion_fwd, 0.03, m_pion, "reco_pion_track", "Reco")
    arrow(dv, pdir, 4.5, 0.03, m_pion, "reco_pion_arrow", "Reco")

    # impact parameter vector d : PV -> PCA (perpendicular to pion track)
    cylinder_between(PV, pca, 0.045, m_reco, "reco_d", "Reco")
    sphere(pca, 0.09, m_reco, "reco_pca", "Reco")

    # --- pedagogical labels ---
    # The reco camera rolls so that pdir is screen-horizontal and dhat is
    # screen-vertical; place labels along those axes for predictable layout.
    dhat0 = dvec.normalized() if dvec.length > 1e-9 else Vector((0, 0, 1))
    d_um = t["impact_param_mag_um"]
    alpha_mrad = t["alpha_rad"] * 1e3
    L_um = t["decay_length_reco_um"]
    billboard_label("PV", PV - dhat0 * 0.8, 0.42, "Reco", "rl_pv")
    billboard_label("tau decay vertex", dv - pdir * 1.4 + dhat0 * 1.0, 0.42,
                    "Reco", "rl_dv")
    billboard_label("impact parameter  d = %.0f um" % d_um,
                    pca - pdir * 0.3 + dhat0 * 1.1, 0.4, "Reco", "rl_d")
    billboard_label("pi track (measured)",
                    (pca + dv) * 0.5 + dhat0 * 0.55, 0.42, "Reco", "rl_pi")
    billboard_label("L = |d| / sin(alpha) = %.0f um" % L_um,
                    (Vector(PV) + dv) * 0.5 - dhat0 * 0.9, 0.4, "Reco", "rl_L")
    billboard_label("alpha = %.0f mrad" % alpha_mrad,
                    dv - taudir * 1.7 - dhat0 * 0.4, 0.38, "Reco", "rl_alpha")

    # the track plane: spanned by pion dir and d_hat, anchored at PV.
    # Rendered as a faint fill + a crisp wireframe outline so it reads as a
    # plane without flooding the frame or hiding the labels.
    dhat = dvec.normalized() if dvec.length > 1e-9 else Vector((0, 0, 1))
    u = pdir
    w = dhat
    s1, s2 = 3.6, 1.9
    center = PV + u * 2.0
    verts = [center + u * a + w * b for (a, b) in
             [(-s1, -s2), (s1, -s2), (s1, s2), (-s1, s2)]]
    # wireframe outline only (a filled quad floods this near-face-on view)
    for i in range(4):
        cylinder_between(verts[i], verts[(i + 1) % 4], 0.016, m_plane,
                         "plane_edge_%d" % i, "Reco", caps=False)
    billboard_label("track plane", verts[3] + w * 0.5 - u * 0.3, 0.34, "Reco",
                    "rl_plane")

    # angle alpha between the pion track and the tau flight -- drawn where they
    # actually cross, at the decay vertex.
    draw_angle_arc(dv, taudir, pdir, radius=1.3, mat=m_reco,
                   name="alpha_arc", collection="Reco", tube=0.03)

    # right-angle marker at the PCA: d is perpendicular to the pion track.
    q = 0.5
    corner = pca - pdir * q
    cylinder_between(corner, corner + pdir * q, 0.014, m_reco, "ra_a", "Reco",
                     caps=False)
    cylinder_between(corner, corner - dhat * q, 0.014, m_reco, "ra_b", "Reco",
                     caps=False)


def draw_angle_arc(apex, dir_a, dir_b, radius, mat, name, collection,
                   segments=48, tube=0.02):
    """Draw a small tube arc between two directions, to denote an angle."""
    a = Vector(dir_a).normalized()
    b = Vector(dir_b).normalized()
    axis = a.cross(b)
    if axis.length < 1e-9:
        return
    axis.normalize()
    total = a.angle(b)
    pts = []
    for i in range(segments + 1):
        ang = total * i / segments
        q = Matrix.Rotation(ang, 4, axis)
        pts.append(Vector(apex) + (q @ a) * radius)
    for i in range(segments):
        cylinder_between(pts[i], pts[i + 1], tube, mat, f"{name}_{i}",
                         collection, caps=False)


# ===========================================================================
#  BUILD: Higgs rest frame  (taus back-to-back, each M_H/2)
# ===========================================================================
def build_rest():
    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.5)
    m_nu = matte_material("neutrino", PALETTE["neutrino"], roughness=0.8, alpha=0.4)

    origin = Vector((0, 0, 0))
    sphere(origin, 0.18, m_higgs, "rest_higgs", "Rest")
    billboard_label("Higgs rest frame", origin + Vector((0.0, 0.0, 1.0)), 0.5,
                    "Rest", "restl_h")

    # scale: map |p| (GeV) to BU
    pscale = 3.8 / 60.0  # ~60 GeV -> 3.8 BU

    sym = {"tau_minus": "tau-", "tau_plus": "tau+"}
    for label in ("tau_minus", "tau_plus"):
        t = DATA["taus"][label]["rest_frame"]
        m_tau = matte_material(label, PALETTE[TAUCOL[label]], roughness=0.5)
        p = p3(t["tau_p4_reco"])
        L = p.length * pscale
        arrow(origin, p.normalized(), L, 0.06, m_tau, "rest_" + label, "Rest")
        pmag = p.length
        billboard_label("%s   |p| = %.0f GeV ~ M_H/2" % (sym[label], pmag),
                        origin + p.normalized() * (L + 0.7), 0.4,
                        "Rest", "restl_" + label)
    billboard_label("taus emitted back-to-back", origin + Vector((0, 0, -1.4)),
                    0.4, "Rest", "restl_b2b")


# ===========================================================================
#  Lighting + world
# ===========================================================================
DOME_RADIUS = 70.0


def build_world_and_lights():
    """A softly lit studio dome enclosing the whole scene.

    The dome is a huge sphere whose interior carries a vertical emission
    gradient -- near-black below the 'horizon', rising to a warm soft white
    overhead. It acts as a giant wraparound softbox: every satin surface
    picks up one long gentle highlight, and the backdrop is a clean,
    seamless cyclorama gradient with no walls, floor, or horizon line.
    Two large area lights add gentle directional modelling on top.
    """
    scn = bpy.context.scene
    world = bpy.data.worlds.new("W")
    scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs[1].default_value = 0.0   # the dome does all the work

    # --- the dome itself ---
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
    # vertical gradient in the sphere's generated coords (Z: 0 bottom, 1 top)
    ramp.color_ramp.interpolation = 'B_SPLINE'
    e = ramp.color_ramp.elements
    e[0].position = 0.0
    e[0].color = (0.010, 0.011, 0.014, 1.0)     # deep slate well below
    e[1].position = 1.0
    e[1].color = (0.85, 0.87, 0.92, 1.0)        # soft cool-white zenith
    mid = ramp.color_ramp.elements.new(0.42)
    mid.color = (0.055, 0.060, 0.075, 1.0)      # long dark sweep at eye level
    hi = ramp.color_ramp.elements.new(0.78)
    hi.color = (0.32, 0.34, 0.40, 1.0)          # gentle rise to the light
    nt.links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], emis.inputs["Color"])
    emis.inputs["Strength"].default_value = 1.6
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    dome.data.materials.append(mat)
    link(dome, "Lights")

    def area(loc, energy, size, rot, color=(1, 1, 1)):
        l = bpy.data.lights.new("area", 'AREA')
        l.energy = energy
        l.size = size
        l.color = color
        o = bpy.data.objects.new("area", l)
        o.location = loc
        o.rotation_euler = rot
        col("Lights").objects.link(o)
        return o

    # key: big warm softbox, high camera-left; rim: cool kicker from behind
    area((16, -18, 20), 14000, 26, (math.radians(42), 0, math.radians(40)),
         color=(1.0, 0.96, 0.90))
    area((-14, 16, 12), 6500, 22, (math.radians(-58), 0, math.radians(-140)),
         color=(0.78, 0.84, 1.0))


# ===========================================================================
#  Cameras  (subject pushed to one side via lens shift -> empty slide area)
# ===========================================================================
def _aim_rotation(location, look_at, up=None):
    """Quaternion that aims a camera at `look_at`, optionally with an explicit
    up vector for roll control (camera looks down -Z, +Y up, +X right)."""
    direction = (Vector(look_at) - Vector(location)).normalized()
    if up is None:
        return direction.to_track_quat('-Z', 'Y')
    f = direction
    right = f.cross(Vector(up)).normalized()
    true_up = right.cross(f).normalized()
    return Matrix((right, true_up, -f)).transposed().to_quaternion()


def add_camera(name, location, look_at, lens=50, shift_x=0.0, shift_y=0.0,
               up=None, fstop=2.8, focus_at=None):
    """A cinematic camera: aimed at `look_at`, shallow DOF focused on
    `focus_at` (defaults to the look_at point)."""
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.shift_x = shift_x
    cam_data.shift_y = shift_y
    cam = bpy.data.objects.new(name, cam_data)
    col("Lights").objects.link(cam)
    cam.location = location
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = _aim_rotation(location, look_at, up)
    # natural, photographic depth of field
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = fstop
    cam_data.dof.aperture_blades = 9          # rounded, filmic bokeh
    focus = Vector(focus_at) if focus_at is not None else Vector(look_at)
    cam_data.dof.focus_distance = (focus - Vector(location)).length
    return cam


def animate_camera(cam, moves, look_at, up=None, focus_at=None):
    """Keyframe a slow cinematic camera move.

    `moves` is a list of (frame, location) pairs; the camera re-aims at
    `look_at` and refocuses on `focus_at` at every key, with ease-in-out
    interpolation between them.
    """
    focus = Vector(focus_at) if focus_at is not None else Vector(look_at)
    for frame, loc in moves:
        loc = Vector(loc)
        cam.location = loc
        cam.rotation_quaternion = _aim_rotation(loc, look_at, up)
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


# ===========================================================================
#  Render settings
# ===========================================================================
def enable_gpu():
    """Try to enable GPU rendering (OPTIX > CUDA > HIP > METAL > ONEAPI).
    Returns the backend name or None if no GPU is available."""
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

    scn.render.resolution_x = 3840
    scn.render.resolution_y = 2160
    scn.render.resolution_percentage = ARGS["res_percent"]
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'

    # Tiling: tile_size = 2160 => exactly 2 tiles for a 3840x2160 frame.
    scn.cycles.use_auto_tile = True
    scn.cycles.tile_size = 2160

    # Filmic-ish view transform for nice matte tonality
    try:
        scn.view_settings.view_transform = 'AgX'
        scn.view_settings.look = 'AgX - Punchy'   # gentle filmic contrast
    except Exception:
        scn.view_settings.view_transform = 'Filmic'
        scn.view_settings.look = 'None'

    # the softly lit dome IS the backdrop -- render it
    scn.render.film_transparent = False
    scn.render.fps = ARGS["fps"]


def set_visibility(lab, reco, rest):
    """Toggle which phase collections render (labels live inside each phase)."""
    mapping = {"Lab": lab, "Reco": reco, "Rest": rest}
    for name, on in mapping.items():
        c = col(name)
        c.hide_render = not on
        c.hide_viewport = not on


# ===========================================================================
#  Shots
# ===========================================================================
def render_still(cam, filename):
    scn = bpy.context.scene
    scn.camera = cam
    scn.render.use_motion_blur = False
    scn.render.image_settings.file_format = 'PNG'
    scn.render.filepath = os.path.join(OUT, filename)
    print(f"  -> rendering {filename}")
    bpy.ops.render.render(write_still=True)


def render_animation(cam, filename):
    """Render the current frame range as an H.264 mp4 with motion blur on."""
    scn = bpy.context.scene
    scn.camera = cam
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


def _frame_range():
    scn = bpy.context.scene
    f0, f1 = ARGS["frame_start"], ARGS["frame_end"]
    scn.frame_start = f0
    scn.frame_end = f1
    return f0, f1


def _arc_moves(f0, f1, center, start, sweep_deg, push_in=0.92, lift=0.6):
    """Camera positions for a slow arc around `center`: orbit by `sweep_deg`
    while drifting closer (push_in < 1) and rising slightly -- the classic
    'product shot' move."""
    center = Vector(center)
    start = Vector(start)
    rel = start - center
    moves = []
    n_keys = 5
    for i in range(n_keys):
        t = i / (n_keys - 1)
        frame = round(f0 + t * (f1 - f0))
        q = Matrix.Rotation(math.radians(sweep_deg) * t, 4, Vector((0, 0, 1)))
        r = (q @ rel) * (1.0 + (push_in - 1.0) * t)
        moves.append((frame, center + r + Vector((0, 0, lift * t))))
    return moves


def _cam_event(animated):
    """Lab-frame event camera; subject on the RIGHT third."""
    look = (0.0, 2.0, 2.0)
    start = (19, -24, 10)
    cam = add_camera("cam_event", location=start, look_at=look,
                     lens=47, shift_x=-0.30, fstop=2.2, focus_at=(0, 0, 0))
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=14),
                       look, focus_at=(0, 0, 0))
    face_labels_to_camera(cam)
    return cam


def _cam_reco(animated):
    """Jeans-geometry camera: rolled so the pion track is screen-horizontal
    and the impact parameter drops vertically (the textbook IP figure).
    Subject on the LEFT."""
    t = DATA["taus"]["tau_plus"]
    dv = bu(t["decay_vertex_reco_um"])
    pdir = Vector(t["pion_dir"]).normalized()
    dhat = bu(t["impact_param_vec_um"]).normalized()
    focus = (Vector(PV) + dv) / 2.0
    n = pdir.cross(dhat).normalized()
    if (focus - PV).dot(n) < 0:
        n = -n
    dist = 20.0
    tilt = math.radians(16)
    loc = focus + n * dist * math.cos(tilt) + dhat * dist * math.sin(tilt)
    cam = add_camera("cam_reco", location=tuple(loc), look_at=tuple(focus),
                     lens=44, shift_x=-0.04, up=dhat, fstop=4.0,
                     focus_at=tuple(Vector(PV)))
    if animated:
        # gentle sweep in the plane spanned by (n, pdir): the triangle
        # geometry stays legible while the parallax brings it to life.
        f0, f1 = _frame_range()
        moves = []
        n_keys = 5
        for i in range(n_keys):
            tt = i / (n_keys - 1)
            ang = math.radians(-8 + 16 * tt)
            d2 = dist * (1.0 - 0.06 * tt)
            l2 = focus + (n * math.cos(ang) + pdir * math.sin(ang)) \
                * d2 * math.cos(tilt) + dhat * d2 * math.sin(tilt)
            moves.append((round(f0 + tt * (f1 - f0)), l2))
        animate_camera(cam, moves, tuple(focus), up=dhat,
                       focus_at=tuple(Vector(PV)))
    face_labels_to_camera(cam)
    return cam


def _cam_rest(animated):
    """Higgs-rest-frame camera; subject on the RIGHT, 3/4 angle."""
    look = (0, 0, 0)
    start = (17, -9, 9)
    cam = add_camera("cam_rest", location=start, look_at=look,
                     lens=52, shift_x=-0.30, fstop=2.0)
    if animated:
        f0, f1 = _frame_range()
        animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=24,
                                       push_in=0.88, lift=-1.2), look)
    face_labels_to_camera(cam)
    return cam


def shot_event(samples, animated=False):
    set_visibility(lab=True, reco=False, rest=False)
    cam = _cam_event(animated)
    if animated:
        render_animation(cam, "01_event_display.mp4")
    else:
        render_still(cam, "01_event_display")


def shot_reco(samples, animated=False):
    set_visibility(lab=False, reco=True, rest=False)
    cam = _cam_reco(animated)
    if animated:
        render_animation(cam, "02_reconstruction_geometry.mp4")
    else:
        render_still(cam, "02_reconstruction_geometry")


def shot_rest(samples, animated=False):
    set_visibility(lab=False, reco=False, rest=True)
    cam = _cam_rest(animated)
    if animated:
        render_animation(cam, "03_higgs_rest_frame.mp4")
    else:
        render_still(cam, "03_higgs_rest_frame")


def shot_boost(samples):
    """Animation: lab-frame tau momenta morph into the rest-frame back-to-back
    configuration, with motion blur on."""
    set_visibility(lab=False, reco=False, rest=False)
    scn = bpy.context.scene

    # Build dedicated, animatable momentum arrows in the Rest collection.
    set_visibility(lab=False, reco=False, rest=True)
    # Clear any existing rest objects to avoid duplication
    for o in list(col("Rest").objects):
        bpy.data.objects.remove(o, do_unlink=True)

    m_higgs = matte_material("higgs", PALETTE["higgs"], roughness=0.5)
    origin = Vector((0, 0, 0))
    sphere(origin, 0.20, m_higgs, "boost_higgs", "Rest")
    billboard_label("boost into the Higgs rest frame",
                    origin + Vector((0, 0, 1.3)), 0.5, "Rest", "bl_title")

    # GeV -> BU; chosen so both lab and rest arrows stay nicely in frame.
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
            d = p.normalized()
            arr.rotation_quaternion = Vector((1, 0, 0)).rotation_difference(d)
            # length-only scale -> arrow grows/shrinks without getting fatter
            arr.scale = (p.length * pscale, 1.0, 1.0)
            arr.keyframe_insert("rotation_quaternion", frame=frame)
            arr.keyframe_insert("scale", frame=frame)

        keyed(f0, p_lab)
        keyed(f1, p_rest)

    # smooth the interpolation
    for o in col("Rest").objects:
        if o.animation_data and o.animation_data.action:
            for fc in o.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.easing = 'EASE_IN_OUT'

    # slow drifting camera underneath the morph (subject on the RIGHT)
    look = (0, 0, 0)
    start = (13, -16, 6)
    cam = add_camera("cam_boost", location=start, look_at=look,
                     lens=55, shift_x=-0.18, fstop=2.5)
    animate_camera(cam, _arc_moves(f0, f1, look, start, sweep_deg=18,
                                   push_in=0.90, lift=0.8), look)
    face_labels_to_camera(cam)
    render_animation(cam, "04_boost_to_rest_frame.mp4")


# ===========================================================================
#  Main
# ===========================================================================
def main():
    reset_scene()
    setup_render(ARGS["samples"])
    build_world_and_lights()
    build_lab()
    build_reco()
    build_rest()

    shot = ARGS["shot"]
    s = ARGS["samples"]
    if shot in ("event", "stills", "all"):
        shot_event(s)
    if shot in ("reco", "stills", "all"):
        shot_reco(s)
    if shot in ("rest", "stills", "all"):
        shot_rest(s)
    if shot in ("event-anim", "anims", "all"):
        shot_event(s, animated=True)
    if shot in ("reco-anim", "anims", "all"):
        shot_reco(s, animated=True)
    if shot in ("rest-anim", "anims", "all"):
        shot_rest(s, animated=True)
    if shot in ("boost", "anims", "all"):
        shot_boost(s)

    # save the .blend for the user to open / tweak
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "tautau_event.blend"))
    print("Done.")


if __name__ == "__main__":
    main()
