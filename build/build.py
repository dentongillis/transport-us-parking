"""v2: geometry measured from a stitched Google Maps satellite mosaic (zoom 19, 0.335 m/px)
by colour segmentation (lot edge, pavement) and line-fitting of the parked units (rows).

Local frame: origin = lat 41.5533, lng -85.8822. x = metres east, y = metres north.
GLB frame:   X = east, Y = up, Z = -north (glTF standard, 1 unit = 1 m).
"""
import json, math, base64, random, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import numpy as np
import trimesh
from shapely.geometry import Polygon, LineString, Point

random.seed(7)
LAT0, LNG0 = 41.5533, -85.8822
M_LAT = 111070.0
M_LNG = 111320.0 * math.cos(math.radians(LAT0))


def ll(x, y):
    return round(LAT0 + y / M_LAT, 7), round(LNG0 + x / M_LNG, 7)


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def ring(geom):
    return [[round(a, 2), round(b, 2)] for a, b in list(geom.exterior.coords)[:-1]]


def path(pts, w):
    return ring(LineString(pts).buffer(w / 2, cap_style=2, join_style=1, resolution=4))


def stadium(cx, cy, rx, ry):
    return ring(LineString([(cx - rx + ry, cy), (cx + rx - ry, cy)]).buffer(ry, resolution=8))


# ---------------- surfaces ----------------
LOT_POLY = [  # measured gravel edge, clockwise from NW
    [-66, 183], [54, 185.5], [60.5, 182.5], [59.5, 120], [59.2, 60], [61.5, -10], [62.5, -40], [63.5, -108.5],
    [35, -111], [23, -111], [22, -143], [10, -143], [9, -128], [-13.5, -128], [-13.5, -104.5], [-30, -104],
    [-30, -83], [-38, -79.5], [-47, -78], [-55, -77.5], [-58, -74], [-56, -50], [-60, -44], [-57, -24],
    [-60, -7], [-58.5, 6.5], [-61, 25], [-58, 45], [-62, 58], [-61, 76], [-63, 93], [-62, 115], [-64, 146], [-65, 170]]
surfaces = [
    dict(id="field_w", kind="field", name="Field (west)", polygon=rect(-300, -84, -81, 292)),
    dict(id="field_e", kind="field", name="Field (east)", polygon=[[77, -112], [300, -112], [300, 480], [130, 480], [130, 262], [77, 262]]),
    dict(id="field_sw", kind="field", name="Field (south-west)", polygon=rect(-145, -290, -8, -207)),
    dict(id="field_se", kind="field", name="Field (south-east)", polygon=rect(82, -290, 125, -207)),
    dict(id="gravel_lot", kind="gravel", name="Storage lot (gravel)", polygon=LOT_POLY),
    dict(id="west_drive", kind="gravel", name="West drive (gravel)",
         polygon=path([(-64, -194), (-64.5, -150), (-69.5, -125), (-66.5, -107), (-58, -100.5), (-29, -100)], 5.5)),
    dict(id="neighbor_drive", kind="gravel", name="Neighbour drive",
         polygon=path([(-108, -194), (-101, -181), (-95.5, -166), (-93, -150)], 4)),
    dict(id="gravel_apron", kind="gravel", name="Gravel apron",
         polygon=[[11, -151], [22, -151], [22, -158], [14, -165], [11, -161]]),
    dict(id="main_drive", kind="pavement", name="Main entrance drive", polygon=[
        [23, -111], [35, -111], [35.5, -140], [36, -166], [41, -182.5], [50.5, -194], [14, -194], [20.5, -184],
        [23, -175]]),
    dict(id="office_lane", kind="pavement", name="Office lane", polygon=rect(-64, -151, 23, -144)),
    dict(id="visitor_parking_n", kind="pavement", name="Visitor parking (north bay)", polygon=rect(-13, -144, 9.5, -138.5)),
    dict(id="visitor_parking_s", kind="pavement", name="Visitor parking (south bay)", polygon=rect(-15, -158, 10, -151)),
    dict(id="office_walk", kind="concrete", name="Office walk", polygon=rect(-16, -129, -13.5, -105.5)),
    dict(id="office_pad", kind="concrete", name="Concrete pad", polygon=rect(-21, -144, -14, -138)),
    dict(id="barn_pad", kind="concrete", name="Barn pad", polygon=rect(-115, -104.5, -84, -100.2)),
    dict(id="cr38", kind="road", name="County Road 38", polygon=rect(-300, -202, 300, -194)),
    dict(id="side_road", kind="road", name="Side road", polygon=rect(73, -290, 79, -202)),
]
# Retention pond: a sunken basin with sloped grass banks, not a raised wall. Three matching rings are lofted:
# the rim at grade, a bench part-way down, and the water's edge.
def basin_ring(cx, cy, rx, ry, wobble=0.0, n=96, e=0.62):
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        k = 1 + wobble * (math.sin(3 * t + 1.0) * 0.6 + math.sin(7 * t + 0.4) * 0.4)
        c, sn = math.cos(t), math.sin(t)
        out.append([round(cx + rx * k * math.copysign(abs(c) ** e, c), 2), round(cy + ry * k * math.copysign(abs(sn) ** e, sn), 2)])
    return out


BASIN_DEPTH, BENCH_DEPTH = 1.3, 0.5
basin_rim = basin_ring(-11, 241.5, 57, 25.5)
basin_bench = basin_ring(-11, 241.5, 53.5, 22.0, 0.012)
basin_shore = basin_ring(-11, 241.5, 50.0, 18.5, 0.025)     # water fills the basin, centred, with even banks
basin_outer = Polygon(basin_rim)
basin = dict(id="retention_basin", name="Retention pond", rim=basin_rim, bench=basin_bench, water=basin_shore,
             bench_depth_m=BENCH_DEPTH, water_depth_below_grade_m=BASIN_DEPTH,
             structures=[[-20, 270.5], [7, 270.5], [-37, 212.5], [27, 212.5]])

# ---------------- buildings ----------------
# roof: "ns"/"ew" = ridge direction, "flat"
buildings = [
    dict(id="office", name="Transport US office", box=[-29, -130, -16, -105], eave=3.7, ridge=5.7, roof="ns",
         wall="wall_grey", roofmat="roof_green"),
    dict(id="office_wing", name="Office west wing", box=[-54, -115.5, -29, -105], eave=3.0, ridge=4.5, roof="ew",
         wall="wall_grey", roofmat="roof_dark"),
    dict(id="office_leanto", name="Lean-to", box=[-50.5, -123, -45, -115.5], eave=2.6, ridge=2.6, roof="flat",
         wall="wall_grey", roofmat="roof_dark"),
    dict(id="gate_shed_w", name="Small shed", box=[10, -110.5, 15.5, -107], eave=2.6, ridge=2.6, roof="flat",
         wall="wall_white", roofmat="wall_white"),
    dict(id="gate_shed_e", name="Small shed", box=[17, -110, 22.5, -106], eave=2.6, ridge=2.6, roof="flat",
         wall="wall_white", roofmat="wall_white"),
    dict(id="barn", name="Barn", box=[-115.2, -100.2, -83.7, -91.1], eave=3.6, ridge=5.6, roof="ew",
         wall="wall_grey", roofmat="roof_grey"),
    dict(id="house", name="House", box=[-45.6, -173.3, -37.2, -162.8], eave=5.4, ridge=7.6, roof="ns",
         wall="wall_grey", roofmat="roof_dark"),
    dict(id="neighbor_bldg", name="Neighbouring building (21875)", box=[-117.2, -175.8, -100.8, -155], eave=4.2,
         ridge=5.8, roof="ns", wall="wall_white", roofmat="wall_white"),
    dict(id="neighbor_leanto", name="Lean-to", box=[-100.8, -169, -98, -156], eave=2.8, ridge=2.8, roof="flat",
         wall="wall_grey", roofmat="roof_dark"),
    dict(id="house_south", name="House (south of CR 38)", box=[32.6, -240, 52, -225.6], eave=3.2, ridge=5.4,
         roof="ew", wall="wall_grey", roofmat="roof_grey"),
]
grain_bin = dict(id="grain_bin", name="Grain bin", x=-54.7, y=-118.6, radius=3.7, wall_height=4.6, roof_height=2.2)

# ---------------- trees ----------------
def scatter_line(x0, y0, x1, y1, step, jitter, r, h):
    n = int(math.hypot(x1 - x0, y1 - y0) / step)
    return [[round(x0 + (x1 - x0) * i / n + random.uniform(-jitter, jitter), 1),
             round(y0 + (y1 - y0) * i / n + random.uniform(-jitter, jitter), 1),
             round(random.uniform(*r), 1), round(random.uniform(*h), 1)] for i in range(n + 1)]


trees = dict(
    west_hedge=scatter_line(-76, -82, -78, 290, 7, 1.2, (2.2, 3.6), (5, 8)),
    east_hedge=scatter_line(67, -106, 64, 262, 9, 1.0, (1.6, 2.8), (3.5, 6)),
    yard=[[-58, -176, 5, 11], [-48, -181, 7, 14], [-38, -180, 6, 12], [-13, -171, 5, 10], [-17, -178, 4, 9],
          [-33, -159, 3, 7], [-83, -93, 3, 7], [-120, -108, 3, 6], [14, -232, 6, 12], [58, -228, 6, 12],
          [22, -262, 6, 12], [60, -268, 6, 12], [-6, -225, 3, 7], [-6, -240, 3, 7], [-6, -255, 3, 7]],
    woods=[[round(x + random.uniform(-3, 3), 1), round(y + random.uniform(-3, 3), 1), round(random.uniform(5, 7.5), 1),
            round(random.uniform(12, 17), 1)] for y in range(298, 345, 9) for x in range(-95, 126, 9)]
          + [[round(x + random.uniform(-3, 3), 1), round(y + random.uniform(-3, 3), 1), round(random.uniform(5, 7), 1),
              round(random.uniform(12, 16), 1)] for y in range(268, 296, 9) for x in range(66, 126, 9)],
)

# ---------------- parking rows: one stall per unit detected in the zoom-20 imagery ----------------
_st = json.load(open(os.path.join(HERE, "stalls.json")))
slots = _st["slots"]
for s_ in slots:
    s_["lat"], s_["lng"] = ll(s_["x"], s_["y"])
rows = _st["rows"]
aisles = _st["aisles"]

zones = [
    dict(id="Z-WEST", name="West equipment area", polygon=rect(-57, -76, -40, -36)),
    dict(id="Z-FLAT", name="Flatbed / trailer staging", polygon=rect(41, -102, 61, -60)),
]
for z in zones:
    c = Polygon(z["polygon"]).centroid
    z["x"], z["y"] = round(c.x, 2), round(c.y, 2)
    z["lat"], z["lng"] = ll(c.x, c.y)

imagery_units = _st["units"]

lot = dict(
    version=4,
    name="Transport US, LLC - RV storage lot",
    address="21825 County Road 38, Goshen, IN 46526",
    source="Measured from Google Maps satellite imagery (zoom 19, 0.335 m/px mosaic) and Street View, Nov 2026",
    accuracy_note="Lot edge, pavement and rows were measured from the imagery by colour segmentation and line "
                  "fitting; expect about +/- 1-1.5 m plus any offset in Google's own georeferencing. Building "
                  "heights are estimated from Street View. The lot has no painted stalls: this is a proposed organized "
                  "layout. Row centre-lines are the lines fitted to the units parked in the zoom-20 imagery, extended "
                  "the full length of the lot at a uniform 4.0 m spacing and 40 degree angle, with cross aisles left "
                  "open. occupied_in_imagery marks stalls that had a unit within 2.6 m in the photo.",
    units="metres",
    local_frame=dict(origin_lat=LAT0, origin_lng=LNG0, x_axis="east", y_axis="north",
                     metres_per_deg_lat=M_LAT, metres_per_deg_lng=round(M_LNG, 2)),
    glb_frame="X = east, Y = up, Z = -north; node names match ids here (slot_A-01, zone_Z-NORTH, bldg_office, ...)",
    entrance=dict(name="CR 38 entrance", x=30, y=-193, lat=ll(30, -193)[0], lng=ll(30, -193)[1]),
    lot_area_m2=round(Polygon(LOT_POLY).area),
    surfaces=surfaces, basin=basin, buildings=buildings, grain_bin=grain_bin, trees=trees,
    rows=rows, aisles=aisles, zones=zones, slots=slots, imagery_units=imagery_units,
)
json.dump(lot, open(os.path.join(ROOT, "model", "transport_us_lot.json"), "w"), indent=1)

# ================= GLB =================
C = dict(ground=[104, 134, 78], field=[172, 152, 116], gravel=[182, 176, 164], pavement=[122, 124, 128],
         road=[84, 86, 90], concrete=[206, 204, 196], berm=[98, 132, 72], basin_floor=[82, 108, 62],
         water=[62, 112, 138], reeds=[122, 128, 70], wall_grey=[196, 196, 192], wall_white=[236, 236, 232], roof_green=[92, 128, 110],
         roof_dark=[78, 80, 86], roof_grey=[150, 154, 160], roof_brown=[124, 92, 70], bin=[176, 180, 182],
         tree=[58, 100, 56], woods=[44, 86, 46], trunk=[92, 72, 52], slot=[244, 243, 238], zone=[226, 180, 90],
         line=[232, 208, 96], stripe=[236, 236, 230])


def mat(name, rgb, alpha=255):
    lin = [round(255 * ((c / 255 / 12.92) if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4)) for c in rgb]
    return trimesh.visual.material.PBRMaterial(name=name, baseColorFactor=[*lin, alpha], metallicFactor=0.0,
                                               roughnessFactor=0.95,
                                               alphaMode="BLEND" if alpha < 255 else "OPAQUE")


MATS = {k: mat(k, v) for k, v in C.items()}
MATS["zone"] = mat("zone", C["zone"], 120)
for _k in ("berm", "basin_floor"):          # open lofted surfaces: draw both faces
    MATS[_k].doubleSided = True
TO_GLTF = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])   # (x, north, up) -> (X, up, -north)
scene = trimesh.Scene()


def add(name, mesh, material, extras=None):
    mesh = mesh.copy()
    mesh.apply_transform(TO_GLTF)
    mesh.visual = trimesh.visual.TextureVisuals(material=MATS[material])
    if extras:
        mesh.metadata.update(extras)
    scene.add_geometry(mesh, node_name=name, geom_name=name)


def ext(poly, height, base=0.0):
    p = poly if isinstance(poly, Polygon) else Polygon(poly)
    m = trimesh.creation.extrude_polygon(p, height)
    m.apply_translation([0, 0, base])
    return m


def prism(x0, y0, x1, y1, eave, ridge, axis, over=0.45):
    x0, y0, x1, y1 = x0 - over, y0 - over, x1 + over, y1 + over
    if axis == "ns":
        xm = (x0 + x1) / 2
        v = [[x0, y0, eave], [x1, y0, eave], [xm, y0, ridge], [x0, y1, eave], [x1, y1, eave], [xm, y1, ridge]]
    else:
        ym = (y0 + y1) / 2
        v = [[x0, y0, eave], [x0, y1, eave], [x0, ym, ridge], [x1, y0, eave], [x1, y1, eave], [x1, ym, ridge]]
    f = [[0, 1, 2], [3, 5, 4], [0, 2, 5], [0, 5, 3], [1, 4, 5], [1, 5, 2], [0, 3, 4], [0, 4, 1]]
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    trimesh.repair.fix_normals(m)
    return m


def tree_mesh(items, trunk=True):
    crowns, trunks = [], []
    for x, y, r, h in items:
        s = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        s.apply_scale([r, r, max(h - r * 0.9, r) * 0.62])
        s.apply_translation([x, y, h - max(h - r * 0.9, r) * 0.62])
        crowns.append(s)
        if trunk:
            t = trimesh.creation.cylinder(radius=0.3, height=h * 0.5, sections=6)
            t.apply_translation([x, y, h * 0.25])
            trunks.append(t)
    return trimesh.util.concatenate(crowns), (trimesh.util.concatenate(trunks) if trunks else None)


# ground + surfaces
add("ground", ext(Polygon(rect(-300, -420, 300, 480)).difference(basin_outer), 0.5, -0.5), "ground")
Z = dict(field=0.01, gravel=0.03, pavement=0.06, concrete=0.09, road=0.06)
for s in surfaces:
    add("surf_" + s["id"], ext(s["polygon"], 0.04, Z[s["kind"]]), s["kind"])
# road markings
add("cr38_centre_line", ext(rect(-300, -198.12, 300, -197.88), 0.02, 0.1), "line")
add("cr38_edge_n", ext(rect(-300, -194.75, 13, -194.6), 0.02, 0.1), "stripe")
add("cr38_edge_n2", ext(rect(51, -194.75, 300, -194.6), 0.02, 0.1), "stripe")
add("cr38_edge_s", ext(rect(-300, -201.4, 300, -201.25), 0.02, 0.1), "stripe")
# visitor parking stripes
vs = [ext(rect(x - 0.06, -143.8, x + 0.06, -138.8), 0.02, 0.1) for x in np.arange(-13, 9.6, 2.8)] + \
     [ext(rect(x - 0.06, -157.8, x + 0.06, -152.6), 0.02, 0.1) for x in np.arange(-15, 10.1, 2.8)]
add("visitor_stripes", trimesh.util.concatenate(vs), "stripe")

# retention pond: sloped banks lofted between the rings, water at the bottom, reeds on the shore
def loft(r0, z0, r1, z1):
    n = len(r0)
    v = [[x, y, z0] for x, y in r0] + [[x, y, z1] for x, y in r1]
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + i], [j, n + j, n + i]]
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


add("basin_bank_upper", loft(basin_rim, 0.0, basin_bench, -BENCH_DEPTH), "berm")
add("basin_bank_lower", loft(basin_bench, -BENCH_DEPTH, basin_shore, -BASIN_DEPTH + 0.08), "basin_floor")
add("basin_water", ext(Polygon(basin_shore).buffer(0.6), 0.3, -BASIN_DEPTH - 0.2), "water")
_reeds = []
for i in range(0, 96, 2):
    if random.random() < 0.55:
        x, y = basin_shore[i]
        c_ = trimesh.creation.cone(radius=random.uniform(0.5, 0.9), height=random.uniform(1.0, 1.7), sections=5)
        c_.apply_translation([x + random.uniform(-0.6, 0.6), y + random.uniform(-0.6, 0.6), -BASIN_DEPTH + 0.05])
        _reeds.append(c_)
add("basin_reeds", trimesh.util.concatenate(_reeds), "reeds")
add("basin_structures", trimesh.util.concatenate([ext(rect(x - 1.5, y - 2, x + 1.5, y + 2), 0.9) for x, y in
                                                  basin["structures"]]), "concrete")

# buildings
for b in buildings:
    x0, y0, x1, y1 = b["box"]
    add("bldg_" + b["id"], ext(rect(x0, y0, x1, y1), b["eave"]), b["wall"], dict(name=b["name"]))
    if b["roof"] == "flat":
        add("roof_" + b["id"], ext(rect(x0 - .25, y0 - .25, x1 + .25, y1 + .25), 0.2, b["eave"]), b["roofmat"])
    else:
        add("roof_" + b["id"], prism(x0, y0, x1, y1, b["eave"], b["ridge"], b["roof"]), b["roofmat"])
# office details from Street View: doors / sign on the south gable, entry doors on the east side
add("office_sign", ext(rect(-26, -130.12, -20.5, -130.0), 1.3, 2.3), "roof_brown")
add("office_doors", trimesh.util.concatenate([ext(rect(-16.0, y, -15.88, y + 1.1), 2.1) for y in
                                              (-110, -114.5, -121, -126)]), "roof_dark")
g = grain_bin
cyl = trimesh.creation.cylinder(radius=g["radius"], height=g["wall_height"], sections=24)
cyl.apply_translation([g["x"], g["y"], g["wall_height"] / 2])
add("bldg_grain_bin", cyl, "bin")
cone = trimesh.creation.cone(radius=g["radius"] + 0.25, height=g["roof_height"], sections=24)
cone.apply_translation([g["x"], g["y"], g["wall_height"]])
add("roof_grain_bin", cone, "roof_brown")

# trees
for key, m in (("west_hedge", "tree"), ("east_hedge", "tree"), ("yard", "tree"), ("woods", "woods")):
    crown, trunk = tree_mesh(trees[key], trunk=key != "woods")
    add("trees_" + key, crown, m)
    if trunk is not None:
        add("trunks_" + key, trunk, "trunk")

# drive-aisle centre dashes
dashes = []
for a_ in aisles:
    pts = np.array(a_["line"], float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
    for d0 in np.arange(2, cum[-1] - 3, 9.0):
        p0 = np.array([np.interp(d0, cum, pts[:, 0]), np.interp(d0, cum, pts[:, 1])])
        p1 = np.array([np.interp(d0 + 3.5, cum, pts[:, 0]), np.interp(d0 + 3.5, cum, pts[:, 1])])
        dashes.append(ext(LineString([p0, p1]).buffer(0.22, cap_style=2), 0.02, 0.08))
add("aisle_dashes", trimesh.util.concatenate(dashes), "stripe")

# zones + stalls (pickable, one node each)
for z in zones:
    add("zone_" + z["id"], ext(z["polygon"], 0.03, 0.1), "zone", dict(zone_id=z["id"], name=z["name"]))
for s in slots:
    th = math.radians(s["heading_deg"])
    c, sn = math.cos(th), math.sin(th)
    hl, hw = s["length_m"] / 2, s["width_m"] / 2 - 0.12
    pts = [[s["x"] + dx * c - dy * sn, s["y"] + dx * sn + dy * c] for dx, dy in
           [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]]
    add("slot_" + s["id"], ext(pts, 0.03, 0.1), "slot", dict(slot_id=s["id"], row=s["row"], lat=s["lat"], lng=s["lng"]))

GLB = os.path.join(ROOT, "model", "transport_us_lot.glb")
scene.export(GLB)
glb = open(GLB, "rb").read()
print(len(slots), "stalls", {r["id"]: r["slot_count"] for r in rows}, "| lot area m2", lot["lot_area_m2"],
      "| glb kB", len(glb) // 1024)

view = {k: lot[k] for k in ("rows", "aisles", "zones", "slots")}
html = (open(os.path.join(HERE, "template.html")).read()
        .replace("/*__LOT__*/null", json.dumps(view, separators=(",", ":")))
        .replace("__GLB__", base64.b64encode(glb).decode()))
open(os.path.join(ROOT, "app", "index.html"), "w").write(html)
