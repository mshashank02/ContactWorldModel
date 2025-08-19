# merge_touch_sites_all_box.py
import re
import math
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

# ----------------------------
# Config knobs (tweak freely)
# ----------------------------
ALPHA = 0.95     # in-plane shrink to keep a small margin to edges
BETA  = 0.90     # normal offset fraction of radius/half-extent
T     = 0.0025   # half-thickness of each box along surface normal (meters)

# Gaps/margins (in-plane)
GAP_U = 0.0015   # gap between sensors tangentially (X direction on front/back)
GAP_Z = 0.0015   # gap between sensors axially (Z direction)
MARGIN = 0.0005  # small perimeter inset (extra safety margin)

# Face keys
FRONT, BACK, LEFT, RIGHT = "front", "back", "left", "right"

FaceLayout = namedtuple("FaceLayout", "axis tang_half ax_half normal_center face")

# ----------------------------
# Helpers: parse touch <site> names → (body, face)
# ----------------------------

def parse_sensor_sites(sensor_xml_path):
    tree = ET.parse(sensor_xml_path)
    root = tree.getroot()
    sites = []
    for touch in root.findall(".//touch"):
        s = touch.get("site")
        if s:
            sites.append(s)
    return sorted(set(sites))

def site_to_body(site_name):
    # same as your original mapping (name → MuJoCo body)
    if ":" not in site_name:
        raise ValueError(f"Unexpected site name (no prefix): {site_name}")
    prefix, tail = site_name.split(":", 1)
    if not tail.startswith("T_"):
        raise ValueError(f"Site does not start with 'T_': {site_name}")
    tag = tail[2:]

    if tag.startswith("palm"):
        return "robot0:palm"
    if tag.startswith("lfmetacarpal"):
        return "robot0:lfmetacarpal"

    m = re.match(r"(ff|mf|rf|lf)(proximal|middle|tip)", tag)
    if m:
        finger, seg = m.groups()
        seg_body = "distal" if seg == "tip" else seg
        return f"robot0:{finger}{seg_body}"

    m = re.match(r"(th)(proximal|middle|tip)", tag)
    if m:
        thumb, seg = m.groups()
        seg_body = "distal" if seg == "tip" else seg
        return f"robot0:{thumb}{seg_body}"

    if tag.startswith("palm_"): return "robot0:palm"
    if tag.startswith("lfmetacarpal_"): return "robot0:lfmetacarpal"
    m = re.match(r"(ff|mf|rf|lf)(proximal|middle|tip)_", tag)
    if m:
        finger, seg = m.groups()
        seg_body = "distal" if seg == "tip" else seg
        return f"robot0:{finger}{seg_body}"
    m = re.match(r"(th)(proximal|middle|tip)_", tag)
    if m:
        thumb, seg = m.groups()
        seg_body = "distal" if seg == "tip" else seg
        return f"robot0:{thumb}{seg_body}"

    raise ValueError(f"Cannot infer body for site '{site_name}'")

def site_to_face(site_name):
    """
    Heuristic face detection from site name.
    Accepts 'front/back/left/right' anywhere after 'T_'.
    Also supports palm shortcuts like _fl/_fm/_fr which are all 'front'.
    """
    _, tail = site_name.split(":", 1)
    tag = tail[2:].lower()

    if "front" in tag or re.search(r"[_:]f(?:ront)?[lrmb]?$", tag):
        return FRONT
    if "back" in tag:
        return BACK
    if "left" in tag and "back" not in tag and "front" not in tag:
        return LEFT
    if "right" in tag and "back" not in tag and "front" not in tag:
        return RIGHT

    # Palm pattern from your XML: fl/fm/fr -> all front
    if re.search(r"palm_?f[lmr]$", tag):
        return FRONT

    # If nothing matches, assume FRONT as a safe default
    return FRONT

# ----------------------------
# Geometry readers (from base XML geoms)
# ----------------------------

def find_body_elem(root, body_name_full):
    for b in root.findall(".//body"):
        if b.get("name") == body_name_full:
            return b
    return None

def find_primary_geom_on_body(body_elem):
    """
    Grab the first collision geom that encodes the body shape and size.
    We rely on the same class/type you already have in hand_base.xml.
    """
    for g in body_elem.findall("geom"):
        t = g.get("type", "mesh")
        if t in ("capsule", "box"):
            return g
    return None

def capsule_dims(geom):
    """Return (r, L) from MuJoCo capsule half-sizes: size="radius half_length"."""
    sz = geom.get("size", "")
    parts = [float(x) for x in sz.split()]
    if len(parts) < 2:
        raise ValueError(f"Capsule geom missing size: {sz}")
    return parts[0], parts[1]

def box_dims(geom):
    """Return (sx, sy, sz) half-extends for box."""
    sz = geom.get("size", "")
    parts = [float(x) for x in sz.split()]
    if len(parts) < 3:
        raise ValueError(f"Box geom missing size: {sz}")
    return parts[0], parts[1], parts[2]

# ----------------------------
# Face coordinate frames
# ----------------------------

def face_layout_for_capsule(face, r, L):
    """
    Define the local plane & usable half extents for a capsule segment.
    We treat the cylindrical part; sensor boxes sit on a tangent plane at +/-X or +/-Y.
    """
    wx = ALPHA * r
    wy = ALPHA * r
    z_half = L * ALPHA  # slight shrink axially to stay clear of rounded ends

    if face == FRONT:   # -Y
        return FaceLayout(axis="capsule_front", tang_half=wx, ax_half=z_half, normal_center=(0.0, -BETA*r, 0.0), face=FRONT)
    if face == BACK:    # +Y
        return FaceLayout(axis="capsule_back",  tang_half=wx, ax_half=z_half, normal_center=(0.0, +BETA*r, 0.0), face=BACK)
    if face == LEFT:    # -X
        return FaceLayout(axis="capsule_left",  tang_half=wy, ax_half=z_half, normal_center=(-BETA*r, 0.0, 0.0), face=LEFT)
    if face == RIGHT:   # +X
        return FaceLayout(axis="capsule_right", tang_half=wy, ax_half=z_half, normal_center=(+BETA*r, 0.0, 0.0), face=RIGHT)
    raise ValueError(face)

def face_layout_for_box(face, sx, sy, sz):
    """
    Plane & usable half extents for a box face.
    For FRONT/BACK: X is tangential, Z is axial (toward fingertips).
    For LEFT/RIGHT: Y is tangential, Z is axial.
    """
    if face in (FRONT, BACK):
        tang_half = ALPHA * sx
        ax_half   = ALPHA * sz
        y = -BETA*sy if face == FRONT else +BETA*sy
        return FaceLayout(axis="box_fb", tang_half=tang_half, ax_half=ax_half, normal_center=(0.0, y, 0.0), face=face)
    if face in (LEFT, RIGHT):
        tang_half = ALPHA * sy
        ax_half   = ALPHA * sz
        x = -BETA*sx if face == LEFT else +BETA*sx
        return FaceLayout(axis="box_lr", tang_half=tang_half, ax_half=ax_half, normal_center=(x, 0.0, 0.0), face=face)
    raise ValueError(face)

# ----------------------------
# Grid selection & full-coverage layout
# ----------------------------

def choose_base_grid(N, aspect_t_over_z):
    """
    Pick a base grid (nx_base, nz_base) near sqrt(N) and matching face aspect.
    """
    if N <= 2:
        return (N, 1)
    best = None
    # search reasonable factors around sqrt
    root = int(math.ceil(math.sqrt(N)))
    for nz in range(1, root+3):
        nx = int(math.ceil(N / nz))
        # try also nx bumped a bit
        for nx_try in (nx, nx+1):
            # bias shapes to the aspect ratio (tangential : axial)
            # target aspect ≈ nx/nz
            ar = nx_try / nz
            cost = abs(ar - aspect_t_over_z) + 0.1*(nx_try*nz - N)  # small cost for overfill
            cand = (cost, nx_try, nz)
            if best is None or cand < best:
                best = cand
    _, nx_base, nz_base = best
    return nx_base, nz_base

def row_distribution(N, nz):
    """
    Distribute N sensors into nz rows.
    Priority: give remainders to TOP rows (+Z).
    Returns list m_per_row (length nz), ordered bottom..top.
    """
    q, r = divmod(N, nz)
    m = [q]*nz
    # put the extra 1's on the top-most rows
    for i in range(r):
        m[nz - 1 - i] += 1
    return m

def layout_cover_full(face_layout, N, gap_u=GAP_U, gap_z=GAP_Z, margin=MARGIN):
    """
    Full-coverage layout on a rectangle: tangential half 'tang_half', axial half 'ax_half'.
    - Ensure every axial band has at least one sensor (nz = min(nz_base, N)).
    - Within each row, expand sensor widths so the row covers full width.
    - Distal priority: extra sensors go to the top rows (+Z).
    Returns list of dicts with 'pos' and 'size' (half-sizes) for MuJoCo <site>.
    """
    if N <= 0:
        return []

    tang_half = max(0.0, face_layout.tang_half - margin)
    ax_half   = max(0.0, face_layout.ax_half   - margin)
    W = 2.0 * tang_half
    H = 2.0 * ax_half

    # choose base grid from aspect
    aspect = (W / H) if H > 1e-9 else 1.0
    nx_base, nz_base = choose_base_grid(N, aspect)

    # ensure no axial gaps: nz cannot exceed N
    nz = min(nz_base, N)
    m_per_row = row_distribution(N, nz)

    # axial row height covering full H
    if nz == 1:
        row_h = H
        row_center_z = [0.0]
    else:
        total_gap_z = gap_z * (nz - 1)
        row_h = (H - total_gap_z) / nz
        # bottom row center starts at -ax_half + row_h/2
        z0 = -ax_half + row_h/2.0
        row_center_z = [z0 + i*(row_h + gap_z) for i in range(nz)]

    out = []
    # For each row (bottom→top)
    for row_idx, m in enumerate(m_per_row):
        zc = row_center_z[row_idx]
        if m <= 0:
            continue
        if m == 1:
            # one wide sensor covering full width
            cell_w = W
            xs = [0.0]
        else:
            total_gap_u = gap_u * (m - 1)
            cell_w = (W - total_gap_u) / m
            # centers left→right
            x0 = -tang_half + cell_w/2.0
            xs = [x0 + j*(cell_w + gap_u) for j in range(m)]

        # sizes (half-sizes) along (tangential, normal, axial)
        half_u = cell_w/2.0
        half_z = row_h/2.0

        for x in xs:
            # map to XYZ depending on face
            nx, ny, nz_pos = face_layout.normal_center
            if face_layout.face in (FRONT, BACK):
                # tangential = X, axial = Z, normal = Y
                pos = (x, ny, zc)
                size = (half_u, T, half_z)
            else:
                # LEFT/RIGHT: tangential = Y, axial = Z, normal = X
                pos = (nx, x, zc)
                size = (T, half_u, half_z)

            out.append({"pos": pos, "size": size})

    return out

# ----------------------------
# Update (or add) <site> in the base XML
# ----------------------------

def site_exists(root, site_name):
    return root.find(f".//site[@name='{site_name}']") is not None

def ensure_site(body_elem, site_name):
    """Find or create the <site> node under body_elem (before first <body>)."""
    site = body_elem.find(f"./site[@name='{site_name}']")
    if site is None:
        site = ET.Element("site", {"name": site_name, "type": "box"})
        # insert before first child <body> for stability
        children = list(body_elem)
        insert_idx = None
        for i, ch in enumerate(children):
            if ch.tag == "body":
                insert_idx = i
                break
        if insert_idx is None:
            body_elem.append(site)
        else:
            body_elem.insert(insert_idx, site)
    else:
        # force box type
        site.set("type", "box")
    return site

def set_site_pose(site_elem, pos, size):
    site_elem.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    site_elem.set("size", f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}")

# ----------------------------
# Main merge & layout
# ----------------------------

def merge_sites_with_layout(base_xml_path, sensor_xml_path, out_xml_path):
    # 1) Read sensor sites from sensors.xml
    sensor_sites = parse_sensor_sites(sensor_xml_path)

    # 2) Load base and index bodies
    base_tree = ET.parse(base_xml_path)
    base_root = base_tree.getroot()

    # 3) Group requested sites by (body, face)
    grouped = defaultdict(list)  # (body, face) -> [site_name,...]
    unresolved = []
    for s in sensor_sites:
        try:
            b = site_to_body(s)
        except Exception as e:
            unresolved.append((s, f"body: {e}"))
            continue
        f = site_to_face(s)
        grouped[(b, f)].append(s)

    # 4) For each (body,face) compute geometry → place sites
    added, updated, skipped = 0, 0, 0
    missing_body = []
    debug_counts = defaultdict(int)

    for (body_name, face), sites in grouped.items():
        body = find_body_elem(base_root, body_name)
        if body is None:
            missing_body.append((body_name, f"Body '{body_name}' not found for sites: {sites[:3]}..."))
            continue

        geom = find_primary_geom_on_body(body)
        if geom is None:
            unresolved.extend([(s, f"no shape geom on {body_name}")] * len(sites))
            continue

        gtype = geom.get("type")
        # geometry to face layout
        if gtype == "capsule":
            r, L = capsule_dims(geom)
            layout_info = face_layout_for_capsule(face, r, L)
        elif gtype == "box":
            sx, sy, sz = box_dims(geom)
            layout_info = face_layout_for_box(face, sx, sy, sz)
        else:
            unresolved.extend([(s, f"unsupported geom type {gtype}")] * len(sites))
            continue

        # Exception: palm & lfmetacarpal → FRONT only
        if body_name in ("robot0:palm", "robot0:lfmetacarpal") and face in (LEFT, RIGHT, BACK):
            # skip any non-front requests for these bodies
            continue

        N = len(sites)

        # compute layout covering the whole face
        rects = layout_cover_full(layout_info, N, GAP_U, GAP_Z, MARGIN)

        # order: bottom→top rows, left→right inside rows
        # assign 1:1 to the sites in the given stable order
        for site_name, spec in zip(sites, rects):
            site_elem = ensure_site(body, site_name)
            set_site_pose(site_elem, spec["pos"], spec["size"])
            updated += 1

        # If more sites than rects (shouldn't happen), set remainder to zero
        for site_name in sites[len(rects):]:
            site_elem = ensure_site(body, site_name)
            set_site_pose(site_elem, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            updated += 1

        debug_counts[(body_name, face)] = N

    # 5) Write out
    ET.indent(base_tree, space="    ")
    base_tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

    # 6) Report
    print("# ---- Merge Touch Sites (box + full-coverage layout) ----")
    print(f"Input base:    {base_xml_path}")
    print(f"Input sensors: {sensor_xml_path}")
    print(f"Output file:   {out_xml_path}")
    print(f"Groups laid out: {len(debug_counts)}")
    if debug_counts:
        print("\nPer (body, face) site counts:")
        for (b, f), c in sorted(debug_counts.items()):
            print(f"  {b:24s} {f:5s}: {c}")

    if unresolved or missing_body:
        print("\nWarnings:")
        for item in missing_body:
            print(" ", item)
        for s, msg in unresolved:
            print(" ", s, ":", msg)

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add/position box sites for touch sensors on a MuJoCo hand model with full-coverage per face.")
    ap.add_argument("--base", required=True, help="Path to hand_base.xml (body tree, has geoms/sizes)")
    ap.add_argument("--sensors", required=True, help="Path to generated sensors XML (with <touch site='...'>)")
    ap.add_argument("--out", default="hand_with_sites_layout.xml", help="Output XML path")
    args = ap.parse_args()
    merge_sites_with_layout(args.base, args.sensors, args.out)
