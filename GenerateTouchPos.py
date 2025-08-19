# merge_touch_sites_all_box.py  (ratio-based face assignment)
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
# Helpers: parse touch <site> names → (body)
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
    # unchanged mapping from your original code
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

# ----------------------------
# Geometry readers (from base XML geoms)
# ----------------------------

def find_body_elem(root, body_name_full):
    for b in root.findall(".//body"):
        if b.get("name") == body_name_full:
            return b
    return None

def find_primary_geom_on_body(body_elem):
    for g in body_elem.findall("geom"):
        t = g.get("type", "mesh")
        if t in ("capsule", "box"):
            return g
    return None

def capsule_dims(geom):
    sz = geom.get("size", "")
    parts = [float(x) for x in sz.split()]
    if len(parts) < 2:
        raise ValueError(f"Capsule geom missing size: {sz}")
    return parts[0], parts[1]

def box_dims(geom):
    sz = geom.get("size", "")
    parts = [float(x) for x in sz.split()]
    if len(parts) < 3:
        raise ValueError(f"Box geom missing size: {sz}")
    return parts[0], parts[1], parts[2]

# ----------------------------
# Face coordinate frames
# ----------------------------

def face_layout_for_capsule(face, r, L):
    wx = ALPHA * r
    wy = ALPHA * r
    z_half = L * ALPHA
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
# Grid selection & full-coverage layout (unchanged)
# ----------------------------

def choose_base_grid(N, aspect_t_over_z):
    if N <= 2:
        return (N, 1)
    best = None
    root = int(math.ceil(math.sqrt(N)))
    for nz in range(1, root+3):
        nx = int(math.ceil(N / nz))
        for nx_try in (nx, nx+1):
            ar = nx_try / nz
            cost = abs(ar - aspect_t_over_z) + 0.1*(nx_try*nz - N)
            cand = (cost, nx_try, nz)
            if best is None or cand < best:
                best = cand
    _, nx_base, nz_base = best
    return nx_base, nz_base

def row_distribution(N, nz):
    q, r = divmod(N, nz)
    m = [q]*nz
    for i in range(r):
        m[nz - 1 - i] += 1
    return m

def layout_cover_full(face_layout, N, gap_u=GAP_U, gap_z=GAP_Z, margin=MARGIN):
    if N <= 0:
        return []

    tang_half = max(0.0, face_layout.tang_half - margin)
    ax_half   = max(0.0, face_layout.ax_half   - margin)
    W = 2.0 * tang_half
    H = 2.0 * ax_half

    aspect = (W / H) if H > 1e-9 else 1.0
    nx_base, nz_base = choose_base_grid(N, aspect)

    nz = min(nz_base, N)
    m_per_row = row_distribution(N, nz)

    if nz == 1:
        row_h = H
        row_center_z = [0.0]
    else:
        total_gap_z = gap_z * (nz - 1)
        row_h = (H - total_gap_z) / nz
        z0 = -ax_half + row_h/2.0
        row_center_z = [z0 + i*(row_h + gap_z) for i in range(nz)]

    out = []
    for row_idx, m in enumerate(m_per_row):
        zc = row_center_z[row_idx]
        if m <= 0:
            continue
        if m == 1:
            cell_w = W
            xs = [0.0]
        else:
            total_gap_u = gap_u * (m - 1)
            cell_w = (W - total_gap_u) / m
            x0 = -tang_half + cell_w/2.0
            xs = [x0 + j*(cell_w + gap_u) for j in range(m)]
        half_u = cell_w/2.0
        half_z = row_h/2.0

        for x in xs:
            nx, ny, nz_pos = face_layout.normal_center
            if face_layout.face in (FRONT, BACK):
                pos = (x, ny, zc)
                size = (half_u, T, half_z)
            else:
                pos = (nx, x, zc)
                size = (T, half_u, half_z)

            out.append({"pos": pos, "size": size})

    return out

# ----------------------------
# NEW: Ratio split per body (7:1:1, priority FRONT)
# ----------------------------

def split_counts_7_1_1(N):
    """
    Split N sensors into FRONT:BACK:SIDES = 7:1:1 (total 9 parts).
    Priority for remainders: FRONT first, then BACK, then SIDES.
    SIDES are later split into LEFT/RIGHT ~ evenly.
    Returns dict {'front': nf, 'back': nb, 'sides': ns}
    """
    total = 9
    base_front = (7 * N) // total
    base_back  = (1 * N) // total
    base_side  = (1 * N) // total

    assigned = base_front + base_back + base_side
    rem = N - assigned

    # distribute remainder with priority to FRONT, then BACK, then SIDES
    nf, nb, ns = base_front, base_back, base_side
    order = ['front', 'back', 'sides']
    i = 0
    while rem > 0:
        tgt = order[i % len(order)]
        if tgt == 'front': nf += 1
        elif tgt == 'back': nb += 1
        else: ns += 1
        rem -= 1
        i += 1

    return {'front': nf, 'back': nb, 'sides': ns}

def split_sides_left_right(nsides):
    left = nsides // 2
    right = nsides - left
    return left, right

def assign_faces_by_ratio(site_names_for_body, body_name):
    """
    Given the list of site names that belong to the same body, ignore any face hints,
    and assign faces via 7:1:1 split with priority to FRONT.
    Also enforce your original 'palm'/'lfmetacarpal' special case: FRONT-only.
    Returns list of (site_name, face) in stable order.
    """
    N = len(site_names_for_body)

    # Palm / lfmetacarpal: force all FRONT
    if body_name in ("robot0:palm", "robot0:lfmetacarpal"):
        return [(s, FRONT) for s in site_names_for_body]

    split = split_counts_7_1_1(N)
    n_front = split['front']
    n_back  = split['back']
    n_sides = split['sides']
    n_left, n_right = split_sides_left_right(n_sides)

    out = []
    idx = 0

    # Assign FRONT first
    for _ in range(min(n_front, N - idx)):
        out.append((site_names_for_body[idx], FRONT))
        idx += 1
    # then BACK
    for _ in range(min(n_back, N - idx)):
        out.append((site_names_for_body[idx], BACK))
        idx += 1
    # then LEFT
    for _ in range(min(n_left, N - idx)):
        out.append((site_names_for_body[idx], LEFT))
        idx += 1
    # then RIGHT
    for _ in range(min(n_right, N - idx)):
        out.append((site_names_for_body[idx], RIGHT))
        idx += 1

    # If any remain (shouldn't), put to FRONT (priority)
    while idx < N:
        out.append((site_names_for_body[idx], FRONT))
        idx += 1

    return out

# ----------------------------
# Update (or add) <site> in the base XML
# ----------------------------

def ensure_site(body_elem, site_name):
    site = body_elem.find(f"./site[@name='{site_name}']")
    if site is None:
        site = ET.Element("site", {"name": site_name, "type": "box"})
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

    # 3) Group requested sites by body ONLY (ignore face in names)
    by_body = defaultdict(list)  # body -> [site_name,...]
    unresolved = []
    for s in sensor_sites:
        try:
            b = site_to_body(s)
        except Exception as e:
            unresolved.append((s, f"body: {e}"))
            continue
        by_body[b].append(s)

    # 4) For each body, compute ratio split and lay out per face
    debug_counts = defaultdict(int)
    missing_body = []
    updated = 0

    for body_name, sites in by_body.items():
        body = find_body_elem(base_root, body_name)
        if body is None:
            missing_body.append((body_name, f"Body '{body_name}' not found for sites: {sites[:3]}..."))
            continue

        geom = find_primary_geom_on_body(body)
        if geom is None:
            for s in sites:
                unresolved.append((s, f"no shape geom on {body_name}"))
            continue

        gtype = geom.get("type")
        if gtype == "capsule":
            r, L = capsule_dims(geom)
            face_layout = {
                FRONT: face_layout_for_capsule(FRONT, r, L),
                BACK:  face_layout_for_capsule(BACK,  r, L),
                LEFT:  face_layout_for_capsule(LEFT,  r, L),
                RIGHT: face_layout_for_capsule(RIGHT, r, L),
            }
        elif gtype == "box":
            sx, sy, sz = box_dims(geom)
            face_layout = {
                FRONT: face_layout_for_box(FRONT, sx, sy, sz),
                BACK:  face_layout_for_box(BACK,  sx, sy, sz),
                LEFT:  face_layout_for_box(LEFT,  sx, sy, sz),
                RIGHT: face_layout_for_box(RIGHT, sx, sy, sz),
            }
        else:
            for s in sites:
                unresolved.append((s, f"unsupported geom type {gtype}"))
            continue

        # Assign faces per 7:1:1 with priority FRONT (ignoring names)
        face_assignments = assign_faces_by_ratio(sites, body_name)

        # Bucket sites by assigned face, preserving order
        grouped = defaultdict(list)  # face -> sites
        for site_name, face in face_assignments:
            grouped[face].append(site_name)

        # Layout each face independently using your coverage logic
        for face, site_list in grouped.items():
            # Exception: palm & lfmetacarpal are FRONT-only → skip non-front (safety)
            if body_name in ("robot0:palm", "robot0:lfmetacarpal") and face in (LEFT, RIGHT, BACK):
                continue
            N = len(site_list)
            rects = layout_cover_full(face_layout[face], N, GAP_U, GAP_Z, MARGIN)

            for site_name, spec in zip(site_list, rects):
                site_elem = ensure_site(body, site_name)
                set_site_pose(site_elem, spec["pos"], spec["size"])
                updated += 1

            # If more sites than rects (shouldn't), zero them
            for site_name in site_list[len(rects):]:
                site_elem = ensure_site(body, site_name)
                set_site_pose(site_elem, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
                updated += 1

            debug_counts[(body_name, face)] += N

    # 5) Write out
    ET.indent(base_tree, space="    ")
    base_tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

    # 6) Report
    print("# ---- Merge Touch Sites (box + full-coverage layout w/ 7:1:1 split) ----")
    print(f"Input base:    {base_xml_path}")
    print(f"Input sensors: {sensor_xml_path}")
    print(f"Output file:   {out_xml_path}")
    if debug_counts:
        print("\nPer (body, face) site counts after ratio split:")
        for (b, f), c in sorted(debug_counts.items()):
            print(f"  {b:24s} {f:5s}: {c}")
    if missing_body or unresolved:
        print("\nWarnings:")
        for item in missing_body:
            print(" ", item)
        for s, msg in unresolved:
            print(" ", s, ":", msg)

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add/position box sites on a MuJoCo hand model using a 7:1:1 (front:back:sides) split per body, with full-coverage per face.")
    ap.add_argument("--base", required=True, help="Path to hand_base.xml (body tree, has geoms/sizes)")
    ap.add_argument("--sensors", required=True, help="Path to generated sensors XML (with <touch site='...'>)")
    ap.add_argument("--out", default="hand_with_sites_layout.xml", help="Output XML path")
    args = ap.parse_args()
    merge_sites_with_layout(args.base, args.sensors, args.out)
