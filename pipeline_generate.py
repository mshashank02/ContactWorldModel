#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Tuple
import os, sys, math, argparse, re, shutil, json
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple

# =========================
# util: formatting + paths
# =========================

def _fmt_num(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"

def make_output_names(out_dir: str, Ntotal: int, Rppx: float, Rpt: float) -> Tuple[str, str]:
    """Return (shared_path, robot_path) obeying the required naming scheme."""
    r1 = _fmt_num(Rppx); r2 = _fmt_num(Rpt)
    shared = f"shared_touch_sensors_{Ntotal}_{r1}_{r2}.xml"
    robot  = f"Sensors_withPos_{Ntotal}_{r1}_{r2}.xml"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, shared), os.path.join(out_dir, robot)

def make_candidate_paths(out_root: str, task: str, Ntotal: int, Rppx: float, Rpt: float) -> Dict[str, str]:
    # file tag (unchanged in filenames)
    tag = f"{Ntotal}_{_fmt_num(Rppx)}_{_fmt_num(Rpt)}"
    # directory tag (now includes the task)
    dir_tag = f"{task}_{tag}"

    cand_dir = os.path.join(out_root, dir_tag)
    os.makedirs(cand_dir, exist_ok=True)

    env_base = f"manipulate_{task}_touch_sensors_{tag}.xml"

    return {
        "dir": cand_dir,                                   # e.g., generated/block_90_1_1
        "shared": os.path.join(cand_dir, f"shared_touch_sensors_{tag}.xml"),
        "robot":  os.path.join(cand_dir, f"Sensors_withPos_{tag}.xml"),
        "env":    os.path.join(cand_dir, env_base),
        "env_basename": env_base,
        "tag":    tag,                                     # keep old tag for filenames/metadata
        "dir_tag": dir_tag,                                # new: task-prefixed directory tag
    }


def save_text_with_header(path: str, xml: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if not xml.lstrip().startswith("<?xml"):
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(xml)
        if not xml.endswith("\n"):
            f.write("\n")

# =========================
# 1) SHARED SENSOR BUILDER
# =========================

def _largest_remainder(values, labels, total_int, priority=None) -> Dict[str, int]:
    floors = [math.floor(v) for v in values]
    result = {lab: f for lab, f in zip(labels, floors)}
    leftover = total_int - sum(floors)
    if leftover <= 0:
        return result
    rema = [v - f for v, f in zip(values, floors)]
    pr_rank = {lab: i for i, lab in enumerate(priority)} if priority else {}
    order = list(range(len(labels)))
    order.sort(key=lambda i: (-rema[i], pr_rank.get(labels[i], 10**9), labels[i]))
    for i in order[:leftover]:
        result[labels[i]] += 1
    return result

def _allocate_groups_int(Ap, Apx, At, Ntotal, Rppx, Rpt) -> Dict[str, int]:
    if min(Ap, Apx, At) <= 0 or min(Rppx, Rpt) <= 0 or Ntotal <= 0:
        raise ValueError("Areas, ratios, and Ntotal must be positive.")
    Dp = Ntotal / (Ap + (Apx / Rppx) + (At / Rpt))
    vals = [Dp * Ap, (Dp / Rppx) * Apx, (Dp / Rpt) * At]
    out = _largest_remainder(vals, ["Np", "Npx", "Nt"], Ntotal, priority=["Nt", "Npx", "Np"])
    assert sum(out.values()) == Ntotal
    return out

def _split_palm_area_weighted(Np: int, Ap1: float, Ap2: float) -> Dict[str, int]:
    tot = Ap1 + Ap2
    if tot <= 0: raise ValueError("Ap1 + Ap2 must be > 0")
    return _largest_remainder(
        [Np*(Ap1/tot), Np*(Ap2/tot)],
        ["TS_palm","TS_lfmetacarpal"], Np,
        priority=["TS_palm","TS_lfmetacarpal"]
    )

SEEDS: Dict[str, List[str]] = {
    "TS_palm": [
        "robot0:T_palm_b0","robot0:T_palm_bl","robot0:T_palm_bm","robot0:T_palm_br",
        "robot0:T_palm_fl","robot0:T_palm_fm","robot0:T_palm_fr","robot0:T_palm_b1",
    ],
    "TS_lfmetacarpal": ["robot0:T_lfmetacarpal_front"],
    "TS_ffproximal": [
        "robot0:T_ffproximal_front_left_bottom","robot0:T_ffproximal_front_right_bottom",
        "robot0:T_ffproximal_front_left_top","robot0:T_ffproximal_front_right_top",
        "robot0:T_ffproximal_back_left","robot0:T_ffproximal_back_right",
        "robot0:T_ffproximal_tip",
    ],
    "TS_ffmiddle": [
        "robot0:T_ffmiddle_front_left","robot0:T_ffmiddle_front_right",
        "robot0:T_ffmiddle_back_left","robot0:T_ffmiddle_back_right",
        "robot0:T_ffmiddle_tip",
    ],
    "TS_fftip": [
        "robot0:T_fftip_front_left","robot0:T_fftip_front_right",
        "robot0:T_fftip_back_left","robot0:T_fftip_back_right","robot0:T_fftip_tip",
    ],
    "TS_mfproximal": [
        "robot0:T_mfproximal_front_left_bottom","robot0:T_mfproximal_front_right_bottom",
        "robot0:T_mfproximal_front_left_top","robot0:T_mfproximal_front_right_top",
        "robot0:T_mfproximal_back_left","robot0:T_mfproximal_back_right",
        "robot0:T_mfproximal_tip",
    ],
    "TS_mfmiddle": [
        "robot0:T_mfmiddle_front_left","robot0:T_mfmiddle_front_right",
        "robot0:T_mfmiddle_back_left","robot0:T_mfmiddle_back_right",
        "robot0:T_mfmiddle_tip",
    ],
    "TS_mftip": [
        "robot0:T_mftip_front_left","robot0:T_mftip_front_right",
        "robot0:T_mftip_back_left","robot0:T_mftip_back_right","robot0:T_mftip_tip",
    ],
    "TS_rfproximal": [
        "robot0:T_rfproximal_front_left_bottom","robot0:T_rfproximal_front_right_bottom",
        "robot0:T_rfproximal_front_left_top","robot0:T_rfproximal_front_right_top",
        "robot0:T_rfproximal_back_left","robot0:T_rfproximal_back_right",
        "robot0:T_rfproximal_tip",
    ],
    "TS_rfmiddle": [
        "robot0:T_rfmiddle_front_left","robot0:T_rfmiddle_front_right",
        "robot0:T_rfmiddle_back_left","robot0:T_rfmiddle_back_right",
        "robot0:T_rfmiddle_tip",
    ],
    "TS_rftip": [
        "robot0:T_rftip_front_left","robot0:T_rftip_front_right",
        "robot0:T_rftip_back_left","robot0:T_rftip_back_right","robot0:T_rftip_tip",
    ],
    "TS_lfproximal": [
        "robot0:T_lfproximal_front_left_bottom","robot0:T_lfproximal_front_right_bottom",
        "robot0:T_lfproximal_front_left_top","robot0:T_lfproximal_front_right_top",
        "robot0:T_lfproximal_back_left","robot0:T_lfproximal_back_right",
        "robot0:T_lfproximal_tip",
    ],
    "TS_lfmiddle": [
        "robot0:T_lfmiddle_front_left","robot0:T_lfmiddle_front_right",
        "robot0:T_lfmiddle_back_left","robot0:T_lfmiddle_back_right",
        "robot0:T_lfmiddle_tip",
    ],
    "TS_lftip": [
        "robot0:T_lftip_front_left","robot0:T_lftip_front_right",
        "robot0:T_lftip_back_left","robot0:T_lftip_back_right","robot0:T_lftip_tip",
    ],
    "TS_thproximal": [
        "robot0:T_thproximal_front_left","robot0:T_thproximal_front_right",
        "robot0:T_thproximal_back_left","robot0:T_thproximal_back_right",
        "robot0:T_thproximal_tip",
    ],
    "TS_thmiddle": [
        "robot0:T_thmiddle_front_left","robot0:T_thmiddle_front_right",
        "robot0:T_thmiddle_back_left","robot0:T_thmiddle_back_right",
        "robot0:T_thmiddle_tip",
    ],
    "TS_thtip": [
        "robot0:T_thtip_front_left","robot0:T_thtip_front_right",
        "robot0:T_thtip_back_left","robot0:T_thtip_back_right","robot0:T_thtip_tip",
    ],
}

PHALANX_KEYS = [
    "TS_ffproximal","TS_ffmiddle",
    "TS_mfproximal","TS_mfmiddle",
    "TS_rfproximal","TS_rfmiddle",
    "TS_lfproximal","TS_lfmiddle",
    "TS_thproximal","TS_thmiddle",
]
TIP_KEYS = ["TS_fftip","TS_mftip","TS_rftip","TS_lftip","TS_thtip"]

PREFIX = {k: k.replace("TS_", "robot0:T_") + "_auto" for k in (
    ["TS_palm","TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS
)}

def _names_for_region(region: str, count: int) -> List[Tuple[str, str]]:
    pairs = []
    seeds = SEEDS.get(region, [])
    take = min(count, len(seeds))
    for s in seeds[:take]:
        pairs.append((s.replace("robot0:T_", "robot0:TS_"), s))
    remain = count - take
    if remain > 0:
        base = PREFIX[region]
        for i in range(1, remain + 1):
            site = f"{base}_{i:03d}"
            pairs.append((site.replace("robot0:T_", "robot0:TS_"), site))
    return pairs

def build_sensor_xml_scaled(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2):
    groups = _allocate_groups_int(Ap, Apx, At, Ntotal, Rppx, Rpt)
    Np, Npx, Nt = groups["Np"], groups["Npx"], groups["Nt"]
    palm = _split_palm_area_weighted(Np, Ap1, Ap2)
    phal = _largest_remainder([Npx/len(PHALANX_KEYS)]*len(PHALANX_KEYS), PHALANX_KEYS, Npx)
    tips = _largest_remainder([Nt/len(TIP_KEYS)]*len(TIP_KEYS), TIP_KEYS, Nt)

    desired = {}
    desired.update(palm); desired.update(phal); desired.update(tips)

    sections = [
        ("PALM", ["TS_palm", "TS_lfmetacarpal"]),
        ("FOREFINGER", ["TS_ffproximal","TS_ffmiddle","TS_fftip"]),
        ("MIDDLE FINGER", ["TS_mfproximal","TS_mfmiddle","TS_mftip"]),
        ("RING FINGER", ["TS_rfproximal","TS_rfmiddle","TS_rftip"]),
        ("LITTLE FINGER", ["TS_lfproximal","TS_lfmiddle","TS_lftip"]),
        ("THUMB", ["TS_thproximal","TS_thmiddle","TS_thtip"]),
    ]

    lines = ['<mujoco>', '    <sensor>']
    for title, keys in sections:
        lines.append(f'\n        <!--{title}-->')
        for k in keys:
            n = desired.get(k, 0)
            for touch_name, site_name in _names_for_region(k, n):
                lines.append(f'        <touch name="{touch_name}" site="{site_name}"></touch>')
    lines += ['\n    </sensor>', '</mujoco>']
    xml = "\n".join(lines)

    stats = {"Np": Np, "Npx": Npx, "Nt": Nt, "Ntotal": Ntotal}
    for k in ["TS_palm", "TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS:
        stats[k] = desired.get(k, 0)
    stats["check_sum"] = sum(stats[k] for k in ["TS_palm", "TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS)
    return xml, stats

# ================================
# 2) MERGE & LAYOUT SITES ON BODIES
# ================================

ALPHA = 0.95; BETA = 0.90; T = 0.0025
GAP_U = 0.0015; GAP_Z = 0.0015; MARGIN = 0.0005
FRONT, BACK, LEFT, RIGHT = "front","back","left","right"
FaceLayout = namedtuple("FaceLayout", "axis tang_half ax_half normal_center face")

def parse_sensor_sites(sensor_xml_path):
    root = ET.parse(sensor_xml_path).getroot()
    return sorted({t.get("site") for t in root.findall(".//touch") if t.get("site")})

def site_to_body(site_name):
    if ":" not in site_name: raise ValueError(f"Bad site name: {site_name}")
    _, tail = site_name.split(":", 1)
    if not tail.startswith("T_"): raise ValueError(f"Site must start with T_: {site_name}")
    tag = tail[2:]
    if tag.startswith("palm"): return "robot0:palm"
    if tag.startswith("lfmetacarpal"): return "robot0:lfmetacarpal"

    m = re.match(r"(ff|mf|rf|lf)(proximal|middle|tip)", tag)
    if m: finger, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{finger}{seg_body}"
    m = re.match(r"(th)(proximal|middle|tip)", tag)
    if m: thumb, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{thumb}{seg_body}"

    if tag.startswith("palm_"): return "robot0:palm"
    if tag.startswith("lfmetacarpal_"): return "robot0:lfmetacarpal"
    m = re.match(r"(ff|mf|rf|lf)(proximal|middle|tip)_", tag)
    if m: finger, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{finger}{seg_body}"
    m = re.match(r"(th)(proximal|middle|tip)_", tag)
    if m: thumb, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{thumb}{seg_body}"

    raise ValueError(f"Cannot infer body for site '{site_name}'")

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
    parts = [float(x) for x in geom.get("size","").split()]
    if len(parts) < 2: raise ValueError("Capsule geom missing size")
    return parts[0], parts[1]

def box_dims(geom):
    parts = [float(x) for x in geom.get("size","").split()]
    if len(parts) < 3: raise ValueError("Box geom missing size")
    return parts[0], parts[1], parts[2]

# ---------- NEW: pose / quaternion helpers (for geom→body transform) ----------
def _parse_vec(s: str | None, n: int):
    vals = [float(x) for x in s.split()] if s else []
    vals += [0.0] * (n - len(vals))
    return tuple(vals[:n])

def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )

def _quat_from_axisangle(axisangle):
    x, y, z, ang = axisangle
    half = 0.5 * ang
    s = math.sin(half)
    c = math.cos(half)
    return (c, x*s, y*s, z*s)

def _quat_from_euler_xyz(angles):
    rx, ry, rz = angles
    cx, sx = math.cos(rx/2), math.sin(rx/2)
    cy, sy = math.cos(ry/2), math.sin(ry/2)
    cz, sz = math.cos(rz/2), math.sin(rz/2)
    qx = (cx, sx, 0.0, 0.0)
    qy = (cy, 0.0, sy, 0.0)
    qz = (cz, 0.0, 0.0, sz)
    return _quat_mul(_quat_mul(qx, qy), qz)

def _quat_rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y*vz - z*vy)
    ty = 2.0 * (z*vx - x*vz)
    tz = 2.0 * (x*vy - y*vx)
    vpx = vx + w*tx + (y*tz - z*ty)
    vpy = vy + w*ty + (z*tx - x*tz)
    vpz = vz + w*tz + (x*ty - y*tx)
    return (vpx, vpy, vpz)

def _geom_pose(geom):
    gpos = _parse_vec(geom.get("pos"), 3)
    if geom.get("quat"):
        gquat = _parse_vec(geom.get("quat"), 4)
    elif geom.get("euler"):
        gquat = _quat_from_euler_xyz(_parse_vec(geom.get("euler"), 3))
    elif geom.get("axisangle"):
        gquat = _quat_from_axisangle(_parse_vec(geom.get("axisangle"), 4))
    else:
        gquat = (1.0, 0.0, 0.0, 0.0)
    return gpos, gquat
# ------------------------------------------------------------------------------

def face_layout_for_capsule(face, r, L):
    wx = ALPHA * r; wy = ALPHA * r; z_half = L * ALPHA
    if face == FRONT:   return FaceLayout("capsule_front", wx, z_half, (0.0, -BETA*r, 0.0), FRONT)
    if face == BACK:    return FaceLayout("capsule_back",  wx, z_half, (0.0, +BETA*r, 0.0), BACK)
    if face == LEFT:    return FaceLayout("capsule_left",  wy, z_half, (-BETA*r, 0.0, 0.0), LEFT)
    if face == RIGHT:   return FaceLayout("capsule_right", wy, z_half, (+BETA*r, 0.0, 0.0), RIGHT)
    raise ValueError(face)

def face_layout_for_box(face, sx, sy, sz):
    if face in (FRONT, BACK):
        y = -BETA*sy if face == FRONT else +BETA*sy
        return FaceLayout("box_fb", ALPHA*sx, ALPHA*sz, (0.0, y, 0.0), face)
    if face in (LEFT, RIGHT):
        x = -BETA*sx if face == LEFT else +BETA*sx
        return FaceLayout("box_lr", ALPHA*sy, ALPHA*sz, (x, 0.0, 0.0), face)
    raise ValueError(face)

def choose_base_grid(N, aspect_t_over_z):
    if N <= 2: return (N, 1)
    best = None; root = int(math.ceil(math.sqrt(N)))
    for nz in range(1, root+3):
        nx = int(math.ceil(N / nz))
        for nx_try in (nx, nx+1):
            ar = nx_try / nz
            cost = abs(ar - aspect_t_over_z) + 0.1*(nx_try*nz - N)
            cand = (cost, nx_try, nz)
            if best is None or cand < best: best = cand
    _, nx_base, nz_base = best
    return nx_base, nz_base

def row_distribution(N, nz):
    q, r = divmod(N, nz)
    m = [q]*nz
    for i in range(r):
        m[nz - 1 - i] += 1
    return m

def layout_cover_full(face_layout, N, gap_u=GAP_U, gap_z=GAP_Z, margin=MARGIN):
    if N <= 0: return []
    tang_half = max(0.0, face_layout.tang_half - margin)
    ax_half   = max(0.0, face_layout.ax_half   - margin)
    W = 2.0 * tang_half; H = 2.0 * ax_half
    aspect = (W / H) if H > 1e-9 else 1.0
    nx_base, nz_base = choose_base_grid(N, aspect)
    nz = min(nz_base, N)
    m_per_row = row_distribution(N, nz)

    if nz == 1:
        row_h = H; row_center_z = [0.0]
    else:
        total_gap_z = gap_z * (nz - 1)
        row_h = (H - total_gap_z) / nz
        z0 = -ax_half + row_h/2.0
        row_center_z = [z0 + i*(row_h + gap_z) for i in range(nz)]

    out = []
    for row_idx, m in enumerate(m_per_row):
        zc = row_center_z[row_idx]
        if m <= 0: continue
        if m == 1:
            cell_w = W; xs = [0.0]
        else:
            total_gap_u = gap_u * (m - 1)
            cell_w = (W - total_gap_u) / m
            x0 = -tang_half + cell_w/2.0
            xs = [x0 + j*(cell_w + gap_u) for j in range(m)]
        half_u = cell_w/2.0; half_z = row_h/2.0
        for x in xs:
            nx, ny, nz_pos = face_layout.normal_center
            if face_layout.face in (FRONT, BACK):
                pos = (x, ny, zc); size = (half_u, T, half_z)
            else:
                pos = (nx, x, zc); size = (T, half_u, half_z)
            out.append({"pos": pos, "size": size})
    return out

def split_counts_7_1_1(N):
    total = 9
    nf = (7*N)//total; nb = (1*N)//total; ns = (1*N)//total
    assigned = nf + nb + ns; rem = N - assigned
    order = ['front', 'back', 'sides']; i = 0
    while rem > 0:
        tgt = order[i % len(order)]
        if tgt == 'front': nf += 1
        elif tgt == 'back': nb += 1
        else: ns += 1
        rem -= 1; i += 1
    return {'front': nf, 'back': nb, 'sides': ns}

def split_sides_left_right(nsides):
    left = nsides // 2
    right = nsides - left
    return left, right

def assign_faces_by_ratio(site_names_for_body, body_name):
    N = len(site_names_for_body)
    if body_name in ("robot0:palm", "robot0:lfmetacarpal"):
        return [(s, FRONT) for s in site_names_for_body]
    split = split_counts_7_1_1(N)
    n_front, n_back, n_sides = split['front'], split['back'], split['sides']
    n_left, n_right = split_sides_left_right(n_sides)

    out = []; idx = 0
    for _ in range(min(n_front, N-idx)): out.append((site_names_for_body[idx], FRONT)); idx += 1
    for _ in range(min(n_back,  N-idx)): out.append((site_names_for_body[idx], BACK));  idx += 1
    for _ in range(min(n_left,  N-idx)): out.append((site_names_for_body[idx], LEFT));  idx += 1
    for _ in range(min(n_right, N-idx)): out.append((site_names_for_body[idx], RIGHT)); idx += 1
    while idx < N: out.append((site_names_for_body[idx], FRONT)); idx += 1
    return out

def ensure_site(body_elem, site_name):
    site = body_elem.find(f"./site[@name='{site_name}']")
    if site is None:
        site = ET.Element("site", {"name": site_name, "type": "box"})
        children = list(body_elem)
        insert_idx = None
        for i, ch in enumerate(children):
            if ch.tag == "body":
                insert_idx = i; break
        if insert_idx is None: body_elem.append(site)
        else: body_elem.insert(insert_idx, site)
    else:
        site.set("type", "box")
    return site

def set_site_pose(site_elem, pos, size):
    site_elem.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    site_elem.set("size", f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}")

def merge_sites_with_layout(base_xml_path, sensor_xml_path, out_xml_path):
    sensor_sites = parse_sensor_sites(sensor_xml_path)
    base_tree = ET.parse(base_xml_path); base_root = base_tree.getroot()

    by_body = defaultdict(list); unresolved = []
    for s in sensor_sites:
        try: b = site_to_body(s)
        except Exception as e: unresolved.append((s, f"body: {e}")); continue
        by_body[b].append(s)

    debug_counts = defaultdict(int); missing_body = []; updated = 0

    for body_name, sites in by_body.items():
        body = find_body_elem(base_root, body_name)
        if body is None: missing_body.append((body_name, f"Body not found")); continue
        geom = find_primary_geom_on_body(body)
        if geom is None:
            for s in sites: unresolved.append((s, f"no geom on {body_name}")); continue

        # NEW: capture geom pose (rotation+translation) to transform local→body
        gpos, gquat = _geom_pose(geom)

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
            for s in sites: unresolved.append((s, f"unsupported geom {gtype}")); continue

        face_assignments = assign_faces_by_ratio(sites, body_name)
        grouped = defaultdict(list)
        for site_name, face in face_assignments:
            grouped[face].append(site_name)

        for face, site_list in grouped.items():
            if body_name in ("robot0:palm", "robot0:lfmetacarpal") and face in (LEFT, RIGHT, BACK):
                continue
            N = len(site_list)
            rects = layout_cover_full(face_layout[face], N, GAP_U, GAP_Z, MARGIN)
            for site_name, spec in zip(site_list, rects):
                # transform local (geom frame) → body frame
                p_local = spec["pos"]
                p_rot   = _quat_rotate(gquat, p_local)
                p_body  = (p_rot[0] + gpos[0], p_rot[1] + gpos[1], p_rot[2] + gpos[2])

                site_elem = ensure_site(body, site_name)
                set_site_pose(site_elem, p_body, spec["size"]); updated += 1
            for site_name in site_list[len(rects):]:
                site_elem = ensure_site(body, site_name)
                set_site_pose(site_elem, (0.0,0.0,0.0), (0.0,0.0,0.0)); updated += 1
            debug_counts[(body_name, face)] += N

    ET.indent(base_tree, space="    ")
    os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
    base_tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

    print("# ---- Merge Touch Sites (7:1:1 split + coverage) ----")
    print(f"Input base:    {base_xml_path}")
    print(f"Input sensors: {sensor_xml_path}")
    print(f"Output file:   {out_xml_path}")
    if debug_counts:
        print("\nPer (body, face) site counts:")
        for (b, f), c in sorted(debug_counts.items()):
            print(f"  {b:24s} {f:5s}: {c}")
    if missing_body or unresolved:
        print("\nWarnings:")
        for item in missing_body: print(" ", item)
        for s, msg in unresolved: print(" ", s, ":", msg)

# =======================
# 3) INCLUDE FILE UPDATER
# =======================

def _find_parent_tag(root: ET.Element, child: ET.Element) -> str | None:
    for elem in root.iter():
        for ch in list(elem):
            if ch is child:
                return elem.tag
    return None

def update_includes_by_prefix(tree: ET.ElementTree, new_shared_basename: str, new_robot_basename: str) -> dict:
    """
    Replace current includes for:
      - shared: filename that starts with 'shared_touch_sensors'
      - robot : filename that starts with 'Sensors_withPos'
    Fallbacks:
      - shared: first non-worldbody include
      - robot : first worldbody include
    Returns {'shared_updated', 'robot_updated'}.
    """
    root = tree.getroot()
    includes = [e for e in root.iter() if e.tag == "include"]
    files = [e.attrib.get("file", "") for e in includes]

    shared_idx = next((i for i, f in enumerate(files) if os.path.basename(f).startswith("shared_touch_sensors")), None)
    robot_idx  = next((i for i, f in enumerate(files) if os.path.basename(f).startswith("Sensors_withPos")), None)

    # If either wasn't found, use structure-based fallbacks
    if shared_idx is None or robot_idx is None:
        parents = []
        for elem in root.iter():
            if elem.tag == "include":
                parents.append(_find_parent_tag(root, elem))
        if robot_idx is None:
            for i, tag in enumerate(parents):
                if tag == "worldbody":
                    robot_idx = i; break
        if shared_idx is None:
            for i, tag in enumerate(parents):
                if tag != "worldbody":
                    shared_idx = i; break

    counts = {'shared_updated': 0, 'robot_updated': 0}
    if shared_idx is not None:
        includes[shared_idx].set("file", new_shared_basename)
        counts['shared_updated'] += 1
    if robot_idx is not None:
        includes[robot_idx].set("file", new_robot_basename)
        counts['robot_updated'] += 1
    return counts

def write_standalone_env(template_xml: str, out_env: str, shared_basename: str, robot_basename: str):
    """Copy template and point its includes to the given basenames (placed in the same folder as out_env)."""
    tree = ET.parse(template_xml)
    update_includes_by_prefix(tree, shared_basename, robot_basename)
    os.makedirs(os.path.dirname(out_env), exist_ok=True)
    tree.write(out_env, encoding="utf-8", xml_declaration=True)

# =======================
# 4) HIGH-LEVEL MODES
# =======================

def build_shared_and_robot(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2, base_xml, out_shared, out_robot, force=False):
    """Generate shared sensors XML and robot-with-sites XML."""
    if (not force) and os.path.exists(out_shared) and os.path.exists(out_robot):
        print(f"[SKIP] Using cached files:\n  {out_shared}\n  {out_robot}")
        return
    xml, stats = build_sensor_xml_scaled(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2)
    save_text_with_header(out_shared, xml)
    print(f"[OK] Wrote shared sensors: {out_shared}")
    print(f"     Totals: Np={stats['Np']} Npx={stats['Npx']} Nt={stats['Nt']} (sum {stats['check_sum']})")
    merge_sites_with_layout(base_xml, out_shared, out_robot)
    print(f"[OK] Wrote robot hand with sites: {out_robot}")

def resolve_task_template(task: str, explicit_template: str | None, main_fallback: str | None) -> str:
    """Choose the template XML for the task."""
    if explicit_template:
        return explicit_template
    if main_fallback:
        return main_fallback
    # Default guesses (edit if your repo uses different paths/names)
    defaults = {
        "block": "assets/manipulate_block_touch_sensors.xml",
        "egg":   "assets/manipulate_egg_touch_sensors.xml",
        "pen":   "assets/manipulate_pen_touch_sensors.xml",
    }
    if task not in defaults:
        raise SystemExit(f"ERROR: unknown task {task!r} and no --template given.")
    return defaults[task]

def build_candidate_standalone(
    task: str,
    Ntotal, Rppx, Rpt, Ap, Apx, At, Ap1, Ap2,
    base_xml: str, template_xml: str, out_root: str, force=False
) -> Dict[str, str]:
    """
    No side effects. Returns dict with paths:
      {dir, shared, robot, env, env_basename, tag}
    """
    paths = make_candidate_paths(out_root, task, Ntotal, Rppx, Rpt)
    # 1) shared + robot
    build_shared_and_robot(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2, base_xml, paths["shared"], paths["robot"], force=force)
    # 2) standalone env that includes the basenames
    if force or (not os.path.exists(paths["env"])):
        write_standalone_env(
            template_xml=template_xml,
            out_env=paths["env"],
            shared_basename=os.path.basename(paths["shared"]),
            robot_basename=os.path.basename(paths["robot"]),
        )
        print(f"[OK] Wrote standalone env: {paths['env']}")
    else:
        print(f"[SKIP] Using cached env: {paths['env']}")
    
    # Copy shared assets needed by the standalone environment
    assets_dir = os.path.dirname(base_xml)
    for fname in ("shared.xml", "shared_asset.xml"):
        src = os.path.join(assets_dir, fname)
        dst = os.path.join(paths["dir"], fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Required asset file not found: {src}")
        if force or (not os.path.exists(dst)):
            shutil.copy2(src, dst)
            print(f"[OK] Copied asset: {dst}")
        else:
            print(f"[SKIP] Using cached asset: {dst}")
    # Copy mesh assets referenced by shared_asset.xml
    shared_asset_path = os.path.join(paths["dir"], "shared_asset.xml")
    asset_tree = ET.parse(shared_asset_path)
    mesh_files = {m.get("file") for m in asset_tree.findall(".//mesh") if m.get("file")}

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(base_xml)))
    mesh_src_dir = os.path.join(repo_root, "stls", "hand")
    mesh_dst_dir = os.path.join(out_root, "stls", "hand")
    os.makedirs(mesh_dst_dir, exist_ok=True)

    for mesh in mesh_files:
        src = os.path.join(mesh_src_dir, mesh)
        dst = os.path.join(mesh_dst_dir, os.path.basename(mesh))
        if not os.path.exists(src):
            raise FileNotFoundError(f"Mesh file referenced in shared_asset.xml not found: {src}")
        if force or (not os.path.exists(dst)):
            shutil.copy2(src, dst)
            print(f"[OK] Copied mesh: {dst}")
        else:
            print(f"[SKIP] Using cached mesh: {dst}")

    # Copy texture assets referenced by the standalone env
    env_tree = ET.parse(paths["env"])
    texture_files = {t.get("file") for t in env_tree.findall(".//texture") if t.get("file")}

    texture_src_dir = os.path.join(repo_root, "textures")
    texture_dst_dir = os.path.join(out_root, "textures")
    os.makedirs(texture_dst_dir, exist_ok=True)

    for tex in texture_files:
        src = os.path.join(texture_src_dir, tex)
        dst = os.path.join(texture_dst_dir, os.path.basename(tex))
        if not os.path.exists(src):
            raise FileNotFoundError(f"Texture file referenced in {paths['env']} not found: {src}")
        if force or (not os.path.exists(dst)):
            shutil.copy2(src, dst)
            print(f"[OK] Copied texture: {dst}")
        else:
            print(f"[SKIP] Using cached texture: {dst}")
    return paths

# ==========
# MAIN CLI
# ==========

def main():
    p = argparse.ArgumentParser(
        description="End-to-end: build sensors → layout sites → update main XML (legacy) OR emit standalone per-candidate env (no side effects)."
    )
    p.add_argument("--base",  required=True, help="Path to base hand XML (bodies + geoms), e.g., assets/hand_base.xml")

    # Task selection
    p.add_argument("--task", choices=["block","egg","pen"], default="block",
                   help="Which task template to use when generating a standalone env (manipulate_<task>_touch_sensors.xml).")

    # Legacy/in-place mode
    p.add_argument("--main",  help="Path to main task XML to update includes, e.g., assets/manipulate_block_touch_sensors.xml")
    p.add_argument("--out-dir", default=None, help="Directory to write shared/robot in legacy mode (and update --main includes)")

    # Standalone mode
    p.add_argument("--standalone", action="store_true", help="Generate a per-candidate folder with shared, robot, and a standalone env (no side effects).")
    p.add_argument("--template", help="Template task XML used to create the standalone env. If omitted, a default matching --task is used, or --main if provided.")
    p.add_argument("--out-root", default="generated", help="Root folder for standalone candidates (each under <N>_<r1>_<r2>/).")
    p.add_argument("--force", action="store_true", help="Overwrite/cached outputs for this candidate.")

    # Allocation / areas / ratios
    p.add_argument("--Ntotal", type=int, required=True)
    p.add_argument("--Rppx", type=float, required=True, help="Palm : Phalanx ratio scale")
    p.add_argument("--Rpt",  type=float, required=True, help="Palm : Tip ratio scale")
    p.add_argument("--Ap",   type=float, default= 6557, help="Area weight: Palm")
    p.add_argument("--Apx",  type=float, default=26885, help="Area weight: Phalanx")
    p.add_argument("--At",   type=float, default=7193, help="Area weight: Tips")
    p.add_argument("--Ap1",  type=float, default=5557, help="Palm sub-area 1 (palm)")
    p.add_argument("--Ap2",  type=float, default=1000, help="Palm sub-area 2 (lfmetacarpal)")

    # legacy safety
    p.add_argument("--backup", action="store_true", help="(Legacy mode) Backup main XML to .bak before editing")

    args = p.parse_args()

    # Decide mode
    if args.standalone:
        template_xml = resolve_task_template(args.task, args.template, args.main)
        paths = build_candidate_standalone(
            task=args.task,
            Ntotal=args.Ntotal, Rppx=args.Rppx, Rpt=args.Rpt,
            Ap=args.Ap, Apx=args.Apx, At=args.At, Ap1=args.Ap1, Ap2=args.Ap2,
            base_xml=args.base, template_xml=template_xml,
            out_root=args.out_root, force=args.force
        )
        # Emit a small machine-friendly summary for BO loops
        print(json.dumps(paths, indent=2))
        return

    # Legacy/in-place mode
    if not args.main:
        sys.exit("ERROR: legacy mode requires --main (the task XML to update). Use --standalone for no-side-effects generation.")
    if not args.out_dir:
        sys.exit("ERROR: legacy mode requires --out-dir (where to put shared/robot). Use --standalone to avoid editing files.")

    shared_path, robot_path = make_output_names(args.out_dir, args.Ntotal, args.Rppx, args.Rpt)
    build_shared_and_robot(args.Ap, args.Apx, args.At, args.Ntotal, args.Rppx, args.Rpt, args.Ap1, args.Ap2, args.base, shared_path, robot_path, force=args.force)

    if args.backup:
        with open(args.main, "rb") as src, open(args.main + ".bak", "wb") as dst:
            dst.write(src.read())
        print(f"[OK] Backup saved: {args.main}.bak")

    main_tree = ET.parse(args.main)
    counts = update_includes_by_prefix(
        main_tree,
        new_shared_basename=os.path.basename(shared_path),
        new_robot_basename=os.path.basename(robot_path)
    )
    main_tree.write(args.main, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Updated main includes in: {args.main}")
    print(f"     Replaced: shared={counts['shared_updated']} robot={counts['robot_updated']}")

if __name__ == "__main__":
    main()
