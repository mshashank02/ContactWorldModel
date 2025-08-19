# merge_touch_sites_all_box.py
import re
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

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
    if ":" not in site_name:
        raise ValueError(f"Unexpected site name (no prefix): {site_name}")
    _, tail = site_name.split(":", 1)
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

def site_type_for(site_name):
    return "box"  # All touch sites are boxes

def find_body_elem(root, body_name_full):
    for b in root.findall(".//body"):
        if b.get("name") == body_name_full:
            return b
    return None

def site_exists(root, site_name):
    return root.find(f".//site[@name='{site_name}']") is not None

def append_site(body_elem, site_name):
    """Insert <site> before first <body> child for consistent layout."""
    stype = site_type_for(site_name)
    site = ET.Element("site", {
        "name": site_name,
        "type": stype,
        "pos": "0 0 0",
        "size": "0 0 0"
    })
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

#Sizing portion of the code 

#Tunable params 
ALPHA = 0.90 #tangential coverage
BETA = 0.60 #offset form center on the face
T_MAX = 0.005 # abs cap for face-normal half-thickness (m)
T_FRAC_CAP = 0.20 #capsule 
T_FRAC_BOX = 0.20  # box:     t = min(0.2 * min(sx,sy), T_MAX)
GAP_FRAC  = 0.5    # axial gap relative to t; g = max(GAP_FRAC * t, 0.0005)
Z_MIN     = 0.003  # minimum half-length per band
FRONT_ONLY_BODIES = {"robot0:palm", "robot0:lfmetacarpal"}

def merge_sites(base_xml_path, sensor_xml_path, out_xml_path):
    sensor_sites = parse_sensor_sites(sensor_xml_path)
    base_tree = ET.parse(base_xml_path)
    base_root = base_tree.getroot()

    added = 0
    skipped = 0
    missing_body = []
    per_body_counts = defaultdict(int)

    for s in sensor_sites:
        if site_exists(base_root, s):
            skipped += 1
            continue
        try:
            body_name = site_to_body(s)
        except ValueError as e:
            missing_body.append((s, str(e)))
            continue

        body_elem = find_body_elem(base_root, body_name)
        if body_elem is None:
            missing_body.append((s, f"Body '{body_name}' not found"))
            continue

        append_site(body_elem, s)
        added += 1
        per_body_counts[body_name] += 1

    ET.indent(base_tree, space="    ")
    base_tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

    print("# ---- Merge Touch Sites (all box type) ----")
    print(f"Input base:    {base_xml_path}")
    print(f"Input sensors: {sensor_xml_path}")
    print(f"Output file:   {out_xml_path}")
    print(f"Sites found in sensor XML: {len(sensor_sites)}")
    print(f"Sites added:   {added}")
    print(f"Sites skipped (already present): {skipped}")
    if per_body_counts:
        print("\nPer-body additions:")
        for b, c in sorted(per_body_counts.items()):
            print(f"  {b}: {c}")
    if missing_body:
        print("\nWarnings (unplaced sites):")
        for s, msg in missing_body:
            print(f"  {s}: {msg}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add placeholder site nodes for touch sensors to a MuJoCo base hand XML.")
    ap.add_argument("--base", required=True, help="Path to hand_base.xml (body tree, no/new touch sites)")
    ap.add_argument("--sensors", required=True, help="Path to generated sensors XML (with <touch site='...'>)")
    ap.add_argument("--out", default="hand_with_sites_stub.xml", help="Output XML path")
    args = ap.parse_args()
    merge_sites(args.base, args.sensors, args.out)