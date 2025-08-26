#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update the <include file="..."> entries in a MuJoCo XML to point to new
touch-sensor include files.

By default, it looks for:
  - shared_touch_sensors_92.xml
  - robot_touch_sensors_92.xml

and replaces them with paths you provide.

Usage:
  python update_includes.py path/to/main.xml \
      --shared new_shared_file.xml \
      --robot new_robot_file.xml \
      [--backup]

If --backup is given, a copy of the original is saved as main.xml.bak
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

# Known default file names to replace
DEFAULT_SHARED = "shared_touch_sensors_92.xml"
DEFAULT_ROBOT  = "robot_touch_sensors_92.xml"

def update_includes(tree: ET.ElementTree, new_shared: str | None, new_robot: str | None) -> dict:
    """
    Update include file attributes in the parsed XML tree.

    Returns a dict with counts:
      {'shared_updated': int, 'robot_updated': int}
    """
    root = tree.getroot()
    counts = {'shared_updated': 0, 'robot_updated': 0}

    # Iterate over all <include .../> elements anywhere in the tree
    for elem in root.iter():
        if elem.tag != "include":
            continue
        f = elem.attrib.get("file", "")
        # Replace shared include
        if new_shared and os.path.normpath(f) == os.path.normpath(DEFAULT_SHARED):
            elem.set("file", new_shared)
            counts['shared_updated'] += 1
        # Replace robot include
        if new_robot and os.path.normpath(f) == os.path.normpath(DEFAULT_ROBOT):
            elem.set("file", new_robot)
            counts['robot_updated'] += 1

    return counts

def main():
    ap = argparse.ArgumentParser(description="Update MuJoCo XML include filenames for touch sensors.")
    ap.add_argument("xml_path", help="Path to the main MuJoCo XML file.")
    ap.add_argument("--shared", dest="new_shared", default=None,
                    help=f'New filename to replace "{DEFAULT_SHARED}".')
    ap.add_argument("--robot", dest="new_robot", default=None,
                    help=f'New filename to replace "{DEFAULT_ROBOT}".')
    ap.add_argument("--backup", action="store_true",
                    help="Save a .bak backup of the original file before writing.")
    args = ap.parse_args()

    if not os.path.isfile(args.xml_path):
        print(f"ERROR: Cannot find XML file: {args.xml_path}", file=sys.stderr)
        sys.exit(1)

    if not args.new_shared and not args.new_robot:
        print("Nothing to update: provide at least one of --shared or --robot", file=sys.stderr)
        sys.exit(2)

    # Parse
    try:
        tree = ET.parse(args.xml_path)
    except ET.ParseError as e:
        print(f"ERROR: Failed to parse XML ({args.xml_path}): {e}", file=sys.stderr)
        sys.exit(3)

    counts = update_includes(tree, args.new_shared, args.new_robot)

    # Report if nothing matched (helps catch typos / different defaults)
    if args.new_shared and counts['shared_updated'] == 0:
        print(f'WARNING: No <include file="{DEFAULT_SHARED}"> found to replace.')
    if args.new_robot and counts['robot_updated'] == 0:
        print(f'WARNING: No <include file="{DEFAULT_ROBOT}"> found to replace.')

    # Backup if requested
    if args.backup:
        bak_path = args.xml_path + ".bak"
        try:
            with open(args.xml_path, "rb") as src, open(bak_path, "wb") as dst:
                dst.write(src.read())
            print(f"Backup saved to: {bak_path}")
        except OSError as e:
            print(f"ERROR: Could not write backup: {e}", file=sys.stderr)
            sys.exit(4)

    # Write back (preserves UTF-8 + XML declaration; formatting may be normalized)
    try:
        tree.write(args.xml_path, encoding="utf-8", xml_declaration=True)
        print(f"Updated XML written to: {args.xml_path}")
        print(f"Replacements: shared={counts['shared_updated']}, robot={counts['robot_updated']}")
    except OSError as e:
        print(f"ERROR: Failed to write updated XML: {e}", file=sys.stderr)
        sys.exit(5)

if __name__ == "__main__":
    main()
