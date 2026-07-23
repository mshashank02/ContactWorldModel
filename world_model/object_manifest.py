"""Object metadata extraction for world-model collection runs."""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _parse_object_id(object_id: str) -> dict[str, str]:
    result: dict[str, str] = {}
    patterns = {
        "size": r"(?:^|_)size-([^_]+)",
        "aspect_ratio": r"(?:^|_)ar-([^_]+)",
        "macro_geometry": r"(?:^|_)macro-([^_]+)",
        "roughness": r"(?:^|_)rough-([^_]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, object_id)
        if match:
            result[key] = match.group(1)
    return result


def _xml_object_descriptors(xml_path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return result
    for body in root.findall(".//body"):
        if body.get("name") != "object":
            continue
        result["spawn_position"] = body.get("pos")
        geoms = []
        for geom in body.findall(".//geom"):
            geoms.append(
                {
                    key: geom.get(key)
                    for key in ("name", "type", "size", "mesh", "material", "friction", "mass", "density")
                    if geom.get(key) is not None
                }
            )
        if geoms:
            result["geometry"] = geoms
        flex = body.find(".//flexcomp")
        if flex is not None:
            result["flexcomp"] = dict(flex.attrib)
        break
    return result


def load_object_row(manifest_csv: str | Path, object_id: str) -> dict[str, str]:
    with Path(manifest_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("object_id") == object_id:
                return {key: value for key, value in row.items() if key and value not in (None, "")}
    raise KeyError(f"Object {object_id!r} not found in {manifest_csv}")


def build_object_manifest(
    object_id: str,
    physics_mode: str,
    xml_path: str | Path,
    manifest_csv: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge repository object metadata, ID-derived labels, and XML descriptors."""

    result: dict[str, Any] = {
        "object_id": object_id,
        "base_object": object_id,
        "size": None,
        "aspect_ratio": None,
        "roughness": None,
        "macro_geometry": None,
        "rigid_deformable": physics_mode,
        "is_deformable": physics_mode == "deformable",
    }
    result.update(_parse_object_id(object_id))
    if manifest_csv:
        row = load_object_row(manifest_csv, object_id)
        result.update(row)
        if "macro" in row:
            result["macro_geometry"] = row["macro"]
    result["rigid_deformable"] = physics_mode
    result["is_deformable"] = physics_mode == "deformable"
    result["xml_descriptors"] = _xml_object_descriptors(xml_path)
    if overrides:
        result.update(overrides)
    return result


def write_object_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
