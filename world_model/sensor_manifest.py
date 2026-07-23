"""Build canonical sensor manifests from MuJoCo XML measurement sites."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


@dataclass
class SensorSite:
    sensor_id: int
    site_name: str
    touch_sensor_name: str | None
    body_name: str
    finger_id: str
    link_id: str
    region_type: str
    local_position_xyz: list[float]
    local_normal_xyz: list[float]
    global_rest_position_xyz: list[float] | None
    active_in_layout: bool
    candidate_id: str
    site_size_xyz: list[float]
    site_quat_wxyz: list[float]
    policy_touch_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numbers(value: str | None, length: int, default: Iterable[float]) -> list[float]:
    values = [float(item) for item in value.split()] if value else list(default)
    if len(values) < length:
        values.extend([0.0] * (length - len(values)))
    return values[:length]


def _iter_xml_documents(path: str | Path, seen: set[Path] | None = None) -> Iterator[tuple[Path, ET.Element]]:
    """Yield an XML document and all recursively included documents."""

    source = Path(path).expanduser().resolve()
    visited = seen if seen is not None else set()
    if source in visited:
        return
    visited.add(source)
    root = ET.parse(source).getroot()
    yield source, root
    for include in root.findall(".//include"):
        filename = include.get("file")
        if not filename:
            continue
        included = (source.parent / filename).resolve()
        if included.is_file():
            yield from _iter_xml_documents(included, visited)


def touch_sensor_map(xml_path: str | Path) -> dict[str, str]:
    """Return ``site_name -> touch_sensor_name`` from an XML include tree."""

    result: dict[str, str] = {}
    for _, root in _iter_xml_documents(xml_path):
        for touch in root.findall(".//sensor/touch") + root.findall(".//touch"):
            site_name = touch.get("site")
            if site_name:
                result[site_name] = touch.get("name") or site_name.replace(":T_", ":TS_")
    return result


def _iter_body_sites(body: ET.Element) -> Iterator[tuple[str, ET.Element]]:
    body_name = body.get("name", "")
    for site in body.findall("./site"):
        yield body_name, site
    for child in body.findall("./body"):
        yield from _iter_body_sites(child)


def _site_elements(xml_path: str | Path) -> dict[str, tuple[str, ET.Element]]:
    result: dict[str, tuple[str, ET.Element]] = {}
    for _, root in _iter_xml_documents(xml_path):
        for worldbody in root.findall(".//worldbody"):
            for body in worldbody.findall("./body"):
                for body_name, site in _iter_body_sites(body):
                    name = site.get("name")
                    if name:
                        result[name] = (body_name, site)
        # Robot fragments sometimes start directly with a body.
        for body in root.findall("./body"):
            for body_name, site in _iter_body_sites(body):
                name = site.get("name")
                if name:
                    result[name] = (body_name, site)
    return result


def _classify(site_name: str, body_name: str) -> tuple[str, str, str]:
    text = f"{site_name} {body_name}".lower()
    finger = "palm"
    for prefix, label in (
        ("ff", "forefinger"),
        ("mf", "middle"),
        ("rf", "ring"),
        ("lf", "little"),
        ("th", "thumb"),
    ):
        if re.search(rf"(?:t_|:){prefix}", text):
            finger = label
            break
    if "lfmetacarpal" in text:
        finger = "little"

    link = "palm"
    for candidate in ("metacarpal", "proximal", "middle", "distal", "tip"):
        if candidate in text:
            link = candidate
            break
    region = "tip" if ("tip" in site_name.lower() or link in {"tip", "distal"}) else (
        "palm" if "palm" in text or "metacarpal" in text else "non_tip"
    )
    return finger, link, region


def _infer_normal(site_name: str, position: list[float], size: list[float]) -> list[float]:
    lower = site_name.lower()
    named_axes = (
        ("front", 1, -1.0),
        ("back", 1, 1.0),
        ("left", 0, -1.0),
        ("right", 0, 1.0),
        ("_tip", 2, 1.0),
    )
    for token, axis, sign in named_axes:
        if token in lower:
            normal = [0.0, 0.0, 0.0]
            normal[axis] = sign
            return normal
    nonzero_sizes = [(value, axis) for axis, value in enumerate(size) if value > 0.0]
    axis = min(nonzero_sizes)[1] if nonzero_sizes else int(np.argmax(np.abs(position)))
    sign = math.copysign(1.0, position[axis]) if position[axis] != 0 else 1.0
    normal = [0.0, 0.0, 0.0]
    normal[axis] = sign
    return normal


def build_sensor_manifest(
    active_xml_path: str | Path,
    oracle_site_xml_path: str | Path,
    candidate_id: str,
    oracle_sensor_xml_path: str | Path | None = None,
) -> list[SensorSite]:
    """Build the dense superset manifest without modifying the MuJoCo model.

    ``oracle_site_xml_path`` supplies passive local site volumes. The active XML
    is used only to mark the subset visible to the policy.
    """

    active = touch_sensor_map(active_xml_path)
    active_order = {
        sensor_name: index
        for index, (site_name, sensor_name) in enumerate(
            (item for item in active.items() if ":T_" in item[0] or item[0].startswith("T_"))
        )
    }
    oracle_touches = touch_sensor_map(oracle_sensor_xml_path) if oracle_sensor_xml_path else {}
    sites = _site_elements(oracle_site_xml_path)
    if oracle_touches:
        site_names = sorted(oracle_touches)
    else:
        site_names = sorted(name for name in sites if ":T_" in name or name.startswith("T_"))
    missing = [name for name in site_names if name not in sites]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} oracle touch sites are absent from the site XML: {preview}")
    uncovered_active = sorted(
        name
        for name in set(active).difference(site_names)
        if ":T_" in name or name.startswith("T_")
    )
    if uncovered_active:
        preview = ", ".join(uncovered_active[:5])
        raise ValueError(
            "Oracle sites are not a superset of the active policy layout; "
            f"{len(uncovered_active)} sites are missing: {preview}"
        )

    manifest: list[SensorSite] = []
    for sensor_id, site_name in enumerate(site_names):
        body_name, elem = sites[site_name]
        position = _numbers(elem.get("pos"), 3, (0.0, 0.0, 0.0))
        size = _numbers(elem.get("size"), 3, (0.0, 0.0, 0.0))
        quat = _numbers(elem.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
        finger, link, region = _classify(site_name, body_name)
        touch_name = (
            active.get(site_name)
            or oracle_touches.get(site_name)
            or site_name.replace(":T_", ":TS_")
        )
        manifest.append(
            SensorSite(
                sensor_id=sensor_id,
                site_name=site_name,
                touch_sensor_name=touch_name,
                body_name=body_name,
                finger_id=finger,
                link_id=link,
                region_type=region,
                local_position_xyz=position,
                local_normal_xyz=_infer_normal(site_name, position, size),
                global_rest_position_xyz=None,
                active_in_layout=site_name in active,
                candidate_id=candidate_id,
                site_size_xyz=size,
                site_quat_wxyz=quat,
                policy_touch_index=active_order.get(touch_name),
            )
        )
    if not manifest:
        raise ValueError(f"No tactile sites found in {oracle_site_xml_path}")
    return manifest


def sensor_counts(
    sensors: Iterable[SensorSite], *, active_only: bool = False
) -> dict[str, int]:
    """Count all canonical sites or only the active policy layout."""

    rows = [
        row for row in sensors if not active_only or row.active_in_layout
    ]
    palm = sum(row.region_type == "palm" for row in rows)
    tip = sum(row.region_type == "tip" for row in rows)
    return {
        "sensor_count_total": len(rows),
        "sensor_count_palm": palm,
        "sensor_count_tip": tip,
        "sensor_count_non_tip": len(rows) - palm - tip,
    }


def write_sensor_manifest(path: str | Path, sensors: Iterable[SensorSite]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_dict() for row in sensors]
    target.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_sensor_manifest(path: str | Path) -> list[SensorSite]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SensorSite(**row) for row in rows]
