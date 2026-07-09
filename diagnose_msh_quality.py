#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MshData:
    path: Path
    nodes: dict[int, tuple[float, float, float]]
    tets: list[tuple[int, int, int, int]]
    element_types: dict[int, int]
    warnings: list[str]


def _read_msh_v2(path: Path) -> MshData:
    raw = path.read_bytes()
    warnings: list[str] = []
    fmt_start = raw.find(b"$MeshFormat")
    if fmt_start < 0:
        raise ValueError("missing $MeshFormat")
    fmt_line_start = raw.find(b"\n", fmt_start) + 1
    fmt_line_end = raw.find(b"\n", fmt_line_start)
    fmt_parts = raw[fmt_line_start:fmt_line_end].strip().split()
    if len(fmt_parts) < 3:
        raise ValueError("invalid $MeshFormat line")
    binary = fmt_parts[1] == b"1"
    data_size = int(fmt_parts[2])

    if not binary:
        return _read_ascii_msh_v2(raw, path, warnings)
    return _read_binary_msh_v2(raw, path, data_size, warnings)


def _section_text(raw: bytes, name: bytes) -> bytes:
    start = raw.find(b"$" + name)
    if start < 0:
        raise ValueError(f"missing ${name.decode()}")
    start = raw.find(b"\n", start) + 1
    end = raw.find(b"$End" + name, start)
    if end < 0:
        raise ValueError(f"missing $End{name.decode()}")
    return raw[start:end]


def _read_ascii_msh_v2(raw: bytes, path: Path, warnings: list[str]) -> MshData:
    node_lines = _section_text(raw, b"Nodes").decode("utf-8", errors="replace").splitlines()
    node_count = int(node_lines[0].strip())
    nodes: dict[int, tuple[float, float, float]] = {}
    for line in node_lines[1 : node_count + 1]:
        parts = line.split()
        if len(parts) >= 4:
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))

    elem_lines = _section_text(raw, b"Elements").decode("utf-8", errors="replace").splitlines()
    elem_count = int(elem_lines[0].strip())
    tets: list[tuple[int, int, int, int]] = []
    element_types: dict[int, int] = {}
    for line in elem_lines[1 : elem_count + 1]:
        parts = line.split()
        if len(parts) < 4:
            continue
        etype = int(parts[1])
        ntags = int(parts[2])
        element_types[etype] = element_types.get(etype, 0) + 1
        node_ids = [int(x) for x in parts[3 + ntags :]]
        if etype == 4 and len(node_ids) >= 4:
            tets.append(tuple(node_ids[:4]))
        elif etype == 11 and len(node_ids) >= 10:
            tets.append(tuple(node_ids[:4]))
    return MshData(path, nodes, tets, element_types, warnings)


def _read_binary_msh_v2(raw: bytes, path: Path, data_size: int, warnings: list[str]) -> MshData:
    nodes_start = raw.find(b"$Nodes")
    if nodes_start < 0:
        raise ValueError("missing $Nodes")
    count_start = raw.find(b"\n", nodes_start) + 1
    count_end = raw.find(b"\n", count_start)
    node_count = int(raw[count_start:count_end].strip())

    node_offset = count_end + 1
    endian = "<"
    rec = struct.Struct(endian + "i" + ("d" if data_size == 8 else "f") * 3)
    nodes: dict[int, tuple[float, float, float]] = {}
    for _ in range(node_count):
        node_id, x, y, z = rec.unpack_from(raw, node_offset)
        node_offset += rec.size
        nodes[int(node_id)] = (float(x), float(y), float(z))

    elem_start = raw.find(b"$Elements")
    if elem_start < 0:
        raise ValueError("missing $Elements")
    count_start = raw.find(b"\n", elem_start) + 1
    count_end = raw.find(b"\n", count_start)
    elem_count = int(raw[count_start:count_end].strip())
    offset = count_end + 1

    node_counts = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 8: 3, 9: 6, 10: 9, 11: 10, 15: 1}
    element_types: dict[int, int] = {}
    tets: list[tuple[int, int, int, int]] = []
    parsed = 0
    block_header = struct.Struct(endian + "iii")
    while parsed < elem_count:
        etype, block_n, tag_n = block_header.unpack_from(raw, offset)
        offset += block_header.size
        n_nodes = node_counts.get(etype)
        if n_nodes is None:
            warnings.append(f"unknown binary element type {etype}")
            break
        rec_ints = 1 + tag_n + n_nodes
        elem_rec = struct.Struct(endian + "i" * rec_ints)
        element_types[etype] = element_types.get(etype, 0) + block_n
        for _ in range(block_n):
            vals = elem_rec.unpack_from(raw, offset)
            offset += elem_rec.size
            node_ids = vals[1 + tag_n :]
            if etype in (4, 11):
                tets.append(tuple(int(x) for x in node_ids[:4]))
        parsed += block_n

    return MshData(path, nodes, tets, element_types, warnings)


def _tet_metrics(points: np.ndarray) -> tuple[float, float, float]:
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(float(np.linalg.norm(points[i] - points[j])))
    min_edge = min(edges)
    max_edge = max(edges)
    volume = abs(float(np.linalg.det(np.stack([points[1] - points[0], points[2] - points[0], points[3] - points[0]], axis=1)))) / 6.0
    edge_ratio = max_edge / min_edge if min_edge > 0 else math.inf
    return volume, min_edge, edge_ratio


def diagnose(path: Path) -> dict[str, object]:
    mesh = _read_msh_v2(path)
    coords = np.array(list(mesh.nodes.values()), dtype=np.float64)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_size = bbox_max - bbox_min

    volumes: list[float] = []
    min_edges: list[float] = []
    edge_ratios: list[float] = []
    missing_refs = 0
    duplicate_node_tets = 0
    for tet in mesh.tets:
        if len(set(tet)) != 4:
            duplicate_node_tets += 1
            continue
        try:
            points = np.array([mesh.nodes[node_id] for node_id in tet], dtype=np.float64)
        except KeyError:
            missing_refs += 1
            continue
        volume, min_edge, edge_ratio = _tet_metrics(points)
        volumes.append(volume)
        min_edges.append(min_edge)
        edge_ratios.append(edge_ratio)

    vol = np.array(volumes, dtype=np.float64)
    edge = np.array(min_edges, dtype=np.float64)
    ratio = np.array(edge_ratios, dtype=np.float64)
    tiny_volume = int(np.sum(vol <= 1e-12)) if vol.size else 0
    tiny_edge = int(np.sum(edge <= 1e-8)) if edge.size else 0
    bad_ratio = int(np.sum(ratio >= 50.0)) if ratio.size else 0

    warnings = list(mesh.warnings)
    if not mesh.tets:
        warnings.append("no linear/quadratic tetrahedra found")
    if tiny_volume:
        warnings.append(f"{tiny_volume} tetrahedra have volume <= 1e-12")
    if tiny_edge:
        warnings.append(f"{tiny_edge} tetrahedra have min edge <= 1e-8")
    if bad_ratio:
        warnings.append(f"{bad_ratio} tetrahedra have edge ratio >= 50")
    if max(bbox_size) > 100.0:
        warnings.append("pre-scale bbox is very large")
    if max(bbox_size) < 1e-3:
        warnings.append("pre-scale bbox is very small")
    if missing_refs:
        warnings.append(f"{missing_refs} tetrahedra reference missing nodes")
    if duplicate_node_tets:
        warnings.append(f"{duplicate_node_tets} tetrahedra repeat node ids")

    risk = 0
    risk += 5 if not mesh.tets else 0
    risk += min(5, tiny_volume)
    risk += min(5, tiny_edge)
    risk += min(5, bad_ratio)
    risk += 3 if max(bbox_size) > 100.0 or max(bbox_size) < 1e-3 else 0

    return {
        "path": str(path),
        "nodes": len(mesh.nodes),
        "tets": len(mesh.tets),
        "element_types": mesh.element_types,
        "bbox_size": bbox_size.tolist(),
        "volume_min": float(np.min(vol)) if vol.size else None,
        "volume_p01": float(np.percentile(vol, 1)) if vol.size else None,
        "min_edge_min": float(np.min(edge)) if edge.size else None,
        "edge_ratio_max": float(np.max(ratio)) if ratio.size else None,
        "edge_ratio_p99": float(np.percentile(ratio, 99)) if ratio.size else None,
        "tiny_volume_tets": tiny_volume,
        "tiny_edge_tets": tiny_edge,
        "bad_edge_ratio_tets": bad_ratio,
        "risk": risk,
        "warnings": warnings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose Gmsh v2 tetrahedral mesh quality for MuJoCo flexcomp.")
    p.add_argument("paths", nargs="+")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    paths: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.msh")))
        else:
            paths.append(path)

    rows = []
    for path in paths:
        try:
            rows.append(diagnose(path))
        except Exception as exc:
            rows.append({"path": str(path), "risk": 999, "warnings": [f"{type(exc).__name__}: {exc}"]})

    rows.sort(key=lambda row: (-int(row["risk"]), str(row["path"])))
    print("risk tets nodes bbox_size volume_min min_edge_min edge_ratio_max warnings path")
    for row in rows[: args.top]:
        bbox = row.get("bbox_size")
        bbox_s = ",".join(f"{float(x):.4g}" for x in bbox) if bbox else "n/a"
        warnings = "; ".join(str(x) for x in row.get("warnings", [])) or "-"
        print(
            f"{row.get('risk')} {row.get('tets', 'n/a')} {row.get('nodes', 'n/a')} "
            f"{bbox_s} {row.get('volume_min')} {row.get('min_edge_min')} "
            f"{row.get('edge_ratio_max')} {warnings} {row['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
