#!/usr/bin/env python3
"""Compare actual mesh extents for study objects that share a family."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "study_objects" / "sphere_study_v1" / "manifest.csv"
MESH_DIR = ROOT / "study_objects" / "sphere_study_v1"


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_bounds(mesh_file: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
    xml = f"""
    <mujoco model="bounds">
      <compiler meshdir="{MESH_DIR}" autolimits="true"/>
      <worldbody>
        <flexcomp name="obj" type="gmsh"
                  file="{mesh_file}"
                  dim="3" dof="trilinear" rigid="true"
                  pos="0 0 0" scale="{scale} {scale} {scale}" radius="0.001"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    start = int(model.flex_vertadr[0])
    count = int(model.flex_vertnum[0])
    verts = model.flex_vert0[start : start + count]
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    return mins, maxs


def fmt_vec(vec: np.ndarray) -> str:
    return "(" + ", ".join(f"{v:.6f}" for v in vec) + ")"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare actual MuJoCo mesh extents across study object sizes.")
    parser.add_argument("--macro", required=True, choices=["low", "high"])
    parser.add_argument("--roughness", required=True, choices=["low", "high"])
    parser.add_argument("--aspect-ratio", required=True, choices=["low", "high"])
    parser.add_argument("--scale", type=float, default=0.1, help="Scale used in generated MuJoCo XMLs.")
    args = parser.parse_args()

    rows = load_manifest()
    selected = [
        row
        for row in rows
        if row["macro"] == args.macro
        and row["roughness"] == args.roughness
        and row["aspect_ratio"] == args.aspect_ratio
    ]
    selected.sort(key=lambda row: ["small", "medium", "large"].index(row["size"]))

    if len(selected) != 3:
        raise SystemExit(f"Expected 3 rows for one family, found {len(selected)}")

    print(
        f"Comparing family: macro={args.macro}, roughness={args.roughness}, "
        f"aspect_ratio={args.aspect_ratio}, scale={args.scale}"
    )
    print()
    print(
        f"{'size':<8} {'mesh_file':<52} {'extent_xyz':<34} {'volume_box':<14} {'mins':<34} {'maxs':<34}"
    )
    print("-" * 190)

    prev_extent = None
    for row in selected:
        mins, maxs = load_bounds(row["msh_file"], args.scale)
        extent = maxs - mins
        volume_box = float(np.prod(extent))
        print(
            f"{row['size']:<8} {row['msh_file']:<52} {fmt_vec(extent):<34} "
            f"{volume_box:<14.8f} {fmt_vec(mins):<34} {fmt_vec(maxs):<34}"
        )
        if prev_extent is not None:
            ratio = extent / prev_extent
            print(f"{'':<8} {'vs prev':<52} {fmt_vec(ratio):<34}")
        prev_extent = extent


if __name__ == "__main__":
    main()
