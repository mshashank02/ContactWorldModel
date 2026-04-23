#!/usr/bin/env python3
"""Generate a MuJoCo scene that drops all study objects onto a floor grid."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "study_objects" / "sphere_study_v1" / "manifest.csv"
OUTPUT_PATH = ROOT / "generated" / "all_48_study_objects_drop.xml"

MESHDIR = "../study_objects/sphere_study_v1"
COLS = 8
X_SPACING = 0.55
Y_SPACING = 0.70
X_START = -((COLS - 1) * X_SPACING) / 2.0
Y_START = 1.75
DROP_HEIGHT = 0.2
RIGID_ROW_OFFSET = -0.22
SOFT_MASS = 0.07
RIGID_MASS = 0.07
SCALE = "0.1 0.1 0.1"
RADIUS = "0.001"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    rows = list(csv.DictReader(MANIFEST_PATH.open("r", encoding="utf-8", newline="")))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append('<mujoco model="All 48 Study Objects Drop">')
    lines.append(f'  <compiler meshdir="{MESHDIR}" autolimits="true"/>')
    lines.append('')
    lines.append('  <option solver="CG" tolerance="1e-8" timestep="1e-4" iterations="100" ls_iterations="50" integrator="implicitfast"/>')
    lines.append('  <size memory="1G" nconmax="4000"/>')
    lines.append('')
    lines.append('  <visual>')
    lines.append('    <headlight ambient="0.55 0.55 0.55" diffuse="0.55 0.55 0.55" specular="0.15 0.15 0.15"/>')
    lines.append('    <rgba haze="0.12 0.16 0.22 1"/>')
    lines.append('    <global offwidth="1600" offheight="900"/>')
    lines.append('  </visual>')
    lines.append('')
    lines.append('  <asset>')
    lines.append('    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.97 0.98 1.0" rgb2="0.73 0.80 0.91" width="512" height="3072"/>')
    lines.append('    <material name="floor_mat" rgba="0.86 0.88 0.90 1" specular="0.1" shininess="0.1"/>')
    lines.append('    <material name="soft_mat" rgba="0.23 0.55 0.93 0.55"/>')
    lines.append('    <material name="rigid_mat" rgba="0.91 0.36 0.28 0.65"/>')
    lines.append('  </asset>')
    lines.append('')
    lines.append('  <worldbody>')
    lines.append('    <light pos="0 0 8" dir="0 0 -1" directional="true"/>')
    lines.append('    <camera name="overview" pos="0 -8.8 5.8" xyaxes="1 0 0 0 0.58 0.82"/>')
    lines.append('    <geom name="floor" type="plane" pos="0 0 0" size="8 8 0.1" material="floor_mat" condim="3" friction="1 0.005 0.0001"/>')

    for idx, row in enumerate(rows):
        grid_row = idx // COLS
        grid_col = idx % COLS
        base_x = X_START + grid_col * X_SPACING
        base_y = Y_START - grid_row * Y_SPACING
        mesh_file = row["msh_file"]
        object_id = row["object_id"]

        lines.append(f'    <!-- {object_id} -->')
        lines.append(
            f'    <flexcomp name="soft_{object_id}" type="gmsh" '
            f'file="{mesh_file}" dim="3" dof="trilinear" '
            f'pos="{fmt(base_x)} {fmt(base_y)} {fmt(DROP_HEIGHT)}" '
            f'scale="{SCALE}" mass="{SOFT_MASS}" radius="{RADIUS}" material="soft_mat">'
        )
        lines.append('      <elasticity young="1000000" poisson="0.45" damping="0.001"/>')
        lines.append('      <contact selfcollide="none" internal="false" friction="1 0.005 0.0001"/>')
        lines.append('    </flexcomp>')

        lines.append(
            f'    <body name="rigid_{object_id}" '
            f'pos="{fmt(base_x)} {fmt(base_y + RIGID_ROW_OFFSET)} {fmt(DROP_HEIGHT)}">'
        )
        lines.append('      <freejoint/>')
        lines.append(f'      <inertial pos="0 0 0" mass="{RIGID_MASS}" diaginertia="1e-4 1e-4 1e-4"/>')
        lines.append(
            f'      <flexcomp name="rigid_copy_{object_id}" type="gmsh" '
            f'file="{mesh_file}" dim="3" dof="trilinear" rigid="true" '
            f'pos="0 0 0" scale="{SCALE}" radius="{RADIUS}" material="rigid_mat">'
        )
        lines.append('        <contact selfcollide="none" internal="false" friction="1 0.005 0.0001"/>')
        lines.append('      </flexcomp>')
        lines.append('    </body>')
        lines.append('')

    lines.append('  </worldbody>')
    lines.append('</mujoco>')
    lines.append('')

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
