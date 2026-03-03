#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import mujoco


def build_welds_from_model(model: mujoco.MjModel) -> list[tuple[str, str]]:
    welds: list[tuple[str, str]] = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name.startswith("robot0:"):
            continue
        parent_id = int(model.body_parentid[body_id])
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or ""
        if parent_name.startswith("robot0:"):
            welds.append((body_name, parent_name))
    return welds


def insert_welds_into_shared(shared_text: str, weld_pairs: list[tuple[str, str]]) -> str:
    marker = "</mujoco>"
    if marker not in shared_text:
        raise ValueError("Could not find </mujoco> in shared.xml")

    weld_lines = [
        "    <equality>",
        "        <!-- Added for rigid-hand diagnostic: weld hand body chain -->",
    ]
    for child, parent in weld_pairs:
        weld_lines.append(
            f'        <weld body1="{child}" body2="{parent}" />'
        )
    weld_lines += ["    </equality>", ""]
    weld_block = "\n".join(weld_lines)

    return shared_text.replace(marker, weld_block + marker, 1)


def rewrite_top_xml_include(top_xml_text: str, old_include: str, new_include: str) -> str:
    old_token = f'<include file="{old_include}" />'
    new_token = f'<include file="{new_include}" />'
    if old_token not in top_xml_text:
        # Fallback: replace first shared.xml include occurrence.
        top_xml_text_new = top_xml_text.replace('file="shared.xml"', f'file="{new_include}"', 1)
        if top_xml_text_new == top_xml_text:
            raise ValueError("Could not find shared.xml include in top-level XML")
        return top_xml_text_new
    return top_xml_text.replace(old_token, new_token, 1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a copy of a generated ShadowHand XML with the hand welded rigid via equality constraints."
    )
    p.add_argument(
        "--xml-path",
        required=True,
        help="Path to the top-level generated XML (e.g., manipulate_block_touch_sensors_...xml)",
    )
    p.add_argument(
        "--out-xml",
        default=None,
        help="Optional output top-level XML path. Defaults to <input>_static_hand_welded.xml",
    )
    p.add_argument(
        "--out-shared",
        default=None,
        help="Optional output shared include path. Defaults to sibling shared_static_hand_welded.xml",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = Path(args.xml_path).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    top_dir = xml_path.parent
    in_shared = top_dir / "shared.xml"
    if not in_shared.is_file():
        raise FileNotFoundError(f"Expected include file not found: {in_shared}")

    out_xml = (
        Path(args.out_xml).expanduser().resolve()
        if args.out_xml
        else xml_path.with_name(xml_path.stem + "_static_hand_welded.xml")
    )
    out_shared = (
        Path(args.out_shared).expanduser().resolve()
        if args.out_shared
        else top_dir / "shared_static_hand_welded.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    weld_pairs = build_welds_from_model(model)
    if not weld_pairs:
        raise RuntimeError("No robot0:* parent-child bodies found to weld")

    shared_text = in_shared.read_text(encoding="utf-8")
    shared_welded = insert_welds_into_shared(shared_text, weld_pairs)
    out_shared.write_text(shared_welded, encoding="utf-8")

    top_xml_text = xml_path.read_text(encoding="utf-8")
    top_xml_welded = rewrite_top_xml_include(
        top_xml_text, old_include="shared.xml", new_include=out_shared.name
    )
    out_xml.write_text(top_xml_welded, encoding="utf-8")

    print(f"Created welded shared include: {out_shared}")
    print(f"Created welded top XML:      {out_xml}")
    print(f"Weld constraints added:      {len(weld_pairs)}")
    print("Test with: freeze_hand_drop_test.py --xml-path <welded_top_xml> --freeze-steps 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
