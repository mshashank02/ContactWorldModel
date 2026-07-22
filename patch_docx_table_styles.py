#!/usr/bin/env python3
"""Add Pandoc's missing Compact/Table styles to a DOCX reference-based build."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def style_id(node: ET.Element) -> str | None:
    return node.get(w("styleId"))


def patch_styles(target_docx: Path, donor_docx: Path) -> None:
    with zipfile.ZipFile(target_docx) as target_zip:
        target_styles = ET.fromstring(target_zip.read("word/styles.xml"))
        members = {item.filename: (item, target_zip.read(item.filename)) for item in target_zip.infolist()}

    with zipfile.ZipFile(donor_docx) as donor_zip:
        donor_styles = ET.fromstring(donor_zip.read("word/styles.xml"))

    existing = {style_id(node) for node in target_styles.findall(w("style"))}
    for node in donor_styles.findall(w("style")):
        if style_id(node) in {"Compact", "Table"} and style_id(node) not in existing:
            target_styles.append(node)

    members["word/styles.xml"] = (
        members["word/styles.xml"][0],
        ET.tostring(target_styles, encoding="utf-8", xml_declaration=True),
    )

    with tempfile.NamedTemporaryFile(dir=target_docx.parent, suffix=".docx", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        with zipfile.ZipFile(temp_path, "w") as output_zip:
            for _, (info, data) in members.items():
                output_zip.writestr(info, data)
        temp_path.replace(target_docx)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_docx_table_styles.py TARGET.docx DONOR.docx")
    patch_styles(Path(sys.argv[1]), Path(sys.argv[2]))
