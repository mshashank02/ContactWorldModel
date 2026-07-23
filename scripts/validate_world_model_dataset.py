#!/usr/bin/env python3
"""Validate that a collected dataset is ready for world-model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model.dataset_index import build_dataset_index
from world_model.dataset_validation import validate_dataset, write_validation_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--empty-contact-is-error",
        action="store_true",
        help="Fail instead of warning for episodes with no oracle contact.",
    )
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = validate_dataset(
        args.dataset_root, empty_contact_is_error=args.empty_contact_is_error
    )
    output = args.report or args.dataset_root / "index" / "validation_report.json"
    write_validation_report(report, output)
    if not args.skip_index and report.ready:
        build_dataset_index(args.dataset_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
