#!/usr/bin/env python3
"""Split mf_go_terms.tsv into per-level TSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-level TSV files from mf_go_terms.tsv.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the mf_go_terms.tsv input file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write per-level TSV files (defaults to the input directory).",
    )
    return parser.parse_args()


def split_by_level(input_path: Path, output_dir: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames or "level" not in fieldnames:
            raise ValueError(
                f"Input file missing 'level' column: {input_path}"
            )

        rows_by_level: Dict[str, List[Dict[str, str]]] = {}
        for row in reader:
            level = row.get("level")
            if level is None or level == "":
                raise ValueError(
                    f"Row missing level value (go_id={row.get('go_id', 'unknown')})."
                )
            rows_by_level.setdefault(level, []).append(row)

    for level, rows in rows_by_level.items():
        level_path = output_dir / f"mf_go_terms-level-{level}.tsv"
        with level_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, delimiter="\t", fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    split_by_level(input_path=input_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
