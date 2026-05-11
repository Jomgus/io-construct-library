#!/usr/bin/env python3
"""
Compare old OpenAlex constructs from unified_master_constructs.csv
against new openalex_final_for_neon.csv.

Outputs:
- Top 20 Lost Constructs (old only), sorted by old paper count.
- Top 20 New Constructs (new only), sorted by new paper count.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old vs new OpenAlex construct sets.")
    parser.add_argument("--old", default="unified_master_constructs.csv", help="Old unified CSV path.")
    parser.add_argument("--new", default="openalex_final_for_neon.csv", help="New OpenAlex CSV path.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of rows to print for each list.")
    return parser.parse_args()


def to_int(value: str) -> int:
    if value is None:
        return 0
    raw = str(value).strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        try:
            return int(float(raw))
        except ValueError:
            return 0


def norm_name(value: str) -> str:
    return (value or "").strip().casefold()


def load_old_openalex(path: Path) -> Dict[str, Tuple[str, int]]:
    # key -> (display_name, paper_count)
    agg: Dict[str, int] = defaultdict(int)
    display: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("Source") or "").strip() != "OpenAlex":
                continue
            name = (row.get("Construct_Name") or "").strip()
            if not name:
                continue
            key = norm_name(name)
            agg[key] += to_int(row.get("Metadata_Value", "0"))
            display.setdefault(key, name)
    return {k: (display[k], v) for k, v in agg.items()}


def load_new(path: Path) -> Dict[str, Tuple[str, int]]:
    # key -> (display_name, paper_count)
    agg: Dict[str, int] = defaultdict(int)
    display: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Construct_Name") or "").strip()
            if not name:
                continue
            key = norm_name(name)
            agg[key] += to_int(row.get("Total_Paper_Count", "0"))
            display.setdefault(key, name)
    return {k: (display[k], v) for k, v in agg.items()}


def print_ranked(title: str, rows: list[Tuple[str, int]], top_k: int) -> None:
    print(title)
    if not rows:
        print("  (none)")
        print("")
        return
    for idx, (name, count) in enumerate(rows[: max(1, top_k)], start=1):
        print(f"{idx:>2}. {name} | Paper_Count={count}")
    print("")


def main() -> None:
    args = parse_args()
    old_map = load_old_openalex(Path(args.old))
    new_map = load_new(Path(args.new))

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    lost_keys = old_keys - new_keys
    new_only_keys = new_keys - old_keys

    lost_rows = sorted(
        ((old_map[k][0], old_map[k][1]) for k in lost_keys),
        key=lambda x: (-x[1], x[0].casefold()),
    )
    new_rows = sorted(
        ((new_map[k][0], new_map[k][1]) for k in new_only_keys),
        key=lambda x: (-x[1], x[0].casefold()),
    )

    print(f"Old OpenAlex unique constructs: {len(old_keys)}")
    print(f"New OpenAlex unique constructs: {len(new_keys)}")
    print(f"Lost constructs (old - new): {len(lost_rows)}")
    print(f"New constructs (new - old): {len(new_rows)}")
    print("")

    print_ranked("Top Lost Constructs:", lost_rows, args.top_k)
    print_ranked("Top New Constructs:", new_rows, args.top_k)


if __name__ == "__main__":
    main()
