#!/usr/bin/env python3
"""
Step 1.6: I/O-focused refinement.

Input: openalex_approved.csv
Outputs:
- openalex_final_for_neon.csv (kept constructs)
- openalex_excluded_non_io.csv (excluded constructs + reason)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


EXCLUDE_KEYWORDS = [
    "semiconductor",
    "chess",
    "geometry",
    "mutation",
    "infectious disease",
    "covid",
    "military",
    "psycinfo",
    "computer user satisfaction",
]

EXCLUDE_EXACT = {
    "PsycINFO",
    "Reliability (semiconductor)",
    "Promotion (chess)",
    "Similarity (geometry)",
    "White (mutation)",
    "Infectious disease (medical specialty)",
    "Coronavirus disease 2019 (COVID-19)",
    "Investment (military)",
}

# Emergency keep-list in case keyword filters catch valid constructs.
KEEP_EXACT = {
    "Reliability",
    "Validity",
    "Construct validity",
    "Criterion validity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply I/O refinement to approved OpenAlex constructs.")
    parser.add_argument("--input", default="openalex_approved.csv")
    parser.add_argument("--output", default="openalex_final_for_neon.csv")
    parser.add_argument("--excluded-output", default="openalex_excluded_non_io.csv")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def to_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def exclusion_reason(name: str) -> str:
    if name in KEEP_EXACT:
        return ""
    if name in EXCLUDE_EXACT:
        return f"exact:{name}"
    lowered = name.casefold()
    for kw in EXCLUDE_KEYWORDS:
        if kw in lowered:
            return f"keyword:{kw}"
    return ""


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: List[Dict[str, str]], top_k: int, title: str) -> None:
    ordered = sorted(
        rows,
        key=lambda r: (-to_int(r.get("Total_Paper_Count", "0")), (r.get("Construct_Name") or "").casefold()),
    )
    print(title)
    for idx, row in enumerate(ordered[: max(1, top_k)], start=1):
        print(f"{idx:>2}. {row['Construct_Name']} | Total_Paper_Count={row['Total_Paper_Count']}")


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input))

    kept: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    for row in rows:
        name = (row.get("Construct_Name") or "").strip()
        if not name:
            continue
        reason = exclusion_reason(name)
        if reason:
            row_with_reason = dict(row)
            row_with_reason["Exclusion_Reason"] = reason
            excluded.append(row_with_reason)
        else:
            kept.append(row)

    kept = sorted(
        kept,
        key=lambda r: (-to_int(r.get("Total_Paper_Count", "0")), (r.get("Construct_Name") or "").casefold()),
    )
    excluded = sorted(
        excluded,
        key=lambda r: (-to_int(r.get("Total_Paper_Count", "0")), (r.get("Construct_Name") or "").casefold()),
    )

    write_rows(
        Path(args.output),
        kept,
        [
            "Construct_Name",
            "Source",
            "Total_Paper_Count",
            "Concept_Levels",
            "Journal_Count",
            "Journals",
            "Year_Min",
            "Year_Max",
        ],
    )
    write_rows(
        Path(args.excluded_output),
        excluded,
        [
            "Construct_Name",
            "Source",
            "Total_Paper_Count",
            "Concept_Levels",
            "Journal_Count",
            "Journals",
            "Year_Min",
            "Year_Max",
            "Exclusion_Reason",
        ],
    )

    print(f"Kept constructs: {len(kept)}")
    print(f"Excluded constructs: {len(excluded)}")
    print("")
    print_top(kept, args.top_k, title=f"Top {args.top_k} after I/O refinement:")


if __name__ == "__main__":
    main()
