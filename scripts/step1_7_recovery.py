#!/usr/bin/env python3
"""
Step 1.7 targeted recovery pass.

Rules:
1) Baseline keep: concept levels >= 3.
2) Recovery keep: level 2 concepts only if old unified OpenAlex paper count > threshold.
3) Hard-drop levels 0 and 1.
4) Hard-drop known obvious noise.
5) Merge + dedupe + aggregate to openalex_ultimate_final.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


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
    "signal processing",
    "genetics",
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
    "Dominance (genetics)",
    "Transition (genetics)",
    "Sampling (signal processing)",
    "Representation (politics)",
}

KEEP_EXACT = {
    "Reliability",
    "Validity",
    "Construct validity",
    "Criterion validity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 1.7 recovery pass.")
    parser.add_argument("--raw-input", default="openalex_filtered_raw.csv")
    parser.add_argument("--old-unified", default="unified_master_constructs.csv")
    parser.add_argument("--output", default="openalex_ultimate_final.csv")
    parser.add_argument("--recovered-output", default="openalex_recovered_level2.csv")
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def to_int(value: str) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        try:
            return int(float(str(value).strip()))
        except Exception:
            return 0


def norm(s: str) -> str:
    return (s or "").strip().casefold()


def exclusion_reason(name: str) -> str:
    if name in KEEP_EXACT:
        return ""
    if name in EXCLUDE_EXACT:
        return f"exact:{name}"
    n = norm(name)
    for kw in EXCLUDE_KEYWORDS:
        if kw in n:
            return f"keyword:{kw}"
    return ""


def load_old_openalex_counts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("Source") or "").strip() != "OpenAlex":
                continue
            name = (row.get("Construct_Name") or "").strip()
            if not name:
                continue
            counts[norm(name)] += to_int(row.get("Metadata_Value", "0"))
    return counts


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_input)
    old_path = Path(args.old_unified)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw input: {raw_path}")
    if not old_path.exists():
        raise FileNotFoundError(f"Missing old unified CSV: {old_path}")

    old_counts = load_old_openalex_counts(old_path)

    # Aggregate into final construct-level records.
    totals: Dict[str, int] = defaultdict(int)
    levels: Dict[str, Set[str]] = defaultdict(set)
    journals: Dict[str, Set[str]] = defaultdict(set)
    year_min: Dict[str, int] = {}
    year_max: Dict[str, int] = {}
    display_name: Dict[str, str] = {}
    sources: Dict[str, Set[str]] = defaultdict(set)
    recovered_level2: Set[str] = set()

    excluded_level01 = 0
    excluded_noise = 0
    excluded_unqualified_level2 = 0

    with raw_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("Construct_Name") or "").strip()
            if not name:
                continue
            key = norm(name)
            level = to_int(row.get("Concept_Level", "0"))
            paper_count = to_int(row.get("Paper_Count", "0"))
            journal = (row.get("Journal") or "").strip()
            year = to_int(row.get("Year", "0"))

            if level in (0, 1):
                excluded_level01 += 1
                continue

            noise = exclusion_reason(name)
            if noise:
                excluded_noise += 1
                continue

            keep = False
            if level >= 3:
                keep = True
            elif level == 2:
                old_count = old_counts.get(key, 0)
                if old_count > args.threshold:
                    keep = True
                    recovered_level2.add(key)
                else:
                    excluded_unqualified_level2 += 1

            if not keep:
                continue

            totals[key] += paper_count
            levels[key].add(str(level))
            if journal:
                journals[key].add(journal)
            if year > 0:
                if key not in year_min or year < year_min[key]:
                    year_min[key] = year
                if key not in year_max or year > year_max[key]:
                    year_max[key] = year
            display_name.setdefault(key, name)
            sources[key].add("OpenAlex")

    final_rows: List[Dict[str, str]] = []
    recovered_rows: List[Dict[str, str]] = []
    for key, total in totals.items():
        row = {
            "Construct_Name": display_name[key],
            "Source": "OpenAlex",
            "Total_Paper_Count": str(total),
            "Concept_Levels": ",".join(sorted(levels[key], key=int)),
            "Journal_Count": str(len(journals[key])),
            "Journals": "; ".join(sorted(journals[key])),
            "Year_Min": str(year_min.get(key, "")),
            "Year_Max": str(year_max.get(key, "")),
            "Recovered_Level2": "True" if key in recovered_level2 else "False",
            "Old_Unified_Paper_Count": str(old_counts.get(key, 0)),
        }
        final_rows.append(row)
        if key in recovered_level2:
            recovered_rows.append(row)

    final_rows.sort(
        key=lambda r: (-to_int(r["Total_Paper_Count"]), r["Construct_Name"].casefold())
    )
    recovered_rows.sort(
        key=lambda r: (-to_int(r["Total_Paper_Count"]), r["Construct_Name"].casefold())
    )

    fieldnames = [
        "Construct_Name",
        "Source",
        "Total_Paper_Count",
        "Concept_Levels",
        "Journal_Count",
        "Journals",
        "Year_Min",
        "Year_Max",
        "Recovered_Level2",
        "Old_Unified_Paper_Count",
    ]
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final_rows)

    with Path(args.recovered_output).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(recovered_rows)

    print(f"Final constructs: {len(final_rows)}")
    print(f"Recovered Level 2 constructs: {len(recovered_rows)} (threshold > {args.threshold})")
    print(f"Dropped rows at levels 0/1: {excluded_level01}")
    print(f"Dropped rows for noise list: {excluded_noise}")
    print(f"Dropped level 2 rows below threshold: {excluded_unqualified_level2}")
    print("")
    print(f"Top {args.top_k} constructs after Step 1.7:")
    for i, r in enumerate(final_rows[: max(1, args.top_k)], start=1):
        mark = " [RECOVERED_L2]" if r["Recovered_Level2"] == "True" else ""
        print(f"{i:>2}. {r['Construct_Name']} | Total_Paper_Count={r['Total_Paper_Count']}{mark}")


if __name__ == "__main__":
    main()
