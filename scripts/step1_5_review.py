#!/usr/bin/env python3
"""
Step 1.5 review pipeline.

Reads openalex_filtered.csv and produces:
1) Aggregate summary + top 20 constructs in terminal.
2) noise_review.csv (only Is_Noise == True rows, sorted by Paper_Count desc).
3) openalex_approved.csv (only Is_Noise == False, aggregated + deduplicated by Construct_Name).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ALLOWED_CONCEPT_LEVELS = {3, 4, 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Step 1.5 review outputs.")
    parser.add_argument("--input", default="openalex_filtered.csv", help="Input CSV path.")
    parser.add_argument("--noise-output", default="noise_review.csv", help="Noise review CSV path.")
    parser.add_argument(
        "--approved-output",
        default="openalex_approved.csv",
        help="Approved aggregated CSV path.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="How many top constructs to print.")
    parser.add_argument(
        "--allow-missing-level",
        action="store_true",
        help="Allow processing inputs that do not include Concept_Level values.",
    )
    return parser.parse_args()


def to_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    noise_rows: List[Dict[str, str]] = []
    construct_totals: Dict[str, int] = defaultdict(int)

    approved_totals: Dict[str, int] = defaultdict(int)
    approved_journals: Dict[str, Set[str]] = defaultdict(set)
    approved_year_min: Dict[str, int] = {}
    approved_year_max: Dict[str, int] = {}
    approved_levels: Dict[str, Set[str]] = defaultdict(set)
    saw_valid_level = False

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            construct = (row.get("Construct_Name") or "").strip()
            paper_count = to_int(row.get("Paper_Count", "0"))
            is_noise = (row.get("Is_Noise") or "").strip().casefold() == "true"
            journal = (row.get("Journal") or "").strip()
            year = to_int(row.get("Year", "0"))
            concept_level = to_int(row.get("Concept_Level", "0"))
            if concept_level in ALLOWED_CONCEPT_LEVELS:
                saw_valid_level = True

            if not construct:
                continue
            if concept_level and concept_level not in ALLOWED_CONCEPT_LEVELS:
                # Safety guard if input file contains mixed levels.
                continue

            construct_totals[construct] += paper_count

            if is_noise:
                noise_rows.append(row)
                continue

            approved_totals[construct] += paper_count
            if concept_level:
                approved_levels[construct].add(str(concept_level))
            if journal:
                approved_journals[construct].add(journal)
            if year > 0:
                if construct not in approved_year_min or year < approved_year_min[construct]:
                    approved_year_min[construct] = year
                if construct not in approved_year_max or year > approved_year_max[construct]:
                    approved_year_max[construct] = year

    if not saw_valid_level and not args.allow_missing_level:
        raise RuntimeError(
            "No Concept_Level values (3/4/5) found in input. "
            "Re-run openalex_step1_pipeline.py to regenerate openalex_filtered.csv with Concept_Level."
        )

    # 1) Print aggregate summary + top constructs
    unique_constructs = len(construct_totals)
    print(f"Unique constructs (overall): {unique_constructs}")
    print("")
    print(f"Top {args.top_k} constructs by total Paper_Count:")

    top_constructs = sorted(
        construct_totals.items(),
        key=lambda kv: (-kv[1], kv[0].casefold()),
    )[: max(1, args.top_k)]

    for rank, (name, total) in enumerate(top_constructs, start=1):
        print(f"{rank:>2}. {name} | Total_Paper_Count={total}")

    # 2) Write noise_review.csv (row-level; sorted by row Paper_Count desc)
    noise_rows_sorted = sorted(
        noise_rows,
        key=lambda r: (-to_int(r.get("Paper_Count", "0")), (r.get("Construct_Name") or "").casefold()),
    )

    noise_output_path = Path(args.noise_output)
    noise_output_path.parent.mkdir(parents=True, exist_ok=True)
    noise_fieldnames = [
        "Construct_Name",
        "Source",
        "Journal",
        "Year",
        "Concept_Level",
        "Paper_Count",
        "Journal_Tier",
        "Is_Noise",
        "Noise_Reasons",
    ]
    with noise_output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=noise_fieldnames)
        writer.writeheader()
        writer.writerows(noise_rows_sorted)

    # 3) Write openalex_approved.csv (construct-level aggregate + dedupe)
    approved_rows: List[Dict[str, str]] = []
    for construct, total in approved_totals.items():
        journals = sorted(approved_journals.get(construct, set()))
        approved_rows.append(
            {
                "Construct_Name": construct,
                "Source": "OpenAlex",
                "Total_Paper_Count": str(total),
                "Concept_Levels": ",".join(sorted(approved_levels.get(construct, set()))),
                "Journal_Count": str(len(journals)),
                "Journals": "; ".join(journals),
                "Year_Min": str(approved_year_min.get(construct, "")),
                "Year_Max": str(approved_year_max.get(construct, "")),
            }
        )

    approved_rows.sort(
        key=lambda r: (-to_int(r["Total_Paper_Count"]), r["Construct_Name"].casefold())
    )

    approved_output_path = Path(args.approved_output)
    approved_output_path.parent.mkdir(parents=True, exist_ok=True)
    approved_fieldnames = [
        "Construct_Name",
        "Source",
        "Total_Paper_Count",
        "Concept_Levels",
        "Journal_Count",
        "Journals",
        "Year_Min",
        "Year_Max",
    ]
    with approved_output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=approved_fieldnames)
        writer.writeheader()
        writer.writerows(approved_rows)

    print("")
    print(f"Wrote noise review: {noise_output_path} ({len(noise_rows_sorted)} rows)")
    print(f"Wrote approved set: {approved_output_path} ({len(approved_rows)} constructs)")


if __name__ == "__main__":
    main()
