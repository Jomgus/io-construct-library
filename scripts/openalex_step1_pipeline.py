#!/usr/bin/env python3
"""
Step 1 OpenAlex pipeline:
- Pull works from a two-tier journal whitelist.
- Extract concept/construct names from each work.
- Aggregate paper counts with provenance (journal + year).
- Soft-flag potential noise without deleting any rows.

Output:
- openalex_filtered.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


TIER_1_CORE_IO = [
    "Journal of Applied Psychology",
    "Personnel Psychology",
    "Journal of Organizational Behavior",
    "Journal of Occupational and Organizational Psychology",
]

TIER_2_BROAD_ADJACENT = [
    "Academy of Management Journal",
    "Journal of Management",
    "Applied Psychology: An International Review",
    "Journal of Vocational Behavior",
]

# Soft flagging list only. Rows are never dropped.
NOISE_KEYWORDS = [
    "buddhism",
    "physics",
    "quantum",
    "nanotechnology",
    "astrophysics",
    "robotics",
    "geology",
    "zoology",
    "botany",
    "neurosurgery",
    "chemistry",
    "mathematics",
    "computer science",
    "political science",
    "mechanical engineering",
    "electrical engineering",
    "civil engineering",
    "materials science",
]

OPENALEX_BASE = "https://api.openalex.org"
# Known title variants to improve source resolution.
JOURNAL_QUERY_ALIASES = {
    "Applied Psychology: An International Review": [
        "Applied Psychology International Review",
        "Applied Psychology",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAlex construct CSV with provenance.")
    parser.add_argument(
        "--output",
        default="openalex_filtered.csv",
        help="Output CSV path (default: openalex_filtered.csv)",
    )
    parser.add_argument(
        "--mail-to",
        default="",
        help="Optional contact email for OpenAlex polite pool.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=200,
        help="OpenAlex page size (max 200).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Delay between API calls.",
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=3,
        help="Minimum OpenAlex concept level to keep (default: 3).",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=5,
        help="Maximum OpenAlex concept level to keep (default: 5).",
    )
    return parser.parse_args()


def api_get(url: str, params: Dict[str, str], sleep_seconds: float) -> Dict:
    if params:
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"
    else:
        full_url = url

    req = urllib.request.Request(
        full_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "construct-library-step1/1.0",
        },
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    time.sleep(sleep_seconds)
    return payload


def normalize_text(value: str) -> str:
    lowered = (value or "").casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def score_source_candidate(journal_name: str, candidate: Dict) -> int:
    target_norm = normalize_text(journal_name)
    candidate_name = candidate.get("display_name") or ""
    cand_norm = normalize_text(candidate_name)

    score = 0
    if cand_norm == target_norm:
        score += 100

    target_tokens = set(target_norm.split())
    cand_tokens = set(cand_norm.split())
    if target_tokens and target_tokens.issubset(cand_tokens):
        score += 40

    # Partial overlap fallback for title variants.
    overlap = len(target_tokens & cand_tokens)
    score += min(overlap * 5, 25)

    if (candidate.get("type") or "").casefold() == "journal":
        score += 10

    works_count = int(candidate.get("works_count") or 0)
    if works_count > 0:
        score += min(works_count // 10000, 10)

    return score


def resolve_source_id(journal_name: str, mail_to: str, sleep_seconds: float) -> str:
    queries = [journal_name]
    queries.extend(JOURNAL_QUERY_ALIASES.get(journal_name, []))
    queries.append(normalize_text(journal_name))

    best_candidate: Dict | None = None
    best_score = -1
    last_results: List[Dict] = []

    for query in queries:
        params = {"search": query, "per-page": "50"}
        if mail_to:
            params["mailto"] = mail_to

        payload = api_get(f"{OPENALEX_BASE}/sources", params, sleep_seconds)
        results = payload.get("results", [])
        if results:
            last_results = results

        for row in results:
            score = score_source_candidate(journal_name, row)
            if score > best_score:
                best_score = score
                best_candidate = row

    if best_candidate:
        return best_candidate["id"]

    sample_names = [r.get("display_name", "") for r in last_results[:5]]
    raise RuntimeError(
        f"Could not resolve OpenAlex source id for journal: {journal_name}. "
        f"Sample candidates: {sample_names}"
    )


def iter_source_works(
    source_id: str,
    mail_to: str,
    per_page: int,
    sleep_seconds: float,
) -> Iterable[Dict]:
    cursor = "*"
    while True:
        params = {
            "filter": f"primary_location.source.id:{source_id}",
            "per-page": str(per_page),
            "cursor": cursor,
        }
        if mail_to:
            params["mailto"] = mail_to

        payload = api_get(f"{OPENALEX_BASE}/works", params, sleep_seconds)
        results = payload.get("results", [])
        for work in results:
            yield work

        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor or not results:
            break


def infer_noise_reasons(construct_name: str) -> List[str]:
    lowered = construct_name.casefold().strip()
    reasons: List[str] = []
    for keyword in NOISE_KEYWORDS:
        if keyword in lowered:
            reasons.append(f"keyword:{keyword}")

    if len(lowered) <= 2:
        reasons.append("too_short")

    return reasons


def extract_works_to_rows(
    source_meta: List[Tuple[str, str, str]],
    mail_to: str,
    per_page: int,
    sleep_seconds: float,
    min_level: int,
    max_level: int,
) -> List[Dict[str, str]]:
    # key: (construct, journal, year, concept_level)
    counts: Dict[Tuple[str, str, int, int], int] = defaultdict(int)

    for journal_name, tier, source_id in source_meta:
        print(f"Fetching works for {journal_name} ({tier})...")
        for work in iter_source_works(source_id, mail_to, per_page, sleep_seconds):
            year = work.get("publication_year")
            if not year:
                continue

            # Deduplicate concepts at work-level by (display_name, level) and enforce level filter.
            unique_concepts: Set[Tuple[str, int]] = set()
            for concept in (work.get("concepts") or []):
                construct = (concept.get("display_name") or "").strip()
                level = concept.get("level")
                if not construct or not isinstance(level, int):
                    continue
                if level < min_level or level > max_level:
                    continue
                unique_concepts.add((construct, level))

            for construct, level in unique_concepts:
                counts[(construct, journal_name, int(year), level)] += 1

    rows: List[Dict[str, str]] = []
    for (construct, journal_name, year, concept_level), paper_count in counts.items():
        reasons = infer_noise_reasons(construct)
        rows.append(
            {
                "Construct_Name": construct,
                "Source": "OpenAlex",
                "Journal": journal_name,
                "Year": str(year),
                "Concept_Level": str(concept_level),
                "Paper_Count": str(paper_count),
                "Journal_Tier": tier_for_journal(journal_name),
                "Is_Noise": "True" if reasons else "False",
                "Noise_Reasons": ";".join(reasons),
            }
        )

    rows.sort(
        key=lambda r: (
            -int(r["Paper_Count"]),
            r["Construct_Name"].casefold(),
            r["Journal"].casefold(),
            int(r["Year"]),
            int(r["Concept_Level"]),
        )
    )
    return rows


def tier_for_journal(journal_name: str) -> str:
    if journal_name in TIER_1_CORE_IO:
        return "Tier_1_Core_IO"
    return "Tier_2_Broad_Adjacent"


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    journals_with_tiers = [
        *( (j, "Tier_1_Core_IO") for j in TIER_1_CORE_IO ),
        *( (j, "Tier_2_Broad_Adjacent") for j in TIER_2_BROAD_ADJACENT ),
    ]

    source_meta: List[Tuple[str, str, str]] = []
    for journal_name, tier in journals_with_tiers:
        source_id = resolve_source_id(journal_name, args.mail_to, args.sleep_seconds)
        source_meta.append((journal_name, tier, source_id))
        print(f"Resolved source: {journal_name} -> {source_id}")

    rows = extract_works_to_rows(
        source_meta=source_meta,
        mail_to=args.mail_to,
        per_page=max(1, min(args.per_page, 200)),
        sleep_seconds=max(0.0, args.sleep_seconds),
        min_level=max(0, args.min_level),
        max_level=max(0, args.max_level),
    )

    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
