#!/usr/bin/env python3
"""
Step 2: Data enrichment via Wikipedia API with resume/checkpoint support.

Features:
- Reads construct rows from openalex_ultimate_final.csv.
- Adds Definition_Text, Definition_Source, Last_Updated.
- Uses a JSON cache to avoid re-querying completed constructs.
- Writes temporary checkpoint CSV every N rows.
- Resumes safely after interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests


DEFAULT_DEFINITION_SOURCE = "Wikidata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich constructs with Wikipedia definitions.")
    parser.add_argument(
        "--input",
        default="data/processed/openalex_ultimate_final.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/openalex_enriched.csv",
        help="Final output CSV path.",
    )
    parser.add_argument(
        "--checkpoint-csv",
        default="data/processed/openalex_enriched.checkpoint.csv",
        help="Intermediate checkpoint CSV path.",
    )
    parser.add_argument(
        "--cache-json",
        default="data/processed/openalex_enrichment_cache.json",
        help="JSON cache path for API responses.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write checkpoint files every N processed rows.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between live API calls.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional limit for test runs (0 = all rows).",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_key(name: str) -> str:
    return (name or "").strip().casefold()


def fetch_wikipedia_definition(
    construct_name: str,
    timeout: float,
    user_agent: str = "construct-library/step2-enrichment (contact: jng6114@mavs.uta.edu)",
) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exsentences": 2,
        "exlimit": 1,
        "titles": construct_name,
        "explaintext": 1,
        "format": "json",
        "redirects": 1,
    }
    headers = {"Accept": "application/json", "User-Agent": user_agent}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        if not pages:
            return "No page data returned by Wikipedia."

        page_id = next(iter(pages.keys()))
        if page_id == "-1":
            return "No exact Wikipedia match found."

        extract = pages[page_id].get("extract") or "No extract available."
        return extract.replace("\n", " ").strip()
    except requests.exceptions.Timeout:
        return "API Error: request timed out."
    except requests.exceptions.RequestException as exc:
        return f"API Error: {exc}"
    except ValueError:
        return "API Error: invalid JSON response."


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_csv_path = Path(args.checkpoint_csv)
    cache_json_path = Path(args.cache_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    if not rows:
        raise RuntimeError("Input CSV has no rows to enrich.")

    base_fieldnames = list(rows[0].keys())
    final_fieldnames = [
        *base_fieldnames,
        "Definition_Text",
        "Definition_Source",
        "Last_Updated",
    ]

    cache = load_cache(cache_json_path)

    processed = 0
    cache_hits = 0
    api_calls = 0

    for idx, row in enumerate(rows, start=1):
        construct_name = (row.get("Construct_Name") or "").strip()
        key = normalize_key(construct_name)

        if not construct_name:
            row["Definition_Text"] = "No construct name provided."
            row["Definition_Source"] = DEFAULT_DEFINITION_SOURCE
            row["Last_Updated"] = now_iso()
            processed += 1
            continue

        if key in cache:
            cached = cache[key]
            row["Definition_Text"] = cached.get("Definition_Text", "")
            row["Definition_Source"] = cached.get("Definition_Source", DEFAULT_DEFINITION_SOURCE)
            row["Last_Updated"] = cached.get("Last_Updated", now_iso())
            cache_hits += 1
        else:
            definition_text = fetch_wikipedia_definition(construct_name, timeout=args.timeout)
            updated_at = now_iso()
            row["Definition_Text"] = definition_text
            row["Definition_Source"] = DEFAULT_DEFINITION_SOURCE
            row["Last_Updated"] = updated_at

            cache[key] = {
                "Construct_Name": construct_name,
                "Definition_Text": definition_text,
                "Definition_Source": DEFAULT_DEFINITION_SOURCE,
                "Last_Updated": updated_at,
            }
            api_calls += 1
            time.sleep(max(0.0, args.sleep_seconds))

        processed += 1

        if processed % max(1, args.checkpoint_every) == 0:
            write_csv(checkpoint_csv_path, rows, final_fieldnames)
            atomic_write_json(cache_json_path, cache)
            print(
                f"[checkpoint] processed={processed}/{len(rows)} "
                f"api_calls={api_calls} cache_hits={cache_hits}"
            )

    write_csv(output_path, rows, final_fieldnames)
    write_csv(checkpoint_csv_path, rows, final_fieldnames)
    atomic_write_json(cache_json_path, cache)

    print(f"Completed enrichment for {len(rows)} rows.")
    print(f"API calls: {api_calls}")
    print(f"Cache hits: {cache_hits}")
    print(f"Output: {output_path}")
    print(f"Checkpoint CSV: {checkpoint_csv_path}")
    print(f"Cache JSON: {cache_json_path}")


if __name__ == "__main__":
    main()
